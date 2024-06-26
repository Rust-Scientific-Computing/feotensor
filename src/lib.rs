pub mod shape;
pub mod iter;
pub mod error;

use num::Num;
use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Div;

use crate::shape::Shape;
use crate::iter::IndexIterator;
use crate::error::ShapeError;

pub type Axes = Vec<usize>;

#[derive(Debug, Clone)]
pub struct Tensor<T: Num> {
    data: Vec<T>,
    shape: Shape
}

impl<T: Num + PartialOrd + Copy> Tensor<T> {

    pub fn new(shape: &Shape, data: &[T]) -> Result<Tensor<T>, ShapeError> {
        if data.len() != shape.size() {
            return Err(ShapeError::new("Data length does not match shape size"));
        }
        Ok(Tensor {data: data.to_vec(), shape: shape.clone()})
    }

    pub fn fill(shape: &Shape, value: T) -> Tensor<T> {
        let total_size = shape.size();
        let mut vec = Vec::with_capacity(total_size);
        for _ in 0..total_size { vec.push(value); }
        Tensor::new(shape, &vec).unwrap()
    }

    pub fn zeros(shape: &Shape) -> Tensor<T> {
        Tensor::fill(shape, T::zero())
    }

    pub fn ones(shape: &Shape) -> Tensor<T> {
        Tensor::fill(shape, T::one())
    }

    // Element-wise operations
    pub fn pow(&self, power: usize) -> Tensor<T> {
        let mut t = self.clone();
        for i in 0..t.size() {
            t.data[i] = num::pow(t.data[i], power);
        }
        t
    }

    // Properties
    pub fn shape(&self) -> &Shape { &self.shape }
    pub fn size(&self) -> usize { self.shape.size() }

    pub fn get_element(&self, indices: &[usize]) -> Result<&T, ShapeError> {
        self.assert_indices(indices)?;
        Ok(&self.data[self.calculate_index(indices)])
    }

    pub fn set_element(&mut self, indices: &[usize], value: T) -> Result<(), ShapeError> {
        self.assert_indices(indices)?;
        let index = self.calculate_index(indices);
        self.data[index] = value;
        Ok(())
    }

    // // Reduction operations
    pub fn sum(&self, axes: Axes) -> Tensor<T> {
        let all_axes = (0..self.shape.len()).collect::<Vec<_>>();
        let remaining_axes = all_axes.clone().into_iter().filter(|&i| !axes.contains(&i)).collect::<Vec<_>>();
        let remaining_dims = remaining_axes.iter().map(|&i| self.shape.dims[i]).collect::<Vec<_>>();
        let removing_dims = axes.iter().map(|&i| self.shape.dims[i]).collect::<Vec<_>>();

        // We resolve to a scalar value
        if axes.is_empty() | (remaining_dims.len() == 0) {
            let sum: T = self.data.iter().fold(T::zero(), |acc, x| acc + *x);
            return Tensor::new(&Shape::new(vec![1]).unwrap(), &[sum]).unwrap();
        }

        // Create new tensor with right shape
        let new_shape = Shape::new(remaining_dims).unwrap();
        let mut t: Tensor<T> = Tensor::zeros(&new_shape);

        for target in IndexIterator::new(&new_shape.dims) {
            let sum_iter = IndexIterator::new(&removing_dims);
            for sum_index in sum_iter {
                let mut indices = target.clone();
                for (i, &axis) in axes.iter().enumerate() {
                    indices.insert(axis, sum_index[i]);
                }

                let value = *t.get_element(&target).unwrap() + *self.get_element(&indices).unwrap();
                let _ = t.set_element(&target, value).unwrap();
            }
        }

        t
    }

    pub fn mean(&self, axes: Axes) -> Tensor<T> {
        let removing_dims = axes.iter().map(|&i| self.shape.dims[i]).collect::<Vec<_>>();
        let removing_dims_t: Vec<T> = removing_dims.iter().map(|&dim| {
            let mut result = T::zero();
            for _ in 0..dim {
                result = result + T::one();
            }
            result
        }).collect();
        let n = removing_dims_t.iter().fold(T::one(), |acc, x| acc * *x);
        self.sum(axes) / n
    }

    pub fn var(&self, axes: Axes) -> Tensor<T> {
        let removing_dims = axes.iter().map(|&i| self.shape.dims[i]).collect::<Vec<_>>();
        let removing_dims_t: Vec<T> = removing_dims.iter().map(|&dim| {
            let mut result = T::zero();
            for _ in 0..dim {
                result = result + T::one();
            }
            result
        }).collect();
        let n = removing_dims_t.iter().fold(T::one(), |acc, x| acc * *x);
        
        let all_axes = (0..self.shape.len()).collect::<Vec<_>>();
        let remaining_axes = all_axes.clone().into_iter().filter(|&i| !axes.contains(&i)).collect::<Vec<_>>();
        let remaining_dims = remaining_axes.iter().map(|&i| self.shape.dims[i]).collect::<Vec<_>>();
        let removing_dims = axes.iter().map(|&i| self.shape.dims[i]).collect::<Vec<_>>();

        // We resolve to a scalar value
        if axes.is_empty() | (remaining_dims.len() == 0) {
            let avg: T = self.data.iter().fold(T::zero(), |acc, x| acc + *x) / n;
            let var: T = self.data.iter().map(|&x| (x - avg) * (x - avg)).fold(T::zero(), |acc, x| acc + x) / n;
            return Tensor::new(&Shape::new(vec![1]).unwrap(), &[var]).unwrap();
        }

        // Create new tensor with right shape
        let new_shape = Shape::new(remaining_dims).unwrap();
        let mut t: Tensor<T> = Tensor::zeros(&new_shape);

        for target in IndexIterator::new(&new_shape.dims) {
            let sum_iter = IndexIterator::new(&removing_dims);
            let mean = self.mean(axes.clone());

            for sum_index in sum_iter {
                let mut indices = target.clone();
                for (i, &axis) in axes.iter().enumerate() {
                    indices.insert(axis, sum_index[i]);
                }

                let centered = *self.get_element(&indices).unwrap() - *mean.get_element(&target).unwrap();
                let value = *t.get_element(&target).unwrap() + centered * centered;
                let _ = t.set_element(&target, value).unwrap();
            }
        }

        t / n
    }

    pub fn max(&self, axes: Axes) -> Tensor<T> {
        let all_axes = (0..self.shape.len()).collect::<Vec<_>>();
        let remaining_axes = all_axes.clone().into_iter().filter(|&i| !axes.contains(&i)).collect::<Vec<_>>();
        let remaining_dims = remaining_axes.iter().map(|&i| self.shape.dims[i]).collect::<Vec<_>>();
        let removing_dims = axes.iter().map(|&i| self.shape.dims[i]).collect::<Vec<_>>();

        // We resolve to a scalar value
        if axes.is_empty() | (remaining_dims.len() == 0) {
            let max: T = self.data.iter().fold(T::zero(), |acc, x| if acc > *x { acc } else { *x });
            return Tensor::new(&Shape::new(vec![1]).unwrap(), &[max]).unwrap();
        }

        // Create new tensor with right shape
        let new_shape = Shape::new(remaining_dims).unwrap();
        let min: T = self.data.iter().fold(T::zero(), |acc, x| if acc < *x { acc } else { *x });
        let mut t: Tensor<T> = Tensor::fill(&new_shape, min);

        for target in IndexIterator::new(&new_shape.dims) {
            let max_iter = IndexIterator::new(&removing_dims);
            for max_index in max_iter {
                let mut indices = target.clone();
                for (i, &axis) in axes.iter().enumerate() {
                    indices.insert(axis, max_index[i]);
                }

                if self.get_element(&indices).unwrap() > t.get_element(&target).unwrap() {
                    let _ = t.set_element(&target, *self.get_element(&indices).unwrap());
                }
            }
        }

        t
    }

    pub fn min(&self, axes: Axes) -> Tensor<T> {
        let all_axes = (0..self.shape.len()).collect::<Vec<_>>();
        let remaining_axes = all_axes.clone().into_iter().filter(|&i| !axes.contains(&i)).collect::<Vec<_>>();
        let remaining_dims = remaining_axes.iter().map(|&i| self.shape.dims[i]).collect::<Vec<_>>();
        let removing_dims = axes.iter().map(|&i| self.shape.dims[i]).collect::<Vec<_>>();

        // We resolve to a scalar value
        if axes.is_empty() | (remaining_dims.len() == 0) {
            let min: T = self.data.iter().fold(T::zero(), |acc, x| if acc < *x { acc } else { *x });
            return Tensor::new(&Shape::new(vec![1]).unwrap(), &[min]).unwrap();
        }

        // Create new tensor with right shape
        let new_shape = Shape::new(remaining_dims).unwrap();
        let max: T = self.data.iter().fold(T::zero(), |acc, x| if acc > *x { acc } else { *x });
        let mut t: Tensor<T> = Tensor::fill(&new_shape, max);

        for target in IndexIterator::new(&new_shape.dims) {
            let min_iter = IndexIterator::new(&removing_dims);
            for min_index in min_iter {
                let mut indices = target.clone();
                for (i, &axis) in axes.iter().enumerate() {
                    indices.insert(axis, min_index[i]);
                }

                if self.get_element(&indices).unwrap() < t.get_element(&target).unwrap() {
                    let _ = t.set_element(&target, *self.get_element(&indices).unwrap());
                }
            }
        }

        t
    }

    // Tensor Product
    // Consistent with numpy.tensordot(a, b, axis=0)
    pub fn prod(&self, other: &Tensor<T>) -> Tensor<T> {
        let mut new_dims = self.shape.dims.clone();
        new_dims.extend(&other.shape.dims);
        let new_shape = Shape::new(new_dims).unwrap();

        let mut new_data = Vec::with_capacity(self.size() * other.size());
        for &a in &self.data {
            for &b in &other.data {
                new_data.push(a * b);
            }
        }

        Tensor::new(&new_shape, &new_data).unwrap()
    }

    /// For the maths see: https://bit.ly/3KQjPa3
    fn calculate_index(&self, indices: &[usize]) -> usize {
        let mut index = 0;
        for k in 0..self.shape.len() {
            let stride = self.shape.dims[k+1..].iter().product::<usize>();
            index += indices[k] * stride;
        }
        index
    }

    fn assert_indices(&self, indices: &[usize]) -> Result<(), ShapeError> {
        if indices.len() != self.shape.len() {
            let msg = format!("incorrect order ({} vs {}).", indices.len(), self.shape.len());
            return Err(ShapeError::new(msg.as_str()));
        }
        for (i, &index) in indices.iter().enumerate() {
            if index >= self.shape.dims[i] {
                return Err(ShapeError::new(format!("out of bounds for dimension {}", i).as_str()));
            }
        }
        Ok(())
    }
}

// Element-wise Multiplication
impl<T: Num + PartialOrd + Copy> Mul<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Tensor<T> {
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i].clone() * rhs;
        }
        result
    }
}

// Vector/Matrix Multiplication
impl<T: Num + PartialOrd + Copy> Mul<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Tensor<T>) -> Tensor<T> {
    if self.shape.len() == 1 && rhs.shape.len() == 1 {
        // Vector-Vector multiplication (dot product)
        assert!(self.shape[0] == rhs.shape[0], "Vectors must be of the same length for dot product.");
        let mut result = T::zero();
        for i in 0..self.shape[0] {
            result = result + self.data[i] * rhs.data[i];
        }
        Tensor::new(&shape![1].unwrap(), &vec![result]).unwrap()
    } else if self.shape.len() == 1 && rhs.shape.len() == 2 {
        // Vector-Matrix multiplication
        assert!(self.shape[0] == rhs.shape[0], "The length of the vector must be equal to the number of rows in the matrix.");
        let mut result = Tensor::zeros(&shape![rhs.shape[1]].unwrap());
        for j in 0..rhs.shape[1] {
            let mut sum = T::zero();
            for i in 0..self.shape[0] {
                sum = sum + self.data[i] * rhs.data[i * rhs.shape[1] + j];
            }
            result.data[j] = sum;
        }
        result
    } else if self.shape.len() == 2 && rhs.shape.len() == 1 {
        // Matrix-Vector multiplication
        assert!(self.shape[1] == rhs.shape[0], "The number of columns in the matrix must be equal to the length of the vector.");
        let mut result = Tensor::zeros(&shape![self.shape[0]].unwrap());
        for i in 0..self.shape[0] {
            let mut sum = T::zero();
            for j in 0..self.shape[1] {
                sum = sum + self.data[i * self.shape[1] + j] * rhs.data[j];
            }
            result.data[i] = sum;
        }
        result
    } else if self.shape.len() == 2 && rhs.shape.len() == 2 {
        // Matrix-Matrix multiplication
        assert!(self.shape[1] == rhs.shape[0], "The number of columns in the first matrix must be equal to the number of rows in the second matrix.");
        let mut result = Tensor::zeros(&shape![self.shape[0], rhs.shape[1]].unwrap());
        for i in 0..self.shape[0] {
            for j in 0..rhs.shape[1] {
                let mut sum = T::zero();
                for k in 0..self.shape[1] {
                    sum = sum + self.data[i * self.shape[1] + k] * rhs.data[k * rhs.shape[1] + j];
                }
                result.data[i * rhs.shape[1] + j] = sum;
            }
        }
        result
    } else {
        panic!("Unsupported shapes for multiplication.");
    }
    }
}

// Element-wise Addition
impl<T: Num + PartialOrd + Copy> Add<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: T) -> Tensor<T> {
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i].clone() + rhs;
        }
        result
    }
}

// Tensor Addition
impl<T: Num + PartialOrd + Copy> Add<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Tensor<T>) -> Tensor<T> {
        assert!(self.shape == rhs.shape);
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i].clone() + rhs.data[i].clone();
        }
        result
    }
}

// Element-wise Subtraction
impl<T: Num + PartialOrd + Copy> Sub<T> for Tensor<T>
{
    type Output = Tensor<T>;

    fn sub(self, rhs: T) -> Tensor<T> {
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i].clone() - rhs;
        }
        result
    }
}

// Tensor Subtraction
impl<T: Num + PartialOrd + Copy> Sub<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Tensor<T>) -> Tensor<T> {
        assert!(self.shape == rhs.shape);
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i].clone() - rhs.data[i].clone();
        }
        result
    }
}

// Element-wise Division
impl<T: Num + PartialOrd + Copy> Div<T> for Tensor<T>
{
    type Output = Tensor<T>;

    fn div(self, rhs: T) -> Tensor<T> {
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i].clone() / rhs;
        }
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tensor() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let tensor = Tensor::new(&shape, &data).unwrap();

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data, data);
    }

    #[test]
    fn test_new_tensor_shape_data_mismatch() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0]; // Mismatched data length

        let result = Tensor::new(&shape, &data);

        assert!(result.is_err());
    }

    #[test]
    fn test_zeros_tensor() {
        let shape = shape![2, 3].unwrap();
        let tensor: Tensor<f32> = Tensor::zeros(&shape);

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data, vec![0.0; shape.size()]);
    }

    #[test]
    fn test_ones_tensor() {
        let shape = shape![2, 3].unwrap();
        let tensor: Tensor<f32> = Tensor::ones(&shape);

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data, vec![1.0; shape.size()]);
    }

    #[test]
    fn test_fill_tensor() {
        let shape = shape![2, 3].unwrap();
        let tensor: Tensor<f32> = Tensor::fill(&shape, 7.0);

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data, vec![7.0; shape.size()]);
    }

    #[test]
    fn test_tensor_shape() {
        let shape = shape![2, 3].unwrap();
        let tensor: Tensor<f32> = Tensor::zeros(&shape);

        assert_eq!(tensor.shape(), &shape);
    }

    #[test]
    fn test_tensor_size() {
        let shape = shape![2, 3].unwrap();
        let tensor: Tensor<f32> = Tensor::zeros(&shape);

        assert_eq!(tensor.size(), 6);
    }

    #[test]
    fn test_tensor_pow() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.pow(2);

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![1.0, 4.0, 9.0, 16.0]);

        let result = tensor.pow(3);

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![1.0, 8.0, 27.0, 64.0]);
    }

    #[test]
    fn test_tensor_get() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        assert_eq!(*tensor.get_element(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*tensor.get_element(&[0, 1]).unwrap(), 2.0);
        assert_eq!(*tensor.get_element(&[1, 0]).unwrap(), 3.0);
        assert_eq!(*tensor.get_element(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_tensor_set() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = Tensor::new(&shape, &data).unwrap();

        tensor.set_element(&[0, 0], 5.0).unwrap();
        tensor.set_element(&[0, 1], 6.0).unwrap();
        tensor.set_element(&[1, 0], 7.0).unwrap();
        tensor.set_element(&[1, 1], 8.0).unwrap();

        assert_eq!(*tensor.get_element(&[0, 0]).unwrap(), 5.0);
        assert_eq!(*tensor.get_element(&[0, 1]).unwrap(), 6.0);
        assert_eq!(*tensor.get_element(&[1, 0]).unwrap(), 7.0);
        assert_eq!(*tensor.get_element(&[1, 1]).unwrap(), 8.0);
    }

    #[test]
    fn test_tensor_get_out_of_bounds() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        assert!(tensor.get_element(&[2, 0]).is_err());
        assert!(tensor.get_element(&[0, 2]).is_err());
        assert!(tensor.get_element(&[2, 2]).is_err());
    }

    #[test]
    fn test_tensor_set_out_of_bounds() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = Tensor::new(&shape, &data).unwrap();

        assert!(tensor.set_element(&[2, 0], 5.0).is_err());
        assert!(tensor.set_element(&[0, 2], 6.0).is_err());
        assert!(tensor.set_element(&[2, 2], 7.0).is_err());
    }

    #[test]
    fn test_tensor_sum_no_axis_1d() {
        let shape = shape![5].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, vec![15.0]);
    }

    #[test]
    fn test_tensor_sum_no_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, vec![21.0]);
    }

    #[test]
    fn test_tensor_sum_no_axis_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, vec![78.0]);
    }

    #[test]
    fn test_tensor_sum_one_axis_1d() {
        let shape = shape![5].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![0]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, vec![15.0]);
    }

    #[test]
    fn test_tensor_sum_one_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![0]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tensor_sum_one_axis_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![0]);

        assert_eq!(result.shape(), &shape![2, 3].unwrap());
        assert_eq!(result.data, vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
    }

    #[test]
    fn test_tensor_sum_multiple_axes_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![0, 1]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, vec![21.0]);
    }

    #[test]
    fn test_tensor_sum_multiple_axes_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, vec![22.0, 26.0, 30.0]);
    }

    #[test]
    fn test_tensor_mean_one_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.mean(vec![0]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_tensor_mean_one_axis_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.mean(vec![0]);

        assert_eq!(result.shape(), &shape![2, 3].unwrap());
        assert_eq!(result.data, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_tensor_mean_multiple_axes_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.mean(vec![0, 1]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, vec![3.5]);
    }

    #[test]
    fn test_tensor_mean_multiple_axes_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.mean(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, vec![5.5, 6.5, 7.5]);
    }

    #[test]
    fn test_tensor_var_one_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.var(vec![0]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, vec![2.25, 2.25, 2.25]);
    }

    #[test]
    fn test_tensor_var_one_axis_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.var(vec![0]);

        assert_eq!(result.shape(), &shape![2, 3].unwrap());
        assert_eq!(result.data, vec![9.0, 9.0, 9.0, 9.0, 9.0, 9.0]);
    }

    #[test]
    fn test_tensor_var_multiple_axes_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 1.0, 1.0, 7.0, 7.0, 7.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.var(vec![0, 1]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, vec![9.0]);
    }

    #[test]
    fn test_tensor_var_multiple_axes_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.var(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, vec![45.0, 45.0, 45.0]);
    }

    #[test]
    fn test_tensor_max_no_axis_1d() {
        let shape = shape![5].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.max(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, vec![5.0]);
    }

    #[test]
    fn test_tensor_max_one_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.max(vec![0]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, vec![1.0, 5.0, 3.0]);
    }

    #[test]
    fn test_tensor_max_multiple_axes_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.max(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, vec![7.0, 11.0, 9.0]);
    }

    #[test]
    fn test_tensor_min_no_axis_1d() {
        let shape = shape![5].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.min(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, vec![-4.0]);
    }

    #[test]
    fn test_tensor_min_one_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.min(vec![0]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, vec![-4.0, -2.0, -6.0]);
    }

    #[test]
    fn test_tensor_min_multiple_axes_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.min(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, vec![-10.0, -8.0, -12.0]);
    }

    #[test]
    fn test_tensor_prod_1d_1d() {
        let shape1 = shape![3].unwrap();
        let data1 = vec![1.0, 2.0, 3.0];
        let tensor1 = Tensor::new(&shape1, &data1).unwrap();

        let shape2 = shape![2].unwrap();
        let data2 = vec![4.0, 5.0];
        let tensor2 = Tensor::new(&shape2, &data2).unwrap();

        let result = tensor1.prod(&tensor2);

        assert_eq!(result.shape(), &shape![3, 2].unwrap());
        assert_eq!(result.data, vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn test_tensor_prod_2d_1d() {
        let shape1 = shape![2, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(&shape1, &data1).unwrap();

        let shape2 = shape![2].unwrap();
        let data2 = vec![5.0, 6.0];
        let tensor2 = Tensor::new(&shape2, &data2).unwrap();

        let result = tensor1.prod(&tensor2);

        assert_eq!(result.shape(), &shape![2, 2, 2].unwrap());
        assert_eq!(result.data, vec![5.0, 6.0, 10.0, 12.0, 15.0, 18.0, 20.0, 24.0]);
    }

    #[test]
    fn test_tensor_prod_2d_2d() {
        let shape1 = shape![2, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(&shape1, &data1).unwrap();

        let shape2 = shape![2, 2].unwrap();
        let data2 = vec![5.0, 6.0, 7.0, 8.0];
        let tensor2 = Tensor::new(&shape2, &data2).unwrap();

        let result = tensor1.prod(&tensor2);

        assert_eq!(result.shape(), &shape![2, 2, 2, 2].unwrap());
        assert_eq!(result.data, vec![5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 15.0, 18.0, 21.0, 24.0, 20.0, 24.0, 28.0, 32.0]);
    }

    #[test]
    fn test_add_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(&shape, &data1).unwrap();

        let result = tensor1 + 3.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_add_tensors() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0, 7.0, 8.0];
        let tensor1 = Tensor::new(&shape, &data1).unwrap();
        let tensor2 = Tensor::new(&shape, &data2).unwrap();

        let result = tensor1 + tensor2;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_sub_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![5.0, 6.0, 7.0, 8.0];

        let tensor1 = Tensor::new(&shape, &data1).unwrap();

        let result = tensor1 - 3.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sub_tensors() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![5.0, 6.0, 7.0, 8.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(&shape, &data1).unwrap();
        let tensor2 = Tensor::new(&shape, &data2).unwrap();

        let result = tensor1 - tensor2;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_mul_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];

        let tensor1 = Tensor::new(&shape, &data1).unwrap();

        let result = tensor1 * 2.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_vec_vec_mul_single() {
        let shape = shape![1].unwrap();
        let data1 = vec![2.0];
        let data2 = vec![5.0];

        let tensor1 = Tensor::new(&shape, &data1).unwrap();
        let tensor2 = Tensor::new(&shape, &data2).unwrap();

        let result = tensor1 * tensor2;

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, vec![10.0]);
    }

    #[test]
    fn test_vec_vec_mul() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];

        let tensor1 = Tensor::new(&shape, &data1).unwrap();
        let tensor2 = Tensor::new(&shape, &data2).unwrap();

        let result = tensor1 * tensor2;

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, vec![40.0]);
    }

    #[test]
    fn test_vec_matrix_mul() {
        let shape_vec = shape![2].unwrap();
        let shape_matrix = shape![2, 3].unwrap();
        let data_vec = vec![1.0, 2.0];
        let data_matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let tensor_vec = Tensor::new(&shape_vec, &data_vec).unwrap();
        let tensor_matrix = Tensor::new(&shape_matrix, &data_matrix).unwrap();

        let result = tensor_vec * tensor_matrix;

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, vec![9.0, 12.0, 15.0]);
    }

    #[test]
    fn test_matrix_vec_mul() {
        let shape_matrix = shape![2, 3].unwrap();
        let shape_vec = shape![3].unwrap();
        let data_matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_vec = vec![1.0, 2.0, 3.0];

        let tensor_matrix = Tensor::new(&shape_matrix, &data_matrix).unwrap();
        let tensor_vec = Tensor::new(&shape_vec, &data_vec).unwrap();

        let result = tensor_matrix * tensor_vec;

        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result.data, vec![14.0, 32.0]);
    }

    #[test]
    fn test_matrix_matrix_mul() {
        let shape1 = shape![2, 3].unwrap();
        let shape2 = shape![3, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data2 = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let tensor1 = Tensor::new(&shape1, &data1).unwrap();
        let tensor2 = Tensor::new(&shape2, &data2).unwrap();

        let result = tensor1 * tensor2;

        assert_eq!(result.shape(), &shape![2, 2].unwrap());
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_div_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![4.0, 6.0, 8.0, 10.0];

        let tensor1 = Tensor::new(&shape, &data1).unwrap();

        let result = tensor1 / 2.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![2.0, 3.0, 4.0, 5.0]);
    }
}

