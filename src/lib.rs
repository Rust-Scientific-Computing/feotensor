pub mod shape;
pub mod iter;

use num::Num;
use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::ops::Div;

use crate::shape::Shape;
use crate::iter::IndexIterator;

pub type Axes = Vec<usize>;

#[derive(Debug, Clone)]
pub struct Tensor<T: Num> {
    data: Vec<T>,
    shape: Shape
}

impl<T: Num + Copy> Tensor<T> {

    pub fn new(shape: &Shape, data: &[T]) -> Tensor<T> {
        assert!(data.len() == shape.size());
        Tensor {data: data.to_vec(), shape: shape.clone()}
    }

    pub fn zeros(shape: &Shape) -> Tensor<T> {
        let total_size = shape.size();
        let mut vec = Vec::with_capacity(total_size);
        for _ in 0..total_size { vec.push(T::zero()); }
        Tensor::new(shape, &vec)
    }

    // Properties
    pub fn shape(&self) -> &Shape { &self.shape }
    pub fn size(&self) -> usize { self.shape.size() }

    // Access methods
    pub fn get_element(&self, indices: &[usize]) -> &T {
        self.assert_indices(indices).unwrap();
        &self.data[self.calculate_index(indices)]
    }
    pub fn set_element(&mut self, indices: &[usize], value: T) {
        self.assert_indices(indices).unwrap();
        let index = self.calculate_index(indices);
        self.data[index] = value;
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
            return Tensor::new(&Shape::new(vec![1]), &[sum]);
        }

        // Create new tensor with right shape
        let new_shape = Shape::new(remaining_dims);
        let mut t: Tensor<T> = Tensor::zeros(&new_shape);

        for target in IndexIterator::new(&new_shape.dims) {
            let sum_iter = IndexIterator::new(&removing_dims);
            for sum_index in sum_iter {
                let mut indices = target.clone();
                for (i, &axis) in axes.iter().enumerate() {
                    indices.insert(axis, sum_index[i]);
                }

                let value = *t.get_element(&target) + *self.get_element(&indices);
                t.set_element(&target, value);
            }
        }

        t
    }

    // pub fn var(&self, axes: Axes) -> Tensor<T> {}
    // pub fn mean(&self, axes: Axes) -> Tensor<T> {}
    // pub fn max(&self, axes: Option<Axes>) -> Tensor<T> {}
    // pub fn min(&self, axes: Axes) -> Tensor<T> {}

    /// For the maths see: https://bit.ly/3KQjPa3
    fn calculate_index(&self, indices: &[usize]) -> usize {
        let mut index = 0;
        for k in 0..self.shape.len() {
            let stride = self.shape.dims[k+1..].iter().product::<usize>();
            index += indices[k] * stride;
        }
        index
    }

    fn assert_indices(&self, indices: &[usize]) -> Result<(), String> {
        if indices.len() != self.shape.len() {
            return Err(format!("Incorrect number of dimensions ({} vs {}).", indices.len(), self.shape.len()));
        }
        for (i, &index) in indices.iter().enumerate() {
            if index >= self.shape.dims[i] {
                return Err(format!("Index out of bounds for dimension {}", i));
            }
        }
        Ok(())
    }

}

// Element-wise Multiplication
impl<T: Num + Copy> Mul<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Tensor<T> {
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i].clone() * rhs;
        }
        result
    }
}

// Element-wise Addition
impl<T: Num + Copy> Add<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: T) -> Tensor<T> {
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i].clone() + rhs;
        }
        result
    }
}

// Element-wise Subtraction
impl<T: Num + Copy> Sub<T> for Tensor<T>
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

// Element-wise Division
impl<T: Num + Copy> Div<T> for Tensor<T>
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
        let shape = shape![2, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let tensor = Tensor::new(&shape, &data);

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data, data);
    }

    #[test]
    fn test_zeros_tensor() {
        let shape = shape![2, 3];
        let tensor: Tensor<f32> = Tensor::zeros(&shape);

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data, vec![0.0; shape.size()]);
    }

    #[test]
    fn test_tensor_shape() {
        let shape = shape![2, 3];
        let tensor: Tensor<f32> = Tensor::zeros(&shape);

        assert_eq!(tensor.shape(), &shape);
    }

    #[test]
    fn test_tensor_size() {
        let shape = shape![2, 3];
        let tensor: Tensor<f32> = Tensor::zeros(&shape);

        assert_eq!(tensor.size(), 6);
    }

    #[test]
    fn test_tensor_get() {
        let shape = shape![2, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&shape, &data);

        assert_eq!(*tensor.get_element(&[0, 0]), 1.0);
        assert_eq!(*tensor.get_element(&[0, 1]), 2.0);
        assert_eq!(*tensor.get_element(&[1, 0]), 3.0);
        assert_eq!(*tensor.get_element(&[1, 1]), 4.0);
    }

    #[test]
    fn test_tensor_set() {
        let shape = shape![2, 2];
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = Tensor::new(&shape, &data);

        tensor.set_element(&[0, 0], 5.0);
        tensor.set_element(&[0, 1], 6.0);
        tensor.set_element(&[1, 0], 7.0);
        tensor.set_element(&[1, 1], 8.0);

        assert_eq!(*tensor.get_element(&[0, 0]), 5.0);
        assert_eq!(*tensor.get_element(&[0, 1]), 6.0);
        assert_eq!(*tensor.get_element(&[1, 0]), 7.0);
        assert_eq!(*tensor.get_element(&[1, 1]), 8.0);
    }

    #[test]
    fn test_tensor_sum_no_axis_1d() {
        let shape = shape![5];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::new(&shape, &data);

        let result = tensor.sum(vec![]);

        assert_eq!(result.shape(), &shape![1]);
        assert_eq!(result.data, vec![15.0]);
    }

    #[test]
    fn test_tensor_sum_no_axis_2d() {
        let shape = shape![2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data);

        let result = tensor.sum(vec![]);

        assert_eq!(result.shape(), &shape![1]);
        assert_eq!(result.data, vec![21.0]);
    }

    #[test]
    fn test_tensor_sum_no_axis_3d() {
        let shape = shape![2, 2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tensor = Tensor::new(&shape, &data);

        let result = tensor.sum(vec![]);

        assert_eq!(result.shape(), &shape![1]);
        assert_eq!(result.data, vec![78.0]);
    }

    #[test]
    fn test_tensor_sum_one_axis_1d() {
        let shape = shape![5];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::new(&shape, &data);

        let result = tensor.sum(vec![0]);

        assert_eq!(result.shape(), &shape![1]);
        assert_eq!(result.data, vec![15.0]);
    }

    #[test]
    fn test_tensor_sum_one_axis_2d() {
        let shape = shape![2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data);

        let result = tensor.sum(vec![0]);

        assert_eq!(result.shape(), &shape![3]);
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tensor_sum_one_axis_3d() {
        let shape = shape![2, 2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tensor = Tensor::new(&shape, &data);

        let result = tensor.sum(vec![0]);

        assert_eq!(result.shape(), &shape![2, 3]);
        assert_eq!(result.data, vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
    }

    #[test]
    fn test_tensor_sum_multiple_axes_2d() {
        let shape = shape![2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data);

        let result = tensor.sum(vec![0, 1]);

        assert_eq!(result.shape(), &shape![1]);
        assert_eq!(result.data, vec![21.0]);
    }

    #[test]
    fn test_tensor_sum_multiple_axes_3d() {
        let shape = shape![2, 2, 3];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let tensor = Tensor::new(&shape, &data);

        let result = tensor.sum(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3]);
        assert_eq!(result.data, vec![22.0, 26.0, 30.0]);
    }

    #[test]
    fn test_add_tensor() {
        let shape = shape![4];
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(&shape, &data1);

        let result = tensor1 + 3.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_sub_tensor() {
        let shape = shape![4];
        let data1 = vec![5.0, 6.0, 7.0, 8.0];

        let tensor1 = Tensor::new(&shape, &data1);

        let result = tensor1 - 3.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_mul_tensor() {
        let shape = shape![4];
        let data1 = vec![1.0, 2.0, 3.0, 4.0];

        let tensor1 = Tensor::new(&shape, &data1);

        let result = tensor1 * 2.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_div_tensor() {
        let shape = shape![4];
        let data1 = vec![4.0, 6.0, 8.0, 10.0];

        let tensor1 = Tensor::new(&shape, &data1);

        let result = tensor1 / 2.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, vec![2.0, 3.0, 4.0, 5.0]);
    }
}

