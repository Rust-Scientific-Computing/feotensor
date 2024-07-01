use num::{Float, Num};
use std::ops::{Add, Div, Mul, Sub};

use crate::axes::Axes;
use crate::coordinate::Coordinate;
use crate::error::ShapeError;
use crate::iter::IndexIterator;
use crate::matrix::DynamicMatrix;
use crate::shape;
use crate::shape::Shape;
use crate::storage::DynamicStorage;
use crate::vector::DynamicVector;

#[derive(Debug)]
pub struct DynamicTensor<T: Num> {
    data: DynamicStorage<T>,
    shape: Shape,
}
pub type Tensor<T> = DynamicTensor<T>; // Alias for convenience

impl<T: Num + PartialOrd + Copy> Tensor<T> {
    pub fn new(shape: &Shape, data: &[T]) -> Result<Tensor<T>, ShapeError> {
        if data.len() != shape.size() {
            return Err(ShapeError::new("Data length does not match shape size"));
        }
        Ok(Tensor {
            data: DynamicStorage::new(data.to_vec()),
            shape: shape.clone(),
        })
    }

    pub fn fill(shape: &Shape, value: T) -> Tensor<T> {
        let mut vec = Vec::with_capacity(shape.size());
        for _ in 0..shape.size() {
            vec.push(value);
        }
        Tensor::new(shape, &vec).unwrap()
    }
    pub fn zeros(shape: &Shape) -> Tensor<T> {
        Tensor::fill(shape, T::zero())
    }
    pub fn ones(shape: &Shape) -> Tensor<T> {
        Tensor::fill(shape, T::one())
    }

    // Properties
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    pub fn get(&self, coord: &Coordinate) -> Result<&T, ShapeError> {
        Ok(&self.data[self.data.flatten(coord, &self.shape)?])
    }

    pub fn get_mut(&mut self, coord: &Coordinate) -> Result<&mut T, ShapeError> {
        let index = self.data.flatten(coord, &self.shape)?;
        Ok(&mut self.data[index])
    }

    pub fn set(&mut self, coord: &Coordinate, value: T) -> Result<(), ShapeError> {
        let index = self.data.flatten(coord, &self.shape)?;
        self.data[index] = value;
        Ok(())
    }

    // // Reduction operations
    pub fn sum(&self, axes: Axes) -> Tensor<T> {
        let all_axes = (0..self.shape.order()).collect::<Vec<_>>();
        let remaining_axes = all_axes
            .clone()
            .into_iter()
            .filter(|&i| !axes.contains(&i))
            .collect::<Vec<_>>();
        let remaining_dims = remaining_axes
            .iter()
            .map(|&i| self.shape[i])
            .collect::<Vec<_>>();
        let removing_dims = axes.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();

        // We resolve to a scalar value
        if axes.is_empty() | remaining_dims.is_empty() {
            let sum: T = self.data.iter().fold(T::zero(), |acc, x| acc + *x);
            return Tensor::new(&shape![1].unwrap(), &[sum]).unwrap();
        }

        // Create new tensor with right shape
        let new_shape = Shape::new(remaining_dims).unwrap();
        let remove_shape = Shape::new(removing_dims).unwrap();
        let mut t: Tensor<T> = Tensor::zeros(&new_shape);

        for target in IndexIterator::new(&new_shape) {
            let sum_iter = IndexIterator::new(&remove_shape);
            for sum_index in sum_iter {
                let mut indices = target.clone();
                for (i, &axis) in axes.iter().enumerate() {
                    indices = indices.insert(axis, sum_index[i]);
                }

                let value = *t.get(&target).unwrap() + *self.get(&indices).unwrap();
                t.set(&target, value).unwrap();
            }
        }

        t
    }

    pub fn mean(&self, axes: Axes) -> Tensor<T> {
        let removing_dims = axes.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();
        let removing_dims_t: Vec<T> = removing_dims
            .iter()
            .map(|&dim| {
                let mut result = T::zero();
                for _ in 0..dim {
                    result = result + T::one();
                }
                result
            })
            .collect();
        let n = if !removing_dims_t.is_empty() {
            removing_dims_t.iter().fold(T::one(), |acc, x| acc * *x)
        } else {
            let mut sum = T::zero();
            for _ in 0..self.shape().size() {
                sum = sum + T::one();
            }
            sum
        };
        self.sum(axes) / n
    }

    pub fn var(&self, axes: Axes) -> Tensor<T> {
        let removing_dims = axes.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();
        let removing_dims_t: Vec<T> = removing_dims
            .iter()
            .map(|&dim| {
                let mut result = T::zero();
                for _ in 0..dim {
                    result = result + T::one();
                }
                result
            })
            .collect();

        let n = if !removing_dims_t.is_empty() {
            removing_dims_t.iter().fold(T::one(), |acc, x| acc * *x)
        } else {
            let mut sum = T::zero();
            for _ in 0..self.shape().size() {
                sum = sum + T::one();
            }
            sum
        };

        let all_axes = (0..self.shape.order()).collect::<Vec<_>>();
        let remaining_axes = all_axes
            .clone()
            .into_iter()
            .filter(|&i| !axes.contains(&i))
            .collect::<Vec<_>>();
        let remaining_dims = remaining_axes
            .iter()
            .map(|&i| self.shape[i])
            .collect::<Vec<_>>();
        let removing_dims = axes.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();

        // We resolve to a scalar value
        if axes.is_empty() | remaining_dims.is_empty() {
            let avg: T = self.data.iter().fold(T::zero(), |acc, x| acc + *x) / n;
            let var: T = self
                .data
                .iter()
                .map(|&x| (x - avg) * (x - avg))
                .fold(T::zero(), |acc, x| acc + x)
                / n;
            return Tensor::new(&Shape::new(vec![1]).unwrap(), &[var]).unwrap();
        }

        // Create new tensor with right shape
        let new_shape = Shape::new(remaining_dims).unwrap();
        let remove_shape = Shape::new(removing_dims).unwrap();
        let mut t: Tensor<T> = Tensor::zeros(&new_shape);

        for target in IndexIterator::new(&new_shape) {
            let sum_iter = IndexIterator::new(&remove_shape);
            let mean = self.mean(axes.clone());

            for sum_index in sum_iter {
                let mut indices = target.clone();
                for (i, &axis) in axes.iter().enumerate() {
                    indices = indices.insert(axis, sum_index[i]);
                }

                let centered = *self.get(&indices).unwrap() - *mean.get(&target).unwrap();
                let value = *t.get(&target).unwrap() + centered * centered;
                t.set(&target, value).unwrap();
            }
        }

        t / n
    }

    pub fn max(&self, axes: Axes) -> Tensor<T> {
        let all_axes = (0..self.shape.order()).collect::<Vec<_>>();
        let remaining_axes = all_axes
            .clone()
            .into_iter()
            .filter(|&i| !axes.contains(&i))
            .collect::<Vec<_>>();
        let remaining_dims = remaining_axes
            .iter()
            .map(|&i| self.shape[i])
            .collect::<Vec<_>>();
        let removing_dims = axes.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();

        // We resolve to a scalar value
        if axes.is_empty() | remaining_dims.is_empty() {
            let min: T = self
                .data
                .iter()
                .fold(T::zero(), |acc, x| if acc < *x { acc } else { *x });
            let max: T = self
                .data
                .iter()
                .fold(min, |acc, x| if acc > *x { acc } else { *x });
            return Tensor::new(&Shape::new(vec![1]).unwrap(), &[max]).unwrap();
        }

        // Create new tensor with right shape
        let new_shape = Shape::new(remaining_dims).unwrap();
        let remove_shape = Shape::new(removing_dims).unwrap();
        let min: T = self
            .data
            .iter()
            .fold(T::zero(), |acc, x| if acc < *x { acc } else { *x });
        let mut t: Tensor<T> = Tensor::fill(&new_shape, min);

        for target in IndexIterator::new(&new_shape) {
            let max_iter = IndexIterator::new(&remove_shape);
            for max_index in max_iter {
                let mut indices = target.clone();
                for (i, &axis) in axes.iter().enumerate() {
                    indices = indices.insert(axis, max_index[i]);
                }

                if self.get(&indices).unwrap() > t.get(&target).unwrap() {
                    let _ = t.set(&target, *self.get(&indices).unwrap());
                }
            }
        }

        t
    }

    pub fn min(&self, axes: Axes) -> Tensor<T> {
        let all_axes = (0..self.shape.order()).collect::<Vec<_>>();
        let remaining_axes = all_axes
            .clone()
            .into_iter()
            .filter(|&i| !axes.contains(&i))
            .collect::<Vec<_>>();
        let remaining_dims = remaining_axes
            .iter()
            .map(|&i| self.shape[i])
            .collect::<Vec<_>>();
        let removing_dims = axes.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();

        // We resolve to a scalar value
        if axes.is_empty() | remaining_dims.is_empty() {
            let max: T = self
                .data
                .iter()
                .fold(T::zero(), |acc, x| if acc > *x { acc } else { *x });
            let min: T = self
                .data
                .iter()
                .fold(max, |acc, x| if acc < *x { acc } else { *x });
            return Tensor::new(&Shape::new(vec![1]).unwrap(), &[min]).unwrap();
        }

        // Create new tensor with right shape
        let new_shape = Shape::new(remaining_dims).unwrap();
        let remove_shape = Shape::new(removing_dims).unwrap();
        let max: T = self
            .data
            .iter()
            .fold(T::zero(), |acc, x| if acc > *x { acc } else { *x });
        let mut t: Tensor<T> = Tensor::fill(&new_shape, max);

        for target in IndexIterator::new(&new_shape) {
            let min_iter = IndexIterator::new(&remove_shape);
            for min_index in min_iter {
                let mut indices = target.clone();
                for (i, &axis) in axes.iter().enumerate() {
                    indices = indices.insert(axis, min_index[i]);
                }

                if self.get(&indices).unwrap() < t.get(&target).unwrap() {
                    let _ = t.set(&target, *self.get(&indices).unwrap());
                }
            }
        }

        t
    }

    pub fn median(&self, axes: Axes) -> Tensor<T> {
        let all_axes = (0..self.shape.order()).collect::<Vec<_>>();
        let remaining_axes = all_axes
            .clone()
            .into_iter()
            .filter(|&i| !axes.contains(&i))
            .collect::<Vec<_>>();
        let remaining_dims = remaining_axes
            .iter()
            .map(|&i| self.shape[i])
            .collect::<Vec<_>>();
        let removing_dims = axes.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();

        // We resolve to a scalar value
        if axes.is_empty() || remaining_dims.is_empty() {
            let mut data = self.data.iter().copied().collect::<Vec<T>>();
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = data.len() / 2;
            let median = if data.len() % 2 == 0 {
                let two = T::one() + T::one();
                (data[mid - 1] + data[mid]) / two
            } else {
                data[mid]
            };
            return Tensor::new(&Shape::new(vec![1]).unwrap(), &[median]).unwrap();
        }

        // Create new tensor with right shape
        let new_shape = Shape::new(remaining_dims).unwrap();
        let remove_shape = Shape::new(removing_dims).unwrap();
        let mut t: Tensor<T> = Tensor::zeros(&new_shape);

        for target in IndexIterator::new(&new_shape) {
            let mut values = Vec::new();
            let median_iter = IndexIterator::new(&remove_shape);
            for median_index in median_iter {
                let mut indices = target.clone();
                for (i, &axis) in axes.iter().enumerate() {
                    indices = indices.insert(axis, median_index[i]);
                }
                values.push(*self.get(&indices).unwrap());
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = values.len() / 2;
            let median = if values.len() % 2 == 0 {
                let two = T::one() + T::one();
                (values[mid - 1] + values[mid]) / two
            } else {
                values[mid]
            };
            let _ = t.set(&target, median);
        }

        t
    }

    // Tensor Product
    // Consistent with numpy.tensordot(a, b, axis=0)
    pub fn prod(&self, other: &Tensor<T>) -> Tensor<T> {
        let new_shape = self.shape.stack(&other.shape);

        let mut new_data = Vec::with_capacity(self.size() * other.size());
        for &a in &self.data {
            for &b in &other.data {
                new_data.push(a * b);
            }
        }

        Tensor::new(&new_shape, &new_data).unwrap()
    }
}

impl<T: Float + PartialOrd + Copy> Tensor<T> {
    pub fn pow(&self, power: T) -> Tensor<T> {
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i].powf(power);
        }
        result
    }
}

// Element-wise Multiplication
impl<T: Num + PartialOrd + Copy> Mul<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: T) -> Tensor<T> {
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i] * rhs;
        }
        result
    }
}

// Element-wise Addition
impl<T: Num + PartialOrd + Copy> Add<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: T) -> Tensor<T> {
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i] + rhs;
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
            result.data[i] = self.data[i] + rhs.data[i];
        }
        result
    }
}

impl<T: Num + PartialOrd + Copy> Add<DynamicVector<T>> for Tensor<T> {
    type Output = DynamicVector<T>;

    fn add(self, rhs: DynamicVector<T>) -> DynamicVector<T> {
        rhs + self
    }
}

impl<T: Num + PartialOrd + Copy> Add<DynamicMatrix<T>> for Tensor<T> {
    type Output = DynamicMatrix<T>;

    fn add(self, rhs: DynamicMatrix<T>) -> DynamicMatrix<T> {
        rhs + self
    }
}

// Element-wise Subtraction
impl<T: Num + PartialOrd + Copy> Sub<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: T) -> Tensor<T> {
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i] - rhs;
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
            result.data[i] = self.data[i] - rhs.data[i];
        }
        result
    }
}

impl<T: Num + PartialOrd + Copy> Sub<DynamicVector<T>> for Tensor<T> {
    type Output = DynamicVector<T>;

    fn sub(self, rhs: DynamicVector<T>) -> DynamicVector<T> {
        (rhs * (T::zero() - T::one())) + self
    }
}

impl<T: Num + PartialOrd + Copy> Sub<DynamicMatrix<T>> for Tensor<T> {
    type Output = DynamicMatrix<T>;

    fn sub(self, rhs: DynamicMatrix<T>) -> DynamicMatrix<T> {
        (rhs * (T::zero() - T::one())) + self
    }
}

// Tensor Multiplication
impl<T: Num + PartialOrd + Copy> Mul<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Tensor<T>) -> Tensor<T> {
        assert!(self.shape == rhs.shape);
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i] * rhs.data[i];
        }
        result
    }
}

impl<T: Num + PartialOrd + Copy> Mul<DynamicVector<T>> for Tensor<T> {
    type Output = DynamicVector<T>;

    fn mul(self, rhs: DynamicVector<T>) -> DynamicVector<T> {
        rhs * self
    }
}

impl<T: Num + PartialOrd + Copy> Mul<DynamicMatrix<T>> for Tensor<T> {
    type Output = DynamicMatrix<T>;

    fn mul(self, rhs: DynamicMatrix<T>) -> DynamicMatrix<T> {
        rhs * self
    }
}

// Element-wise Division
impl<T: Num + PartialOrd + Copy> Div<T> for Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: T) -> Tensor<T> {
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i] / rhs;
        }
        result
    }
}

// Tensor Division
impl<T: Num + PartialOrd + Copy> Div<Tensor<T>> for Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Tensor<T>) -> Tensor<T> {
        assert!(self.shape == rhs.shape);
        let mut result = Tensor::zeros(&self.shape);
        for i in 0..self.size() {
            result.data[i] = self.data[i] / rhs.data[i];
        }
        result
    }
}

impl<T: Num + PartialOrd + Copy> Div<DynamicVector<T>> for Tensor<T> {
    type Output = DynamicVector<T>;

    fn div(self, rhs: DynamicVector<T>) -> DynamicVector<T> {
        DynamicVector::<T>::from_tensor(self).unwrap() / rhs
    }
}

impl<T: Num + PartialOrd + Copy> Div<DynamicMatrix<T>> for Tensor<T> {
    type Output = DynamicMatrix<T>;

    fn div(self, rhs: DynamicMatrix<T>) -> DynamicMatrix<T> {
        DynamicMatrix::<T>::from_tensor(self).unwrap() / rhs
    }
}

impl<T: Num + PartialOrd + Copy + std::fmt::Display> Tensor<T> {
    pub fn display(&self) -> String {
        fn format_tensor<T: Num + PartialOrd + Copy + std::fmt::Display>(
            data: &DynamicStorage<T>,
            shape: &Shape,
            level: usize,
        ) -> String {
            if shape.order() == 1 {
                let mut result = String::from("[");
                for (i, item) in data.iter().enumerate() {
                    result.push_str(&format!("{}", item));
                    if i < data.size() - 1 {
                        result.push_str(", ");
                    }
                }
                result.push(']');
                return result;
            }

            let mut result = String::from("[");
            let sub_size = Shape::new(shape[1..].to_vec()).unwrap().size();
            for i in 0..shape[0] {
                if i > 0 {
                    result.push_str(",\n");
                    for _ in 0..shape.order() - 2 {
                        result.push('\n');
                    }
                    for _ in 0..level {
                        result.push(' ');
                    }
                }
                let sub_data = DynamicStorage::new(data[i * sub_size..(i + 1) * sub_size].to_vec());
                result.push_str(&format_tensor(
                    &sub_data,
                    &Shape::new(shape[1..].to_vec()).unwrap(),
                    level + 1,
                ));
            }
            result.push(']');
            result
        }

        format_tensor(&self.data, &self.shape, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coord;

    #[test]
    fn test_new_tensor() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let tensor = Tensor::new(&shape, &data).unwrap();

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data, DynamicStorage::new(data));
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
        assert_eq!(tensor.data, DynamicStorage::new(vec![0.0; shape.size()]));
    }

    #[test]
    fn test_ones_tensor() {
        let shape = shape![2, 3].unwrap();
        let tensor: Tensor<f32> = Tensor::ones(&shape);

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data, DynamicStorage::new(vec![1.0; shape.size()]));
    }

    #[test]
    fn test_fill_tensor() {
        let shape = shape![2, 3].unwrap();
        let tensor: Tensor<f32> = Tensor::fill(&shape, 7.0);

        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.data, DynamicStorage::new(vec![7.0; shape.size()]));
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
    fn test_tensor_get() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        assert_eq!(*tensor.get(&coord![0, 0].unwrap()).unwrap(), 1.0);
        assert_eq!(*tensor.get(&coord![0, 1].unwrap()).unwrap(), 2.0);
        assert_eq!(*tensor.get(&coord![1, 0].unwrap()).unwrap(), 3.0);
        assert_eq!(*tensor.get(&coord![1, 1].unwrap()).unwrap(), 4.0);
    }

    #[test]
    fn test_tensor_set() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = Tensor::new(&shape, &data).unwrap();

        tensor.set(&coord![0, 0].unwrap(), 5.0).unwrap();
        tensor.set(&coord![0, 1].unwrap(), 6.0).unwrap();
        tensor.set(&coord![1, 0].unwrap(), 7.0).unwrap();
        tensor.set(&coord![1, 1].unwrap(), 8.0).unwrap();

        assert_eq!(*tensor.get(&coord![0, 0].unwrap()).unwrap(), 5.0);
        assert_eq!(*tensor.get(&coord![0, 1].unwrap()).unwrap(), 6.0);
        assert_eq!(*tensor.get(&coord![1, 0].unwrap()).unwrap(), 7.0);
        assert_eq!(*tensor.get(&coord![1, 1].unwrap()).unwrap(), 8.0);
    }

    #[test]
    fn test_tensor_get_out_of_bounds() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        assert!(tensor.get(&coord![2, 0].unwrap()).is_err());
        assert!(tensor.get(&coord![0, 2].unwrap()).is_err());
        assert!(tensor.get(&coord![2, 2].unwrap()).is_err());
    }

    #[test]
    fn test_tensor_set_out_of_bounds() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut tensor = Tensor::new(&shape, &data).unwrap();

        assert!(tensor.set(&coord![2, 0].unwrap(), 5.0).is_err());
        assert!(tensor.set(&coord![0, 2].unwrap(), 6.0).is_err());
        assert!(tensor.set(&coord![2, 2].unwrap(), 7.0).is_err());
    }

    #[test]
    fn test_tensor_sum_no_axis_1d() {
        let shape = shape![5].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![15.0]));
    }

    #[test]
    fn test_tensor_sum_no_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![21.0]));
    }

    #[test]
    fn test_tensor_sum_no_axis_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![78.0]));
    }

    #[test]
    fn test_tensor_sum_one_axis_1d() {
        let shape = shape![5].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![0]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![15.0]));
    }

    #[test]
    fn test_tensor_sum_one_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![0]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![5.0, 7.0, 9.0]));
    }

    #[test]
    fn test_tensor_sum_one_axis_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![0]);

        assert_eq!(result.shape(), &shape![2, 3].unwrap());
        assert_eq!(
            result.data,
            DynamicStorage::new(vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
        );
    }

    #[test]
    fn test_tensor_sum_multiple_axes_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![0, 1]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![21.0]));
    }

    #[test]
    fn test_tensor_sum_multiple_axes_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.sum(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![22.0, 26.0, 30.0]));
    }

    #[test]
    fn test_tensor_mean_no_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.mean(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![3.5]));
    }

    #[test]
    fn test_tensor_mean_one_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.mean(vec![0]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![2.5, 3.5, 4.5]));
    }

    #[test]
    fn test_tensor_mean_one_axis_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.mean(vec![0]);

        assert_eq!(result.shape(), &shape![2, 3].unwrap());
        assert_eq!(
            result.data,
            DynamicStorage::new(vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        );
    }

    #[test]
    fn test_tensor_mean_multiple_axes_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.mean(vec![0, 1]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![3.5]));
    }

    #[test]
    fn test_tensor_mean_multiple_axes_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.mean(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![5.5, 6.5, 7.5]));
    }

    #[test]
    fn test_tensor_var_no_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 1.0, 1.0, 7.0, 7.0, 7.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.var(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![9.0]));
    }

    #[test]
    fn test_tensor_var_one_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.var(vec![0]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![2.25, 2.25, 2.25]));
    }

    #[test]
    fn test_tensor_var_one_axis_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.var(vec![0]);

        assert_eq!(result.shape(), &shape![2, 3].unwrap());
        assert_eq!(
            result.data,
            DynamicStorage::new(vec![9.0, 9.0, 9.0, 9.0, 9.0, 9.0])
        );
    }

    #[test]
    fn test_tensor_var_multiple_axes_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, 1.0, 1.0, 7.0, 7.0, 7.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.var(vec![0, 1]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![9.0]));
    }

    #[test]
    fn test_tensor_var_multiple_axes_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![
            1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0,
        ];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.var(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![45.0, 45.0, 45.0]));
    }

    #[test]
    fn test_tensor_max_no_axis_1d() {
        let shape = shape![5].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.max(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![5.0]));
    }

    #[test]
    fn test_tensor_max_one_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.max(vec![0]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![1.0, 5.0, 3.0]));
    }

    #[test]
    fn test_tensor_max_multiple_axes_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0,
        ];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.max(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![7.0, 11.0, 9.0]));
    }

    #[test]
    fn test_tensor_min_no_axis_1d() {
        let shape = shape![5].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.min(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![-4.0]));
    }

    #[test]
    fn test_tensor_min_one_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.min(vec![0]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![-4.0, -2.0, -6.0]));
    }

    #[test]
    fn test_tensor_min_multiple_axes_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0,
        ];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.min(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![-10.0, -8.0, -12.0]));
    }

    #[test]
    fn test_tensor_median_no_axis_1d_odd() {
        let shape = shape![5].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.median(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![1.0]));
    }

    #[test]
    fn test_tensor_median_no_axis_1d_even() {
        let shape = shape![6].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, 6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.median(vec![]);

        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![2.0]));
    }

    #[test]
    fn test_tensor_median_one_axis_2d() {
        let shape = shape![2, 3].unwrap();
        let data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.median(vec![0]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![-1.5, 1.5, -1.5]));
    }

    #[test]
    fn test_tensor_median_multiple_axes_3d() {
        let shape = shape![2, 2, 3].unwrap();
        let data = vec![
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0,
        ];
        let tensor = Tensor::new(&shape, &data).unwrap();

        let result = tensor.median(vec![0, 1]);

        assert_eq!(result.shape(), &shape![3].unwrap());
        assert_eq!(result.data, DynamicStorage::new(vec![-1.5, 1.5, -1.5]));
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
        assert_eq!(
            result.data,
            DynamicStorage::new(vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0])
        );
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
        assert_eq!(
            result.data,
            DynamicStorage::new(vec![5.0, 6.0, 10.0, 12.0, 15.0, 18.0, 20.0, 24.0])
        );
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
        assert_eq!(
            result.data,
            DynamicStorage::new(vec![
                5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 15.0, 18.0, 21.0, 24.0, 20.0, 24.0,
                28.0, 32.0
            ])
        );
    }

    #[test]
    fn test_add_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = Tensor::new(&shape, &data1).unwrap();

        let result = tensor1 + 3.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, DynamicStorage::new(vec![4.0, 5.0, 6.0, 7.0]));
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
        assert_eq!(result.data, DynamicStorage::new(vec![6.0, 8.0, 10.0, 12.0]));
    }

    #[test]
    fn test_sub_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![5.0, 6.0, 7.0, 8.0];

        let tensor1 = Tensor::new(&shape, &data1).unwrap();

        let result = tensor1 - 3.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, DynamicStorage::new(vec![2.0, 3.0, 4.0, 5.0]));
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
        assert_eq!(result.data, DynamicStorage::new(vec![4.0, 4.0, 4.0, 4.0]));
    }

    #[test]
    fn test_mul_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];

        let tensor1 = Tensor::new(&shape, &data1).unwrap();

        let result = tensor1 * 2.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, DynamicStorage::new(vec![2.0, 4.0, 6.0, 8.0]));
    }

    #[test]
    fn test_div_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![4.0, 6.0, 8.0, 10.0];

        let tensor1 = Tensor::new(&shape, &data1).unwrap();

        let result = tensor1 / 2.0;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, DynamicStorage::new(vec![2.0, 3.0, 4.0, 5.0]));
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
        assert_eq!(result.data, DynamicStorage::new(vec![10.0]));
    }

    #[test]
    fn test_vec_vec_mul() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];

        let tensor1 = Tensor::new(&shape, &data1).unwrap();
        let tensor2 = Tensor::new(&shape, &data2).unwrap();

        let result = tensor1 * tensor2;

        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, DynamicStorage::new(vec![2.0, 6.0, 12.0, 20.0]));
    }

    #[test]
    fn test_matrix_matrix_mul() {
        let shape1 = shape![2, 3].unwrap();
        let shape2 = shape![2, 3].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data2 = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let tensor1 = Tensor::new(&shape1, &data1).unwrap();
        let tensor2 = Tensor::new(&shape2, &data2).unwrap();

        let result = tensor1 * tensor2;

        assert_eq!(result.shape(), &shape![2, 3].unwrap());
        assert_eq!(
            result.data,
            DynamicStorage::new(vec![7.0, 16.0, 27.0, 40.0, 55.0, 72.0])
        );
    }

    #[test]
    fn test_add_tensor_vector() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let tensor = DynamicTensor::new(&shape, &data1).unwrap();
        let vector = DynamicVector::new(&data2).unwrap();
        let result = tensor + vector;
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 7.0);
        assert_eq!(result[3], 9.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_sub_tensor_vector() {
        let shape = shape![4].unwrap();
        let data1 = vec![2.0, 3.0, 4.0, 5.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = DynamicTensor::new(&shape, &data1).unwrap();
        let vector = DynamicVector::new(&data2).unwrap();
        let result = tensor - vector;
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_mul_tensor_vector() {
        let shape = shape![4].unwrap();
        let data1 = vec![2.0, 3.0, 4.0, 5.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = DynamicTensor::new(&shape, &data1).unwrap();
        let vector = DynamicVector::new(&data2).unwrap();
        let result = tensor * vector;
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 6.0);
        assert_eq!(result[2], 12.0);
        assert_eq!(result[3], 20.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_div_tensor_vector() {
        let shape = shape![4].unwrap();
        let data1 = vec![2.0, 4.0, 6.0, 8.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = DynamicTensor::new(&shape, &data1).unwrap();
        let vector = DynamicVector::new(&data2).unwrap();
        let result = tensor / vector;
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 2.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_add_tensor_matrix() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let tensor = DynamicTensor::new(&shape, &data1).unwrap();
        let matrix = DynamicMatrix::new(&shape, &data2).unwrap();
        let result = tensor + matrix;
        assert_eq!(result[coord![0, 0].unwrap()], 3.0);
        assert_eq!(result[coord![0, 1].unwrap()], 5.0);
        assert_eq!(result[coord![1, 0].unwrap()], 7.0);
        assert_eq!(result[coord![1, 1].unwrap()], 9.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_sub_tensor_matrix() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![2.0, 3.0, 4.0, 5.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = DynamicTensor::new(&shape, &data1).unwrap();
        let matrix = DynamicMatrix::new(&shape, &data2).unwrap();
        let result = tensor - matrix;
        assert_eq!(result[coord![0, 0].unwrap()], 1.0);
        assert_eq!(result[coord![0, 1].unwrap()], 1.0);
        assert_eq!(result[coord![1, 0].unwrap()], 1.0);
        assert_eq!(result[coord![1, 1].unwrap()], 1.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_mul_tensor_matrix() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![2.0, 3.0, 4.0, 5.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = DynamicTensor::new(&shape, &data1).unwrap();
        let matrix = DynamicMatrix::new(&shape, &data2).unwrap();
        let result = tensor * matrix;
        assert_eq!(result[coord![0, 0].unwrap()], 2.0);
        assert_eq!(result[coord![0, 1].unwrap()], 6.0);
        assert_eq!(result[coord![1, 0].unwrap()], 12.0);
        assert_eq!(result[coord![1, 1].unwrap()], 20.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_div_tensor_matrix() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![2.0, 4.0, 6.0, 8.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = DynamicTensor::new(&shape, &data1).unwrap();
        let matrix = DynamicMatrix::new(&shape, &data2).unwrap();
        let result = tensor / matrix;
        assert_eq!(result[coord![0, 0].unwrap()], 2.0);
        assert_eq!(result[coord![0, 1].unwrap()], 2.0);
        assert_eq!(result[coord![1, 0].unwrap()], 2.0);
        assert_eq!(result[coord![1, 1].unwrap()], 2.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_display_1d_tensor() {
        let shape = shape![3].unwrap();
        let data = vec![1.0, 2.0, 3.0];
        let tensor = Tensor::new(&shape, &data).unwrap();
        let display = tensor.display();
        assert_eq!(display, "[1, 2, 3]");
    }

    #[test]
    fn test_display_2d_tensor() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&shape, &data).unwrap();
        let display = tensor.display();
        assert_eq!(display, "[[1, 2],\n [3, 4]]");
    }

    #[test]
    fn test_display_3d_tensor() {
        let shape = shape![2, 2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::new(&shape, &data).unwrap();
        let display = tensor.display();
        assert_eq!(display, "[[[1, 2],\n  [3, 4]],\n\n [[5, 6],\n  [7, 8]]]");
    }

    #[test]
    fn test_display_4d_tensor() {
        let shape = shape![2, 2, 2, 2].unwrap();
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let tensor = Tensor::new(&shape, &data).unwrap();
        let display = tensor.display();
        assert_eq!(display, "[[[[1, 2],\n   [3, 4]],\n\n  [[5, 6],\n   [7, 8]]],\n\n\n [[[9, 10],\n   [11, 12]],\n\n  [[13, 14],\n   [15, 16]]]]");
    }

    #[test]
    fn test_pow_tensor_square() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = Tensor::new(&shape, &data).unwrap();
        let result = tensor.pow(2.0);
        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, DynamicStorage::new(vec![1.0, 4.0, 9.0, 16.0]));
    }

    #[test]
    fn test_pow_tensor_sqrt() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 4.0, 9.0, 16.0];
        let tensor = Tensor::new(&shape, &data).unwrap();
        let result = tensor.pow(0.5);
        assert_eq!(result.shape(), &shape);
        assert_eq!(result.data, DynamicStorage::new(vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_pow_tensor_negative_exponent() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 4.0, 8.0];
        let tensor = Tensor::new(&shape, &data).unwrap();
        let result = tensor.pow(-1.0);
        assert_eq!(result.shape(), &shape);
        assert_eq!(
            result.data,
            DynamicStorage::new(vec![1.0, 0.5, 0.25, 0.125])
        );
    }
}
