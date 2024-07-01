use std::ops::{Add, Deref, DerefMut, Div, Index, IndexMut, Mul, Sub};

use crate::coord;
use crate::error::ShapeError;
use crate::matrix::DynamicMatrix;
use crate::shape;
use crate::shape::Shape;
use crate::tensor::DynamicTensor;
use num::Float;
use num::Num;

pub struct DynamicVector<T: Num> {
    tensor: DynamicTensor<T>,
}
pub type Vector<T> = DynamicVector<T>;

impl<T: Num + PartialOrd + Copy> DynamicVector<T> {
    pub fn new(data: &[T]) -> Result<DynamicVector<T>, ShapeError> {
        Ok(DynamicVector {
            tensor: DynamicTensor::new(&shape![data.len()].unwrap(), data)?,
        })
    }

    pub fn from_tensor(tensor: DynamicTensor<T>) -> Result<DynamicVector<T>, ShapeError> {
        if tensor.shape().order() != 1 {
            return Err(ShapeError::new("Shape must have order of 1"));
        }
        Ok(DynamicVector { tensor })
    }

    pub fn tile(tensor: &DynamicVector<T>, reps: &Shape) -> Result<DynamicVector<T>, ShapeError> {
        let result = DynamicTensor::tile(tensor, reps)?;
        Ok(DynamicVector { tensor: result })
    }

    pub fn fill(shape: &Shape, value: T) -> Result<DynamicVector<T>, ShapeError> {
        if shape.order() != 1 {
            return Err(ShapeError::new("Shape must have order of 1"));
        }
        let data = vec![value; shape[0]];
        DynamicVector::new(&data)
    }
    pub fn zeros(shape: &Shape) -> Result<DynamicVector<T>, ShapeError> {
        Self::fill(shape, T::zero())
    }
    pub fn ones(shape: &Shape) -> Result<DynamicVector<T>, ShapeError> {
        Self::fill(shape, T::one())
    }

    pub fn sum(&self) -> DynamicVector<T> {
        let result = self.tensor.sum(vec![]);
        DynamicVector::from_tensor(result).unwrap()
    }

    pub fn mean(&self) -> DynamicVector<T> {
        let result = self.tensor.mean(vec![]);
        DynamicVector::from_tensor(result).unwrap()
    }

    pub fn var(&self) -> DynamicVector<T> {
        let result = self.tensor.var(vec![]);
        DynamicVector::from_tensor(result).unwrap()
    }

    pub fn max(&self) -> DynamicVector<T> {
        let result = self.tensor.max(vec![]);
        DynamicVector::from_tensor(result).unwrap()
    }

    pub fn min(&self) -> DynamicVector<T> {
        let result = self.tensor.min(vec![]);
        DynamicVector::from_tensor(result).unwrap()
    }

    // Vector/Matrix Multiplication
    pub fn vecmul(&self, rhs: &DynamicVector<T>) -> DynamicVector<T> {
        assert!(self.shape() == rhs.shape());
        let mut result = T::zero();
        for i in 0..self.size() {
            result = result + self[i] * rhs[i];
        }
        DynamicVector::new(&[result]).unwrap()
    }

    pub fn matmul(&self, rhs: &DynamicMatrix<T>) -> DynamicVector<T> {
        assert_eq!(self.shape()[0], rhs.shape()[0]);
        let mut result = DynamicTensor::zeros(&shape![rhs.shape()[1]].unwrap());
        for j in 0..rhs.shape()[1] {
            let mut sum = T::zero();
            for i in 0..self.shape()[0] {
                sum = sum + self[i] * rhs[coord![i, j].unwrap()];
            }
            result.set(&coord![j].unwrap(), sum).unwrap();
        }
        DynamicVector::from_tensor(result).unwrap()
    }
}

impl<T: Float + PartialOrd + Copy> DynamicVector<T> {
    pub fn pow(&self, power: T) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor.pow(power)).unwrap()
    }
}

// Scalar Addition
impl<T: Num + PartialOrd + Copy> Add<T> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn add(self, rhs: T) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor + rhs).unwrap()
    }
}

// Tensor Addition
impl<T: Num + PartialOrd + Copy> Add<DynamicVector<T>> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn add(self, rhs: DynamicVector<T>) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor + rhs.tensor).unwrap()
    }
}

impl<T: Num + PartialOrd + Copy> Add<DynamicTensor<T>> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn add(self, rhs: DynamicTensor<T>) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor + rhs).unwrap()
    }
}

// Scalar Subtraction
impl<T: Num + PartialOrd + Copy> Sub<T> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn sub(self, rhs: T) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor - rhs).unwrap()
    }
}

// Tensor Subtraction
impl<T: Num + PartialOrd + Copy> Sub<DynamicVector<T>> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn sub(self, rhs: DynamicVector<T>) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor - rhs.tensor).unwrap()
    }
}

impl<T: Num + PartialOrd + Copy> Sub<DynamicTensor<T>> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn sub(self, rhs: DynamicTensor<T>) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor - rhs).unwrap()
    }
}

// Scalar Multiplication
impl<T: Num + PartialOrd + Copy> Mul<T> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn mul(self, rhs: T) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor * rhs).unwrap()
    }
}

// Tensor Multiplication
impl<T: Num + PartialOrd + Copy> Mul<DynamicVector<T>> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn mul(self, rhs: DynamicVector<T>) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor * rhs.tensor).unwrap()
    }
}

impl<T: Num + PartialOrd + Copy> Mul<DynamicTensor<T>> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn mul(self, rhs: DynamicTensor<T>) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor * rhs).unwrap()
    }
}

// Scalar Division
impl<T: Num + PartialOrd + Copy> Div<T> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn div(self, rhs: T) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor / rhs).unwrap()
    }
}

// Tensor Division
impl<T: Num + PartialOrd + Copy> Div<DynamicVector<T>> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn div(self, rhs: DynamicVector<T>) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor / rhs.tensor).unwrap()
    }
}

impl<T: Num + PartialOrd + Copy> Div<DynamicTensor<T>> for DynamicVector<T> {
    type Output = DynamicVector<T>;

    fn div(self, rhs: DynamicTensor<T>) -> DynamicVector<T> {
        DynamicVector::from_tensor(self.tensor / rhs).unwrap()
    }
}

impl<T: Num> Deref for DynamicVector<T> {
    type Target = DynamicTensor<T>;

    fn deref(&self) -> &DynamicTensor<T> {
        &self.tensor
    }
}

impl<T: Num> DerefMut for DynamicVector<T> {
    fn deref_mut(&mut self) -> &mut DynamicTensor<T> {
        &mut self.tensor
    }
}

impl<T: Num + Copy + PartialOrd> Index<usize> for DynamicVector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.tensor.get(&coord![index].unwrap()).unwrap()
    }
}

impl<T: Num + Copy + PartialOrd> IndexMut<usize> for DynamicVector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.tensor.get_mut(&coord![index].unwrap()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let shape = shape![4].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        assert_eq!(vector.shape(), &shape);
        assert_eq!(vector[0], 1.0);
        assert_eq!(vector[1], 2.0);
        assert_eq!(vector[2], 3.0);
        assert_eq!(vector[3], 4.0);
    }

    #[test]
    fn test_from_tensor() {
        let shape = shape![4].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = DynamicTensor::new(&shape, &data).unwrap();
        let vector = DynamicVector::from_tensor(tensor).unwrap();
        assert_eq!(vector.shape(), &shape);
        assert_eq!(vector[0], 1.0);
        assert_eq!(vector[1], 2.0);
        assert_eq!(vector[2], 3.0);
        assert_eq!(vector[3], 4.0);
    }

    #[test]
    fn test_from_tensor_fail() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = DynamicTensor::new(&shape, &data).unwrap();
        let result = DynamicVector::from_tensor(tensor);
        assert!(result.is_err());
    }

    #[test]
    fn test_tile() {
        let data = vec![1.0, 2.0];
        let vector = DynamicVector::new(&data).unwrap();
        let reps = shape![3].unwrap();

        let result = DynamicVector::tile(&vector, &reps).unwrap();

        assert_eq!(result.shape(), &shape![6].unwrap());
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 2.0);
        assert_eq!(result[4], 1.0);
        assert_eq!(result[5], 2.0);
    }

    #[test]
    fn test_fill() {
        let shape = shape![4].unwrap();
        let vector = DynamicVector::fill(&shape, 3.0).unwrap();
        assert_eq!(vector.shape(), &shape);
        assert_eq!(vector[0], 3.0);
        assert_eq!(vector[1], 3.0);
        assert_eq!(vector[2], 3.0);
        assert_eq!(vector[3], 3.0);
    }

    #[test]
    fn test_zeros() {
        let shape = shape![4].unwrap();
        let vector = DynamicVector::<f64>::zeros(&shape).unwrap();
        assert_eq!(vector.shape(), &shape);
        assert_eq!(vector[0], 0.0);
        assert_eq!(vector[1], 0.0);
        assert_eq!(vector[2], 0.0);
        assert_eq!(vector[3], 0.0);
    }

    #[test]
    fn test_ones() {
        let shape = shape![4].unwrap();
        let vector = DynamicVector::<f64>::ones(&shape).unwrap();
        assert_eq!(vector.shape(), &shape);
        assert_eq!(vector[0], 1.0);
        assert_eq!(vector[1], 1.0);
        assert_eq!(vector[2], 1.0);
        assert_eq!(vector[3], 1.0);
    }

    #[test]
    fn test_size() {
        let shape = shape![4].unwrap();
        let vector = DynamicVector::<f64>::zeros(&shape).unwrap();
        assert_eq!(vector.size(), 4);
    }

    #[test]
    fn test_get() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        assert_eq!(vector[2], 3.0);
    }

    #[test]
    fn test_get_mut() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut vector = DynamicVector::new(&data).unwrap();
        vector[2] = 5.0;
        assert_eq!(vector[2], 5.0);
    }

    #[test]
    fn test_set() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut vector = DynamicVector::new(&data).unwrap();
        vector.set(&coord![2].unwrap(), 5.0).unwrap();
        assert_eq!(*vector.get(&coord![2].unwrap()).unwrap(), 5.0);
    }

    #[test]
    fn test_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector.sum();
        assert_eq!(result[0], 10.0);
        assert_eq!(result.shape(), &shape![1].unwrap());
    }

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector.mean();
        assert_eq!(result[0], 2.5);
        assert_eq!(result.shape(), &shape![1].unwrap());
    }

    #[test]
    fn test_var() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector.var();
        assert_eq!(result[0], 1.25);
        assert_eq!(result.shape(), &shape![1].unwrap());
    }

    #[test]
    fn test_min() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector.min();
        assert_eq!(result[0], 1.0);
        assert_eq!(result.shape(), &shape![1].unwrap());
    }

    #[test]
    fn test_max() {
        let data = vec![-1.0, -2.0, -3.0, -4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector.max();
        assert_eq!(result[0], -1.0);
        assert_eq!(result.shape(), &shape![1].unwrap());
    }

    #[test]
    fn test_vecmul() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1.vecmul(&vector2);
        assert_eq!(result[0], 40.0);
        assert_eq!(result.shape(), &shape![1].unwrap());
    }

    #[test]
    fn test_matmul() {
        let data_vector = vec![1.0, 2.0];
        let data_matrix = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data_vector).unwrap();
        let matrix = DynamicMatrix::new(&shape![2, 2].unwrap(), &data_matrix).unwrap();
        let result = vector.matmul(&matrix);
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], 7.0);
        assert_eq!(result[1], 10.0);
    }

    #[test]
    fn test_prod() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1.prod(&vector2);

        let expected_data = vec![
            2.0, 3.0, 4.0, 5.0, 4.0, 6.0, 8.0, 10.0, 6.0, 9.0, 12.0, 15.0, 8.0, 12.0, 16.0, 20.0,
        ];
        let expected_shape = shape![4, 4].unwrap();
        let expected_tensor = DynamicTensor::new(&expected_shape, &expected_data).unwrap();

        assert_eq!(result.shape(), &expected_shape);
        for i in 0..result.shape()[0] {
            for j in 0..result.shape()[1] {
                let x = result.get(&coord![i, j].unwrap()).unwrap();
                let y = expected_tensor.get(&coord![i, j].unwrap()).unwrap();
                assert_eq!(*x, *y);
            }
        }
    }

    #[test]
    fn test_add_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector + 2.0;
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 4.0);
        assert_eq!(result[2], 5.0);
        assert_eq!(result[3], 6.0);
        assert_eq!(result.shape(), &shape![4].unwrap());
    }

    #[test]
    fn test_add_vector() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1 + vector2;
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 7.0);
        assert_eq!(result[3], 9.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_add_vector_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector = DynamicVector::new(&data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = vector + tensor;
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 7.0);
        assert_eq!(result[3], 9.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_sub_scalar() {
        let shape = shape![4].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector - 2.0;
        assert_eq!(result[0], -1.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 2.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_sub_vector() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1 - vector2;
        assert_eq!(result[0], -1.0);
        assert_eq!(result[1], -1.0);
        assert_eq!(result[2], -1.0);
        assert_eq!(result[3], -1.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_sub_vector_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector = DynamicVector::new(&data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = vector - tensor;
        assert_eq!(result[0], -1.0);
        assert_eq!(result[1], -1.0);
        assert_eq!(result[2], -1.0);
        assert_eq!(result[3], -1.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_mul_scalar() {
        let shape = shape![4].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector * 2.0;
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 4.0);
        assert_eq!(result[2], 6.0);
        assert_eq!(result[3], 8.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_mul_vector() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1 * vector2;
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 6.0);
        assert_eq!(result[2], 12.0);
        assert_eq!(result[3], 20.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_mul_vector_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector = DynamicVector::new(&data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = vector * tensor;
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 6.0);
        assert_eq!(result[2], 12.0);
        assert_eq!(result[3], 20.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_div_scalar() {
        let shape = shape![4].unwrap();
        let data = vec![4.0, 6.0, 8.0, 10.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector / 2.0;
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 4.0);
        assert_eq!(result[3], 5.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_div_vector() {
        let shape = shape![4].unwrap();
        let data1 = vec![4.0, 6.0, 8.0, 10.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1 / vector2;
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 2.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_div_vector_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![4.0, 6.0, 8.0, 10.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector = DynamicVector::new(&data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = vector / tensor;
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 2.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_pow_vector() {
        let shape = shape![4].unwrap();
        let data = vec![2.0, 3.0, 4.0, 5.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector.pow(2.0);
        assert_eq!(result[0], 4.0);
        assert_eq!(result[1], 9.0);
        assert_eq!(result[2], 16.0);
        assert_eq!(result[3], 25.0);
        assert_eq!(result.shape(), &shape);
    }
}
