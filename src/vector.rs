use std::ops::{Add, Deref, DerefMut, Div, Index, IndexMut, Mul, Sub};

use crate::coord;
use crate::error::ShapeError;
use crate::matrix::DynamicMatrix;
use crate::shape;
use crate::shape::Shape;
use crate::tensor::DynamicTensor;
use num::Float;
use num::Num;

#[derive(Debug, PartialEq)]
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
        assert_eq!(vector, DynamicVector::new(&data).unwrap());
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
    fn test_fill() {
        let shape = shape![4].unwrap();
        let vector = DynamicVector::fill(&shape, 3.0).unwrap();
        assert_eq!(vector, DynamicVector::new(&[3.0; 4]).unwrap());
    }

    #[test]
    fn test_zeros() {
        let shape = shape![4].unwrap();
        let vector = DynamicVector::<f64>::zeros(&shape).unwrap();
        assert_eq!(vector, DynamicVector::new(&[0.0; 4]).unwrap());
    }

    #[test]
    fn test_ones() {
        let shape = shape![4].unwrap();
        let vector = DynamicVector::<f64>::ones(&shape).unwrap();
        assert_eq!(vector, DynamicVector::new(&[1.0; 4]).unwrap());
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
        assert_eq!(result, DynamicVector::new(&[10.0]).unwrap());
    }

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector.mean();
        assert_eq!(result, DynamicVector::new(&[2.5]).unwrap());
    }

    #[test]
    fn test_var() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector.var();
        assert_eq!(result, DynamicVector::new(&[1.25]).unwrap());
    }

    #[test]
    fn test_min() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector.min();
        assert_eq!(result, DynamicVector::new(&[1.0]).unwrap());
    }

    #[test]
    fn test_max() {
        let data = vec![-1.0, -2.0, -3.0, -4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector.max();
        assert_eq!(result, DynamicVector::new(&[-1.0]).unwrap());
    }

    #[test]
    fn test_vecmul() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1.vecmul(&vector2);
        assert_eq!(result, DynamicVector::new(&[40.0]).unwrap());
    }

    #[test]
    fn test_matmul() {
        let data_vector = vec![1.0, 2.0];
        let data_matrix = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data_vector).unwrap();
        let matrix = DynamicMatrix::new(&shape![2, 2].unwrap(), &data_matrix).unwrap();
        let result = vector.matmul(&matrix);
        assert_eq!(result, DynamicVector::new(&[7.0, 10.0]).unwrap());
    }

    #[test]
    fn test_prod() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1.prod(&vector2);

        let expected_tensor = DynamicTensor::new(&shape![4, 4].unwrap(), &[
            2.0, 3.0, 4.0, 5.0, 4.0, 6.0, 8.0, 10.0, 6.0, 9.0, 12.0, 15.0, 8.0, 12.0, 16.0, 20.0,
        ]).unwrap();
        assert_eq!(result, expected_tensor);
    }

    #[test]
    fn test_add_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector + 2.0;
        assert_eq!(result, DynamicVector::new(&[3.0, 4.0, 5.0, 6.0]).unwrap());
    }

    #[test]
    fn test_add_vector() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1 + vector2;
        assert_eq!(result, DynamicVector::new(&[3.0, 5.0, 7.0, 9.0]).unwrap());
    }

    #[test]
    fn test_add_vector_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector = DynamicVector::new(&data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = vector + tensor;
        assert_eq!(result, DynamicVector::new(&[3.0, 5.0, 7.0, 9.0]).unwrap());
    }

    #[test]
    fn test_sub_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector - 2.0;
        assert_eq!(result, DynamicVector::new(&[-1.0, 0.0, 1.0, 2.0]).unwrap());
    }

    #[test]
    fn test_sub_vector() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1 - vector2;
        assert_eq!(result, DynamicVector::new(&[-1.0; 4]).unwrap());
    }

    #[test]
    fn test_sub_vector_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector = DynamicVector::new(&data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = vector - tensor;
        assert_eq!(result, DynamicVector::new(&[-1.0; 4]).unwrap());
    }

    #[test]
    fn test_mul_scalar() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector * 2.0;
        assert_eq!(result, DynamicVector::new(&[2.0, 4.0, 6.0, 8.0]).unwrap());
    }

    #[test]
    fn test_mul_vector() {
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1 * vector2;
        assert_eq!(result, DynamicVector::new(&[2.0, 6.0, 12.0, 20.0]).unwrap());
    }

    #[test]
    fn test_mul_vector_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector = DynamicVector::new(&data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = vector * tensor;
        assert_eq!(result, DynamicVector::new(&[2.0, 6.0, 12.0, 20.0]).unwrap());
    }

    #[test]
    fn test_div_scalar() {
        let data = vec![4.0, 6.0, 8.0, 10.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector / 2.0;
        assert_eq!(result, DynamicVector::new(&[2.0, 3.0, 4.0, 5.0]).unwrap());
    }

    #[test]
    fn test_div_vector() {
        let data1 = vec![4.0, 6.0, 8.0, 10.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector1 = DynamicVector::new(&data1).unwrap();
        let vector2 = DynamicVector::new(&data2).unwrap();
        let result = vector1 / vector2;
        assert_eq!(result, DynamicVector::new(&[2.0; 4]).unwrap());
    }

    #[test]
    fn test_div_vector_tensor() {
        let shape = shape![4].unwrap();
        let data1 = vec![4.0, 6.0, 8.0, 10.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let vector = DynamicVector::new(&data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = vector / tensor;
        assert_eq!(result, DynamicVector::new(&[2.0; 4]).unwrap());
    }

    #[test]
    fn test_pow_vector() {
        let data = vec![2.0, 3.0, 4.0, 5.0];
        let vector = DynamicVector::new(&data).unwrap();
        let result = vector.pow(2.0);
        assert_eq!(result, DynamicVector::new(&[4.0, 9.0, 16.0, 25.0]).unwrap());
    }
}
