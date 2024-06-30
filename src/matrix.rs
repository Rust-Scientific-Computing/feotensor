use std::ops::{Add, Deref, DerefMut, Div, Index, IndexMut, Mul, Sub};

use crate::axes::Axes;
use crate::coord;
use crate::coordinate::Coordinate;
use crate::error::ShapeError;
use crate::shape;
use crate::shape::Shape;
use crate::tensor::DynamicTensor;
use crate::vector::DynamicVector;
use num::{Float, Num};

pub struct DynamicMatrix<T: Num> {
    tensor: DynamicTensor<T>,
}
pub type Matrix<T> = DynamicMatrix<T>;

impl<T: Num + PartialOrd + Copy> DynamicMatrix<T> {
    pub fn new(shape: &Shape, data: &[T]) -> Result<DynamicMatrix<T>, ShapeError> {
        Ok(DynamicMatrix {
            tensor: DynamicTensor::new(shape, data)?,
        })
    }

    pub fn from_tensor(tensor: DynamicTensor<T>) -> Result<DynamicMatrix<T>, ShapeError> {
        if tensor.shape().order() != 2 {
            return Err(ShapeError::new("Shape must have order of 2"));
        }
        Ok(DynamicMatrix { tensor })
    }

    pub fn fill(shape: &Shape, value: T) -> Result<DynamicMatrix<T>, ShapeError> {
        let data = vec![value; shape.size()];
        DynamicMatrix::new(shape, &data)
    }
    pub fn eye(shape: &Shape) -> Result<DynamicMatrix<T>, ShapeError> {
        let mut result = DynamicMatrix::zeros(shape).unwrap();
        for i in 0..shape[0] {
            result.set(&coord![i, i].unwrap(), T::one()).unwrap();
        }
        Ok(result)
    }
    pub fn zeros(shape: &Shape) -> Result<DynamicMatrix<T>, ShapeError> {
        Self::fill(shape, T::zero())
    }
    pub fn ones(shape: &Shape) -> Result<DynamicMatrix<T>, ShapeError> {
        Self::fill(shape, T::one())
    }

    pub fn sum(&self, axes: Axes) -> DynamicVector<T> {
        let result = self.tensor.sum(axes);
        DynamicVector::from_tensor(result).unwrap()
    }

    pub fn mean(&self, axes: Axes) -> DynamicVector<T> {
        let result = self.tensor.mean(axes);
        DynamicVector::from_tensor(result).unwrap()
    }

    pub fn var(&self, axes: Axes) -> DynamicVector<T> {
        let result = self.tensor.var(axes);
        DynamicVector::from_tensor(result).unwrap()
    }

    pub fn max(&self, axes: Axes) -> DynamicVector<T> {
        let result = self.tensor.max(axes);
        DynamicVector::from_tensor(result).unwrap()
    }

    pub fn min(&self, axes: Axes) -> DynamicVector<T> {
        let result = self.tensor.min(axes);
        DynamicVector::from_tensor(result).unwrap()
    }

    // Vector/Matrix Multiplication
    pub fn matmul(&self, rhs: &Self) -> DynamicMatrix<T> {
        // Matrix-Matrix multiplication
        assert_eq!(self.shape()[1], rhs.shape()[0]);
        let mut result = DynamicTensor::zeros(&shape![self.shape()[0], rhs.shape()[1]].unwrap());
        for i in 0..self.shape()[0] {
            for j in 0..rhs.shape()[1] {
                let mut sum = T::zero();
                for k in 0..self.shape()[1] {
                    sum = sum + self[coord![i, k].unwrap()] * rhs[coord![k, j].unwrap()];
                }
                result.set(&coord![i, j].unwrap(), sum).unwrap();
            }
        }
        DynamicMatrix::from_tensor(result).unwrap()
    }

    pub fn vecmul(&self, rhs: &DynamicVector<T>) -> DynamicVector<T> {
        assert_eq!(self.shape()[1], rhs.shape()[0]);
        let mut result = DynamicTensor::zeros(&shape![self.shape()[0]].unwrap());
        for i in 0..self.shape()[0] {
            let mut sum = T::zero();
            for j in 0..self.shape()[1] {
                sum = sum + self[coord![i, j].unwrap()] * rhs[j];
            }
            result.set(&coord![i].unwrap(), sum).unwrap();
        }
        DynamicVector::from_tensor(result).unwrap()
    }
}

impl<T: Float + PartialOrd + Copy> DynamicMatrix<T> {
    pub fn pow(&self, power: T) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor.pow(power)).unwrap()
    }
}

// Scalar Addition
impl<T: Num + PartialOrd + Copy> Add<T> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn add(self, rhs: T) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor + rhs).unwrap()
    }
}

// Tensor Addition
impl<T: Num + PartialOrd + Copy> Add<DynamicMatrix<T>> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn add(self, rhs: DynamicMatrix<T>) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor + rhs.tensor).unwrap()
    }
}

impl<T: Num + PartialOrd + Copy> Add<DynamicTensor<T>> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn add(self, rhs: DynamicTensor<T>) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor + rhs).unwrap()
    }
}

// Scalar Subtraction
impl<T: Num + PartialOrd + Copy> Sub<T> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn sub(self, rhs: T) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor - rhs).unwrap()
    }
}

// Tensor Subtraction
impl<T: Num + PartialOrd + Copy> Sub<DynamicMatrix<T>> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn sub(self, rhs: DynamicMatrix<T>) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor - rhs.tensor).unwrap()
    }
}

impl<T: Num + PartialOrd + Copy> Sub<DynamicTensor<T>> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn sub(self, rhs: DynamicTensor<T>) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor - rhs).unwrap()
    }
}

// Scalar Multiplication
impl<T: Num + PartialOrd + Copy> Mul<T> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn mul(self, rhs: T) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor * rhs).unwrap()
    }
}

// Tensor Multiplication
impl<T: Num + PartialOrd + Copy> Mul<DynamicMatrix<T>> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn mul(self, rhs: DynamicMatrix<T>) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor * rhs.tensor).unwrap()
    }
}

impl<T: Num + PartialOrd + Copy> Mul<DynamicTensor<T>> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn mul(self, rhs: DynamicTensor<T>) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor * rhs).unwrap()
    }
}

// Scalar Division
impl<T: Num + PartialOrd + Copy> Div<T> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn div(self, rhs: T) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor / rhs).unwrap()
    }
}

// Tensor Division
impl<T: Num + PartialOrd + Copy> Div<DynamicMatrix<T>> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn div(self, rhs: DynamicMatrix<T>) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor / rhs.tensor).unwrap()
    }
}

impl<T: Num + PartialOrd + Copy> Div<DynamicTensor<T>> for DynamicMatrix<T> {
    type Output = DynamicMatrix<T>;

    fn div(self, rhs: DynamicTensor<T>) -> DynamicMatrix<T> {
        DynamicMatrix::from_tensor(self.tensor / rhs).unwrap()
    }
}

impl<T: Num> Deref for DynamicMatrix<T> {
    type Target = DynamicTensor<T>;

    fn deref(&self) -> &DynamicTensor<T> {
        &self.tensor
    }
}

impl<T: Num> DerefMut for DynamicMatrix<T> {
    fn deref_mut(&mut self) -> &mut DynamicTensor<T> {
        &mut self.tensor
    }
}

impl<T: Num + Copy + PartialOrd> Index<Coordinate> for DynamicMatrix<T> {
    type Output = T;

    fn index(&self, index: Coordinate) -> &Self::Output {
        &self.tensor.get(&index).unwrap()
    }
}

impl<T: Num + Copy + PartialOrd> IndexMut<Coordinate> for DynamicMatrix<T> {
    fn index_mut(&mut self, index: Coordinate) -> &mut Self::Output {
        self.tensor.get_mut(&index).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        assert_eq!(matrix.shape(), &shape);
        assert_eq!(matrix[coord![0, 0].unwrap()], 1.0);
        assert_eq!(matrix[coord![0, 1].unwrap()], 2.0);
        assert_eq!(matrix[coord![1, 0].unwrap()], 3.0);
        assert_eq!(matrix[coord![1, 1].unwrap()], 4.0);
    }

    #[test]
    fn test_from_tensor() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = DynamicTensor::new(&shape, &data).unwrap();
        let matrix = DynamicMatrix::from_tensor(tensor).unwrap();
        assert_eq!(matrix.shape(), &shape);
        assert_eq!(matrix[coord![0, 0].unwrap()], 1.0);
        assert_eq!(matrix[coord![0, 1].unwrap()], 2.0);
        assert_eq!(matrix[coord![1, 0].unwrap()], 3.0);
        assert_eq!(matrix[coord![1, 1].unwrap()], 4.0);
    }

    #[test]
    fn test_from_tensor_fail() {
        let shape = shape![2, 2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = DynamicTensor::new(&shape, &data).unwrap();
        let result = DynamicMatrix::from_tensor(tensor);
        assert!(result.is_err());
    }

    #[test]
    fn test_fill() {
        let shape = shape![2, 2].unwrap();
        let matrix = DynamicMatrix::fill(&shape, 3.0).unwrap();
        assert_eq!(matrix.shape(), &shape);
        assert_eq!(matrix[coord![0, 0].unwrap()], 3.0);
        assert_eq!(matrix[coord![0, 1].unwrap()], 3.0);
        assert_eq!(matrix[coord![1, 0].unwrap()], 3.0);
        assert_eq!(matrix[coord![1, 1].unwrap()], 3.0);
    }

    #[test]
    fn test_eye() {
        let shape = shape![3, 3].unwrap();
        let matrix = DynamicMatrix::<f64>::eye(&shape).unwrap();
        assert_eq!(matrix.shape(), &shape);
        assert_eq!(matrix[coord![0, 0].unwrap()], 1.0);
        assert_eq!(matrix[coord![0, 1].unwrap()], 0.0);
        assert_eq!(matrix[coord![0, 2].unwrap()], 0.0);
        assert_eq!(matrix[coord![1, 0].unwrap()], 0.0);
        assert_eq!(matrix[coord![1, 1].unwrap()], 1.0);
        assert_eq!(matrix[coord![1, 2].unwrap()], 0.0);
        assert_eq!(matrix[coord![2, 0].unwrap()], 0.0);
        assert_eq!(matrix[coord![2, 1].unwrap()], 0.0);
        assert_eq!(matrix[coord![2, 2].unwrap()], 1.0);
    }

    #[test]
    fn test_zeros() {
        let shape = shape![2, 2].unwrap();
        let matrix = DynamicMatrix::<f64>::zeros(&shape).unwrap();
        assert_eq!(matrix.shape(), &shape);
        assert_eq!(matrix[coord![0, 0].unwrap()], 0.0);
        assert_eq!(matrix[coord![0, 1].unwrap()], 0.0);
        assert_eq!(matrix[coord![1, 0].unwrap()], 0.0);
        assert_eq!(matrix[coord![1, 1].unwrap()], 0.0);
    }

    #[test]
    fn test_ones() {
        let shape = shape![2, 2].unwrap();
        let matrix = DynamicMatrix::<f64>::ones(&shape).unwrap();
        assert_eq!(matrix.shape(), &shape);
        assert_eq!(matrix[coord![0, 0].unwrap()], 1.0);
        assert_eq!(matrix[coord![0, 1].unwrap()], 1.0);
        assert_eq!(matrix[coord![1, 0].unwrap()], 1.0);
        assert_eq!(matrix[coord![1, 1].unwrap()], 1.0);
    }

    #[test]
    fn test_size() {
        let shape = shape![2, 2].unwrap();
        let matrix = DynamicMatrix::<f64>::zeros(&shape).unwrap();
        assert_eq!(matrix.size(), 4);
    }

    #[test]
    fn test_get() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        assert_eq!(matrix[coord![1, 0].unwrap()], 3.0);
    }

    #[test]
    fn test_get_mut() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut matrix = DynamicMatrix::new(&shape, &data).unwrap();
        matrix[coord![1, 0].unwrap()] = 5.0;
        assert_eq!(matrix.shape(), &shape);
        assert_eq!(matrix[coord![0, 0].unwrap()], 1.0);
        assert_eq!(matrix[coord![0, 1].unwrap()], 2.0);
        assert_eq!(matrix[coord![1, 0].unwrap()], 5.0);
        assert_eq!(matrix[coord![1, 1].unwrap()], 4.0);
    }

    #[test]
    fn test_set() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut matrix = DynamicMatrix::new(&shape, &data).unwrap();
        matrix.set(&coord![1, 0].unwrap(), 5.0).unwrap();
        assert_eq!(matrix.shape(), &shape);
        assert_eq!(matrix[coord![0, 0].unwrap()], 1.0);
        assert_eq!(matrix[coord![0, 1].unwrap()], 2.0);
        assert_eq!(matrix[coord![1, 0].unwrap()], 5.0);
        assert_eq!(matrix[coord![1, 1].unwrap()], 4.0);
    }

    #[test]
    fn test_sum() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        let result = matrix.sum(vec![0, 1]);
        assert_eq!(result[0], 10.0);
        assert_eq!(result.shape(), &shape![1].unwrap());
    }

    #[test]
    fn test_mean() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        let result = matrix.mean(vec![0, 1]);
        assert_eq!(result[0], 2.5);
        assert_eq!(result.shape(), &shape![1].unwrap());
    }

    #[test]
    fn test_var() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        let result = matrix.var(vec![0, 1]);
        assert_eq!(result[0], 1.25);
        assert_eq!(result.shape(), &shape![1].unwrap());
    }

    #[test]
    fn test_min() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        let result = matrix.min(vec![0, 1]);
        assert_eq!(result[0], 1.0);
        assert_eq!(result.shape(), &shape![1].unwrap());
    }

    #[test]
    fn test_max() {
        let shape = shape![2, 2].unwrap();
        let data = vec![-1.0, -2.0, -3.0, -4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        let result = matrix.max(vec![0, 1]);
        assert_eq!(result[0], -1.0);
        assert_eq!(result.shape(), &shape![1].unwrap());
    }

    #[test]
    fn test_matmul() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let matrix1 = DynamicMatrix::new(&shape, &data1).unwrap();
        let matrix2 = DynamicMatrix::new(&shape, &data2).unwrap();
        let result = matrix1.matmul(&matrix2);
        assert_eq!(result.shape(), &shape);
        assert_eq!(result[coord![0, 0].unwrap()], 10.0);
        assert_eq!(result[coord![0, 1].unwrap()], 13.0);
        assert_eq!(result[coord![1, 0].unwrap()], 22.0);
        assert_eq!(result[coord![1, 1].unwrap()], 29.0);
    }

    #[test]
    fn test_vecmul() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        let vector_data = vec![1.0, 2.0];
        let vector = DynamicVector::new(&vector_data).unwrap();
        let result = matrix.vecmul(&vector);
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], 5.0);
        assert_eq!(result[1], 11.0);
    }

    #[test]
    fn test_add_scalar() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        let result = matrix + 2.0;
        assert_eq!(result[coord![0, 0].unwrap()], 3.0);
        assert_eq!(result[coord![0, 1].unwrap()], 4.0);
        assert_eq!(result[coord![1, 0].unwrap()], 5.0);
        assert_eq!(result[coord![1, 1].unwrap()], 6.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_add_matrix() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let matrix1 = DynamicMatrix::new(&shape, &data1).unwrap();
        let matrix2 = DynamicMatrix::new(&shape, &data2).unwrap();
        let result = matrix1 + matrix2;
        assert_eq!(result[coord![0, 0].unwrap()], 3.0);
        assert_eq!(result[coord![0, 1].unwrap()], 5.0);
        assert_eq!(result[coord![1, 0].unwrap()], 7.0);
        assert_eq!(result[coord![1, 1].unwrap()], 9.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_add_matrix_tensor() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let matrix = DynamicMatrix::new(&shape, &data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = matrix + tensor;
        assert_eq!(result[coord![0, 0].unwrap()], 3.0);
        assert_eq!(result[coord![0, 1].unwrap()], 5.0);
        assert_eq!(result[coord![1, 0].unwrap()], 7.0);
        assert_eq!(result[coord![1, 1].unwrap()], 9.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_sub_scalar() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        let result = matrix - 2.0;
        assert_eq!(result[coord![0, 0].unwrap()], -1.0);
        assert_eq!(result[coord![0, 1].unwrap()], 0.0);
        assert_eq!(result[coord![1, 0].unwrap()], 1.0);
        assert_eq!(result[coord![1, 1].unwrap()], 2.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_sub_matrix() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let matrix1 = DynamicMatrix::new(&shape, &data1).unwrap();
        let matrix2 = DynamicMatrix::new(&shape, &data2).unwrap();
        let result = matrix1 - matrix2;
        assert_eq!(result[coord![0, 0].unwrap()], -1.0);
        assert_eq!(result[coord![0, 1].unwrap()], -1.0);
        assert_eq!(result[coord![1, 0].unwrap()], -1.0);
        assert_eq!(result[coord![1, 1].unwrap()], -1.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_sub_matrix_tensor() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let matrix = DynamicMatrix::new(&shape, &data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = matrix - tensor;
        assert_eq!(result[coord![0, 0].unwrap()], -1.0);
        assert_eq!(result[coord![0, 1].unwrap()], -1.0);
        assert_eq!(result[coord![1, 0].unwrap()], -1.0);
        assert_eq!(result[coord![1, 1].unwrap()], -1.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_mul_scalar() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        let result = matrix * 2.0;
        assert_eq!(result[coord![0, 0].unwrap()], 2.0);
        assert_eq!(result[coord![0, 1].unwrap()], 4.0);
        assert_eq!(result[coord![1, 0].unwrap()], 6.0);
        assert_eq!(result[coord![1, 1].unwrap()], 8.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_mul_matrix() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let matrix1 = DynamicMatrix::new(&shape, &data1).unwrap();
        let matrix2 = DynamicMatrix::new(&shape, &data2).unwrap();
        let result = matrix1 * matrix2;
        assert_eq!(result[coord![0, 0].unwrap()], 2.0);
        assert_eq!(result[coord![0, 1].unwrap()], 6.0);
        assert_eq!(result[coord![1, 0].unwrap()], 12.0);
        assert_eq!(result[coord![1, 1].unwrap()], 20.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_mul_matrix_tensor() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let matrix = DynamicMatrix::new(&shape, &data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = matrix * tensor;
        assert_eq!(result[coord![0, 0].unwrap()], 2.0);
        assert_eq!(result[coord![0, 1].unwrap()], 6.0);
        assert_eq!(result[coord![1, 0].unwrap()], 12.0);
        assert_eq!(result[coord![1, 1].unwrap()], 20.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_div_scalar() {
        let shape = shape![2, 2].unwrap();
        let data = vec![4.0, 6.0, 8.0, 10.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        let result = matrix / 2.0;
        assert_eq!(result[coord![0, 0].unwrap()], 2.0);
        assert_eq!(result[coord![0, 1].unwrap()], 3.0);
        assert_eq!(result[coord![1, 0].unwrap()], 4.0);
        assert_eq!(result[coord![1, 1].unwrap()], 5.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_div_matrix() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![4.0, 6.0, 8.0, 10.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let matrix1 = DynamicMatrix::new(&shape, &data1).unwrap();
        let matrix2 = DynamicMatrix::new(&shape, &data2).unwrap();
        let result = matrix1 / matrix2;
        assert_eq!(result[coord![0, 0].unwrap()], 2.0);
        assert_eq!(result[coord![0, 1].unwrap()], 2.0);
        assert_eq!(result[coord![1, 0].unwrap()], 2.0);
        assert_eq!(result[coord![1, 1].unwrap()], 2.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_div_matrix_tensor() {
        let shape = shape![2, 2].unwrap();
        let data1 = vec![4.0, 6.0, 8.0, 10.0];
        let data2 = vec![2.0, 3.0, 4.0, 5.0];
        let matrix = DynamicMatrix::new(&shape, &data1).unwrap();
        let tensor = DynamicTensor::new(&shape, &data2).unwrap();
        let result = matrix / tensor;
        assert_eq!(result[coord![0, 0].unwrap()], 2.0);
        assert_eq!(result[coord![0, 1].unwrap()], 2.0);
        assert_eq!(result[coord![1, 0].unwrap()], 2.0);
        assert_eq!(result[coord![1, 1].unwrap()], 2.0);
        assert_eq!(result.shape(), &shape);
    }

    #[test]
    fn test_index_coord() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        assert_eq!(matrix[coord![0, 0].unwrap()], 1.0);
        assert_eq!(matrix[coord![0, 1].unwrap()], 2.0);
        assert_eq!(matrix[coord![1, 0].unwrap()], 3.0);
        assert_eq!(matrix[coord![1, 1].unwrap()], 4.0);
    }

    #[test]
    fn test_index_mut_coord() {
        let shape = shape![2, 2].unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let mut matrix = DynamicMatrix::new(&shape, &data).unwrap();
        matrix[coord![0, 0].unwrap()] = 5.0;
        assert_eq!(matrix[coord![0, 0].unwrap()], 5.0);
        assert_eq!(matrix[coord![0, 1].unwrap()], 2.0);
        assert_eq!(matrix[coord![1, 0].unwrap()], 3.0);
        assert_eq!(matrix[coord![1, 1].unwrap()], 4.0);
    }

    #[test]
    fn test_pow_matrix() {
        let shape = shape![2, 2].unwrap();
        let data = vec![2.0, 3.0, 4.0, 5.0];
        let matrix = DynamicMatrix::new(&shape, &data).unwrap();
        let result = matrix.pow(2.0);
        assert_eq!(result[coord![0, 0].unwrap()], 4.0);
        assert_eq!(result[coord![0, 1].unwrap()], 9.0);
        assert_eq!(result[coord![1, 0].unwrap()], 16.0);
        assert_eq!(result[coord![1, 1].unwrap()], 25.0);
        assert_eq!(result.shape(), &shape);
    }
}
