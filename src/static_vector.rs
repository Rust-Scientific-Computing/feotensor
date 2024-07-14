use std::ops::{Add, Deref, DerefMut, Div, Index, IndexMut, Mul, Sub};

use crate::error::ShapeError;
use crate::shape;
use crate::static_matrix::StaticMatrix;
use crate::tensor::Tensor;
use num::Float;
use num::Num;

pub struct StaticVector<T: Num, const N: usize> {
    data: [T; N],
}

impl<T: Num + PartialOrd + Copy, const N: usize> StaticVector<T, N> {
    pub fn new(data: [T; N]) -> Result<StaticVector<T, N>, ShapeError> {
        assert!(N > 0, "N must be greater than 0");
        Ok(Self { data })
    }

    pub fn to_tensor(&self) -> Tensor<T> {
        Tensor::new(&shape![N].unwrap(), &self.data).unwrap()
    }

    pub fn fill(value: T) -> StaticVector<T, N> {
        let data = [value; N];
        StaticVector::new(data).unwrap()
    }

    pub fn ones() -> StaticVector<T, N> {
        StaticVector::fill(T::one())
    }

    pub fn zeros() -> StaticVector<T, N> {
        StaticVector::fill(T::zero())
    }

    pub fn matmul<const P: usize>(&self, rhs: &StaticMatrix<T, N, P>) -> StaticVector<T, P> {
        let mut result = StaticVector::zeros();
        for i in 0..P {
            for j in 0..N {
                result.data[i] = result.data[i] + self.data[j] * rhs[(j, i)];
            }
        }
        result
    }

    pub fn dot(&self, rhs: &Self) -> T {
        let mut result = T::zero();
        for i in 0..N {
            result = result + self.data[i] * rhs.data[i];
        }
        result
    }

    pub fn sum(&self) -> T {
        let mut sum = T::zero();
        for &item in self.data.iter() {
            sum = sum + item;
        }
        sum
    }

    pub fn mean(&self) -> T {
        self.sum() / self.size()
    }

    pub fn variance(&self) -> T {
        let mean = self.mean();
        let mut result = T::zero();
        for i in 0..N {
            let diff = self.data[i] - mean;
            result = result + diff * diff;
        }
        result / self.size()
    }

    pub fn max(&self) -> T {
        let mut max_value = self.data[0];
        for i in 1..N {
            if self.data[i] > max_value {
                max_value = self.data[i];
            }
        }
        max_value
    }

    pub fn min(&self) -> T {
        let mut min_value = self.data[0];
        for i in 1..N {
            if self.data[i] < min_value {
                min_value = self.data[i];
            }
        }
        min_value
    }

    pub fn dims(&self) -> usize {
        N
    }

    pub fn size(&self) -> T {
        let mut n = T::zero();
        for _ in 0..N {
            n = n + T::one();
        }
        n
    }
}

impl<T: Float + PartialOrd + Copy, const N: usize> StaticVector<T, N> {
    pub fn pow(&self, power: T) -> StaticVector<T, N> {
        let mut result = [T::zero(); N];
        for (i, &item) in self.data.iter().enumerate() {
            result[i] = item.powf(power);
        }
        StaticVector { data: result }
    }
}

// Scalar Addition
impl<T: Num + PartialOrd + Copy, const N: usize> Add<T> for StaticVector<T, N> {
    type Output = StaticVector<T, N>;

    fn add(self, rhs: T) -> StaticVector<T, N> {
        let mut result = [T::zero(); N];
        for (i, &item) in self.data.iter().enumerate() {
            result[i] = item + rhs;
        }
        StaticVector { data: result }
    }
}

// Vector Addition
impl<T: Num + PartialOrd + Copy, const N: usize> Add<StaticVector<T, N>> for StaticVector<T, N> {
    type Output = StaticVector<T, N>;

    fn add(self, rhs: StaticVector<T, N>) -> StaticVector<T, N> {
        let mut result = [T::zero(); N];
        for (i, &item) in self.data.iter().enumerate() {
            result[i] = item + rhs.data[i];
        }
        StaticVector { data: result }
    }
}

// Scalar Subtraction
impl<T: Num + PartialOrd + Copy, const N: usize> Sub<T> for StaticVector<T, N> {
    type Output = StaticVector<T, N>;

    fn sub(self, rhs: T) -> StaticVector<T, N> {
        let mut result = [T::zero(); N];
        for (i, &item) in self.data.iter().enumerate() {
            result[i] = item - rhs;
        }
        StaticVector { data: result }
    }
}

// Vector Subtraction
impl<T: Num + PartialOrd + Copy, const N: usize> Sub<StaticVector<T, N>> for StaticVector<T, N> {
    type Output = StaticVector<T, N>;

    fn sub(self, rhs: StaticVector<T, N>) -> StaticVector<T, N> {
        let mut result = [T::zero(); N];
        for (i, &item) in self.data.iter().enumerate() {
            result[i] = item - rhs.data[i];
        }
        StaticVector { data: result }
    }
}

// Scalar Multiplication
impl<T: Num + PartialOrd + Copy, const N: usize> Mul<T> for StaticVector<T, N> {
    type Output = StaticVector<T, N>;

    fn mul(self, rhs: T) -> StaticVector<T, N> {
        let mut result = [T::zero(); N];
        for (i, &item) in self.data.iter().enumerate() {
            result[i] = item * rhs;
        }
        StaticVector { data: result }
    }
}

// Vector Multiplication
impl<T: Num + PartialOrd + Copy, const N: usize> Mul<StaticVector<T, N>> for StaticVector<T, N> {
    type Output = StaticVector<T, N>;

    fn mul(self, rhs: StaticVector<T, N>) -> StaticVector<T, N> {
        let mut result = [T::zero(); N];
        for (i, &item) in self.data.iter().enumerate() {
            result[i] = item * rhs.data[i];
        }
        StaticVector { data: result }
    }
}

// Scalar Division
impl<T: Num + PartialOrd + Copy, const N: usize> Div<T> for StaticVector<T, N> {
    type Output = StaticVector<T, N>;

    fn div(self, rhs: T) -> StaticVector<T, N> {
        let mut result = [T::zero(); N];
        for (i, &item) in self.data.iter().enumerate() {
            result[i] = item / rhs;
        }
        StaticVector { data: result }
    }
}

// Vector Division
impl<T: Num + PartialOrd + Copy, const N: usize> Div<StaticVector<T, N>> for StaticVector<T, N> {
    type Output = StaticVector<T, N>;

    fn div(self, rhs: StaticVector<T, N>) -> StaticVector<T, N> {
        let mut result = [T::zero(); N];
        for (i, &item) in self.data.iter().enumerate() {
            result[i] = item / rhs.data[i];
        }
        StaticVector { data: result }
    }
}

impl<T: Num + Copy + PartialOrd, const N: usize> Index<usize> for StaticVector<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Num + Copy + PartialOrd, const N: usize> IndexMut<usize> for StaticVector<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Num, const N: usize> Deref for StaticVector<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &[T; N] {
        &self.data
    }
}

impl<T: Num, const N: usize> DerefMut for StaticVector<T, N> {
    fn deref_mut(&mut self) -> &mut [T; N] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data = [1, 2, 3];
        let vector = StaticVector::new(data).unwrap();
        assert_eq!(vector.data, data);
    }

    #[test]
    #[should_panic(expected = "N must be greater than 0")]
    fn test_new_failure_zero_length() {
        let _ = StaticVector::<i32, 0>::new([]).unwrap();
    }

    #[test]
    fn test_fill() {
        let vector = StaticVector::fill(5);
        assert_eq!(vector.data, [5; 3]);
    }

    #[test]
    fn test_ones() {
        let vector = StaticVector::<i32, 3>::ones();
        assert_eq!(vector.data, [1; 3]);
    }

    #[test]
    fn test_zeros() {
        let vector = StaticVector::<i32, 3>::zeros();
        assert_eq!(vector.data, [0; 3]);
    }

    #[test]
    fn test_matmul() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        let matrix = StaticMatrix::new([[1, 2], [3, 4], [5, 6]]).unwrap();
        let result = vector.matmul(&matrix);
        assert_eq!(result.data, [22, 28]);
    }

    #[test]
    fn test_dot() {
        let vector1 = StaticVector::new([1, 2, 3]).unwrap();
        let vector2 = StaticVector::new([4, 5, 6]).unwrap();
        let result = vector1.dot(&vector2);
        assert_eq!(result, 32);
    }

    #[test]
    fn test_sum() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        let result = vector.sum();
        assert_eq!(result, 6);
    }

    #[test]
    fn test_mean() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        let result = vector.mean();
        assert_eq!(result, 2);
    }

    #[test]
    fn test_variance() {
        let vector = StaticVector::new([1.0, 2.5, 4.0]).unwrap();
        let result = vector.variance();
        assert_eq!(result, 1.5);
    }

    #[test]
    fn test_max() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        let result = vector.max();
        assert_eq!(result, 3);
    }

    #[test]
    fn test_min() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        let result = vector.min();
        assert_eq!(result, 1);
    }

    #[test]
    fn test_dims() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        let result = vector.dims();
        assert_eq!(result, 3);
    }

    #[test]
    fn test_len() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        let result = vector.len();
        assert_eq!(result, 3);
    }

    #[test]
    fn test_pow() {
        let vector = StaticVector::new([1.0, 2.0, 3.0]).unwrap();
        let result = vector.pow(2.0);
        assert_eq!(result.data, [1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_scalar_add() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        let result = vector + 1;
        assert_eq!(result.data, [2, 3, 4]);
    }

    #[test]
    fn test_vector_add() {
        let vector1 = StaticVector::new([1, 2, 3]).unwrap();
        let vector2 = StaticVector::new([4, 5, 6]).unwrap();
        let result = vector1 + vector2;
        assert_eq!(result.data, [5, 7, 9]);
    }

    #[test]
    fn test_scalar_sub() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        let result = vector - 1;
        assert_eq!(result.data, [0, 1, 2]);
    }

    #[test]
    fn test_vector_sub() {
        let vector1 = StaticVector::new([4, 5, 6]).unwrap();
        let vector2 = StaticVector::new([1, 2, 3]).unwrap();
        let result = vector1 - vector2;
        assert_eq!(result.data, [3, 3, 3]);
    }

    #[test]
    fn test_scalar_mul() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        let result = vector * 2;
        assert_eq!(result.data, [2, 4, 6]);
    }

    #[test]
    fn test_vector_mul() {
        let vector1 = StaticVector::new([1, 2, 3]).unwrap();
        let vector2 = StaticVector::new([4, 5, 6]).unwrap();
        let result = vector1 * vector2;
        assert_eq!(result.data, [4, 10, 18]);
    }

    #[test]
    fn test_scalar_div() {
        let vector = StaticVector::new([2, 4, 6]).unwrap();
        let result = vector / 2;
        assert_eq!(result.data, [1, 2, 3]);
    }

    #[test]
    fn test_vector_div() {
        let vector1 = StaticVector::new([2, 4, 6]).unwrap();
        let vector2 = StaticVector::new([1, 2, 3]).unwrap();
        let result = vector1 / vector2;
        assert_eq!(result.data, [2, 2, 2]);
    }

    #[test]
    fn test_index() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        assert_eq!(vector[1], 2);
    }

    #[test]
    fn test_index_mut() {
        let mut vector = StaticVector::new([1, 2, 3]).unwrap();
        vector[1] = 5;
        assert_eq!(vector[1], 5);
    }

    #[test]
    fn test_deref() {
        let vector = StaticVector::new([1, 2, 3]).unwrap();
        assert_eq!(*vector, [1, 2, 3]);
    }

    #[test]
    fn test_deref_mut() {
        let mut vector = StaticVector::new([1, 2, 3]).unwrap();
        vector[1] = 5;
        assert_eq!(*vector, [1, 5, 3]);
    }
}
