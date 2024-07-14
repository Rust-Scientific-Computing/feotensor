use std::ops::{Add, Deref, DerefMut, Div, Index, IndexMut, Mul, Sub};

use crate::dynamic_vector::DynamicVector;
use crate::error::ShapeError;
use crate::shape;
use crate::static_vector::StaticVector;
use crate::tensor::Tensor;
use num::Float;
use num::Num;

pub struct StaticMatrix<T: Num, const M: usize, const N: usize> {
    data: [[T; N]; M],
}

impl<T: Num + PartialOrd + Copy, const M: usize, const N: usize> StaticMatrix<T, M, N> {
    pub fn new(data: [[T; N]; M]) -> Result<StaticMatrix<T, M, N>, ShapeError> {
        assert!(M > 0 && N > 0, "M and N must be greater than 0");
        if data.len() != M || data[0].len() != N {
            return Err(ShapeError::new("Data dimensions must be equal to M and N"));
        }
        Ok(Self { data })
    }

    pub fn to_tensor(&self) -> Tensor<T> {
        Tensor::new(&shape![M, N].unwrap(), &self.data.concat()).unwrap()
    }

    pub fn fill(value: T) -> StaticMatrix<T, M, N> {
        let data = [[value; N]; M];
        StaticMatrix::new(data).unwrap()
    }

    pub fn ones() -> StaticMatrix<T, M, N> {
        StaticMatrix::fill(T::one())
    }

    pub fn zeros() -> StaticMatrix<T, M, N> {
        StaticMatrix::fill(T::zero())
    }

    pub fn vecmul(&self, rhs: &StaticVector<T, N>) -> StaticVector<T, M> {
        let mut result = StaticVector::zeros();
        for i in 0..M {
            for j in 0..N {
                result[i] = result[i] + self.data[i][j] * rhs[j];
            }
        }
        result
    }

    pub fn matmul<const P: usize>(&self, rhs: &StaticMatrix<T, N, P>) -> StaticMatrix<T, M, P> {
        let mut result = StaticMatrix::<T, M, P>::zeros();
        for i in 0..M {
            for j in 0..P {
                for k in 0..N {
                    result.data[i][j] = result.data[i][j] + self.data[i][k] * rhs.data[k][j];
                }
            }
        }
        result
    }

    pub fn sum(&self, axis: Option<usize>) -> DynamicVector<T> {
        match axis {
            Some(0) => {
                let mut result = vec![T::zero(); N];
                for row in self.data.iter() {
                    for (j, &item) in row.iter().enumerate() {
                        result[j] = result[j] + item;
                    }
                }
                DynamicVector::new(&result).unwrap()
            }
            Some(1) => {
                let mut result = vec![T::zero(); M];
                for (i, row) in self.data.iter().enumerate() {
                    for &item in row.iter() {
                        result[i] = result[i] + item;
                    }
                }
                DynamicVector::new(&result).unwrap()
            }
            None => {
                let mut sum = T::zero();
                for row in self.data.iter() {
                    for &item in row.iter() {
                        sum = sum + item;
                    }
                }
                DynamicVector::new(&[sum]).unwrap()
            }
            _ => panic!("Axis out of bounds"),
        }
    }

    pub fn mean(&self, axis: Option<usize>) -> DynamicVector<T> {
        let sum = self.sum(axis);
        match axis {
            Some(0) => sum / self.size().0,
            Some(1) => sum / self.size().1,
            None => sum / (self.size().0 * self.size().1),
            _ => panic!("Axis out of bounds"),
        }
    }

    pub fn variance(&self, axis: Option<usize>) -> DynamicVector<T> {
        let mean = self.mean(axis);
        match axis {
            Some(0) => {
                let mut result = vec![T::zero(); N];
                for row in self.data.iter() {
                    for (j, &item) in row.iter().enumerate() {
                        let diff = item - mean[j];
                        result[j] = result[j] + diff * diff;
                    }
                }
                DynamicVector::new(&result).unwrap() / self.size().0
            }
            Some(1) => {
                let mut result = vec![T::zero(); M];
                for (i, row) in self.data.iter().enumerate() {
                    for &item in row.iter() {
                        let diff = item - mean[i];
                        result[i] = result[i] + diff * diff;
                    }
                }
                DynamicVector::new(&result).unwrap() / self.size().1
            }
            None => {
                let mut result = T::zero();
                for row in self.data.iter() {
                    for &item in row.iter() {
                        let diff = item - mean[0];
                        result = result + diff * diff;
                    }
                }
                DynamicVector::new(&[result / (self.size().0 * self.size().1)]).unwrap()
            }
            _ => panic!("Axis out of bounds"),
        }
    }

    pub fn max(&self, axis: Option<usize>) -> DynamicVector<T> {
        let mut min_value = self.data[0][0];
        for row in self.data.iter() {
            for &item in row.iter() {
                if item < min_value {
                    min_value = item;
                }
            }
        }
        match axis {
            Some(0) => {
                let mut result = vec![min_value; N];
                for row in self.data.iter() {
                    for (j, &item) in row.iter().enumerate() {
                        if item > result[j] {
                            result[j] = item;
                        }
                    }
                }
                DynamicVector::new(&result).unwrap()
            }
            Some(1) => {
                let mut result = vec![min_value; M];
                for (i, row) in self.data.iter().enumerate() {
                    for &item in row.iter() {
                        if item > result[i] {
                            result[i] = item;
                        }
                    }
                }
                DynamicVector::new(&result).unwrap()
            }
            None => {
                let mut max_value = min_value;
                for row in self.data.iter() {
                    for &item in row.iter() {
                        if item > max_value {
                            max_value = item;
                        }
                    }
                }
                DynamicVector::new(&[max_value]).unwrap()
            }
            _ => panic!("Axis out of bounds"),
        }
    }

    pub fn min(&self, axis: Option<usize>) -> DynamicVector<T> {
        let mut max_value = self.data[0][0];
        for row in self.data.iter() {
            for &item in row.iter() {
                if item > max_value {
                    max_value = item;
                }
            }
        }
        match axis {
            Some(0) => {
                let mut result = vec![max_value; N];
                for row in self.data.iter() {
                    for (j, &item) in row.iter().enumerate() {
                        if item < result[j] {
                            result[j] = item;
                        }
                    }
                }
                DynamicVector::new(&result).unwrap()
            }
            Some(1) => {
                let mut result = vec![max_value; M];
                for (i, row) in self.data.iter().enumerate() {
                    for &item in row.iter() {
                        if item < result[i] {
                            result[i] = item;
                        }
                    }
                }
                DynamicVector::new(&result).unwrap()
            }
            None => {
                let mut min_value = max_value;
                for row in self.data.iter() {
                    for &item in row.iter() {
                        if item < min_value {
                            min_value = item;
                        }
                    }
                }
                DynamicVector::new(&[min_value]).unwrap()
            }
            _ => panic!("Axis out of bounds"),
        }
    }

    pub fn dims(&self) -> (usize, usize) {
        (M, N)
    }

    pub fn size(&self) -> (T, T) {
        let mut n = T::zero();
        let mut m = T::zero();
        for _ in 0..M {
            m = m + T::one();
        }
        for _ in 0..N {
            n = n + T::one();
        }
        (m, n)
    }
}

impl<T: Float + PartialOrd + Copy, const M: usize, const N: usize> StaticMatrix<T, M, N> {
    pub fn pow(&self, power: T) -> StaticMatrix<T, M, N> {
        let mut result = [[T::zero(); N]; M];
        for (i, row) in self.data.iter().enumerate() {
            for (j, &item) in row.iter().enumerate() {
                result[i][j] = item.powf(power);
            }
        }
        StaticMatrix { data: result }
    }
}

// Scalar Addition
impl<T: Num + PartialOrd + Copy, const M: usize, const N: usize> Add<T> for StaticMatrix<T, M, N> {
    type Output = StaticMatrix<T, M, N>;

    fn add(self, rhs: T) -> StaticMatrix<T, M, N> {
        let mut result = [[T::zero(); N]; M];
        for (i, row) in self.data.iter().enumerate() {
            for (j, &item) in row.iter().enumerate() {
                result[i][j] = item + rhs;
            }
        }
        StaticMatrix { data: result }
    }
}

// Matrix Addition
impl<T: Num + PartialOrd + Copy, const M: usize, const N: usize> Add<StaticMatrix<T, M, N>>
    for StaticMatrix<T, M, N>
{
    type Output = StaticMatrix<T, M, N>;

    fn add(self, rhs: StaticMatrix<T, M, N>) -> StaticMatrix<T, M, N> {
        let mut result = [[T::zero(); N]; M];
        for (i, row) in self.data.iter().enumerate() {
            for (j, &item) in row.iter().enumerate() {
                result[i][j] = item + rhs.data[i][j];
            }
        }
        StaticMatrix { data: result }
    }
}

// Scalar Subtraction
impl<T: Num + PartialOrd + Copy, const M: usize, const N: usize> Sub<T> for StaticMatrix<T, M, N> {
    type Output = StaticMatrix<T, M, N>;

    fn sub(self, rhs: T) -> StaticMatrix<T, M, N> {
        let mut result = [[T::zero(); N]; M];
        for (i, row) in self.data.iter().enumerate() {
            for (j, &item) in row.iter().enumerate() {
                result[i][j] = item - rhs;
            }
        }
        StaticMatrix { data: result }
    }
}

// Matrix Subtraction
impl<T: Num + PartialOrd + Copy, const M: usize, const N: usize> Sub<StaticMatrix<T, M, N>>
    for StaticMatrix<T, M, N>
{
    type Output = StaticMatrix<T, M, N>;

    fn sub(self, rhs: StaticMatrix<T, M, N>) -> StaticMatrix<T, M, N> {
        let mut result = [[T::zero(); N]; M];
        for (i, row) in self.data.iter().enumerate() {
            for (j, &item) in row.iter().enumerate() {
                result[i][j] = item - rhs.data[i][j];
            }
        }
        StaticMatrix { data: result }
    }
}

// Scalar Multiplication
impl<T: Num + PartialOrd + Copy, const M: usize, const N: usize> Mul<T> for StaticMatrix<T, M, N> {
    type Output = StaticMatrix<T, M, N>;

    fn mul(self, rhs: T) -> StaticMatrix<T, M, N> {
        let mut result = [[T::zero(); N]; M];
        for (i, row) in self.data.iter().enumerate() {
            for (j, &item) in row.iter().enumerate() {
                result[i][j] = item * rhs;
            }
        }
        StaticMatrix { data: result }
    }
}

// Matrix Multiplication
impl<T: Num + PartialOrd + Copy, const M: usize, const N: usize> Mul<StaticMatrix<T, M, N>>
    for StaticMatrix<T, M, N>
{
    type Output = StaticMatrix<T, M, N>;

    fn mul(self, rhs: StaticMatrix<T, M, N>) -> StaticMatrix<T, M, N> {
        let mut result = [[T::zero(); N]; M];
        for (i, row) in self.data.iter().enumerate() {
            for (j, &item) in row.iter().enumerate() {
                result[i][j] = item * rhs.data[i][j];
            }
        }
        StaticMatrix { data: result }
    }
}

// Scalar Division
impl<T: Num + PartialOrd + Copy, const M: usize, const N: usize> Div<T> for StaticMatrix<T, M, N> {
    type Output = StaticMatrix<T, M, N>;

    fn div(self, rhs: T) -> StaticMatrix<T, M, N> {
        let mut result = [[T::zero(); N]; M];
        for (i, row) in self.data.iter().enumerate() {
            for (j, &item) in row.iter().enumerate() {
                result[i][j] = item / rhs;
            }
        }
        StaticMatrix { data: result }
    }
}

// Matrix Division
impl<T: Num + PartialOrd + Copy, const M: usize, const N: usize> Div<StaticMatrix<T, M, N>>
    for StaticMatrix<T, M, N>
{
    type Output = StaticMatrix<T, M, N>;

    fn div(self, rhs: StaticMatrix<T, M, N>) -> StaticMatrix<T, M, N> {
        let mut result = [[T::zero(); N]; M];
        for (i, row) in self.data.iter().enumerate() {
            for (j, &item) in row.iter().enumerate() {
                result[i][j] = item / rhs.data[i][j];
            }
        }
        StaticMatrix { data: result }
    }
}

impl<T: Num + Copy + PartialOrd, const M: usize, const N: usize> Index<(usize, usize)>
    for StaticMatrix<T, M, N>
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

impl<T: Num + Copy + PartialOrd, const M: usize, const N: usize> IndexMut<(usize, usize)>
    for StaticMatrix<T, M, N>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0][index.1]
    }
}

impl<T: Num, const M: usize, const N: usize> Deref for StaticMatrix<T, M, N> {
    type Target = [[T; N]; M];

    fn deref(&self) -> &[[T; N]; M] {
        &self.data
    }
}

impl<T: Num, const M: usize, const N: usize> DerefMut for StaticMatrix<T, M, N> {
    fn deref_mut(&mut self) -> &mut [[T; N]; M] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let data = [[1, 2], [3, 4]];
        let matrix = StaticMatrix::new(data).unwrap();
        assert_eq!(matrix.data, data);
    }

    #[test]
    #[should_panic(expected = "M and N must be greater than 0")]
    fn test_new_failure_zero_length() {
        let _ = StaticMatrix::<i32, 1, 0>::new([[]]).unwrap();
    }

    #[test]
    fn test_fill() {
        let matrix = StaticMatrix::fill(5);
        assert_eq!(matrix.data, [[5; 2]; 2]);
    }

    #[test]
    fn test_ones() {
        let matrix = StaticMatrix::<i32, 2, 2>::ones();
        assert_eq!(matrix.data, [[1; 2]; 2]);
    }

    #[test]
    fn test_zeros() {
        let matrix = StaticMatrix::<i32, 2, 2>::zeros();
        assert_eq!(matrix.data, [[0; 2]; 2]);
    }

    #[test]
    fn test_vecmul() {
        let matrix = StaticMatrix::new([[1, 2], [3, 4], [5, 6]]).unwrap();
        let vector = StaticVector::new([1, 2]).unwrap();
        let result = matrix.vecmul(&vector);
        assert_eq!(*result, [5, 11, 17]);
    }

    #[test]
    fn test_matmul() {
        let matrix1 = StaticMatrix::new([[1, 2, 3], [4, 5, 6]]).unwrap();
        let matrix2 = StaticMatrix::new([[7, 8], [9, 10], [11, 12]]).unwrap();
        let result = matrix1.matmul(&matrix2);
        assert_eq!(result.data, [[58, 64], [139, 154]]);
    }

    #[test]
    fn test_sum() {
        let matrix = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        let result = matrix.sum(None);
        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result[0], 10);

        let result = matrix.sum(Some(0));
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], 4);
        assert_eq!(result[1], 6);

        let result = matrix.sum(Some(1));
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], 3);
        assert_eq!(result[1], 7);
    }

    #[test]
    fn test_mean() {
        let matrix = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let result = matrix.mean(None);
        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result[0], 2.5);

        let result = matrix.mean(Some(0));
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 3.0);

        let result = matrix.mean(Some(1));
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], 1.5);
        assert_eq!(result[1], 3.5);
    }

    #[test]
    fn test_variance() {
        let matrix = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let result = matrix.variance(None);
        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result[0], 1.25);

        let result = matrix.variance(Some(0));
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 1.0);

        let result = matrix.variance(Some(1));
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], 0.25);
        assert_eq!(result[1], 0.25);
    }

    #[test]
    fn test_max() {
        let matrix = StaticMatrix::new([[-1, -2], [-3, -4]]).unwrap();
        let result = matrix.max(None);
        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result[0], -1);

        let result = matrix.max(Some(0));
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], -1);
        assert_eq!(result[1], -2);

        let result = matrix.max(Some(1));
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], -1);
        assert_eq!(result[1], -3);
    }

    #[test]
    fn test_min() {
        let matrix = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        let result = matrix.min(None);
        assert_eq!(result.shape(), &shape![1].unwrap());
        assert_eq!(result[0], 1);

        let result = matrix.min(Some(0));
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);

        let result = matrix.min(Some(1));
        assert_eq!(result.shape(), &shape![2].unwrap());
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 3);
    }

    #[test]
    fn test_dims() {
        let matrix = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        let result = matrix.dims();
        assert_eq!(result, (2, 2));
    }

    #[test]
    fn test_size() {
        let matrix = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        let result = matrix.size();
        assert_eq!(result, (2, 2));
    }

    #[test]
    fn test_add() {
        let matrix1 = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        let matrix2 = StaticMatrix::new([[5, 6], [7, 8]]).unwrap();
        let result = matrix1 + matrix2;
        assert_eq!(result.data, [[6, 8], [10, 12]]);
    }

    #[test]
    fn test_add_scalar() {
        let matrix = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        let result = matrix + 2;
        assert_eq!(result.data, [[3, 4], [5, 6]]);
    }

    #[test]
    fn test_sub() {
        let matrix1 = StaticMatrix::new([[5, 6], [7, 8]]).unwrap();
        let matrix2 = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        let result = matrix1 - matrix2;
        assert_eq!(result.data, [[4, 4], [4, 4]]);
    }

    #[test]
    fn test_sub_scalar() {
        let matrix = StaticMatrix::new([[5, 6], [7, 8]]).unwrap();
        let result = matrix - 2;
        assert_eq!(result.data, [[3, 4], [5, 6]]);
    }

    #[test]
    fn test_mul() {
        let matrix1 = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        let matrix2 = StaticMatrix::new([[2, 0], [1, 2]]).unwrap();
        let result = matrix1 * matrix2;
        assert_eq!(result.data, [[2, 0], [3, 8]]);
    }

    #[test]
    fn test_mul_scalar() {
        let matrix = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        let result = matrix * 2;
        assert_eq!(result.data, [[2, 4], [6, 8]]);
    }

    #[test]
    fn test_div() {
        let matrix1 = StaticMatrix::new([[4, 8], [12, 16]]).unwrap();
        let matrix2 = StaticMatrix::new([[2, 2], [3, 4]]).unwrap();
        let result = matrix1 / matrix2;
        assert_eq!(result.data, [[2, 4], [4, 4]]);
    }

    #[test]
    fn test_div_scalar() {
        let matrix = StaticMatrix::new([[2, 4], [6, 8]]).unwrap();
        let result = matrix / 2;
        assert_eq!(result.data, [[1, 2], [3, 4]]);
    }

    #[test]
    fn test_pow() {
        let matrix = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let result = matrix.pow(2.0);
        assert_eq!(result.data, [[1.0, 4.0], [9.0, 16.0]]);
    }

    #[test]
    fn test_index() {
        let matrix = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        assert_eq!(matrix[(0, 1)], 2);
    }

    #[test]
    fn test_index_mut() {
        let mut matrix = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        matrix[(0, 1)] = 5;
        assert_eq!(matrix[(0, 1)], 5);
    }

    #[test]
    fn test_deref() {
        let matrix = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        assert_eq!(*matrix, [[1, 2], [3, 4]]);
    }

    #[test]
    fn test_deref_mut() {
        let mut matrix = StaticMatrix::new([[1, 2], [3, 4]]).unwrap();
        matrix[(0, 1)] = 5;
        assert_eq!(*matrix, [[1, 5], [3, 4]]);
    }
}
