use std::ops::{Index, IndexMut};

use crate::coordinate::Coordinate;
use crate::shape::Shape;
use crate::error::ShapeError;

#[derive(Debug, PartialEq)]
pub struct DynamicStorage<T> {
    data: Vec<T>,
}

impl<T> DynamicStorage<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// For the row-wise maths see: https://bit.ly/3KQjPa3
    pub fn flatten(&self, coord: &Coordinate, shape: &Shape) -> Result<usize, ShapeError> {
        if coord.len() != shape.order() {
            let msg = format!("incorrect order ({} vs {}).", coord.len(), shape.order());
            return Err(ShapeError::new(msg.as_str()));
        }

        for (i, &dim) in coord.iter().enumerate() {
            if dim >= shape[i] {
                return Err(ShapeError::new(format!("out of bounds for dimension {}", i).as_str()));
            }
        }

        let mut index = 0;
        for k in 0..shape.order() {
            let stride = shape[k+1..].iter().product::<usize>();
            index += coord[k] * stride;
        }
        Ok(index)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<T> Index<usize> for DynamicStorage<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for DynamicStorage<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T> IntoIterator for DynamicStorage<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a DynamicStorage<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut DynamicStorage<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter() {
        let storage = DynamicStorage::new(vec![1, 2, 3, 4]);
        let mut iter = storage.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_mut() {
        let mut storage = DynamicStorage::new(vec![1, 2, 3, 4]);
        {
            let mut iter = storage.iter_mut();
            if let Some(x) = iter.next() {
                *x = 10;
            }
        }
        assert_eq!(storage.data, vec![10, 2, 3, 4]);
    }

    #[test]
    fn test_index() {
        let storage = DynamicStorage::new(vec![1, 2, 3, 4]);
        assert_eq!(storage[0], 1);
        assert_eq!(storage[1], 2);
        assert_eq!(storage[2], 3);
        assert_eq!(storage[3], 4);
    }

    #[test]
    fn test_index_mut() {
        let mut storage = DynamicStorage::new(vec![1, 2, 3, 4]);
        storage[0] = 10;
        assert_eq!(storage[0], 10);
    }

    #[test]
    fn test_into_iter() {
        let storage = DynamicStorage::new(vec![1, 2, 3, 4]);
        let mut iter = storage.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_ref() {
        let storage = DynamicStorage::new(vec![1, 2, 3, 4]);
        let mut iter = (&storage).into_iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_mut() {
        let mut storage = DynamicStorage::new(vec![1, 2, 3, 4]);
        {
            let mut iter = (&mut storage).into_iter();
            if let Some(x) = iter.next() {
                *x = 10;
            }
        }
        assert_eq!(storage.data, vec![10, 2, 3, 4]);
    }
}
