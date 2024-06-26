use std::ops::Index;
use std::ops::IndexMut;

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
            if dim >= shape.dims[i] {
                return Err(ShapeError::new(format!("out of bounds for dimension {}", i).as_str()));
            }
        }

        let mut index = 0;
        for k in 0..shape.order() {
            let stride = shape.dims[k+1..].iter().product::<usize>();
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
