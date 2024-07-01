use std::fmt;
use std::ops::{Index, IndexMut};

use crate::error::ShapeError;

#[derive(Debug, Clone, PartialEq)]
pub struct Coordinate {
    indices: Vec<usize>,
}

impl Coordinate {
    pub fn new(indices: Vec<usize>) -> Result<Self, ShapeError> {
        if indices.is_empty() {
            return Err(ShapeError::new("Coordinate cannot be empty"));
        }
        Ok(Self { indices })
    }

    pub fn order(&self) -> usize {
        self.indices.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.indices.iter()
    }

    pub fn insert(&self, index: usize, axis: usize) -> Self {
        let mut new_indices = self.indices.clone();
        new_indices.insert(index, axis);
        Self {
            indices: new_indices,
        }
    }
}

impl Index<usize> for Coordinate {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.indices[index]
    }
}

impl IndexMut<usize> for Coordinate {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.indices[index]
    }
}

impl fmt::Display for Coordinate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use itertools::Itertools;
        let idxs = self.indices.iter().map(|&x| format!("{}", x)).join(", ");
        write!(f, "({})", idxs)
    }
}

#[macro_export]
macro_rules! coord {
    ($($index:expr),*) => {
        {
            use $crate::coordinate::Coordinate;
            Coordinate::new(vec![$($index),*]).unwrap()
        }
    };

    ($index:expr; $count:expr) => {
        {
            use $crate::coordinate::Coordinate;
            Coordinate::new(vec![$index; $count]).unwrap()
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order() {
        let coord = coord![1, 2, 3];
        assert_eq!(coord.order(), 3);
    }

    #[test]
    fn test_iter() {
        let coord = coord![1, 2, 3];
        let mut iter = coord.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_insert() {
        let coord = coord![1, 2, 3];
        let new_coord = coord.insert(1, 4);
        assert_eq!(new_coord, coord![1, 4, 2, 3]);
    }

    #[test]
    fn test_index() {
        let coord = coord![1, 2, 3];
        assert_eq!(coord[0], 1);
        assert_eq!(coord[1], 2);
        assert_eq!(coord[2], 3);
    }

    #[test]
    fn test_index_mut() {
        let mut coord = coord![1, 2, 3];
        coord[1] = 4;
        assert_eq!(coord[1], 4);
    }

    #[test]
    fn test_display() {
        let coord = coord![1, 2, 3];
        assert_eq!(format!("{}", coord), "(1, 2, 3)");
    }

    #[test]
    fn test_coord_macro() {
        let coord = coord![1, 2, 3];
        assert_eq!(coord, Coordinate::new(vec![1, 2, 3]).unwrap());

        let coord_repeated = coord![1; 3];
        assert_eq!(coord_repeated, Coordinate::new(vec![1, 1, 1]).unwrap());
    }
}
