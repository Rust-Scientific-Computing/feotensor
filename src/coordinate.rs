use std::fmt;
use std::ops::Index;
use std::ops::IndexMut;

#[derive(Debug, Clone, PartialEq)]
pub struct Coordinate {
    indices: Vec<usize>,
}

impl Coordinate {
    pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.indices.iter()
    }

    pub fn insert(&self, index: usize, axis: usize) -> Self {
        let mut new_indices = self.indices.clone();
        new_indices.insert(index, axis);
        Self { indices: new_indices }
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
        Coordinate::new(vec![$($index),*])
    };

    ($index:expr; $count:expr) => {
        Coordinate::new(vec![$index; $count])
    };
}
