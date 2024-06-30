use crate::coord;
use crate::coordinate::Coordinate;
use crate::shape::Shape;
use std::cmp::max;

pub struct IndexIterator {
    shape: Shape,
    current: Coordinate,
    done: bool,
}

impl IndexIterator {
    pub fn new(shape: &Shape) -> Self {
        // (shape.order() == 0) => `next` returns None before `current` is used
        let current = coord![0; max(shape.order(), 1)].unwrap();
        IndexIterator {
            shape: shape.clone(),
            current,
            done: false,
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Coordinate;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.shape.order() == 0 {
            return None;
        }

        let result = self.current.clone();

        for i in (0..self.shape.order()).rev() {
            if self.current[i] + 1 < self.shape[i] {
                self.current[i] += 1;
                break;
            } else {
                self.current[i] = 0;
                if i == 0 {
                    self.done = true;
                }
            }
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape;

    #[test]
    fn test_index_iterator() {
        let shape = shape![2, 3].unwrap();
        let mut iter = IndexIterator::new(&shape);

        assert_eq!(iter.next(), Some(coord![0, 0].unwrap()));
        assert_eq!(iter.next(), Some(coord![0, 1].unwrap()));
        assert_eq!(iter.next(), Some(coord![0, 2].unwrap()));
        assert_eq!(iter.next(), Some(coord![1, 0].unwrap()));
        assert_eq!(iter.next(), Some(coord![1, 1].unwrap()));
        assert_eq!(iter.next(), Some(coord![1, 2].unwrap()));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_index_iterator_single_dimension() {
        let shape = shape![4].unwrap();
        let mut iter = IndexIterator::new(&shape);

        assert_eq!(iter.next(), Some(coord![0].unwrap()));
        assert_eq!(iter.next(), Some(coord![1].unwrap()));
        assert_eq!(iter.next(), Some(coord![2].unwrap()));
        assert_eq!(iter.next(), Some(coord![3].unwrap()));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_index_iterator_empty_tensor() {
        let shape = shape![].unwrap();
        let mut iter = IndexIterator::new(&shape);

        assert_eq!(iter.next(), None);
    }
}
