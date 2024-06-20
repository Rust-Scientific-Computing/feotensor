pub struct IndexIterator {
    shape: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl IndexIterator {
    pub fn new(shape: &[usize]) -> Self {
        let current = vec![0; shape.len()];
        IndexIterator {
            shape: shape.to_vec(),
            current,
            done: false,
        }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.shape.len() == 0 {
            return None;
        }

        let result = self.current.clone();

        for i in (0..self.shape.len()).rev() {
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

    #[test]
    fn test_index_iterator() {
        let shape = vec![2, 3];
        let mut iter = IndexIterator::new(&shape);

        assert_eq!(iter.next(), Some(vec![0, 0]));
        assert_eq!(iter.next(), Some(vec![0, 1]));
        assert_eq!(iter.next(), Some(vec![0, 2]));
        assert_eq!(iter.next(), Some(vec![1, 0]));
        assert_eq!(iter.next(), Some(vec![1, 1]));
        assert_eq!(iter.next(), Some(vec![1, 2]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_index_iterator_single_dimension() {
        let shape = vec![4];
        let mut iter = IndexIterator::new(&shape);

        assert_eq!(iter.next(), Some(vec![0]));
        assert_eq!(iter.next(), Some(vec![1]));
        assert_eq!(iter.next(), Some(vec![2]));
        assert_eq!(iter.next(), Some(vec![3]));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_index_iterator_empty_tensor() {
        let shape = vec![];
        let mut iter = IndexIterator::new(&shape);

        assert_eq!(iter.next(), None);
    }
}