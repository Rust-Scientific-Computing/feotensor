use std::fmt;
use std::ops::Index;

use crate::error::ShapeError;

#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Result<Shape, ShapeError> {
        if dims.iter().any(|&x| x == 0) {
            return Err(ShapeError::new("Dimension cannot be zero"));
        }
        Ok(Shape { dims })
    }

    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn order(&self) -> usize {
        self.dims.len()
    }

    pub fn stack(&self, rhs: &Shape) -> Shape {
        let mut new_dims = self.dims.clone();
        new_dims.extend(rhs.dims.iter());
        Shape { dims: new_dims }
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl Index<std::ops::RangeFrom<usize>> for Shape {
    type Output = [usize];

    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        &self.dims[index]
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use itertools::Itertools;
        let dims = self.dims.iter().map(|&x| format!("{}", x)).join(", ");
        write!(f, "({})", dims)
    }
}

#[macro_export]
macro_rules! shape {
    ($($dim:expr),*) => {
        Shape::new(vec![$($dim),*])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_new() {
        let dims = vec![2, 3, 4];
        let shape = Shape::new(dims.clone()).unwrap();
        assert_eq!(shape.dims, dims);
    }

    #[test]
    fn test_shape_new_with_zero_dimension() {
        let dims = vec![2, 0, 4];
        let result = Shape::new(dims);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_size() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(shape.size(), 24);
    }

    #[test]
    fn test_shape_display() {
        let shape = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(format!("{}", shape), "(2, 3, 4)");
    }

    #[test]
    fn test_shape_macro() {
        let shape = shape![2, 3, 4].unwrap();
        assert_eq!(shape.dims, vec![2, 3, 4]);
    }

    
    #[test]
    fn test_shape_extend() {
        let shape1 = Shape::new(vec![2, 3]).unwrap();
        let shape2 = Shape::new(vec![4, 5]).unwrap();
        let extended_shape = shape1.stack(&shape2);
        assert_eq!(extended_shape.dims, vec![2, 3, 4, 5]);
    }
}