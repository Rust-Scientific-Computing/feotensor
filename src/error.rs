use std::fmt;

#[derive(Debug, Clone)]
pub struct ShapeError {
    reason: String,
}

impl ShapeError {
    pub fn new(reason: &str) -> Self {
        ShapeError {
            reason: reason.to_string(),
        }
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ShapeError: {}", self.reason)
    }
}
