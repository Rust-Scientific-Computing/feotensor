pub mod axes;
pub mod coordinate;
pub mod error;
pub mod iter;
pub mod matrix;
pub mod shape;
pub mod storage;
pub mod tensor;
pub mod vector;

// Re-export important structs
pub use axes::Axes;
pub use shape::Shape;
pub use tensor::Tensor;
pub use vector::Vector;
pub use matrix::Matrix;
pub use coordinate::Coordinate;
pub use error::ShapeError;
