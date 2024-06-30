use criterion::{criterion_group, criterion_main, Criterion};
use feotensor::{Matrix, Shape, Tensor, Vector};

// Contraction Methods

fn bench_sum(c: &mut Criterion) {
    let shape = Shape::new(vec![500, 500]).unwrap();
    let tensor = Tensor::<f64>::ones(&shape);
    c.bench_function("tensor_sum", |b| b.iter(|| tensor.sum(vec![])));
}

fn bench_mean(c: &mut Criterion) {
    let shape = Shape::new(vec![500, 500]).unwrap();
    let tensor = Tensor::<f64>::ones(&shape);
    c.bench_function("tensor_mean", |b| b.iter(|| tensor.mean(vec![])));
}

fn bench_var(c: &mut Criterion) {
    let shape = Shape::new(vec![500, 500]).unwrap();
    let tensor = Tensor::<f64>::ones(&shape);
    c.bench_function("tensor_var", |b| b.iter(|| tensor.var(vec![])));
}

fn bench_max(c: &mut Criterion) {
    let shape = Shape::new(vec![500, 500]).unwrap();
    let tensor = Tensor::<f64>::ones(&shape);
    c.bench_function("tensor_max", |b| b.iter(|| tensor.max(vec![])));
}

fn bench_min(c: &mut Criterion) {
    let shape = Shape::new(vec![500, 500]).unwrap();
    let tensor = Tensor::<f64>::ones(&shape);
    c.bench_function("tensor_min", |b| b.iter(|| tensor.min(vec![])));
}

// Tensor Product

fn bench_tensor_product(c: &mut Criterion) {
    let shape_a = Shape::new(vec![100, 100]).unwrap();
    let shape_b = Shape::new(vec![100, 100]).unwrap();
    let tensor_a = Tensor::<f64>::ones(&shape_a);
    let tensor_b = Tensor::<f64>::ones(&shape_b);
    c.bench_function("tensor_product", |b| b.iter(|| tensor_a.prod(&tensor_b)));
}

// Matrix Multiplication

fn bench_matmul(c: &mut Criterion) {
    let shape_a = Shape::new(vec![100, 100]).unwrap();
    let shape_b = Shape::new(vec![100, 100]).unwrap();
    let matrix_a = Matrix::<f64>::ones(&shape_a).unwrap();
    let matrix_b = Matrix::<f64>::ones(&shape_b).unwrap();
    c.bench_function("matrix_multiplication", |b| {
        b.iter(|| matrix_a.matmul(&matrix_b))
    });
}

// Vector Multiplication

fn bench_vecmul(c: &mut Criterion) {
    let shape = Shape::new(vec![100]).unwrap();
    let vector_a = Vector::<f64>::ones(&shape).unwrap();
    let vector_b = Vector::<f64>::ones(&shape).unwrap();
    c.bench_function("vector_multiplication", |b| {
        b.iter(|| vector_a.vecmul(&vector_b))
    });
}

criterion_group!(
    benches,
    bench_sum,
    bench_mean,
    bench_var,
    bench_max,
    bench_min,
    bench_matmul,
    bench_vecmul,
    bench_tensor_product,
);
criterion_main!(benches);
