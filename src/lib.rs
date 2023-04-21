use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

fn max_elementwise(a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
    a.zip_map(b, |a_i, b_i| a_i.max(b_i))
}

fn project_onto_hyperplane(x: &DVector<f64>, a: &DVector<f64>, e: f64) -> DVector<f64> {
    let t = (e - a.dot(&x)) / a.dot(&a);
    x + t * a
}

pub fn projected_subgradient_descent(
    a: &DMatrix<f64>,
    q: &DVector<f64>,
    b: &DVector<f64>,
    s: &DVector<f64>,
    x0: &DVector<f64>,
    max_iter: usize,
    step_size: f64,
    a_hyperplane: &DVector<f64>,
    e: f64,
) -> DVector<f64> {
    let mut x = x0.clone();

    for _ in 0..max_iter {
        let grad_f = a * &x + q;
        let subgrad_g = max_elementwise(&(b - &x), &(x.clone() - s));
        let subgrad = grad_f + subgrad_g;

        x = x - step_size * subgrad;
        x = project_onto_hyperplane(&x, a_hyperplane, e);
    }

    x
}

/// Gives an approximate solution to convex optimization functions of the form
/// f(x) = x^T A x + q^T x + Σ max(x_i b_i, x_i s_i), such that a^T x = E,
/// given matrix A, vectors q, b, s, a, and real E
#[pyfunction]
fn optimize(
    a_matrix: Vec<f64>,
    q_vector: Vec<f64>,
    b_vector: Vec<f64>,
    s_vector: Vec<f64>,
    x0: Vec<f64>,
    max_iter: usize,
    step_size: f64,
    a_hyperplane: Vec<f64>,
    e: f64,
) -> PyResult<Vec<f64>> {
    let n = x0.len();
    let a_matrix = DMatrix::from_row_slice(n, n, &a_matrix);
    let q_vector = DVector::from_vec(q_vector);
    let b_vector = DVector::from_vec(b_vector);
    let s_vector = DVector::from_vec(s_vector);
    let x0_vector = DVector::from_vec(x0);
    let a_hyperplane_vector = DVector::from_vec(a_hyperplane);

    let result = projected_subgradient_descent(
        &a_matrix,
        &q_vector,
        &b_vector,
        &s_vector,
        &x0_vector,
        max_iter,
        step_size,
        &a_hyperplane_vector,
        e,
    );

    Ok(result.iter().cloned().collect::<Vec<f64>>())
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn convex(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    Ok(())
}
