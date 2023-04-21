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
    Q: &DMatrix<f64>,
    p: &DVector<f64>,
    b: &DVector<f64>,
    s: &DVector<f64>,
    x0: &DVector<f64>,
    max_iter: usize,
    a: &DVector<f64>,
    e: f64,
    alpha: f64,
    gamma: f64,
) -> DVector<f64> {
    let mut x = x0.clone();

    for iter in 0..max_iter {
        let grad_f = 2.0 * Q * &x + p;
        let subgrad_g = max_elementwise(b, s);
        let subgrad = grad_f + subgrad_g;
        let dynamic_step_size = alpha / (gamma + iter as f64);
        x = x - dynamic_step_size * subgrad;
        x = project_onto_hyperplane(&x, a, e);
    }

    x
}

/// Gives an approximate solution to convex optimization functions of the form
/// f(x) = x^T Q x + p^T x + Σ max(x_i b_i, x_i s_i), such that a^T x = E,
/// given symmetric matrix Q, vectors q, b, s, a, and real E
#[pyfunction]
fn optimize(
    Q: Vec<f64>,
    p: Vec<f64>,
    b: Vec<f64>,
    s: Vec<f64>,
    x0: Vec<f64>,
    max_iter: usize,
    a: Vec<f64>,
    e: f64,
    alpha: f64,
    gamma: f64,
) -> PyResult<Vec<f64>> {
    let n = x0.len();
    let Q = DMatrix::from_row_slice(n, n, &Q);
    let p = DVector::from_vec(p);
    let b = DVector::from_vec(b);
    let s = DVector::from_vec(s);
    let x0 = DVector::from_vec(x0);
    let a = DVector::from_vec(a);

    let result = projected_subgradient_descent(&Q, &p, &b, &s, &x0, max_iter, &a, e, alpha, gamma);

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
