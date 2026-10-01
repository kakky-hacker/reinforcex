//! Global gradient clipping without overflowing the norm reduction for finite
//! f32 gradients. Reject invalid gradients before scaling or stepping anything.

use super::autograd::no_grad;
use tch::{nn, Kind};

/// Return the pre-clip L2 norm and clip all defined optimizer gradients together.
///
/// Accumulate squared gradients in f64; retain tch's `maximum / (norm + 1e-6)`
/// coefficient. The first pass validates every gradient, so a failed check does
/// not partially scale them. The caller remains responsible for checking its
/// loss before backward and invoking the optimizer only after this succeeds.
pub(crate) fn clip_grad_norm_f64(optimizer: &nn::Optimizer, maximum: f64, agent: &str) -> f64 {
    assert!(maximum.is_finite() && maximum > 0.0);
    let parameters = optimizer.trainable_variables();
    no_grad(|| {
        let mut squared_norm = 0.0;
        for parameter in &parameters {
            let gradient = parameter.grad();
            if gradient.defined() {
                assert!(
                    gradient.isfinite().all().int64_value(&[]) != 0,
                    "{} gradient must be finite",
                    agent
                );
                squared_norm += gradient
                    .to_kind(Kind::Double)
                    .square()
                    .sum(Kind::Double)
                    .double_value(&[]);
            }
        }
        let norm = squared_norm.sqrt();
        assert!(norm.is_finite(), "{} gradient norm must be finite", agent);
        let coefficient = maximum / (norm + 1e-6);
        if coefficient < 1.0 {
            for parameter in &parameters {
                let mut gradient = parameter.grad();
                if gradient.defined() {
                    let _ = gradient.g_mul_scalar_(coefficient);
                }
            }
        }
        norm
    })
}
