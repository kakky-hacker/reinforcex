//! Local inference/diagnostic scopes must restore autograd even when an agent
//! panics and the FFI catches that panic on a thread used by another agent.

pub(crate) fn no_grad<T>(operation: impl FnOnce() -> T) -> T {
    // tch 0.20's closure helper restores the previous mode only on normal
    // return. Its RAII guard also restores it during Rust stack unwinding.
    let _guard = tch::no_grad_guard();
    operation()
}

#[cfg(test)]
mod tests {
    use super::no_grad;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

    fn operation_tracks_gradient() -> bool {
        let value = Tensor::from_slice(&[1.0_f32]).set_requires_grad(true);
        (&value * 2.0).requires_grad()
    }

    #[test]
    fn restores_enabled_mode_on_return_and_unwind() {
        assert!(operation_tracks_gradient());
        let result = no_grad(|| {
            assert!(!operation_tracks_gradient());
            17
        });
        assert_eq!(result, 17);
        assert!(operation_tracks_gradient());
        assert!(catch_unwind(|| no_grad(|| {
            assert!(!operation_tracks_gradient());
            panic!("model/diagnostic error");
        }))
        .is_err());
        assert!(operation_tracks_gradient());
    }

    #[test]
    fn preserves_disabled_outer_mode_after_nested_unwind() {
        assert!(operation_tracks_gradient());
        {
            let _outer_guard = tch::no_grad_guard();
            no_grad(|| assert!(!operation_tracks_gradient()));
            assert!(catch_unwind(|| no_grad(|| {
                no_grad(|| panic!("nested error"));
            }))
            .is_err());
            assert!(!operation_tracks_gradient());
        }
        assert!(operation_tracks_gradient());
    }

    #[test]
    fn independent_optimizer_can_learn_after_caught_tensor_error() {
        let store = nn::VarStore::new(Device::Cpu);
        let parameter = store.root().var("healthy", &[1], nn::Init::Const(1.0));
        let mut optimizer = nn::Adam::default().build(&store, 0.01).unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| no_grad(|| {
            // Exercise a real tch error, not only an explicit panic.
            let _ = Tensor::zeros([2], (Kind::Float, Device::Cpu)).view([3]);
        })))
        .is_err());
        let loss = parameter.square().sum(Kind::Float);
        assert!(loss.requires_grad());
        optimizer.zero_grad();
        loss.backward();
        assert_eq!(parameter.grad().double_value(&[0]), 2.0);
        optimizer.step();
        assert!(parameter.double_value(&[0]) < 1.0);
    }
}
