use candle_core::{Device, Tensor};

/// The default method for making batch of observations.
/// Args:
///     states (list): list of observations from an environment.
///     xp (module): numpy or cupy
/// Return:
///     the object which will be given as input to the model.

pub(crate) fn batch_states(states: &Vec<Tensor>, is_cuda: bool) -> Tensor {
    let device = if is_cuda && Device::cuda_if_available().is_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    Tensor::stack(&states, 0).to(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_states_cpu() {
        let states = vec![
            Tensor::from_slice(&[1.0, 2.0, 3.0]),
            Tensor::from_slice(&[4.0, 5.0, 6.0]),
        ];

        let result = batch_states(&states, false);
        assert_eq!(result.device(), Device::Cpu);

        let expected = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_batch_states_cuda() {
        if Device::cuda_if_available().is_cuda() {
            let states = vec![
                Tensor::from_slice(&[1.0, 2.0, 3.0]),
                Tensor::from_slice(&[4.0, 5.0, 6.0]),
            ];

            let result = batch_states(&states, true);

            assert_eq!(result.device(), Device::Cuda(0));

            let expected = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .reshape(&[2, 3])
                .to(Device::Cuda(0));
            assert_eq!(result, expected);
        } else {
            println!("Cuda is not available.");
        }
    }
}
