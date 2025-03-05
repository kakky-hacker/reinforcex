use candle_core::{Device, Result, Tensor};

/// The default method for making batch of observations.
/// Args:
///     states (list): list of observations from an environment.
///     xp (module): numpy or cupy
/// Return:
///     the object which will be given as input to the model.

pub(crate) fn batch_states(states: &Vec<Tensor>, is_cuda: bool) -> Result<Tensor> {
    let device;
    if is_cuda {
        device = Device::cuda_if_available(0)?;
    } else {
        device = Device::Cpu;
    }
    let res = Tensor::stack(&states, 0)?.to_device(&device)?;
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_states_cpu() {
        let states = vec![
            Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
            Tensor::from_slice(&[4.0, 5.0, 6.0], &[3], &Device::Cpu).unwrap(),
        ];

        let result = batch_states(&states, false).unwrap();
        assert!(result.device().is_cpu());

        let expected =
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &Device::Cpu).unwrap();
        assert_eq!(
            result.to_vec2::<f64>().unwrap(),
            expected.to_vec2::<f64>().unwrap()
        );
    }

    #[test]
    fn test_batch_states_cuda() {
        if Device::cuda_if_available(0).unwrap().is_cuda() {
            let states = vec![
                Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap(),
                Tensor::from_slice(&[4.0, 5.0, 6.0], &[3], &Device::Cpu).unwrap(),
            ];

            let result = batch_states(&states, true).unwrap();

            assert!(result.device().is_cuda());
        } else {
            println!("Cuda is not available.");
        }
    }
}
