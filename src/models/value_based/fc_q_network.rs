use super::base_q_network::BaseQFunction;
use crate::misc::weight_initializer::{he_init, xavier_init};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder, VarMap};

pub struct FCQNetwork {
    layers: Vec<Linear>,
    n_input_channels: usize,
    action_size: usize,
    n_hidden_layers: usize,
    n_hidden_channels: usize,
}

impl FCQNetwork {
    pub fn new(
        vb: VarBuilder,
        n_input_channels: usize,
        action_size: usize,
        n_hidden_layers: usize,
        n_hidden_channels: Option<usize>,
    ) -> Self {
        let mut layers: Vec<Linear> = Vec::new();
        let n_hidden_channels = n_hidden_channels.unwrap_or(256);

        layers.push(Linear::new(
            vb.get_with_hints(
                (n_hidden_channels, n_input_channels),
                "weight_input",
                he_init(n_input_channels),
            )
            .unwrap(),
            Some(
                vb.get_with_hints(1, "bias_input", Init::Const(0.0))
                    .unwrap(),
            ),
        ));
        for i in 0..n_hidden_layers {
            layers.push(Linear::new(
                vb.get_with_hints(
                    (n_hidden_channels, n_hidden_channels),
                    format!("weight_medium_{}", i).as_str(),
                    he_init(n_hidden_channels),
                )
                .unwrap(),
                Some(
                    vb.get_with_hints(1, format!("bias_medium_{}", i).as_str(), Init::Const(0.0))
                        .unwrap(),
                ),
            ));
        }
        layers.push(Linear::new(
            vb.get_with_hints(
                (action_size, n_hidden_channels),
                "weight_output",
                xavier_init(n_hidden_channels, action_size),
            )
            .unwrap(),
            Some(
                vb.get_with_hints(1, "bias_output", Init::Const(0.0))
                    .unwrap(),
            ),
        ));

        FCQNetwork {
            layers,
            n_input_channels,
            action_size,
            n_hidden_layers,
            n_hidden_channels,
        }
    }
}

impl BaseQFunction for FCQNetwork {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.reshape(((), self.n_input_channels))?;
        for i in 0..self.layers.len() {
            h = self.layers[i].forward(&h)?;
            if i < self.layers.len() - 1 {
                h = h.relu()?;
            }
        }
        h.reshape(((), self.action_size))?.to_dtype(DType::F64)
    }

    fn is_cuda(&self) -> bool {
        self.layers[0].weight().device().is_cuda()
    }

    fn clone(&self) -> Box<dyn BaseQFunction> {
        Box::new(FCQNetwork {
            layers: self.layers.clone(),
            n_input_channels: self.n_input_channels,
            action_size: self.action_size,
            n_hidden_layers: self.n_hidden_layers,
            n_hidden_channels: self.n_hidden_channels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_fcqnetwork_forward() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let n_input_channels = 4;
        let action_size = 2;
        let n_hidden_layers = 2;
        let n_hidden_channels = Some(64);

        let network = FCQNetwork::new(
            vb,
            n_input_channels,
            action_size,
            n_hidden_layers,
            n_hidden_channels,
        );

        let input = Tensor::randn(0.0, 1.0, &[1, n_input_channels], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let output = network.forward(&input).unwrap();

        assert_eq!(output.dims(), vec![1, action_size]);
    }

    #[test]
    fn test_fcqnetwork_clone() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let n_input_channels = 4;
        let action_size = 2;
        let n_hidden_layers = 2;
        let n_hidden_channels = Some(64);

        let network = FCQNetwork::new(
            vb,
            n_input_channels,
            action_size,
            n_hidden_layers,
            n_hidden_channels,
        );
        let cloned_network = network.clone();

        let input = Tensor::randn(0.0, 1.0, &[1, n_input_channels], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let output_original = network.forward(&input).unwrap();
        let output_cloned = cloned_network.forward(&input).unwrap();

        assert_eq!(output_original.dims(), output_cloned.dims());
        let output_original_vec = output_original
            .squeeze(0)
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let output_cloned_vec = output_cloned.squeeze(0).unwrap().to_vec1::<f64>().unwrap();
        assert!(output_original_vec
            .iter()
            .zip(output_cloned_vec.iter())
            .all(|(x, y)| (x - y).abs() <= 1e-6));
    }
}
