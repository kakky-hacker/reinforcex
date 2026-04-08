use super::base_q_network::BaseQFunction;
use crate::misc::weight_initializer::{he_init, xavier_init};
use tch::nn::{linear, Init, Linear, LinearConfig, Module, VarStore};
use tch::{no_grad, Device, Tensor};

pub struct FCQNetwork {
    vs: VarStore,
    layers: Vec<Linear>,
    n_input_channels: i64,
    action_size: i64,
    n_hidden_layers: usize,
    n_hidden_channels: i64,
}

impl FCQNetwork {
    pub fn new(
        vs: VarStore,
        n_input_channels: i64,
        action_size: i64,
        n_hidden_layers: usize,
        n_hidden_channels: i64,
    ) -> Self {
        let root = (&vs).root();
        let mut layers: Vec<Linear> = Vec::new();

        layers.push(linear(
            &root,
            n_input_channels,
            n_hidden_channels,
            LinearConfig {
                ws_init: he_init(n_input_channels),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        ));
        for _ in 0..n_hidden_layers {
            layers.push(linear(
                &root,
                n_hidden_channels,
                n_hidden_channels,
                LinearConfig {
                    ws_init: he_init(n_hidden_channels),
                    bs_init: Some(Init::Const(0.0)),
                    bias: true,
                },
            ));
        }
        layers.push(linear(
            &root,
            n_hidden_channels,
            action_size,
            LinearConfig {
                ws_init: xavier_init(n_hidden_channels, action_size),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            },
        ));

        FCQNetwork {
            vs,
            layers,
            n_input_channels,
            action_size,
            n_hidden_layers,
            n_hidden_channels,
        }
    }
}

impl BaseQFunction for FCQNetwork {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.view([-1, self.n_input_channels]);
        for i in 0..self.layers.len() {
            h = self.layers[i].forward(&h);
            if i < self.layers.len() - 1 {
                h = h.relu();
            }
        }
        h.view([-1, self.action_size])
    }

    fn device(&self) -> Device {
        self.vs.device()
    }

    fn clone(&self) -> Box<dyn BaseQFunction> {
        let vs = VarStore::new(self.device());
        let mut cloned_network = FCQNetwork::new(
            vs,
            self.n_input_channels,
            self.action_size,
            self.n_hidden_layers,
            self.n_hidden_channels,
        );

        no_grad(|| {
            for (cloned_layer, original_layer) in cloned_network.layers.iter_mut().zip(&self.layers)
            {
                cloned_layer.ws.copy_(&original_layer.ws);
                if let Some(ref mut cloned_bs) = cloned_layer.bs {
                    if let Some(ref original_bs) = &original_layer.bs {
                        cloned_bs.copy_(original_bs);
                    }
                }
            }
        });

        Box::new(cloned_network)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Tensor};

    #[test]
    fn test_fcqnetwork_forward() {
        let vs = VarStore::new(Device::Cpu);
        let n_input_channels = 4;
        let action_size = 2;
        let n_hidden_layers = 2;
        let n_hidden_channels = 64;

        let network = FCQNetwork::new(
            vs,
            n_input_channels,
            action_size,
            n_hidden_layers,
            n_hidden_channels,
        );

        let input = Tensor::randn([1, n_input_channels], (tch::Kind::Float, Device::Cpu));
        let output = network.forward(&input);

        assert_eq!(output.size(), vec![1, action_size]);
    }

    #[test]
    fn test_fcqnetwork_clone() {
        let vs = VarStore::new(Device::Cpu);
        let n_input_channels = 4;
        let action_size = 2;
        let n_hidden_layers = 2;
        let n_hidden_channels = 64;

        let network = FCQNetwork::new(
            vs,
            n_input_channels,
            action_size,
            n_hidden_layers,
            n_hidden_channels,
        );
        let cloned_network = network.clone();

        let input = Tensor::randn([1, n_input_channels], (tch::Kind::Float, Device::Cpu));
        let output_original = network.forward(&input);
        let output_cloned = cloned_network.forward(&input);

        assert_eq!(output_original.size(), output_cloned.size());
        assert!(output_original.allclose(&output_cloned, 1e-6, 1e-6, false));
    }
}
