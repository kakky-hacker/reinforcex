use super::base_q_network::BaseQFunction;
use tch::{Tensor, nn};
use tch::nn::{Module, Linear, LinearConfig, Init};
use crate::misc::weight_initializer::he_init;

pub struct FCQNetwork {
    layers: Vec<Linear>,
    n_input_channels: i64,
    action_size: i64,
}


impl FCQNetwork {
    pub fn new(vs: &nn::VarStore, n_input_channels: i64, action_size: i64, 
           n_hidden_layers: usize, n_hidden_channels: Option<i64>) -> Self {

        let root = vs.root(); 
        let mut layers: Vec<Linear> = Vec::new();
        let n_hidden_channels = n_hidden_channels.unwrap_or(256);

        layers.push(nn::linear(&root, n_input_channels, n_hidden_channels, LinearConfig {
            ws_init: he_init(n_input_channels),
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        }));
        for _ in 0..n_hidden_layers - 1 {
            layers.push(nn::linear(&root, n_hidden_channels, n_hidden_channels, LinearConfig {
                ws_init: he_init(n_hidden_channels),
                bs_init: Some(Init::Const(0.0)),
                bias: true,
            }));
        }
        layers.push(nn::linear(&root, n_hidden_channels, action_size, LinearConfig {
            ws_init: he_init(n_input_channels),
            bs_init: Some(Init::Const(0.0)),
            bias: true,
        }));
        
        FCQNetwork {
            layers,
            n_input_channels,
            action_size,
        }
    }
}

impl BaseQFunction for FCQNetwork {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.view([-1, self.n_input_channels]);
        for layer in &self.layers {
            h = (layer.forward(&h)).relu();
        }
        h.view([-1, self.action_size])
    }

    fn is_cuda(&self) -> bool {
        false
    }

    fn clone(&self) -> Box<dyn BaseQFunction> {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let mut cloned_network = FCQNetwork::new(
            &vs,
            self.n_input_channels,
            self.action_size,
            self.layers.len() - 1,
            Some(self.layers[0].ws.size()[0]),
        );

        for (cloned_layer, original_layer) in cloned_network.layers.iter_mut().zip(&self.layers) {
            cloned_layer.ws.copy_(&original_layer.ws);
            if let Some(ref mut cloned_bs) = cloned_layer.bs {
                if let Some(ref original_bs) = original_layer.bs {
                    cloned_bs.copy_(original_bs);
                }
            }
        }

        Box::new(cloned_network)
    }
}