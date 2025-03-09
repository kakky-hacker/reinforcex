use super::base_policy_network::BasePolicy;

use crate::misc::weight_initializer::{he_init, xavier_init};
use crate::prob_distributions::BaseDistribution;
use crate::prob_distributions::GaussianDistribution;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder, VarMap};

pub struct FCGaussianPolicy {
    layers: Vec<Linear>,
    mean_layer: Linear,
    var_layer: Linear,
    n_input_channels: usize,
    bound_mean: bool,
    min_action: Option<Tensor>,
    max_action: Option<Tensor>,
    min_var: f64,
}

pub struct FCGaussianPolicyWithValue {
    base_policy: FCGaussianPolicy,
    value_layer: Linear,
}

impl FCGaussianPolicy {
    pub fn new(
        vb: VarBuilder,
        n_input_channels: usize,
        action_size: usize,
        n_hidden_layers: usize,
        n_hidden_channels: Option<usize>,
        min_action: Option<Tensor>,
        max_action: Option<Tensor>,
        bound_mean: bool,
        var_type: &str,
        min_var: f64,
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

        let mean_layer = Linear::new(
            vb.get_with_hints(
                (action_size, n_hidden_channels),
                "weight_mean",
                xavier_init(n_hidden_channels, action_size),
            )
            .unwrap(),
            Some(vb.get_with_hints(1, "bias_mean", Init::Const(0.0)).unwrap()),
        );
        let var_size = if var_type == "spherical" {
            1
        } else {
            action_size
        };
        let var_layer = Linear::new(
            vb.get_with_hints(
                (var_size, n_hidden_channels),
                "weight_var",
                he_init(n_hidden_channels),
            )
            .unwrap(),
            Some(vb.get_with_hints(1, "bias_var", Init::Const(0.0)).unwrap()),
        );

        FCGaussianPolicy {
            layers,
            mean_layer,
            var_layer,
            n_input_channels,
            bound_mean,
            min_action,
            max_action,
            min_var,
        }
    }

    fn compute_medium_layer(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.reshape(((), self.n_input_channels))?;
        for i in 0..self.layers.len() {
            h = self.layers[i].forward(&h)?;
            if i < self.layers.len() - 1 {
                h = h.relu()?;
            }
        }
        Ok(h)
    }

    fn compute_mean_and_var(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = self.mean_layer.forward(&x)?;
        let mean = if self.bound_mean {
            self.bound_by_tanh(mean)?
        } else {
            mean
        };
        let var = ((1.0 + self.var_layer.forward(&x)?.exp()?)?.log()? + self.min_var)?;
        let var = var.broadcast_as(mean.dims())?;
        Ok((mean, var))
    }

    fn bound_by_tanh(&self, x: Tensor) -> Result<Tensor> {
        if self.min_action.is_none() || self.max_action.is_none() {
            return Ok(x);
        }
        let min_action = self.min_action.as_ref().unwrap();
        let max_action = self.max_action.as_ref().unwrap();
        let scale = ((max_action - min_action)? / 2.0)?.broadcast_as(x.dims())?;
        let x_mean = ((max_action + min_action)? / 2.0)?.broadcast_as(x.dims())?;
        x.tanh() * scale + x_mean
    }
}

impl BasePolicy for FCGaussianPolicy {
    fn forward(&self, x: &Tensor) -> Result<(Box<dyn BaseDistribution>, Option<Tensor>)> {
        let h = self.compute_medium_layer(x)?;
        let (mean, var) = self.compute_mean_and_var(&h)?;
        Ok((Box::new(GaussianDistribution::new(mean, var)), None))
    }

    fn is_cuda(&self) -> bool {
        self.layers[0].weight().device().is_cuda()
    }

    fn get_device(&self) -> &Device {
        self.layers[0].weight().device()
    }
}

impl FCGaussianPolicyWithValue {
    pub fn new(
        vb: VarBuilder,
        n_input_channels: usize,
        action_size: usize,
        n_hidden_layers: usize,
        n_hidden_channels: Option<usize>,
        min_action: Option<Tensor>,
        max_action: Option<Tensor>,
        bound_mean: bool,
        var_type: &str,
        min_var: f64,
    ) -> Self {
        let n_hidden_channels = n_hidden_channels.unwrap_or(256);
        let value_layer = Linear::new(
            vb.get_with_hints(
                (n_hidden_channels, 1),
                "weight_value",
                he_init(n_hidden_channels),
            )
            .unwrap(),
            Some(
                vb.get_with_hints(1, "bias_value", Init::Const(0.0))
                    .unwrap(),
            ),
        );

        let base_policy = FCGaussianPolicy::new(
            vb,
            n_input_channels,
            action_size,
            n_hidden_layers,
            Some(n_hidden_channels),
            min_action,
            max_action,
            bound_mean,
            var_type,
            min_var,
        );

        FCGaussianPolicyWithValue {
            base_policy,
            value_layer,
        }
    }

    fn compute_value(&self, x: &Tensor) -> Result<Tensor> {
        self.value_layer.forward(&x)
    }
}

impl BasePolicy for FCGaussianPolicyWithValue {
    fn forward(&self, x: &Tensor) -> Result<(Box<dyn BaseDistribution>, Option<Tensor>)> {
        let h = self.base_policy.compute_medium_layer(x)?;
        let (mean, var) = self.base_policy.compute_mean_and_var(&h)?;
        let value = self.compute_value(&h)?;
        Ok((Box::new(GaussianDistribution::new(mean, var)), Some(value)))
    }

    fn is_cuda(&self) -> bool {
        self.base_policy.is_cuda()
    }

    fn get_device(&self) -> &Device {
        self.base_policy.get_device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_initialization() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let n_input_channels = 4;
        let action_size = 2;
        let n_hidden_layers = 2;
        let n_hidden_channels = Some(64);
        let min_action = Tensor::from_slice(&[-1.0], &[], &Device::Cpu).unwrap();
        let max_action = Tensor::from_slice(&[1.0], &[], &Device::Cpu).unwrap();
        let bound_mean = true;
        let var_type = "spherical";
        let min_var = 1e-3;

        let policy = FCGaussianPolicy::new(
            vb,
            n_input_channels,
            action_size,
            n_hidden_layers,
            n_hidden_channels,
            Some(min_action.clone()),
            Some(max_action.clone()),
            bound_mean,
            var_type,
            min_var,
        );

        assert_eq!(policy.n_input_channels, n_input_channels);
        assert_eq!(policy.layers.len(), n_hidden_layers + 1);
        assert_eq!(policy.bound_mean, bound_mean);
        assert!(policy.min_action.is_some());
        assert!(policy.max_action.is_some());
        assert_eq!(policy.min_var, min_var);
    }

    #[test]
    fn test_compute_mean_and_var() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let n_input_channels = 4;
        let action_size = 6;
        let policy = FCGaussianPolicy::new(
            vb,
            n_input_channels,
            action_size,
            2,
            Some(64),
            None,
            None,
            false,
            "spherical",
            1e-3,
        );

        let input = Tensor::randn(0.0, 1.0, &[3, n_input_channels], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let h = policy.compute_medium_layer(&input).unwrap();
        let (mean, var) = policy.compute_mean_and_var(&h).unwrap();

        assert_eq!(mean.dims()[0], 3);
        assert_eq!(mean.dims()[1], action_size);
        assert_eq!(var.dims()[0], 3);
        assert_eq!(var.dims()[1], action_size);
        assert!(var.min_all().unwrap().to_scalar::<f32>().unwrap() >= 1e-3);
    }

    #[test]
    fn test_bound_by_tanh() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let n_input_channels = 4;
        let action_size = 2;
        let min_action = Tensor::from_slice(&[-1.0], &[], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let max_action = Tensor::from_slice(&[1.0], &[], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let policy = FCGaussianPolicy::new(
            vb,
            n_input_channels,
            action_size,
            2,
            Some(64),
            Some(min_action.copy().unwrap()),
            Some(max_action.copy().unwrap()),
            true,
            "spherical",
            1e-3,
        );

        let unbounded_mean = Tensor::from_slice(&[-2.0, 0.0, 2.0], &[1, 3], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let bounded_mean = policy.bound_by_tanh(unbounded_mean).unwrap();

        assert!(
            bounded_mean.min_all().unwrap().to_scalar::<f32>().unwrap()
                >= min_action.to_scalar::<f32>().unwrap()
        );
        assert!(
            bounded_mean.max_all().unwrap().to_scalar::<f32>().unwrap()
                <= max_action.to_scalar::<f32>().unwrap()
        );
    }

    #[test]
    fn test_forward() {
        let var_map = VarMap::new();
        let device = Device::Cpu;
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);
        let n_input_channels = 4;
        let action_size = 6;
        let policy = FCGaussianPolicy::new(
            vb,
            n_input_channels,
            action_size,
            2,
            Some(64),
            None,
            None,
            false,
            "spherical",
            1e-3,
        );

        let input = Tensor::randn(0.0, 1.0, &[3, n_input_channels], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let action_distribution = policy.forward(&input).unwrap().0;

        let (mean, var) = action_distribution.params();

        assert!(mean.dims() == vec![3, 6]);
        assert!(var.dims() == vec![3, 6]);
    }
}
