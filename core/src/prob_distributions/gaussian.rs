use super::base_distribution::BaseDistribution;
use tch::{Kind, Tensor};

pub struct GaussianDistribution {
    mean: Tensor,
    var: Tensor,
    min_action: Option<f64>,
    max_action: Option<f64>,
}

unsafe impl Sync for GaussianDistribution {}
unsafe impl Send for GaussianDistribution {}

impl GaussianDistribution {
    pub fn new(mean: Tensor, var: Tensor) -> Self {
        assert_eq!(mean.size(), var.size(), "mean and var must have same shape");
        GaussianDistribution {
            mean,
            var,
            min_action: None,
            max_action: None,
        }
    }

    pub fn new_bounded(mean: Tensor, var: Tensor, min_action: f64, max_action: f64) -> Self {
        assert!(min_action.is_finite());
        assert!(max_action.is_finite());
        assert!(min_action < max_action);
        let mut distribution = Self::new(mean, var);
        distribution.min_action = Some(min_action);
        distribution.max_action = Some(max_action);
        distribution
    }

    fn bound_action(&self, action: Tensor) -> Tensor {
        match (self.min_action, self.max_action) {
            (Some(min_action), Some(max_action)) => action.clamp(min_action, max_action),
            _ => action,
        }
    }
}

impl BaseDistribution for GaussianDistribution {
    fn is_discrete(&self) -> bool {
        false
    }

    fn params(&self) -> (&Tensor, &Tensor) {
        (&self.mean, &self.var)
    }

    fn kl(&self, q: &Box<dyn BaseDistribution>) -> Tensor {
        let (q_mean, q_var) = q.params();
        let mean_diff = (&self.mean - q_mean).pow_tensor_scalar(2.0);
        let term1 = q_var.log() - &self.var.log();
        let term2 = (&self.var + mean_diff) / q_var;
        0.5 * (term1 + term2 - 1.0).sum_dim_intlist([-1].as_ref(), false, Kind::Float)
    }

    fn entropy(&self) -> Tensor {
        let dim = self.mean.size()[1];
        let log_term = 0.5 * ((2.0 * std::f64::consts::PI).ln() + 1.0);
        log_term * dim as f64
            + 0.5
                * self
                    .var
                    .log()
                    .sum_dim_intlist([-1].as_ref(), false, Kind::Float)
    }

    fn sample(&self) -> Tensor {
        let std = self.var.sqrt();
        let noise = Tensor::randn_like(&self.mean);
        self.bound_action(&self.mean + &std * noise).detach()
    }

    fn prob(&self, x: &Tensor) -> Tensor {
        self.log_prob(x).exp()
    }

    fn log_prob(&self, x: &Tensor) -> Tensor {
        let diff = (x - &self.mean).pow_tensor_scalar(2.0);
        let log_prob_each_dim: Tensor =
            -0.5 * ((2.0 * std::f64::consts::PI).ln() + &self.var.log() + &diff / &self.var);
        log_prob_each_dim.sum_dim_intlist([-1].as_ref(), false, Kind::Float)
    }

    fn copy(&self) -> Box<dyn BaseDistribution> {
        let mut copy = Self::new(
            self.mean.shallow_clone().detach(),
            self.var.shallow_clone().detach(),
        );
        copy.min_action = self.min_action;
        copy.max_action = self.max_action;
        Box::new(copy)
    }

    fn most_probable(&self) -> Tensor {
        self.bound_action(self.mean.shallow_clone())
    }

    fn detach(&mut self) {
        self.mean = self.mean.detach();
        self.var = self.var.detach();
    }

    fn concat(&mut self, others: Vec<Box<dyn BaseDistribution>>) {
        let means = others
            .iter()
            .map(|d| d.params().0.shallow_clone())
            .collect::<Vec<Tensor>>();
        let vars = others
            .iter()
            .map(|d| d.params().1.shallow_clone())
            .collect::<Vec<Tensor>>();
        self.mean = Tensor::cat(&means, 0);
        self.var = Tensor::cat(&vars, 0);
    }

    fn all_prob(&self) -> Tensor {
        Tensor::new()
    }

    fn all_log_prob(&self) -> Tensor {
        Tensor::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_new_and_params() {
        let mean = Tensor::from_slice(&[0.0, 0.0, 0.0]).view([1, 3]);
        let var = Tensor::from_slice(&[1.0, 1.0, 1.0]).view([1, 3]);
        let gaussian: GaussianDistribution =
            GaussianDistribution::new(mean.shallow_clone(), var.shallow_clone());

        let (mean_out, var_out) = gaussian.params();
        assert_eq!(mean_out, &mean);
        assert_eq!(var_out, &var);
    }

    #[test]
    fn test_most_probable() {
        let mean = Tensor::from_slice(&[0.0, 1.0]).view([1, 2]);
        let var = Tensor::from_slice(&[1.0, 4.0]).view([1, 2]);
        let gaussian: GaussianDistribution = GaussianDistribution::new(mean.shallow_clone(), var);

        let most_probable = gaussian.most_probable();
        assert_eq!(most_probable, mean);
    }

    #[test]
    fn test_sample() {
        let mean = Tensor::from_slice(&[0.0, 1.0]).view([1, 2]);
        let var = Tensor::from_slice(&[1.0, 4.0]).view([1, 2]);
        let gaussian: GaussianDistribution = GaussianDistribution::new(mean, var);

        let sample = gaussian.sample();
        assert_eq!(sample.size(), vec![1, 2]);
    }

    #[test]
    fn test_bounded_sample_and_mode_stay_in_action_space() {
        let mean = Tensor::from_slice(&[-5.0, 5.0]).view([1, 2]);
        let var = Tensor::from_slice(&[100.0, 100.0]).view([1, 2]);
        let gaussian = GaussianDistribution::new_bounded(mean, var, -1.0, 1.0);

        for _ in 0..100 {
            let sample = gaussian.sample();
            assert!(sample.ge(-1.0).all().int64_value(&[]) == 1);
            assert!(sample.le(1.0).all().int64_value(&[]) == 1);
        }
        let mode = gaussian.most_probable();
        assert_eq!(mode.double_value(&[0, 0]), -1.0);
        assert_eq!(mode.double_value(&[0, 1]), 1.0);
    }

    #[test]
    fn test_log_prob() {
        let mean = Tensor::from_slice(&[0.0]).view([1, 1]);
        let var = Tensor::from_slice(&[1.0]).view([1, 1]);
        let gaussian = GaussianDistribution::new(mean, var);

        let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).view([1, 3]);
        let log_prob = gaussian.log_prob(&x);
        let expected_log_prob: f64 = -9.7568156;
        assert!((log_prob.double_value(&[]) - expected_log_prob).abs() < 1e-6);
    }

    #[test]
    fn test_kl_divergence() {
        let mean_p = Tensor::from_slice(&[0.0]).view([1, 1]);
        let var_p = Tensor::from_slice(&[1.0]).view([1, 1]);
        let gaussian_p: GaussianDistribution = GaussianDistribution::new(mean_p, var_p);

        let mean_q = Tensor::from_slice(&[0.8]).view([1, 1]);
        let var_q = Tensor::from_slice(&[1.5]).view([1, 1]);
        let gaussian_q: Box<dyn BaseDistribution> =
            Box::new(GaussianDistribution::new(mean_q, var_q));

        let kl_div = gaussian_p.kl(&gaussian_q);
        let expected_kl: f64 = 0.249399221;
        assert!((kl_div.double_value(&[]) - expected_kl).abs() < 1e-6);
    }

    #[test]
    fn test_entropy() {
        let mean = Tensor::from_slice(&[0.0]).view([1, 1]);
        let var = Tensor::from_slice(&[1.0]).view([1, 1]);
        let gaussian: GaussianDistribution = GaussianDistribution::new(mean, var);

        let entropy = gaussian.entropy();
        let expected_entropy: f64 = 1.418938533;
        assert!((entropy.double_value(&[]) - expected_entropy).abs() < 1e-6);

        let mean = Tensor::from_slice(&[0.0, 0.0]).view([1, 2]);
        let var = Tensor::from_slice(&[1.0, 1.0]).view([1, 2]);
        let gaussian: GaussianDistribution = GaussianDistribution::new(mean, var);

        let entropy = gaussian.entropy();
        let expected_entropy: f64 = 1.418938533 * 2.0;
        assert!((entropy.double_value(&[]) - expected_entropy).abs() < 1e-6);
    }
}
