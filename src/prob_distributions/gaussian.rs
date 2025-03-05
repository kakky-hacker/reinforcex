use super::base_distribution::BaseDistribution;
use candle_core::{shape, DType, Device, Result, Tensor};

pub struct GaussianDistribution {
    mean: Tensor,   // [n]
    var: Tensor,    // [n]
    ln_var: Tensor, // [n]
}

impl GaussianDistribution {
    pub fn new(mean: Tensor, var: Tensor) -> Self {
        let ln_var = var.log().unwrap();
        GaussianDistribution { mean, var, ln_var }
    }
}

impl BaseDistribution for GaussianDistribution {
    fn params(&self) -> (&Tensor, &Tensor) {
        (&self.mean, &self.var)
    }

    fn kl(&self, q: Box<dyn BaseDistribution>) -> Result<Tensor> {
        let (q_mean, q_var) = q.params();
        assert_eq!(self.mean.dims(), q_mean.dims());
        assert_eq!(self.var.dims(), q_var.dims());
        let mean_diff = (&self.mean - q_mean)?.powf(2.0)?; // [n]
        let term1 = (q_var.log() - &self.ln_var)?; // [n]
        let term2 = ((&self.var + &mean_diff) / q_var)?; // [n]
        let res = (0.5
            * (term1 + term2
                - Tensor::new(&[1.0], &Device::Cpu)?.broadcast_as(mean_diff.dims())?)?)?; // [n]
        Ok(res)
    }

    fn entropy(&self) -> Result<Tensor> {
        let log_term: f64 = 0.5 * ((2.0 * std::f64::consts::PI).ln() + 1.0);
        let res = (log_term + (0.5 * &self.ln_var)?)?.sum_all()?; // [1]
        Ok(res)
    }

    fn sample(&self) -> Result<Tensor> {
        let std = self.var.sqrt()?; // [n]
        let shape = self.mean.shape();

        let noise = Tensor::randn(0.0, 1.0, shape, &Device::Cpu)?; // [n]
        &self.mean + &std * noise
    }

    fn prob(&self, x: &Tensor) -> Result<Tensor> {
        let res = self.log_prob(x)?.exp()?; // [n]
        Ok(res)
    }

    fn log_prob(&self, x: &Tensor) -> Result<Tensor> {
        // x.dims() = [n,m]
        let diff = (x - &self.mean.unsqueeze(1)?.broadcast_as(x.dims())?)?.powf(2.0)?; // [n,m]
        let eltwise_log_prob = (-0.5
            * ((2.0 * std::f64::consts::PI).ln()
                + &self.ln_var.unsqueeze(1)?.broadcast_as(x.dims())?
                + (diff / &self.var.unsqueeze(1)?.broadcast_as(x.dims())?)?)?)?; // [n,m]
        let res = eltwise_log_prob.sum(1)?; // [n]
        Ok(res)
    }

    fn copy(&self) -> Box<dyn BaseDistribution> {
        Box::new(GaussianDistribution {
            mean: self.mean.clone(),
            var: self.var.clone(),
            ln_var: self.ln_var.clone(),
        })
    }

    fn most_probable(&self) -> Result<Tensor> {
        let res = self.mean.clone(); // [n]
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn test_new_and_params() {
        let mean = Tensor::from_slice(&[0.0, 0.0, 0.0], &[3], &Device::Cpu).unwrap();
        let var = Tensor::from_slice(&[1.0, 1.0, 1.0], &[3], &Device::Cpu).unwrap();
        let gaussian: GaussianDistribution = GaussianDistribution::new(mean.clone(), var.clone());

        let (mean_out, var_out) = gaussian.params();
        let mean_out_vec: Vec<f64> = mean_out.to_vec1().unwrap();
        let mean_vec: Vec<f64> = mean.to_vec1().unwrap();
        let var_out_vec: Vec<f64> = var_out.to_vec1().unwrap();
        let var_vec: Vec<f64> = var.to_vec1().unwrap();
        assert_eq!(mean_out_vec, mean_vec);
        assert_eq!(var_out_vec, var_vec);
    }

    #[test]
    fn test_most_probable() {
        let mean = Tensor::from_slice(&[0.0, 1.0], &[2], &Device::Cpu).unwrap();
        let var = Tensor::from_slice(&[1.0, 4.0], &[2], &Device::Cpu).unwrap();
        let gaussian: GaussianDistribution = GaussianDistribution::new(mean.clone(), var);

        let most_probable = gaussian.most_probable().unwrap();
        let most_probable_vec: Vec<f64> = most_probable.to_vec1().unwrap();
        let mean_vec: Vec<f64> = mean.to_vec1().unwrap();
        assert_eq!(most_probable_vec, mean_vec);
    }

    #[test]
    fn test_sample() {
        let mean = Tensor::from_slice(&[0.0, 1.0], &[2], &Device::Cpu).unwrap();
        let var = Tensor::from_slice(&[1.0, 4.0], &[2], &Device::Cpu).unwrap();
        let gaussian: GaussianDistribution = GaussianDistribution::new(mean, var);

        let sample = gaussian.sample().unwrap();
        assert_eq!(sample.dims(), vec![2]);
    }

    #[test]
    fn test_log_prob() {
        let mean = Tensor::from_slice(&[0.0, 0.0], &[2], &Device::Cpu).unwrap();
        let var = Tensor::from_slice(&[1.0, 1.0], &[2], &Device::Cpu).unwrap();
        let gaussian = GaussianDistribution::new(mean, var);

        let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &Device::Cpu)
            .unwrap()
            .broadcast_as(&[2, 3])
            .unwrap();
        let log_prob: Vec<f64> = gaussian.log_prob(&x).unwrap().to_vec1().unwrap();
        let expected_log_prob: Vec<f64> = vec![-9.7568156, -9.7568156];
        assert!(log_prob
            .iter()
            .zip(expected_log_prob.iter())
            .all(|(x, y)| (x - y).abs() <= 1e-6));
    }

    #[test]
    fn test_kl_divergence() {
        let mean_p = Tensor::from_slice(&[0.0, 0.0], &[2], &Device::Cpu).unwrap();
        let var_p = Tensor::from_slice(&[1.0, 1.0], &[2], &Device::Cpu).unwrap();
        let gaussian_p: GaussianDistribution = GaussianDistribution::new(mean_p, var_p);

        let mean_q = Tensor::from_slice(&[0.8, 0.8], &[2], &Device::Cpu).unwrap();
        let var_q = Tensor::from_slice(&[1.5, 1.5], &[2], &Device::Cpu).unwrap();
        let gaussian_q: Box<dyn BaseDistribution> =
            Box::new(GaussianDistribution::new(mean_q, var_q));

        let kl_div: Vec<f64> = gaussian_p.kl(gaussian_q).unwrap().to_vec1().unwrap();
        let expected_kl: Vec<f64> = vec![0.249399221, 0.249399221];
        assert!(kl_div
            .iter()
            .zip(expected_kl.iter())
            .all(|(x, y)| (x - y).abs() <= 1e-6));
    }

    #[test]
    fn test_entropy() {
        let mean = Tensor::from_slice(&[0.0], &[1], &Device::Cpu).unwrap();
        let var = Tensor::from_slice(&[1.0], &[1], &Device::Cpu).unwrap();
        let gaussian: GaussianDistribution = GaussianDistribution::new(mean, var);

        let entropy: f64 = gaussian.entropy().unwrap().to_vec0().unwrap();
        let expected_entropy: f64 = 1.418938533;
        assert!((entropy - expected_entropy).abs() < 1e-6);

        let mean = Tensor::from_slice(&[0.0, 0.0], &[2], &Device::Cpu).unwrap();
        let var = Tensor::from_slice(&[1.0, 1.0], &[2], &Device::Cpu).unwrap();
        let gaussian: GaussianDistribution = GaussianDistribution::new(mean, var);

        let entropy: f64 = gaussian.entropy().unwrap().to_vec0().unwrap();
        let expected_entropy: f64 = 1.418938533 * 2.0;
        assert!((entropy - expected_entropy).abs() < 1e-6);
    }
}
