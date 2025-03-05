use super::base_distribution::BaseDistribution;
use candle_core::{shape, DType, Device, Result, Tensor};
use candle_nn::ops;

pub struct SoftmaxDistribution {
    logits: Tensor, // [n,m]
    beta: f64,
    min_prob: f64,
    n: f64,
}

impl SoftmaxDistribution {
    pub fn new(logits: Tensor, beta: f64, min_prob: f64) -> Self {
        let n = logits.dims()[1] as f64;
        assert!(min_prob * n <= 1.0, "Invalid min_prob value");
        Self {
            logits,
            beta,
            min_prob,
            n,
        }
    }

    fn all_prob(&self) -> Result<Tensor> {
        let scaled_logits = (&self.logits * self.beta)?;
        let softmax = ops::softmax(&scaled_logits, 1)?; // [n,m]
        let res;
        if self.min_prob > 0.0 {
            res = ((softmax * (1.0 - self.min_prob * self.n))? + self.min_prob)?;
        } else {
            res = softmax;
        }
        Ok(res)
    }

    fn all_log_prob(&self) -> Result<Tensor> {
        self.all_prob()?.log() // [n,m]
    }
}

impl BaseDistribution for SoftmaxDistribution {
    fn params(&self) -> (&Tensor, &Tensor) {
        (&self.logits, &self.logits)
    }

    fn kl(&self, q: Box<dyn BaseDistribution>) -> Result<Tensor> {
        let q_log_prob = q.log_prob(&self.all_prob()?)?;
        (self.all_prob() * (self.all_log_prob()? - q_log_prob)?)?.sum(1)
    }

    fn entropy(&self) -> Result<Tensor> {
        (-1.0 * (&self.all_prob()? * self.all_log_prob()?)?)?.sum_all()
    }

    fn sample(&self) -> Result<Tensor> {
        let log_probs = self.all_log_prob()?;
        let noise = (Tensor::rand(0.0, 1.0, log_probs.dims(), &Device::Cpu)?.log()? * -1.0)?;
        let res = (log_probs + noise)?.argmax(1)?;
        Ok(res)
    }

    fn prob(&self, x: &Tensor) -> Result<Tensor> {
        // x:[n]
        let probs = self.all_prob()?.gather(x, 1)?; // [n,1]
        let res = probs.squeeze(1)?; // [n]
        Ok(res)
    }

    fn log_prob(&self, x: &Tensor) -> Result<Tensor> {
        let log_probs = self.all_log_prob()?.gather(x, 1)?; // [n,1]
        let res = log_probs.squeeze(1)?; // [n]
        Ok(res)
    }

    fn copy(&self) -> Box<dyn BaseDistribution> {
        Box::new(Self::new(self.logits.clone(), self.beta, self.min_prob))
    }

    fn most_probable(&self) -> Result<Tensor> {
        self.all_prob()?.argmax(1) // [n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn test_all_prob() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);

        // Test probabilities sum to 1
        let all_prob = dist.all_prob().unwrap();
        let sum_all_prob = all_prob.sum_all().unwrap().to_scalar::<f64>().unwrap();
        assert!(sum_all_prob - 1.0 < 1e-6);
    }

    #[test]
    fn test_all_prob_with_min_prob() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
        let beta = 1.0;
        let min_prob = 0.1;
        let dist = SoftmaxDistribution::new(logits, beta, min_prob);

        // Ensure minimum probability constraint is applied
        let all_prob = dist.all_prob().unwrap();
        let min_vals: Vec<f64> = all_prob.min(1).unwrap().to_vec1::<f64>().unwrap();
        assert!(min_vals.iter().all(|&v| v >= min_prob));
    }

    #[test]
    fn test_all_log_prob() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);

        let all_log_prob = dist.all_log_prob().unwrap();
        let log_sum = all_log_prob.sum_all().unwrap().to_scalar::<f64>().unwrap();
        let expected_log_sum = -7.760758794;
        assert!((log_sum - expected_log_sum).abs() < 1e-6);
    }

    #[test]
    fn test_sample() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);
        let sample = dist.sample().unwrap().to_vec1::<u32>().unwrap();
        assert!([0, 1, 2, 3].contains(&sample[0]));
    }

    #[test]
    fn test_prob() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);
        let prob = dist
            .prob(&Tensor::from_slice(&[0 as i64], &[1, 1], &Device::Cpu).unwrap())
            .unwrap();
        assert!((prob.to_vec1::<f64>().unwrap()[0] - 0.032058604).abs() < 1e-6);
        let prob = dist
            .prob(&Tensor::from_slice(&[1 as i64], &[1, 1], &Device::Cpu).unwrap())
            .unwrap();
        assert!((prob.to_vec1::<f64>().unwrap()[0] - 0.087144318).abs() < 1e-6);
        let prob = dist
            .prob(&Tensor::from_slice(&[2 as i64], &[1, 1], &Device::Cpu).unwrap())
            .unwrap();
        assert!((prob.to_vec1::<f64>().unwrap()[0] - 0.236882818).abs() < 1e-6);
        let prob = dist
            .prob(&Tensor::from_slice(&[3 as i64], &[1, 1], &Device::Cpu).unwrap())
            .unwrap();
        assert!((prob.to_vec1::<f64>().unwrap()[0] - 0.643914260).abs() < 1e-6);
    }

    #[test]
    fn test_log_prob() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);
        let log_prob = dist
            .log_prob(&Tensor::from_slice(&[0 as i64], &[1, 1], &Device::Cpu).unwrap())
            .unwrap();
        assert!((log_prob.to_vec1::<f64>().unwrap()[0] - (-3.440189702)).abs() < 1e-6);
        let log_prob = dist
            .log_prob(&Tensor::from_slice(&[1 as i64], &[1, 1], &Device::Cpu).unwrap())
            .unwrap();
        assert!((log_prob.to_vec1::<f64>().unwrap()[0] - (-2.440189702)).abs() < 1e-6);
        let log_prob = dist
            .log_prob(&Tensor::from_slice(&[2 as i64], &[1, 1], &Device::Cpu).unwrap())
            .unwrap();
        assert!((log_prob.to_vec1::<f64>().unwrap()[0] - (-1.440189702)).abs() < 1e-6);
        let log_prob = dist
            .log_prob(&Tensor::from_slice(&[3 as i64], &[1, 1], &Device::Cpu).unwrap())
            .unwrap();
        assert!((log_prob.to_vec1::<f64>().unwrap()[0] - (-0.440189702)).abs() < 1e-6);
    }

    #[test]
    fn test_entropy() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);
        let entropy = dist.entropy().unwrap();
        assert!(entropy.to_scalar::<f64>().unwrap() >= 0.0);
        assert!((entropy.to_scalar::<f64>().unwrap() - 0.947536964).abs() < 1e-6)
    }

    #[test]
    fn test_most_probable() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);
        let most_probable = dist.most_probable().unwrap();
        assert_eq!(most_probable.to_vec1::<u32>().unwrap()[0], 3);

        let logits = Tensor::from_slice(&[1.0, 3.5, 1.0, 2.0], &[1, 4], &Device::Cpu).unwrap();
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.1);
        let most_probable = dist.most_probable().unwrap();
        assert_eq!(most_probable.to_vec1::<u32>().unwrap()[0], 1);

        let logits = Tensor::from_slice(&[1.0, 3.5, 5.0, 2.0], &[1, 4], &Device::Cpu).unwrap();
        let dist = SoftmaxDistribution::new(logits, 2.0, 0.1);
        let most_probable = dist.most_probable().unwrap();
        assert_eq!(most_probable.to_vec1::<u32>().unwrap()[0], 2);

        let logits = Tensor::from_slice(&[5.1, 3.5, 5.0, 4.0], &[1, 4], &Device::Cpu).unwrap();
        let dist = SoftmaxDistribution::new(logits, 1.5, 0.0);
        let most_probable = dist.most_probable().unwrap();
        assert_eq!(most_probable.to_vec1::<u32>().unwrap()[0], 0);
    }
}
