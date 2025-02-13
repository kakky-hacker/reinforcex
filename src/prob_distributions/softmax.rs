use tch::Tensor;
use super::base_distribution::BaseDistribution;

pub struct SoftmaxDistribution {
    logits: Tensor,
    beta: f64,
    min_prob: f64,
    n: f64,
}

impl SoftmaxDistribution {
    pub fn new(logits: Tensor, beta: f64, min_prob: f64) -> Self {
        let n = logits.size()[1] as f64;
        assert!(min_prob * n <= 1.0, "Invalid min_prob value");
        Self {
            logits,
            beta,
            min_prob,
            n,
        }
    }

    fn all_prob(&self) -> Tensor {
        let scaled_logits = &self.logits * self.beta;
        if self.min_prob > 0.0 {
            let softmax = scaled_logits.softmax(-1, tch::Kind::Float);
            softmax * (1.0 - self.min_prob * self.n) + self.min_prob
        } else {
            scaled_logits.softmax(-1, tch::Kind::Float)
        }
    }

    fn all_log_prob(&self) -> Tensor {
        if self.min_prob > 0.0 {
            self.all_prob().log()
        } else {
            (&self.logits * self.beta).log_softmax(-1, tch::Kind::Float)
        }
    }
}

impl BaseDistribution for SoftmaxDistribution {
    fn params(&self) -> (&Tensor, &Tensor) {
        (&self.logits, &self.logits)
    }

    fn kl(&self, q: Box<dyn BaseDistribution>) -> Tensor {
        let q_log_prob = q.log_prob(&self.all_prob());
        self.all_prob() * (self.all_log_prob() - q_log_prob)
            .sum_dim_intlist([-1].as_ref(), false, tch::Kind::Float)
    }

    fn entropy(&self) -> Tensor {
        -(&self.all_prob() * self.all_log_prob())
            .sum_dim_intlist([-1].as_ref(), false, tch::Kind::Float)
    }

    fn sample(&self) -> Tensor {
        let probs = self.all_prob();
        let noise = Tensor::rand(&probs.size(), (tch::Kind::Float, probs.device())).log() * -1.0;
        let logits_with_noise = (probs.log() + noise).argmax(-1, false);
        logits_with_noise
    }

    fn prob(&self, x: &Tensor) -> Tensor {
        self.all_prob().gather(-1, x, false).squeeze_dim(-1)
    }

    fn log_prob(&self, x: &Tensor) -> Tensor {
        self.all_log_prob().gather(-1, &x.reshape(&[1, 1]), false).squeeze_dim(-1)
    }

    fn copy(&self) -> Box<dyn BaseDistribution> {
        Box::new(Self::new(self.logits.shallow_clone(), self.beta, self.min_prob))
    }

    fn most_probable(&self) -> Tensor {
        self.all_prob().argmax(-1, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Tensor, Kind};

    #[test]
    fn test_all_prob() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[1, 4]);
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);

        // Test probabilities sum to 1
        let all_prob = dist.all_prob();
        assert!(all_prob.sum(Kind::Float).double_value(&[]) - 1.0 < 1e-6);
    }

    #[test]
    fn test_all_prob_with_min_prob() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[1, 4]);
        let beta = 1.0;
        let min_prob = 0.1;
        let dist = SoftmaxDistribution::new(logits, beta, min_prob);

        // Ensure minimum probability constraint is applied
        let all_prob = dist.all_prob();
        let min_val = all_prob.min().double_value(&[]);
        assert!(min_val >= min_prob);
    }

    #[test]
    fn test_all_log_prob() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[1, 4]);
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);

        let all_log_prob = dist.all_log_prob();
        let log_sum = all_log_prob.sum(Kind::Float).double_value(&[]);
        assert!(log_sum.is_finite());
    }

    #[test]
    fn test_sample() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[1, 4]);
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);
        let sample = dist.sample();
        assert_eq!(sample.size(), [1]);
        assert!(0 <= sample.double_value(&[]) as i64);
        assert!(sample.double_value(&[]) as i64 <= 3);
    }

    #[test]
    fn test_prob() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[1, 4]);
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);
        let prob = dist.prob(&Tensor::from_slice(&[0 as i64]).reshape(&[1, 1]));
        assert!((prob.double_value(&[]) - 0.032058604).abs() < 1e-6);
        let prob = dist.prob(&Tensor::from_slice(&[1 as i64]).reshape(&[1, 1]));
        assert!((prob.double_value(&[]) - 0.087144318).abs() < 1e-6);
        let prob = dist.prob(&Tensor::from_slice(&[2 as i64]).reshape(&[1, 1]));
        assert!((prob.double_value(&[]) - 0.236882818).abs() < 1e-6);
        let prob = dist.prob(&Tensor::from_slice(&[3 as i64]).reshape(&[1, 1]));
        assert!((prob.double_value(&[]) - 0.643914260).abs() < 1e-6);
    }

    #[test]
    fn test_log_prob() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[1, 4]);
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);
        let log_prob = dist.log_prob(&Tensor::from_slice(&[0 as i64]).reshape(&[1, 1]));
        assert!((log_prob.double_value(&[]) - (-3.440189702)).abs() < 1e-6);
        let log_prob = dist.log_prob(&Tensor::from_slice(&[1 as i64]).reshape(&[1, 1]));
        assert!((log_prob.double_value(&[]) - (-2.440189702)).abs() < 1e-6);
        let log_prob = dist.log_prob(&Tensor::from_slice(&[2 as i64]).reshape(&[1, 1]));
        assert!((log_prob.double_value(&[]) - (-1.440189702)).abs() < 1e-6);
        let log_prob = dist.log_prob(&Tensor::from_slice(&[3 as i64]).reshape(&[1, 1]));
        assert!((log_prob.double_value(&[]) - (-0.440189702)).abs() < 1e-6);
    }

    #[test]
    fn test_entropy() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[1, 4]);
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);
        let entropy = dist.entropy();
        assert!(entropy.double_value(&[]) >= 0.0);
        assert!((entropy.double_value(&[]) - 0.947536964).abs() < 1e-6)
    }

    #[test]
    fn test_most_probable() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[1, 4]);
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.0);
        let most_probable = dist.most_probable();
        assert_eq!(most_probable.int64_value(&[]), 3);

        let logits = Tensor::from_slice(&[1.0, 3.5, 1.0, 2.0]).reshape(&[1, 4]);
        let dist = SoftmaxDistribution::new(logits, 1.0, 0.1);
        let most_probable = dist.most_probable();
        assert_eq!(most_probable.int64_value(&[]), 1);

        let logits = Tensor::from_slice(&[1.0, 3.5, 5.0, 2.0]).reshape(&[1, 4]);
        let dist = SoftmaxDistribution::new(logits, 2.0, 0.1);
        let most_probable = dist.most_probable();
        assert_eq!(most_probable.int64_value(&[]), 2);

        let logits = Tensor::from_slice(&[5.1, 3.5, 5.0, 4.0]).reshape(&[1, 4]);
        let dist = SoftmaxDistribution::new(logits, 1.5, 0.0);
        let most_probable = dist.most_probable();
        assert_eq!(most_probable.int64_value(&[]), 0);
    }

    #[test]
    fn test_invalid_min_prob() {
        let logits = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape(&[1, 4]);
        let min_prob = 0.3;
        let result = std::panic::catch_unwind(|| {
            SoftmaxDistribution::new(logits, 1.0, min_prob);
        });
        assert!(result.is_err());
    }
}