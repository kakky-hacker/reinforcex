use super::base_distribution::BaseDistribution;
use super::softmax::SoftmaxDistribution;
use tch::{Kind, Tensor};

pub struct MultiSoftmaxDistribution {
    distributions: Vec<SoftmaxDistribution>,
}

unsafe impl Send for MultiSoftmaxDistribution {}
unsafe impl Sync for MultiSoftmaxDistribution {}

impl MultiSoftmaxDistribution {
    pub fn new(distributions: Vec<SoftmaxDistribution>) -> Self {
        assert!(!distributions.is_empty());

        let batch_size = distributions[0].params().0.size()[0];

        for d in &distributions {
            assert_eq!(
                d.params().0.size()[0],
                batch_size,
                "All branches must have same batch size"
            );
        }

        Self { distributions }
    }

    pub fn n_branches(&self) -> usize {
        self.distributions.len()
    }
}

impl BaseDistribution for MultiSoftmaxDistribution {
    fn params(&self) -> (&Tensor, &Tensor) {
        panic!("MultiSoftmaxDistribution::params() is unsupported");
    }

    fn all_prob(&self) -> Tensor {
        panic!("MultiSoftmaxDistribution::all_prob() is unsupported");
    }

    fn all_log_prob(&self) -> Tensor {
        panic!("MultiSoftmaxDistribution::all_log_prob() is unsupported");
    }

    fn detach(&mut self) {
        for d in &mut self.distributions {
            d.detach();
        }
    }

    fn kl(&self, q: &Box<dyn BaseDistribution>) -> Tensor {
        panic!("MultiSoftmaxDistribution::kl() is unsupported");
    }

    fn entropy(&self) -> Tensor {
        let entropies: Vec<Tensor> = self.distributions.iter().map(|d| d.entropy()).collect();

        Tensor::stack(&entropies, -1).sum_dim_intlist([-1].as_ref(), false, Kind::Float)
    }

    fn sample(&self) -> Tensor {
        let samples: Vec<Tensor> = self.distributions.iter().map(|d| d.sample()).collect();

        Tensor::stack(&samples, -1)
    }

    fn prob(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.size()[1] as usize, self.distributions.len());

        let probs: Vec<Tensor> = self
            .distributions
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let action = x.narrow(1, i as i64, 1);

                d.prob(&action)
            })
            .collect();

        Tensor::stack(&probs, -1).prod_dim_int(-1, false, Kind::Float)
    }

    fn log_prob(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.size()[1] as usize, self.distributions.len());

        let log_probs: Vec<Tensor> = self
            .distributions
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let action = x.narrow(1, i as i64, 1);

                d.log_prob(&action)
            })
            .collect();

        Tensor::stack(&log_probs, -1).sum_dim_intlist([-1].as_ref(), false, Kind::Float)
    }

    fn copy(&self) -> Box<dyn BaseDistribution> {
        let copied = self
            .distributions
            .iter()
            .map(|d| SoftmaxDistribution::new(d.params().0.detach(), d.beta(), d.min_prob()))
            .collect();

        Box::new(Self::new(copied))
    }

    fn most_probable(&self) -> Tensor {
        let actions: Vec<Tensor> = self
            .distributions
            .iter()
            .map(|d| d.most_probable())
            .collect();

        Tensor::stack(&actions, -1)
    }

    fn concat(&mut self, _others: Vec<Box<dyn BaseDistribution>>) {
        panic!("MultiSoftmaxDistribution::concat() is unsupported");
    }
}

#[cfg(test)]
mod tests {
    use super::super::softmax::SoftmaxDistribution;
    use super::*;
    use tch::{Kind, Tensor};

    fn make_dist() -> MultiSoftmaxDistribution {
        let logits0 = Tensor::from_slice(&[1.0, 2.0, 3.0, 1.0, 3.0, 2.0]).reshape(&[2, 3]);

        let logits1 = Tensor::from_slice(&[4.0, 1.0, 1.0, 4.0]).reshape(&[2, 2]);

        MultiSoftmaxDistribution::new(vec![
            SoftmaxDistribution::new(logits0, 1.0, 0.0),
            SoftmaxDistribution::new(logits1, 1.0, 0.0),
        ])
    }

    #[test]
    fn test_new() {
        let dist = make_dist();
        assert_eq!(dist.n_branches(), 2);
    }

    #[test]
    fn test_sample() {
        let dist = make_dist();
        let sample = dist.sample();

        assert_eq!(sample.size(), [2, 2]);

        for b in 0..2 {
            let a0 = sample.int64_value(&[b, 0]);
            let a1 = sample.int64_value(&[b, 1]);

            assert!(0 <= a0 && a0 < 3);
            assert!(0 <= a1 && a1 < 2);
        }
    }

    #[test]
    fn test_most_probable() {
        let dist = make_dist();
        let action = dist.most_probable();

        assert_eq!(action.size(), [2, 2]);

        // batch 0:
        // branch 0 logits = [1, 2, 3] -> 2
        // branch 1 logits = [4, 1]    -> 0
        assert_eq!(action.int64_value(&[0, 0]), 2);
        assert_eq!(action.int64_value(&[0, 1]), 0);

        // batch 1:
        // branch 0 logits = [1, 3, 2] -> 1
        // branch 1 logits = [1, 4]    -> 1
        assert_eq!(action.int64_value(&[1, 0]), 1);
        assert_eq!(action.int64_value(&[1, 1]), 1);
    }

    #[test]
    fn test_log_prob() {
        let dist = make_dist();

        let actions = Tensor::from_slice(&[2_i64, 0_i64, 1_i64, 1_i64]).reshape(&[2, 2]);

        let log_prob = dist.log_prob(&actions);

        assert_eq!(log_prob.size(), [2]);

        let d0 = SoftmaxDistribution::new(
            Tensor::from_slice(&[1.0, 2.0, 3.0, 1.0, 3.0, 2.0]).reshape(&[2, 3]),
            1.0,
            0.0,
        );

        let d1 = SoftmaxDistribution::new(
            Tensor::from_slice(&[4.0, 1.0, 1.0, 4.0]).reshape(&[2, 2]),
            1.0,
            0.0,
        );

        let expected =
            d0.log_prob(&actions.narrow(1, 0, 1)) + d1.log_prob(&actions.narrow(1, 1, 1));

        assert!(log_prob.allclose(&expected, 1e-6, 1e-6, true));
    }

    #[test]
    fn test_prob() {
        let dist = make_dist();

        let actions = Tensor::from_slice(&[2_i64, 0_i64, 1_i64, 1_i64]).reshape(&[2, 2]);

        let prob = dist.prob(&actions);
        let log_prob = dist.log_prob(&actions).exp();

        assert_eq!(prob.size(), [2]);
        assert!(prob.allclose(&log_prob, 1e-6, 1e-6, true));
    }

    #[test]
    fn test_entropy() {
        let dist = make_dist();
        let entropy = dist.entropy();

        assert_eq!(entropy.size(), [2]);

        let d0 = SoftmaxDistribution::new(
            Tensor::from_slice(&[1.0, 2.0, 3.0, 1.0, 3.0, 2.0]).reshape(&[2, 3]),
            1.0,
            0.0,
        );

        let d1 = SoftmaxDistribution::new(
            Tensor::from_slice(&[4.0, 1.0, 1.0, 4.0]).reshape(&[2, 2]),
            1.0,
            0.0,
        );

        let expected = d0.entropy() + d1.entropy();

        assert!(entropy.allclose(&expected, 1e-6, 1e-6, true));

        let min_entropy = entropy.min().double_value(&[]);
        assert!(min_entropy >= 0.0);
    }

    #[test]
    fn test_copy() {
        let dist = make_dist();
        let copied = dist.copy();

        let actions = Tensor::from_slice(&[2_i64, 0_i64, 1_i64, 1_i64]).reshape(&[2, 2]);

        let log_prob = dist.log_prob(&actions);
        let copied_log_prob = copied.log_prob(&actions);

        assert!(log_prob.allclose(&copied_log_prob, 1e-6, 1e-6, true));
    }

    #[test]
    fn test_detach() {
        let mut dist = make_dist();
        dist.detach();

        let actions = Tensor::from_slice(&[2_i64, 0_i64, 1_i64, 1_i64]).reshape(&[2, 2]);

        let log_prob = dist.log_prob(&actions);
        assert_eq!(log_prob.size(), [2]);
    }

    #[test]
    #[should_panic]
    fn test_invalid_action_shape() {
        let dist = make_dist();

        let invalid_actions =
            Tensor::from_slice(&[0_i64, 1_i64, 2_i64, 0_i64, 1_i64, 2_i64]).reshape(&[2, 3]);

        dist.log_prob(&invalid_actions);
    }

    #[test]
    #[should_panic]
    fn test_empty_distributions() {
        MultiSoftmaxDistribution::new(vec![]);
    }
}
