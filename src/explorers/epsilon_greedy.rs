use super::base_explorer::BaseExplorer;
use rand::Rng;

pub struct EpsilonGreedy {
    start_epsilon: f64,
    end_epsilon: f64,
    decay_steps: usize,
}

impl EpsilonGreedy {
    pub fn new(start_epsilon: f64, end_epsilon: f64, decay_steps: usize) -> Self {
        assert!((0.0..=1.0).contains(&start_epsilon));
        assert!((0.0..=1.0).contains(&end_epsilon));
        EpsilonGreedy {
            start_epsilon,
            end_epsilon,
            decay_steps,
        }
    }
}

impl BaseExplorer for EpsilonGreedy {
    fn select_action(
        &self,
        t: usize,
        random_action_func: &dyn Fn() -> usize,
        greedy_action_func: &dyn Fn() -> usize,
    ) -> usize {
        let epsilon;
        if t > self.decay_steps {
            epsilon = self.end_epsilon
        } else {
            epsilon = self.start_epsilon
                + (self.end_epsilon - self.start_epsilon) * (t as f64 / self.decay_steps as f64)
        }

        let action = if rand::thread_rng().gen::<f64>() < epsilon {
            (random_action_func)()
        } else {
            greedy_action_func()
        };

        action
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let explorer = EpsilonGreedy::new(0.9, 0.1, 100);
        assert_eq!(explorer.start_epsilon, 0.9);
        assert_eq!(explorer.end_epsilon, 0.1);
        assert_eq!(explorer.decay_steps, 100);
    }

    #[test]
    #[should_panic]
    fn test_new_invalid_epsilon() {
        EpsilonGreedy::new(1.2, 0.1, 100);
    }

    #[test]
    fn test_select_action_exploration() {
        let explorer = EpsilonGreedy::new(1.0, 1.0, 100);
        let random_action = || 456;
        let greedy_action = || 123;

        let action = explorer.select_action(0, &random_action, &greedy_action);
        assert_eq!(action, 456);
    }

    #[test]
    fn test_select_action_exploitation() {
        let explorer = EpsilonGreedy::new(0.0, 0.0, 100);
        let random_action = || 456;
        let greedy_action = || 123;

        let action = explorer.select_action(50, &random_action, &greedy_action);
        assert_eq!(action, 123);
    }

    #[test]
    fn test_select_action_decay() {
        let explorer = EpsilonGreedy::new(1.0, 0.3, 100);
        let random_action = || 456;
        let greedy_action = || 123;

        let mut random_count = 0;
        let mut greedy_count = 0;

        for t in 0..100 {
            let action = explorer.select_action(t, &random_action, &greedy_action);
            if action == 456 {
                random_count += 1;
            } else {
                greedy_count += 1;
            }
        }

        assert!(random_count > 0);
        assert!(greedy_count > 0);
    }
}
