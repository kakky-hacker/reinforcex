use rand::Rng;

pub struct EpsilonGreedy<F>
where
    F: Fn() -> usize,
{
    start_epsilon: f64,
    end_epsilon: f64,
    decay_steps: usize,
    random_action_func: F,
}

impl<F> EpsilonGreedy<F>
where
    F: Fn() -> usize,
{
    pub fn new(start_epsilon: f64, end_epsilon: f64, decay_steps: usize, random_action_func: F) -> Self {
        assert!((0.0..=1.0).contains(&start_epsilon));
        assert!((0.0..=1.0).contains(&end_epsilon));
        assert!(decay_steps >= 0);
        EpsilonGreedy {
            start_epsilon,
            end_epsilon,
            decay_steps,
            random_action_func,
        }
    }

    pub fn select_action<G>(&self, t: usize, greedy_action_func: G) -> usize
    where
        G: Fn() -> usize,
    {
        let epsilon;
        if t > self.decay_steps {
            epsilon = self.end_epsilon
        } else {
            epsilon = self.start_epsilon + (self.end_epsilon - self.start_epsilon) * (t as f64 / self.decay_steps as f64)
        }
        
        let action = if rand::thread_rng().gen::<f64>() < epsilon { (self.random_action_func)() } else { greedy_action_func() };

        action
    }
}
