pub trait BaseExplorer {
    fn select_action(&self, t: usize, random_action_func: &dyn Fn() -> usize, greedy_action_func: &dyn Fn() -> usize) -> usize;
}