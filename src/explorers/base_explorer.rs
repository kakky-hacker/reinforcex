pub trait BaseExplorer {
    fn select_action<F, G>(&self, t: usize, random_action_func: F, greedy_action_func: G) -> usize
    where
        F: Fn() -> usize, G: Fn() -> usize;
}