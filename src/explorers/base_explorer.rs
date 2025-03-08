use candle_core::Result;

pub trait BaseExplorer {
    fn select_action(
        &self,
        t: usize,
        random_action_func: &dyn Fn() -> Result<usize>,
        greedy_action_func: &dyn Fn() -> Result<usize>,
    ) -> Result<usize>;
}
