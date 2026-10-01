/// Explorers must be safe to move with an agent to a worker thread.
///
/// Non-thread-safe shared state is rejected at compile time:
/// ```compile_fail
/// use reinforcex::explorers::BaseExplorer;
/// use std::{cell::Cell, rc::Rc};
/// struct LocalExplorer(Rc<Cell<usize>>);
/// impl BaseExplorer for LocalExplorer {
///     fn select_action(&self, _: usize, _: &dyn Fn() -> usize, greedy: &dyn Fn() -> usize) -> usize {
///         greedy()
///     }
/// }
/// ```
pub trait BaseExplorer: Send {
    fn select_action(
        &self,
        t: usize,
        random_action_func: &dyn Fn() -> usize,
        greedy_action_func: &dyn Fn() -> usize,
    ) -> usize;
}
