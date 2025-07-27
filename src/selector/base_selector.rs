use crate::memory::Experience;
use ulid::Ulid;

pub trait BaseSelector {
    fn observe(&mut self, experience: &Experience);
    fn prune(&self, agent_id: &Ulid) -> bool;
    fn select_next_parents(&self, agent_id: &Ulid) -> Vec<&Ulid>;
}
