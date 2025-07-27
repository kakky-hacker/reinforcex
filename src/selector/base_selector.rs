use crate::memory::Experience;
use ulid::Ulid;

pub trait BaseSelector {
    fn step(&mut self, experience: &Experience);
    fn prune(&self, agent_id: &Ulid) -> bool;
}
