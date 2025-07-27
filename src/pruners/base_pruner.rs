use ulid::Ulid;

pub trait BasePruner {
    fn step(&mut self);
    fn prune(&self, agent_id: Ulid) -> bool;
}
