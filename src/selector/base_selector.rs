use crate::memory::Experience;
use ulid::Ulid;

pub trait BaseSelector: Send + Sync {
    fn observe(&self, experience: &Experience);
    fn delete(&self, agent_id: &Ulid);
    fn find_pareto_dominant(&self, agent_id: &Ulid) -> Vec<Ulid>;
}
