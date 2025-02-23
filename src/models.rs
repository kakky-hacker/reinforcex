mod policy_based {
    pub mod base_policy_network;
    pub mod fc_gaussian_policy;
    pub mod fc_softmax_policy;
}

mod value_based {
    pub mod base_q_network;
}

pub use policy_based::base_policy_network::BasePolicy;
pub use policy_based::fc_gaussian_policy::{FCGaussianPolicy, FCGaussianPolicyWithValue};
pub use policy_based::fc_softmax_policy::{FCSoftmaxPolicy, FCSoftmaxPolicyWithValue};

pub use value_based::base_q_network::BaseQFunction;