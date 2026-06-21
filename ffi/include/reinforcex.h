#ifndef REINFORCEX_H
#define REINFORCEX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    RX_OK = 0,
    RX_ERROR_NULL_POINTER = -1,
    RX_ERROR_INVALID_ARGUMENT = -2,
    RX_ERROR_NOT_FOUND = -3,
    RX_ERROR_BUFFER_TOO_SMALL = -4,
    RX_ERROR_PANIC = -5,
    RX_ERROR_INTERNAL = -6
};

enum {
    RX_ACTION_DISCRETE = 0,
    RX_ACTION_CONTINUOUS = 1
};

typedef struct RxAgentConfig {
    uint64_t obs_size;
    uint64_t action_size;
    uint64_t hidden_layers;
    uint64_t hidden_size;
    double gamma;
} RxAgentConfig;

typedef struct RxDqnConfig {
    RxAgentConfig agent;
    double learning_rate;
    uint64_t batch_size;
    uint64_t replay_capacity;
    uint64_t replay_n_steps;
    uint64_t update_interval;
    uint64_t target_update_interval;
    double epsilon_start;
    double epsilon_end;
    uint64_t epsilon_decay_steps;
} RxDqnConfig;

typedef struct RxPpoConfig {
    RxAgentConfig agent;
    uint32_t action_space;
    double learning_rate;
    double gae_lambda;
    uint64_t update_interval;
    uint64_t epochs;
    uint64_t minibatch_size;
    double policy_clip_epsilon;
    double value_clip_range;
    double value_loss_coefficient;
    double entropy_coefficient;
    uint32_t standardize_gae;
    double min_action;
    double max_action;
    double min_variance;
} RxPpoConfig;

typedef struct RxSacConfig {
    RxAgentConfig agent;
    uint32_t action_space;
    double actor_learning_rate;
    double critic_learning_rate;
    uint64_t replay_capacity;
    uint64_t replay_start_size;
    uint64_t batch_size;
    uint64_t replay_n_steps;
    uint64_t update_interval;
    uint64_t target_update_interval;
    double tau;
    double alpha;
    double min_variance;
    uint32_t squash_action;
} RxSacConfig;

int32_t rx_dqn_config_default(
    RxDqnConfig *out_config,
    uint64_t obs_size,
    uint64_t action_size);

int32_t rx_ppo_config_default(
    RxPpoConfig *out_config,
    uint64_t obs_size,
    uint64_t action_size);

int32_t rx_sac_config_default(
    RxSacConfig *out_config,
    uint64_t obs_size,
    uint64_t action_size);

int32_t rx_dqn_create(const RxDqnConfig *config, uint64_t *out_id);
int32_t rx_ppo_create(const RxPpoConfig *config, uint64_t *out_id);
int32_t rx_sac_create(const RxSacConfig *config, uint64_t *out_id);

/* Returns the number of floats written, or a negative RX_ERROR_* value. */
int64_t rx_agent_act(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    float *out,
    uint64_t out_len);

/* Returns the number of floats written, or a negative RX_ERROR_* value. */
int64_t rx_agent_act_and_train(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    float reward,
    float *out,
    uint64_t out_len);

int32_t rx_agent_stop_episode(
    uint64_t id,
    const float *obs,
    uint64_t obs_len,
    float reward);

int32_t rx_agent_destroy(uint64_t id);

#ifdef __cplusplus
}
#endif

#endif
