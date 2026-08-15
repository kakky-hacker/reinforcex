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

enum {
    RX_STAT_NAME_LEN = 64
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
    double discrete_target_entropy_ratio;
    double min_variance;
    uint32_t squash_action;
} RxSacConfig;

typedef struct RxReplayBufferConfig {
    uint64_t capacity;
    uint64_t n_steps;
} RxReplayBufferConfig;

typedef struct RxRndConfig {
    uint64_t obs_size;
    uint64_t feature_size;
    uint64_t hidden_layers;
    uint64_t hidden_size;
    double learning_rate;
    uint64_t update_interval;
} RxRndConfig;

typedef struct RxStatistic {
    char name[RX_STAT_NAME_LEN];
    double value;
} RxStatistic;

uint32_t rx_cuda_is_available(void);
int32_t rx_manual_seed(int64_t seed);

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

int32_t rx_replay_buffer_config_default(
    RxReplayBufferConfig *out_config,
    uint64_t capacity,
    uint64_t n_steps);

int32_t rx_rnd_config_default(
    RxRndConfig *out_config,
    uint64_t obs_size);

int32_t rx_dqn_create(const RxDqnConfig *config, uint64_t *out_id);
int32_t rx_ppo_create(const RxPpoConfig *config, uint64_t *out_id);
int32_t rx_sac_create(const RxSacConfig *config, uint64_t *out_id);

int32_t rx_dqn_create_with_paths(
    const RxDqnConfig *config,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_dqn_create_with_replay(
    const RxDqnConfig *config,
    uint64_t replay_id,
    uint64_t *out_id);

int32_t rx_dqn_create_with_replay_and_paths(
    const RxDqnConfig *config,
    uint64_t replay_id,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_ppo_create_with_paths(
    const RxPpoConfig *config,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_ppo_create_with_replay(
    const RxPpoConfig *config,
    uint64_t replay_id,
    uint64_t *out_id);

int32_t rx_ppo_create_with_replay_and_paths(
    const RxPpoConfig *config,
    uint64_t replay_id,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_ppo_create_with_rnd(
    const RxPpoConfig *config,
    uint64_t rnd_id,
    double curiosity_reward_coefficient,
    uint64_t *out_id);

int32_t rx_ppo_create_with_rnd_and_paths(
    const RxPpoConfig *config,
    uint64_t rnd_id,
    double curiosity_reward_coefficient,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_ppo_create_with_rnd_and_replay(
    const RxPpoConfig *config,
    uint64_t rnd_id,
    uint64_t replay_id,
    double curiosity_reward_coefficient,
    uint64_t *out_id);

int32_t rx_ppo_create_with_rnd_and_replay_and_paths(
    const RxPpoConfig *config,
    uint64_t rnd_id,
    uint64_t replay_id,
    double curiosity_reward_coefficient,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_sac_create_with_paths(
    const RxSacConfig *config,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_replay_buffer_create(
    const RxReplayBufferConfig *config,
    uint64_t *out_id);

int32_t rx_sac_create_with_replay(
    const RxSacConfig *config,
    uint64_t replay_id,
    uint64_t *out_id);

int32_t rx_sac_create_with_replay_and_paths(
    const RxSacConfig *config,
    uint64_t replay_id,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

int32_t rx_rnd_create(const RxRndConfig *config, uint64_t *out_id);

int32_t rx_rnd_create_with_paths(
    const RxRndConfig *config,
    const char *save_path,
    const char *load_path,
    uint64_t *out_id);

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

int32_t rx_agent_statistics_len(uint64_t id, uint64_t *out_len);

/* Returns the number of statistics written, or a negative RX_ERROR_* value. */
int64_t rx_agent_statistics(
    uint64_t id,
    RxStatistic *out_stats,
    uint64_t out_len);

int32_t rx_agent_save(uint64_t id);
int32_t rx_agent_load(uint64_t id);

int32_t rx_replay_buffer_destroy(uint64_t id);
int32_t rx_replay_buffer_len(uint64_t id, uint64_t *out_len);

int32_t rx_rnd_save(uint64_t id);
int32_t rx_rnd_load(uint64_t id);
int32_t rx_rnd_destroy(uint64_t id);

int32_t rx_agent_destroy(uint64_t id);

#ifdef __cplusplus
}
#endif

#endif
