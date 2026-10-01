#!/usr/bin/env python3
"""Read-only aggregation/plots for the new core improvement study.

python benchmarks/render_improvement_report.py --no-plots  # standard library
python benchmarks/render_improvement_report.py             # matplotlib needed
Never loads a checkpoint, calls an environment, or runs learning/inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / 'reports/core_improvements_20261001'
FINAL_EPISODES = 100
THRESHOLDS = {'CartPole-v1': 475., 'HalfCheetah-v5': 4800., 'Hopper-v5': 3800.}
TARGET_CASES = {'cartpole_dqn', 'cartpole_ppo', 'halfcheetah_hybrid', 'hopper_sac'}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def evaluation_protocol(stage: str, validation_seed: int, validation_episodes: int,
                        final_seed: int, final_episodes: int) -> dict:
    if stage not in ('development', 'confirmation'):
        raise ValueError('unknown study stage')
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1
           for x in (validation_seed, validation_episodes, final_seed, final_episodes)):
        raise ValueError('invalid evaluation seed/count')
    # Confirmation rounds may reserve a new block; never silently reuse development.
    if validation_seed != 1100000 or (stage == 'development' and final_seed != 1100000) or (
            stage == 'confirmation' and (final_seed < 1200000 or final_seed % 100000)):
        raise ValueError('evaluation seeds differ from the improvement protocol')
    if not (2 <= validation_episodes <= 100000 and 2 <= final_episodes <= 100000):
        raise ValueError('evaluation episode count leaves its reserved seed block')
    return {'validation_seed': validation_seed, 'validation_episodes': validation_episodes,
            'final_seed': final_seed, 'final_episodes': final_episodes,
            'final_split': 'test' if stage == 'confirmation' else 'development', 'deterministic': True}


def comparison_settings(configuration: dict, agents: list[dict], max_episode_steps: Any) -> dict:
    return {'env_id': configuration.get('env_id'), 'reward_mode': configuration['reward_mode'],
            'agents': agents, 'shared_replay': bool(configuration.get('shared_replay', False)),
            'max_episode_steps': max_episode_steps}


def apply_decisions(runs: list[dict], path: Path | None) -> dict | None:
    if path is None:
        return None
    document = read_json(path)
    if document.get('schema_version') != 1 or not isinstance(document.get('decisions'), list):
        raise ValueError('invalid report decision schema')
    seen = set()
    for decision in document['decisions']:
        name = decision['run']
        selected = [run for run in runs if run['run'] == name]
        if name in seen or len(selected) != 1:
            raise ValueError(f'decision must select exactly one unique run: {name}')
        seen.add(name)
        run = selected[0]
        if decision.get('metadata_sha256') != run['metadata_sha256']:
            raise ValueError(f'decision metadata SHA mismatch: {name}')
        if decision.get('interpretation') not in ('redevelopment', 'provenance_discrepancy') or not decision.get('reason'):
            raise ValueError(f'unsupported or unexplained interpretation: {name}')
        run['interpretation'] = decision['interpretation']
        run['decision'] = decision
        run['confirmation_evidence_eligible'] = False
        run['achievement_evidence_eligible'] = False
    return {'source': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **document}


def relative(path: Path, output: Path) -> str:
    return Path(os.path.relpath(path.resolve(), output.resolve())).as_posix()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected JSON object')
    return value


def read_jsonl(path: Path, live: bool, warnings: list[str]) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_text()
    rows = []
    lines = data.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('expected JSON object')
            rows.append(row)
        except ValueError:
            if live and index == len(lines) - 1 and not data.endswith('\n'):
                warnings.append(f'{path}: ignored incomplete trailing line while writer is active')
                break
            raise ValueError(f'{path}:{index + 1}: malformed JSON record') from None
    return rows


def variant_identity(metadata: dict) -> dict:
    sha = metadata['library_sha256']
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('library_sha256 must be a complete SHA256')
    identity = {'build_sha256': sha, 'overrides': metadata.get('overrides'),
                'case': metadata['case'], 'requested_total_steps': metadata['requested_total_steps']}
    # Worker topology changes the experiment even with identical constructor
    # overrides (e.g. two agents sharing one RND predictor versus private RND).
    # Keep these conditions separate before duplicate-seed/contract checks.
    topology = {key: metadata[key] for key in
                ('worker_count', 'concurrent_workers', 'share_rnd')}
    if type(topology['worker_count']) is not int or topology['worker_count'] < 1:
        raise ValueError('worker_count must be a positive integer')
    if any(type(topology[key]) is not bool for key in ('concurrent_workers', 'share_rnd')):
        raise ValueError('concurrent_workers and share_rnd must be booleans')
    identity['worker_topology'] = topology
    schedules = [agent.get('learning_rate_schedule') for agent in metadata.get('effective_agents', [])]
    if any(schedule is not None for schedule in schedules):
        identity['learning_rate_schedules'] = schedules
    return identity


def variant_id(metadata: dict) -> str:
    return hashlib.sha256(canonical(variant_identity(metadata)).encode()).hexdigest()[:16]


def evaluation_mean(row: dict, where: str) -> float:
    values = row.get('returns', [])
    if not values or not all(finite(value) for value in values):
        raise ValueError(f'{where}: missing or nonfinite raw evaluation returns')
    if row.get('reward') != 'raw' or row.get('episodes') != len(values):
        raise ValueError(f'{where}: raw reward/episode-count mismatch')
    mean = statistics.fmean(values)
    if not finite(row.get('mean')) or not math.isclose(mean, row['mean'], rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError(f'{where}: stored mean disagrees with returns')
    return mean


def moving_average(values: list[float], requested_window: int = 100) -> tuple[list[int], list[float], int]:
    """Full trailing windows only; short runs use their total observed length."""
    if requested_window < 1:
        raise ValueError('window must be positive')
    window = min(requested_window, len(values))
    if not window:
        return [], [], 0
    queue: deque[float] = deque()
    total = 0.
    positions, result = [], []
    for index, value in enumerate(values):
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.popleft()
        if len(queue) == window:
            positions.append(index)
            result.append(total / window)
    return positions, result, window


def load_run(path: Path, output: Path, window: int) -> dict:
    meta_path = path / 'metadata.json'
    meta = read_json(meta_path)
    if meta.get('backend') != 'reinforcex':
        raise ValueError(f'{meta_path}: only native runs are supported; external controls require separate grouping')
    identity = variant_identity(meta)
    stage = meta.get('study_stage')
    protocol = evaluation_protocol(stage, meta.get('validation_seed'), meta.get('validation_episodes'),
                                   meta.get('final_test_seed'), meta.get('final_test_episodes'))
    agents = meta['effective_agents']
    if len(agents) != meta['worker_count'] or not agents:
        raise ValueError(f'{meta_path}: worker-count mismatch')
    warnings: list[str] = []
    final_path = path / 'final.json'
    final = read_json(final_path) if final_path.exists() else None
    status = 'running_or_interrupted' if final is None else 'invalid_final'
    if final is None and (path / 'failure.json').exists():
        status = 'failed'
    threshold = meta.get('reward_threshold')
    if meta['env_id'] in THRESHOLDS and threshold != THRESHOLDS[meta['env_id']]:
        raise ValueError(f'{meta_path}: threshold differs from the improvement protocol')
    if threshold is not None and not finite(threshold):
        raise ValueError(f'{meta_path}: nonfinite threshold')
    evaluations = read_jsonl(path / 'evaluations.jsonl', final is None, warnings)
    for row in evaluations:
        worker = row.get('worker')
        if not isinstance(worker, int) or worker < 0 or worker >= len(agents):
            raise ValueError(f'{path}: evaluation for unknown worker')
        if row.get('algorithm') != agents[worker]['algorithm']:
            raise ValueError(f'{path}: evaluation algorithm mismatch')
        evaluation_mean(row, str(path))
        if row.get('deterministic') is not True:
            raise ValueError(f'{path}: evaluation is not deterministic')
        if row.get('split') == 'validation' and (row.get('seed_start') != meta['validation_seed']
                                                or row['episodes'] != meta['validation_episodes']):
            raise ValueError(f'{path}: validation seed/episode-count mismatch')
        if not finite(row.get('aggregate_steps')) or not 0 <= row['aggregate_steps'] <= identity['requested_total_steps']:
            raise ValueError(f'{path}: evaluation step outside requested budget')
    final_rows: dict[int, dict] = {}
    counter_rows: dict[int, dict] = {}
    if final is not None:
        if final.get('status') != 'complete' or final.get('actual_total_steps') != identity['requested_total_steps']:
            raise ValueError(f'{final_path}: status/budget is not complete')
        for key in ('case', 'env_id', 'seed', 'study_stage'):
            if final.get(key) != meta.get(key):
                raise ValueError(f'{final_path}: {key} disagrees with metadata')
        if final.get('threshold') != threshold:
            raise ValueError(f'{final_path}: threshold mismatch')
        for key, destination in [('test', final_rows), ('workers', counter_rows)]:
            for row in final.get(key, []):
                index = row.get('worker')
                if index in destination or not isinstance(index, int) or not 0 <= index < len(agents):
                    raise ValueError(f'{final_path}: duplicate/unknown {key} worker')
                if row.get('algorithm') != agents[index]['algorithm']:
                    raise ValueError(f'{final_path}: {key} algorithm mismatch')
                destination[index] = row
            if set(destination) != set(range(len(agents))):
                raise ValueError(f'{final_path}: missing {key} workers')
        if sum(row['steps'] for row in counter_rows.values()) != identity['requested_total_steps']:
            raise ValueError(f'{final_path}: worker steps do not sum to budget')
        status = 'complete' if all(len(row.get('returns', [])) == FINAL_EPISODES for row in final_rows.values()) else 'complete_nonstandard_evaluation'
    workers = []
    for worker, spec in enumerate(agents):
        train_path = path / f'train_worker{worker}.jsonl'
        train = read_jsonl(train_path, final is None, warnings)
        previous_episode = 0
        for row in train:
            if row.get('worker') != worker or row.get('algorithm') != spec['algorithm']:
                raise ValueError(f'{train_path}: worker/algorithm mismatch')
            if not finite(row.get('reward')) or row['episode'] <= previous_episode:
                raise ValueError(f'{train_path}: invalid raw reward or episode sequence')
            previous_episode = row['episode']
        natural = [row for row in train if not row.get('budget_cut', False)]
        cut = [row for row in train if row.get('budget_cut', False)]
        _, _, actual_window = moving_average([row['reward'] for row in natural], window)
        test = final_rows.get(worker)
        worker_final = None
        if test is not None:
            mean = evaluation_mean(test, str(final_path))
            expected_split = 'test' if stage == 'confirmation' else 'development'
            if test.get('split') != expected_split or test.get('seed_start') != meta['final_test_seed']:
                raise ValueError(f'{final_path}: final evaluation split/seed mismatch')
            if test['episodes'] != meta['final_test_episodes']:
                raise ValueError(f'{final_path}: final evaluation episode-count mismatch')
            if test.get('aggregate_steps') != identity['requested_total_steps']:
                raise ValueError(f'{final_path}: final checkpoint step mismatch')
            # The final record must occur exactly once in the append-only log.
            matches = [row for row in evaluations if row['worker'] == worker and row.get('split') == expected_split]
            if len(matches) != 1 or matches[0] != test:
                raise ValueError(f'{final_path}: final/log records disagree')
            worker_final = {'mean': mean, 'episodes': len(test['returns']),
                            'pass': None if threshold is None else mean >= threshold,
                            'seed_start': test['seed_start'], 'split': test['split'],
                            'statistics': counter_rows[worker].get('statistics', {}),
                            'steps': counter_rows[worker]['steps']}
        workers.append({'worker': worker, 'algorithm': spec['algorithm'], 'final': worker_final,
                        'natural_episodes': len(natural), 'budget_cut_episodes': len(cut),
                        'smoothing_window': actual_window, 'requested_smoothing_window': window,
                        'seed_configuration': (meta['worker_seeds'][worker] if meta.get('worker_seeds') else None),
                        'training_source': relative(train_path, output),
                        '_training': natural, '_budget_cut': cut,
                        '_validation': [row for row in evaluations if row['worker'] == worker and row['split'] == 'validation']})
    return {'run': path.name, 'backend': 'reinforcex', 'stage': stage, 'variant': variant_id(meta), 'identity': identity,
            'interpretation': 'as_recorded', 'confirmation_evidence_eligible': stage == 'confirmation',
            'achievement_evidence_eligible': True,
            'metadata_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
            'evaluation_protocol': protocol, 'packages': meta.get('packages', {}),
            'comparison_settings': comparison_settings(meta['configuration'], agents, meta.get('max_episode_steps')),
            'comparison_eligible': len(agents) == 1 and agents[0].get('rnd_config') is None,
            'seed': meta['seed'], 'env_id': meta['env_id'], 'threshold': threshold, 'status': status,
            'worker_count': len(agents), 'workers': workers, 'warnings': warnings,
            'metadata_source': relative(meta_path, output),
            'final_source': relative(final_path, output) if final else None,
            'evaluations_source': relative(path / 'evaluations.jsonl', output),
            'seconds': None if final is None else final.get('seconds'),
            'training_seconds': None if final is None else final.get('training_seconds'),
            'evaluation_seconds': None if final is None else final.get('evaluation_seconds'),
            # Reject ambiguous requested keys instead of merging hidden differences.
            '_contract': canonical({'agents': agents, 'environment': meta['env_id'],
                                    'concurrent_workers': meta.get('concurrent_workers'),
                                    'share_rnd': meta.get('share_rnd'),
                                    'shared_replay': meta['configuration'].get('shared_replay'),
                                    'reward_mode': meta['configuration']['reward_mode'],
                                    'validation_seed': meta['validation_seed'],
                                    'validation_episodes': meta['validation_episodes'],
                                    'final_test_seed': meta['final_test_seed'],
                                    'final_test_episodes': meta['final_test_episodes']})}


def reference_mapping_issues(args: dict, constructor: dict) -> list[str]:
    """Conservative eligibility check; this does not claim algorithmic equivalence."""
    option_names = ('learning_rate gamma batch_size buffer_size learning_starts train_freq gradient_steps tau '
                    'n_steps n_epochs gae_lambda clip_range vf_coef max_grad_norm target_kl target_update_interval '
                    'exploration_fraction exploration_initial_eps exploration_final_eps ent_coef target_entropy '
                    'activation net_arch initial_log_std').split()
    issues = []
    if args.get('hyperparams') or any(args.get(name) is not None for name in option_names):
        issues.append('Explicit SB3 CLI/hyperparameter overrides: settings match is not certified.')
    config = args['native_config']
    if len(config['agents']) != 1:
        return issues + ['Reference does not reproduce multiple workers/shared replay.']
    spec = config['agents'][0]
    if spec.get('rnd_config') is not None:
        issues.append('Reference does not implement RND.')
    cfg, algo = spec['config'], spec['algorithm']
    c = cfg['agent']
    expected = {'gamma': c['gamma']}
    if algo == 'ppo':
        mapping = {'learning_rate': 'learning_rate', 'n_steps': 'update_interval', 'batch_size': 'minibatch_size',
                   'n_epochs': 'epochs', 'gae_lambda': 'gae_lambda', 'clip_range': 'policy_clip_epsilon',
                   'ent_coef': 'entropy_coefficient', 'vf_coef': 'value_loss_coefficient'}
        expected.update({key: cfg[value] for key, value in mapping.items()})
        expected.update(clip_range_vf=cfg['value_clip_range'] or None,
                        normalize_advantage=bool(cfg['standardize_gae']), max_grad_norm=.5,
                        target_kl=cfg.get('target_kl') or None)
    elif algo == 'sac':
        expected.update(learning_rate=cfg['actor_learning_rate'], batch_size=cfg['batch_size'],
                        buffer_size=cfg['replay_capacity'], learning_starts=cfg['replay_start_size'],
                        n_steps=cfg['replay_n_steps'], train_freq=cfg['update_interval'], gradient_steps=1,
                        target_update_interval=1, target_entropy='auto', ent_coef=f"auto_{cfg['alpha']}",
                        tau=1 - (1 - cfg['tau']) ** (cfg['update_interval'] / cfg['target_update_interval']))
        if cfg['actor_learning_rate'] != cfg['critic_learning_rate']:
            issues.append('Unequal SAC actor/critic rates cannot match one reference rate.')
    else:
        issues.append('Automatic matched display currently certifies PPO/SAC mappings only.')
    schedule = spec.get('learning_rate_schedule')
    if schedule is not None:
        if (algo not in ('ppo', 'dqn') or not isinstance(schedule, dict)
                or set(schedule) != {'kind', 'final_fraction'} or schedule.get('kind') != 'linear'
                or not finite(schedule.get('final_fraction')) or not 0 < schedule['final_fraction'] <= 1
                or not finite(cfg.get('learning_rate')) or cfg['learning_rate'] <= 0):
            issues.append('Unsupported or invalid learning rate schedule mapping.')
        else:
            # Dictionary equality below is exact, including keys; no tolerance or
            # opaque callable representation can certify the schedule contract.
            expected['learning_rate'] = {'kind': 'linear', 'initial': cfg['learning_rate'],
                                         'final_fraction': schedule['final_fraction']}
    for key, value in expected.items():
        actual = constructor.get(key)
        if not (math.isclose(value, actual, rel_tol=1e-12, abs_tol=1e-14)
                if finite(value) and finite(actual) else value == actual):
            issues.append(f'Resolved constructor differs: {key}.')
    policy = constructor.get('policy_kwargs', {})
    activation = 'Tanh' if algo == 'ppo' and cfg.get('model') == 1 and cfg.get('activation', 0) == 0 else 'ReLU'
    if policy.get('net_arch') != [c['hidden_size']] * (c['hidden_layers'] + 1) or (
            policy.get('activation_fn') != f"<class 'torch.nn.modules.activation.{activation}'>"):
        issues.append('Resolved policy architecture differs from mapped settings.')
    if algo == 'ppo':
        if policy.get('optimizer_kwargs', {}).get('eps') != cfg.get('adam_epsilon', 1e-8):
            issues.append('Resolved PPO Adam epsilon differs.')
        if cfg.get('model') == 1 and (policy.get('ortho_init') is not True or (
                cfg['action_space'] == 1 and policy.get('log_std_init') != cfg.get('initial_log_std', 0.))):
            issues.append('Resolved separate PPO initialization differs.')
    transform = {'raw': 'raw', 'cartpole': 'cartpole', 'scale0.1': 'scale', 'hopper': 'hopper', 'ant_shared': 'ant'}
    if args.get('reward_transform') != transform.get(config['reward_mode']) or (
            config['reward_mode'] == 'scale0.1' and args.get('reward_scale') != .1):
        issues.append('Reward preprocessing differs from native settings.')
    return issues


def load_reference_run(path: Path, output: Path, window: int) -> dict:
    meta_path = path / 'metadata.json'
    meta, config = read_json(meta_path), read_json(path / 'config.json')
    args, constructor = config['arguments'], config['resolved_constructor']
    stage, seed, case = meta['study_stage'], args['seed'], meta['case']
    if args['case'] != case or args['stage'] != stage or meta.get('n_envs') != 1:
        raise ValueError(f'{path}: inconsistent reference identity or environment count')
    if constructor.get('seed') != seed or config['environment_spec']['id'] != args['env']:
        raise ValueError(f'{path}: reference constructor seed/environment mismatch')
    native = args['native_config']
    agents = native['agents']
    if len(agents) != 1 or agents[0]['algorithm'] != args['algo'] or native['env_id'] != args['env']:
        raise ValueError(f'{path}: reference native configuration mismatch')
    protocol = evaluation_protocol(stage, args['validation_seed'], args['progress_eval_episodes'],
                                   args['final_eval_seed'], args['eval_episodes'])
    if meta['validation_seed'] != protocol['validation_seed'] or meta['final_eval_seed'] != protocol['final_seed']:
        raise ValueError(f'{path}: metadata evaluation protocol mismatch')
    resolved = {key: value for key, value in constructor.items() if key not in ('seed', 'verbose')}
    identity = {'backend': 'stable-baselines3', 'runner_sha256': meta['script_sha256'],
                'packages': meta['packages'], 'native_config': native, 'resolved_constructor': resolved,
                'normalization_source_sha256': meta.get('normalization_source_sha256'),
                'case': case, 'requested_total_steps': args['steps']}
    variant = hashlib.sha256(canonical(identity).encode()).hexdigest()[:16]
    final_path = path / 'final.json'
    final = read_json(final_path) if final_path.exists() else None
    warnings: list[str] = []
    summaries = read_jsonl(path / 'evaluations.jsonl', final is None, warnings)
    episode_rows = read_jsonl(path / 'eval_episodes.jsonl', final is None, warnings)
    by_point = defaultdict(list)
    for row in episode_rows:
        by_point[row['point']].append(row)
    evaluations, point_ids = [], set()
    for row in summaries:
        if row['point'] in point_ids:
            raise ValueError(f'{path}: duplicate reference evaluation point')
        point_ids.add(row['point'])
        episodes = sorted(by_point[row['point']], key=lambda item: item['episode'])
        if len(episodes) != row['episodes'] or [x['episode'] for x in episodes] != list(range(1, row['episodes'] + 1)):
            raise ValueError(f'{path}: reference evaluation episode count/sequence mismatch')
        for index, episode in enumerate(episodes):
            if episode['seed'] != row['seed_start'] + index or any(episode[key] != row[key]
                    for key in ('phase', 'split', 'study_stage', 'env_steps', 'requested_step')):
                raise ValueError(f'{path}: reference episode/summary protocol mismatch')
        expected_final = row['phase'] == 'final'
        if row['study_stage'] != stage or row['deterministic'] is not True or (
                row['split'] != (protocol['final_split'] if expected_final else 'validation')) or (
                row['seed_start'] != protocol['final_seed' if expected_final else 'validation_seed']) or (
                row['episodes'] != protocol['final_episodes' if expected_final else 'validation_episodes']):
            raise ValueError(f'{path}: reference deterministic evaluation protocol mismatch')
        converted = {**row, 'worker': 0, 'algorithm': args['algo'], 'aggregate_steps': row['env_steps'],
                     'returns': [x['return'] for x in episodes], 'reward': 'raw', 'mean': row['mean_return']}
        evaluation_mean(converted, str(path))
        evaluations.append(converted)
    if final is not None and set(by_point) != point_ids:
        raise ValueError(f'{path}: completed reference contains unpaired evaluation episodes')
    status = 'failed' if meta.get('status') == 'failed' else 'running_or_interrupted'
    worker_final = None
    threshold = args['success_threshold']
    if threshold != THRESHOLDS.get(args['env']):
        raise ValueError(f'{path}: reference threshold mismatch')
    if final is not None:
        for key, expected in {'status': 'complete', 'case': case, 'study_stage': stage,
                              'engine': 'stable-baselines3', 'algorithm': args['algo'], 'environment': args['env'],
                              'seed': seed, 'requested_steps': args['steps']}.items():
            if final.get(key) != expected:
                raise ValueError(f'{path}: inconsistent reference final {key}')
        matches = [row for row in summaries if row['phase'] == 'final']
        if len(matches) != 1 or matches[0] != final['final_evaluation'] or matches[0]['env_steps'] != final['actual_steps']:
            raise ValueError(f'{path}: reference final/log records disagree')
        if final['actual_steps'] < args['steps']:
            raise ValueError(f'{path}: reference actual budget below request')
        test = next(row for row in evaluations if row['phase'] == 'final')
        mean = evaluation_mean(test, str(path))
        worker_final = {'mean': mean, 'episodes': test['episodes'], 'pass': mean >= threshold,
                        'seed_start': test['seed_start'], 'split': test['split'], 'steps': final['actual_steps'],
                        'statistics': {'sb3_updates': final['updates']}}
        status = 'complete' if test['episodes'] == FINAL_EPISODES else 'complete_nonstandard_evaluation'
        if meta.get('status') == 'failed':
            status = 'failed'
    train_path = path / 'train_episodes.jsonl'
    train = read_jsonl(train_path, final is None, warnings)
    previous = 0
    for row in train:
        if not finite(row.get('return')) or row['episode'] <= previous:
            raise ValueError(f'{path}: invalid reference training return/episode sequence')
        previous = row['episode']
    training = [{**row, 'reward': row['return']} for row in train]
    issues = reference_mapping_issues(args, constructor)
    actual_steps = None if final is None else final['actual_steps']
    if actual_steps is not None and actual_steps != args['steps']:
        issues.append('Actual reference step budget exceeds native requested budget; no matched final comparison.')
    settings = comparison_settings(native, agents, config['environment_spec']['max_episode_steps'])
    return {'run': path.name, 'backend': 'stable-baselines3', 'stage': stage, 'variant': variant, 'identity': identity,
            'seed': seed, 'env_id': args['env'], 'threshold': threshold, 'status': status, 'worker_count': 1,
            'interpretation': 'as_recorded', 'confirmation_evidence_eligible': stage == 'confirmation',
            'achievement_evidence_eligible': True,
            'metadata_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(), 'evaluation_protocol': protocol,
            'packages': meta['packages'], 'comparison_settings': settings, 'comparison_eligible': not issues,
            'comparison_exclusions': issues, 'comparison_notes': meta['comparison_notes'],
            'config_source': relative(path / 'config.json', output),
            'eval_episodes_source': relative(path / 'eval_episodes.jsonl', output),
            'metadata_source': relative(meta_path, output), 'final_source': relative(final_path, output) if final else None,
            'evaluations_source': relative(path / 'evaluations.jsonl', output), 'warnings': warnings,
            'workers': [{'worker': 0, 'algorithm': args['algo'], 'final': worker_final,
                         'natural_episodes': len(train), 'budget_cut_episodes': 0,
                         'smoothing_window': min(window, len(train)), 'requested_smoothing_window': window,
                         'seed_configuration': {'training_seed': seed}, 'training_source': relative(train_path, output),
                         '_training': training, '_budget_cut': [],
                         '_validation': [row for row in evaluations if row['split'] == 'validation']}],
            'seconds': None if final is None else final.get('total_seconds'),
            '_contract': canonical({'settings': settings, 'evaluation': protocol, 'constructor': resolved})}


def group_key(group: dict) -> str:
    return f"{group['stage']}-{group['variant']}-{group['cohort']}-{group['algorithm']}"


def aggregate(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    contracts = {}
    seen = set()
    for run in runs:
        cohort = hashlib.sha256(canonical({'evaluation': run['evaluation_protocol'],
                                          'interpretation': run['interpretation']}).encode()).hexdigest()[:8]
        base = (run['stage'], run['variant'], cohort)
        seed_key = (*base, run['seed'])
        if seed_key in seen:
            raise ValueError(f'duplicate seed within stage/variant: {seed_key}; separate reruns explicitly')
        seen.add(seed_key)
        if base in contracts and contracts[base] != run['_contract']:
            raise ValueError(f'variant {base} has conflicting effective configurations/evaluation contracts')
        contracts[base] = run['_contract']
        for algorithm in sorted({worker['algorithm'] for worker in run['workers']}):
            grouped[(*base, algorithm)].append(run)
    result = []
    for (stage, variant, cohort, algorithm), members in sorted(grouped.items()):
        first = members[0]
        complete = [run for run in members if run['status'] == 'complete']
        per_seed = []
        model_count = passed_models = 0
        for run in sorted(complete, key=lambda item: item['seed']):
            workers = [worker for worker in run['workers'] if worker['algorithm'] == algorithm]
            means = [worker['final']['mean'] for worker in workers]
            model_count += len(workers)
            passed_models += sum(worker['final']['pass'] is True for worker in workers)
            seed_mean = statistics.fmean(means)
            per_seed.append({'seed': run['seed'], 'mean': seed_mean, 'worker_count': len(workers),
                             'all_workers_pass': None if run['threshold'] is None else all(worker['final']['pass'] for worker in workers),
                             'seed_mean_pass': None if run['threshold'] is None else seed_mean >= run['threshold'],
                             'worker_results': [{'worker': worker['worker'], **worker['final']} for worker in workers]})
        means = [seed['mean'] for seed in per_seed]
        expected_models = sum(sum(worker['algorithm'] == algorithm for worker in run['workers']) for run in members)
        result.append({'stage': stage, 'variant': variant, 'cohort': cohort, 'algorithm': algorithm,
                       'backend': first['backend'], 'interpretation': first['interpretation'],
                       'confirmation_evidence_eligible': all(run['confirmation_evidence_eligible'] for run in members),
                       'achievement_evidence_eligible': all(run['achievement_evidence_eligible'] for run in members),
                       'evaluation_protocol': first['evaluation_protocol'],
                       'study_role': ('reference' if first['backend'] != 'reinforcex' else 'target' if first['identity']['case'] in TARGET_CASES
                                      and (first['identity']['case'] != 'halfcheetah_hybrid' or algorithm == 'ppo')
                                      else 'supplementary'),
                       'identity': first['identity'], 'env_id': first['env_id'], 'threshold': first['threshold'],
                       'registered_seeds': sorted(run['seed'] for run in members),
                       'completed_seeds': len(complete), 'incomplete_seeds': len(members) - len(complete),
                       'expected_models_from_metadata': expected_models, 'evaluated_models': model_count,
                       'passed_models': None if first['threshold'] is None else passed_models,
                       'all_workers_passed_seeds': None if first['threshold'] is None else sum(seed['all_workers_pass'] for seed in per_seed),
                       'mean_passed_seeds': None if first['threshold'] is None else sum(seed['seed_mean_pass'] for seed in per_seed),
                       'mean': statistics.fmean(means) if means else None,
                       'seed_sd': statistics.stdev(means) if len(means) >= 2 else None,
                       'seeds': per_seed, 'runs': [run['run'] for run in members], 'plots': {}, '_members': members})
    return result


def render_group(group: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    workers = sorted({worker['worker'] for run in group['_members'] for worker in run['workers'] if worker['algorithm'] == group['algorithm']})
    seeds = group['registered_seeds']
    colors = {seed: plt.get_cmap('tab10')(index % 10) for index, seed in enumerate(seeds)}
    key = group_key(group)
    for kind in ('training', 'evaluation'):
        figure, axes = plt.subplots(len(workers), 1, figsize=(10.4, 3.8 * len(workers)), squeeze=False)
        for axis, worker_index in zip(axes[:, 0], workers):
            for run in sorted(group['_members'], key=lambda item: item['seed']):
                worker = next(worker for worker in run['workers'] if worker['worker'] == worker_index)
                color = colors[run['seed']]
                label = f"seed {run['seed']}" + ('' if run['status'] == 'complete' else f" ({run['status']})")
                if kind == 'training':
                    rows = worker['_training']
                    x, y = [row['episode'] for row in rows], [row['reward'] for row in rows]
                    axis.plot(x, y, color=color, alpha=.14, linewidth=.65)
                    indices, average, window = moving_average(y, worker['requested_smoothing_window'])
                    axis.plot([x[index] for index in indices], average, color=color, linewidth=1.7,
                              marker='.' if len(indices) == 1 else None, label=f'{label}; MA window {window}')
                    for row in worker['_budget_cut']:
                        axis.scatter([row['episode']], [row['reward']], marker='x', color=color, s=26, zorder=4)
                    axis.set_xlabel('Completed training episode (per worker); x = budget-cut fragment')
                    axis.set_ylabel('Raw environment episode return')
                else:
                    rows = sorted(worker['_validation'], key=lambda row: row['aggregate_steps'])
                    axis.plot([row['aggregate_steps'] for row in rows], [evaluation_mean(row, run['run']) for row in rows],
                              marker='o', markersize=3, color=color, linewidth=1.5, label=label)
                    if worker['final'] is not None:
                        axis.scatter([worker['final']['steps']], [worker['final']['mean']],
                                     color=color, marker='D', s=44, edgecolors='black', linewidths=.7, zorder=5)
                    axis.set_xlabel('Aggregate environment steps (all workers combined)')
                    axis.set_ylabel('Mean raw deterministic evaluation return')
                if not worker['_training'] and kind == 'training':
                    axis.text(.03, .08, 'No completed training episodes yet', transform=axis.transAxes)
            if group['threshold'] is not None:
                axis.axhline(group['threshold'], color='#666666', linestyle='--', linewidth=.8, label=f"threshold {group['threshold']:g}")
            axis.set_title(f"Worker {worker_index} | {group['algorithm'].upper()}", fontsize=10, loc='left')
            axis.grid(alpha=.18)
            axis.legend(fontsize=8, loc='best')
        figure.suptitle(f"{group['identity']['case']} | {group['backend']} | {group['stage']} | {group['interpretation']}\n"
                       f"variant {group['variant']} | total budget {group['identity']['requested_total_steps']:,}", fontsize=11)
        note = ('Thin lines: individual raw returns. Thick lines: full trailing-window mean; short runs use the stated smaller window.\n'
                'Budget-cut fragments (x) are excluded from the mean. Training returns are exploratory, not the final pass criterion.'
                if kind == 'training' else
                'Circles: validation means at observed checkpoints only. Diamonds: last-checkpoint final evaluation.\n'
                'No interpolation or best-checkpoint selection. Validation and final evaluation may use different episode counts/seeds.')
        figure.text(.02, .012, note, fontsize=8, va='bottom')
        figure.tight_layout(rect=(0, .07 / len(workers), 1, .93 if len(workers) == 1 else .96))
        for suffix in ('png', 'pdf'):
            destination = output / 'figures' / f'{key}_{kind}.{suffix}'
            figure.savefig(destination, dpi=155)
            group['plots'][f'{kind}_{suffix}'] = relative(destination, output)
        plt.close(figure)


def matched_comparisons(runs: list[dict]) -> list[dict]:
    """Display paired individual runs, never pool native and external estimates."""
    result = []
    for native in runs:
        if native['backend'] != 'reinforcex' or not native['comparison_eligible']:
            continue
        for reference in runs:
            if reference['backend'] != 'stable-baselines3' or not reference['comparison_eligible']:
                continue
            fields = ('stage', 'seed', 'env_id', 'threshold', 'comparison_settings', 'evaluation_protocol', 'interpretation')
            if native['interpretation'] != 'as_recorded' or any(native[key] != reference[key] for key in fields):
                continue
            if any(native['identity'][key] != reference['identity'][key] for key in ('case', 'requested_total_steps')):
                continue
            # An explicit environment/policy seed override is not the base training seed.
            seeds = native['workers'][0]['seed_configuration'] or {}
            if any(seeds.get(key, native['seed']) != native['seed'] for key in ('environment_seed', 'policy_seed')):
                continue
            finals = [run['workers'][0]['final'] for run in (native, reference)]
            both_final = all(run['status'] == 'complete' for run in (native, reference))
            key = hashlib.sha256(canonical([native['run'], reference['run']]).encode()).hexdigest()[:16]
            result.append({'key': key, 'case': native['identity']['case'], 'seed': native['seed'],
                           'stage': native['stage'], 'steps': native['identity']['requested_total_steps'],
                           'native_run': native['run'], 'reference_run': reference['run'],
                           'native_variant': native['variant'], 'reference_variant': reference['variant'],
                           'status': 'complete' if both_final else 'pending',
                           'native_mean': None if not both_final else finals[0]['mean'],
                           'reference_mean': None if not both_final else finals[1]['mean'],
                           'native_minus_reference': None if not both_final else finals[0]['mean'] - finals[1]['mean'],
                           'plots': {}, '_members': [native, reference]})
    return result


def render_comparison(pair: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in ('training', 'evaluation'):
        figure, axis = plt.subplots(figsize=(10.4, 4.8))
        for run, color in zip(pair['_members'], ('#2369a6', '#d76b21')):
            worker = run['workers'][0]
            label = f"{run['backend']} | seed {run['seed']} | {run['status']}"
            if kind == 'training':
                rows = worker['_training']
                x, y = [row['episode'] for row in rows], [row['reward'] for row in rows]
                axis.plot(x, y, color=color, alpha=.13, linewidth=.65)
                indices, average, window = moving_average(y, worker['requested_smoothing_window'])
                axis.plot([x[i] for i in indices], average, color=color, label=f'{label}; MA {window}', linewidth=1.7)
                axis.set_xlabel('Completed training episode (single worker)')
                axis.set_ylabel('Raw environment episode return')
            else:
                rows = sorted(worker['_validation'], key=lambda row: row['aggregate_steps'])
                axis.plot([row['aggregate_steps'] for row in rows], [evaluation_mean(row, run['run']) for row in rows],
                          marker='o', markersize=3, linewidth=1.5, color=color, label=label)
                if worker['final'] is not None:
                    axis.scatter([worker['final']['steps']], [worker['final']['mean']], marker='D', color=color,
                                 s=45, edgecolors='black', linewidths=.7, zorder=4)
                axis.set_xlabel('Environment steps (one worker in each implementation)')
                axis.set_ylabel('Mean raw deterministic evaluation return')
        axis.grid(alpha=.18)
        axis.legend(fontsize=8, loc='best')
        axis.set_title(f"{pair['case']} | matched nominal settings | {pair['steps']:,} steps | {pair['stage']}", fontsize=11)
        note = ('Raw exploratory training return; trailing episode means. No pooling between implementations.' if kind == 'training'
                else 'Circles: observed validation. Diamonds: last-model final. Collection/update timing differs; see report notes.')
        figure.text(.02, .02, note + '\nSame seed numbers do not imply identical random streams. This single-seed pair is not a confidence interval.', fontsize=8)
        figure.tight_layout(rect=(0, .095, 1, 1))
        for suffix in ('png', 'pdf'):
            target = output / 'figures' / f"comparison_{pair['key']}_{kind}.{suffix}"
            figure.savefig(target, dpi=155)
            pair['plots'][f'{kind}_{suffix}'] = relative(target, output)
        plt.close(figure)


def stripped(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: stripped(item) for key, item in value.items() if not key.startswith('_')}
    if isinstance(value, list):
        return [stripped(item) for item in value]
    return value


def fmt(value: Any) -> str:
    return '—' if value is None else f'{value:,.2f}'


def write_report(runs: list[dict], groups: list[dict], output: Path, window: int,
                 comparisons: list[dict] | None = None, decisions: dict | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    payload = {'generated_at': now, 'final_evaluation_episodes_required': FINAL_EPISODES,
               'requested_smoothing_window': window, 'runs': stripped(runs), 'groups': stripped(groups),
               'comparisons': stripped(comparisons or []), 'decisions': decisions}
    (output / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# コア改善実験：自動集計', '', f'生成日時: {now}', '',
             '開発用（development）と最終確認用（confirmation）は別集計。variant は build SHA256・overrides の canonical JSON・case・合計 step 予算・LR schedule・worker数/並行実行/RND共有の構成で識別する。異なる有効設定や評価条件が同じ識別子になる場合は結合せずエラーにする。', '',
             '最終値は最後の checkpoint の raw return 100 episode から再計算。worker 平均を同じ training seed 内で等重みにし、seed 平均を seed 間で等重みにする。± は training seed 間の標本 SD（信頼区間ではない）。評価 episode や worker を独立 seed として数えない。', '',
             'model 達成数は各 worker の最終平均で判定。「全 worker 達成 seed」と「seed 平均達成」は別指標であり、HalfCheetah PPO では前者に両 PPO の成功が必要。開発段階の達成は最終確認の合格証拠ではない。', '',
             '進行中・失敗・100 episode 未満/超の非標準評価は最終集計から除外する。分母の登録数は metadata が存在する run のみで、未開始 job の予定数ではない。性能や速度の因果はこの集計だけでは断定できない。', '',
             'HalfCheetah PPO 単独診断などの補助 case と hybrid 内 SAC の参考成績は、改善対象の 4 case/algorithm と別表にする。単独診断の成功は 2 PPO + 2 SAC hybrid の成功証拠に数えない。', '',
             '外部対照は backend・runner SHA・package 版・resolved constructor ごとの別 group。評価 seed block と判断区分も cohort を分ける。元 metadata の段階は上書きしない。redevelopment は確認後に再開発/設定選定へ再利用された履歴、provenance_discrepancy は出典整合性に問題がある履歴で、いずれも確認合格・達成の根拠から除外する。数値と探索的学習曲線は残す。']
    if decisions:
        lines.extend(['', f"判断記録: [decision JSON]({relative(Path(decisions['source']), output)})（SHA `{decisions['sha256']}`）。", '',
                      '| run | 判断 | 理由 |', '|---|---|---|'])
        for decision in decisions['decisions']:
            lines.append(f"| {decision['run']} | {decision['interpretation']} | {decision['reason'].replace('|', '/')} |")
    for role, title in [('target', '対象 4 case'), ('supplementary', '補助診断（対象 case の合否とは別）'),
                        ('reference', '外部対照（native と平均を混合しない）')]:
        selected = [group for group in groups if group['study_role'] == role]
        if not selected:
            continue
        lines.extend(['', f'## {title}', '',
                      '| 履歴段階 / 判断 | case / algorithm | variant | 最終 seed / 登録 | 最終平均 ± seed SD | model 閾値以上 / 評価済（登録） | 全 worker 閾値以上 seed / 完了 | seed 平均閾値以上 / 完了 |',
                      '|---|---|---|---:|---:|---:|---:|---:|'])
        for group in selected:
            number = group['completed_seeds']
            lines.append(f"| {group['stage']} / {group['interpretation']} | {group['identity']['case']} / {group['algorithm']} | [{group['variant']}](#{group_key(group)}) | {number}/{len(group['registered_seeds'])} | {fmt(group['mean'])} ± {fmt(group['seed_sd'])} | {group['passed_models']}/{group['evaluated_models']} ({group['expected_models_from_metadata']}) | {group['all_workers_passed_seeds']}/{number} | {group['mean_passed_seeds']}/{number} |")
    lines.extend(['', '## 設定・seed が一致する対照', '',
                  'case、環境、報酬加工、全 agent 設定・正規化、予算、時間制限、training seed、段階、決定的評価の split/seed/count が一致する単独 worker のみ。同じ設定へマッピングされた実装の対照であり、同一アルゴリズム実装の証明ではない。異なる package・RNG・更新順序・損失関数等の残差は下記の元 notes を参照する。差は1本ずつの最終100episode平均の差であり、seed 間信頼区間ではない。', '',
                  '| case / seed | native variant | SB3 variant | 状態 | native | SB3 | native − SB3 |',
                  '|---|---|---|---|---:|---:|---:|'])
    for pair in comparisons or []:
        lines.append(f"| {pair['case']} / {pair['seed']} | {pair['native_variant']} | {pair['reference_variant']} | {pair['status']} | {fmt(pair['native_mean'])} | {fmt(pair['reference_mean'])} | {fmt(pair['native_minus_reference'])} |")
        for kind in ('training', 'evaluation'):
            if f'{kind}_png' in pair['plots']:
                lines.extend(['', f"![matched {pair['case']} {kind}]({pair['plots'][kind + '_png']})", '',
                              f"[{kind} PDF]({pair['plots'][kind + '_pdf']})"])
    for group in groups:
        lines.extend(['', f"## {group_key(group)}", '',
                      f"`{group['identity']['case']}`、合計 {group['identity']['requested_total_steps']:,} step、閾値 {fmt(group['threshold'])}。", '',
                      f"backend: `{group['backend']}`、判断: `{group['interpretation']}`。", '',
                      'variant / 評価条件:', '', '```json', json.dumps({'identity': group['identity'], 'evaluation': group['evaluation_protocol']}, ensure_ascii=False, sort_keys=True, indent=2), '```', '',
                      '| training seed | worker | 最終 raw 平均（episode 数） | 閾値達成 | worker step | 状態 | 元ログ |',
                      '|---:|---:|---:|---|---:|---|---|'])
        for run in sorted(group['_members'], key=lambda item: item['seed']):
            for worker in run['workers']:
                if worker['algorithm'] != group['algorithm']:
                    continue
                final = worker['final']
                result = 'pending' if final is None else ('yes' if final['pass'] else 'no')
                if final is not None and run['status'] != 'complete':
                    result += ' (非標準評価・集計除外)'
                mean = '—' if final is None else f"{fmt(final['mean'])} ({final['episodes']})"
                sources = f"[metadata]({run['metadata_source']}) / [train]({worker['training_source']}) / [eval]({run['evaluations_source']})"
                if run['final_source']:
                    sources += f" / [final]({run['final_source']})"
                lines.append(f"| {run['seed']} | {worker['worker']} | {mean} | {result} | {'—' if final is None else final['steps']} | {run['status']} | {sources} |")
        if group['backend'] == 'stable-baselines3':
            for run in group['_members']:
                lines.extend(['', f"参照設定: [config]({run['config_source']}) / [各評価 episode]({run['eval_episodes_source']})。", '',
                              f"packages: `{canonical(run['packages'])}`。", '',
                              '元 metadata の比較制約（全件）:', '', *[f'- {note}' for note in run['comparison_notes']]])
                if run['comparison_exclusions']:
                    lines.extend(['', '自動対照から除外: ' + '; '.join(run['comparison_exclusions'])])
                elif not any(pair['reference_run'] == run['run'] for pair in comparisons or []):
                    lines.extend(['', 'case/settings/step/seed/evaluation が厳密に一致する native run は現時点でないため別表示。'])
        for kind, title in [('training', 'episode 対 raw reward'), ('evaluation', '合計 step 対評価 return')]:
            if f'{kind}_png' in group['plots']:
                lines.extend(['', f"![{title}]({group['plots'][f'{kind}_png']})", '', f"[{title} PDF]({group['plots'][f'{kind}_pdf']})"])
    lines.extend(['', '## 読み方と制約', '',
                  f'学習図の薄線は episode ごとの生値、太線は末尾 {window} episode の移動平均。完了 episode が不足する run はその数を窓にし、凡例に実窓長を明記する。先頭の不足区間に平均値を補完しない。予算で途中終了した断片は × 印とし平均から除外する。', '',
                  '評価図は観測済み checkpoint のみを結んだ線で、欠測を生成しない。丸は validation、菱形は最終評価。通常 validation は 10 episode、最終は 100 episode だが、正確な数と seed は元ログに保存される。最高値の checkpoint や最良 worker を選ばない。', '',
                  '並列条件の横軸は全 worker 合計 step。worker 当たりの経験量や更新数は summary.json の worker_results と元ログを参照する。並列実行順序と Rust thread_rng は完全固定でなく、同 seed のみで同一軌跡や bit 単位の再現を主張できない。', '',
                  '再生成: `python benchmarks/render_improvement_report.py`。`--no-plots` は標準ライブラリのみ、描画には Matplotlib が必要。入力を読み取るだけで学習・推論・checkpoint load は行わない。', ''])
    warnings = [warning for run in runs for warning in run['warnings']]
    if warnings:
        lines.extend(['## 読み取り時の注意', '', *[f'- {warning}' for warning in warnings], ''])
    (output / 'REPORT.ja.md').write_text('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=DEFAULT_STUDY / 'runs')
    parser.add_argument('--reference-runs', type=Path, help='Default: sibling reference_runs directory')
    parser.add_argument('--decisions', type=Path, help='Default: sibling report_decisions.json when present')
    parser.add_argument('--output', type=Path, default=DEFAULT_STUDY / 'rendered')
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    output = args.output.resolve()
    reference_root = args.reference_runs or args.runs.parent / 'reference_runs'
    decisions_path = args.decisions or args.runs.parent / 'report_decisions.json'
    if any(output == root.resolve() or output.is_relative_to(root.resolve()) for root in (args.runs, reference_root)):
        parser.error('--output must not be inside the input runs directory')
    output.mkdir(parents=True, exist_ok=True)
    runs = [load_run(path.parent, output, args.window) for path in sorted(args.runs.glob('*/metadata.json'))]
    runs += [load_reference_run(path.parent, output, args.window) for path in sorted(reference_root.glob('*/metadata.json'))]
    if not runs:
        parser.error('no metadata.json runs found')
    decisions = apply_decisions(runs, decisions_path if decisions_path.exists() else None)
    if args.decisions and not decisions_path.exists():
        parser.error('--decisions does not exist')
    groups = aggregate(runs)
    comparisons = matched_comparisons(runs)
    if not args.no_plots:
        os.environ.setdefault('MPLCONFIGDIR', str(output / '.mplconfig'))
        (output / 'figures').mkdir(exist_ok=True)
        for group in groups:
            render_group(group, output)
        for pair in comparisons:
            render_comparison(pair, output)
    write_report(runs, groups, output, args.window, comparisons, decisions)
    print(json.dumps({'runs': len(runs), 'groups': len(groups), 'complete_runs': sum(run['status'] == 'complete' for run in runs),
                      'matched_comparisons': len(comparisons),
                      'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
