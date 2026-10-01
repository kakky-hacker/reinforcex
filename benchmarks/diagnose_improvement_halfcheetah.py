#!/usr/bin/env python3
"""No-update development-only HalfCheetah posture/action diagnostics.

Runs ten deterministic episodes from seed 1100000 for each selected checkpoint.
Loads the exact native library/configuration and normalization moments from its
metadata. Never calls training/save, and verifies all input hashes and statistics.
No torch Python import, rendering, or confirmation seeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ctypes as C
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'reports/core_improvements_20261001'
DEFAULTS = [
    ('raw_standard', 'tanh_standard'),
    ('observations_only', 'tanh_standard_normalize_observations'),
    ('observations_and_rewards', 'tanh_standard_normalize_both'),
]
EPISODES = 10
SEED_START = 1100000


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprints(paths):
    return {str(path.resolve()): {'sha256': sha256(path), 'bytes': path.stat().st_size,
                                'mtime_ns': path.stat().st_mtime_ns}
            for path in sorted(set(paths))}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def summary(values, np):
    values = np.asarray(values, dtype=np.float64)
    require(values.size > 0 and np.isfinite(values).all(), 'empty or nonfinite metric')
    return {'n': int(values.size), 'mean': float(values.mean()), 'sd_population': float(values.std()),
            'min': float(values.min()), 'q05': float(np.quantile(values, .05)),
            'median': float(np.median(values)), 'q95': float(np.quantile(values, .95)), 'max': float(values.max())}


def diagnose(label, run, output, libraries, np, gym, rx, as_dict, fill, NormalizedAgent):
    metadata = read_json(run / 'metadata.json')
    final = read_json(run / 'final.json')
    require(metadata['backend'] == 'reinforcex' and metadata['env_id'] == 'HalfCheetah-v5', 'expected native HalfCheetah-v5')
    require(metadata['study_stage'] == final['study_stage'] == 'development', 'confirmation runs are forbidden')
    require(final['status'] == 'complete' and final['actual_total_steps'] == metadata['requested_total_steps'], 'run is incomplete')
    require(metadata['worker_count'] == 1 and len(metadata['effective_agents']) == 1, 'diagnostic expects the standalone PPO case')
    require(metadata['final_test_seed'] == SEED_START and metadata['final_test_episodes'] >= EPISODES, 'development seed reference mismatch')
    for package in ('gymnasium', 'mujoco', 'numpy'):
        require(importlib.metadata.version(package) == metadata['packages'][package], f'{package} differs from training runtime')
    spec = metadata['effective_agents'][0]
    require(spec['algorithm'] == 'ppo' and spec.get('rnd_config') is None, 'only PPO without RND is supported')
    library = Path(metadata['library']).resolve(strict=True)
    weights = run / 'worker0.ot'
    normalization = run / 'worker0.normalization.json'
    sources = [Path(__file__).resolve(), ROOT / 'examples/reinforcex_ffi.py',
               ROOT / 'examples/reinforcex_normalization.py', ROOT / 'benchmarks/improvement_configs.py',
               run / 'metadata.json', run / 'final.json', library, weights]
    if spec.get('normalization') is not None:
        sources.append(normalization)
    before_hashes = fingerprints(sources)
    require(before_hashes[str(library)]['sha256'] == metadata['library_sha256'], 'library SHA mismatch')
    if str(library) not in libraries:
        libraries[str(library)] = C.CDLL(str(library))
        rx.configure_ffi(libraries[str(library)])
    lib = libraries[str(library)]
    require(not rx.cuda_is_available(lib), 'CPU-only diagnostic required')
    config = rx.RxPpoConfigV2() if 'model' in spec['config'] else rx.RxPpoConfig()
    require(set(as_dict(config)) == set(spec['config']), 'missing/extra PPO configuration fields')
    fill(config, spec['config'])
    require(as_dict(config) == spec['config'], 'PPO configuration reconstruction mismatch')
    native = rx.create_ppo(lib, config, None, str(weights))
    agent = native
    env = None
    started = time.monotonic()
    forbidden_calls = {'training': 0, 'save': 0}

    def training_forbidden(*args, **kwargs):
        forbidden_calls['training'] += 1
        raise RuntimeError('training is forbidden in this diagnostic')

    def save_forbidden(*args, **kwargs):
        forbidden_calls['save'] += 1
        raise RuntimeError('checkpoint saving is forbidden in this diagnostic')

    try:
        if spec.get('normalization') is not None:
            agent = NormalizedAgent(native, config.agent.obs_size, config.agent.gamma,
                                    load_path=normalization, **spec['normalization'])
        for obj in {id(native): native, id(agent): agent}.values():
            for name in ('act_and_train', 'act_and_train_with_replay_input', 'stop_episode', 'stop_episode_with_replay_input'):
                setattr(obj, name, training_forbidden)
            obj.save = save_forbidden
        statistics_before = agent.statistics()
        require(statistics_before['updates'] == statistics_before.get('optimizer_steps') == 0, 'nonzero fresh update counter')
        normalization_before = agent.state_dict() if hasattr(agent, 'state_dict') else None
        env = gym.make(metadata['env_id'])
        require(env.spec.max_episode_steps == metadata['max_episode_steps'], 'environment time limit mismatch')
        require(env.unwrapped.observation_structure['skipped_qpos'] == 1, 'unexpected observation coordinates')
        torso = env.unwrapped.model.body('torso').id
        torso_origin_z = float(env.unwrapped.model.body_pos[torso, 2])
        samples = {name: [] for name in ('pitch_radians', 'wrapped_pitch_radians', 'torso_origin_world_height',
                    'rootz_coordinate', 'x_velocity', 'reward_forward', 'reward_ctrl', 'action_rms')}
        episode_rows = []
        boundary_counts = np.zeros(env.action_space.shape, dtype=np.int64)
        near_boundary_counts = np.zeros(env.action_space.shape, dtype=np.int64)
        inverted = 0
        steps = 0
        trace_path = output / f'{label}.steps.jsonl.gz'
        with gzip.open(trace_path, 'wt') as trace:
            for episode in range(EPISODES):
                seed = SEED_START + episode
                obs, _ = env.reset(seed=seed)
                total = forward = control = 0.
                begin = steps
                for length in range(1, env.spec.max_episode_steps + 1):
                    action = rx.gym_action(agent, agent.act(obs), env.action_space)
                    require(np.isfinite(action).all(), 'nonfinite action')
                    obs, reward, terminated, truncated, info = env.step(action)
                    require(np.isfinite(obs).all() and math.isfinite(float(reward)), 'nonfinite transition')
                    reward_forward, reward_ctrl = float(info['reward_forward']), float(info['reward_ctrl'])
                    require(math.isclose(float(reward), reward_forward + reward_ctrl, abs_tol=1e-10, rel_tol=1e-10), 'reward component mismatch')
                    pitch = float(env.unwrapped.data.qpos[2])
                    rootz = float(env.unwrapped.data.qpos[1])
                    wrapped = math.atan2(math.sin(pitch), math.cos(pitch))
                    # Root torso slides in z around model body_pos, with pitch
                    # hinge anchored at its origin. qpos avoids derived xpos lag.
                    height = torso_origin_z + rootz
                    boundary = (action == env.action_space.low) | (action == env.action_space.high)
                    near = ((action <= env.action_space.low + 1e-6) | (action >= env.action_space.high - 1e-6))
                    row = {'episode_seed': seed, 'step': length, 'pitch_radians': pitch,
                           'wrapped_pitch_radians': wrapped, 'torso_origin_world_height': height,
                           'rootz_coordinate': rootz, 'x_velocity': float(info['x_velocity']),
                           'reward_forward': reward_forward, 'reward_ctrl': reward_ctrl,
                           'action_rms': float(np.sqrt(np.mean(np.square(action))))}
                    for name in samples:
                        samples[name].append(row[name])
                    row['action'] = np.asarray(action).tolist()
                    row['boundary_components'] = int(boundary.sum())
                    trace.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                    boundary_counts += boundary
                    near_boundary_counts += near
                    inverted += int(math.cos(pitch) < 0)
                    steps += 1
                    total += float(reward)
                    forward += reward_forward
                    control += reward_ctrl
                    if terminated or truncated:
                        break
                reference = final['test'][0]
                require(reference['seed_start'] == SEED_START and reference['split'] == 'development', 'final reference protocol mismatch')
                difference = abs(total - reference['returns'][episode])
                require(length == reference['lengths'][episode] and math.isclose(total, reference['returns'][episode], rel_tol=1e-5, abs_tol=1e-4), 'reloaded development return differs from saved final')
                episode_rows.append({'seed': seed, 'length': length, 'raw_return': total,
                    'saved_final_return': reference['returns'][episode], 'absolute_return_difference': difference,
                    'reward_forward_sum': forward, 'reward_ctrl_sum': control,
                    'mean_torso_origin_world_height': statistics.fmean(samples['torso_origin_world_height'][begin:]),
                    'mean_x_velocity': statistics.fmean(samples['x_velocity'][begin:]),
                    'inverted_pitch_fraction': statistics.fmean(float(math.cos(x) < 0) for x in samples['pitch_radians'][begin:])})
        statistics_after = agent.statistics()
        normalization_after = agent.state_dict() if hasattr(agent, 'state_dict') else None
        require(statistics_before == statistics_after, 'agent statistics changed during inference')
        require(normalization_before == normalization_after, 'normalization moments changed during inference')
        after_hashes = fingerprints(sources)
        require(before_hashes == after_hashes, 'checkpoint/RMS/library/source inputs changed during diagnostic')
        require(forbidden_calls == {'training': 0, 'save': 0}, 'training/save was attempted')
        result = {'label': label, 'run': str(run), 'status': 'passed', 'study_stage': 'development',
                  'seed_start': SEED_START, 'episodes': EPISODES, 'deterministic': True,
                  'inference_steps': steps, 'new_training_steps': 0, 'new_updates': 0,
                  'library_sha256': metadata['library_sha256'], 'configuration': spec,
                  'episode_results': episode_rows, 'return_summary': summary([row['raw_return'] for row in episode_rows], np),
                  'metrics': {name: summary(values, np) for name, values in samples.items()},
                  'pitch_circular_mean_radians': math.atan2(statistics.fmean(map(math.sin, samples['pitch_radians'])),
                                                            statistics.fmean(map(math.cos, samples['pitch_radians']))),
                  'inverted_pitch_fraction': inverted / steps,
                  'action_boundary_fraction': int(boundary_counts.sum()) / (steps * boundary_counts.size),
                  'action_boundary_fraction_per_dimension': (boundary_counts / steps).tolist(),
                  'action_near_boundary_fraction_1e6': int(near_boundary_counts.sum()) / (steps * boundary_counts.size),
                  'action_boundary_interpretation': 'Observed executed-action endpoint frequency; latent actions unavailable, so this is NOT a measured clipping probability.',
                  'statistics_before': statistics_before, 'statistics_after': statistics_after,
                  'statistics_unchanged': True, 'normalization_state_before': normalization_before,
                  'normalization_state_after': normalization_after, 'normalization_state_unchanged': True,
                  'input_fingerprints_before': before_hashes, 'input_fingerprints_after': after_hashes,
                  'inputs_unchanged': True, 'forbidden_calls': forbidden_calls,
                  'step_trace': trace_path.name, 'step_trace_sha256': sha256(trace_path),
                  'seconds': time.monotonic() - started,
                  'packages': {name: importlib.metadata.version(name) for name in ('gymnasium', 'mujoco', 'numpy')}}
        write_json(output / f'{label}.json', result)
        return result
    finally:
        if env is not None:
            env.close()
        native.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', help='LABEL=run directory; default is the three selected development checkpoints')
    parser.add_argument('--output', type=Path, default=STUDY / 'diagnostics/halfcheetah_ppo_posture_development_1001')
    args = parser.parse_args()
    selected = args.run or [f'{label}={STUDY / "runs" / ("halfcheetah_ppo_diagnostic_core_v1_" + suffix + "_s1001")}' for label, suffix in DEFAULTS]
    runs = []
    for value in selected:
        label, path = value.split('=', 1)
        require(label and all(char.isascii() and (char.isalnum() or char == '_') for char in label), 'invalid output label')
        runs.append((label, Path(path).resolve(strict=True)))
    require(len({label for label, _ in runs}) == len(runs), 'duplicate output label')
    output = args.output.resolve()
    require(not output.exists(), 'output already exists; choose a new diagnostic directory')
    require(not output.is_relative_to(STUDY / 'runs'), 'diagnostic output must be outside original runs')
    output.mkdir(parents=True)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path[:0] = [str(ROOT / 'examples'), str(ROOT / 'benchmarks')]
    import numpy as np
    import gymnasium as gym
    import reinforcex_ffi as rx
    from reinforcex_normalization import NormalizedAgent
    from improvement_configs import as_dict, fill
    require(importlib.metadata.version('gymnasium') == '1.3.0', 'use the fixed study environment')
    libraries = {}
    results = [diagnose(label, run, output, libraries, np, gym, rx, as_dict, fill, NormalizedAgent)
               for label, run in runs]
    summary_json = {'generated_at': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
                    'scope': 'post-hoc development-only behavior diagnostic; no training and no final-confirmation replacement',
                    'inference_steps': sum(row['inference_steps'] for row in results),
                    'new_training_steps': 0, 'results': results}
    write_json(output / 'summary.json', summary_json)
    lines = ['# HalfCheetah PPO：保存モデルの無更新行動診断', '',
             '開発用 seed 1100000–1100009 の各 10 episode。各条件の保存済み最終 checkpoint を決定的に実行し、保存 final の同じ 10 episode と return/長さを照合した。確認用 seed は使用していない。この診断を最終100episodeの性能値や基準達成の判定に置き換えない。', '',
             '| 条件 | 開発10episode平均 | torso高さ平均 (m) | pitch円平均 (rad) | 反転姿勢割合 | x速度平均 (m/s) | forward/step | ctrl/step | 実行action端値率 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        m = row['metrics']
        lines.append(f"| [{row['label']}]({row['label']}.json) | {row['return_summary']['mean']:.2f} | {m['torso_origin_world_height']['mean']:.3f} | {row['pitch_circular_mean_radians']:.3f} | {row['inverted_pitch_fraction']:.1%} | {m['x_velocity']['mean']:.3f} | {m['reward_forward']['mean']:.3f} | {m['reward_ctrl']['mean']:.3f} | {row['action_boundary_fraction']:.1%} |")
    lines += ['', '姿勢・報酬は各 environment step 後に計測。torso 高さは root torso のモデル上の原点高さと rootz 座標の和（観測 rootz 単独とは異なる）。pitch は rooty 角度、反転姿勢は cos(pitch)<0 の時点割合。円平均は ±π のまたぎを考慮し、詳細 JSON に未 wrap 値・wrap 値の分位点も保存した。', '',
              '実行 action 端値率は、6次元 action の各成分が環境の下限/上限に厳密に等しい割合。FFI は latent action を返さないため、これは clip 前後を直接比較した clip 率ではない。near-boundary（1e-6）と次元別の割合も JSON にある。', '',
              'reward_ctrl は負の制御コスト。毎 step、reward_forward + reward_ctrl = raw reward を検証した。実行結果を reward scaling や最終確認成績の代わりに用いない。', '',
              f"全 {len(results)} 条件、計 {summary_json['inference_steps']:,} step の推論のみ。学習・保存呼び出しを禁止し、更新 counter は 0 のまま、agent statistics と normalization moments の前後一致を確認した。checkpoint・RMS・library・入力 metadata/final と使用 source の SHA256/サイズ/mtime が前後で不変。source trace は各条件の *.steps.jsonl.gz に保存。", '',
              '同じ seed1001 のモデル 3 本に対する事後診断であり、独立 training seed を増やした結果ではない。姿勢の傾向とスコアの相関から、局所解や特定設定の因果を確定しない。', '']
    (output / 'REPORT.ja.md').write_text('\n'.join(lines))
    print(json.dumps({'status': 'passed', 'output': str(output), 'inference_steps': summary_json['inference_steps'],
                      'returns': {row['label']: row['return_summary']['mean'] for row in results}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
