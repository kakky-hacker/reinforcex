"""Numerical/contract fixtures for the improvement-study reporting tool."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('improvement_report', Path(__file__).with_name('render_improvement_report.py'))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'report'

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, name='run', seed=1001, stage='development', means=(5000., 4700.), complete=True,
                episodes=100, case='halfcheetah_hybrid'):
        path = self.root / name
        path.mkdir()
        agents = [{'algorithm': 'ppo', 'config': {'lr': .001}} for _ in means]
        meta = {'backend': 'reinforcex', 'case': case, 'env_id': 'HalfCheetah-v5', 'seed': seed,
                'study_stage': stage, 'library_sha256': 'a' * 64, 'overrides': {'z': 1, 'a': {'v': 2}},
                'effective_agents': agents, 'worker_count': len(agents), 'configuration': {'reward_mode': 'raw'},
                'requested_total_steps': 1000 * len(agents), 'reward_threshold': 4800.,
                'concurrent_workers': True, 'share_rnd': False, 'validation_seed': 1100000,
                'validation_episodes': 10, 'final_test_seed': 1200000 if stage == 'confirmation' else 1100000,
                'final_test_episodes': episodes}
        (path / 'metadata.json').write_text(json.dumps(meta))
        final = {'status': 'complete', 'case': case, 'env_id': 'HalfCheetah-v5', 'seed': seed,
                 'study_stage': stage, 'actual_total_steps': 1000 * len(agents), 'threshold': 4800.,
                 'workers': [], 'test': []}
        evals = []
        for worker, mean in enumerate(means):
            train = [{'worker': worker, 'algorithm': 'ppo', 'episode': i, 'reward': value,
                      'learning_reward': -1000., 'budget_cut': i == 4}
                     for i, value in enumerate([10., 20., 30., 999.], 1)]
            (path / f'train_worker{worker}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in train))
            final['workers'].append({'worker': worker, 'algorithm': 'ppo', 'steps': 1000, 'statistics': {'optimizer_steps': 17.}})
            validation = {'worker': worker, 'algorithm': 'ppo', 'aggregate_steps': 0, 'split': 'validation',
                          'returns': [9999.] * 10, 'mean': 9999., 'episodes': 10, 'seed_start': 1100000, 'reward': 'raw', 'deterministic': True}
            evals.append(validation)
            if complete:
                test = {'worker': worker, 'algorithm': 'ppo', 'aggregate_steps': 1000 * len(agents),
                        'split': 'test' if stage == 'confirmation' else 'development',
                        'returns': [mean] * episodes, 'mean': mean, 'episodes': episodes,
                        'seed_start': meta['final_test_seed'], 'reward': 'raw', 'deterministic': True}
                evals.append(test)
                final['test'].append(test)
        (path / 'evaluations.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in evals))
        if complete:
            (path / 'final.json').write_text(json.dumps(final))
        return path

    def load(self, path):
        return r.load_run(path, self.output, 100)

    def test_variant_uses_canonical_overrides_build_case_and_budget(self):
        meta = json.loads((self.fixture() / 'metadata.json').read_text())
        reordered = copy.deepcopy(meta)
        reordered['overrides'] = {'a': {'v': 2}, 'z': 1}
        self.assertEqual(r.variant_id(meta), r.variant_id(reordered))
        for key, value in [('library_sha256', 'b' * 64), ('case', 'halfcheetah_ppo_diagnostic'),
                           ('requested_total_steps', 4000), ('overrides', {'z': 2})]:
            changed = {**meta, key: value}
            self.assertNotEqual(r.variant_id(meta), r.variant_id(changed))

    def test_worker_failure_survives_passing_seed_average(self):
        group = r.aggregate([self.load(self.fixture())])[0]
        self.assertEqual(group['mean'], 4850.)
        self.assertEqual(group['passed_models'], 1)
        self.assertEqual(group['evaluated_models'], 2)
        self.assertEqual(group['mean_passed_seeds'], 1)
        self.assertEqual(group['all_workers_passed_seeds'], 0)
        self.assertEqual(group['seeds'][0]['worker_results'][1]['mean'], 4700.)
        self.assertEqual(group['seeds'][0]['worker_results'][0]['statistics']['optimizer_steps'], 17.)

    def test_worker_topology_is_part_of_variant_identity(self):
        meta = json.loads((self.fixture() / 'metadata.json').read_text())
        for key, value in [('worker_count', 1), ('concurrent_workers', False), ('share_rnd', True)]:
            with self.subTest(key=key):
                self.assertNotEqual(r.variant_id(meta), r.variant_id({**meta, key: value}))
        for key, value in [('worker_count', True), ('worker_count', 0),
                           ('concurrent_workers', 1), ('share_rnd', 'false')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                r.variant_id({**meta, key: value})

    def test_same_seed_private_and_shared_rnd_are_separate_groups(self):
        private = self.load(self.fixture('private'))
        shared_path = self.fixture('shared')
        meta_path = shared_path / 'metadata.json'
        meta = json.loads(meta_path.read_text())
        meta['share_rnd'] = True
        meta_path.write_text(json.dumps(meta))
        groups = r.aggregate([private, self.load(shared_path)])
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(group['completed_seeds'] == 1 for group in groups))
        self.assertTrue(all(group['evaluated_models'] == 2 for group in groups))

    def test_equal_seed_weight_and_seed_sd_not_evaluation_sd(self):
        a = self.load(self.fixture('a', seed=1001, means=(5000., 4700.)))
        b = self.load(self.fixture('b', seed=1002, means=(5500., 6500.)))
        group = r.aggregate([a, b])[0]
        self.assertEqual(group['mean'], (4850. + 6000.) / 2)
        self.assertEqual(group['seed_sd'], statistics.stdev([4850., 6000.]))
        self.assertEqual(group['all_workers_passed_seeds'], 1)
        self.assertEqual(group['mean_passed_seeds'], 2)

    def test_stage_and_diagnostic_case_are_separate(self):
        runs = [self.load(self.fixture('dev')), self.load(self.fixture('confirmation', stage='confirmation')),
                self.load(self.fixture('diagnostic', case='halfcheetah_ppo_diagnostic', means=(6000.,)))]
        groups = r.aggregate(runs)
        self.assertEqual(len(groups), 3)
        self.assertEqual(sum(group['study_role'] == 'supplementary' for group in groups), 1)
        self.assertEqual({group['stage'] for group in groups}, {'confirmation', 'development'})

    def test_pending_and_nonstandard_final_excluded(self):
        finished = self.load(self.fixture('complete'))
        pending = self.load(self.fixture('pending', seed=1002, complete=False))
        group = r.aggregate([finished, pending])[0]
        self.assertEqual(group['completed_seeds'], 1)
        self.assertEqual(group['incomplete_seeds'], 1)
        self.assertEqual(group['expected_models_from_metadata'], 4)
        self.assertEqual(group['evaluated_models'], 2)
        short = self.load(self.fixture('short', episodes=10))
        self.assertEqual(short['status'], 'complete_nonstandard_evaluation')
        self.assertEqual(r.aggregate([short])[0]['completed_seeds'], 0)

    def test_no_best_checkpoint_selection_and_raw_training_only(self):
        run = self.load(self.fixture(means=(1.,)))
        self.assertEqual(r.aggregate([run])[0]['mean'], 1.)
        worker = run['workers'][0]
        self.assertEqual(worker['natural_episodes'], 3)
        self.assertEqual(worker['budget_cut_episodes'], 1)
        self.assertEqual(worker['smoothing_window'], 3)
        self.assertEqual(r.moving_average([row['reward'] for row in worker['_training']]), ([2], [20.], 3))
        indices, means, window = r.moving_average(list(range(101)))
        self.assertEqual(indices, [99, 100])
        self.assertEqual(means, [49.5, 50.5])
        self.assertEqual(window, 100)
        self.assertEqual(r.moving_average([]), ([], [], 0))

    def test_duplicate_seed_and_hidden_contract_difference_rejected(self):
        run = self.load(self.fixture())
        with self.assertRaisesRegex(ValueError, 'duplicate seed'):
            r.aggregate([run, copy.deepcopy(run)])
        other = copy.deepcopy(run)
        other['seed'] = 1002
        other['_contract'] = 'different effective model'
        with self.assertRaisesRegex(ValueError, 'conflicting effective'):
            r.aggregate([run, other])

    def test_mean_tampering_and_final_log_mismatch_rejected(self):
        path = self.fixture()
        final = r.read_json(path / 'final.json')
        final['test'][0]['mean'] += 1
        (path / 'final.json').write_text(json.dumps(final))
        with self.assertRaisesRegex(ValueError, 'stored mean disagrees'):
            self.load(path)
        final['test'][0]['mean'] -= 1
        final['test'][0]['extra_field'] = 'not in evaluations'
        (path / 'final.json').write_text(json.dumps(final))
        with self.assertRaisesRegex(ValueError, 'final/log records disagree'):
            self.load(path)

    def test_live_partial_tail_only_is_tolerated(self):
        path = self.root / 'log.jsonl'
        path.write_text('{"x": 1}\n{"x":')
        warnings = []
        self.assertEqual(r.read_jsonl(path, True, warnings), [{'x': 1}])
        self.assertEqual(len(warnings), 1)
        with self.assertRaises(ValueError):
            r.read_jsonl(path, False, [])
        path.write_text('{broken}\n{"x": 1}\n')
        with self.assertRaises(ValueError):
            r.read_jsonl(path, True, [])

    def test_unknown_stage_or_confirmation_seed_contamination_rejected(self):
        path = self.fixture(stage='confirmation')
        meta = r.read_json(path / 'metadata.json')
        meta['study_stage'] = 'developmnt'
        (path / 'metadata.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'unknown study stage'):
            self.load(path)
        meta['study_stage'] = 'confirmation'
        meta['final_test_seed'] = 1100000
        (path / 'metadata.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'evaluation seeds differ'):
            self.load(path)

    def test_json_and_markdown_link_to_sources_without_internal_arrays(self):
        run = self.load(self.fixture())
        groups = r.aggregate([run])
        self.output.mkdir()
        r.write_report([run], groups, self.output, 100)
        result = r.read_json(self.output / 'summary.json')
        self.assertNotIn('_members', result['groups'][0])
        self.assertNotIn('_training', result['runs'][0]['workers'][0])
        self.assertEqual(result['groups'][0]['seeds'][0]['worker_results'][1]['mean'], 4700.)
        markdown = (self.output / 'REPORT.ja.md').read_text()
        self.assertIn('../run/final.json', markdown)
        self.assertIn('4,700.00 (100)', markdown)
        self.assertIn('信頼区間ではない', markdown)

    def reference_fixture(self, name='sb3', complete=True, steps=2000):
        path = self.root / name
        path.mkdir()
        cfg = {'agent': {'gamma': .99, 'hidden_size': 16, 'hidden_layers': 1}, 'action_space': 1,
               'learning_rate': .0003, 'update_interval': 1000, 'minibatch_size': 100, 'epochs': 2,
               'gae_lambda': .95, 'policy_clip_epsilon': .2, 'value_clip_range': 0.,
               'value_loss_coefficient': .5, 'entropy_coefficient': 0., 'standardize_gae': 1,
               'model': 1, 'activation': 0, 'initial_log_std': 0., 'adam_epsilon': 1e-8, 'target_kl': 0.}
        native = {'env_id': 'HalfCheetah-v5', 'reward_mode': 'raw', 'shared_replay': False,
                  'agents': [{'algorithm': 'ppo', 'config': cfg, 'rnd_config': None}]}
        args = {'case': 'halfcheetah_ppo_diagnostic', 'stage': 'development', 'seed': 1001,
                'env': 'HalfCheetah-v5', 'algo': 'ppo', 'steps': steps, 'success_threshold': 4800.,
                'native_config': native, 'validation_seed': 1100000, 'final_eval_seed': 1100000,
                'progress_eval_episodes': 10, 'eval_episodes': 100, 'reward_transform': 'raw'}
        constructor = {'seed': 1001, 'gamma': .99, 'learning_rate': .0003, 'n_steps': 1000, 'batch_size': 100,
                       'n_epochs': 2, 'gae_lambda': .95, 'clip_range': .2, 'clip_range_vf': None,
                       'ent_coef': 0., 'vf_coef': .5, 'normalize_advantage': True, 'max_grad_norm': .5,
                       'target_kl': None, 'policy_kwargs': {'net_arch': [16, 16], 'optimizer_kwargs': {'eps': 1e-8},
                       'activation_fn': "<class 'torch.nn.modules.activation.Tanh'>", 'ortho_init': True, 'log_std_init': 0.}}
        meta = {'case': args['case'], 'study_stage': 'development', 'n_envs': 1, 'script_sha256': 'b' * 64,
                'packages': {'torch': '2.14.0', 'stable-baselines3': '2.9.0'}, 'validation_seed': 1100000,
                'final_eval_seed': 1100000, 'comparison_notes': ['Population vs sample advantage SD differs.']}
        (path / 'metadata.json').write_text(json.dumps(meta))
        (path / 'config.json').write_text(json.dumps({'arguments': args, 'resolved_constructor': constructor,
                         'environment_spec': {'id': args['env'], 'max_episode_steps': 1000}}))
        summaries, episodes = [], []
        for point, phase, count, mean, step in [(1, 'initial', 10, -10., 0)] + ([(2, 'final', 100, 2000., steps)] if complete else []):
            row = {'point': point, 'phase': phase, 'split': 'development' if phase == 'final' else 'validation',
                   'study_stage': 'development', 'env_steps': step, 'requested_step': step,
                   'episodes': count, 'seed_start': 1100000, 'mean_return': mean, 'deterministic': True}
            summaries.append(row)
            episodes += [{**{k: row[k] for k in ('point', 'phase', 'split', 'study_stage', 'env_steps', 'requested_step')},
                          'episode': i + 1, 'seed': 1100000 + i, 'return': mean, 'length': 1000} for i in range(count)]
        for filename, rows in [('evaluations.jsonl', summaries), ('eval_episodes.jsonl', episodes),
                               ('train_episodes.jsonl', [{'episode': 1, 'return': 42., 'train_return': -100., 'env_steps': 1000}])]:
            (path / filename).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        if complete:
            final = {'status': 'complete', 'case': args['case'], 'study_stage': 'development',
                     'engine': 'stable-baselines3', 'algorithm': 'ppo', 'environment': args['env'], 'seed': 1001,
                     'requested_steps': steps, 'actual_steps': steps, 'updates': 20, 'final_evaluation': summaries[-1]}
            (path / 'final.json').write_text(json.dumps(final))
        return path

    def test_reference_recomputes_raw_episodes_and_keeps_backend_separate(self):
        path = self.reference_fixture()
        reference = r.load_reference_run(path, self.output, 100)
        self.assertEqual(reference['workers'][0]['final']['mean'], 2000.)
        self.assertEqual(reference['workers'][0]['_training'][0]['reward'], 42.)
        self.assertTrue(reference['comparison_eligible'])
        native = self.load(self.fixture(means=(3000.,), case='halfcheetah_ppo_diagnostic'))
        self.assertEqual(len(r.aggregate([reference, native])), 2)
        rows = [json.loads(x) for x in (path / 'eval_episodes.jsonl').read_text().splitlines()]
        rows[-1]['return'] += 100
        (path / 'eval_episodes.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
        with self.assertRaisesRegex(ValueError, 'stored mean disagrees'):
            r.load_reference_run(path, self.output, 100)

    def test_reference_matching_requires_settings_seed_budget_and_protocol(self):
        reference = r.load_reference_run(self.reference_fixture(), self.output, 100)
        native = self.load(self.fixture(means=(3000.,), case='halfcheetah_ppo_diagnostic'))
        native['comparison_settings'] = copy.deepcopy(reference['comparison_settings'])
        native['identity']['requested_total_steps'] = 2000
        self.assertEqual(r.matched_comparisons([native, reference])[0]['native_minus_reference'], 1000.)
        for field, value in [('seed', 999), ('interpretation', 'provenance_discrepancy'),
                             ('comparison_eligible', False), ('comparison_settings', {}), ('evaluation_protocol', {})]:
            changed = {**reference, field: value}
            self.assertEqual(r.matched_comparisons([native, changed]), [])
        reference['identity']['requested_total_steps'] = 4000
        self.assertEqual(r.matched_comparisons([native, reference]), [])

    def test_reference_overrides_and_overshoot_not_misrepresented_as_matched(self):
        path = self.reference_fixture()
        config = r.read_json(path / 'config.json')
        config['arguments']['hyperparams'] = {'n_epochs': 3}
        (path / 'config.json').write_text(json.dumps(config))
        self.assertFalse(r.load_reference_run(path, self.output, 100)['comparison_eligible'])
        config['arguments'].pop('hyperparams')
        config['arguments']['steps'] = 1999
        (path / 'config.json').write_text(json.dumps(config))
        final = r.read_json(path / 'final.json')
        final['requested_steps'] = 1999
        (path / 'final.json').write_text(json.dumps(final))
        run = r.load_reference_run(path, self.output, 100)
        self.assertFalse(run['comparison_eligible'])
        self.assertEqual(run['status'], 'complete')
        self.assertEqual(run['workers'][0]['final']['steps'], 2000)

    def test_linear_schedule_exact_mapping_and_group_identity(self):
        path = self.reference_fixture()
        config = r.read_json(path / 'config.json')
        native_spec = config['arguments']['native_config']['agents'][0]
        native_spec['learning_rate_schedule'] = {'kind': 'linear', 'final_fraction': .05}
        schedule = {'kind': 'linear', 'initial': native_spec['config']['learning_rate'], 'final_fraction': .05}
        config['resolved_constructor']['learning_rate'] = schedule
        (path / 'config.json').write_text(json.dumps(config))
        reference = r.load_reference_run(path, self.output, 100)
        self.assertTrue(reference['comparison_eligible'])
        self.assertEqual(reference['comparison_settings']['agents'][0]['learning_rate_schedule']['final_fraction'], .05)
        for change in ({'kind': 'exponential'}, {'initial': .0004}, {'final_fraction': .050000000000001}, {'extra': 1}):
            altered = copy.deepcopy(config['resolved_constructor'])
            altered['learning_rate'].update(change)
            self.assertTrue(r.reference_mapping_issues(config['arguments'], altered))
        altered = copy.deepcopy(config['resolved_constructor'])
        altered['learning_rate'] = .0003
        self.assertTrue(r.reference_mapping_issues(config['arguments'], altered))
        altered_args = copy.deepcopy(config['arguments'])
        altered_args['native_config']['agents'][0]['learning_rate_schedule']['final_fraction'] = True
        self.assertTrue(r.reference_mapping_issues(altered_args, config['resolved_constructor']))

        native_path = self.fixture(means=(3000.,), case='halfcheetah_ppo_diagnostic')
        meta = r.read_json(native_path / 'metadata.json')
        original_id = r.variant_id(meta)
        meta['effective_agents'][0]['learning_rate_schedule'] = {'kind': 'linear', 'final_fraction': .05}
        scheduled_id = r.variant_id(meta)
        self.assertNotEqual(original_id, scheduled_id)
        self.assertEqual(r.variant_identity(meta)['learning_rate_schedules'], [{'kind': 'linear', 'final_fraction': .05}])
        meta['effective_agents'][0]['learning_rate_schedule']['final_fraction'] = .1
        self.assertNotEqual(scheduled_id, r.variant_id(meta))
        native_spec['learning_rate_schedule']['final_fraction'] = .1
        config['resolved_constructor']['learning_rate']['final_fraction'] = .1
        (path / 'config.json').write_text(json.dumps(config))
        changed = r.load_reference_run(path, self.output, 100)
        self.assertTrue(changed['comparison_eligible'])
        self.assertNotEqual(reference['variant'], changed['variant'])

    def test_decisions_preserve_stage_exclude_evidence_and_require_metadata_hash(self):
        path = self.fixture(stage='confirmation')
        run = self.load(path)
        document = {'schema_version': 1, 'decisions': [{'run': path.name,
                    'metadata_sha256': hashlib.sha256((path / 'metadata.json').read_bytes()).hexdigest(),
                    'interpretation': 'redevelopment', 'reason': 'Stability gate failed.'}]}
        decision = self.root / 'decisions.json'
        decision.write_text(json.dumps(document))
        r.apply_decisions([run], decision)
        self.assertEqual(run['stage'], 'confirmation')
        self.assertFalse(run['confirmation_evidence_eligible'])
        self.assertFalse(run['achievement_evidence_eligible'])
        self.assertEqual(r.aggregate([run])[0]['passed_models'], 1)  # historical numeric count remains visible
        document['decisions'][0]['metadata_sha256'] = '0' * 64
        decision.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, 'metadata SHA mismatch'):
            r.apply_decisions([run], decision)

    def test_confirmation_blocks_and_pending_reference_remain_separate(self):
        first = self.load(self.fixture(stage='confirmation'))
        second = copy.deepcopy(first)
        second['seed'] = 99
        second['evaluation_protocol']['final_seed'] = 1300000
        second['_contract'] = 'new evaluation block'
        self.assertEqual(len(r.aggregate([first, second])), 2)
        r.evaluation_protocol('confirmation', 1100000, 10, 1300000, 100)
        with self.assertRaisesRegex(ValueError, 'evaluation seeds differ'):
            r.evaluation_protocol('confirmation', 1100000, 10, 1300001, 100)
        pending = r.load_reference_run(self.reference_fixture(complete=False), self.output, 100)
        self.assertEqual(r.aggregate([pending])[0]['completed_seeds'], 0)


if __name__ == '__main__':
    unittest.main()
