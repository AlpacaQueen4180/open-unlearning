"""Independent local audit of native four-update smoke and preserved gradient gates."""
import argparse, hashlib, json, math
from pathlib import Path
from audit_native_evidence import ks_exact_two_sided


def audit(root, short):
    model = {'llama31_8b': 'Llama-3.1-8B-Instruct', 'llama2_7b': 'Llama-2-7b-chat-hf'}[short]
    read = lambda path: json.loads((root / path).read_text(encoding='utf-8'))
    evidence = read('evidence-manifest.json')
    for name, meta in evidence.items():
        content = (root / name).read_bytes()
        assert len(content) == meta['bytes'] and hashlib.sha256(content).hexdigest() == meta['sha256']
    cell = Path('runs') / (short + '_forget01_s0_native_smoke4')
    manifest = read(cell / 'manifest.json')
    assert manifest['model'] == model and manifest['seed'] == 0 and manifest['split'] == 'forget01'
    assert manifest['smoke_only'] and manifest['native_no_legacy_patch']
    assert [manifest[k] for k in ['alpha', 'gamma', 'beta']] == [1, 1, .1]
    assets = read('assets_' + model + '.json')
    assert manifest['assets'] == assets
    command = read(short + '_train_smoke4.command.json')
    for item in ['scripts/reproduction/npo/gb200/short_native_probe.py', 'trainer=NPO',
                 'model=' + model, 'forget_split=forget01', 'retain_split=retain99',
                 'trainer.args.seed=0', 'trainer.args.per_device_train_batch_size=4',
                 'trainer.args.gradient_accumulation_steps=8', '+trainer.args.max_steps=4']:
        assert item in command
    frozen = read(short + '-tofu-frozen-40.json')
    assert frozen['assets'] == assets and frozen['indices'] == list(range(40)) and frozen['retain_sampling_seed'] == 0
    assert all(len(frozen['effective_tokens'][term]) == 40 for term in ['forget', 'retain'])
    runtime = read(cell / 'checkpoint/runtime_final_rank0.json')
    trainer = read(cell / 'checkpoint/trainer_state.json')
    assert runtime['trainer_global_step'] == runtime['engine_global_steps'] == trainer['global_step'] == 4
    assert runtime['microsteps'] == 20 and runtime['epoch'] == trainer['epoch'] == 2
    assert runtime['world_size'] == 1 and runtime['micro_batch'] == 4 and runtime['configured_gas'] == 8
    assert runtime['attention'] == 'flash_attention_2' and runtime['optimizer'] == 'paged_adamw_32bit'
    ds = runtime['resolved_deepspeed']
    assert ds['train_batch_size'] == 32 and ds['gradient_accumulation_steps'] == 8 and ds['zero_optimization']['stage'] == 3
    trace = runtime['loss_scaling_trace']
    assert len(trace) == 20 and [x['microstep'] for x in trace] == list(range(1, 21))
    assert [x['actual_window_microbatches'] for x in trace] == ([8] * 8 + [2] * 2) * 2
    assert [x['microstep'] for x in trace if x['boundary']] == [8, 10, 18, 20]
    for entry in trace:
        expected = entry['raw_npo_loss'] / entry['actual_window_microbatches']
        assert math.isfinite(expected) and math.isclose(entry['loss_entering_ds_backward'], expected, rel_tol=2e-6, abs_tol=1e-8)
        assert math.isclose(entry['ratio'], 1 / entry['actual_window_microbatches'], rel_tol=2e-6, abs_tol=1e-8)
        assert entry['kwargs']['scale_wrt_gas'] is False
    proof = read('checkpoint-proof.json')
    config = read(cell / 'checkpoint/config.json')
    assert proof['header_and_required_layers_complete'] and proof['model'] == model and proof['counts'] == runtime
    assert proof['num_hidden_layers'] == config['num_hidden_layers'] == 32
    assert proof['tensor_count'] >= 32 * 9 + 2 and sum(s['tensor_count'] for s in proof['shards']) == proof['tensor_count']
    assert sum(s['bytes'] for s in proof['shards']) == proof['weights_bytes']
    assert all(s['bytes'] > 100_000_000 and len(s['sha256']) == 64 for s in proof['shards'])
    gradients = []
    parameter_sha = None
    for window, offset in [(8, 0), (2, 32)]:
        g = read(short + '_TOFU_w%d_combined_bf16.json' % window)
        assert g['versions']['transformers'] == '5.5.4' and g['versions']['accelerate'] == '1.13.0' and g['versions']['deepspeed'] == '0.15.4'
        assert g['world_size'] == 1 and g['gas'] == g['engine_gas'] == 8 and g['window'] == window and g['batch_offset'] == offset
        assert g['model'] == assets['model_path'] and g['dtype'] == 'bf16' and g['term'] == 'combined'
        assert g['num_items_in_batch'] is None and g['model_accepts_loss_kwargs']
        assert g['optimizer_lr'] == g['clipping'] == 0 and not g['legacy_tail_boundary_forced']
        assert g['engine_updates'] == g['ds_reference_updates'] == 1 and g['engine_total_updates'] == 2
        assert g['initial_parameter_sha256'] == g['final_parameter_sha256']
        if parameter_sha is not None:
            assert parameter_sha == g['initial_parameter_sha256']
        parameter_sha = g['initial_parameter_sha256']
        observed, reference = g['engine_backward_trace'], g['ds_reference_backward_trace']
        assert len(observed) == len(reference) == window
        for rows in (observed, reference):
            assert [i + 1 for i, r in enumerate(rows) if r['boundary']] == [window]
            assert all(r['kwargs']['scale_wrt_gas'] is False for r in rows)
        assert observed == reference
        matched = g['vs_ds_explicit_mean_reference']
        assert abs(matched['projected_scale'] - 1) < .05 and matched['relative_error_after_scale'] < .05
        autograd = g['vs_mean_microbatch_objective']
        independent_gate = abs(autograd['projected_scale'] - 1) < .05 and autograd['relative_error_after_scale'] < .05
        assert independent_gate == g['autograd_reference_gate_passed']
        assert (g['mean_micro_vs_large']['relative_error'] < .05) == g['autograd_micro_vs_large_gate_passed']
        gradients.append({'window': window, 'matched_ds_reference': matched, 'matched_gate_passed': True,
                          'autograd_reference': autograd, 'autograd_gate_passed': independent_gate,
                          'mean_micro_vs_large': g['mean_micro_vs_large'], 'parameter_sha256': parameter_sha})
    ref_bytes = (root / 'reference/retain99_TOFU_EVAL.json').read_bytes()
    assert hashlib.sha256(ref_bytes).hexdigest() == assets['retain_sha256']
    reference = json.loads(ref_bytes)
    raw, summary = read(cell / 'eval/TOFU_EVAL.json'), read(cell / 'eval/TOFU_SUMMARY.json')
    scores = lambda data: [v['score'] for v in data['forget_truth_ratio']['value_by_index'].values()]
    left, right = scores(raw), scores(reference)
    assert len(left) == len(right) == 40
    assert all(len(raw[k]['value_by_index']) == 40 for k in ['forget_Q_A_Prob', 'forget_Q_A_ROUGE'])
    statistic, pvalue = ks_exact_two_sided(left, right)
    keys = ['retain_Q_A_Prob', 'retain_Q_A_ROUGE', 'retain_Truth_Ratio', 'ra_Q_A_Prob_normalised', 'ra_Q_A_ROUGE', 'ra_Truth_Ratio', 'wf_Q_A_Prob_normalised', 'wf_Q_A_ROUGE', 'wf_Truth_Ratio']
    utility = [raw[k]['agg_value'] for k in keys]
    assert all(math.isfinite(v) and v >= 0 for v in utility)
    computed = {'forget_quality': pvalue, 'model_utility': 0 if 0 in utility else len(utility) / sum(1 / v for v in utility),
                'forget_truth_ratio': sum(min(v, 1 / (v + 1e-10)) for v in left) / len(left)}
    deltas = {k: abs(v - summary[k]) for k, v in computed.items()}
    assert max(deltas.values()) < 1e-12
    result = {'model': model, 'scope': 'forget01 seed0 four-update smoke, not full reproduction or safety acceptance',
              'raw_files_sha_verified': len(evidence), 'native_updates_boundary_scaling_gate_passed': True,
              'trainer_updates': 4, 'ds_updates': 4, 'microbatches': 20, 'epoch': 2,
              'boundaries': [8, 10, 18, 20], 'full_window_divisor': 8, 'tail_window_divisor': 2,
              'gradient_cases': gradients, 'all_autograd_gates_passed': all(g['autograd_gate_passed'] for g in gradients),
              'summary': computed, 'ks_D': statistic, 'samples': [40, 40], 'summary_deltas': deltas,
              'ks_method': 'two-sided exact integer lattice paths', 'checkpoint_proof': proof,
              'checkpoint_proof_method': 'remote CPU streamed file SHA, safetensors headers/index/offsets/shapes/all layers; weights not downloaded',
              'assets': assets, 'frozen_batch_sha256': frozen['sha256']}
    (root / 'independent-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return {k: v for k, v in result.items() if k not in ['checkpoint_proof', 'assets', 'gradient_cases']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--short', choices=['llama31_8b', 'llama2_7b'], required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.short), indent=2))
