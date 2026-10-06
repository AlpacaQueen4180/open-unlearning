"""SPF NPO smoke mask adapter and trace audit; no optimizer or loss modification."""
import math


def source_mask_collator(base_class):
    class SPFSourceMaskCollator(base_class):
        def __call__(self, instances):
            if 'input_ids' not in instances[0]:
                return {key: self([item[key] for item in instances]) for key in instances[0]}
            for item in instances:
                if not 0 < len(item['input_ids']) <= 512:
                    raise ValueError('SPF NPO QA exceeds the fixed 512-token limit')
                if len(item['attention_mask']) != len(item['input_ids']):
                    raise ValueError('Source attention mask length mismatch')
                if item['attention_mask'].tolist() != [1] * len(item['input_ids']):
                    raise ValueError('Expected native unpadded QA source mask')
            result = super().__call__(instances)
            # Preserve native input/label/index padding, including valid EOS.
            # Token-ID inequality cannot distinguish EOS from padding when tied.
            result['attention_mask'] = self._pad_tokens(
                [item['attention_mask'] for item in instances], 0).to(
                    dtype=result['attention_mask'].dtype)
            return result
    return SPFSourceMaskCollator


def audit_trace(runtime):
    required = dict(trainer_global_step=2, engine_global_steps=2, microsteps=10,
                    epoch=1.0, world_size=1, micro_batch=4, configured_gas=8,
                    alpha=0.0, gamma=1.0, beta=0.1, retain_loss_type='NLL',
                    learning_rate=1e-7, dataset_examples=40, retain_pool_examples=3800,
                    forget_exposures=40, retain_exposures=40, unique_forget_examples=40,
                    attention='flash_attention_2',optimizer='paged_adamw_32bit',zero_stage=3,
                    reference_fingerprint_unchanged=True, initial_target_reference_equal=True)
    if any(runtime.get(key) != value for key,value in required.items()):
        raise ValueError('SPF smoke identity, reference or actual counter mismatch')
    trace = runtime.get('loss_scaling_trace', [])
    windows = [8] * 8 + [2] * 2
    if (len(trace) != 10 or [r['actual_window_microbatches'] for r in trace] != windows
            or [i+1 for i,r in enumerate(trace) if r['boundary']] != [8,10]
            or any(r['kwargs'].get('scale_wrt_gas') is not False for r in trace)):
        raise ValueError('Native full/tail window or DeepSpeed scaling mismatch')
    for row, window in zip(trace, windows):
        raw, actual = row['raw_npo_loss'], row['loss_entering_ds_backward']
        if (not math.isfinite(raw) or not math.isfinite(actual)
                or not math.isclose(actual, raw/window, rel_tol=2e-6, abs_tol=1e-8)
                or row['mask_valid'] is not True):
            raise ValueError('Loss scalar or source-mask audit failed')
    return dict(status='NATIVE_SCALING_MASK_REFERENCE_PASS_PENDING_FRESH_RELOAD',
                fresh_process_reload_verified=False, pilot_candidates_frozen=False)
