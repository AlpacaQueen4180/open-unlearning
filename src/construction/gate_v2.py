"""Development interval gate. Legacy gate remains available for old evidence."""
import hashlib
import json
import math
from pathlib import Path


def target_gate_v2(evidence, *, expected_target=None):
    path = Path(__file__).resolve().parents[2] / 'configs/construction/development_gate_policy_v2.json'
    policy = json.loads(path.read_text(encoding='utf-8'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    failed, missing = [], []
    if evidence.get('selection_source') != 'development':
        failed.append('Only development evidence may select targets')
    if not evidence.get('target_id') or (expected_target and evidence['target_id'] != expected_target):
        failed.append('Target identity missing or mismatched')
    if evidence.get('policy_sha256') != digest:
        missing.append('Evidence must bind the frozen policy hash')
    if evidence.get('overlap_audit_status') == 'fail':
        failed.append('Overlap audit failed')
    elif evidence.get('overlap_audit_status') != 'pass':
        missing.append('Passing overlap audit')
    for field in ('paired_data_integrity', 'cluster_mapping_verified', 'adjudication_complete'):
        if evidence.get(field) is not True:
            missing.append(field)
    if evidence.get('unresolved_judgments') != 0:
        missing.append('Unresolved judgments or missing coverage')
    minimums, maximums = dict(policy['minimums']), dict(policy['maximums'])
    method = evidence.get('method')
    if method not in ('standard', 'mixing', 'spf'):
        missing.append('Known construction method')
    if method in ('mixing', 'spf'):
        minimums.update(policy['matched_standard_minimums'])
        maximums.update(policy['matched_standard_maximums'])
        if not evidence.get('standard_target_id') or evidence.get('standard_learning_pass') is not True:
            missing.append('Matched Standard identity and passing learning gate')
    for direction, bounds in [('minimum', minimums), ('maximum', maximums)]:
        for name, threshold in bounds.items():
            item = evidence.get('intervals', {}).get(name, {})
            lo, hi = item.get('lower'), item.get('upper')
            if (not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in (lo, hi))
                    or lo > hi or item.get('confidence_level') != policy['confidence_level']
                    or item.get('method') != policy['interval_method']
                    or item.get('replicates') != policy['bootstrap_replicates']
                    or item.get('seed') != policy['bootstrap_seed']):
                missing.append(name + ': valid preregistered interval required')
                continue
            if direction == 'minimum':
                if hi < threshold: failed.append(name)
                elif lo < threshold: missing.append(name + ': interval crosses threshold')
            else:
                if lo > threshold: failed.append(name)
                elif hi > threshold: missing.append(name + ': interval crosses threshold')
    return dict(status='fail' if failed else 'insufficient_evidence' if missing else 'pass',
                reasons=failed, missing=missing, policy_sha256=digest, schema_version=2)
