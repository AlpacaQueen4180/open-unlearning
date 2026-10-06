"""Prepare pinned development executors for an audited SPF target; never launch evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import re

SOURCE_SHA256 = {
    'evaluate_construction_baseline.py': '4197f8b323b146d643119195f1e5a1f623a4fa87a5d69a7f066c33d14e25d4ec',
    'generate_development.py': '9a99410fb62985d124a19772d621628728fa20334efc36aa4d441cdf2a59dfe2',
    'generate_development_conversation.py': '95db77e44ea26e010e380cd9abf5b21ea9b94eee734f03f64da4d3887f9216fc',
    'evaluate_development_knowledge.py': '22796a344c14db1f3cf275c7d82d3e639d40a71eff4e92dd2049076ddee4a847',
    'score_development_ifbench.py': 'c590e30aff720c6e747dc3c7fee946b76f83470c6e82078057ef7d69e2517c1d',
}
META_REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
TOFU_REVISION = '324592d84ae4f482ac7249b9285c2ecdb53e3a68'
DATA_SHA256 = dict(full='cf6f9c9bd844b60661f1c923e6b188628f00b92c1834efb8f9f7c552274a9a33',
                   retain95='e1bfcea25c3237064c11676e9d8f52145032ccb45a0e7f858e7f8446475e8a3e')
TEMPLATE_SHA256 = '7d3e75bbebc8ca06f05579ff90e1730db88e7982d6c4ff524c55c736b21a10a0'


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_target_audit(audit):
    expected = dict(status='pass', method='spf', split='full', construction_seed=0,
                    actual_epochs=5.0, actual_updates=625, actual_examples=20000,
                    unique_examples=4000)
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError('Complete seed0 SPF full audit required')
    checkpoint = audit.get('final_checkpoint', {})
    if (checkpoint.get('parameter_tensors') != 291 or checkpoint.get('layers') != 32
            or not checkpoint.get('files') or audit.get('fresh_reload', {}).get('status') != 'pass'):
        raise ValueError('Complete 8B checkpoint and fresh reload required')
    for entry in checkpoint['files']:
        name = Path(entry['name'])
        if (name.name != entry['name'] or '/' in entry['name'] or '\\' in entry['name'] or name.suffix != '.safetensors'
                or not re.fullmatch('[0-9a-f]{64}', entry.get('sha256', ''))
                or not isinstance(entry.get('size'), int) or entry['size'] <= 0):
            raise ValueError('Invalid audited shard identity')
    if len({entry['name'] for entry in checkpoint['files']}) != len(checkpoint['files']):
        raise ValueError('Duplicate audited shard')


def adapt_source(name, raw, *, target, identity, repository):
    """Reject any source drift and prove the inverse edits restore the exact source."""
    if name not in SOURCE_SHA256 or sha_bytes(raw) != SOURCE_SHA256[name]:
        raise ValueError('Development execution source SHA mismatch: ' + name)
    if not re.fullmatch('[0-9a-f]{64}', identity):
        raise ValueError('Checkpoint identity must be SHA256')
    text = raw.decode('utf8')
    changes = []

    def replace_once(old, new):
        nonlocal text
        if text.count(old) != 1 or new in text:
            raise ValueError('Expected unique adapter edit: ' + name)
        text = text.replace(old, new, 1)
        changes.append((old, new))

    if name == 'score_development_ifbench.py':
        return raw, dict(source_sha256=sha_bytes(raw), adapter_sha256=sha_bytes(raw), edits=[])
    replace_once('def main():\n',
                 "def main():\n    if (int(os.environ.get('WORLD_SIZE', '1')) != 1\n"
                 "            or int(os.environ.get('RANK', '0')) != 0\n"
                 "            or int(os.environ.get('LOCAL_RANK', '0')) != 0):\n"
                 "        raise ValueError('SPF development adapter requires world1/rank0/local0')\n")
    if name == 'evaluate_construction_baseline.py':
        replace_once('AutoModelForCausalLM.from_pretrained(model_id, revision=revision,',
                     f'AutoModelForCausalLM.from_pretrained({target!r}, revision=None,')
        replace_once('repo = Path(__file__).resolve().parents[3]', f'repo = Path({repository!r})')
        replace_once("'model_id':model_id,'model_revision':revision,",
                     f"'model_id':{target!r},'model_revision':None,'checkpoint_identity_sha256':{identity!r},")
    elif name == 'evaluate_development_knowledge.py':
        replace_once("AutoModelForCausalLM.from_pretrained(m['model_id'],revision=m['model_revision'],",
                     f'AutoModelForCausalLM.from_pretrained({target!r},revision=None,')
        replace_once("model=m['model_id'],revision=m['model_revision'],",
                     f'model={target!r},revision=None,checkpoint_identity_sha256={identity!r},world_size=1,')
    else:
        replace_once("AutoModelForCausalLM.from_pretrained(manifest['model_id'],revision=manifest['model_revision'],",
                     f'AutoModelForCausalLM.from_pretrained({target!r},revision=None,')
        replace_once("model=manifest['model_id'],",
                     f'model={target!r},checkpoint_identity_sha256={identity!r},world_size=world,')
        replace_once("revision=manifest['model_revision'],tokenizer_revision=",
                     'revision=None,tokenizer_revision=')
    if 'model.generate(**' in text:
        replace_once('model.generate(**', 'model.generate(use_model_defaults=False, do_sample=False, **')
    compile(text, name, 'exec')
    restored = text
    for old, new in reversed(changes):
        if restored.count(new) != 1:
            raise ValueError('Noninvertible adapter edit: ' + name)
        restored = restored.replace(new, old, 1)
    if restored.encode('utf8') != raw:
        raise ValueError('Unexpected development arithmetic or protocol change: ' + name)
    adapted = text.encode('utf8')
    return adapted, dict(source_sha256=sha_bytes(raw), adapter_sha256=sha_bytes(adapted),
                         edits=[dict(before=old, after=new) for old, new in changes],
                         exact_source_restored_after_inverse_edits=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repository', 'manifest', 'target', 'checkpoint-audit', 'checkpoint-audit-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    repo, target, output = [Path(value).resolve() for value in (args.repository, args.target, args.output)]
    audit_bytes = Path(args.checkpoint_audit).read_bytes()
    if sha_bytes(audit_bytes) != args.checkpoint_audit_sha256:
        raise ValueError('Checkpoint audit hash mismatch')
    audit = json.loads(audit_bytes)
    validate_target_audit(audit)
    manifest_bytes = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get('model_id') != 'meta-llama/Llama-3.1-8B-Instruct'
            or manifest.get('model_revision') != META_REVISION
            or manifest.get('tokenizer_revision') != META_REVISION
            or manifest.get('tofu_revision') != TOFU_REVISION
            or any(manifest.get('datasets', {}).get(split, {}).get('sha256') != digest
                   for split, digest in DATA_SHA256.items())
            or manifest.get('readiness', {}).get('status') != 'pass'
            or manifest['readiness'].get('world_size') != 1
            or manifest['readiness'].get('global_batch') != 32
            or sha_bytes(manifest_bytes) != audit.get('manifest_sha256')):
        raise ValueError('Audited frozen Meta construction manifest required')
    if sha_bytes((repo / 'configs/model/Llama-3.1-8B-Instruct.yaml').read_bytes()) != TEMPLATE_SHA256:
        raise ValueError('Pinned teacher-forcing template changed')
    record = json.loads((target / 'target.json').read_text(encoding='utf8'))
    state = json.loads((target / 'trainer_state.json').read_text(encoding='utf8'))
    if (record.get('method') != 'spf' or record.get('split') != 'full' or record.get('mode') != 'full'
            or record.get('construction_seed') != 0 or record.get('manifest_sha256') != audit['manifest_sha256']
            or record.get('reload') != audit['fresh_reload'] or state.get('global_step') != 625
            or state.get('epoch') != 5.0):
        raise ValueError('Target metadata does not match the completed audit')
    shards = sorted(path.name for path in target.glob('*.safetensors'))
    if shards != sorted(entry['name'] for entry in audit['final_checkpoint']['files']):
        raise ValueError('Audited target shard set changed')
    for entry in audit['final_checkpoint']['files']:
        if (target / entry['name']).stat().st_size != entry['size']:
            raise ValueError('Audited target shard size changed')
    identities = {entry['name']: entry['sha256'] for entry in audit['final_checkpoint']['files']}
    for name in ('config.json', 'generation_config.json', 'model.safetensors.index.json',
                 'target.json', 'tokenizer_config.json', 'trainer_state.json'):
        path = target / name
        if path.exists():
            identities[name] = sha_bytes(path.read_bytes())
    identity = sha_bytes(json.dumps(identities, sort_keys=True).encode())
    sources, prepared = {}, {}
    for name in SOURCE_SHA256:
        raw = (repo / 'scripts/reproduction/safety' / name).read_bytes()
        prepared[name], sources[name] = adapt_source(name, raw, target=str(target),
                                                     identity=identity, repository=str(repo))
    output.mkdir(parents=True, exist_ok=False)
    code = output / 'code'
    code.mkdir()
    for name, raw in prepared.items():
        (code / name).write_bytes(raw)
    provenance = dict(status='prepared_not_evaluated', checkpoint=str(target),
                      checkpoint_identity_sha256=identity, checkpoint_files=identities,
                      checkpoint_audit_sha256=args.checkpoint_audit_sha256,
                      weight_byte_hash_verification='delegated to bound completed checkpoint audit',
                      sources=sources, world_size=1, template_sha256=TEMPLATE_SHA256,
                      base_tokenizer_revision=META_REVISION, base_generation_config_revision=META_REVISION,
                      generation_caps=dict(tofu=200, safety=512, ifbench=2048, conversation=1024),
                      tofu_batch_size=8, knowledge_protocol='auxiliary-zero-shot-ABCD-summed-continuation-logprob-v1',
                      external_judge_executed=False, runtime_validation='not_run',
                      target_gate_status='not_evaluated')
    (output / 'adapter-audit.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf8')
    print(json.dumps({key: provenance[key] for key in ('status', 'checkpoint_identity_sha256', 'world_size')}))


if __name__ == '__main__':
    main()
