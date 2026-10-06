"""Read-only export of one completed smoke model; weights are verified, not sent."""
import base64, hashlib, io, json, os, struct, zipfile
from pathlib import Path

task = Path(os.environ.get('NPO_EVIDENCE_TASK', '/data/npo-gb200-20261004/model-validation-20261006-r1'))
short = os.environ.get('NPO_EVIDENCE_MODEL', 'llama31_8b')
model = {'llama31_8b': 'Llama-3.1-8B-Instruct', 'llama2_7b': 'Llama-2-7b-chat-hf'}[short]
cell = task / 'runs' / (short + '_forget01_s0_native_smoke4')
checkpoint = cell / 'checkpoint'
assert (cell / 'eval/TOFU_SUMMARY.json').exists()
assets_file = task / ('assets_' + model + '.json')
assets = json.loads(assets_file.read_text())
config = json.loads((checkpoint / 'config.json').read_text())
index_file = checkpoint / 'model.safetensors.index.json'
index = json.loads(index_file.read_text()) if index_file.exists() else None
shards = sorted(set(index['weight_map'].values())) if index else ['model.safetensors']
all_tensors, shard_proofs, total_payload = {}, [], 0
for shard in shards:
    path = checkpoint / shard
    with path.open('rb') as stream:
        header_length = struct.unpack('<Q', stream.read(8))[0]
        assert header_length < 100_000_000
        header = json.loads(stream.read(header_length))
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    cursor = 0
    for name, tensor in sorted(tensors.items(), key=lambda item: item[1]['data_offsets']):
        start, end = tensor['data_offsets']
        assert start == cursor and end >= start, (shard, name)
        bits = {'BF16': 16, 'F16': 16, 'F32': 32, 'F64': 64, 'I64': 64, 'I32': 32}[tensor['dtype']]
        elements = 1
        for dim in tensor['shape']:
            elements *= dim
        assert end - start == elements * bits // 8
        assert name not in all_tensors
        all_tensors[name] = shard
        cursor = end
    assert 8 + header_length + cursor == path.stat().st_size
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    total_payload += cursor
    shard_proofs.append({'file': shard, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest(), 'tensor_count': len(tensors)})
if index:
    assert all_tensors == index['weight_map']
    assert total_payload == index['metadata']['total_size']
required = ['model.embed_tokens.weight', 'model.norm.weight']
if not config.get('tie_word_embeddings', False):
    required.append('lm_head.weight')
for layer in range(config['num_hidden_layers']):
    required.extend('model.layers.%d.%s.weight' % (layer, suffix) for suffix in [
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
        'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'input_layernorm', 'post_attention_layernorm'])
assert all(name in all_tensors for name in required)
proof = {'checkpoint': str(checkpoint), 'model': model, 'num_hidden_layers': config['num_hidden_layers'],
         'tensor_count': len(all_tensors), 'weights_bytes': sum(x['bytes'] for x in shard_proofs),
         'payload_bytes': total_payload, 'shards': shard_proofs, 'header_and_required_layers_complete': True,
         'counts': json.loads((checkpoint / 'runtime_final_rank0.json').read_text())}
reference = Path(assets['retain_path'])
assert hashlib.sha256(reference.read_bytes()).hexdigest() == assets['retain_sha256']
files = {assets_file: assets_file.name, reference: 'reference/retain99_TOFU_EVAL.json'}
for path in list(task.glob(short + '*')) + list(cell.rglob('*')) + list((task / 'as-run').glob('*')):
    if path.is_file() and path.suffix in ('.json', '.yaml', '.log', '.py') and path.stat().st_size < 2_000_000:
        # Tokenizer bytes are model assets, not necessary runtime evidence.
        if path.name not in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
            files[path] = str(path.relative_to(task)).replace('\\', '/')
previous = task.with_name('model-validation-20261006')
for path in previous.glob('*'):
    if path.is_file() and ('metadata' in path.name or 'preflight' in path.name) and path.suffix in ('.json', '.log'):
        files[path] = 'pre-recovery/' + path.name
if short == 'llama2_7b':
    # Final queue evidence is immutable once DONE; do not recollect 8B weights.
    for path in task.iterdir():
        if path.is_file() and path.suffix in ('.json', '.log') and not path.name.startswith('llama31_8b') and path.stat().st_size < 2_000_000:
            files[path] = str(path.relative_to(task))
    root = task.parent
    for directory in (previous, task):
        for path in directory.rglob('*'):
            if path.is_file() and any('metadata' in part or 'preflight' in part for part in path.relative_to(directory).parts) and path.suffix in ('.json', '.log') and path.stat().st_size < 2_000_000:
                files[path] = 'recovery-evidence/' + directory.name + '/' + str(path.relative_to(directory))
    for path in root.glob('*metadata*'):
        if path.is_file() and path.suffix in ('.json', '.log'):
            files[path] = 'recovery-evidence/root/' + path.name
    source = root / 'repo/scripts/reproduction/npo/gb200/gradient_audit.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == '2d13942540ad74d21804bf06c6e40219f675a9b0605655f9fac26ad18de0d3fb'
    files[source] = 'as-run/gradient_audit.py'
buffer, manifest = io.BytesIO(), {}
with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path, name in sorted(files.items(), key=lambda item: item[1]):
        content = path.read_bytes()
        archive.writestr(name, content)
        manifest[name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    archive.writestr('checkpoint-proof.json', json.dumps(proof, indent=2))
    archive.writestr('evidence-manifest.json', json.dumps(manifest, indent=2))
archive_bytes = buffer.getvalue()
print(json.dumps({'archive_base64': base64.b64encode(archive_bytes).decode(),
                  'archive_sha256': hashlib.sha256(archive_bytes).hexdigest(), 'files': len(files)}))
