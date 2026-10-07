"""CPU-only actual native tokenizer cache mapping; no weights or generation."""
import pathlib,hashlib,os,json,sys
checks={'/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/huggingface_hub/file_download.py': '2ee4e4898858191f67bbcd3667c4ceb3a699cacfecefc5ab69c07e638ab7ca58', '/data/npo-gb200-20261004/validation-20261006/venv/lib/python3.12/site-packages/transformers/utils/hub.py': 'f9ed705661d670f15a5b88840c1bd10f668c9ada2a510e03e74601768cd7a845', '/data/spf-development-20261007-r1/repo/src/model/__init__.py': '894e36a8471b3346ef672db896f51f40000270167ed70b955cec80351534a252'}
for name,expected in checks.items(): assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==expected
sys.path.insert(0,'/data/spf-development-20261007-r1/repo/src')
import torch
from omegaconf import OmegaConf
from model import get_tokenizer,hf_home
from transformers import AutoTokenizer
from huggingface_hub import constants
cache='/data/smart-mfg/jimmy-lin/hf_cache/hub'
assert hf_home==cache and constants.HF_HUB_CACHE==cache and constants.HF_HUB_OFFLINE
cfg=dict(pretrained_model_name_or_path='meta-llama/Llama-3.1-8B-Instruct',revision='0e9e39f249a16976918f6564b8830bc894c89659',local_files_only=True)
try: AutoTokenizer.from_pretrained(**cfg,cache_dir='/data/smart-mfg/jimmy-lin/hf_cache')
except (OSError,RuntimeError): negative=True
else: raise AssertionError('Old cache path unexpectedly resolved')
tok=get_tokenizer(OmegaConf.create(cfg)); direct=AutoTokenizer.from_pretrained(**cfg,cache_dir=cache)
assert tok.get_vocab()==direct.get_vocab() and tok.chat_template==direct.chat_template
assert tok.eos_token_id==direct.eos_token_id and tok.pad_token_id==tok.eos_token_id
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='NATIVE_OFFLINE_TOKENIZER_CACHE_MAPPING_PASS',native_source_sha256=checks['/data/spf-development-20261007-r1/repo/src/model/__init__.py'],cache_dir=hf_home,hf_hub_cache=constants.HF_HUB_CACHE,model_id=cfg['pretrained_model_name_or_path'],revision=cfg['revision'],vocab_size=len(tok),eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,old_cache_path_rejected=negative,exact_vocab_and_template=True,cuda_initialized=False,weights_read=False,generation_calls=0,npo_training_executed=False)))
