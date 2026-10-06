import copy
import pytest
from construction.profile import accumulation
from construction.protocol import validate_manifest, ARCHIVE_SHA256

@pytest.mark.parametrize('world,micro,expected',[(1,4,8),(1,2,16),(1,1,32),(2,4,4)])
def test_global32(world,micro,expected):
    assert accumulation(micro,world)==expected

@pytest.mark.parametrize('micro,world',[(3,1),(4,3),(0,1)])
def test_nonintegral_or_unsupported(micro,world):
    with pytest.raises(ValueError): accumulation(micro,world)

def test_old_dual_readiness_cannot_masquerade_as_world1():
    manifest={'schema_version':1,'archive_sha256':ARCHIVE_SHA256,'selection_source':'development',
              **{k:'a'*40 for k in ('model_revision','tokenizer_revision','tofu_revision','repository_commit')},
              'audit':{'status':'pass'},'readiness':{'status':'pass','microbatch':4,'accumulation':4,'world_size':1}}
    with pytest.raises(ValueError,match='profile'):validate_manifest(manifest,full=True)
    manifest['readiness']['accumulation']=8
    validate_manifest(manifest,full=True)
