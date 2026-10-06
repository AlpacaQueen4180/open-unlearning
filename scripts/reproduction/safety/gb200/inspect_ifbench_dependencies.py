"""Read distribution metadata from the historical CPU scorer; do not score."""
import datetime
import importlib.metadata as metadata
import json
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement

roots = ['absl-py', 'langdetect', 'nltk', 'immutabledict', 'spacy', 'emoji',
         'syllapy', 'pydantic', 'pydantic-settings', 'packaging']
environment = default_environment()
environment['extra'] = ''
pending = list(roots)
distributions = {}
missing = []
while pending:
    requested = pending.pop(0)
    normalized = requested.lower().replace('_', '-').replace('.', '-')
    if normalized in distributions or normalized in missing:
        continue
    try:
        dist = metadata.distribution(requested)
    except metadata.PackageNotFoundError:
        missing.append(normalized)
        continue
    active = []
    for raw in dist.requires or []:
        requirement = Requirement(raw)
        if requirement.marker is None or requirement.marker.evaluate(environment):
            active.append(raw)
            pending.append(requirement.name)
    distributions[normalized] = {
        'name': dist.metadata['Name'], 'version': dist.version,
        'requires_python': dist.metadata.get('Requires-Python'),
        'active_requires_dist': active,
    }
print(json.dumps({
    'scope': 'Read-only installed distribution metadata; no scorer import or evaluation',
    'queried_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'python_version': sys.version.split()[0],
    'interpreter': sys.executable,
    'marker_environment': environment,
    'roots': roots,
    'distributions': dict(sorted(distributions.items())),
    'missing': sorted(missing),
    'scoring_executed': False,
    'runai_query_performed': False,
    'credentials_read': False,
}, sort_keys=True))
