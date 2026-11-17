#!/usr/bin/env python3
"""Regenerate frontend/public/models/avatar/manifest.json.

Keeps the legacy `animationUrl` (VRM path expects it) and adds:
  clip         /models/avatar/clips/<gloss>.json   or null when no clip ships
  source       "retarget" | "authored-approximation" | null
  samplesUsed  number of recorded samples behind the clip (0 for authored)
  needsReview  true for single-sample / authored clips
Glosses without a clip keep `clip: null`; the runtime must show a
"no clip yet" state instead of falling back to the generic demo.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLIPS = REPO / 'frontend/public/models/avatar/clips'
MANIFEST = REPO / 'frontend/public/models/avatar/manifest.json'

LABELS = {
    'HELLO': 'Hello', 'THANK_YOU': 'Thank you', 'SORRY': 'Sorry', 'PLEASE': 'Please',
    'YES': 'Yes', 'NO': 'No', 'HELP': 'Help me', 'WATER': 'Water', 'FOOD': 'Food',
    'NAME': 'Name', 'YOU': 'You', 'ME': 'Me', 'WHERE': 'Where', 'HOW': 'How',
}
NO_CLIP_REASON = {
    'SORRY': 'no clip yet — only 2 poor one-handed captures (58–66% frame presence); do not ship',
    'YES': 'no clip yet — 1 poor single-signer capture (75/12% hand presence)',
    'NO': 'no clip yet — 1 poor single-signer capture (79/1% hand presence)',
    'HELP': 'no clip yet — 1 poor single-signer capture (42/17% hand presence)',
    'WATER': 'no clip yet — 1 poor single-signer capture (2/53% hand presence)',
    'ME': 'no clip yet — 2 poor single-signer captures (≤40% one-hand presence)',
    'YOU': 'no clip yet — 1 poor single-signer capture (2/37% hand presence)',
}

clip_meta = {}
for f in CLIPS.glob('*.json'):
    data = json.loads(f.read_text())
    meta = data.get('meta', {})
    clip_meta[data['gloss']] = {
        'clip': f'/models/avatar/clips/{f.name}',
        'source': meta.get('source', 'retarget'),
        'authored': bool(meta.get('authored', False)),
        'samplesUsed': meta.get('samplesUsed', 0),
        'needsReview': bool(meta.get('needsReview', False)),
        'durationSeconds': round(data.get('duration', 0), 3),
    }

manifest = {}
for gloss, label in LABELS.items():
    legacy = f'/models/avatar/signs/{gloss.lower()}.glb'
    entry = {
        'id': gloss,
        'label': label,
        'animationUrl': legacy,  # VRM controller path (unchanged)
    }
    entry.update(clip_meta.get(gloss, {
        'clip': None,
        'source': None,
        'authored': False,
        'samplesUsed': 0,
        'needsReview': True,
    }))
    if entry['clip'] is None and gloss in NO_CLIP_REASON:
        entry['noClipReason'] = NO_CLIP_REASON[gloss]
    manifest[gloss] = entry

MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=1))
