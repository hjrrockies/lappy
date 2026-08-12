"""Append-only storage for the basis-knob program: one JSONL row per build, plus domain cards.

WHY JSONL AND NOT NPZ. The archived harnesses (`benchmarks/archive/new_suite`,
`benchmarks/archive/benchmark_suite`) stored `np.savez` dumps keyed by sweep axis. That works
while the axes are fixed and fails the moment a record needs a null knob, a verbatim warning
string, or a field that did not exist when the first row was written -- which is every row here.
A row must answer "what was held fixed?" on its own, months later, without re-running anything.

TWO RULES THAT MAKE THAT TRUE:

* Every knob field is present on every row, null where the family ignores it, so a missing knob
  is never confused with a defaulted one. `probe.KNOB_FIELDS` is the authority; `append` checks.
* A row points at an immutable DOMAIN CARD by id. The card freezes `lam_star`, its provenance,
  the off-eigenvalues, the probe grid, the reference floor and the censor level. Rows never
  recompute those, so a later change of reference cannot silently retro-fit old rows -- it
  creates a new card and the old rows stay attributable to the old one.

RESUMABILITY keys on `record_id`, a hash of the full knob dict plus the card id plus the
collocation and solver settings. Not an index: adding a value to a sweep mid-run must not
invalidate or renumber prior rows. A re-run that produces an already-present `record_id` is
skipped by `pending`, so an interrupted sweep resumes by simply being restarted.
"""
import hashlib
import json
import os
import time

RUN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run')
CARDS = os.path.join(RUN_DIR, 'cards.jsonl')

# Fields that identify a measurement setup. Anything here changing means a different row.
_ID_FIELDS = ('domain_card_id', 'family', 'n_requested', 'fs_frac', 'fb_strategy',
              'fs_placement', 'fs_d', 'fs_order', 'fs_spacing', 'fs_C', 'fs_sigma',
              'check_exterior', 'colloc_mode', 'bdry_mult', 'int_ratio', 'n_int',
              'n_bdry_total', 'rtol', 'ttol', 'seed', 'stage')


def _ensure_dir():
    os.makedirs(RUN_DIR, exist_ok=True)


def domain_path(domain_key):
    return os.path.join(RUN_DIR, f'{domain_key}.jsonl')


def record_id(rec):
    """Stable hash of the identifying fields. Floats are formatted, not repr'd, so that a value
    that round-trips through JSON hashes the same on reload."""
    parts = []
    for k in _ID_FIELDS:
        v = rec.get(k)
        parts.append(f'{k}=' + ('null' if v is None else
                                (f'{v:.12g}' if isinstance(v, float) else str(v))))
    return hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16]


def _append_line(path, obj):
    _ensure_dir()
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as fh:
        fh.write(json.dumps(obj, sort_keys=True, default=float) + '\n')
    with open(tmp) as fh:
        line = fh.read()
    with open(path, 'a') as fh:          # append is atomic enough for single-line writes
        fh.write(line)
    os.unlink(tmp)


def put_card(card):
    """Freeze a domain card. Returns its id. Re-putting an identical card is a no-op."""
    body = {k: v for k, v in card.items() if k != 'card_id'}
    cid = hashlib.sha1(json.dumps(body, sort_keys=True, default=float).encode()).hexdigest()[:12]
    for existing in load_cards():
        if existing.get('card_id') == cid:
            return cid
    out = dict(body, card_id=cid, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    _append_line(CARDS, out)
    return cid


def load_cards():
    if not os.path.exists(CARDS):
        return []
    with open(CARDS) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def get_card(card_id):
    for c in load_cards():
        if c.get('card_id') == card_id:
            return c
    raise KeyError(f'no domain card {card_id!r}')


def append(domain_key, rec, knob_fields):
    """Write one row. Fails loudly if a knob field is absent -- absence is never a default."""
    missing = [k for k in knob_fields if k not in rec]
    if missing:
        raise KeyError(f'record is missing knob fields {missing}; null them explicitly')
    if 'domain_card_id' not in rec:
        raise KeyError('record must carry domain_card_id')
    out = dict(rec)
    out.setdefault('domain_key', domain_key)
    out['record_id'] = record_id(out)
    out.setdefault('created_utc', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    _append_line(domain_path(domain_key), out)
    return out['record_id']


def load(domain_key):
    path = domain_path(domain_key)
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def seen_ids(domain_key):
    return {r.get('record_id') for r in load(domain_key)}


def pending(domain_key, planned):
    """Which of `planned` (dicts carrying the identifying fields) have no row yet."""
    have = seen_ids(domain_key)
    return [p for p in planned if record_id(p) not in have]
