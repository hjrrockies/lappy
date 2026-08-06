"""Re-bucket many domains, one subprocess each, against the recorded baseline.

    python -m benchmarks.suite.bucket_sweep --all --tag orth
    python -m benchmarks.suite.bucket_sweep --keys stadium,mushroom --tag orth

`bucket.py` handles one domain; this drives it over the suite and prints the
comparison. The baseline for each domain is its **best previous record** in
`run/buckets.jsonl` (lowest bucket, then most digits), which is what BUCKETS.md
tabulates -- and, importantly, the `n_basis` that produced it, since `n_basis`
has a domain-specific optimum that is not monotone (BUCKETS.md, corner tier).
Re-running at `entry.n_basis` instead would compare two different experiments.

Guards, both needed (see `guards.py` and `sweep.py`):

* in-process (`bucket.py --timeout/--swap-mb`) aborts cleanly and records;
* out-of-process here kills the process group if the child stops responding to
  its own guard -- a solve that is thrashing may never reach the watchdog's next
  poll.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN = os.path.join(HERE, 'run')
JSONL = os.path.join(RUN, 'buckets.jsonl')
LOGS = os.path.join(RUN, 'logs')

# Slack over the in-process timeout, so the child's own guard normally wins and
# writes a record; this one is the backstop.
OUTER_SLACK_S = 120.0


def baseline(exclude_tags=()):
    """Best previous record per domain: lowest bucket, then most digits."""
    best = {}
    if not os.path.exists(JSONL):
        return best
    with open(JSONL) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('tag', '') in exclude_tags:
                continue
            key = r['key']
            if key not in best or _rank(r) < _rank(best[key]):
                best[key] = r
    return best


def _rank(r):
    """Sort key: bucket first, then digits (true where known, else certified)."""
    dig = r.get('true_digits')
    if dig is None:
        dig = r.get('certified')
    return (r.get('bucket', 9), -(dig if dig is not None else -99))


def digits_of(r):
    if r is None:
        return None, ''
    if r.get('true_digits') is not None:
        return r['true_digits'], 'true'
    if r.get('certified') is not None:
        return r['certified'], 'cert'
    return None, ''


def run_one(key, n_basis, tag, timeout, swap_mb, seed=0, extra=()):
    os.makedirs(LOGS, exist_ok=True)
    cmd = [sys.executable, '-m', 'benchmarks.suite.bucket', key,
           '--tag', tag, '--seed', str(seed), '--timeout', str(timeout),
           '--swap-mb', str(swap_mb), *extra]
    if n_basis:
        cmd += ['--n-basis', str(n_basis)]
    log = os.path.join(LOGS, f'{key}__{tag}.bucket.log')
    t0 = time.time()
    with open(log, 'w') as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh,
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout + OUTER_SLACK_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                p.kill()
            p.wait(timeout=30)
            rc = -9
    return rc, time.time() - t0, log


def latest(key, tag):
    """Last record written for (key, tag)."""
    out = None
    with open(JSONL) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('key') == key and r.get('tag') == tag:
                out = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default=None)
    ap.add_argument('--tier', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--tag', default='orth')
    ap.add_argument('--timeout', type=float, default=1200)
    ap.add_argument('--swap-mb', type=float, default=3500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-basis', type=int, default=None,
                    help='override; default is the baseline record\'s n_basis')
    ap.add_argument('--no-orthonorm', action='store_true',
                    help='pass through to bucket.py (A/B against cubature)')
    ap.add_argument('--out', default=None, help='write a JSON comparison table')
    args = ap.parse_args(argv)

    sys.path.insert(0, ROOT)
    from benchmarks.suite.domains import SUITE, select

    if args.keys:
        keys = [k.strip() for k in args.keys.split(',')]
    elif args.tier:
        keys = [d.key for d in select(tier=args.tier)]
    elif args.all:
        keys = list(SUITE)
    else:
        ap.error('need --keys, --tier or --all')

    # 'orthtest' was the smoke test of the boundary-norm certification path;
    # like the sweep's own tag it is a *new*-path record, not a baseline.
    base = baseline(exclude_tags=(args.tag, 'orthtest'))
    extra = ['--no-orthonorm'] if args.no_orthonorm else []
    rows = []
    print(f'# {len(keys)} domains, tag={args.tag}, timeout={args.timeout:.0f}s',
          flush=True)
    for i, key in enumerate(keys, 1):
        b = base.get(key)
        nb = args.n_basis or (b or {}).get('n_basis') or SUITE[key].n_basis
        rc, secs, log = run_one(key, nb, args.tag, args.timeout, args.swap_mb,
                                seed=args.seed, extra=extra)
        new = latest(key, args.tag)
        bd, bsrc = digits_of(b)
        nd, nsrc = digits_of(new)
        bbk = (b or {}).get('bucket')
        nbk = (new or {}).get('bucket')
        delta = (nd - bd) if (nd is not None and bd is not None) else None
        flag = ''
        if bbk is not None and nbk is not None and nbk != bbk:
            flag = f'  BUCKET {bbk}->{nbk}'
        elif new is None:
            flag = f'  NO RECORD (rc={rc})'
        rows.append(dict(key=key, n_basis=nb, rc=rc, seconds=secs,
                         base_bucket=bbk, new_bucket=nbk,
                         base_digits=bd, new_digits=nd, digit_source=nsrc,
                         base_seconds=(b or {}).get('seconds'),
                         l2_methods=(new or {}).get('l2_methods'),
                         l2_spread_max=(new or {}).get('l2_spread_max'),
                         bq_nodes=(new or {}).get('bq_nodes'),
                         bq_precision=(new or {}).get('bq_precision'),
                         error=(new or {}).get('error')))
        print(f'[{i}/{len(keys)}] {key:20s} nb={nb or 0:4d} '
              f'bucket {bbk}->{nbk}  '
              f'digits {bd if bd is None else round(bd, 2)} -> '
              f'{nd if nd is None else round(nd, 2)} '
              f'({nsrc}, d={delta if delta is None else round(delta, 2)})  '
              f'{secs:.0f}s vs {(b or {}).get("seconds") or float("nan"):.0f}s'
              f'{flag}', flush=True)

    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(rows, fh, indent=1)
        print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
