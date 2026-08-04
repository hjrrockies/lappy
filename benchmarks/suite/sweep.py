"""Drive `runner.py` over many domains, one subprocess each, with timeouts.

    python -m benchmarks.suite.sweep --tier analytic --timeout 900
    python -m benchmarks.suite.sweep --keys L_shape,H_shape --tag nb480 --n-basis 480
    python -m benchmarks.suite.sweep --all --timeout 1200

Updates `run/queue.json` as it goes, so the run is resumable at any point and a
fresh context can see exactly where things stand. Prints one line per domain.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUN = os.path.join(HERE, 'run')
QUEUE = os.path.join(RUN, 'queue.json')
LOGS = os.path.join(RUN, 'logs')
RESULTS = os.path.join(RUN, 'results')

TARGET_DIGITS = 8.0


def load_queue():
    if os.path.exists(QUEUE):
        with open(QUEUE) as fh:
            return json.load(fh)
    from benchmarks.suite.domains import SUITE
    q = {k: dict(status='pending', attempts=0, best_digits=None,
                 best_tag=None, best_n_basis=None, notes='')
         for k in SUITE}
    save_queue(q)
    return q


def save_queue(q):
    os.makedirs(RUN, exist_ok=True)
    tmp = QUEUE + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(q, fh, indent=1, sort_keys=True)
    os.replace(tmp, QUEUE)   # atomic; a crash mid-write can't corrupt the queue


def run_one(key, tag, n_basis, n_eigs, timeout, no_sym=False, workers=4):
    os.makedirs(LOGS, exist_ok=True)
    cmd = [sys.executable, '-m', 'benchmarks.suite.runner', key,
           '--tag', tag, '--workers', str(workers)]
    if n_basis:
        cmd += ['--n-basis', str(n_basis)]
    if n_eigs:
        cmd += ['--n-eigs', str(n_eigs)]
    if no_sym:
        cmd += ['--no-sym']
    log = os.path.join(LOGS, f'{key}__{tag}.log')
    t0 = time.time()
    with open(log, 'w') as fh:
        try:
            p = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout)
            rc, note = p.returncode, ''
        except subprocess.TimeoutExpired:
            rc, note = -9, f'timeout after {timeout}s'
    return rc, note, time.time() - t0


def read_result(key, tag):
    path = os.path.join(RESULTS, f'{key}__{tag}.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--keys', default=None, help='comma-separated domain keys')
    ap.add_argument('--tier', default=None)
    ap.add_argument('--status', default=None,
                    help='only run domains currently in this queue status')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--tag', default='base')
    ap.add_argument('--n-basis', type=int, default=None)
    ap.add_argument('--n-eigs', type=int, default=None)
    ap.add_argument('--timeout', type=int, default=900)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-sym', action='store_true')
    ap.add_argument('--redo', action='store_true',
                    help='rerun even if already done')
    args = ap.parse_args(argv)

    from benchmarks.suite.domains import SUITE, select

    if args.keys:
        keys = [k.strip() for k in args.keys.split(',')]
    elif args.tier:
        keys = [d.key for d in select(tier=args.tier)]
    elif args.all:
        keys = list(SUITE)
    else:
        ap.error('need --keys, --tier or --all')

    q = load_queue()
    if args.status:
        keys = [k for k in keys if q.get(k, {}).get('status') == args.status]
    if not args.redo:
        keys = [k for k in keys if q.get(k, {}).get('status') != 'done']

    print(f'# {len(keys)} domains, tag={args.tag}, timeout={args.timeout}s')
    for i, key in enumerate(keys, 1):
        ent = q.setdefault(key, dict(status='pending', attempts=0,
                                     best_digits=None, best_tag=None,
                                     best_n_basis=None, notes=''))
        ent['status'] = 'running'
        ent['attempts'] = ent.get('attempts', 0) + 1
        save_queue(q)

        rc, note, secs = run_one(key, args.tag, args.n_basis, args.n_eigs,
                                 args.timeout, args.no_sym, args.workers)
        res = read_result(key, args.tag) if rc == 0 else None

        if res and res.get('ok'):
            dig = res['min_digits']
            best = ent.get('best_digits')
            if best is None or dig > best:
                ent.update(best_digits=dig, best_tag=args.tag,
                           best_n_basis=res['n_basis'])
            ent['status'] = 'done' if (ent['best_digits'] or 0) >= TARGET_DIGITS \
                else 'short'
            extra = ''
            if 'analytic_min_digits' in res:
                extra = f" analytic={res['analytic_min_digits']:.1f}"
            print(f'[{i}/{len(keys)}] {key:20s} {ent["status"]:6s} '
                  f'digits={dig:5.1f}{extra} nb={res["n_basis"]} {secs:.0f}s')
        else:
            ent['status'] = 'failed'
            ent['notes'] = (note or f'rc={rc}')[:200]
            print(f'[{i}/{len(keys)}] {key:20s} FAILED {ent["notes"]} {secs:.0f}s')
        save_queue(q)

    return 0


if __name__ == '__main__':
    sys.exit(main())
