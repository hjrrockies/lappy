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
import signal
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


def swap_used_mb():
    """System-wide swap in use, MB. macOS only; 0.0 elsewhere."""
    try:
        out = subprocess.run(['sysctl', '-n', 'vm.swapusage'],
                             capture_output=True, text=True, timeout=5).stdout
        # "total = 2048.00M  used = 619.62M  free = 1428.38M  (encrypted)"
        used = out.split('used =')[1].split()[0]
        return float(used.rstrip('MG')) * (1024 if used.endswith('G') else 1)
    except Exception:
        return 0.0


# Guard the *machine*, not the process. macOS refuses to lower RLIMIT_AS or
# RLIMIT_DATA ("current limit exceeds maximum limit") and `ulimit -v` is a
# no-op, so a hard per-process cap is not available. Worse, an RSS-based
# watchdog is actively misleading here: memory is compressed, so a runaway
# `rect_thin` read 4.7GB resident while its real footprint was 59.8GB with
# 40GB of swap. Swap growth is the signal that actually correlates with the
# machine becoming unusable, and it catches every cause at once.
SWAP_LIMIT_MB = float(os.environ.get('LAPPY_RUN_SWAP_MB', '4000'))


def run_one(key, tag, n_basis, n_eigs, timeout, no_sym=False, workers=1,
            seed=0):
    os.makedirs(LOGS, exist_ok=True)
    cmd = [sys.executable, '-m', 'benchmarks.suite.runner', key,
           '--tag', tag, '--workers', str(workers), '--seed', str(seed)]
    if n_basis:
        cmd += ['--n-basis', str(n_basis)]
    if n_eigs:
        cmd += ['--n-eigs', str(n_eigs)]
    if no_sym:
        cmd += ['--no-sym']
    log = os.path.join(LOGS, f'{key}__{tag}.log')
    t0 = time.time()
    baseline = swap_used_mb()
    budget = baseline + SWAP_LIMIT_MB
    rc, note = None, ''
    with open(log, 'w') as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh,
                             stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            try:
                # Poll fast: reg_ngon_8 grew from under budget to 11.5GB of
                # swap inside a single 5s window. The guard still fired, but
                # by then the machine had already taken the hit, which is the
                # thing the guard exists to prevent.
                rc = p.wait(timeout=1.5)
                break
            except subprocess.TimeoutExpired:
                pass
            if time.time() - t0 > timeout:
                note = f'timeout after {timeout}s'
                break
            sw = swap_used_mb()
            if sw > budget:
                note = (f'swap guard: system swap {sw:.0f}MB exceeded '
                        f'baseline+{SWAP_LIMIT_MB:.0f}MB')
                break
        if rc is None:
            # kill the whole process group: manual_solve may have forked
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                p.kill()
            p.wait(timeout=30)
            rc = -9
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
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--retries', type=int, default=2,
                    help='on failure, retry on seed+1, seed+2, ...')
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
        # 'hard' means diagnosed and parked: re-running it each pass only burns
        # time (rect_thin sorts first in the pending list and costs ~340s to
        # fail). Use --redo, or --keys, to revisit one deliberately.
        keys = [k for k in keys
                if q.get(k, {}).get('status') not in ('done', 'hard')]

    print(f'# {len(keys)} domains, tag={args.tag}, timeout={args.timeout}s',
          flush=True)
    for i, key in enumerate(keys, 1):
        ent = q.setdefault(key, dict(status='pending', attempts=0,
                                     best_digits=None, best_tag=None,
                                     best_n_basis=None, notes=''))
        ent['status'] = 'running'
        ent['attempts'] = ent.get('attempts', 0) + 1
        save_queue(q)

        # Retry a failure on a different seed. Interior collocation points are
        # drawn randomly, and an unlucky draw can produce a badly conditioned
        # system whose tension curve is noisy enough to send the bracket search
        # (and its memory) out of control: reg_ngon_6 reaches 12.8 digits on
        # some draws and trips the swap guard on seed 0. Seeding made that
        # deterministic, which is right for reproducibility but means a single
        # bad seed would permanently lose the domain.
        res = None
        for attempt in range(args.retries + 1):
            seed = args.seed + attempt
            tag = args.tag if attempt == 0 else f'{args.tag}s{seed}'
            rc, note, secs = run_one(key, tag, args.n_basis, args.n_eigs,
                                     args.timeout, args.no_sym, args.workers,
                                     seed=seed)
            res = read_result(key, tag) if rc == 0 else None
            if res and res.get('ok'):
                args_tag_used = tag
                break
            if attempt < args.retries:
                print(f'      {key}: seed {seed} failed ({note or rc}), '
                      f'retrying on seed {seed + 1}', flush=True)

        if res and res.get('ok'):
            dig = res['min_digits']
            best = ent.get('best_digits')
            if best is None or dig > best:
                ent.update(best_digits=dig, best_tag=res.get('tag', args.tag),
                           best_n_basis=res['n_basis'],
                           best_seed=res.get('seed'))
            ent['status'] = 'done' if (ent['best_digits'] or 0) >= TARGET_DIGITS \
                else 'short'
            extra = ''
            if 'analytic_min_digits' in res:
                extra = f" analytic={res['analytic_min_digits']:.1f}"
            print(f'[{i}/{len(keys)}] {key:20s} {ent["status"]:6s} '
                  f'digits={dig:5.1f}{extra} nb={res["n_basis"]} {secs:.0f}s',
                  flush=True)
        else:
            ent['status'] = 'failed'
            ent['notes'] = (note or f'rc={rc}')[:200]
            print(f'[{i}/{len(keys)}] {key:20s} FAILED {ent["notes"]} '
                  f'{secs:.0f}s', flush=True)
        save_queue(q)

    return 0


if __name__ == '__main__':
    sys.exit(main())
