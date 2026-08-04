"""Compact run status. The first thing a fresh context should run.

    python -m benchmarks.suite.status          # summary + per-domain state
    python -m benchmarks.suite.status --short  # just the counts and what's next

Reads `run/queue.json`. Output is deliberately small enough to paste into a
context window without crowding it out.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(HERE, 'run', 'queue.json')
ORDER = ('done', 'short', 'hard', 'running', 'failed', 'pending')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--short', action='store_true')
    args = ap.parse_args(argv)

    if not os.path.exists(QUEUE):
        print('no queue yet; run: python -m benchmarks.suite.sweep --all')
        return 0
    with open(QUEUE) as fh:
        q = json.load(fh)

    from benchmarks.suite.domains import SUITE

    counts = {s: 0 for s in ORDER}
    for e in q.values():
        counts[e.get('status', 'pending')] = counts.get(e.get('status'), 0) + 1
    total = len(q)
    print(f'{total} domains: ' + '  '.join(
        f'{s}={counts.get(s, 0)}' for s in ORDER if counts.get(s)))

    if not args.short:
        for s in ORDER:
            rows = [(k, e) for k, e in sorted(q.items())
                    if e.get('status') == s]
            if not rows:
                continue
            print(f'\n-- {s} ({len(rows)})')
            for k, e in rows:
                d = e.get('best_digits')
                d = f'{d:5.1f}' if d is not None else '  -- '
                nb = e.get('best_n_basis') or ''
                tier = SUITE[k].tier if k in SUITE else '?'
                note = (e.get('notes') or '')[:60]
                print(f'  {k:22s} {tier:8s} digits={d} nb={nb:<4} {note}')

    nxt = [k for k, e in sorted(q.items()) if e.get('status') == 'pending']
    if nxt:
        print(f'\nnext pending: {", ".join(nxt[:8])}'
              + (' ...' if len(nxt) > 8 else ''))
    short = [k for k, e in sorted(q.items()) if e.get('status') == 'short']
    if short:
        print(f'below 8 digits, need work: {", ".join(short)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
