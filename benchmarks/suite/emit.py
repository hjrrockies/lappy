"""Turn the run's JSON results into consumable reference values.

    python -m benchmarks.suite.emit --summary        # digit table, stdout
    python -m benchmarks.suite.emit --write          # write RESULTS.md + values

Two outputs:

* ``run/RESULTS.md`` -- the human table: certified digits, true error where a
  closed form exists, method, timing, and the verdict (good / hard).
* ``run/reference_values.py`` -- a paste-ready module of arrays keyed by domain,
  with the certified error bar recorded alongside each. Intended as the source
  for test fixtures, so tests can assert to the accuracy actually achieved
  rather than to a hopeful constant.

Only results that pass the consistency checks are emitted as reference values;
everything else is listed with the reason it was withheld.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, 'run')
RESULTS = os.path.join(RUN, 'results')

TARGET = 8.0


def load_best():
    """Best (highest certified min_digits) successful result per domain."""
    best = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, '*.json'))):
        with open(path) as fh:
            r = json.load(fh)
        if not r.get('ok'):
            continue
        key = r['key']
        if key not in best or r['min_digits'] > best[key]['min_digits']:
            best[key] = r
    return best


def checks(key, r):
    """Consistency checks. Returns (list_of_problems, list_of_notes)."""
    from benchmarks.suite.domains import SUITE
    problems, notes = [], []
    ent = SUITE.get(key)

    if r['n_found'] < r['n_eigs']:
        problems.append(f"found {r['n_found']} of {r['n_eigs']}")

    eigs = np.asarray(r['eigs'], dtype=float)
    if len(eigs) and (np.any(~np.isfinite(eigs)) or np.any(eigs <= 0)):
        problems.append('non-finite or non-positive eigenvalue')
    if len(eigs) > 1 and np.any(np.diff(eigs) < -1e-12):
        problems.append('eigenvalues not sorted')

    # Weyl completeness: the two-term count at the last eigenvalue should be
    # close to how many we found. Asymptotic, so only a large gap is a signal.
    nw = r.get('weyl_count_at_last')
    if nw is not None and np.isfinite(nw):
        gap = r['n_found'] - nw
        # Threshold was 3, which is too lax: iso_tri_h1 missed exactly one
        # eigenvalue (lambda=98.696, the (4,2) mode) and the Weyl gap of ~1
        # sailed through, while the analytic check caught it at 0.6 digits.
        # A missed eigenvalue is the worst failure mode for a reference table,
        # so flag at 1.5 and accept some false positives from the asymptotics.
        if abs(gap) > 1.5:
            problems.append(f'Weyl mismatch: found {r["n_found"]}, '
                            f'predicted ~{nw:.1f}')
        else:
            notes.append(f'Weyl {nw:.1f}')

    # The strongest check available, where it exists.
    if 'analytic_min_digits' in r:
        a = r['analytic_min_digits']
        notes.append(f'analytic {a:.1f}')
        # a certified bound that claims far more accuracy than the true error
        # shows is a bookkeeping bug, not conservatism
        if a < r['min_digits'] - 2.0:
            problems.append(f'certified {r["min_digits"]:.1f} digits but true '
                            f'error only {a:.1f} -- misaligned table?')
    if ent is not None and ent.digit_ceiling:
        notes.append(f'was {ent.digit_ceiling:.1f}')
    return problems, notes


def summary_rows():
    from benchmarks.suite.domains import SUITE
    best = load_best()
    rows = []
    for key in SUITE:
        r = best.get(key)
        if r is None:
            rows.append((key, None, None, [], ['no successful run']))
            continue
        problems, notes = checks(key, r)
        rows.append((key, r, r['min_digits'], problems, notes))
    return rows


def cmd_summary():
    from benchmarks.suite.domains import SUITE
    rows = summary_rows()
    ok = [x for x in rows if x[1] and not x[3] and x[2] >= TARGET]
    short = [x for x in rows if x[1] and not x[3] and x[2] < TARGET]
    bad = [x for x in rows if x[1] and x[3]]
    none = [x for x in rows if not x[1]]

    print(f'{"domain":22s} {"tier":8s} {"cert":>5} {"true":>5} {"nb":>4} '
          f'{"secs":>5}  notes')
    for key, r, dig, problems, notes in rows:
        if r is None:
            print(f'{key:22s} {SUITE[key].tier:8s} {"--":>5} {"--":>5} '
                  f'{"--":>4} {"--":>5}  {"; ".join(notes)}')
            continue
        true = r.get('analytic_min_digits')
        true_s = f'{true:5.1f}' if true is not None else '   --'
        flag = '' if not problems else '  !! ' + '; '.join(problems)
        print(f'{key:22s} {SUITE[key].tier:8s} {dig:5.1f} {true_s} '
              f'{r["n_basis"]:4d} {r["seconds"]:5.0f}  {"; ".join(notes)}{flag}')

    print(f'\n>= {TARGET:.0f} digits and clean : {len(ok)}')
    print(f'below target             : {len(short)}'
          + (f'  ({", ".join(k for k, *_ in short)})' if short else ''))
    print(f'flagged by checks        : {len(bad)}'
          + (f'  ({", ".join(k for k, *_ in bad)})' if bad else ''))
    print(f'no successful run        : {len(none)}'
          + (f'  ({", ".join(k for k, *_ in none)})' if none else ''))
    return 0


def cmd_write():
    from benchmarks.suite.domains import SUITE
    rows = summary_rows()

    lines = ['"""Reference Dirichlet eigenvalues from the benchmark suite run.',
             '',
             'GENERATED by `python -m benchmarks.suite.emit --write`. Do not edit.',
             '',
             'Each entry carries the certified Moler--Payne relative error bound',
             'actually achieved, so tests can assert to the accuracy that was',
             'demonstrated rather than to a hopeful constant. `certified_digits`',
             'is -log10 of the worst relative bound over the listed eigenvalues.',
             '"""',
             'import numpy as np', '', 'REFERENCE = {']
    emitted, withheld = 0, []
    for key, r, dig, problems, notes in rows:
        if r is None or problems or dig < TARGET:
            withheld.append((key, problems or (['no run'] if r is None
                                               else [f'{dig:.1f} digits'])))
            continue
        vals = ', '.join(f'{x:.15g}' for x in r['eigs'])
        lines += [f"    {key!r}: dict(",
                  f"        eigs=np.array([{vals}]),",
                  f"        certified_digits={dig:.2f},",
                  f"        n_basis={r['n_basis']}, method={r['method']!r},",
                  f"    ),"]
        emitted += 1
    lines += ['}', '']
    if withheld:
        lines += ['# Withheld (did not meet the 8-digit bar or failed a check):']
        lines += [f'#   {k}: {"; ".join(p)}' for k, p in withheld]
        lines += ['']
    out_py = os.path.join(RUN, 'reference_values.py')
    with open(out_py, 'w') as fh:
        fh.write('\n'.join(lines))

    md = ['# Reference run results', '',
          f'{emitted} domains met the 8-digit bar; {len(withheld)} withheld.',
          '', '| domain | tier | certified digits | vs exact | n_basis | s |',
          '|---|---|---|---|---|---|']
    for key, r, dig, problems, notes in rows:
        if r is None:
            md.append(f'| `{key}` | {SUITE[key].tier} | — | — | — | — |')
            continue
        t = r.get('analytic_min_digits')
        md.append(f'| `{key}` | {SUITE[key].tier} | {dig:.1f} | '
                  f'{t:.1f} | {r["n_basis"]} | {r["seconds"]:.0f} |'
                  if t is not None else
                  f'| `{key}` | {SUITE[key].tier} | {dig:.1f} | — | '
                  f'{r["n_basis"]} | {r["seconds"]:.0f} |')
    out_md = os.path.join(RUN, 'RESULTS.md')
    with open(out_md, 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    print(f'wrote {out_py} ({emitted} domains) and {out_md}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args(argv)
    return cmd_write() if args.write else cmd_summary()


if __name__ == '__main__':
    sys.exit(main())
