"""Build/update the JSON cubature-rule registry at lappy/data/cubature_rules/.

Usage:
    python scripts/build_cubature_registry.py

Reads rule data from lappy.cubature_rules.build_cubature_rules() (the legacy
hardcoded-literal source), computes diagnostics for each (kind, degree) rule,
and writes one JSON file per family to lappy/data/cubature_rules/.

This is also the tool to use when adding a new rule family: add nodes/weights
to a `rules[kind][deg]` dict (an (n,4) array of [bary1, bary2, bary3, weight]
rows) below, or extend build_cubature_rules() in the legacy module, and rerun.
"""
import json
import os

from cubature_common import compute_diagnostics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'lappy', 'data', 'cubature_rules')

SOURCE = "https://www.math.unipd.it/~alvise/SETS_CUBATURE_TRIANGLE/rules_triangle.html"


def main():
    from lappy.cubature_rules import build_cubature_rules

    os.makedirs(DATA_DIR, exist_ok=True)
    rules = build_cubature_rules()

    for kind in sorted(rules):
        print(f"{kind}:")
        family = {
            'kind': kind,
            'source': SOURCE,
            'rules': {},
        }
        for deg in sorted(rules[kind]):
            arr = rules[kind][deg]
            print(f"  degree {deg}: {len(arr)} pts")
            family['rules'][str(deg)] = compute_diagnostics(arr, deg)
        out_path = os.path.join(DATA_DIR, f"{kind}.json")
        with open(out_path, 'w') as f:
            json.dump(family, f, indent=2)


if __name__ == '__main__':
    main()
