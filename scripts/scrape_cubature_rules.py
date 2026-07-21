"""Scrape additional cubature rules from the unipd rule library into the JSON registry.

Usage:
    python scripts/scrape_cubature_rules.py

Fetches barycentric-coordinate .m files from
https://www.math.unipd.it/~alvise/SETS_CUBATURE_TRIANGLE/, parses every
(degree -> (n,4) [bary1,bary2,bary3,weight]) rule in each file, computes the
same diagnostics as build_cubature_registry.py, and keeps only degrees whose
weights are all positive and whose nodes lie inside the closed reference
triangle. Writes/overwrites one lappy/data/cubature_rules/<kind>.json per
family covered by FAMILY_URLS below.

Re-run this script to re-sync if the source page adds/corrects rules, or to
add a new family (add one entry to FAMILY_URLS).

After adding/changing rules, also re-run scripts/precompute_capacity.py so the
new/changed entries get their plane-wave capacity calibration persisted too
(otherwise lappy.cubature recomputes it, once, the first time each new rule
is used at runtime).
"""
import json
import os
import re
import urllib.request

from cubature_common import compute_diagnostics
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'lappy', 'data', 'cubature_rules')

BASE = "https://www.math.unipd.it/~alvise/SETS_CUBATURE_TRIANGLE"

# New families (not previously in the registry).
NEW_FAMILY_URLS = {
    'hammer_marlowe_stroud1': f"{BASE}/hammer_marlowe_stroud/I/set_hammer_marlowe_stroud_I_barycentric.m",
    'hammer_marlowe_stroud2': f"{BASE}/hammer_marlowe_stroud/II/set_hammer_marlowe_stroud_II_barycentric.m",
    'hammer_stroud': f"{BASE}/hammer_stroud/set_hammer_stroud_barycentric.m",
    'hillion1': f"{BASE}/hillion/I/set_hillion_I_barycentric.m",
    'hillion2': f"{BASE}/hillion/II/set_hillion_II_barycentric.m",
    'hillion3': f"{BASE}/hillion/III/set_hillion_III_barycentric.m",
    'hillion4': f"{BASE}/hillion/IV/set_hillion_IV_barycentric.m",
    'hillion5': f"{BASE}/hillion/V/set_hillion_V_barycentric.m",
    'laurie': f"{BASE}/laurie/set_laurie_barycentric.m",
    'laursen_gellert1': f"{BASE}/laursen_gellert/I/set_laursen_gellert_I_barycentric.m",
    'laursen_gellert2': f"{BASE}/laursen_gellert/II/set_laursen_gellert_II_barycentric.m",
    'lyness_jespersen1': f"{BASE}/lyness_jespersen/I_C/set_lyness_jespersen_I_C_barycentric.m",
    'lyness_jespersen2': f"{BASE}/lyness_jespersen/II_C/set_lyness_jespersen_II_C_barycentric.m",
    'lyness_jespersen3': f"{BASE}/lyness_jespersen/III/set_lyness_jespersen_III_barycentric.m",
    'papanicolopulos_a1': f"{BASE}/papanicolopulos_A/I/set_papanicolopulos_A_I_barycentric.m",
    'papanicolopulos_a2': f"{BASE}/papanicolopulos_A/II/set_papanicolopulos_A_II_barycentric.m",
    'papanicolopulos_a3': f"{BASE}/papanicolopulos_A/III_C/set_papanicolopulos_A_III_C_barycentric.m",
    'papanicolopulos_a4': f"{BASE}/papanicolopulos_A/IV_C/set_papanicolopulos_A_IV_C_barycentric.m",
    'papanicolopulos_c1': f"{BASE}/papanicolopulos_C/I/set_papanicolopulos_C_I_barycentric.m",
    'papanicolopulos_c2': f"{BASE}/papanicolopulos_C/II/set_papanicolopulos_C_II_barycentric.m",
    'papanicolopulos_c3': f"{BASE}/papanicolopulos_C/III_C/set_papanicolopulos_C_III_C_barycentric.m",
    'papanicolopulos_c4': f"{BASE}/papanicolopulos_C/IV/set_papanicolopulos_C_IV_barycentric.m",
    'papanicolopulos_c5': f"{BASE}/papanicolopulos_C/V/set_papanicolopulos_C_V_barycentric.m",
    'radon': f"{BASE}/radon/set_radon_barycentric.m",
    'taylor': f"{BASE}/taylor/set_taylor_barycentric.m",
    'taylor_wingate_bos': f"{BASE}/taylor_wingate_bos/C/set_taylor_wingate_bos_C_barycentric.m",
    # Corrected variant's link on the source page is dead (404); original file only.
    'taylor_wingate_bos_siam': f"{BASE}/taylor_wingate_bos_siam/set_taylor_wingate_bos_siam_barycentric.m",
    'walkington': f"{BASE}/walkington/set_walkington_barycentric.m",
    'wandzura_xiao': f"{BASE}/wandzura_xiao/set_wandzura_xiao_barycentric.m",
    'williams_shunn_jameson': f"{BASE}/williams_shunn_jameson/set_williams_shunn_jameson_barycentric.m",
    'witherden_vincent': f"{BASE}/witherden_vincent/set_witherden_vincent_barycentric.m",
    'zhang_cui_liu': f"{BASE}/zhang_cui_liu/set_zhang_cui_liu_barycentric.m",
}

# Existing families being refreshed with superseding "[Corrected]" source data.
REFRESH_FAMILY_URLS = {
    'bern_esp1': f"{BASE}/berntsen_espelid_I/I_C/set_berntsen_espelid_I_C_barycentric.m",
    'dedon_rob': f"{BASE}/dedoncker_robinson/C/set_dedoncker_robinson_C_barycentric.m",
    'dunavant': f"{BASE}/dunavant/C/set_dunavant_C_barycentric.m",
    'gatermann': f"{BASE}/gatermann/C/set_gatermann_C_barycentric.m",
    'vior_rok': f"{BASE}/vioreanu_rokhlin/C/set_vioreanu_rokhlin_C_barycentric.m",
}

# Existing families that were originally hand-transcribed at only their single
# highest degree, missing the lower-degree rules the same source file also
# provides (found by re-parsing the source and diffing against the registry).
INCOMPLETE_FAMILY_URLS = {
    'lether': f"{BASE}/lether/set_lether_barycentric.m",
    'stroud': f"{BASE}/stroud/set_stroud_barycentric.m",
    'xiao_gim': f"{BASE}/xiao_gimbutas/set_xiao_gimbutas_barycentric.m",
}

FAMILY_URLS = {**NEW_FAMILY_URLS, **REFRESH_FAMILY_URLS, **INCOMPLETE_FAMILY_URLS}

_NUM_RE = r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?'
_CASE_RE = re.compile(r'case\s+(\d+)\s*(.*?)xyw_bar\s*=\s*\[(.*?)\];', re.DOTALL)
_SINGLE_XYW_RE = re.compile(r'xyw_bar\s*=\s*\[(.*?)\];', re.DOTALL)
_STATS_RE = re.compile(r'pointset_stats\s*=\s*\[(.*?)\];', re.DOTALL)


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode('latin-1')


def _parse_rows(body):
    nums = [float(x) for x in re.findall(_NUM_RE, body)]
    return np.array(nums).reshape(-1, 4)


def parse_m_file(text):
    """Returns {degree: (n,4) ndarray} of [bary1, bary2, bary3, weight] rows."""
    blocks = _CASE_RE.findall(text)
    if blocks:
        return {int(deg_str): _parse_rows(body) for deg_str, _, body in blocks}

    # Single-degree file: one bare xyw_bar=[...] block; degree comes from pointset_stats.
    xyw_match = _SINGLE_XYW_RE.search(text)
    stats_match = _STATS_RE.search(text)
    deg = int(float(stats_match.group(1).split()[0]))
    return {deg: _parse_rows(xyw_match.group(1))}


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for kind in sorted(FAMILY_URLS):
        url = FAMILY_URLS[kind]
        print(f"{kind}: fetching {url}")
        text = fetch(url)
        rules = parse_m_file(text)

        family = {'kind': kind, 'source': url, 'rules': {}}
        kept, dropped = [], []
        for deg in sorted(rules):
            arr = rules[deg]
            diag = compute_diagnostics(arr, deg)
            if diag['positive'] and diag['inside_triangle']:
                family['rules'][str(deg)] = diag
                kept.append(deg)
            else:
                dropped.append(deg)

        print(f"  kept degrees {kept}" + (f", dropped (not PI) {dropped}" if dropped else ""))

        if not family['rules']:
            print(f"  no PI-quality degrees survived for '{kind}', skipping file write")
            continue

        out_path = os.path.join(DATA_DIR, f"{kind}.json")
        with open(out_path, 'w') as f:
            json.dump(family, f, indent=2)


if __name__ == '__main__':
    main()
