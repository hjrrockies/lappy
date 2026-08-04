"""
Demonstrates the estimate_multiplicity bug with synthetic tension values.

The GSVD returns tensions sorted ascending at each lambda independently.
The ordering can swap between lambda=eig, lambda=a, lambda=b, causing
is_locmin to have a False at index 0 even when a later index is a valid
local minimum with small tension.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from lappy.mps import estimate_multiplicity, ttol_default

# Use numeric sentinel constants so that tensions() can be called with floats
# and estimate_multiplicity can format them normally.
_EIG = 10.0
_A   = 9.9
_B   = 10.1


def synthetic_tensions(t_eig, t_a, t_b):
    """Return a tensions() callable that returns the given arrays at eig, a, b."""
    t_eig = np.array(t_eig, dtype=float)
    t_a   = np.array(t_a,   dtype=float)
    t_b   = np.array(t_b,   dtype=float)
    def tensions(lam):
        if   np.isclose(lam, _EIG): return t_eig
        elif np.isclose(lam, _A):   return t_a
        elif np.isclose(lam, _B):   return t_b
        raise ValueError(f"unexpected lam={lam}")
    return tensions


def run_case(label, t_eig, t_a, t_b, ttol, expected_mult):
    tensions = synthetic_tensions(t_eig, t_a, t_b)
    got = estimate_multiplicity(tensions, _EIG, _A, _B, ttol=ttol, verbose=2)

    t_eig = np.array(t_eig)
    t_a   = np.array(t_a)
    t_b   = np.array(t_b)
    n = min(len(t_eig), len(t_a), len(t_b))
    t_eig, t_a, t_b = t_eig[:n], t_a[:n], t_b[:n]

    is_locmin = (t_eig <= t_a) & (t_eig <= t_b) & ((t_eig != t_a) | (t_eig != t_b))
    is_small  = t_eig <= ttol

    print(f"\n{'='*60}")
    print(f"Case: {label}")
    print(f"  ttol      = {ttol}")
    print(f"  t_eig     = {t_eig}")
    print(f"  t_a       = {t_a}")
    print(f"  t_b       = {t_b}")
    print(f"  is_locmin = {is_locmin.astype(int)}")
    print(f"  is_small  = {is_small.astype(int)}")
    print(f"  combined  = {(is_locmin & is_small).astype(int)}")
    print(f"  argmin(combined) = {np.argmin(is_locmin & is_small)}")
    print(f"  --> got mult={got}, expected mult={expected_mult}  "
          f"{'OK' if got == expected_mult else '*** BUG ***'}")


# ── Case 1: normal multiplicity-1, works correctly ────────────────────────────
# Tension at eig is small and a local min for index 0.
run_case(
    label="multiplicity-1, no bug",
    t_eig = [0.0005, 0.8, 0.9],
    t_a   = [0.01,   0.7, 0.85],
    t_b   = [0.01,   0.75, 0.88],
    ttol  = 1e-2,
    expected_mult = 1,
)

# ── Case 2: normal multiplicity-2, works correctly ────────────────────────────
run_case(
    label="multiplicity-2, no bug",
    t_eig = [0.0005, 0.0008, 0.9],
    t_a   = [0.01,   0.01,   0.85],
    t_b   = [0.01,   0.01,   0.88],
    ttol  = 1e-2,
    expected_mult = 2,
)

# ── Case 3: THE BUG ───────────────────────────────────────────────────────────
# The tension ordering swaps between eig and a.
# At lambda=eig: t[0]=0.001 is the smallest tension.
# At lambda=a:   t[0]=0.0003 — smaller than t_eig[0], so index 0 is NOT a
#                local minimum at eig.
# At index 1:    t_eig[1]=0.002 < t_a[1]=0.008, so index 1 IS a local min
#                and IS small.
#
# Correct answer: mult=1 (index 1 is a valid eigenvalue).
# argmin returns:  0  (sees False at index 0, stops there).
run_case(
    label="THE BUG: ordering swap, valid eig at index 1 is dropped",
    t_eig = [0.001,  0.002, 0.9],
    t_a   = [0.0003, 0.008, 0.85],   # <-- t_a[0] < t_eig[0]: not a local min
    t_b   = [0.008,  0.009, 0.88],
    ttol  = 1e-2,
    expected_mult = 1,
)

# ── Case 4: tension above ttol, should return 0, currently does ───────────────
# All tensions at eig are above ttol. No eigenvalue here.
# argmin returns 0 "correctly" but only because argmin([0,0,0]) == 0 by
# convention — it would return the same thing even if ttol were not checked.
run_case(
    label="tension above ttol, returns 0 (accidentally correct)",
    t_eig = [0.5, 0.6, 0.9],
    t_a   = [0.8, 0.8, 0.95],
    t_b   = [0.8, 0.8, 0.95],
    ttol  = 1e-2,
    expected_mult = 0,
)
