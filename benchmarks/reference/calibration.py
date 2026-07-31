"""Pipeline sanity check: verify make_default_basis + a manually-assembled
MPSEigensolver reproduce exact closed-form eigenvalues to near machine
precision, before trusting the pipeline on domains without closed forms.

DIGIT CEILING: ~10-14 digits at n_basis=60 for the first few eigenvalues,
degrading to ~7-9 digits by the 8th for both rect(2,1) and
disk_sector(1,pi/3) -- this is expected: n_basis=60 spreads across a fixed
number of Fourier-Bessel/rectangle modes, so later (higher-index)
eigenvalues get comparatively fewer basis functions resolving them at the
same nominal n_basis. This isn't a domain-specific limitation like the
other scripts document -- it's just confirming the tension heuristic
predicts the right order of magnitude of error against a known-exact
answer, at a deliberately modest, cheap n_basis (this script's only job is
a quick correctness check, not a precision push).
"""
import numpy as np
from lappy import geometry, reference
from common import solve_domain, report


def check_rect():
    dom = geometry.rect(2.0, 1.0)
    n_eigs = 8
    eigs, tensions = solve_domain(dom, n_basis=60, n_eigs=n_eigs, verbose=1)
    ref = reference.rect_eigs(n_eigs, 2.0, 1.0)
    report('rect(2,1)', eigs, tensions)
    print(f"ref:      {ref}")
    print(f"abs diff: {np.abs(eigs - ref)}")
    return eigs, tensions, ref


def check_disk_sector():
    dom = geometry.disk_sector(1.0, np.pi / 3)
    n_eigs = 8
    eigs, tensions = solve_domain(dom, n_basis=60, n_eigs=n_eigs, verbose=1)
    ref = reference.sector_eigs(n_eigs, 1.0, np.pi / 3)
    report('disk_sector(1, pi/3)', eigs, tensions)
    print(f"ref:      {ref}")
    print(f"abs diff: {np.abs(eigs - ref)}")
    return eigs, tensions, ref


if __name__ == "__main__":
    check_rect()
    check_disk_sector()
