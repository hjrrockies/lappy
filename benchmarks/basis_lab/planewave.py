"""A plane-wave particular-solution basis, as an experiment.

`cos(k d.x)` and `sin(k d.x)` with `k = sqrt(lam)` and `d` a unit direction solve the
Helmholtz equation exactly, so they are legitimate MPS columns. They are the classical
alternative to corner-centred Fourier--Bessel and to fundamental solutions, and they have a
different failure mode: no localization at all. That makes them a candidate exactly where the
localized families do badly -- a domain whose modes live in the *bulk* of two regions joined by
a thin neck, where `FundamentalBasis` sources on an offset boundary buy nothing (measured:
`mushroom_neck01` is unmoved, 3.0e-07 against the default's 3.2e-07, at every offset tried).

Prototype in `benchmarks/` rather than `lappy/` until it earns its place.
"""
import numpy as np

from lappy.bases import ParticularBasis


class PlaneWaveBasis(ParticularBasis):
    """Plane waves in `n_dirs` equally spaced directions, cos and sin of each.

    `len` is `2*n_dirs`. Directions are offset by half a step from the axes by default, which
    avoids putting a node line exactly on a symmetric domain's axis.
    """

    def __init__(self, n_dirs, offset=None):
        n_dirs = int(n_dirs)
        if n_dirs < 1:
            raise ValueError('n_dirs must be >= 1')
        if offset is None:
            offset = 0.5*np.pi/n_dirs
        self.angles = offset + np.pi*np.arange(n_dirs)/n_dirs
        self.dirs = np.exp(1j*self.angles)

    def __len__(self):
        return 2*len(self.dirs)

    def _phase(self, lam, pts):
        k = np.sqrt(max(float(lam), 0.0))
        z = pts.pts if hasattr(pts, 'pts') else np.asarray(pts)
        z = np.asarray(z).ravel()
        return k, (k*(z.real[:, None]*self.dirs.real[None, :]
                      + z.imag[:, None]*self.dirs.imag[None, :]))

    def _eval_pointset(self, lam, pts, cols=None):
        _, ph = self._phase(lam, pts)
        A = np.concatenate([np.cos(ph), np.sin(ph)], axis=1)
        return A if cols is None else A[:, cols]

    def _grad_pointset(self, lam, pts, cols=None):
        k, ph = self._phase(lam, pts)
        d = self.dirs[None, :]
        G = np.concatenate([-k*d*np.sin(ph), k*d*np.cos(ph)], axis=1)
        return G if cols is None else G[:, cols]
