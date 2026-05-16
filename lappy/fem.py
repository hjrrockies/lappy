import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from .core import BaseEigensolver, BaseDomain


def _cluster_eigs(eigs, ltol=1e-8):
    """Merge sorted eigenvalues within relative tolerance; return (unique, mults)."""
    if len(eigs) == 0:
        return np.array([]), np.array([], dtype=int)
    unique, mults = [], []
    i = 0
    while i < len(eigs):
        j = i + 1
        while j < len(eigs) and (eigs[j] - eigs[i]) <= ltol * max(abs(eigs[i]), 1.0):
            j += 1
        unique.append(np.mean(eigs[i:j]))
        mults.append(j - i)
        i = j
    return np.array(unique), np.array(mults, dtype=int)


class FEMEigensolver(BaseEigensolver):
    """P1 finite element eigensolver for the Dirichlet Laplacian.

    Assembles sparse stiffness (K) and mass (M) matrices on a triangular mesh
    and solves K v = λ M v using ARPACK shift-invert.

    Parameters
    ----------
    domain : BaseDomain
        Domain with bc_type == 'dir'. Must have a .vertices attribute (Polygon).
    mesh_size : float, optional
        Target element size for gmsh. Defaults to domain.diameter / 30.
    k_max : int
        Maximum number of eigenvalues to request in one ARPACK call.
    """

    def __init__(self, domain, mesh_size=None, k_max=200):
        if not isinstance(domain, BaseDomain):
            raise TypeError("'domain' must be a Domain object")
        if domain.bc_type != 'dir':
            raise NotImplementedError("FEMEigensolver only supports Dirichlet BC")
        self.domain = domain
        self.mesh_size = mesh_size
        self.k_max = k_max
        self._assembled = False

    @classmethod
    def from_domain(cls, domain, mesh_size=None, k_max=200):
        return cls(domain, mesh_size=mesh_size, k_max=k_max)

    def _build_and_assemble(self):
        from .quad import polygon_triangular_mesh

        if self.mesh_size is None:
            self.mesh_size = self.domain.diameter / 30

        mesh = polygon_triangular_mesh(self.domain.vertices, self.mesh_size)
        pts2d = mesh.points[:, :2]                    # (N_nodes, 2)
        triangles = mesh.cells_dict['triangle']       # (N_tri, 3) int

        N_nodes = pts2d.shape[0]

        # --- Identify boundary nodes via edge counting ---
        edge_count = {}
        for tri in triangles:
            for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]:
                e = (min(a, b), max(a, b))
                edge_count[e] = edge_count.get(e, 0) + 1

        bdry_nodes = set()
        for (a, b), cnt in edge_count.items():
            if cnt == 1:
                bdry_nodes.add(a)
                bdry_nodes.add(b)

        interior_mask = np.ones(N_nodes, dtype=bool)
        interior_mask[list(bdry_nodes)] = False
        interior_idx = np.where(interior_mask)[0]    # global indices of interior nodes
        N_dof = len(interior_idx)

        global_to_dof = np.full(N_nodes, -1, dtype=int)
        global_to_dof[interior_idx] = np.arange(N_dof)

        # --- Vectorized P1 stiffness and mass assembly ---
        v = pts2d[triangles]                                   # (N_tri, 3, 2)
        # b[t,i] = edge vector opposite node i = v[(i+2)%3] - v[(i+1)%3]
        b = v[:, [2, 0, 1], :] - v[:, [1, 2, 0], :]          # (N_tri, 3, 2)

        # Signed area via cross product of first two edge-from-v0 vectors
        e0 = v[:, 1, :] - v[:, 0, :]                          # (N_tri, 2)
        e1 = v[:, 2, :] - v[:, 0, :]
        A = 0.5 * np.abs(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0])  # (N_tri,)

        # Stiffness: K_e[i,j] = dot(b[i], b[j]) / (4*A)
        Ke = np.einsum('tia,tja->tij', b, b) / (4 * A[:, None, None])  # (N_tri,3,3)

        # Consistent mass: M_e[i,j] = A/12 * (1 + delta_ij)
        Me = (A[:, None, None] / 12) * (np.ones((1, 3, 3)) + np.eye(3)[None, :, :])

        # --- COO accumulation (skip boundary nodes) ---
        dof_idx = global_to_dof[triangles]   # (N_tri, 3), -1 for boundary

        rows_list, cols_list, kvals_list, mvals_list = [], [], [], []
        for li in range(3):
            for lj in range(3):
                r = dof_idx[:, li]
                c = dof_idx[:, lj]
                valid = (r >= 0) & (c >= 0)
                rows_list.append(r[valid])
                cols_list.append(c[valid])
                kvals_list.append(Ke[valid, li, lj])
                mvals_list.append(Me[valid, li, lj])

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        kvals = np.concatenate(kvals_list)
        mvals = np.concatenate(mvals_list)

        self._K = scipy.sparse.csr_array(
            (kvals, (rows, cols)), shape=(N_dof, N_dof))
        self._M = scipy.sparse.csr_array(
            (mvals, (rows, cols)), shape=(N_dof, N_dof))
        self._N_dof = N_dof
        self._assembled = True

    def solve_interval(self, a, b, n_pts=None, **kwargs):
        """Find all Dirichlet eigenvalues in [a, b].

        Parameters
        ----------
        a, b : float
            Search interval.
        n_pts : ignored
            Accepted for API compatibility with MPSEigensolver; not used.

        Returns
        -------
        eigs : ndarray
        mults : ndarray of int
        fevals : int
        """
        if not self._assembled:
            self._build_and_assemble()

        # Weyl estimate of how many eigenvalues lie in [0, b]
        k_weyl = max(10, int(np.ceil(self.domain.area * b / (4 * np.pi))) + 5)
        k_req = min(k_weyl, self._N_dof - 1, self.k_max)

        eigs_all, _ = scipy.sparse.linalg.eigsh(
            self._K, k=k_req, M=self._M, sigma=a, which='LM',
            tol=0, maxiter=10 * k_req,
        )
        eigs_all = np.sort(np.real(eigs_all))

        mask = (eigs_all >= a) & (eigs_all <= b)
        eigs_in = eigs_all[mask]

        ltol = kwargs.get('ltol', 1e-8)
        eigs, mults = _cluster_eigs(eigs_in, ltol=ltol)

        return eigs, mults, 1
