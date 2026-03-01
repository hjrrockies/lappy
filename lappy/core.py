from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

# --- base classes ---
# eigenproblem base class
class BaseEigenproblem(ABC):
    """
    Base class for eigenvalue problems.

    This abstract base class defines the interface for solving eigenvalue problems
    within a specified domain. Subclasses must implement the methods to solve for
    eigenvalues and eigenvectors.

    Parameters
    ----------
    domain : BaseDomain
        An instance of a class derived from BaseDomain that defines the problem's domain.

    Methods
    -------
    solve(a, b)
        Solve the eigenvalue problem for eigenvalues within the interval [a, b].
    """

    def __init__(self, domain):
        if not isinstance(domain, BaseDomain):
            raise ValueError("'domain' must be a Domain object")
        self.domain = domain

    @abstractmethod
    def solve(self, a, b, solver=None, **solver_kwargs):
        pass

    @property
    def bc_type(self):
        return self.domain.bc_type

# eigensolver base class
class BaseEigensolver(ABC):
    """eigensolver base class"""
    @abstractmethod
    def solve_interval(self, a, b):
        pass

# segment base class
class BaseSegment(ABC):
    def __init__(self, bc='dir'):
        if bc == 'dir': bc = 0.0
        elif bc == 'neu': bc = 1.0
        if not (np.isreal(bc) and np.isscalar(bc)):
            raise ValueError("boundary condition must be 'dir', 'neu', or a real scalar")
        self.bc = bc

    @property
    def bc_type(self):
        if self.bc == 0: return 'dir'
        elif self.bc == 1: return 'neu'
        else: return 'rob'
    
    @property
    @abstractmethod
    def p0(self):
        """starting point of the segment"""
        pass

    @property
    @abstractmethod
    def pf(self):
        """ending point of the segment"""
        pass

    @property
    @abstractmethod
    def T0(self):
        """unit tangent vector at the starting point"""
        pass

    @property
    @abstractmethod
    def Tf(self):
        """unit tangent vector at the ending point"""
        pass

    @property
    @abstractmethod
    def is_simple(self):
        pass

    @property
    @abstractmethod
    def is_closed(self):
        pass

    @property
    @abstractmethod
    def len(self):
        pass

    @abstractmethod
    def pts(self, n, kind='legendre', weights=False):
        pass
    
    @abstractmethod
    def tangents(self, n, kind='legendre', weights=False):
        pass
        
    @abstractmethod
    def normals(self, n, kind='legendre', weights=False):
        pass

    def plot(self, nsamp=None, ax=None, showbc=False, **plot_kwargs):
        if nsamp is None:
            nsamp = self.nsamp
        pts = self.p(np.linspace(0,1,nsamp))
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
        if showbc:
            if self.bc==0: plot_kwargs['linestyle'] = '-'
            elif self.bc==1: plot_kwargs['linestyle'] = '--'
            else: plot_kwargs['linestyle'] = '-.'
        return ax.plot(pts.real, pts.imag, **plot_kwargs)

    def plot_tangents(self, nsamp=None, ax=None, **plot_kwargs):
        if nsamp is None:
            nsamp = int(np.ceil(self.nsamp/10))
        pts = self.pts(nsamp, kind='even', weights=False)
        tangents = self.tangents(nsamp, kind='even', weights=False)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
        return ax.quiver(pts.real, pts.imag, tangents.real, tangents.imag, angles='xy', **plot_kwargs)

    def plot_normals(self, nsamp=None, ax=None, **plot_kwargs):
        if nsamp is None:
            nsamp = int(np.ceil(self.nsamp/10))
        pts = self.pts(nsamp, kind='even', weights=False)
        normals = self.normals(nsamp, kind='even', weights=False)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect('equal')
        return ax.quiver(pts.real, pts.imag, normals.real, normals.imag, angles='xy', **plot_kwargs)

# domain base class
class BaseDomain(ABC):
    """
    Abstract base class for domain definitions.

    This class serves as an interface for domain implementations that require
    boundary data extraction and manipulation. Subclasses must implement the
    boundary data retrieval method to support various discretization schemes.
    """
    def init(self):
        if not hasattr(self, 'bdry'):
            raise AttributeError(f"{self.__class__.__name__} must set self.bdry")
        if not hasattr(self.bdry, 'bcs'):
            raise AttributeError("boundary must have boundary conditions")

    @property
    @abstractmethod
    def area(self):
        pass

    @property
    @abstractmethod
    def perimeter(self):
        pass

    @property
    @abstractmethod
    def corner_idx(self):
        pass

    @property
    @abstractmethod
    def corners(self):
        pass

    @property
    @abstractmethod
    def corner_angles(self):
        pass
    
    @abstractmethod
    def int_pts(self):
        pass

    @abstractmethod
    def contains(self, pts):
        pass

