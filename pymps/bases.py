import numpy as np
from scipy.special import jv, jvp
import scipy.linalg as la
from .utils import complex_form, interior_angles, edge_angles
from functools import cache, lru_cache

class PlanarBasis:
    """Base class for function bases on the plane. Can be parametric, as with FourierBesselBasis,
    which depends on the spectral parameter lambda"""
    def __init__(self):
        pass
    def __call__(self,x,y):
        """Evaluate the basis on a given set of points in the plane"""
        pass
    def grad(self,x,y):
        """Evaluate the gradients of the basis functions on a given set of points in the plane"""
        pass

class FourierBesselBasis(PlanarBasis):
    """A class for Fourier-Bessel bases on polygons. Allows the user to fix an evaluation
    set to reduce computation when evaluating on the same grid for several lambdas.
    Also allows the user to evaluate the basis on an arbitrary set."""
    def __init__(self,vertices,orders,normalize=True,branch_cuts='middle_out'):
        # unpack vertices, put in complex form
        vertices = np.array(vertices)
        if vertices.ndim > 1:
            if vertices.shape[0] == 2:
                vertices = complex_form(vertices[0],vertices[1])
            elif vertices.shape[1] == 2:
                vertices = complex_form(vertices[:,0],vertices[:,1])
        self.vertices = vertices
        self.n_vert = len(self.vertices)
        self.normalize = normalize

        if isinstance(orders,int):
            self.orders = orders*np.ones(self.n_vert,dtype='int')
        else:
            self.orders = np.array(orders,dtype='int')

        if self.orders.shape[0] != self.n_vert:
            raise ValueError('orders must match length of vertices')

        self._set_alphak()
        self.branch_cuts = branch_cuts

    def _set_alphak(self):
        alpha = np.pi/interior_angles(self.vertices)
        self.alphak = [alphai*np.arange(1,ki+1) for alphai,ki in zip(alpha,self.orders)]
        self.alphak_vec = np.concatenate(self.alphak)[np.newaxis]

    @staticmethod
    def fourier_bessel_angles(points,vertices,branch_cuts='middle_out'):
        """Computes the angles of the given points with respect to the bases vertices
        with branch cuts as needed."""
        psis = edge_angles(vertices)
        int_angles = interior_angles(vertices)
        ext_angles = 2*np.pi-int_angles
        angles = np.angle(np.subtract.outer(points,vertices))-psis
        angles[angles<0] += 2*np.pi # puts into form for 'positive_edge' mode
        if branch_cuts == 'positive_edge':
            pass
        elif branch_cuts == 'negative_edge':
            angles[angles>int_angles] -= 2*np.pi
        elif branch_cuts == 'middle_out':
            angles[angles>(2*np.pi - ext_angles/2)] -= 2*np.pi
        elif type(branch_cuts) is float:
            c = 1-branch_cuts
            if c > 1 or c < 0:
                raise ValueError('branch cuts must be placed at a fraction of the exterior angle')
            angles[angles>(2*np.pi - c*ext_angles)] -= 2*np.pi
        elif type(branch_cuts) in [list,np.ndarray]:
            c = 1-np.asarray(branch_cuts)
            if np.any(c > 1) or np.any(c < 0):
                raise ValueError('branch cuts must be placed at a fraction of the exterior angle')
            angles[angles>(2*np.pi - c*ext_angles)] -= 2*np.pi
        else:
            raise ValueError(f"'{branch_cuts}' not a valid branch cut type")
        
        return angles

    def _set_basis_eval(self,points,y=None):
        """
        Set up evaluation of a Fourier-Bessel basis. Useful for when the evaluation
        points will be held constant for a variety of lambdas.
        """
        if y is not None:
            points = complex_form(points,y)

        # unpack attributes
        n = self.n_vert
        m = len(points)

        # compute radii and angles relative to corners with expansions
        orders = self.orders
        r = np.abs(np.subtract.outer(points,self.vertices))
        theta = self.fourier_bessel_angles(points,self.vertices,self.branch_cuts)

        # set up evaluations of Fourier-Bessel
        # first calculate the fourier part (independent of λ!)
        alphak = self.alphak
        sin = np.empty((m,orders.sum()))
        cumk = np.concatenate(([0],np.cumsum(orders)))
        for i in range(n):
            if orders[i] > 0:
                sin[:,cumk[i]:cumk[i+1]] = np.sin(np.outer(theta[:,i],alphak[i]))

        # set up evaluations of bessel part
        r_rep = np.repeat(r,orders,axis=1)

        return r_rep,sin

    def set_default_points(self,points,y=None):
        if y is not None:
            points = complex_form(points,y)
        self.points = points
        self.r_rep,self.sin = self._set_basis_eval(points)
        self.cos, self.dr_dx, self.dr_dy, self.dtheta_dx, self.dtheta_dy = self._set_gradient_eval(points)

    @cache
    def basis_norms(self,lam):
        if self.r_rep is None:
            raise ValueError('Basis has no default points. Provide evaluation '\
                            'points or use FourierBesselBasis.set_default_points')
        r_rep,sin = self.r_rep,self.sin
        out = jv(self.alphak_vec,np.sqrt(lam)*r_rep)*sin
        return la.norm(out,axis=0)

    def __call__(self,lam,points=None,y=None):
        if (points is None) and (y is not None):
            raise ValueError('x coordinates must be provided when y coordinates are provided')
        elif points is None:
            if self.r_rep is None:
                raise ValueError('Basis has no default points. Provide evaluation '\
                                'points or use FourierBesselBasis.set_default_points')
            r_rep,sin = self.r_rep,self.sin
        else:
            if y is not None:
                points = complex_form(points,y)
            r_rep,sin = self._set_basis_eval(points)

        out = jv(self.alphak_vec,np.sqrt(lam)*r_rep)*sin
        if self.normalize:
            out = out/self.basis_norms(lam)
        return out

    def _set_gradient_eval(self,points,y=None):
        if y is None:
            x,y = points.real,points.imag
        else:
            x,y = np.asarray(x),np.asarray(y)
        if (not x.shape or not y.shape) or (type(x) is float and type(y) is float):
            x,y = np.array([x]),np.array([y])
        if x.shape != y.shape:
            raise ValueError('x and y must have the same shape')
        x,y = x.flatten(),y.flatten()

        # unpack attributes
        x_v,y_v = self.vertices.real, self.vertices.imag
        n = self.n_vert
        m = len(x)

        # compute radii and angles relative to corners with expansions
        orders = self.orders
        mask = orders>0
        r = np.abs(np.subtract.outer(points,self.vertices))
        theta = self.fourier_bessel_angles(points,self.vertices,self.branch_cuts)

        # set up evaluations of Fourier-Bessel
        # first calculate the fourier part (independent of λ!)
        alphak = self.alphak
        cos = np.empty((m,orders.sum()))
        cumk = np.concatenate(([0],np.cumsum(orders)))
        for i in range(n):
            if orders[i] > 0:
                cos[:,cumk[i]:cumk[i+1]] = np.cos(np.outer(theta[:,i],alphak[i]))

        # set up x & y partial derivatives
        x_ve, y_ve = x_v[mask], y_v[mask]
        deltax = np.subtract.outer(x,x_ve)
        deltay = np.subtract.outer(y,y_ve)

        dr_dx = np.repeat(deltax/r,orders[mask],axis=1)
        dr_dy = np.repeat(deltay/r,orders[mask],axis=1)
        dtheta_dx = np.repeat(-deltay/(r**2),orders[mask],axis=1)
        dtheta_dy = np.repeat(deltax/(r**2),orders[mask],axis=1)

        return cos, dr_dx, dr_dy, dtheta_dx, dtheta_dy

    def grad(self,lam,points=None,y=None):
        """Computes the gradients of the basis functions at the given points, returning
        the values in complex form (real part for partials w.r.t. to x_i, imaginary part 
        for partials w.r.t. y_i)"""
        if y is not None:
            points = complex_form(points,y)
        if points is not None:
            r_rep,sin = self._set_basis_eval(points)
            cos, dr_dx, dr_dy, dtheta_dx, dtheta_dy = self._set_gradient_eval(points)
        else:
            r_rep,sin = self.r_rep,self.sin
            cos, dr_dx, dr_dy, dtheta_dx, dtheta_dy = self.cos, self.dr_dx, self.dr_dy, self.dtheta_dx, self.dtheta_dy

        sqrtlam_r = np.sqrt(lam)*r_rep
        jv_ = jv(self.alphak_vec,sqrtlam_r)
        jvp_ = jvp(self.alphak_vec,sqrtlam_r)
        dr = np.sqrt(lam)*jvp_*sin
        dtheta = self.alphak_vec*jv_*cos

        if self.normalize:
            out = jv_*sin
            norms = la.norm(out,axis=0)
            dr = dr/norms
            dtheta = dtheta/norms

        return dr*dr_dx + dtheta*dtheta_dx + 1j*(dr*dr_dy + dtheta*dtheta_dy)
    
    def Aprime(self,lam,points=None,y=None):
        """Computes the derivatives of the basis functions with respect to the spectral
        parameter lambda. In other words, computes the matrix A'(lambda)."""
        if (points is None) and (y is not None):
            raise ValueError('x coordinates must be provided when y coordinates are provided')
        elif points is None:
            if self.r_rep is None:
                raise ValueError('Basis has no default points. Provide evaluation '\
                                'points or use FourierBesselBasis.set_default_points')
            r_rep,sin = self.r_rep,self.sin
        else:
            if y is not None:
                points = complex_form(points,y)
            r_rep,sin = self._set_basis_eval(points)

        return (0.5/np.sqrt(lam))*r_rep*jvp(self.alphak_vec,np.sqrt(lam)*r_rep)*sin