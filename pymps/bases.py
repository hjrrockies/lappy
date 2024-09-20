import numpy as np
from scipy.special import jv, jvp, jve
import scipy.linalg as la
from .utils import *

# TODO
# 1. Add "plot_subspace_angles" method or similar

class FourierBesselBasis:
    """A class for Fourier-Bessel bases on polygons. Allows the user to fix an evaluation
    set to reduce computation when evaluating on the same grid for several lambdas.
    Also allows the user to evaluate the basis on an arbitrary set."""
    def __init__(self,vertices,orders):
        # process vertices array to be of shape (m,2)
        self.vertices = np.array(vertices)
        if self.vertices.shape[0] == 2:
            self.vertices = self.vertices.T
        if self.vertices.shape[1] != 2 or self.vertices.ndim != 2:
            raise ValueError('vertices must be a 2-dimensional array of x & y coordinates')
        self.n_vert = self.vertices.shape[0]

        if isinstance(orders,int):
            self.orders = orders*np.ones(self.n_vert,dtype='int')
        else:
            self.orders = np.array(orders,dtype='int')

        if self.orders.shape[0] != self.n_vert:
            raise ValueError('orders must match length of vertices')

        self._set_alphak()

    def _set_alphak(self):
        x_v,y_v = self.vertices.T
        alpha = np.pi/calc_angles(x_v,y_v)
        self.alphak = [alphai*np.arange(1,ki+1) for alphai,ki in zip(alpha,self.orders)]
        self.alphak_vec = np.concatenate(self.alphak)[np.newaxis]

    def _set_basis_eval(self,x,y):
        """
        Set up evaluation of a Fourier-Bessel basis. Useful for when the evaluation
        points will be held constant for a variety of lambdas.
        """
        x,y = np.asarray(x),np.asarray(y)
        if not x.shape or not y.shape:
            x,y = np.array([x]),np.array([y])
        if x.shape != y.shape:
            raise ValueError('x and y must have the same shape')
        x,y = x.flatten(),y.flatten()

        # unpack attributes
        x_v,y_v = self.vertices.T
        n = self.n_vert
        m = len(x)

        # compute radii and angles relative to corners with expansions
        orders = self.orders
        mask = orders>0
        x_ve, y_ve = x_v[mask], y_v[mask]
        r = radii(x,y,x_ve,y_ve)
        theta = thetas(x,y,x_v,y_v)
        self.theta = theta

        # set up evaluations of Fourier-Bessel
        # first calculate the fourier part (independent of λ!)
        alphak = self.alphak
        sin = np.empty((m,orders.sum()))
        cumk = np.concatenate(([0],np.cumsum(orders)))
        for i in range(n):
            if orders[i] > 0:
                sin[:,cumk[i]:cumk[i+1]] = np.sin(np.outer(theta[:,i],alphak[i]))

        # set up evaluations of bessel part
        r_rep = np.repeat(r,orders[mask],axis=1)

        return r_rep,sin

    def set_default_points(self,x,y):
        self.x,self.y = x,y
        self.r_rep,self.sin = self._set_basis_eval(x,y)

    def __call__(self,lambda_,x=None,y=None):
        if (x is None) != (y is None):
            raise ValueError('x and y must both be set, or left unset')
        elif x is None:
            if self.r_rep is None:
                raise ValueError('Basis has no default points. Provide evaluation '\
                                 'points or use FourierBesselBasis.set_default_points')
            r_rep,sin = self.r_rep,self.sin
        else:
            r_rep,sin = self._set_basis_eval(x,y)

        out = jve(self.alphak_vec,np.sqrt(lambda_)*r_rep)*sin
        if np.any(np.isnan(out)):
            print(lambda_)
            print(np.where(np.isnan(out)))
        if np.any(np.isinf(out)):
            print(lambda_)
            print(np.where(np.isinf(out)))
        return out

    def _set_derivative_eval(self,x,y):
        x,y = np.asarray(x),np.asarray(y)
        if not x.shape or not y.shape:
            x,y = np.array([x]),np.array([y])
        if x.shape != y.shape:
            raise ValueError('x and y must have the same shape')
        x,y = x.flatten(),y.flatten()

        # unpack attributes
        x_v,y_v = self.vertices.T
        n = self.n_vert
        m = len(x)

        # compute radii and angles relative to corners with expansions
        orders = self.orders
        mask = orders>0
        x_ve, y_ve = x_v[mask], y_v[mask]
        r = radii(x,y,x_ve,y_ve)
        theta = thetas(x,y,x_v,y_v)

        # set up evaluations of Fourier-Bessel
        # first calculate the fourier part (independent of λ!)
        alphak = self.alphak
        cos = np.empty((m,orders.sum()))
        cumk = np.concatenate(([0],np.cumsum(orders)))
        for i in range(n):
            if orders[i] > 0:
                cos[:,cumk[i]:cumk[i+1]] = np.cos(np.outer(theta[:,i],alphak[i]))

        # set up x & y partial derivatives
        deltax = np.subtract.outer(x,x_ve)
        deltay = np.subtract.outer(y,y_ve)

        dr_dx = np.repeat(deltax/r,orders[mask],axis=1)
        dr_dy = np.repeat(deltay/r,orders[mask],axis=1)
        dtheta_dx = np.repeat(-deltay/(r**2),orders[mask],axis=1)
        dtheta_dy = np.repeat(deltax/(r**2),orders[mask],axis=1)

        return cos, dr_dx, dr_dy, dtheta_dx, dtheta_dy

    def grad(self,lambda_,x,y):
        r_rep,sin = self._set_basis_eval(x,y)
        cos, dr_dx, dr_dy, dtheta_dx, dtheta_dy = self._set_derivative_eval(x,y)

        dr = np.sqrt(lambda_)*jvp(self.alphak_vec,np.sqrt(lambda_)*r_rep)*sin
        dtheta = self.alphak_vec*jv(self.alphak_vec,np.sqrt(lambda_)*r_rep)*cos

        return dr*dr_dx + dtheta*dtheta_dx, dr*dr_dy + dtheta*dtheta_dy
