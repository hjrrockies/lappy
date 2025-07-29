from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import norm
from pygmsh.geo import Geometry
import matplotlib.pyplot as plt
from .utils import (polygon_area, polygon_perimeter, polygon_diameter, complex_form, real_form, rand_interior_points,
                    interior_angles, edge_lengths, side_normals, polygon_edges, plot_polygon)
from .quad import triangular_mesh, tri_quad, boundary_nodes_polygon
import shapely.geometry as geo
from shapely import points

class Domain2D(ABC):
    """Base class for 2D domains for eigenproblems
        should have attributes:
        - self.bdry_pts
        - self.bdry_wts
        - self.int_pts
        - self.int_wts
        - self.area
        - self.perimeter
        - self.diameter
        - self.normals

        and methods:
        - self.integrate(f) (integrates f on the domain)
        - self.bdry_integrate(f) (integrates f on the boundary of the domain)
        - self.plot(f=None) (plots the domain, and optionally a function)
        - self.contains(pts) (checks if a point is inside the domain)
        - self.bdry_contains(pts) (checks if a point is on the boundary)

        inherited classes need to define:
        - self._compute_area()
        - self._compute_perimeter()
        - self._compute_diameter()
        - self._compute_normals()
        - self.contains()
        - self.bdry_contains()
    """
    def __init__(self,bdry_pts,bdry_wts,int_pts,int_wts):

        # get boundary & interior points and weights
        self.bdry_pts, self.bdry_wts = bdry_pts, bdry_wts
        self.int_pts, self.int_wts = int_pts, int_wts

        # empty attributes for cached attributes
        self._area = None
        self._perimeter = None
        self._diameter = None

    @property
    def area(self):
        if self._area is None:
            self._area = self._compute_area()
        return self._area
    
    @property
    def perimeter(self):
        if self._perimeter is None:
            self._perimeter = self._compute_perimeter()
        return self._perimeter
    
    @property
    def diameter(self):
        if self._diameter is None:
            self._diameter = self._compute_diameter()
        return self._diameter
    
    @property
    def normals(self):
        if self._normals is None:
            self._normals = self._compute_normals()
        return self._normals
    
    @abstractmethod
    def _compute_area(self):
        pass

    @abstractmethod
    def _compute_perimeter(self):
        pass

    @abstractmethod
    def _compute_diameter(self):
        pass

    @abstractmethod
    def contains(self,pts):
        pass

    @abstractmethod
    def bdry_contains(self,pts):
        pass

    @abstractmethod
    def _compute_normals(self):
        pass

    def plot(self,ax=None,**kwargs):
        raise NotImplementedError
    
    def integrate(self,f,complex_arg=False):
        """integrates f on the domain using the interior points and weights"""
        if complex_arg:
            return np.dot(self.int_wts,f(self.int_pts))
        else:
            return np.dot(self.int_wts,f(self.int_pts.real,self.int_pts.imag))
        
    def bdry_integrate(self,f,complex_arg=False):
        """integrates f on the boundary of the domain using the boundary points and weights"""
        if complex_arg:
            return np.dot(self.bdry_wts,f(self.bdry_pts))
        else:
            return np.dot(self.bdry_wts,f(self.bdry_pts.real,self.bdry_pts.imag))

class Polygon(Domain2D):
    def __init__(self,vertices,bdry_pts=None,bdry_wts=None,int_pts=None,int_wts=None):
        super().__init__(bdry_pts,bdry_wts,int_pts,int_wts)
        self.vertices = complex_form(vertices)

        # shapely objects (for convenience)
        self.sh_poly = geo.Polygon(real_form(self.vertices))
        closed_vertices = np.append(self.vertices,vertices[0])
        rvert = real_form(closed_vertices)
        self.sh_bdry = geo.LineString(rvert)
        self.sh_edges = [geo.LineString(rvert[i:i+2]) for i in range(len(self.vertices))]

        # boundary points and weights
        if bdry_pts is not None:
            if not np.all(self.bdry_contains(bdry_pts)):
                raise ValueError("bdry_pts not all on boundary")
        self.bdry_pts, self.bdry_wts = bdry_pts, bdry_wts

        # interior points and weights
        if int_pts is not None:
            if not np.all(self.contains(int_pts)):
                raise ValueError("int_pts not all inside domain")
        self.int_pts, self.int_wts = int_pts, int_wts

        # if bdry_pts provided, label them with corresponding edge
        self.label_bdry_pts(self.bdry_pts)
        
        # geometric data
        self._edges = None
        self._edge_lengths = None
        self._int_angles = None
        self._edge_normals = None

        # shape derivatives
        self._shape_derivatives = None

    @property
    def edges(self):
        if self._edges is None:
            self._edges = polygon_edges(self.vertices)
        return self._edges
    
    @property
    def edge_lengths(self):
        if self._edge_lengths is None:
            self._edge_lengths = edge_lengths(self.vertices)
        return self._edge_lengths
    
    @property
    def int_angles(self):
        if self._int_angles is None:
            self._int_angles = interior_angles(self.vertices)
        return self._int_angles
    
    @property
    def edge_normals(self):
        if self._edge_normals is None:
            self._edge_normals = side_normals(self.vertices)
        return self._edge_normals
    
    @property
    def shape_derivatives(self):
        if self._shape_derivatives is None:
            self._shape_derivatives = self._compute_shape_derivatives()
        return self._shape_derivatives

    def _compute_area(self):
        return polygon_area(self.vertices)

    def _compute_perimeter(self):
        return polygon_perimeter(self.vertices)

    def _compute_diameter(self):
        return polygon_diameter(self.vertices)
    
    def _compute_normals(self):
        return side_normals(self.vertices)[self.bdry_pts_edge_idx]
    
    def contains(self,pts,y=None):
        """returns a boolean array which is True for pts inside the domain"""
        pts = real_form(complex_form(pts,y))
        sh_points = points(pts)
        return self.sh_poly.contains(sh_points)

    def bdry_contains(self,pts,y=None,tol=1e-15):
        """returns a boolean array which is True for pts on the boundary of the domain, up to tolerance"""
        pts = real_form(complex_form(pts,y))
        sh_points = points(pts)
        return self.sh_bdry.distance(sh_points) <= np.maximum(tol,tol*norm(pts,axis=0))
    
    def bdry_edge_idx(self,pts,y=None,tol=1e-15):
        """returns an integer array labeling each point with the index of the edge it lies on"""
        if not np.all(self.bdry_contains(pts,y,tol)):
            raise ValueError("pts not all on the boundary!")
        
        pts = real_form(complex_form(pts,y))
        sh_points = points(pts)
        edge_idx = np.zeros(len(pts),dtype='int')
        mindists = np.full(len(pts),np.inf)
        for i,sh_edge in enumerate(self.sh_edges):
            dists = sh_edge.distance(sh_points)
            edge_idx[dists<mindists] = i
            mindists = np.minimum(mindists,dists)
        return edge_idx

    def make_boundary_pts(self,n_pts,rule="chebyshev",reentrant_mult=1.25):
        """makes boundary points (and weights) for the polygon"""

        # space ~n_pts in total around edges
        if isinstance(n_pts,(int,np.integer)):
            pfrac = self.edge_lengths/self.perimeter

            # put more points on edges with reentrant corners
            pfrac_new = pfrac.copy()
            reentrant = self.int_angles > np.pi
            reentrant_ = np.roll(reentrant,-1)
            pfrac_new[reentrant] = reentrant_mult*pfrac[reentrant]
            pfrac_new[reentrant_] = reentrant_mult*pfrac[reentrant_]

            # n_pts proportional to edge length and reentrant weighting
            pts_per_edge = np.ceil(n_pts*pfrac_new).astype('int')
            self.bdry_pts, self.bdry_wts = boundary_nodes_polygon(self.vertices,n_pts=pts_per_edge,rule=rule)
            self.label_bdry_pts(pts_per_edge=pts_per_edge)
        else:
            self.bdry_pts, self.bdry_wts = boundary_nodes_polygon(self.vertices,n_pts=n_pts,rule=rule)
            self.label_bdry_pts(pts_per_edge=n_pts)

        return self.bdry_pts, self.bdry_wts

    def label_bdry_pts(self,pts=None,pts_per_edge=None):
        # if built sequentially, labels are in blocks
        if pts_per_edge is not None:
            self.bdry_pts_edge_idx = np.concatenate(tuple([i]*pts_per_edge[i] for i in range(len(self.vertices))))

        # if provided manually, must label each point
        elif pts is not None:
            self.bdry_pts_edge_idx = self.bdry_edge_idx(pts)

    def make_interior_pts(self,method="mesh",mesh_size=None,kind='dunavant',order=14,eps=1e-14,npts_rand=50):
        """makes interior points (and weights) for the polygon"""

        if method == "mesh":
            if mesh_size is None: self.mesh_size = self.diameter*(eps**(1/order))
            else: self.mesh_size = mesh_size
            mesh = triangular_mesh(self.vertices,self.mesh_size)
            self.int_pts, self.int_wts = tri_quad(mesh,kind=kind,deg=order)
        elif method == "random":
            self.int_pts = rand_interior_points(self.vertices,npts_rand)
            self.int_wts = (self.area/npts_rand)*np.ones(len(self.int_pts))
        return self.int_pts, self.int_wts

    def _compute_shape_derivatives(self):
        """computes shape derivatives w.r.t. polygon vertex coordinates. For each coordinate x_i (or y_i),
        the shape derivative $$\\frac{\\partial \\Omega}{\\partial x_i}$$ is a function defined along the boundary
        which gives the 'normal velocity' with respect to perturbations in x_i. Returns the evaluations of shape
        derivatives at the boundary points"""
        edge_idx = self.bdry_pts_edge_idx
        edges = self.edges[edge_idx]
        normals = self.normals
        disp = self.bdry_pts-self.vertices[edge_idx]

        # equals zero at v_i, equals 1 at v_{i+1}
        edge_param = (disp.real*edges.real + disp.imag*edges.imag)/(self.edge_lengths[edge_idx])

        # fill array with shape derivative evaluations
        arr = np.zeros((len(self.bdry_pts),len(self.vertices)),dtype='complex128')
        arr[np.arange(arr.shape[0]),edge_idx] = (1-edge_param)*normals
        arr[np.arange(arr.shape[0]),(edge_idx+1)%arr.shape[1]] = edge_param*normals

        return arr
    
    def plot(self,ax=None,**plotkwargs):
        return plot_polygon(self.vertices,ax,**plotkwargs)







        