import numpy as np
from .utils import *
from .cubature import get_cubature_rule
from pygmsh.geo import Geometry
from numpy.polynomial.chebyshev import chebgauss
from numpy.polynomial.legendre import leggauss
from functools import cache

@cache
def cached_leggauss(order):
    return leggauss(order)

@cache
def cached_chebgauss(order):
    nodes,weights = chebgauss(order)
    # note: we adjust the weights to cancel-out the Gauss-Cheb weighting function!
    weights = weights*np.sqrt(1-nodes**2)
    return nodes,weights

def boundary_nodes(vertices,order=20,method='legendre',skip=None):
    """Computes boundary nodes and weights using Chebyshev or Gauss-Legendre
    quadrature rules. Transforms the nodes to lie along the given polygon vertices"""
    if method == 'chebyshev': qnodes,qweights = cached_chebgauss(order)
    elif method == 'legendre': qnodes,qweights = cached_leggauss(order)
    else: raise(NotImplementedError(f"method {method} is not implemented"))
    mask = np.ones(len(vertices),dtype=bool)
    if skip is not None:
        mask[skip] = 0
    n_nodes = order*mask.sum()
    nodes = np.empty((2,n_nodes))
    weights = np.empty(n_nodes)
    x_v,y_v = vertices.T
    j = 0
    for i in range(len(vertices)):
        if mask[i-1]:
            a,b = (x_v[i]-x_v[i-1])/2,(x_v[i]+x_v[i-1])/2
            c,d = (y_v[i]-y_v[i-1])/2,(y_v[i]+y_v[i-1])/2
            sprime = np.sqrt(a**2+c**2)
            nodes[:,j*order:(j+1)*order] = a*qnodes + b, c*qnodes + d
            weights[j*order:(j+1)*order] = qweights*sprime
            j += 1
    return nodes.T, weights

def transform_quad(xi,eta,x_v,y_v):
    """Computes a transformation from the reference square [-1,1]^2 to a
    quadrilateral with given vertices. Also computes the Jacobian determinant"""
    a,b = x_v[2]-x_v[3],x_v[2]+x_v[3]
    c,d = x_v[1]-x_v[0],x_v[1]+x_v[0]
    e,f = y_v[2]-y_v[3],y_v[2]+y_v[3]
    g,h = y_v[1]-y_v[0],y_v[1]+y_v[0]
    etap1 = eta+1
    etam1 = eta-1
    dx_dxi = ((a-c)*eta+a+c)/4
    dx_deta = ((a-c)*xi+b-d)/4
    dy_dxi = ((e-g)*eta+e+g)/4
    dy_deta = ((e-g)*xi+f-h)/4
    x = (etap1*(a*xi+b) - etam1*(c*xi+d))/4
    y = (etap1*(e*xi+f) - etam1*(g*xi+h))/4
    detJ = dx_dxi*dy_deta-dx_deta*dy_dxi
    return  x,y,detJ

def gauss_quad_nodes(mesh_vertices,quads,order=5):
    """Tensor-product Gauss-Legendre quadrature for a quadrilateral mesh"""
    # get Gauss-Legendre points and weights for [-1,1]^2
    pts,wts = cached_leggauss(order)
    Wts = np.outer(wts,wts)
    Xi,Eta = np.meshgrid(pts,pts,indexing='ij')

    # set up data structures
    k = order**2
    n_nodes = k*len(quads)
    nodes = np.empty((2,n_nodes))
    weights = np.empty(n_nodes)

    for i,quad in enumerate(quads):
        x,y = mesh_vertices[quad].T
        x_nodes, y_nodes, detJ = transform_quad(Xi,Eta,x,y)
        quad_weights = detJ*Wts
        nodes[:,i*k:(i+1)*k] = x_nodes.flatten(),y_nodes.flatten()
        weights[i*k:(i+1)*k] = quad_weights.flatten()

    return nodes.T,weights

def triangle_areas(mesh_vertices,triangles):
    v = mesh_vertices[triangles]
    return 0.5*np.abs((v[:,0,0]-v[:,2,0])*(v[:,1,1]-v[:,0,1])-(v[:,0,0]-v[:,1,0])*(v[:,2,1]-v[:,0,1]))

def aggregate_weights(weights,nodes_idx):
    w = np.zeros(int(nodes_idx.max()+1))
    for i in range(nodes_idx.shape[0]):
        for j in range(nodes_idx.shape[1]):
            w[nodes_idx[i,j]] += weights[i,j]
    return w

def tri_midpoints(mesh_vertices,triangles):
    midpoints = np.empty((mesh_vertices.shape[0],mesh_vertices.shape[0],2))
    midpoints[:,:,0] = np.add.outer(mesh_vertices[:,0],mesh_vertices[:,0])/2
    midpoints[:,:,1] = np.add.outer(mesh_vertices[:,1],mesh_vertices[:,1])/2

    edges = {}
    nodes = []
    nodes_idx = []
    j = 0
    for triangle in triangles:
        for i in range(3):
            edge = (triangle[i-1],triangle[i])
            if (edge not in edges.keys()) and (edge[::-1] not in edges.keys()):
                edges[edge] = j
                nodes.append(midpoints[edge])
                j += 1
            try: nodes_idx.append(edges[edge])
            except: nodes_idx.append(edges[edge[::-1]])
    return np.array(nodes),np.array(nodes_idx).reshape(-1,3)

def triangular_mesh(vertices,mesh_size):
    vertices = np.array(vertices)
    if vertices.shape[0] == 2:
        vertices = vertices.T
    if vertices.shape[1] != 2 or vertices.ndim != 2:
        raise ValueError('vertices must be a 2-dimensional array of x & y coordinates')

    # build triangular mesh with pygmsh
    with Geometry() as geom:
        geom.add_polygon(vertices,mesh_size)
        mesh = geom.generate_mesh()

    return mesh

def quadrilateral_mesh(vertices,mesh_size):
    vertices = np.array(vertices)
    if vertices.shape[0] == 2:
        vertices = vertices.T
    if vertices.shape[1] != 2 or vertices.ndim != 2:
        raise ValueError('vertices must be a 2-dimensional array of x & y coordinates')

    # build quadrilateral mesh with pygmsh
    with Geometry() as geom:
        polygon = geom.add_polygon(vertices,mesh_size)
        geom.set_recombined_surfaces([polygon.surface])
        mesh = geom.generate_mesh(dim=2,algorithm=8)
    return mesh

def quad_quad(mesh,order=5):
    mesh_vertices = mesh.points[:,:2]
    quads = mesh.cells[1].data
    return gauss_quad_nodes(mesh_vertices,quads,order)

def tri_quad2(mesh):
    # extract mesh vertices and triangle-to-vertex array
    mesh_vertices = mesh.points[:,:2]
    triangles = mesh.cells[1].data

    # compute quadrature nodes
    nodes, nodes_idx = tri_midpoints(mesh_vertices,triangles)

    areas = triangle_areas(mesh_vertices,triangles)
    weights = np.repeat((areas/3)[:,np.newaxis],3,axis=1)
    weights = aggregate_weights(weights,nodes_idx)

    return nodes, weights

def tri_quad(mesh,kind='dunavant',deg=10):
    # extract mesh vertices and triangle-to-vertex array
    mesh_vertices = mesh.points[:,:2]
    triangles = mesh.cells[1].data

    tri_vertices = mesh_vertices[triangles]
    areas = triangle_areas(mesh_vertices,triangles)
    bary_coords, bary_weights = get_cubature_rule(kind,deg)

    x = tri_vertices[:,:,0]@(bary_coords.T)
    y = tri_vertices[:,:,1]@(bary_coords.T)
    nodes = np.array((x.flatten(),y.flatten())).T
    weights = np.outer(areas,bary_weights).flatten()
    return nodes, weights
