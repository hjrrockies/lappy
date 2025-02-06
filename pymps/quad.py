import numpy as np
from .utils import *
from pygmsh.geo import Geometry
from numpy.polynomial.chebyshev import chebgauss
from numpy.polynomial.legendre import leggauss
from functools import cache


### Quadrature rules for domain boundaries
@cache
def cached_leggauss(order):
    nodes,weights = leggauss(order)
    nodes = (nodes+1)/2 #adjust nodes to interval [0,1]
    weights = weights/2 #adjust weights to interval of unit length
    return nodes,weights

@cache
def cached_chebgauss(order):
    nodes,weights = chebgauss(order)
    # adjust the weights to cancel-out the Gauss-Cheb weighting function
    weights = weights*np.sqrt(1-nodes**2)
    nodes = (nodes+1)/2 #adjust nodes to interval [0,1]
    weights = weights/2 #adjust weights to interval of unit length
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

### Triangular meshes and cubature rules
def load_cubature_rules(path='cubature_rules/'):
    kinds = ['7pts','alb_col','bern_esp1','bern_esp2,','bern_esp4','cowper','day_taylor',
             'dedon_rob','dunavant','vior_rok','xiao_gim','lether','stroud']
    rules = {}
    for kind in kinds:
        try:
            rules[kind] = np.loadz(path+kind)
        except:
            from .cubature_rules import build_cubature_rules, save_cubature_rules
            save_cubature_rules(build_cubature_rules(),path)
            rules[kind] = np.loadz(path+kind)
    return rules

rules = load_cubature_rules()
def get_cubature_rule(kind,deg):
    """Returns a cubature rule of a specified kind and degree in barycentric form"""
    try: arr = rules[kind][deg]
    except: raise ValueError(f"rule of kind '{kind}' and degree {deg} is not defined")
    bary_coords = arr[:,:3]
    bary_weights = arr[:,3]
    return bary_coords, bary_weights

def triangle_areas(mesh_vertices,triangles):
    """Computes the areas of triangles in a triangular mesh"""
    v = mesh_vertices[triangles]
    return 0.5*np.abs((v[:,0,0]-v[:,2,0])*(v[:,1,1]-v[:,0,1])-(v[:,0,0]-v[:,1,0])*(v[:,2,1]-v[:,0,1]))

def triangular_mesh(vertices,mesh_size):
    """Builds a triangular mesh with pygmsh"""
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

def tri_quad(mesh,kind='dunavant',deg=10):
    """For a given mesh, computes nodes and weights for a specified cubature rule"""
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

### Quadrilateral meshes and quadrature rules
def quadrilateral_mesh(vertices,mesh_size):
    """Builds a quadrilateral mesh using pygmsh. NOTE: This function does not always
    give purely quadrilateral meshes. It is retained only for convenience, and should
    not be relied on in general."""
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

def quadrilateral_quad(mesh,order=5):
    """Sets up a quadrature rule for a quadrilateral mesh"""
    mesh_vertices = mesh.points[:,:2]
    quads = mesh.cells[1].data
    return gauss_quad_nodes(mesh_vertices,quads,order)

### Updates for complex form
if __name__ == "__main__":
    def boundary_nodes_polygon(vertices,y=None,orders=20,rule='chebyshev',skip=None):
        """Computes boundary nodes and weights using Chebyshev or Gauss-Legendre
        quadrature rules. Transforms the nodes to lie along the edges of the polygon with
        the given vertices."""
        if y is not None:
            vertices = complex_form(vertices,y)
        
        # select quadrature rule
        if rule == 'chebyshev': quadfunc = cached_chebgauss
        elif rule == 'legendre': quadfunc = cached_leggauss
        elif rule == 'even': quadfunc = lambda n: (np.linspace(0,1,n+2)[1:-1], np.ones(n)/n)
        else: raise(NotImplementedError(f"quadrature rule {rule} is not implemented"))

        # build array of orders (number of nodes/weights) for each edge
        if type(orders) is int:
            orders = orders*np.ones(len(vertices),dtype='int')
            if skip is not None:
                orders[skip] = 0
        elif len(orders) != len(vertices):
            raise ValueError("quadrature orders do not match number of polygon edges")
        else:
            if skip is not None:
                raise ValueError("skip must be 'None' if orders are provided for each edge")

        # set up arrays for nodes and weights
        n_nodes = np.sum(orders)
        nodes = np.empty(n_nodes,dtype='complex')
        weights = np.empty(n_nodes,dtype='float')

        # get polygon edges and lengths
        edges = polygon_edges(vertices)
        lens = edge_lengths(vertices)
        for i in range(len(vertices)):
            if orders[i] > 0:
                start = np.sum(orders[:i])
                end = np.sum(orders[:i+1])
                qnodes,qweights = quadfunc(orders[i]) # get quadrature nodes and weights for interval [0,1]
                # space nodes along edge, adjust weights for edge length
                nodes[start:end] = edges[i]*qnodes + vertices[i]
                weights[start:end] = qweights*lens[i]
        return nodes, weights
    
    def tri_quad(mesh,kind='dunavant',deg=10):
        """"Sets up a cubature rule for a given mesh, in complex form"""
        # extract mesh vertices and triangle-to-vertex array
        mesh_vertices = mesh.points[:,:2]
        triangles = mesh.cells[1].data

        # get triangle vertices in complex form
        tri_vertices = mesh_vertices[triangles]
        tri_vertices_complex = tri_vertices[:,:,0] + 1j*tri_vertices[:,:,1]

        # get cubature nodes and weights in barycentric form
        # convert to array of nodes in complex form
        bary_coords, bary_weights = get_cubature_rule(kind,deg)
        nodes = np.flatten(tri_vertices_complex@(bary_coords.T))

        # get areas of triangles, scale weights appropriately
        areas = triangle_areas(mesh_vertices,triangles)
        weights = np.outer(areas,bary_weights).flatten()
        return nodes, weights
