import numpy as np
from .utils import complex_form, real_form, polygon_edges, edge_lengths
from numpy.polynomial.chebyshev import chebgauss
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from functools import cache


### Quadrature rules for domain boundaries
@cache
def cached_leggauss(order):
    nodes,weights = leggauss(order)
    nodes = (nodes+1)/2 #adjust nodes to interval [0,1]
    weights = weights/2 #adjust weights to interval of unit length
    return nodes, weights

@cache
def cached_chebgauss(order):
    nodes,weights = chebgauss(order)
    # adjust the weights to cancel-out the Gauss-Cheb weighting function
    weights = weights*np.sqrt(1-nodes**2)
    nodes = (nodes+1)/2 # adjust nodes to interval [0,1]
    weights = weights/2 # adjust weights to interval of unit length
    return nodes[::-1], weights[::-1]

def boundary_nodes_polygon(vertices,n_pts=20,rule='legendre',skip=None):
    """Computes boundary nodes and weights using Chebyshev or Gauss-Legendre
    quadrature rules. Transforms the nodes to lie along the edges of the polygon with
    the given vertices."""
    vertices = np.asarray(vertices)
    if vertices.ndim > 1:
        vertices = complex_form(vertices)

    # select quadrature rule
    if rule == 'chebyshev': quadfunc = cached_chebgauss
    elif rule == 'legendre': quadfunc = cached_leggauss
    elif rule == 'even': quadfunc = lambda n: (np.linspace(0,1,n+2)[1:-1], np.ones(n)/n)
    else: raise(NotImplementedError(f"quadrature rule {rule} is not implemented"))

    # build array of n_pts (number of nodes/weights) for each edge
    if isinstance(n_pts,(int,np.integer)):
        n_pts = n_pts*np.ones(len(vertices),dtype='int')
        if skip is not None:
            n_pts[skip] = 0
    elif len(n_pts) != len(vertices):
        raise ValueError("quadrature n_pts do not match number of polygon edges")
    else:
        if skip is not None:
            raise ValueError("skip must be 'None' if n_pts are provided for each edge")

    # set up arrays for nodes and weights
    n_nodes = int(np.sum(n_pts))
    nodes = np.empty(n_nodes,dtype='complex')
    weights = np.empty(n_nodes,dtype='float')

    # get polygon edges and lengths
    edges = polygon_edges(vertices)
    lens = edge_lengths(vertices)
    for i in range(len(vertices)):
        if n_pts[i] > 0:
            start = np.sum(n_pts[:i])
            end = np.sum(n_pts[:i+1])
            # get quadrature nodes and weights for interval [0,1]
            qnodes,qweights = quadfunc(n_pts[i])
            # space nodes along edge, adjust weights for edge length
            nodes[start:end] = edges[i]*qnodes + vertices[i]
            weights[start:end] = qweights*lens[i]
    return nodes, weights

### Triangular meshes and cubature rules
def load_cubature_rules(path='.cubature_rules/'):
    kinds = ['7pts','alb_col','bern_esp1','bern_esp2','bern_esp4','cowper','day_taylor',
             'dedon_rob','dunavant','vior_rok','xiao_gim','lether','stroud']
    rules = {}
    for kind in kinds:
        try:
            arrs = dict(np.load(path+kind+'.npz'))
            rules[kind] = {int(deg):arr for deg,arr in arrs.items()}
        except:
            from .cubature_rules import build_cubature_rules, save_cubature_rules
            save_cubature_rules(build_cubature_rules(),path)
            arrs = dict(np.load(path+kind+'.npz'))
            rules[kind] = {int(deg):arr for deg,arr in arrs.items()}
    return rules

_rules = None

def _get_rules():
    global _rules
    if _rules is None:
        _rules = load_cubature_rules()
    return _rules

def get_cubature_rule(kind,deg):
    """Returns a cubature rule of a specified kind and degree in barycentric form"""
    rules = _get_rules()
    try: arr = rules[kind][deg]
    except: raise ValueError(f"rule of kind '{kind}' and degree {deg} is not defined")
    bary_coords = arr[:,:3]
    bary_weights = arr[:,3]
    return bary_coords, bary_weights

def triangle_areas(mesh_vertices,triangles):
    """Computes the areas of triangles in a triangular mesh"""
    v = mesh_vertices[triangles]
    return 0.5*np.abs((v[:,0,0]-v[:,2,0])*(v[:,1,1]-v[:,0,1])-(v[:,0,0]-v[:,1,0])*(v[:,2,1]-v[:,0,1]))

def tri_quad(mesh, kind='dunavant', deg=4):
    """"Sets up a cubature rule for a given mesh, in complex form"""
    # extract mesh vertices and triangle-to-vertex array
    mesh_vertices = mesh.points[:,:2]
    triangles = mesh.cells[1].data

    # get triangle vertices in complex form
    tri_vertices = mesh_vertices[triangles]
    # tri_vertices_complex = tri_vertices[:,:,0] + 1j*tri_vertices[:,:,1]

    # get cubature nodes and weights in barycentric form
    # convert to array of nodes in complex form
    bary_coords, bary_weights = get_cubature_rule(kind,deg)
    nodes = (tri_vertices[:,:,0]@(bary_coords.T) + 1j*(tri_vertices[:,:,1]@(bary_coords.T))).flatten()

    # get areas of triangles, scale weights appropriately
    areas = triangle_areas(mesh_vertices,triangles)
    weights = np.outer(areas,bary_weights).flatten()
    return nodes, weights

# mesh building
def polygon_triangular_mesh(vertices, mesh_size, mesh_size_min=0.05, mesh_size_max=0.5):
    """Builds a triangular mesh on a polygon with pygmsh"""
    import gmsh
    from pygmsh.geo import Geometry
    vertices = np.asarray(vertices)
    if vertices.dtype == 'complex128':
        vertices = real_form(vertices)
    if vertices.shape[0] == 2:
        vertices = vertices.T
    if vertices.shape[1] != 2 or vertices.ndim != 2:
        raise ValueError('vertices must be a 2-dimensional array of x & y coordinates')

    # build triangular mesh with pygmsh
    with Geometry() as geom:
        geom.add_polygon(vertices, mesh_size)

        # Set meshing options
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)  # Use point sizes
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)  # Extend to interior
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        mesh = geom.generate_mesh()
    return mesh

def curvature_sampling(spline, t0, tf, pts_per_2pi=20):
    """Gets samples from a SciPy BSpline with density based on curvature."""
    if not isinstance(spline, BSpline):
        raise TypeError("'spline' must be a SciPy BSpline.")
    
    t_fine = np.linspace(t0, tf, 1000)
    dr = spline.derivative(nu=1)(t_fine)
    ddr = spline.derivative(nu=2)(t_fine)
    
    speed = np.sqrt(dr[:, 0]**2 + dr[:, 1]**2)
    cross = dr[:, 0] * ddr[:, 1] - dr[:, 1] * ddr[:, 0]
    curvature = np.abs(cross) / (speed**3 + 1e-10)
    
    # Point density: higher curvature = more points
    # Scale by arc length element (speed * dt) to account for parameterization
    dt = np.diff(t_fine, prepend=t_fine[0])
    arc_element = speed * dt
    
    # Points needed per segment based on curvature
    points_per_segment = (curvature / (2 * np.pi)) * pts_per_2pi * arc_element
    
    # Add minimum to handle straight sections
    points_per_segment = np.maximum(points_per_segment, 0.02)
    
    cumulative_points = np.cumsum(points_per_segment)
    total_points = max(2, int(np.ceil(cumulative_points[-1])))
    
    target_positions = np.linspace(0, cumulative_points[-1], total_points)
    t_samples = np.interp(target_positions, cumulative_points, t_fine)
    
    return t_samples

def spline_mesh_with_curvature(segments, pts_per_2pi=20, mesh_size_min=0.05, mesh_size_max=0.5):
    """
    Creates a mesh from a list of SciPy BSpline objects with curvature-adaptive sampling.

    Parameters
    ----------
    splines : list of BSpline
        Closed loop of splines defining the boundary
    pts_per_2pi : float
        Points per 2π radians of curvature for boundary sampling
    mesh_size_min : float
        Minimum mesh size (at high-curvature regions)
    mesh_size_max : float
        Maximum mesh size (in interior/low-curvature regions)

    Returns
    -------
    mesh : meshio.Mesh
        Generated mesh
    """
    import gmsh
    from pygmsh.geo import Geometry
    # Sample each spline with curvature-adaptive spacing
    boundary_points = []
    boundary_curvatures = []
    
    for seg in segments:
        # Get curvature-sampled points
        t_samples = curvature_sampling(seg.spline, seg.t0, seg.tf, pts_per_2pi)
        pts = seg.spline(t_samples)
        
        # Also get curvature at these points for mesh sizing
        dr = seg.spline.derivative(nu=1)(t_samples)
        ddr = seg.spline.derivative(nu=2)(t_samples)
        speed = np.sqrt(dr[:, 0]**2 + dr[:, 1]**2)
        cross = dr[:, 0] * ddr[:, 1] - dr[:, 1] * ddr[:, 0]
        curvature = np.abs(cross) / (speed**3 + 1e-10)
        
        boundary_points.append(pts)
        boundary_curvatures.append(curvature)

    # Concatenate all boundary points (remove duplicate endpoints between splines)
    all_points = []
    all_curvatures = []
    for i, (pts, curv) in enumerate(zip(boundary_points, boundary_curvatures)):
        if i == 0:
            all_points.append(pts)
            all_curvatures.append(curv)
        else:
            # Skip first point (duplicate of previous spline's last point)
            all_points.append(pts[1:])
            all_curvatures.append(curv[1:])
            
    all_points = np.vstack(all_points)[:-1]
    all_curvatures = np.concatenate(all_curvatures)[:-1]
    
    # Remove last point if it duplicates the first (closing the loop)
    if np.allclose(all_points[-1], all_points[0]):
        all_points = all_points[:-1]
        all_curvatures = all_curvatures[:-1]
    
    # Compute mesh sizes based on curvature
    # Higher curvature -> smaller mesh size
    max_curv = np.percentile(all_curvatures, 95)  # Use 95th percentile to avoid outliers
    normalized_curv = np.clip(all_curvatures / (max_curv + 1e-10), 0, 1)
    
    # Interpolate between min and max size (inverse relationship with curvature)
    mesh_sizes = mesh_size_max - normalized_curv * (mesh_size_max - mesh_size_min)
    
    with Geometry() as geom: 
        # Create gmsh points with prescribed mesh sizes
        gmsh_points = []
        for pt, size in zip(all_points, mesh_sizes):
            gmsh_points.append(geom.add_point([pt[0], pt[1], 0], mesh_size=size))
        
        # Create line segments connecting consecutive points
        lines = []
        n_pts = len(gmsh_points)
        for i in range(n_pts):
            lines.append(geom.add_line(gmsh_points[i], gmsh_points[(i + 1) % n_pts]))
        
        # Create curve loop and surface
        curve_loop = geom.add_curve_loop(lines)
        surface = geom.add_plane_surface(curve_loop)
        
        # Set meshing options
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)  # Use point sizes
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)  # Extend to interior
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        
        # Generate mesh
        mesh = geom.generate_mesh(dim=2)
    
    return mesh

### Quadrilateral meshes and quadrature rules
def quadrilateral_mesh(vertices,mesh_size):
    """Builds a quadrilateral mesh using pygmsh. NOTE: This function does not always
    give purely quadrilateral meshes. It is retained only for convenience, and should
    not be relied on in general."""
    import gmsh
    from pygmsh.geo import Geometry
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