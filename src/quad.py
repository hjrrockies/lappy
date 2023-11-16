import numpy as np
from utils import *
from pygmsh.geo import Geometry

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
    self.vertices = np.array(vertices)
    if self.vertices.shape[0] == 2:
        self.vertices = self.vertices.T
    if self.vertices.shape[1] != 2 or self.vertices.ndim != 2:
        raise ValueError('vertices must be a 2-dimensional array of x & y coordinates')

    # build triangular mesh with pygmsh
    with Geometry() as geom:
        geom.add_polygon(vertices.T,mesh_size)
        mesh = geom.generate_mesh()

    return mesh

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
