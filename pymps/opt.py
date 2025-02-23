import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from .utils import edge_lengths

def discrete_local_min_idx(x,y):
    x = x.flatten()
    sort_idx = x.argsort()
    x = x[sort_idx]
    y = y[sort_idx]

    min_idx = np.argwhere((y[1:-1]<y[:-2])*(y[1:-1]<y[2:]))
    return min_idx+1

def parabola_vertex(x,y):
    """Finds the vertex of a parabola passing through the points
    {(x[0],y[0]),(x[1],y[1]),(x[2],y[2])}"""
    dx0,dx1,dx2 = x[1]-x[0],x[2]-x[1],x[2]-x[0]
    dy0,dy1,dy2 = y[1]-y[0],y[2]-y[1],y[2]-y[0]

    C = (dx0*dy1 - dx1*dy0)/(dx0*dx1*dx2)
    if C<=0 or np.isnan(C): vertex = x[np.argmin(y)]
    else: vertex = (x[1]+x[2]-dy1/(dx1*C))/2
    return vertex

def parabolic_iter_min(f,x,y,xtol=1e-8,maxiter=10,maxresc=2,resc_param=0.1,nrecurse=0,verbose=False):
    """Finds a local minimum of f in the interval [x[0],x[2]] by repeated parabolic interpolation."""
    tabs = min(nrecurse,10)*"\t"
    if x[1]<=x[0] or x[2]<=x[1]:
        raise ValueError('x not increasing!')

    # shift left endpoint to zero for maximum precisioon
    z = x-x[0]

    # shifted interval bounds
    zlo, zhi = z[0],z[2]

    # function evaluation & rescue counters
    fevals = 0
    rescues = 0

    # previous vertex
    vold = np.nan

    if verbose: print(tabs+f"parabolic_iter_min on ({x[0]:.3e},{x[1]:.3e},{x[2]:.3e})")
    # iterative parabolic interpolation
    for i in range(maxiter):
        v = parabola_vertex(z,y)

        # if the vertex hasn't changed much, conclude iteration
        if np.abs(v-vold)<xtol:
            break

        # vertex falls off left, rescue if possible
        elif v<zlo-2*xtol:
            if verbose: print(tabs+'fell off left')
            if rescues < maxresc:
                rescues += 1
                v = (1-resc_param)*zlo + resc_param*z[1]
            else:
                if verbose: print(tabs+'too many rescues')
                return None, fevals
        # vertex falls off right, rescue if possible
        elif v>zhi+2*xtol:
            if verbose: print(tabs+'fell off right')
            if rescues < maxresc:
                rescues += 1
                v = (1-resc_param)*zhi + resc_param*z[1]
            else:
                if verbose: print(tabs+'too many rescues')
                return None, fevals
        # if vertex too close to old points, wiggle it
        if np.abs(z-v).min() < xtol/2: v += xtol

        # interpolate using new vertex and adjacent points in z
        vold = v
        yv = f(v+x[0])
        fevals += 1
        if v<z[1]:
            z = np.array([z[0],v,z[1]])
            y = np.array([y[0],yv,y[1]])
        elif z[1]<v:
            z = np.array([z[1],v,z[2]])
            y = np.array([y[1],yv,y[2]])
        if verbose: print(tabs+f"z={np.array_str(z,precision=3)}")

    # return final vertex, shifted back to original interval
    if verbose: print(tabs+f'parabolic_iter_min concluded, x_min={v+x[0]}')
    return float(v+x[0]), fevals

def minsearch(f,a,b,n,xtol=1e-8,maxdepth=10,fargs=(),verbose=False):
    """Finds all minima in the interval [start,end] of a function f, where the second output
    of f is treated as a proxy for nearby minima."""
    spaces = (10-maxdepth)*('  ')

    x = np.linspace(a,b,n+1)
    h = (b-a)/n
    # override maxdepth if at tolerance
    if h <= xtol:
        maxdepth = 0

    # threshold to flag at endpoints
    fm = 1.1*np.sqrt(h**2+xtol**2)
    if verbose: print(spaces+f"fm={fm}")
    
    # set up arrays for function outputs
    recurse_flag = np.zeros(n+2,dtype=int)
    checked = np.zeros(n,dtype=bool)
    y = np.empty((2,n+1),dtype='float')
    y[:,0] = f(x[0],*fargs)
    y[:,1] = f(x[1],*fargs)
    fevals = 2
    
    if verbose:
        print(spaces+f'minsearch on [{a:.3e},{b:.3e}]')
        print(spaces+f'h={h}, n_intervals={n}')
        print(spaces+f"fevals={fevals}")

    if np.min(y[1,:2]) < 1.1*h and maxdepth > 0:
        if verbose:
            print(spaces+f'sigma2 = {np.min(y[1,:2]):.3e}<1.1*h={1.1*h:.3e}')
            print(spaces+f"[{x[0]:.3e},{x[1]:.3e}] flagged for recursion")
        recurse_flag[1] = True

    def islocmin(i):
        if (y[0,i-1]<y[0,i-2] and y[0,i-1]<y[0,i]):
            return True
        elif i==2 and np.min(y[0,:3])<fm:
            return True
        elif i==n and np.min(y[0,-3:])<fm:
            return True
        else:
            return False
    
    minima = []
    for i in range(2,n+1):
        y[:,i] = f(x[i],*fargs)
        fevals += 1
        if verbose:
            print(spaces+str(i))
            print(spaces+f'x={np.array_str(x[i-2:i+1],precision=3)}')
            print(spaces+f'sigma1={np.array_str(y[0,i-2:i+1],precision=3)}')
            print(spaces+f'sigma2={np.array_str(y[1,i-2:i+1],precision=3)}')
            print(spaces+f"fevals={fevals}")

        # catch small second subspace angle, flag for recursion
        if np.min(y[1,i-1:i+1]) < 1.1*h and maxdepth > 0:
            if verbose:
                print(spaces+f'sigma2 = {np.min(y[1,i-1:i+1]):.3e}<1.1*h={1.1*h:.3e}')
                print(spaces+f"[{x[i-1]:.3e},{x[i]:.3e}] flagged for recursion")
            recurse_flag[i] = 1
            if i==2:
                recurse_flag[1] = 1
                if verbose: print(spaces+f"[{x[i-2]:.3e},{x[i-1]:.3e}] also flagged for recursion")
        # catch discrete local min
        elif islocmin(i) and not np.any(checked[i-2:i+1]):
            if verbose: print(spaces+f"discrete local min at x={x[i-1]:.3e}")
            if recurse_flag[i-1]:
                if verbose: print(spaces+f"[{x[i-1]:.3e},{x[i]:.3e}] flagged for recursion")
                recurse_flag[i] = 1
            else:
                if verbose: print('********')
                mins, fe = parabolic_iter_min(lambda x: f(x,*fargs)[0]**2,x[i-2:i+1],y[0,i-2:i+1]**2,xtol=xtol,verbose=verbose)
                fevals += fe
                if verbose: print('********\n'+spaces+f"fevals={fevals}")
                if mins is not None: 
                    minima.append(mins)
                    checked[i-2:i+1] = True
                

    # recurse on flagged stretches
    diffs = np.diff(recurse_flag)
    starts = np.where(diffs==1)[0]
    ends = np.where(diffs==-1)[0]
    for start,end in zip(starts,ends):
        length = end-start
        if verbose: print(spaces+f"recursing on [{x[start]:.3e},{x[end]:.3e}]\n")
        mins,fe = minsearch(f,x[start],x[end],2*length,xtol=xtol,verbose=verbose,maxdepth=maxdepth-1,fargs=fargs)
        minima += mins
        fevals += fe

    if verbose:
        print(spaces+f"minsearch on [{a:.3e},{b:.3e}] concluded")
        print(spaces+f"found minima: {minima}")
    return minima, fevals

def minsearch2(f,a,b,n,xtol=1e-8,fargs=(),maxdepth=10,verbose=False):
    """Finds all minima in the interval [start,end] of a function f, where the second output
    of f is treated as a proxy for nearby minima."""
    spaces = (10-maxdepth)*('  ')

    x = np.linspace(a,b,n+1)
    h = (b-a)/n
    # override maxdepth if at tolerance
    if h <= xtol:
        maxdepth = 0

    # threshold to flag at endpoints
    fm = 1.1*np.sqrt(h**2+xtol**2)
    if verbose: print(spaces+f"fm={fm}")
    
    # set up arrays for function outputs
    recurse_flag = np.zeros(n,dtype=int)

    checked = np.zeros(n,dtype=bool)
    y = np.empty((2,n+1),dtype='float')

    # fill in beginning and end
    y[:,0] = f(float(x[0]),*fargs)
    y[:,1] = f(float(x[1]),*fargs)
    y[:,-1] = f(float(x[-2]),*fargs)
    y[:,-2] = f(float(x[-1]),*fargs)
    fevals = 4
    
    if verbose:
        print(spaces+f'minsearch on [{a:.3e},{b:.3e}]')
        print(spaces+f'h={h}, n_intervals={n}')
        print(spaces+f"fevals={fevals}")

    def islocmin(i):
        if (y[0,i]<y[0,i-1] and y[0,i]<y[0,i+1]):
            return True
        elif i==1 and np.min(y[0,:3])<fm:
            return True
        elif i==n-1 and np.min(y[0,-3:])<fm:
            return True
        else:
            return False
    
    minima = []
    # check if x[i] is a local min
    for i in range(1,n):
        y[:,i+1] = f(float(x[i+1]),*fargs)
        fevals += 1
        if verbose:
            print(spaces+str(i))
            print(spaces+f'x={np.array_str(x[i-1:i+2],precision=3)}')
            print(spaces+f'sigma1={np.array_str(y[0,i-1:i+2],precision=3)}')
            print(spaces+f'sigma2={np.array_str(y[1,i-1:i+2],precision=3)}')
            print(spaces+f"fevals={fevals}")

        # check for local min
        if islocmin(i) and not checked[i-1]:
            if verbose: print(spaces+f"discrete local min at x={x[i]:.3e}")

            # if second subspace angle is close to first, flag for recursion
            mingap = np.min(y[1,i-1:i+2]-y[0,i-1:i+2])
            if mingap < 1.1*h and maxdepth > 0:
                if verbose:
                    print(spaces+f'mingap = {mingap:.3e}<1.1*h={1.1*h:.3e}')
                    print(spaces+f"[{x[i-1]:.3e},{x[i+1]:.3e}] flagged for recursion")
                recurse_flag[i-1] = 1
                recurse_flag[i] = 1
            else:
                if verbose: print('********')
                mins, fe = parabolic_iter_min(lambda x: f(float(x),*fargs)[0]**2,x[i-1:i+2],y[0,i-1:i+2]**2,xtol=xtol,verbose=verbose)
                fevals += fe
                if verbose: print('********\n'+spaces+f"fevals={fevals}")
                if mins is not None:
                    minima.append(mins)
                    checked[i-1] = True
                    checked[i] = True

    # recurse on flagged stretches
    padded_flags = np.zeros(len(recurse_flag)+2,dtype=int)
    padded_flags[1:-1] = recurse_flag
    diffs = np.diff(padded_flags)
    starts = np.nonzero(diffs==1)[0]
    ends = np.nonzero(diffs==-1)[0]
    for start,end in zip(starts,ends):
        length = end-start
        if verbose: print(spaces+f"recursing on [{x[start]:.3e},{x[end]:.3e}]\n")
        mins,fe = minsearch2(f,x[start],x[end],4*length,xtol=xtol,fargs=fargs,maxdepth=maxdepth-1,verbose=verbose)
        minima += mins
        fevals += fe

    if verbose:
        print(spaces+f"minsearch on [{a:.3e},{b:.3e}] concluded")
        print(spaces+f"found minima: {minima}")
    return minima, fevals

def normalized_reciprocals(x,jac=False):
    if jac:
        col = (1/x[1:]).reshape(-1,1)
        diag = -x[0]/(x[1:]**2)
        jacobian = np.hstack((col,np.diag(diag)))
        return x[0]/x[1:], jacobian
    return x[0]/x[1:]

def l2_loss(x,y,jac=False):
    diff = x-y
    if jac:
        return la.norm(diff)**2, 2*diff
    return la.norm(diff)**2

def polygon_vertices(p,perim=1,jac=False):
    N = len(p)//2 + 2
    x,y = p[:N-2],p[N-2:]
    vertices = np.zeros(N,dtype='complex')
    vertices[1:-1] = x + 1j*y
    C = perim - np.abs(vertices[2:]-vertices[1:-1]).sum()
    vertices[0] = 0.5*(x[0] + C + y[0]**2/(x[0]-C))
    if jac:
        l = edge_lengths(vertices)
        alpha = y[0]/(x[0]-C)
        jacobian = np.zeros((2*N,len(p)))
        print(jacobian.shape)
        x,y = vertices.real, vertices.imag

        # partial derivatives of x_1 w.r.t. to p
        jacobian[0,0] = 0.5*((1-alpha**2) + (1+alpha**2)*(x[2]-x[1])/l[1])
        jacobian[0,N-2] = alpha + 0.5*(1+alpha**2)*(y[2]-y[1])/l[1]
        jacobian[0,1:N-3] = 0.5*(1+alpha**2)*((x[3:-1]-x[2:-2])/l[2:-2]-(x[2:-2]-x[1:-3])/l[1:-3])
        jacobian[0,N-1:-1] = 0.5*(1+alpha**2)*((y[3:-1]-y[2:-2])/l[2:-2]-(y[2:-2]-y[1:-3])/l[1:-3])
        jacobian[0,N-3] = -0.5*(1+alpha**2)*(x[-2]/np.abs(vertices[-2]) + (x[-2]-x[-3])/l[-2])
        jacobian[0,-1] = -0.5*(1+alpha**2)*(y[-2]/np.abs(vertices[-2]) + (y[-2]-y[-3])/l[-2])

        # partial derivatives of x_j w.r.t. themselves are identity
        jacobian[1:N-1,:N-2] = np.eye(N-2)
        # partial derivatives of y_j w.r.t. themselves are identity
        jacobian[N+1:-1,N-2:] = np.eye(N-2)
        
        return vertices, jacobian
    return vertices

def polygon_perimeter(vertices,jac=False):
    """computes the perimeter, and optionally the gradient of the perimeter, of a polygon
    with given vertices, written in complex form"""
    dv = vertices-np.roll(vertices,1)
    lengths = np.abs(dv)
    perim = lengths.sum()
    if jac:
        grad = dv/lengths
        grad = grad - np.roll(grad,-1)
        return perim, grad
    return perim

def eig_obj(p,eigs_target,perim_tol=1e-15):
    # check number of eigenvalues, convert targets to normalized reciprocals
    K = len(eigs_target)
    targets = normalized_reciprocals(eigs_target)
    
    # get vertices from parameter vector
    vertices = polygon_vertices(p)
    N = len(vertices)

    # get perimeter and perimeter gradient
    perim, perim_grad = polygon_perimeter(vertices,jac=True)
    if np.abs(perim-1) > perim_tol:
        raise ValueError('perimeter is not 1')
    implicit_grad = -np.concatenate((perim_grad[1:-1].real,perim_grad[1:-1].imag))/(perim_grad[0].real)

    # get eigenvalues and eigenderivatives w.r.t vertices
    eigs, eigs_jac = dirichlet_eigenvalues(vertices,K,jac=True)

    # get normalized reciprocals
    nus, nus_jac = normalized_reciprocals(eigs,jac=True)

    # evaluate loss function
    loss, loss_grad = l2_loss(nus,targets,jac=True)

    # compose derivatives with chain rule
    obj_grad = (loss_grad.reshape(1,-1)@nus_jac)@(np.outer(eigs_jac[:,0],implicit_grad) + np.delete(eigs_jac,[0,N,N+1,2*N],axis=1))

    return loss, obj_grad

rho = (3-5**0.5)/2
def golden_search(f,a,b,tol=1e-14,maxiter=100):
    """Golden ratio minimization search"""
    h = b-a
    u, v = a+rho*h, b-rho*h
    fu, fv = f(u), f(v)
    i = 0
    while (b-a>=tol)&(i<=maxiter):
        i += 1
        if fu < fv:
            b = v
            h = b-a
            v = u
            u = a+rho*h
            fv = fu
            fu = f(u)
        else:
            a = u
            h = b-a
            u = v
            v = b-rho*h
            fu = fv
            fv = f(v)
    if f(a)<f(b): return a,i
    else: return b,i

def dirichlet_eigenvalues(*args):
    pass