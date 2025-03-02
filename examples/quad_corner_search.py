import numpy as np
from pymps import *
from joblib import Parallel, delayed

def quad_eig(v1,k,L,H,tol=1e-5):
    v = np.array([0,L,v1[0]+1j*v1[1],1j*H])
    phis = interior_angles(v)
    alpha = np.pi/phis
    if np.ceil(alpha.max()) <= 100:
        try:
            evp1 = PolygonEVP(v,order=20)
            eigs = evp1.solve_eigs_ordered(k+1)[0][:k]
            out = np.full(k,np.nan)
            out[:len(eigs)] = eigs
            return out
        except:
            pass
    return np.full(k,np.nan)

def run_test(n,k,L,H,dx):
    x,y = np.linspace(L-dx,L+dx,n),np.linspace(H-dx,H+dx,n)
    X,Y = np.meshgrid(x,y,indexing='ij')
    XY = np.vstack((X.flatten(),Y.flatten())).T

    res = Parallel(n_jobs=-1,verbose=10)(delayed(quad_eig)(xy,k) for xy in XY)
    eigs = np.array(res)

    np.savez(f"corner_eig_{n}_{k}_{np.rint(L)}_{np.rint(H)}.npz",X=X,Y=Y,eigs=eigs)


if __name__=="__main__":
    from sys import argv
    n = int(argv[1])
    k = int(argv[2])
    L = float(argv[3])
    H = float(argv[4])
    dx = float(argv[5])

    run_test(n,k,L,H,dx)