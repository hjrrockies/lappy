import numpy as np
from lappy import *
from joblib import Parallel, delayed

def tri_eig(p,k,tol=1e-5):
    if param.perim_constraint(p) > tol:
        v = param.polygon_vertices(p)
        A,P = polygon_area(v),1
        weyl_1 = ((P+np.sqrt(P**2+16*np.pi*A))/(2*A))**2
        v *= np.sqrt(weyl_1)
        phis = interior_angles(v)
        alpha = np.pi/phis
        if np.ceil(alpha.max()) <= 100:
            try:
                evp1 = PolygonEVP(v,order=20)
                evp1.rtol = 1e-7
                eigs = evp1.solve_eigs_ordered(k+1)[0][:k]
                out = np.full(k,np.nan)
                out[:len(eigs)] = eigs
                return weyl_1*out
            except:
                pass
    return np.full(k,np.nan)

def run_test(n,k):
    p1,p2 = np.linspace(0.15,0.35,n),np.linspace(0.15,0.35,n)
    P1,P2 = np.meshgrid(p1,p2,indexing='ij')
    P = np.vstack((P1.flatten(),P2.flatten())).T

    res = Parallel(n_jobs=-1,verbose=10)(delayed(tri_eig)(p,k) for p in P)
    eigs = np.array(res)

    np.savez(f"tri_eig_{n}_{k}.npz",P=P,eigs=eigs)


if __name__=="__main__":
    from sys import argv
    n = int(argv[1])
    k = int(argv[2])

    run_test(n,k)