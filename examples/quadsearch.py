import numpy as np
from lappy import *
from joblib import Parallel, delayed

def quad_eig(v1,k,tol=1e-5):
    v = np.array([0,1,v1[0]+1j*v1[1],1j])
    A,P = polygon_area(v),1
    weyl_1 = ((P+np.sqrt(P**2+16*np.pi*A))/(2*A))**2
    v *= np.sqrt(weyl_1)
    phis = interior_angles(v)
    alpha = np.pi/phis
    if np.ceil(alpha.max()) <= 100:
        try:
            evp1 = PolygonEVP(v,order=20)
            eigs = evp1.solve_eigs_ordered(k+1,ppl=30)[0][:k]
            out = np.full(k,np.nan)
            out[:len(eigs)] = eigs
            return weyl_1*out
        except:
            pass
    return np.full(k,np.nan)

def run_test(n,k):
    x,y = np.linspace(0.5,1.5,n),np.linspace(0.5,1.5,n)
    X,Y = np.meshgrid(x,y,indexing='ij')
    XY = np.vstack((X.flatten(),Y.flatten())).T

    res = Parallel(n_jobs=-1,verbose=10)(delayed(quad_eig)(xy,k) for xy in XY)
    eigs = np.array(res)

    np.savez(f"quad_eig_{n}_{k}.npz",X=X,Y=Y,eigs=eigs)


if __name__=="__main__":
    from sys import argv
    n = int(argv[1])
    k = int(argv[2])

    run_test(n,k)