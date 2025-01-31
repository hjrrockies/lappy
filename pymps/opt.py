import numpy as np
import matplotlib.pyplot as plt

def eig_ratios(eigs):
    """Computes the ratios lambda_j/lambda_1, assuming the eigenvalues are
    indexed along the last dimension of the array"""
    return (eigs.T[1:]/eigs.T[0]).T

def weighted_p_loss(X,y,weights=1,p=2):
    """Computes the weighted Euclidean p-loss for X against y, where X is an
    array of simulated values and y is a vector of targets"""
    return (np.abs(X-y)**p).sum(axis=-1)
