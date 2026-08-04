# Regularization of the GSVD subproblem
The core of the MPS eigensolve pipeline is to compute tension values from a GSVD of the matrix pencil ${A_B(\lambda), A_N(\lambda)}$ as lambda varies. The generalized singular values of the pencil can be ill-conditioned (as functions of the matrix entries), which makes the tension curves (the generalized singular values as functions of $\lambda$) themselves into ill-conditioned functions. There are several extant approaches to regularizing the GSVD subproblem:

1. 