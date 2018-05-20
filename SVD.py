import numpy as np
nrow, ncol = 30, 4
A = np.random.uniform(low=-20, high=20, size=nrow*ncol).reshape(nrow,ncol)

U, S, VT = np.linalg.svd(A)

def svd(A):
    B = A.T.dot(A)
    B_eigenvalue, V = np.linalg.eig(B)
    C = A.dot(A.T)
    C_eigenvalue, U = np.linalg.eig(C)
    S = np.linalg.inv(U).dot(A.dot(V))
    U = np.round(U, 7)
    S = np.round(S, 7)
    V = np.round(V, 7)
    return U, S, V.T
U, S, VT = svd(A)
sigma = S
sigma[sigma != 0]
