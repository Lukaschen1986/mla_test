# -*- coding: utf-8 -*-
import numpy as np
A = np.random.randn(5,4)

# np.linalg.svd
U, S, VT = np.linalg.svd(A)

# svd_manual
def svd(A):
    # ATA
    ATA = A.T.dot(A)
    V_eigenvalue, V = np.linalg.eig(ATA)
    # AAT
    AAT = A.dot(A.T)
    U_eigenvalue, U = np.linalg.eig(AAT)
    # Sigma
    S = np.sqrt(V_eigenvalue)
    #S = np.linalg.inv(U).dot(A.dot(V))
    return U, S, V.T
