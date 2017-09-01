# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:32:36 2017

@author: c00370249
"""

a = np.zeros((n,1))
b = 0
C = 0.6
max_iters = 200
tol = 0.001

    iters = 0
    while iters < max_iters:
        iters += 1
        a_prev = copy.deepcopy(a)
        
        for i in range(n):
            Ki = poly_kernel(X, X[i], 0, 1, 1)
            ui = (a*y).T.dot(Ki) + b
            Ei = ui - y[i]
            # weifan KKT: i
            if y[i]*Ei >= 1 and a[i] > 0 or y[i]*Ei <= 1 and a[i] < C:
                # Pick random i
                i = random_idx(n, j)
                # Error for i
                Ki = poly_kernel(X, X[i], 0, 1, 1)
                ui = (a*y).T.dot(Ki) + b # y_hat
                Ei = ui - y[i]
                # 更新上下限
                L, H = find_bounds(y, i, j)
                # 计算eta
                eta = K[i,i] + K[j,j] - 2*K[i,j]
#                if eta <= 0:
#                    continue
                # Save old alphas
                ai_old, aj_old = a[i], a[j]
                # Update alpha
                a[j] = aj_old + y[j]*(Ei-Ej)/eta
                a[j] = clip(a[j], L, H)
                a[i] = ai_old + y[j]/y[i]*(aj_old-a[j])
                # Find intercept
                b1 = b - y[i] + ui - y[i]*(a[i]-ai_old)*K[i,i] - y[j]*(a[j]-aj_old)*K[i,j]
                b2 = b - y[j] + uj - y[i]*(a[i]-ai_old)*K[i,j] - y[j]*(a[j]-aj_old)*K[j,j]
                if 0 < a[i] < C:
                    b = b1
                elif 0 < a[j] < C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)
            else:
                continue
            # Check convergence
            diff = np.sqrt(np.sum((a-a_prev)**2))
            if diff < tol:
                break
