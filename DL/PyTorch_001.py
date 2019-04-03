import time
import numpy as np
import pandas as pd
import torch as tc
from torch import autograd
import torch.nn.functional as f
print("Pytorch version:", tc.__version__)
print("use gpu:", tc.cuda.is_available())

# 1-无约束二次优化
# 参数初始化
w1 = tc.tensor(0.0, requires_grad=True)
w2 = tc.tensor(0.0, requires_grad=True)
# 目标函数
objt = lambda w1,w2: (w1**2 + w2 - 11)**2 + (w1 + w2**2 - 7)**2
# 优化算法
opti = tc.optim.SGD(params=[w1,w2], lr=0.01)

for step in range(1000):
    loss = objt(w1, w2)
    # 手工的方法
#    grads = autograd.grad(loss, [w1,w2])
#    w1 = w1 - 0.01*grads[0]
#    w2 = w2 - 0.01*grads[1]
    # 接口的方法
    opti.zero_grad()
    loss.backward()
    opti.step()
    # log
    if step % 100 == 0:
        print("step {0}: w1 = {1}, w2 = {2}, loss = {3}".format(step,w1,w2,loss))
    

# 2-一层全连接
# 模拟真实数据
x = tc.rand(100,2)
y = 0.2*tc.randn(100,1)
# 参数初始化
w = tc.randn([2,1], requires_grad=True)
b = tc.zeros([1], requires_grad=True)
# 假设函数
hypr = lambda x,w,b: tc.matmul(x,w) + b
# 目标函数
objt = lambda y,hypr: tc.mean((y-hypr)**2)
#objt = f.mse_loss(y, hypr)
# 优化算法
opti = tc.optim.SGD(params=[w,b], lr=0.01)

for step in range(1000):
    # 假设函数
    y_pred = hypr(x,w,b)
    # 目标函数
    loss = objt(y,y_pred)
    # 手工的方法
#    grads = autograd.grad(loss, [w,b])
#    w = w - 0.01*grads[0]
#    b = b - 0.01*grads[1]
    # 接口的方法
    opti.zero_grad()
    loss.backward()
    opti.step()
    # log
    if step % 100 == 0:
        print("step {0}: w = {1}, b = {2}, loss = {3}".format(step,w,b,loss))
