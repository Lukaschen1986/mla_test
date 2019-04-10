# -*- coding: utf-8 -*-
# 对比 交叉熵 和 平方误差 两个损失函数，交叉熵 在分类问题中能够获得更大的梯度，且损失函数值不至于过小，因此更有利于优化
import numpy as np

# 交叉熵
objt_1 = lambda y, yhat: -y*np.log(yhat) - (1-y)*np.log(1-yhat)
grad_1 = lambda y, yhat: -y/yhat + (1-y)/(1-yhat)

# 平方误差
objt_2 = lambda y, yhat: (y - yhat)**2
grad_2 = lambda y, yhat: -2*(y-yhat)

y = 1
yhat = 0.999
objt_1(y, yhat)
objt_2(y, yhat)
grad_1(y, yhat)
grad_2(y, yhat)
'''
y	yhat	objt_1	objt_2	grad_1	grad_2
1	0.999	0.0010 	0.0000 	-1.0010 	-0.0020 
1	0.99	0.0101 	0.0001 	-1.0101 	-0.0200 
1	0.9	0.1054 	0.0100 	-1.1111 	-0.2000 
1	0.8	0.2231 	0.0400 	-1.2500 	-0.4000 
1	0.1	2.3026 	0.8100 	-10.0000 	-1.8000 
'''
