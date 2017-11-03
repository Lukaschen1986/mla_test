https://zhuanlan.zhihu.com/p/23748037
http://nooverfit.com/wp/keras-%E6%89%8B%E6%8A%8A%E6%89%8B%E5%85%A5%E9%97%A81-%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98/
https://zhuanlan.zhihu.com/p/25249694
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import os
os.getcwd()
import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from sklearn import datasets
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


# 训练集，测试集收集非常方便
(X_init, y), (X_predict_init, y_predict) = mnist.load_data()
X_init.shape # (60000, 28, 28)

# 输入的图片是28*28像素的灰度图
img_rows, img_cols = X_init.shape[1], X_init.shape[2]

X_init = X_init.reshape(X_init.shape[0], img_rows*img_cols)
X_predict_init = X_predict_init.reshape(X_predict_init.shape[0], img_rows*img_cols)
val_max = np.max(X_init)
X = X_init / val_max
X_predict = X_predict_init / val_max

# svm
clf = svm.SVC()
t0 = pd.Timestamp.now()
clf.fit(X, y)
t1 = pd.Timestamp.now(); t1-t0 # 00:10:16.450259
#clf.support_vectors_
#len(clf.support_) / len(y) # 0.3271
y_pred = clf.predict(X_predict)
sum(y_pred == y_predict) / len(y_predict) # 0.9446

# LR
clf = LogisticRegression(penalty="l2", tol=0.0001, solver="lbfgs", max_iter=100, multi_class="multinomial")
t0 = pd.Timestamp.now()
clf.fit(X, y)
t1 = pd.Timestamp.now(); t1-t0 # 00:00:48.229758
y_pred = clf.predict(X_predict)
sum(y_pred == y_predict) / len(y_predict) # 0.9256


# MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(70,70,70,70,70), activation="relu", solver="adam", alpha=0.01, batch_size=128, learning_rate_init=0.001, max_iter=10, tol=10**-4, beta_1=0.9, beta_2=0.999, epsilon=10**-8)
t0 = pd.Timestamp.now()
clf.fit(X, y)
t1 = pd.Timestamp.now(); t1-t0 # 00:01:09.082951
y_pred = clf.predict(X_predict)
sum(y_pred == y_predict) / len(y_predict) # 0.9723



# keras输入数据有两种格式，一种是通道数放在前面，一种是通道数放在后面，其实就是格式差别而已
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 把数据变成float32更精确
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

val_max = np.max(x_train)
x_train /= val_max
x_test /= val_max
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train_2 = x_train.reshape(x_train.shape[0], img_rows*img_cols)



# batch_size 太小会导致训练慢，过拟合等问题，太大会导致欠拟合。所以要适当选择
batch_size = 128
# 0-9手写数字一个有10个类别
num_classes = 10
# 12次完整迭代，差不多够了
epochs = 10


# 把类别0-9变成2进制，方便训练
#one_hot = lambda y: np.eye(len(set(y)))[y]
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

# 牛逼的Sequential类可以让我们灵活地插入不同的神经网络层
model = Sequential()
# 加上一个2D卷积层， 32个输出（也就是卷积通道），激活函数选用relu，
# 卷积核的窗口选用3*3像素窗口
model.add(Conv2D(32, activation='relu', input_shape=input_shape, nb_row=3, nb_col=3))
# 64个通道的卷积层
model.add(Conv2D(64, activation='relu', nb_row=3, nb_col=3))
# 池化层是2*2像素的
model.add(MaxPooling2D(pool_size=(2, 2)))
# 对于池化层的输出，采用0.35概率的Dropout
model.add(Dropout(0.35))
# 展平所有像素，比如[28*28] -> [784]
model.add(Flatten())
# 对所有像素使用全连接层，输出为128，激活函数选用relu
model.add(Dense(128, activation='relu'))
# 对输入采用0.5概率的Dropout
model.add(Dropout(0.5))
# 对刚才Dropout的输出采用softmax激活函数，得到最后结果0-9
model.add(Dense(num_classes, activation='softmax'))
# 模型我们使用交叉熵损失函数，最优化方法选用Adadelta
#model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=0.0))
# 令人兴奋的训练过程
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
