# -*- coding: utf-8 -*-
# 快速开始：30s上手Keras
from keras.models import Sequential
model = Sequential() # 序贯模型

from keras.layers import Dense, Activation
model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(x_train, y_train, epochs=5, batch_size=32)
#model.train_on_batch(x_batch, y_batch)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)

from keras.models import load_model
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
model = load_model('my_model.h5')

from keras.models import model_from_json
json_string = model.to_json() # save as JSON
model = model_from_json(json_string)

model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5', by_name=True)

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='auto')
model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])

hist = model.fit(X, y, validation_split=0.2, shuffle=True)
print(hist.history)

# 冻结
frozen_layer = Dense(32,trainable=False) # “冻结”一个层指的是该层将不参加网络训练，即该层的权重永不会更新。在进行fine-tune时我们经常会需要这项操作。 在使用固定的embedding层处理文本输入时，也需要这个技术。
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # this does NOT update the weights of `layer`
trainable_model.fit(data, labels)  # this updates the weights of `layer`

# pop
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))
print(len(model.layers))  # "2"
model.pop()
print(len(model.layers))  # "1"

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
model = VGG16(weights='imagenet', include_top=True)

