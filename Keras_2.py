# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

model = Sequential([Dense(32, units=784), Activation('relu'), Dense(10), Activation('softmax'),])

# set model and add layers
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
#model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

# compile
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# loss: http://keras-cn.readthedocs.io/en/latest/other/objectives/
# optimizers: http://keras-cn.readthedocs.io/en/latest/other/optimizers/
# metrics: http://keras-cn.readthedocs.io/en/latest/other/metrices/

# train http://keras-cn.readthedocs.io/en/latest/models/sequential/
# fit
fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None, 
    validation_split=0.0, validation_data=None, shuffle=True, 
    class_weight=None, sample_weight=None, initial_epoch=0)
fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)


# evaluate
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
train_on_batch(self, x, y, class_weight=None, sample_weight=None)
test_on_batch(self, x, y, sample_weight=None)

# predict
predict(self, x, batch_size=32, verbose=0)
predict_on_batch(self, x)

labels = np.random.randint(10, size=(1, 1000))
to_categorical(labels, num_classes=10)


# Functional
from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training








layer = Dense(32)
layer.get_weights()
config = layer.get_config()
Dense.from_config(config)

layer.input









