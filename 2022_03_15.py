from multiprocessing.dummy import active_children
import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(), #Batch Normalization 적용
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

print("\nmodel summary:\n", model.summary())


#활성화 함수 전에 배치 정규화 층을 추가하기 위해 은닉층에서 활성화 함수 지정 x. 배치 정규화 층 뒤에 별도의 층 추가
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(), #Batch Normalization 적용
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal", use_bias=False), #use_bias=False -> 층 만들기
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(10, activation="softmax")
])

optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="mse", optimizer=optimizer)

model_A = keras.models.load_model("my_model_a.h5")
model_B_on_A = keras.models.Sequential(model_A.layers[:1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

#model_A에 영향이 가기 때문에 일단 clone 작업
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
    
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

#재사용된 층의 동결을 해제한 후 학습률을 낮추는 것이 가중치가 망가지는 것 막아줌

#모멘텀 최적화
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

#네스테로프 가속 경사
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

#RMSProp
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)

#Adam
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#거듭제곱 기반 스케줄링
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
'''
#지수 기반 스케줄링
def exponential_decay_fn(epoch):
    return 0.01 * 0.1 ** (epoch / 20)
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train_scaled, y_train, [...], callbacks = [lr_scheduler])
'''
def exponential_decay_fn(epoch, lr):
    return lr* 0.1 ** (1/20)

#page 448 참고 및 복습

#l2 규제
layer = keras.layers.Dense(100, activation="elu",
                           kernel_initializer = "he_normal",
                           kernel_regularizer = keras.regularizers.l2(0.01))

#드롭아웃
model = keras.models.Seequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
])


import numpy as np
import pandas as pd

#monte carlo(MC) dropout
y_probas = np.stack([model(X_test_scaled, training=True)
                     for sample in range(100)])
y_proba = y_probas.mean(axis=0)
