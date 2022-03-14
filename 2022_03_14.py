import tensorflow as tf
from tensorflow import keras

print("\ntensorflow version:\n", tf.__version__)

print("\nkeras version:\m", keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape)
print(X_train_full.dtype)

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_tst = X_test / 255.0

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(class_names[y_train[0]])

#신경망 만들기
model = keras.models.Sequential() #Sequential 모델 만들기
model.add(keras.layers.Flatten(input_shape=[28, 28])) #첫번째 층을 만들고 모델 추가 -> flatten: 입력 이미지를 1D 배열로 변환 (X.reshape(-1, 1))
model.add(keras.layers.Dense(300, activation = "relu")) #뉴런 300개 가진 Dense 은닉층 추가: ReLU 활성화 함수 사용
model.add(keras.layers.Dense(100, activation = "relu")) #뉴런 100개 가진 Dense 은닉층 추가: ReLU 활성화 함수 사용
model.add(keras.layers.Dense(10, activation = "softmax")) #뉴런 10개 가진 Dense 출력층 추가: 소프트맥스 활성화 함수 사용

#activaiton = "relu" == actiavation = keras.activations.relu
''' # 위와 동일한 코드
model =keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28, 28]),
    keras.layers.Dense(300, activation = "relu"),
    keras.layers.Dense(100, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax")
])
'''

print("\nmodel summary:\n", model.summary())

print("\nmodel layers:\n", model.layers)

hidden1 = model.layers[1]
print("\nhidden1 layer name:\n", hidden1.name)

#Dense 층에는 weights와 biases 가 모두 포함되어 있음
#모델을 만들고 나서 compile() 메서드를 호출하여 사용할 손실 함수와 옵티마이저를 지정

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "sgd",
              metrics = ["accuracy"])
              
#optimizer = keras.optimizers.SGD(lr=???) -> 학습률을 튜닝하는 것이 중요. 기본값 = 0.01

#모델 훈련과 평가
#모델을 훈련하려면 fit() 메서드 호출

history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_valid, y_valid))

import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
#모델의 검증 정확도가 만족스러울 때 모델을 상용 환경으로 배포하기 전 테스트 세트로 모델 평가하여 일반화 오차 추정

print("\nmodel evaluate:\n", model.evaluate(X_test, y_test))

#모델 사용하여 예측 만들기
#predict() -> 새로운 샘플에 대해 예측 만들기

X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))

import numpy as np

#predict_classes() -> 가장 높은 확률을 가진 클래스에만 관심이 있을 때
y_pred = model.predict_classes(X_new) #version 차이로 오류


print(y_pred)
print(np.array(class_names)[y_pred])

y_new = y_test[:3]
print(y_new)

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full)

#스케일 조정 -> 훈련세트, 검증세트, 테스트세트
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X.train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", optiizer="sgd")
history = model.fit(X_train, y_train, epochs = 20,
                    validation_data = (X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3] #new sample
y_pred = model.predict(X_new)







