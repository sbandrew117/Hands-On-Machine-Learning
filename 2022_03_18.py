import pandas as pd
import numpy as np

#시계열 예측하기
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10)) # 사인 곡선 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + 사인 곡선 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5) # + 잡음
    return series[..., np.newaxis].astype(np.float32)

n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1] # 훈련세트
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1] # 검증 세트
X_test, y_test = series[9000:, :n_steps], series[9000:, -1] # 테스트 세트

#순진한 예측 만들어 성능 비교하기
#순진한 예측
import tensorflow as tf
from tensorflow import keras

y_pred = X_valid[:, -1]
print(np.mean(keras.losses.mean_squared_error(y_valid, y_pred)))

#or
#완전 연결 네트워크 사용
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(1)
])

#간단한 RNN 구현하기
model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1])
])

#심층 RNN
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.SimpleRNN(1),
])

#SImple RNN은 활성화 함수를 tanh 사용. 출력층을 Dense 층으로 바꾸기 -> 더 빠르고 정확도는 비슷
model = keras.smodels.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])

#다음 스텝 예측하기
#1. 이미 훈련된 모델을 사용하여 다음 값을 예측한 다음 이 값을 입력으로 추가하는 것
series = generate_time_series(1, n_steps + 10)
X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
X = X_new
for step_ahead in range(10):
    y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
    X = np.concatenate([X, y_pred_one], axis=1)
    
Y_pred = X[:, n_steps:]

#2. RNN을 훈련하여 다음 값 10개를 한번에 예측하는 것 -> 시퀀스-투-벡터
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

#1개의 유닛이 아니라 10개의 유닛을 가진 출력층이 필요
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
])

Y_pred = model.predict(X_new)

#Seq-2-Seq 모델로 변환 -> 모든 층에(마지막 층도) return_sequences=True 지정해야 함
#그 다음 모든 타임 스텝에서 출력을 .Dense 층에 적용해야함 -> TimeDistributed층 사용
model = keras.models.Sequential([
     keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10)) #출력층 10개
])

#마지막 출력에 대한 MSE만 계산하는 사용자 정의 지표
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

optimizer = keras.optimizers.Adam(lr=0.01)
model.compile(loss="mse", optimizer=optimizer, metrics=[last_time_step_mse])

#긴 시쿼스 다루기







