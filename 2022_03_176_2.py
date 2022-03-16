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

    