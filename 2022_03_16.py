import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

#텐서플로

'''
넘파이와 매우 비슷하지만 GPU 지원
분산 컴퓨팅 지원
계산 그래프 추출, 최적화에 좋음
계산 그래프를 다른 환경(linux)에서 실행 가능
자동 미분 기능, RMSProp, Nadam 같은 고성능 옵티마이저 제공
'''

#텐서와 연산
print("\n행렬:\n", tf.constant([[1, 2, 3], [4, 5, 6]]))
print("\n스칼라:\n", tf.constant(42))

t = tf.constant([[1, 2, 3], [4, 5, 6]])
print("\n shape of t:\n", t.shape) #t.shape -> 크기

print("\n t + 10\n", t + 10)

#numpy와 함께 이용하기
a = np.array([2, 4, 5])
tf.constant(a)
t.numpy()

#타입 변환
#tf.cast 이용
t2 = tf.constant(40, dtype=tf.float64)
print("\ntype change: \n", tf.constant(2.0) + tf.cast(t2, tf.float32))

#변수
#tf.Variable
v = tf.Variable([[1, 2, 3], [4, 5, 6]])
#.assign() 으로 변숫값 바꿀 수 있음
v.assign(2 * v) #-> [[2, 4, 6], [8, 10, 12]]
v[0, 1].assign(42) #-> [[2, 42, 6], [8, 10, 12]]
v[:, 2].assign([0, 1]) #-> [[2, 42, 0], [8, 10, 1]]
v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100, 200])
#-> [[100, 42, 0], [8, 10, 200]]
                
#후버 손실 -> 평균 제곱 오차 대신 이용
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

from keras.layers import Input, Dense
from keras.models import Model

#사용자 정의 요쇼를 가진 모델을 저장하고 로드하기

model = keras.models.load_model("my_model_with_a_custom_loss.h5",
                                custom_objects={"huber_fn" : huber_fn})

######## ~ page 478 ########