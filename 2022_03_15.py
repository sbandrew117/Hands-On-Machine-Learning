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

