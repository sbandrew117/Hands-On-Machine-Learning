'''
#조기 종료 (early stopping)
from sklearn.base import clone

#데이터 준비
poy_scaler = Pipeline([
    ("poly_features", PolynomialFeatures(degree = 90, include_bias=False)),
    ("std_scaler", StandardScaler())
])
x_train_poly_scaled = poly_scaler.fit_transform(x_train)
x_val_poly_scaled = poly.scaler.transform(x_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(x_train_poly_scaled, y_train) #훈련을 이어서 진행
    y_val_predict = sgd.reg.predict(x_val_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)
'''
##################################################################

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import datasets
iris = datasets.load_iris()
print("\niris keys:\n", list(iris.keys()))\
    
x = iris["data"][:, 3:] #꽃잎의 너비
y =(iris["target"] == 2).astype(np.int)
    
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x, y)

x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(x_new)

print(plt.plot(x_new, y_proba[:, 1], "g-", label = "Iris virginica"))
print(plt.plot(x_new, y_proba[:, 0], "b--", label = "Not Iris virginica"))

x = iris["data"][:, (2, 3)]
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(x, y)

print("\nsoftmax regression prediction:\n", softmax_reg.predict([[5, 2]]))
print("\nsoftmax regression prediction probability:\n", softmax_reg.predict_proba([[5, 2]]))

