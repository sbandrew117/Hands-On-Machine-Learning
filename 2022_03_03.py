import numpy as np
import matplotlib.pyplot as plt



X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

print(y_predict)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

##################################################################
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

print("\n intercept, coefficient: \n", lin_reg.intercept_, lin_reg.coef_)
print("\n new prediction: \n", lin_reg.predict(X_new))

##################################################################
eta = 0.1 #Learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1) #random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print(theta)
##################################################################
#확률적 경사 하강법 (learning_schedule 사용한 간단한 학습 스케줄)
n_epochs = 50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

theta1 = np.random.randn(2, 1) #random initialization

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) -yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
print(theta1)

##################################################################
#SGDRegressor, 학습률 0.1로 기본 학습 스케줄 사용, 에포크 50번 수행
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter = 50, penalty = None, eta0 = 0.1)
sgd_reg.fit(X, y.ravel())

print(sgd_reg.intercept_, sgd_reg.coef_)
##################################################################
from sklearn.preprocessing import PolynomialFeatures
#Polynomial regression

m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x ** 2 + x + 2 + np.random.randn(m, 1)
print(y)

poly_features = PolynomialFeatures(degree = 2, include_bias = False)
x_poly = poly_features.fit_transform(x)
print("\nx[0]:\n", x[0])

print("\nx_poly[0]:\n", x_poly[0])

lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)
print("\nintercept and coefficient:\n", lin_reg.intercept_, lin_reg.coef_)

##################################################################




