from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version = 1)


X, y = mnist["data"], mnist["target"]


some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.show(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
plt.axis("off")
plt.show()
