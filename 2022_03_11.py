import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, dbscan
from sklearn.datasets import make_moons
from sympy import N

#DBSCAN
x, y = make_moons(n_samples=1000, noise=0.05)
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(x)

print("\ndbscan labels:\n", dbscan.labels_) #결과물이 -1일시 이 샘플을 이상치로 판단했다는 의미

print("\ncore sample indices:\n", dbscan.core_sample_indices_) #핵심 샘플 인덱스 확인

print("\ncore components\n", dbscan.components_)


#Gaussian Mixture Model

from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(x)

print("\ngm weights:\n", gm.weights_)

print("\ngm means:\n", gm.means_)

print("\ngm covariances:\n", gm.covariances_)

print("\nBIC:\n", gm.bic(x))
print("\nAIC:\n", gm.aic(x))

#BayesianGaussianMixture -> 자동으로 불필요한 클러스터의 가중치를 0으로 만듦
from sklearn.mixture import BayesianGaussianMixture
bgm = BayesianGaussianMixture(n_components=10, n_init=10)
bgm.fit(x)
print("\nbgm weights:\n", np.round(bgm.weights_, 2))


#Rest is in the Word(pdf) file attached inside "Clickup"