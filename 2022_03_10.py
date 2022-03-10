from random import Random

from statistics import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

#make_moons dataset
x, y = make_moons(n_samples=500, noise=0.30, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

#다른 알고리즘으로 학습 -> 앙상블 모델의 정확성 높임.
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting = 'hard'
)

voting_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
#테스트셋의 정확도 확인
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("\naccuracy of each algo:\n", clf.__class__.__name__, accuracy_score(y_test, y_pred))

#bagging ensemble (중복 허용)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1) #n_jobs = CPU 수, bootstrap=True(False) -> bagging(pasting)

bag_clf.fit(x_train, y_train)

#oob(out-of-bag) -> 선택되지 않은 훈련 샘플의 나머지도 평가하기
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, oob_score=True) #oob_score=True

bag_clf.fit(x_train, y_train)
#oob평가에서의 정확도
print("\nwith oob:\n", bag_clf.oob_score_)

#테스트세트에서의 정확도
from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(x_test)
print("\naccuracy score:\n", accuracy_score(y_test, y_pred))

#훈련 샘플이 양성/음성 클래스에 속할 확률 구하기
print("\nprob of oob/non-oob:\n", bag_clf.oob_decision_function_)


#random-forest classifier ->  baggingClassifier에 DecisionTreeClassifier --> RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

#RandomForestClassifier 이용 (전체 특성 중에서 최선의 특성을 찾는 대신
# 무작위로 선택한 특성 후보 중에서 최적의 특성을 찾는 식으로 무작위성을 더 주입)
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(x_train, y_train)

y_pred_rf = rnd_clf.predict(x_test)

#BaggingCLassifier로 유사하게 만들기
'''
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features = "auto", max_leaf_nodes = 16),
    n_estimators = 500, max_samples = 1.0, bootstrap = True, n_jobs = -1
)
'''

#ExtraTreesClassifier -> 일반적인 랜덤 포레스트보다 훨씬 빠름(최적의 임곗값을 찾는 대신
# 후보 특성을 사용해 무작위로 분할한 다음 그 중에서 최상의 분할을 선택)

#특성 중요도
#adaboost -> 반복마다 샘플의 가중치 수정
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(x_train, y_train)

#gradient boosting -> 이전 예측기가 만든 residual error에 새로운 예측기 학습
#GBRT(gradient boosted regression tree)
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(x, y)

#두번째 예측기
y2 = y - tree_reg1.predict(x)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(x, y2)

#세번째 예측기
y3 = y2 - tree_reg2.predict(x)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(x, y3)

x_new = np.array([[0.8]])

#모든 트리의 예측 더하기
#y_pred = sum(tree.predict(x_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
y_pred = sum(tree.predict(x_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=0.1, random_state=42)
gbrt.fit(x, y)

#조기종료(학습률이 낮을 때 최적의 트리 수로 축소)를 사용한 gradient boosting
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

x_train, x_val, y_train, y_val = train_test_split(x, y, random_state=49)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42) #120개의 트리
gbrt.fit(x_train, y_train)

errors = [mean_squared_error(y_val, y_pred)
          for y_pred in gbrt.staged_predict(x_val)] #staged_predict() -> 조기 종료
bst_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(x_train, y_train)

#다섯 번의 반복 동안 검증 오차가 향상되지 않으면 훈련 멈춤
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(x_train, y_train)
    y_pred = gbrt.predict(x_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # 조기 종료

#Stochastic Gradient boosting
#subsample -> GradientBoostingRegressor이 각 트리가 훈련할 때 사용할 훈련 샘플의 비율 지정
#편향 높아지고 분산 낮아지며 훈련 속도 높임

import xgboost
xgb_reg = xgboost.XGBRegressor(random_state=42)
xgb_reg.fit(x_train, y_train)
y_pred = xgb_reg.predict(x_val)

#자동 조기 종료
xgb_reg.fit(x_train, y_train,
            eval_set=[(x_val, y_val)], early_stopping_rounds=2)
y_pred = xgb_reg.predict(x_val)


#차원 축소
x_centered = x - x.mean(axis = 0)
u, s, vt = np.linalg.svd(x_centered)
c1 = vt.T[:, 0]
c2 = vt.T[:, 1]

print("\n", c1)
print("\n", c2)

#d차원으로 투영하기
w2 = vt.T[:, :2]
x2d = x_centered.dot(w2)

#scikit learn 활용하기
from sklearn.decomposition import PCA

pca = PCA(n_components=2) #데이터셋의 차원을 2로 줄이기
x2d = pca.fit_transform(x)

#설명된 분산의 비율
print("\nexplained variance ratio:\n", pca.explained_variance_ratio_)

'''
#적절한 차원 수 선택하기
pca = PCA()
pca.fit(x_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

pca = PCA(n_components=0.95)
x_reduced = pca.fit_transform(x_train)

pca = PCA(n_components=154)
x_reduced = pca.fit_transform(x_train)
x_recovered = pca.inverse_transform(x_reduced)

#d가 n보다 많이 작을시 svd보다 빠른 방법
#randomized
rnd_pca = PCA(n_components=154, svd_solver = "randomized") #svd_solver의 기본값: auto
x_reduced = rnd_pca.fit_transform(x_train)

#점진적 PCA(IPCA) -> 훈련 세트를 미니배치로 나눈 뒤 IPCA 알고리즘 한번에 하나씩 주입
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for x_batch in np.array_split(x_train, n_batches):
    print(".", end="") # not shown in the book
    inc_pca.partial_fit(x_batch) #fit이 아닌 partial_fit

x_reduced = inc_pca.transform(x_train)

#memmap 과 fit 이용하기
filename = "my_mnist.data"
m, n = x_train.shape

X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))
X_mm[:] = x_train

x_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m,n))

batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(x_mm)

#kernel PCA
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(x)

#hyperparameter 튜닝
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)), #2차원으로 축소
        ("log_reg", LogisticRegression(solver='liblinear'))
    ])

param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

grid_search = GridSearchCV(clf, param_grid, cv=3) #kPCA의 가장 좋은 커널과 gamma 파라미터 찾기 -> 가장 높은 분류 정확도를 얻기 위해
grid_search.fit(x, y)

print(grid_search.best_params_)

#재구성하는 code
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
                    fit_inverse_transform=True)
x_reduced = rbf_pca.fit_transform(x)
x_preimage = rbf_pca.inverse_transform(x_reduced)

from sklearn.metrics import mean_squared_error

print("\npre-image error:\n", mean_squared_error(x, x_preimage))

'''

#LLE -> 비선형
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
x_reduced = lle.fit_transform(x)


