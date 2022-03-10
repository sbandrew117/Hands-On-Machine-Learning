from random import Random
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

