import os
import pandas as pd
import numpy as np

import math
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine # dataset
from sklearn.datasets import load_iris, load_boston
import warnings
warnings.filterwarnings('ignore')

# 1. dataset loading
X, y = load_wine(return_X_y = True,)
X.shape # (178, 13)
y.shape # (178,)


# 2. train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# 3. RF model 생성
model = RandomForestClassifier() # 분류 트리(default) 객체 생성
model.fit(X=X_train, y=y_train)


# 4. model 평가
y_pred = model.predict(X = X_test)
y_true = y_test

acc = accuracy_score(y_true, y_pred)
print('분류정확도 =', acc)

# con_mat = confusion_matrix(y_true, y_pred)
# print(con_mat)

report = classification_report(y_true, y_pred)
print(report)

'''
n_estimators=10 : raw sampling 수
min_samples_split=2 : 설명변수 개수 = 변수갯수에 루트
'''

model = RandomForestClassifier(n_estimators=400, min_samples_split=3)
model.fit(X=X_train, y=y_train)

y_pred = model.predict(X=X_test)
y_true = y_test 

acc = accuracy_score(y_true, y_pred) 

print('분류정확도 =', acc)



## Difficult
print("\n\nIris Example")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[x] for x in iris.target]

X = df.drop('species', axis=1)
y = df['species']


grid_params = {
    'n_estimators': (100, 300),
    'max_depth': (2, 10),
    'min_samples_leaf': (5, 18),
    'min_samples_split': (5, 16)
    # 'criterion': 'entropy'
}

# n_jobs=-1 : CPU Full Power
# CV : Coss Validation
# rf_clf = RandomForestClassifier(random_state=100, n_jobs=-1, criterion='entropy',
#                                 oob_score=True)

rf_clf = RandomForestClassifier(
    criterion='entropy', ## 불순도 측도
    # max_features='sqrt', ## 매 분리시 랜덤으로 뽑을 변수 개수
    # max_samples=1.0, ## 붓스트랩 샘플 비율 => 1이면 학습데이터를 모두 샘플링한다.
    # bootstrap=True, ## 복원 추출,  False이면 비복원 추출
    oob_score=True, ## Out-of-bag 데이터를 이용한 성능 계산
    random_state=100
)
grid_cv = GridSearchCV(rf_clf, param_grid=grid_params, cv=4, n_jobs=-1)
grid_cv.fit(X, y)

print('Grid CV :', grid_cv.best_params_)
print('Gird CV 최고 정확도:', grid_cv.best_score_)

# Random Forest 정확도 판별
rf_clf_validation_run = grid_cv.best_estimator_
y_valid = rf_clf_validation_run.predict(X)

print(y_valid)
print(y)

## 성능 평가
# print(rf_clf_validation_run.oob_score_) ## Out-of-bag 성능 평가 점수
print('Grid CV 정확도 : ', accuracy_score(y_valid,y)) ## 테스트 성능 평가 점수(Accuracy)
print()


clf = RandomForestClassifier(
    # n_estimators=50, ## 붓스트랩 샘플 개수 또는 base_estimator 개수
    # criterion='entropy', ## 불순도 측도
    # max_depth=5, ## 개별 나무의 최대 깊이
    # max_features='sqrt', ## 매 분리시 랜덤으로 뽑을 변수 개수
    # max_samples=1.0, ## 붓스트랩 샘플 비율 => 1이면 학습데이터를 모두 샘플링한다.
    # bootstrap=True, ## 복원 추출,  False이면 비복원 추출
    # oob_score=True, ## Out-of-bag 데이터를 이용한 성능 계산
    # random_state=100
).fit(X, y)



## 예측
y_pred=clf.predict(X)
## 성능 평가
# print(clf.oob_score_) ## Out-of-bag 성능 평가 점수
print('일반 정확도 : ', accuracy_score(y_pred,y)) ## 테스트 성능 평가 점수(Accuracy)
print()

importances_values = clf.feature_importances_
ftr_importances = pd.Series(importances_values, index=X.columns)
ftr_top = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8, 6))
sns.barplot(x=ftr_top, y=ftr_top.index)
plt.show()







#


# # y_check = y_t_test.to_numpy()
# sample = y_check - y_valid
# Random Forest의 유의 변수 확인하기