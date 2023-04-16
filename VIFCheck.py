import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import math

import seaborn as sns
import matplotlib.pyplot as plt

# 열을 지워가면서 VIF 확인
x_train.drop('Time', axis=11, inplace=True)

# heatmap을 통하여 상관관계 확인
sns.heatmap(x_train.corr(), annot=True, fmt='3.1f')
plt.show()

# OLS 사용하기 위해서 추가한 상수항
x_train_add = sm.add_constant(x_train, has_constant="add")
model_OLS = sm.OLS(y_train, x_train_add)
fit_model_OLS = model_OLS.fit()

print(fit_model_OLS.summary())

# 다중공산성 분석을 위한 코드
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(x_train.values, i)
              for i in range(x_train.shape[1])]

vif['Feature'] = x_train.columns

print(vif)

