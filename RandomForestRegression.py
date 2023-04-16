import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy import stats
import math

import matplotlib.pyplot as plt
import seaborn as sns

# Random Forest를 위한 Parameter Grid Search
grid_params = {
    'n_estimators': (100, 300),
    'max_depth': (5, 10),
    'min_samples_leaf': (5, 18),
    'min_samples_split': (5, 16)
}

# n_jobs=-1 : CPU Full Power
# CV : Coss Validation
rf_run = RandomForestRegressor(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_run, param_grid=grid_params, cv=4, n_jobs=-1)
grid_cv.fit(x_t_train, np.ravel(y_t_train))

print('Grid CV :', grid_cv.best_params_)
print('Gird CV 최고 정확도:', grid_cv.best_score_)

# rf_run = RandomForestRegressor(random_state=0, max_depth=10, min_samples_leaf=5, min_samples_split=5,n_estimators=200)
# rf_run.fit(x_t_train, np.ravel(y_t_train))

# train_predict = rf_run.predict(x_t_train)
# print("RMSE':{}".format(math.sqrt(mean_squared_error(train_predict, y_t_train))))

# Random Forest 정확도 판별
rf_validation_run = grid_cv.best_estimator_
y_valid = rf_validation_run.predict(x_t_test)
# valid_predict = rf_run.predict(x_t_test)
# print("RMSE':{}".format(math.sqrt(mean_squared_error(valid_predict, y_t_test))))
print("RMSE':{}".format(math.sqrt(mean_squared_error(y_valid, y_t_test))))

# Random Forest의 유의 변수 확인하기
importances_values = rf_run.feature_importances_
ftr_importances = pd.Series(importances_values, index=x_t_train.columns)
ftr_top = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8, 6))
sns.barplot(x=ftr_top, y=ftr_top.index)
plt.show()


