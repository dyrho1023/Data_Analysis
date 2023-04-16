import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy import stats
import math

trainData = pd.read_csv("./train.csv")         # train.csv 값 로드
print(trainData.info())                        # train.csv 데이터 형태 확인
print(trainData['price'].unique())              # object에서 숫자로 변환을 위한 고유값 확인
trainData['price'].replace({'J': 0, 'K': 1, 'O': 2}, inplace=True)
print(trainData['price'].unique())              # 변환된 숫자값 확인
# trainData[['Oil']].astype(float)       # train.csv 데이터 형태 단순 변환

testData = pd.read_csv("./test.csv")
submissionData = pd.read_csv("./sample_submission.csv")


# train_train set / train_test set 나누기
y_train = trainData[['Oil']]
x_train = trainData.drop('Oil', axis=1)


# train_train set / train_test set 나누기
testData['price'].replace({'J': 0, 'K': 1, 'O': 2}, inplace=True)
x_test = testData
y_submission = submissionData.drop('Oil', axis=1)


# 이상값 처리 : Q1 - 1.5IQR < Tukey < Q3 + 1.5IQR
def Tukey(data):
    q3 = data.quantile(0.75)
    q1 = data.quantile(0.25)
    iqr = q3-q1

    upperBound = q3 + (iqr*1.5)
    lowerBound = q1 - (iqr*1.5)
    return data[(data > upperBound)|(data < lowerBound)].index    # 이상값 행 인덱스 찾기


outliers = Tukey(trainData.Oil)
print(len(outliers))                                              # 이상값 존재 여부 확인
# print(trainData.iloc[outliers])                                 # 이상값 출력
trainData.drop(outliers, inplace=True)                            # 이상값 삭제
print(trainData.info())                                           # 이상값 삭제 확인

# 상관관계 검증 - 여기서 이상한 값을 뱉는 Data는 직접 보고 데이터 분석에서 제외

columName = x_train.columns.values
for i in columName:
    print('\n', i)
    print(stats.spearmanr(x_train[[i]], y_train))                # 스피어만 p value가 0.05 이하만
    print(stats.pearsonr(x_train[[i]], y_train))                 # 피어슨 p value가 0.05 이하만

x_train.drop('Time', axis=1, inplace=True)
# print(x_train.info())

line_fitter = LinearRegression()
line_fitter.fit(x_train, y_train)

y_test = line_fitter.predict(x_test)

print('정확도:', line_fitter.score(x_train, y_train))

# Target value의 range 확인
for i in range(len(y_test)):
    if y_test[i] < 0:
        y_test[i] = 0
    else:
        pass

# Submission File
y_temp = pd.DataFrame(y_test)
y_temp.columns = ['Oil']
y_submission = pd.concat([y_submission, y_temp], axis=1)
print(y_submission)

y_submission.to_csv('result.csv', index=False)

