import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

trainData = pd.read_csv("./train.csv")         # train.csv 값 로드
print(trainData.info())                        # train.csv 데이터 형태 확인
print(trainData['price'].unique())              # object에서 숫자로 변환을 위한 고유값 확인
trainData['price'].replace({'J': 0, 'K': 1, 'O': 2}, inplace=True)
print(trainData['price'].unique())              # 변환된 숫자값 확인
# trainData[['Oil']].astype(float)       # train.csv 데이터 형태 단순 변환

















# 결측치 처리 : Oil
print(trainData.isnull().sum())
# train.csv 내 Null 값이 있는지 확인
trainData.loc[train_data.Oil.isnull(), 'Oil'] = 0                              # 처리 : 0
trainData.loc[train_data.Oil.isnull(), 'Oil'] = train_data.F_Consump.mean()    # 처리 : 평균
trainData.loc[train_data.Oil.isnull(), 'Oil'] = np.nan                         # 처리 : NAN
trainData.dropna(inplcae = True)                                                           # 처리 : NAN 값 삭제


# 이상값 처리 : Q1 - 1.5IQR < Tukey < Q3 + 1.5IQR
def OVutlineProcess(data):
    q3 = data.quantile(0.75)
    q1 = data.quantile(0.25)
    iqr = q3-q1

    upperBound = q3 + (iqr*1.5)
    lowerBound = q1 - (iqr*1.5)
    return data[(data > upperBound)|(data < lowerBound)].index    # 이상값 행 인덱스 찾기

outliers = OutlineProcess(trainData.Oil)
print(len(outliers))                                              # 이상값 존재 여부 확인
print(trainData.iloc[outliers])                                 # 이상값 출력
trainData.drop(outliers, inplace=True)                            # 이상값 삭제
print(trainData.info())                                           # 이상값 삭제 확인

# 상관 검증을 통해서 독립 변수만 걸러내는 작업 수행 필요

# 독립변수/종속변수 나누기
y_train = trainData[['Oil']]
x_train = trainData.drop('Oil', axis=1)

print(x_train.columns.values)
print(y_train.columns.values)

# train내에서 train/test set으로 나누기
x_t_train, x_t_test, y_t_train, y_t_test = train_test_split(x_train, y_train, train_size=0.8, test_size=0.2)
# train set을 지속적으로 바꾸고 싶으면, shuffle=True 값으로 설정 할 것


# Target value의 range 확인 - 정리 필요
for i in range(len(y_test)):
    if y_test[i] < 0:
        y_test[i] = 0
    else:
        pass
