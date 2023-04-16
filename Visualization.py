import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from scipy import stats
import matplotlib.pyplot as plt

y_valid = y_valid.reshape(4177,1)
print(y_t_test.shape, y_valid.shape)

plt.plot(np.array(y_t_test-y_predict), label="diff")
plt.legend()
plt.show()
