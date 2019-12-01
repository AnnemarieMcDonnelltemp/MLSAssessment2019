import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams

from sklearn.datasets import load_boston
boston = load_boston()

print(boston.keys())

print(boston.data.shape)

print(boston.feature_names)

print(boston.DESCR)

bos = pd.DataFrame(boston.data)
print(bos.head())

bos.columns = boston.feature_names
print(bos.head())

print(boston.target.shape)

bos['PRICE'] = boston.target
print(bos.head())

print(bos.describe())



#new part

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import sklearn.model_selection

X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('seaborn-whitegrid')
import numpy as np

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print(mse)


