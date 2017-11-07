# -*- coding: utf-8 -*-
"""
@author: xuefei.yang
"""
# =============================================================================
# Import Packages
# =============================================================================
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# =============================================================================
# Load Data
# =============================================================================
boston = load_boston()
boston.keys()
boston.data.shape
boston.feature_names
boston.DESCR

bos = pd.DataFrame(boston.data)    # numpy.ndarray -> pandas.Dataframe
bos.head()

bos.columns = boston.feature_names
bos['MEDV'] = boston.target
bos.head()

X = bos.drop('MEDV', axis = 1)
y = bos["MEDV"] 

"""
- CRIM per capita crime rate by town
- ZN proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS proportion of non-retail business acres per town
- CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX nitric oxides concentration (parts per 10 million)
- RM average number of rooms per dwelling
- AGE proportion of owner-occupied units built prior to 1940
- DIS weighted distances to five Boston employment centres
- RAD index of accessibility to radial highways
- TAX full-value property-tax rate per $10,000
- PTRATIO pupil-teacher ratio by town
- B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT % lower status of the population
- MEDV Median value of owner-occupied homes in $1000's
"""

# =============================================================================
# EDA
# =============================================================================
# describe info
bos.describe()

# distinct values in each column
bos.apply(lambda x: x.nunique(), axis=0)   # Usually axis=0 is said to be "column-wise" and axis=1 "row-wise"

# missing values
bos.isnull().sum()

# correlation matrix
corr = bos.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

# distribution of each feature
bos.iloc[:,:13].hist(layout=(4,4), figsize = (10,10))
np.log(bos['DIS']).hist()    # take log on columns with heavy tail
np.log(bos['LSTAT']).hist()

# pairplot: check linear relationship (seems only RM and LSTAT has linear relationship with MEDV)
sns.pairplot(bos, hue="CHAS")
sns.plt.show()

# multicolinearity
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif


# =============================================================================
# Model 1: scipy.stats (simple linear regression) 
# =============================================================================
X_s = bos["RM"]       
y_s = bos["MEDV"]   

slope, intercept, r_value, p_value, std_err = stats.linregress(X_s, y_s)

print("r-squared:", r_value**2)
print("p-value:", p_value)

plt.plot(X_s, y_s, 'o', label='original data')
plt.plot(X_s, intercept + slope*X_s, 'r', label='fitted line')
plt.legend()
plt.show()


# =============================================================================
# Model 2: statsmodels.api
# =============================================================================
model = sm.OLS(y, X).fit()
model.summary()

pred = model.predict(X)  
res = model.resid         

# Distribution of residuals
stats.probplot(res, dist='norm', plot=plt)
plt.title("Normal Q-Q plot")
plt.show()


# Feature Selection
# Fitting models using R-style formulas
model1 = smf.ols(formula='MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT', data=bos).fit()
#model1.summary()

model2 = smf.ols(formula='MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + np.log(DIS) + RAD + TAX + PTRATIO + B + np.log(LSTAT)', data=bos).fit()
model2.summary()  

# remove features with high VIF: PTRATIO, NOX
model3 = smf.ols(formula='MEDV ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + np.log(DIS) + RAD + B + np.log(LSTAT)', data=bos).fit()
model3.summary()

# remove insignificant features
model4 = smf.ols(formula='MEDV ~ CRIM + CHAS + NOX + RM + np.log(DIS) + B + np.log(LSTAT)', data=bos).fit()
model4.summary()


# =============================================================================
# Model 3: sklearn.linear_model
# =============================================================================
model_sk = LinearRegression()
model_sk.fit(X, y)
pred = model_sk.predict(X)
mse_lr = np.mean((y - pred)**2)
mse_lr

# Distribution of residuals
plt.hist(y-pred, bins = 30, color = 'blue', alpha = 0.5, )
plt.title("Error Distribution")
plt.show()

# =============================================================================
# Ridge Regression
# =============================================================================
# closed form solution
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
pred_ridge = ridge_reg.predict(X)
mse_ridge = np.mean((y - pred_ridge)**2)
mse_ridge


# stochastic gradient descent
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
pred_ridge2 = sgd_reg.predict(X)
mse_ridge2 = np.mean((y - pred_ridge2)**2)
mse_ridge2


# =============================================================================
# Lasso Regression
# =============================================================================
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
pred_lasso = lasso_reg.predict(X)
mse_lasso = np.mean((y - pred_lasso)**2)
mse_lasso


# =============================================================================
# Elastic Net
# =============================================================================
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)   # l1_ratio corresponds to the mix ratio r
elastic_net.fit(X, y)
pred_en = elastic_net.predict(X)
mse_en = np.mean((y - pred_en)**2)
mse_en
