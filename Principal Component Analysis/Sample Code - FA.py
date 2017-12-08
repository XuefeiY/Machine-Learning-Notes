# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:22:26 2017

@author: xuefei.yang
"""

# =============================================================================
# Import Packages
# =============================================================================
import pandas as pd
from sklearn import decomposition, preprocessing

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv')
data = data.drop(["Channel", "Region"], axis=1)
data.head()


# =============================================================================
# sklearn.decomposition.FactorAnalysis
# =============================================================================
# scaling the data before FA
data_normal = preprocessing.scale(data)

fa = decomposition.FactorAnalysis(n_components=2)
fa.fit(data_normal)

print (fa.components_)
df = pd.DataFrame(fa.components_.transpose(), index=data.columns, columns=['factor 1', 'factor 2'])
df


"""
                  factor 1  factor 2
Fresh            -0.047160  0.423627
Milk              0.732284  0.360762
Grocery           0.968583  0.058966
Frozen           -0.072645  0.564214
Detergents_Paper  0.961895 -0.122233
Delicassen        0.167762  0.722710


factor1: grocery, detergents_paper, milk
factor2: delicassen, frozen, fresh

The result is similar to that of PCA
"""



