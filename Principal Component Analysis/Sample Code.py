# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 14:53:06 2017

@author: xuefei.yang
"""

# =============================================================================
# sklearn.decomposition.PCA
# =============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv')
data = data.drop(["Channel", "Region"], axis=1)
data.head()



from sklearn.decomposition import PCA

pca = PCA(n_components=6).fit(data)
pca_samples = pca.transform(data)

plt.plot(pca.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# https://analyticsdefined.com/implementing-principal-component-analysis/