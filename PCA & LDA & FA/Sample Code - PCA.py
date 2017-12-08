# =============================================================================
# Import Packages
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv')
data = data.drop(["Channel", "Region"], axis=1)
data.head()

# =============================================================================
# EDA
# =============================================================================
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,10), diagonal='kde');
plt.show()

"""
From the above scatter-plots, it seems there is a linear relationship between the spending habits of milk, 
]grocery and detergents_paper items. Also, there might be a linear relationship between spending habits on fresh and frozen products. 
Let’s now try to analyze the dataset by creating six principal components
"""
# =============================================================================
# sklearn.decomposition.PCA
# =============================================================================
# scaling the data before PCA
data = pd.DataFrame(scale(data), columns=['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])

# implementing PCA
pca = PCA(n_components=6).fit(data)
pca_samples = pca.transform(data)

np.corrcoef(pca_samples.T)     #  the samples components do no longer carry any linear correlation

# =============================================================================
# Visualization
# =============================================================================
def pca_results(data, pca):
    
    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = data.keys()) 
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1) 
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance']) 
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar')
    ax.set_ylabel("Feature Weights") 
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios# 
    for i, ev in enumerate(pca.explained_variance_ratio_): 
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n %.4f"%(ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

pca_results = pca_results(data, pca)
pca_results.cumsum()

# Scree plot
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

"""
From the below plot we can observe that we got that elbow in the plot corresponding to the 2nd number of principal components. 
Hence we should use only two number of principal components in our analysis.
"""

# Biplot
pca = PCA(n_components=2).fit(data)
reduced_data = pca.transform(data)
pca_samples = pca.transform(data)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

def biplot(data, reduced_data, pca):
    
    fig, ax = plt.subplots(figsize = (14,8))
    
    # scatterplot of the reduced data 
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'], facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    feature_vectors = pca.components_.T

    # using scaling factors to make the arrows
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, data.columns[i], color='black', ha='center', va='center', fontsize=18)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax

biplot(data, reduced_data, pca)

"""
The biplot above shows that the products milk, grocery, and detergents_paper are aligned towards the principal component 1 or dimension 1. 
Whereas the fresh and frozen products are aligned towards the principal component 2 or dimension 2. 
These seem intuitive as we have already seen their relationship in the scatter plot above where there seems to be a linear relationship 
between the group of products milk, grocery and detergents_paper and fresh and frozen products. 
Hence principal component analysis reduced the overall dimension of the dataset from six variables to two variables 
by also removed multicollinearity in the data by aligning the related variables into their respective principal components or dimensions
"""

# https://analyticsdefined.com/implementing-principal-component-analysis/
