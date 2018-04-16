# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:19:45 2018

@author: xuefei.yang
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


#from sklearn.model_selection import train_test_split
# https://www.kaggle.com/sociopath00/random-forest-using-gridsearchcv


df = pd.read_csv('titanic.csv')
df.shape
df.head()

df.isnull().sum()
df.describe()
df.nunique()


df['Survived'].value_counts().plot(kind='bar')


df.columns
cat = ['Pclass', 'Sex', 'Embarked']
num = ['Age', 'SibSp', 'Parch', 'Fare']


# categorical
df['Pclass'].value_counts().plot(kind='bar')
df['Sex'].value_counts().plot(kind='bar')
df['Embarked'].value_counts().plot(kind='bar')


# numerical
# https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/
corr = df[num].corr(method='pearson')
corr
# plot correlation matrix
plt.title('Correlation Matrix')
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

# Density Plot
df[num].plot(kind='density', subplots=True, layout=(2,2), sharex=False)
plt.show()

# Histogram
df[num].hist()
plt.show()


# pairplot
sns.pairplot(df, hue='Sex')
sns.plt.show()