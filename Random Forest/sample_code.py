# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:19:45 2018

@author: xuefei.yang
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

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
todo = ['Name', 'Ticket', 'Cabin']


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
            yticklabels=corr.columns.values, 
            cmap=sns.diverging_palette(220, 10, as_cmap=True), 
            square=True, vmin= -1, vmax= 1)
plt.show()

# Density Plot
df[num].plot(kind='density', subplots=True, layout=(2,2), sharex=False)
plt.show()

# Histogram
df[num].hist()
plt.show()


# pairplot
#sns.pairplot(df[num], dropna=True)
#sns.plt.show()


#The Chi Square statistic is commonly used for testing relationships between 
#categorical variables. The null hypothesis of the Chi-Square test is that no 
#relationship exists on the categorical variables in the population; 
#they are independent. 
chi2_contingency(pd.crosstab(df['Survived'], df['Sex']))[1]
chi2_contingency(pd.crosstab(df['Survived'], df['Pclass']))[1]
chi2_contingency(pd.crosstab(df['Survived'], df['Embarked']))[1]

sns.countplot(x='Survived', hue='Sex', data=df)
sns.countplot(x='Survived', hue='Pclass', data=df)
sns.countplot(x='Survived', hue='Embarked', data=df)

sns.boxplot(x='Survived', y='Fare', data=df)
sns.boxplot(x='Survived', y='Age', data=df)
sns.boxplot(x='Survived', y='SibSp', data=df)
sns.boxplot(x='Survived', y='Parch', data=df)


# Missing value
df.isnull().sum()

# replace na with median
df['Age'].describe()
med_age = np.nanmedian(df['Age'])
df['Age'] = df['Age'].fillna(med_age)

#replace na with 'Unknown'
df['Cabin'].value_counts()
df['Cabin'] = df['Cabin'].fillna('Unknown')


# replace na with mode
df['Embarked'].value_counts()
df['Embarked'] = df['Embarked'].fillna('S')

df.isnull().sum()



# Feature Engineering


# Model
X = df[num+cat]
y = df['Survived']

X = pd.get_dummies(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train)

CV_rfc.best_params_

rfc1=RandomForestClassifier(random_state=42, max_features='auto', 
                            n_estimators= 200, max_depth=6, criterion='entropy')
rfc1.fit(x_train, y_train)
pred = rfc1.predict(x_test)
accuracy_score(y_test,pred)
