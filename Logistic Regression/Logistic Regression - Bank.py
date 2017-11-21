# =============================================================================
# Import
# =============================================================================
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE   # recursive feature selection
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# =============================================================================
# Load Data
# =============================================================================
data = pd.read_csv('banking.csv', header=0)
data = data.dropna()
print(data.shape)
print(list(data.columns))
data.head()


# Modify the 'education' column
data['education'].unique()

data['education'] = np.where(data['education']=='basic.4y', 'basic', data['education'])
data['education'] = np.where(data['education']=='basic.6y', 'basic', data['education'])
data['education'] = np.where(data['education']=='basic.9y', 'basic', data['education'])

data['education'].unique()

# =============================================================================
# Data Exploration
# =============================================================================
data['y'].value_counts()

sns.countplot(x='y', data=data, palette='hls')
plt.show()

data.groupby('y').mean()

data.groupby('job').mean()
data.groupby('marital').mean()
data.groupby('education').mean()

# =============================================================================
# Visualization
# =============================================================================
# job seems a strong predictor
pd.crosstab(data.job, data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')

# marital not a strong predictor
table=pd.crosstab(data.marital, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')

# education seems a strong predictor
table=pd.crosstab(data.education, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')

# day of week not a strong predictor
pd.crosstab(data.day_of_week, data.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')

# month seems a strong predictor
table=pd.crosstab(data.month, data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')

# poutcome seems a strong predictor
table=pd.crosstab(data.poutcome,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')

data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')



# =============================================================================
# Create dummy variables
# =============================================================================
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']

for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1
    
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]

data_final=data[to_keep]
data_final.columns.values

data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]


# =============================================================================
# Feature Selection
# =============================================================================
logreg = LogisticRegression()
rfe = RFE(logreg, 18)
rfe = rfe.fit(data_final[X], data_final[y] )
print(rfe.support_)
print(rfe.ranking_)

cols = data_final.columns[rfe.support_]
X=data_final[cols]
y=data_final['y']


# =============================================================================
# Implementing the model
# =============================================================================
logit_model=sm.Logit(y, X)
result=logit_model.fit()
print(result.summary())

# =============================================================================
# Logistic Regression Model Fitting
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# =============================================================================
# Cross Validation
# =============================================================================
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# =============================================================================
# Confusion Matrix
# =============================================================================
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# =============================================================================
# Compute precision, recall, F-measure and support
# =============================================================================
print(classification_report(y_test, y_pred))

# =============================================================================
# ROC Curve
# =============================================================================
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()