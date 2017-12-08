# =============================================================================
# Import Packages
# =============================================================================
import pandas as pd
import numpy as np
import pylab 
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



# =============================================================================
# Load Packages
# =============================================================================
df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
    )

feature_dict = {i:label for i,label in zip(
                range(4),
                  ('sepal length',
                  'sepal width',
                  'petal length',
                  'petal width', ))}
df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True) # to drop the empty line at file-end
df.tail()

# =============================================================================
# EDA
# =============================================================================
fig, ax = plt.subplots(nrows=2, ncols=2)

for i in range(0,4):
    plt.subplot(2, 2, i+1)
    stats.probplot(df.iloc[:,i], dist='norm', plot=pylab)
    pylab.title('QQ plot of ' + df.columns[i])

pylab.show()


# =============================================================================
# Label Encode
# =============================================================================
X = df.iloc[:,0:4].values
y = df['class label'].values

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}


# =============================================================================
# Splitting the dataset into the Training set and Test set
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# =============================================================================
# Feature Scaling to Dataset
# =============================================================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# =============================================================================
# LDA
# =============================================================================
sklearn_lda = LDA(n_components=2)
X_train_lda_sklearn = sklearn_lda.fit_transform(X_train, y_train)
X_test_lda_sklearn = sklearn_lda.transform(X_test)


# =============================================================================
# Visualization 
# =============================================================================
def plot_scikit_lda(X, y, title):

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X[:,0][y == label],
                    y=X[:,1][y == label] * -1, # flip the figure
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()
    
plot_scikit_lda(X_train_lda_sklearn, y_train, title='Default LDA via scikit-learn')


# =============================================================================
# Prediction
# =============================================================================
# prior probability
print(sklearn_lda.priors_)

# confusion matrix
pred_test = sklearn_lda.predict(X_test)
print(confusion_matrix(pred_test, y_test))

# accuracy
print(np.mean(pred_test==y_test))

