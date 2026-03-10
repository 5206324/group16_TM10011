#%% importeren

import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import metrics




path = ("/Users/fenne/Documents/Technical Medicine/TM10011/Project/group16_TM10011/Lipo_radiomicFeatures.csv")

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Lipo_radiomicFeatures.csv'), index_col=0)

    return data


data = load_data()

# %% feature summary
feature_summary = {}

for col in data.columns:
    if "_sf_" in col:
        base = col.split("_sf_")[1].split("_")[0]
        stat = col.split(base + "_")[1].split("_")[0]
        
        if base not in feature_summary:
            feature_summary[base] = []
            
        feature_summary[base].append(stat)

for k, v in feature_summary.items():
    print(k, ":", sorted(v))
# %% printing verdeling
lipoma = data[data["label"] == "lipoma"]
liposarcoma = data[data["label"] == "liposarcoma"]

print("Lipoma:", lipoma.shape)
print("Liposarcoma:", liposarcoma.shape)




# %%
# Features en labels
X = data.drop(columns=["label"])
y = data["label"]

# Stratified K-Fold met 5 splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Itereer over de folds
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    print(f"Fold {fold+1}")
    print("Train:", X_train.shape, y_train.value_counts().to_dict())
    print("Validation:", X_val.shape, y_val.value_counts().to_dict())
    print("-"*30)




 #%%
lipoma = data[data["label"] == "lipoma"]
liposarcoma = data[data["label"] == "liposarcoma"]

 ## Find columns containing 'area'
area_cols = [col for col in data.columns if "area" in col.lower()]
print(area_cols)   
# %% 



area_cols = [
'PREDICT_original_sf_area_avg_2.5D',
'PREDICT_original_sf_area_max_2.5D',
'PREDICT_original_sf_area_min_2.5D',
'PREDICT_original_sf_area_std_2.5D'
]

fig, axes = plt.subplots(2, 2, figsize=(12,8))

for i, col in enumerate(area_cols):
    ax = axes[i//2, i%2]
    
    ax.hist(lipoma[col], bins=30, alpha=0.5, label="lipoma")
    ax.hist(liposarcoma[col], bins=30, alpha=0.5, label="liposarcoma")
    
    ax.set_title(col.split("area_")[1])  # shorter title
    ax.set_xlabel("Area value")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout()
plt.show()
ax.hist(lipoma[col], bins=30, alpha=0.5, label="Lipoma", density=True)
ax.hist(liposarcoma[col], bins=30, alpha=0.5, label="Liposarcoma", density=True)

# %% Classifiers

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# %%
# some function that we will use
from sklearn.decomposition import PCA

def colorplot(clf, ax, x, y, h=100, precomputer=None):
    '''
    Overlay the decision areas as colors in an axes.

    Input:
        clf: trained classifier
        ax: axis to overlay color mesh on
        x: feature on x-axis
        y: feature on y-axis
        h(optional): steps in the mesh
    '''
    # Create a meshgrid the size of the axis
    xstep = (x.max() - x.min() ) / 20.0
    ystep = (y.max() - y.min() ) / 20.0
    x_min, x_max = x.min() - xstep, x.max() + xstep
    y_min, y_max = y.min() - ystep, y.max() + ystep
    h = max((x_max - x_min, y_max - y_min))/h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    features = np.c_[xx.ravel(), yy.ravel()]
    if precomputer is not None:
        if type(precomputer) is RBFSampler:
            features = precomputer.transform(features)
        elif precomputer is rbf_kernel:
            features = rbf_kernel(features, X)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(features)
    else:
        Z = clf.predict_proba(features)
    if len(Z.shape) > 1:
        Z = Z[:, 1]

    # Put the result into a color plot
    cm = plt.cm.RdBu_r
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    del xx, yy, x_min, x_max, y_min, y_max, Z, cm

def load_data(n_features=2):
    '''
    Load the sklearn breast data set, but reduce the number of features with PCA.
    '''
    data = ds.load_data()
    x = data['data']
    y = data['target']

    p = PCA(n_components=n_features)
    p = p.fit(x)
    x = p.transform(x)
    return x, y
# %%
X1, Y1 = ds.make_classification(n_samples=100, n_features=2, n_redundant=0,
                                n_informative=2,
                                n_clusters_per_class=1)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.set_title("Two informative features, one cluster per class",
             fontsize='small')
ax.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,
           s=25, edgecolor='k', cmap=plt.cm.Paired)
lda = LinearDiscriminantAnalysis()
lda = lda.fit(X1, Y1)
y_pred = lda.predict(X1)
colorplot(lda, ax, X1[:, 0], X1[:, 1])
print("Number of mislabeled points out of a total %d points : %d" % (X1.shape[0], (Y1 != y_pred).sum()))


# %%
X2, Y2 = ds.make_classification(n_samples=100, n_features=2, n_redundant=0,
                                n_informative=1,
                                n_clusters_per_class=1)
fig = plt.figure(figsize=(24, 8))
ax = fig.add_subplot(131)
ax.set_title("One informative feature, one cluster per class", fontsize='small')
ax.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2,
           s=25, edgecolor='k', cmap=plt.cm.Paired)

X3, Y3 = ds.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=5)
ax = fig.add_subplot(132)
ax.set_title("Two blobs, two classes", fontsize='small')
ax.scatter(X3[:, 0], X3[:, 1], marker='o', c=Y3, s=25, edgecolor='k', cmap=plt.cm.Paired)

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X4 = pca.fit_transform(X_scaled)

# Zet de labels (y) om naar nummers (0 en 1) voor de plotter
# 'lipoma' wordt 0, 'liposarcoma' wordt 1 (of andersom)
Y4 = pd.factorize(y)[0]

ax = fig.add_subplot(133)
ax.set_title("A more complicated problem", fontsize='small')
ax.scatter(X4[:, 0], X4[:, 1], marker='o', c=Y4, s=25, edgecolor='k', cmap=plt.cm.Paired)


# %%
#   - GaussianNB
#   - LinearDiscriminantAnalysis
#   - QuadraticDiscriminantAnalysis
#   - LogisticRegression
#   - SGDClassifier
#   - KNeighborsClassifier
#   Motivate your choice. You can use the example code below to loop over both
#   the datasets and the classifiers at the same time:


clsfs = [LinearDiscriminantAnalysis(),QuadraticDiscriminantAnalysis(),GaussianNB(),
         LogisticRegression(),SGDClassifier(),KNeighborsClassifier()]
Xs = [X2, X3, X4]
Ys = [Y2, Y3, Y4]
clfs_fit = list()

# First make a plot without classifiers:
fig = plt.figure(figsize=(21,7*len(clsfs)))
num = 0  # Iteration number for the subplots
for X, Y in zip(Xs, Ys):
    ax = fig.add_subplot(6, 3, num + 1)
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=Y,
               s=25, edgecolor='k', cmap=plt.cm.Paired)
    num += 1

# Fit the classifiers and add them to the plot
num=0
Xt=list()
Yt=list()
for clf in clsfs:
    for X, Y in zip(Xs, Ys):
        # Fit classifier
        clf.fit(X,Y)
        y_pred=clf.predict(X)
        # Predict labels using fitted classifier

        # Make scatterplot of features
        ax = fig.add_subplot(6, 3, num + 1)
        ax.scatter(X[:, 0], X[:, 1], marker='o', c=Y,
               s=25, edgecolor='k', cmap=plt.cm.Paired)
        colorplot(clf, ax, X[:,0], X[:,1])
        # Add overlay through colorplot function
        t=("Misclass: %d / %d" % ((Y!=y_pred).sum(), X.shape[0]))
        ax.set_title(t)
        num+=1

        clfs_fit.append(clf)
        Xt.append(X)
        Yt.append(Y)
# %%