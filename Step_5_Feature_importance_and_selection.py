# %% importeren

import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

path = (r"C:\Bestanden\Technische Universiteit Delft\Master Technical Medicine\Machine learning TM10011\GroepsprojectML\group16_TM10011\Lipo_radiomicFeatures.csv")

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Lipo_radiomicFeatures.csv'), index_col=0)

    return data

data = load_data()
print(data.columns)

#%% feature importance
# 1. Gebruik je eigen ingeladen data
X = data.drop('label', axis=1) 
y = data['label']

# 2. Data opschonen (Essentieel: Random Forest kan niet tegen NaN-waardes)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
feature_names = X.columns # Bewaar de namen voor de grafiek

# 3. Train het model op jouw ECHTE data
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_imputed, y)

# 4. Bereken importance en variatie
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# 5. Plot de Top 20 belangrijkste features van jouw dataset
top_n = 20
plt.figure(figsize=(12, 6))
plt.title("Belangrijkste Radiomic Features voor Lipo Dataset")
plt.bar(range(top_n), importances[indices[:top_n]],
       color="r", yerr=std[indices[:top_n]], align="center")

# Gebruik de echte namen van de kolommen op de x-as
plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
plt.xlim([-1, top_n])
plt.tight_layout()
plt.show()

#%% Feature selection

# nodig: X_train, X_test, X

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),     # Missing data invullen
    ('variance', VarianceThreshold(threshold=0.01)), # Verwijder constante features
    ('scaler', StandardScaler()),                    # PCA vereist gestandaardiseerde data
    ('pca', PCA(n_components=2))                     # Reduceer naar 2 componenten
])

X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

print(f"Oorspronkelijke vorm: {X.shape}")
print(f"Vorm na selectie en PCA: {X_transformed.shape}")


# %%
