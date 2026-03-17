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
from sklearn.model_selection import GridSearchCV

path = (r"C:\Bestanden\Technische Universiteit Delft\Master Technical Medicine\Machine learning TM10011\GroepsprojectML\group16_TM10011\Lipo_radiomicFeatures.csv")

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Lipo_radiomicFeatures.csv'), index_col=0)

    return data

data = load_data()
print(data.columns)


#%% feature importance + selection (zonder PCA)

X = data.drop('label', axis=1) 
y = data['label']
feature_names = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# lege cellen vullen, kolommen met zelfde waarden weghalen, features op zelfde schaal zetten, pca 
common_steps = [
    ('imputer', SimpleImputer(strategy='mean')),
    ('variance', VarianceThreshold(threshold=0.01)),
    ('scaler', StandardScaler()),
]

# classifiers bepalen en hun hyperparameters
classifiers = {
    'RandomForest': (RandomForestClassifier(random_state=42), {
        'clf__n_estimators': [50, 100, 150],
        'clf__max_depth': [None, 10]
    }),
    'LogisticRegression': (LogisticRegression(max_iter=1000), {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }),
    'KNN': (KNeighborsClassifier(), {
        'clf__n_neighbors': [3, 5, 11]
    })
}

results = {}
best_estimators = {}

# common_steps koppelen aan classifiers, optimale hyperparameters bepalen met grid, crossvalidation (knipt in 5 stukken:4x trainen 1x testen)
for name, (clf, params) in classifiers.items():
    # Maak de volledige pipeline voor DIT model
    full_pipeline = Pipeline(steps = common_steps + [('clf', clf)])
    
    # GridSearch setup
    grid = GridSearchCV(full_pipeline, param_grid=params, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    # Sla resultaten op
    results[name] = {
        'best_score': grid.best_score_,
        'test_score': grid.score(X_test, y_test),
        'best_params': grid.best_params_
    }
    best_estimators[name] = grid.best_estimator_
    
    print(f"{name} heeft beste train-score van: {grid.best_score_:.3f}")

# 5. Resultaten bekijken
print("\nEindresultaten op de test-set:")
for name, res in results.items():
    print(f"{name}: {res['test_score']:.3f} (Params: {res['best_params']})")

best_rf_pipeline = best_estimators['RandomForest']
importances = best_rf_pipeline.named_steps['clf'].feature_importances_

# Let op: VarianceThreshold heeft mogelijk features verwijderd. 
# We moeten de namen matchen met de overgebleven features.
mask = best_rf_pipeline.named_steps['variance'].get_support()
reduced_feature_names = feature_names[mask]

indices = np.argsort(importances)[::-1]
top_n = 10

plt.figure(figsize=(12, 6))
plt.title("Top 10 Belangrijkste Radiomic Features (Random Forest)")
plt.bar(range(top_n), importances[indices[:top_n]], color="r", align="center")
plt.xticks(range(top_n), reduced_feature_names[indices[:top_n]], rotation=45, ha='right')
plt.ylabel("Importance score")
plt.tight_layout()
plt.show()
#%% Feature selection met PCA

X = data.drop('label', axis=1) 
y = data['label']
feature_names = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# lege cellen vullen, kolommen met zelfde waarden weghalen, features op zelfde schaal zetten, pca 
common_steps = [
    ('imputer', SimpleImputer(strategy='mean')),
    ('variance', VarianceThreshold(threshold=0.01)),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
]

# classifiers bepalen en hun hyperparameters
classifiers = {
    'RandomForest': (RandomForestClassifier(random_state=42), {
        'clf__n_estimators': [50, 100, 150],
        'clf__max_depth': [None, 10]
    }),
    'LogisticRegression': (LogisticRegression(max_iter=1000), {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }),
    'KNN': (KNeighborsClassifier(), {
        'clf__n_neighbors': [3, 5, 11]
    })
}

results = {}
best_estimators = {}

# common_steps koppelen aan classifiers, optimale hyperparameters bepalen met grid, crossvalidation (knipt in 5 stukken:4x trainen 1x testen)
for name, (clf, params) in classifiers.items():
    # Maak de volledige pipeline voor DIT model
    full_pipeline = Pipeline(steps = common_steps + [('clf', clf)])
    
    # GridSearch setup
    grid = GridSearchCV(full_pipeline, param_grid=params, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    # Sla resultaten op
    results[name] = {
        'best_score': grid.best_score_,
        'test_score': grid.score(X_test, y_test),
        'best_params': grid.best_params_
    }
    best_estimators[name] = grid.best_estimator_
    
    print(f"{name} heeft beste train-score van: {grid.best_score_:.3f}")

# 5. Resultaten bekijken
print("\nEindresultaten op de test-set:")
for name, res in results.items():
    print(f"{name}: {res['test_score']:.3f} (Params: {res['best_params']})")


# %%
