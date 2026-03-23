# %% importeren

import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import metrics

from sklearn.preprocessing import StandardScaler, RobustScaler
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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE


path = (r"C:\Bestanden\Technische Universiteit Delft\Master Technical Medicine\Machine learning TM10011\GroepsprojectML\group16_TM10011\Lipo_radiomicFeatures.csv")

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Lipo_radiomicFeatures.csv'), index_col=0)

    return data

data = load_data()
print(data.columns)


#%% feature importance + selection

X = data.drop('label', axis=1) 
y = data['label']
feature_names = X.columns

#%% Preprocessing: lege cellen vullen, kolommen met zelfde waarden weghalen, features op zelfde schaal zetten, pca 
common_steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('variance', VarianceThreshold(threshold=0.01)),
    ('scaler', RobustScaler()),
    ('rfe', RFE(estimator=LogisticRegression(max_iter=5000), n_features_to_select=20))
]
# %% Classifiers bepalen en hun hyperparameters
classifiers = {
    'RandomForest': (RandomForestClassifier(random_state=42), {
        'clf__n_estimators': [50, 100, 150],
        'clf__max_depth': [None, 10]
    }),
    'LogisticRegression': (LogisticRegression(max_iter=5000), {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100]
    }),
    'KNN': (KNeighborsClassifier(), {
        'clf__n_neighbors': [3, 5, 11]
    })
}

results = {}
best_estimators = {}
#%% Inner-outer loop maken
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nested_results = {}

for name, (clf, params) in classifiers.items():
    # De Pipeline maken: common steps + classifier
    full_pipeline = Pipeline(steps = common_steps + [('clf', clf)])
    
    # hij splits hem nu op in 3 groepen om te testen op accuracy van verschillende waardes voor de hyperparameters
    inner_testing = GridSearchCV(full_pipeline, param_grid=params, cv=StratifiedKFold(n_splits=3), scoring='accuracy')
    
    # De hiervoor gekozen waardes gebruiken voor score berekenen op de outer loop
    outer_scores = cross_val_score(inner_testing, X, y, cv=outer_cv)
    
    # Bepalen van de resultaten (gemiddelde van de 5 waardes en de std)
    nested_results[name] = {
        'scores': outer_scores,
        'mean_score': outer_scores.mean(),
        'std_score': outer_scores.std()
    }
    
    print(f"{name} gemiddelde score zonder PCA: {outer_scores.mean():.3f} (+/- {outer_scores.std():.3f})")

# %% Feature importance bepalen

best_rf_pipeline = Pipeline(steps=common_steps + [('clf', classifiers['RandomForest'][0])])
best_rf_pipeline.fit(X, y)
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
# lege cellen vullen, kolommen met zelfde waarden weghalen, features op zelfde schaal zetten, pca 
common_steps = [
    ('imputer', SimpleImputer(strategy='mean')),
    ('variance', VarianceThreshold(threshold=0.01)),
    ('scaler', StandardScaler()),
    ('pca', PCA())
]
results = {}
best_estimators = {}

classifiers = {
    'RandomForest': (RandomForestClassifier(random_state=42), {
        'pca__n_components': [5, 10, 15, 0.95], # Probeer 5, 10, 15 óf genoeg voor 95% variantie
        'clf__n_estimators': [50, 100, 150],
        'clf__max_depth': [None, 10]
    }),
    'LogisticRegression': (LogisticRegression(max_iter=1000), {
        'pca__n_components': [5, 10, 15, 0.95],
        'clf__C': [0.001, 0.01, 0.1, 1, 10]
    }),
    'KNN': (KNeighborsClassifier(), {
        'pca__n_components': [5, 10, 15, 0.95],
        'clf__n_neighbors': [3, 5, 11]
    })
}

# %% inner-outer loop maken
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nested_results = {}

for name, (clf, params) in classifiers.items():
    # De Pipeline maken: common steps + classifier
    full_pipeline = Pipeline(steps = common_steps + [('clf', clf)])
    
    # hij splits hem nu op in 3 groepen om te testen op accuracy van verschillende waardes voor de hyperparameters
    inner_testing = GridSearchCV(full_pipeline, param_grid=params, cv=StratifiedKFold(n_splits=3), scoring='accuracy')
    
    # De hiervoor gekozen waardes gebruiken voor score berekenen op de outer loop
    outer_scores = cross_val_score(inner_testing, X, y, cv=outer_cv)
    
    # Bepalen van de resultaten (gemiddelde van de 5 waardes en de std)
    nested_results[name] = {
        'scores': outer_scores,
        'mean_score': outer_scores.mean(),
        'std_score': outer_scores.std()
    }
    
    print(f"{name} gemiddelde score met PCA: {outer_scores.mean():.3f} (+/- {outer_scores.std():.3f})")

# %%
