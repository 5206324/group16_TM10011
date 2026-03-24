#%%
# Stap_3_Model_Manager.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
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


def inner_loop(X_train, y_train):
    # De EXACTE stappen van je studiegenoot
    common_steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('variance', VarianceThreshold(threshold=0.01)),
        ('scaler', RobustScaler()),
        ('select_k', SelectKBest(f_classif, k=50)),
        ('rfe', RFE(estimator=LogisticRegression(max_iter=5000), n_features_to_select=10))

    ]

    # De EXACTE classifiers en hyperparameters van je studiegenoot
    classifiers = {
        'RandomForest': (RandomForestClassifier(random_state=42), {
            'clf__n_estimators': [50, 100, 150],
            'clf__max_depth': [None, 5, 10],           # 'Pruning': voorkom te diepe, complexe bomen
            'clf__min_samples_leaf': [1, 5, 10],       # Hoeveel patiënten moeten er minimaal in een 'blaadje' zitten?
            'clf__max_features': ['sqrt', 'log2'],     # Hoeveel features bekijkt hij per split? (Heel belangrijk bij 500 features)
            'clf__class_weight': ['balanced', None]    # Corrigeer voor scheve verhoudingen tussen groepen
        }),
        'LogisticRegression': (LogisticRegression(max_iter=1000), {
            'clf__penalty': ['l1', 'l2'],            # Test zowel Lasso als Ridge
            'clf__C': [0.01, 0.1, 1, 10, 100],      # Sterkte van de straf
            'clf__class_weight': [None, 'balanced'], # Corrigeer voor scheve verhoudingen
            'clf__solver': ['liblinear', 'saga']    # Verschillende rekenmethodes
        }),
        'KNN': (KNeighborsClassifier(), {
            'clf__n_neighbors': [3, 5, 11, 21],    # Aantal buren (oneven getallen voorkomen 'gelijkspel')
            'clf__weights': ['uniform', 'distance'], # Hoe zwaar telt een buur mee?
            'clf__metric': ['euclidean', 'manhattan'] # Hoe berekenen we de afstand?
        })
    }

    best_score = -1
    inner_loop = None
    

    # We lopen door de classifiers heen (de 'alle combi's' op je whiteboard)
    for name, (clf_model, params) in classifiers.items():
            full_pipeline = Pipeline(steps=common_steps + [('clf', clf_model)])
        
        # De Inner Loop (5-fold cross validation binnen de training data)
        grid = GridSearchCV(full_pipeline, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
      
        print(f"> Best voor {name}: {grid.best_score_:.4f} met {grid.best_params_}")
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            inner_loop = grid.best_estimator_
            
    return inner_loop
# %%
