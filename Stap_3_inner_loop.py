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
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV



def inner_loop(X_train, y_train):
    # De EXACTE stappen van je studiegenoot
    estimator = LogisticRegression(max_iter=5000, solver='liblinear', penalty='l2') # we gebruiken een LogisticRegression voor de rangschrikking van de features
    

    # de RFE feature selection definiëren
    rfecv =  RFECV(
        estimator=estimator, 
        step=1,               # Verwijder 1 feature per stap
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),                 # Gebruik 5-fold CV om het beste aantal te vinden
        scoring='accuracy',   # Optimaliseer op nauwkeurigheid MISS MOET DIT AUC WORDEN!!
        min_features_to_select=10, # Stop niet voordat er nog maar 10 over zijn
        n_jobs=-1             # Gebruik alle processors
    )

    common_steps = [
        ('imputer', SimpleImputer(strategy='median')),
        ('variance', VarianceThreshold(threshold=0.01)),
        ('scaler', RobustScaler()),
        ('rfecv', rfecv)
    ]

    # De classfiers
    classifiers = {
        'RandomForest': (RandomForestClassifier(random_state=42), {
            'clf__n_estimators': [50, 100, 150],
            'clf__max_depth': [None, 5, 10],
            'clf__min_samples_leaf': [1, 5, 10],
            'clf__max_features': ['sqrt', 'log2'],
            'clf__class_weight': ['balanced', None]
        }),
        'LogisticRegression': (LogisticRegression(max_iter=1000), {
           'clf__penalty': ['l1', 'l2'],
           'clf__C': [0.01, 0.1, 1, 10, 100],
           'clf__class_weight': [None, 'balanced'],
           'clf__solver': ['liblinear', 'saga']
        }),
        'KNN': (KNeighborsClassifier(), {
            'clf__n_neighbors': [3, 5, 11, 21],
            'clf__weights': ['uniform', 'distance'],
            'clf__metric': ['euclidean', 'manhattan']
        }),
        'SVC': (SVC(probability=True, random_state=42), {
            'clf__C': [0.1, 1, 10, 100],
            'clf__kernel': ['linear', 'rbf', 'poly'],
            'clf__gamma': ['scale', 'auto'],
            'clf__class_weight': ['balanced', None]
        # }),
        # 'LDA': (LinearDiscriminantAnalysis(), {
        #     'clf__solver': ['svd', 'lsqr', 'eigen']
        # }),
        # 'QDA': (QuadraticDiscriminantAnalysis(), {
        #     'clf__reg_param': [0.0, 0.1, 0.5]
        }),
        'GaussianNB': (GaussianNB(), {
            'clf__var_smoothing': [1e-9, 1e-8, 1e-7]
        }),
        'SGD': (SGDClassifier(loss='log_loss', random_state=42), {
            'clf__alpha': [0.0001, 0.001, 0.01, 0.1],
            'clf__penalty': ['l2', 'l1', 'elasticnet']
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
