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

def inner_loop(X_train, y_train):
    # De EXACTE stappen van je studiegenoot
    common_steps = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('variance', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
    ]

    # De EXACTE classifiers en hyperparameters van je studiegenoot
    classifiers = {
        'Lasso_Logistic': (LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000), {
            'clf__C': [0.01, 0.1, 1, 10, 100] # C is de omgekeerde sterkte van Lasso
        }),
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

    best_score = -1
    inner_loop = None

    # We lopen door de classifiers heen (de 'alle combi's' op je whiteboard)
    for name, (clf_model, params) in classifiers.items():
        full_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('variance', VarianceThreshold()),
            ('select_k', SelectKBest(f_classif, k=10)), # Kies alleen de 10 beste features
            ('clf', clf_model) 
    ])
        
        # De Inner Loop (5-fold cross validation binnen de training data)
        grid = GridSearchCV(full_pipeline, param_grid=params, cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        
        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            inner_loop = grid.best_estimator_
            
    return inner_loop
# %%
