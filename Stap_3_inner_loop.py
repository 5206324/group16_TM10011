#%%
# Stap_3_Model_Manager.py
from Stap_3B_var_cor_feat_select import VarianceCorrelationFilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def inner_loop(X_train, y_train):
    # De EXACTE stappen van je studiegenoot
    estimator = LogisticRegression(max_iter=5000, solver='liblinear', penalty='l2') # we gebruiken een LogisticRegression voor de rangschrikking van de features
    

    # de RFE feature selection definiëren
    rfecv =  RFECV(
        estimator=estimator, 
        step=1,               # Verwijder 1 feature pe-r stap
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),                 # Gebruik 5-fold CV om het beste aantal te vinden
        scoring='accuracy',   # Optimaliseer op nauwkeurigheid MISS MOET DIT AUC WORDEN!!
        min_features_to_select=10, # Stop niet voordat er nog maar 10 over zijn
        n_jobs=-1             # Gebruik alle processors
    )


        # 1. De pre-processing stappen (Feature Selection & Scaling)
    common_steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('feature_filter', VarianceCorrelationFilter(
                variance_threshold=0.01,
                correlation_threshold=0.95
            )),
            ('scaler', RobustScaler()),
            ('rfecv', rfecv)
        ]
        
        # 2. De lijst met classifiers en hun hyperparameters
    classifiers = {
            'RandomForest': (RandomForestClassifier(random_state=42), {
                'clf__n_estimators': [150, 300, 450], # Aantal bomen
                'clf__max_depth': [None, 3, 5, 10], # Max diepte van elke boom (voorkomt overfitting)
                'clf__min_samples_leaf': [2, 4, 6, 8], # Minimaal aantal samles in laatste leaf (voorkomt overfitting)
                'clf__max_features': ['sqrt', 'log2'] # Hoeveel features mag elke boom gebruiken van het totaal aantal features
            }),
            'LogisticRegression': (LogisticRegression(max_iter=1000), {
               # 'clf__penalty': ['l1_ratio=1', 'l1_ratio=0'],
                'clf__C': [0.01, 0.1, 1, 10, 100],
               # 'clf__class_weight': [None, 'balanced'],
               # 'clf__solver': ['liblinear', 'saga']
            }),
            'XGBoost': (XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), {
                'clf__n_estimators': [50, 100, 200],      # Aantal bomen
               # 'clf__max_depth': [3, 5, 7],             # Hoe diep mag elke boom gaan? (3-5 is vaak zat voor radiomics)
               # 'clf__learning_rate': [0.01, 0.1, 0.2],  # Hoe snel leert het model? (Lagere waarde = stabieler)
               # 'clf__subsample': [0.8, 1.0],            # Gebruik een deel van de patiënten per boom (tegen overfitting)
               # 'clf__colsample_bytree': [0.7, 1.0],     # Gebruik een deel van de features per boom
               # 'clf__gamma': [0, 1, 5]                  # Minimale verliesreductie om een split te maken
            })
        }

    best_score = -1
    inner_loop = None
        
        # 3. De loop die alle classifiers test (De Inner Loop)
    for name, (clf_model, params) in classifiers.items():
            # Belangrijk: De inspringing hieronder moet exact kloppen
            full_pipeline = Pipeline(steps=common_steps + [('clf', clf_model)])
            
            # Voer de GridSearch uit
            grid = GridSearchCV(full_pipeline, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
        
            print(f"> Best voor {name}: {grid.best_score_:.4f}")
            
            # Sla het allerbeste model op
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                inner_loop = grid.best_estimator_
                # Pak de naam van de klasse van de classifier (bijv. 'SVC' of 'RandomForestClassifier')
                best_classifier = type(inner_loop.named_steps['clf']).__name__
                
        # We geven nu zowel de pipeline als de naam terug
    return inner_loop, best_classifier

# %%
