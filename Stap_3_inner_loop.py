#%% === STEP 3: INNER LOOP ===
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
    estimator = LogisticRegression(max_iter=5000, solver='liblinear', penalty='l2')
    min_features_to_select = 5

    # Keep DataFrame column names after imputation so downstream feature reporting stays meaningful.
    imputer = SimpleImputer(strategy='median')
    if hasattr(imputer, "set_output"):
        imputer = imputer.set_output(transform="pandas")

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        min_features_to_select=min_features_to_select,
        n_jobs=1
    )

    common_steps = [
        ('imputer', imputer),
        ('feature_filter', VarianceCorrelationFilter(
            variance_threshold=0.01,
            correlation_threshold=0.95
        )),
        ('scaler', RobustScaler()),
        ('rfecv', rfecv)
    ]

    classifiers = {
        'RandomForest': (RandomForestClassifier(random_state=42), {
            'clf__n_estimators': [450],#[150, 300, 450],
            'clf__max_depth': [3, 5, 7],
            'clf__min_samples_leaf': [2, 4, 6, 8],
            'clf__max_features': ['sqrt', 'log2']
        }),
        'LogisticRegression': (LogisticRegression(max_iter=1000), {
            'clf__penalty': ['l2'],
            'clf__C': [0.001, 0.01, 0.1, 1, 10],
            'clf__solver': ['liblinear']
        }),
        'XGBoost': (XGBClassifier(random_state=42, eval_metric='logloss'), {
            'clf__n_estimators': [50],#, 75, 100],
            'clf__max_depth': [2, 3, 4],
            'clf__min_child_weight': [2, 4, 6],
            'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__subsample': [0.8, 1.0],
            'clf__colsample_bytree': [0.8, 1.0],
            # 'clf__gamma': [0, 0.5, 1],
            # 'clf__reg_lambda': [1, 5, 10]
        })
    }

    getrainde_modellen = {}

    for name, (clf_model, params) in classifiers.items():
        full_pipeline = Pipeline(steps=common_steps + [('clf', clf_model)])
        grid = GridSearchCV(full_pipeline, param_grid=params, cv=5, scoring='accuracy', n_jobs=1)
        grid.fit(X_train, y_train)

        getrainde_modellen[name] = grid.best_estimator_
        print(f"> Best voor {name}: {grid.best_score_:.4f}")

    return getrainde_modellen
# %%
