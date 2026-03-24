#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier

def plot_baseline_comparison(X, y):
    # Alle gewenste classifiers
    clsfs = [
        RandomForestClassifier(random_state=42),
        LogisticRegression(max_iter=1000),
        KNeighborsClassifier(),
        svm.SVC(probability=True, random_state=42),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        GaussianNB(),
        SGDClassifier(loss='modified_huber', random_state=42),
        XGBClassifier(random_state=42)
    ]
    
    clf_names = ["RandomForest", "LogisticReg", "KNN", "SVC", "LDA", "QDA", "GaussianNB", "SGD", "XGBoost"]  
    
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Maak het raster voor de plots
    fig, axes = plt.subplots(len(clsfs), 2, figsize=(18, 5 * len(clsfs)))
    
    print(f"Start Baseline Analyse voor {len(clsfs)} modellen...")

    for i, (clf, name) in enumerate(zip(clsfs, clf_names)):
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('clf', clf)
        ])
        
        # --- 1. Learning Curve Berekening ---
        ax_lc = axes[i, 0]
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Plot Learning Curve met schaduw
        ax_lc.grid(True)
        ax_lc.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        ax_lc.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        ax_lc.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        ax_lc.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        ax_lc.set_title(f"{name} Learning Curve")
        ax_lc.set_xlabel("Training examples")
        ax_lc.set_ylabel("Accuracy")
        ax_lc.set_ylim(0.4, 1.05)
        ax_lc.legend(loc="best")

        # --- 2. ROC Curves & Statistiek ---
        ax_roc = axes[i, 1]
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        # We gebruiken .values om index-fouten te voorkomen
        for train, test in cv.split(X, y):
            pipeline.fit(X.values[train], y.values[train])
            probs = pipeline.predict_proba(X.values[test])[:, 1]
            fpr, tpr, _ = roc_curve(y.values[test], probs)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))
            ax_roc.plot(fpr, tpr, lw=1, alpha=0.2, color='blue')

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        ax_roc.plot(mean_fpr, mean_tpr, color='b', label=f'Mean AUC = {mean_auc:.2f} ± {std_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], 'r--', label='Kansniveau (0.5)')
        ax_roc.set_title(f"{name} ROC (5-fold CV)")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True)

        # Opslaan voor de ranking tabel
        results.append({
            'Model': name,
            'Mean AUC': mean_auc,
            'Std AUC': std_auc,
            'Gap (Overfit)': train_scores_mean[-1] - test_scores_mean[-1]
        })

    # Toon de objectieve ranking in de console
    ranking_df = pd.DataFrame(results).sort_values(by='Mean AUC', ascending=False)
    print("\n" + "="*50)
    print("      OBJECTIEVE MODEL RANKING (Baseline)")
    print("="*50)
    print(ranking_df.to_string(index=False))
    print("="*50)

    plt.tight_layout()
    plt.show()
# %%