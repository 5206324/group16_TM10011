import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_compare_train_test_roc(classifiers_dict, X_train, y_train, X_test, y_test):
    """
    Maakt een dubbele ROC-curve vergelijking voor Train en Test sets.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Instellingen voor beide assen
    for ax in [ax1, ax2]:
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5) # Diagonale lijn (toeval)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(True, linestyle='--', alpha=0.7)

    ax1.set_title('ROC Curve Comparison - TRAIN Set')
    ax2.set_title('ROC Curve Comparison - TEST Set')

    # Loop door de getrainde modellen
    for i, (name, pipeline) in enumerate(classifiers_dict.items()):
        color = colors[i % len(colors)]
        
        # --- TRAIN DATA ---
        y_probs_train = pipeline.predict_proba(X_train)[:, 1]
        fpr_tr, tpr_tr, _ = roc_curve(y_train, y_probs_train)
        auc_tr = roc_auc_score(y_train, y_probs_train)
        ax1.plot(fpr_tr, tpr_tr, color=color, lw=2, label=f'{name} (AUC = {auc_tr:.2f})')
        
        # --- TEST DATA ---
        y_probs_test = pipeline.predict_proba(X_test)[:, 1]
        fpr_te, tpr_te, _ = roc_curve(y_test, y_probs_test)
        auc_te = roc_auc_score(y_test, y_probs_test)
        ax2.plot(fpr_te, tpr_te, color=color, lw=2, label=f'{name} (AUC = {auc_te:.2f})')

    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")
    plt.tight_layout()
    plt.show()