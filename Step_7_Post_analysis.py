# %% functie voor bepalen uitkomstmaten


from sklearn import metrics

def resultaten(model, X_train, y_train, X_test, y_test, feature_names, naam):
    # 1. Voorspellingen doen (kansen voor de ROC, klasses voor de rest)
    y_score_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = model.predict(X_test)
    
    # 2. ROC data berekenen
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score_test)
    auc_score = metrics.auc(fpr, tpr)
    
    # 3. Alle statistieken in een dictionary stoppen
    res_dict = {
        'model_naam': naam,
        'auc': auc_score,
        'fpr': fpr,            # CRUCIAAL voor de plot
        'tpr': tpr,            # CRUCIAAL voor de plot
        'accuracy': metrics.accuracy_score(y_test, y_pred_test),
        'precision': metrics.precision_score(y_test, y_pred_test, zero_division=0),
        'recall': metrics.recall_score(y_test, y_pred_test),
        'f1': metrics.f1_score(y_test, y_pred_test)
    }
    
    return res_dict

#%% ROC Curve

def plot_roc_curves(models_data, y_test):
    from sklearn import metrics
    import matplotlib.pyplot as plt

    plt.figure()

    for name, y_score in models_data:
        fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
        auc = metrics.roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.show()

#%% Precision-Recall Curve
def plot_pr_curves(models_data, y_test):
    from sklearn import metrics
    import matplotlib.pyplot as plt

    plt.figure()

    for name, y_score in models_data:
        precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)
        pr_auc = metrics.auc(recall, precision)
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='lower left')
    plt.show()

