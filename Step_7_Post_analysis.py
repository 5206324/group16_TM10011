# %% functie voor bepalen uitkomstmaten

# wat erin moet:
# y_test = Test data
# y_score = voorspelde kans : clf.predict_proba(X_test)[:, 1] (hierbij is X_test dus vd testdata)
# y_pred = = voorspelde klasse : clf.predict(X_test) (hierbij is X_test dus vd testdata)
# model = classifier model

def outcomes(y_test, y_score, y_pred, model):
    from sklearn import metrics

    auc = metrics.roc_auc_score(y_test, y_score)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    F1 = metrics.f1_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    print(type(model))
    print('Acc:' + str(accuracy))
    print('AUC:' + str(auc))
    print('F1:' + str(F1))
    print('precision:' + str(precision))
    print('recall:' + str(recall))

    return auc, accuracy, F1, precision, recall

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

