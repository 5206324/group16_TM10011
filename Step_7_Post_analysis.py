# %% functie voor bepalen uitkomstmaten

# wat erin moet:
# y_test = Test data
# y_score = voorspelde kans : clf.predict_proba(X_test)[:, 1] (hierbij is X_test dus vd testdata)
# y_pred = = voorspelde klasse : clf.predict(X_test) (hierbij is X_test dus vd testdata)
# model = classifier model

def outcomes(y_test, y_score, y_pred, model):
    from sklearn import metrics
    import matplotlib.pyplot as plt

    auc=metrics.roc_auc_score(y_test, y_score)
    accuracy=metrics.accuracy_score(y_test, y_pred)
    F1=metrics.f1_score(y_test,y_pred)
    precision=metrics.precision_score(y_test,y_pred)
    recall=metrics.recall_score(y_test, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)

# accuracy, AUC, f1score, precision, recall
    print(type(model))
    print('Acc:' +str(accuracy))
    print('AUC:' +str(auc))
    print('F1:' +str(F1))
    print('precision:' +str(precision))
    print('recall:' +str(recall))

    # ROC Curve

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], linestyle='--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    return auc, accuracy, F1, precision, recall
# %%
