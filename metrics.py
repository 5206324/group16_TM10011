#%%
from sklearn import metrics

def metrics_best_fold(folds_data, best_results_dict):
    for model_naam in ['RandomForest', 'LogisticRegression', 'XGBoost']:
        if model_naam not in best_results_dict:
            continue
            
        # Pak de opgeslagen info van je beste fold
        best_info = best_results_dict[model_naam]
        best_pipeline = best_info['pipeline']
        target_auc = best_info['auc'] # De 0.886 die je zocht
        
        # We moeten nu de fold terugvinden die deze AUC produceerde
        found_data = False
        for pakketje in folds_data:
            xtest = pakketje['X_test']
            ytest = pakketje['y_test']
            
            # Check of dit de fold is die bij je opgeslagen AUC hoort
            probs = best_pipeline.predict_proba(xtest)[:, 1]
            current_auc = metrics.roc_auc_score(ytest, probs)
            
            # Als de AUC (bijna) gelijk is, hebben we de juiste test-data te pakken
            if abs(current_auc - target_auc) < 1e-5:
                preds = best_pipeline.predict(xtest)
                tn, fp, fn, tp = metrics.confusion_matrix(ytest, preds, labels=[0, 1]).ravel()
                
                # Bereken de rest van de lijst
                accuracy = metrics.accuracy_score(ytest, preds)
                sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
                specificity = tn / (tn + fp) if (tn + fp) else 0.0
                ppv = tp / (tp + fp) if (tp + fp) else 0.0
                npv = tn / (tn + fn) if (tn + fn) else 0.0
                f1 = metrics.f1_score(ytest, preds, zero_division=0)
                
                n_features = 0
                if 'rfecv' in best_pipeline.named_steps:
                    n_features = best_pipeline.named_steps['rfecv'].n_features_
                
                print("-" * 35)
                print(f"{model_naam.upper()} - best scorend")
                print(f"{'n features':<20}: {n_features}")
                print(f"{'Test AUC':<20}: {current_auc:.4f}")
                print(f"{'Accuracy':<20}: {accuracy:.4f}")
                print(f"{'Sensitivity':<20}: {sensitivity:.4f}")
                print(f"{'Specificity':<20}: {specificity:.4f}")
                print(f"{'PPV':<20}: {ppv:.4f}")
                print(f"{'NPV':<20}: {npv:.4f}")
                print(f"{'F1':<20}: {f1:.4f}")
                found_data = True
                break
        
        if not found_data:
            print(f"Kon de test-data voor de beste {model_naam} AUC ({target_auc}) niet matchen.")



