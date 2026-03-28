#%% === Step 0: Imports ===
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path.cwd()))
from Stap_1_Data_inladen import data_lipo 
from Stap_1b_visualisatie import plot_baseline_comparison
from Stap_2_kfolds_splitsing import kfold
from Stap_3_inner_loop import inner_loop
from metrics import metrics_best_fold

# --- Extra definities ---
def plot_train_test_roc(best_results_dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    for model_name, data in best_results_dict.items():
        auc_train = metrics.auc(data['fpr_train'], data['tpr_train'])
        # TRAIN
        ax1.plot(data['fpr_train'], data['tpr_train'],
                 label=f"{model_name} (AUC = {auc_train:.2f})")
        
        # TEST
        ax2.plot(data['fpr_test'], data['tpr_test'],
                 label=f"{model_name} (AUC = {data['auc']:.2f})")

    for ax, title in zip([ax1, ax2], ["TRAIN ROC", "TEST ROC"]):
        ax.plot([0, 1], [0, 1], '--')
        ax.set_title(title)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()


# %% --- Step 1: Data exploration ---
data = data_lipo("Lipo_radiomicFeatures.csv")
if 'label' in data.columns:
    data = data.set_index(data.columns[0])

#%% --- Step 1: Data prep ---
X_baseline = data.select_dtypes(include=[np.number])
if 'label' in X_baseline.columns:
    X_baseline = X_baseline.drop(columns=['label'])

y_baseline = data['label'].map({'lipoma': 0, 'liposarcoma': 1})
plot_baseline_comparison(X_baseline, y_baseline)

le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

# %% --- Step 2: k-fold cross validation ---
folds = kfold(data, target_column='label', n_splits=5)

best_results_per_model = {}
auc_tabel_data = []

for i, pakketje in enumerate(folds):
    print(f"\n--- Fold {i+1} ---")
    
    X_train_outer, X_test_outer = pakketje['X_train'], pakketje['X_test']
    y_train_outer, y_test_outer = pakketje['y_train'], pakketje['y_test']

    getrainde_pipelines = inner_loop(X_train_outer, y_train_outer)
    
    fold_scores = {'Fold': i + 1}
    for naam, model in getrainde_pipelines.items():
        # --- TEST ---
        y_score_test = model.predict_proba(X_test_outer)[:, 1]
        fpr_test, tpr_test, _ = metrics.roc_curve(y_test_outer, y_score_test)
        auc_test = metrics.auc(fpr_test, tpr_test)
        
        # --- TRAIN ---
        y_score_train = model.predict_proba(X_train_outer)[:, 1]
        fpr_train, tpr_train, _ = metrics.roc_curve(y_train_outer, y_score_train)
        auc_train = metrics.auc(fpr_train, tpr_train)
        
        # Sla op voor de tabel (DIT MOET OVEREENKOMEN MET JE PRINT STAP)
        fold_scores[f"{naam}_Test"] = round(auc_test, 3)
        fold_scores[f"{naam}_Train"] = round(auc_train, 3)

        # Update Beste Fold op basis van TEST AUC
        if naam not in best_results_per_model or auc_test > best_results_per_model[naam]['auc']:
            best_results_per_model[naam] = {
                'auc': auc_test,
                'fpr_test': fpr_test,
                'tpr_test': tpr_test,
                'fpr_train': fpr_train,
                'tpr_train': tpr_train,
                'pipeline': model
            }

    auc_tabel_data.append(fold_scores)

    # --- FEATURE ANALYSE ---
    test_kolommen = {k: v for k, v in fold_scores.items() if k.endswith('_Test')}
    winnaar_kolom = max(test_kolommen, key=test_kolommen.get) # Bijv: 'RandomForest_Test'
    winnaar_naam = winnaar_kolom.replace('_Test', '')        # Wordt: 'RandomForest'
    
    winnaar_model = getrainde_pipelines[winnaar_naam]    
    rfecv = winnaar_model.named_steps['rfecv']
    filter_stap = winnaar_model.named_steps['feature_filter']
    namen_na_filter = filter_stap.columns_to_keep_

    winnende_features = [
        namen_na_filter[j] for j in range(len(namen_na_filter)) if rfecv.support_[j]
    ]
    
    current_fold_df = pd.DataFrame({
        'Fold': i + 1,
        'Model': winnaar_naam,
        'Feature': winnende_features
    })

    alle_fold_features.append(current_fold_df)

#%%# --- RESULTATEN RAPPORT ---
df_auc_final = pd.DataFrame(auc_tabel_data)

cols = ['Fold'] + sorted([c for c in df_auc_final.columns if c != 'Fold'])
df_auc_final = df_auc_final[cols]

print("\n📊 AUC VERGELIJKING: TRAIN VS TEST PER FOLD:")
print("=========================================================================")
print(df_auc_final.to_string(index=False))
print("=========================================================================")

# ===== ROC TRAIN VS TEST =====
plot_train_test_roc(best_results_per_model)


# %% --- Stap 4: Optimale Parameters ---
print("\n⚙️ OPTIMALE PARAMETERS PER TOPMODEL:")
print("============================================")

for naam, data in best_results_per_model.items():
    pipeline = data['pipeline']
    best_clf = pipeline.named_steps['clf']
    
    print(f"\nModel: {naam} (Beste Test AUC: {data['auc']:.3f})")

    params = best_clf.get_params()
    
    if naam == 'RandomForest':
        items = ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features']
    elif naam == 'LogisticRegression':
        items = ['C', 'penalty', 'solver']
    elif naam == 'XGBoost':
        items = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 
                 'colsample_bytree', 'min_child_weight', 'gamma', 'reg_lambda']
    else:
        items = params.keys()

    for item in items:
        waarde = params.get(item)
        if waarde is not None:
            print(f"    - {item:20}: {waarde}")

print("\n====================================================")

# %% mean and std
df_auc_final_scores = df_auc_final.drop(columns=['Fold'])
mean_std = df_auc_final_scores.agg(['mean', 'std']).T
mean_std.columns = ['Gemiddelde', 'SD']
mean_std = mean_std.round(2)
print(mean_std)

# %% --- Extra: BOXPLOT ---
df_plot = df_auc_final.melt(id_vars=['Fold'], var_name='Metric', value_name='AUC')
df_plot[['Model', 'Type']] = df_plot['Metric'].str.split('_', expand=True)

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
ax = sns.boxplot(x='Model', y='AUC', hue='Type', data=df_plot, palette='Set2')
sns.stripplot(x='Model', y='AUC', hue='Type', data=df_plot, 
              dodge=True, color='black', alpha=0.3, jitter=True, ax=ax)

plt.title('Verdeling van AUC scores over 5 Folds (Train vs Test)', fontsize=14, fontweight='bold')
plt.ylabel('AUC Score')
plt.xlabel('Model')
plt.ylim(0.4, 1.05) # AUC loopt altijd tot 1.0
plt.legend(title='Fase', loc='lower right')

plt.tight_layout()
plt.show()

#%% --- Extra: Metrics ---
metrics_best_fold(folds, best_results_per_model)

# %%
