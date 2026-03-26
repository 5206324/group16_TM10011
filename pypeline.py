# %% --- Step 1: Data exploration ---
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import sys
import numpy as np
from sklearn import metrics 

import matplotlib.pyplot as plt
from Stap_4_visualisatie_ROC_curve import plot_compare_train_test_roc

# Data inladen
sys.path.append(str(Path.cwd()))
from Stap_1_Data_inladen import data_lipo 
data = data_lipo("Lipo_radiomicFeatures.csv")
if 'label' in data.columns:
    data = data.set_index(data.columns[0])

#%% --- Step 1b: Data visualisatie ---
from Stap_1b_visualisatie import plot_baseline_comparison

X_baseline = data.select_dtypes(include=[np.number])
if 'label' in X_baseline.columns:
    X_baseline = X_baseline.drop(columns=['label'])
y_baseline = data['label'].map({'lipoma': 0, 'liposarcoma': 1})

plot_baseline_comparison(X_baseline, y_baseline)


# %% --- Step 2: k-fold cross validation ---
from sklearn.preprocessing import LabelEncoder

# Direct na het inladen van je data:
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

print(f"Klassen succesvol omgezet: {dict(zip(le.classes_, le.transform(le.classes_)))}")

from Stap_2_kfolds_splitsing import kfold
from Stap_3_inner_loop import inner_loop
from Stap_7_Data_verzameling import analyse, resultaten

folds = kfold(data, target_column='label', n_splits=5)

# feature_names = data.drop(columns=['label']).columns
# alle_analyse_data =[]
# alle_fold_features = []
# mijn_getrainde_modellen = {}
# outer_results = []
best_results_per_model = {}
auc_tabel_data = []
alle_fold_features = []

for i, pakketje in enumerate(folds):
    print(f"\n--- Fold {i+1} ---")
    X_train_outer, X_test_outer = pakketje['X_train'], pakketje['X_test']
    y_train_outer, y_test_outer = pakketje['y_train'], pakketje['y_test']

    # 1. Krijg ALLE getrainde modellen (RF, LG, XGB) van de inner loop
    getrainde_pipelines = inner_loop(X_train_outer, y_train_outer)
    
    # Maak een rij aan voor de tabel van deze fold
    fold_scores = {'Fold': i + 1}

    # 2. Loop door de dictionary (.items() werkt nu omdat we de inner_loop hebben aangepast)
    for naam, model in getrainde_pipelines.items():
        # Bereken score op de test-set (outer fold)
        y_score = model.predict_proba(X_test_outer)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test_outer, y_score)
        auc_val = metrics.auc(fpr, tpr)
        
        # Sla score op voor de tabel
        fold_scores[naam] = round(auc_val, 3)

        # 3. Update de "Beste Fold" voor de uiteindelijke ROC curve
        if naam not in best_results_per_model or auc_val > best_results_per_model[naam]['auc']:
    
    # TEST
        y_score_test = model.predict_proba(X_test_outer)[:, 1]
        fpr_test, tpr_test, _ = metrics.roc_curve(y_test_outer, y_score_test)
    
    # TRAIN
        y_score_train = model.predict_proba(X_train_outer)[:, 1]
        fpr_train, tpr_train, _ = metrics.roc_curve(y_train_outer, y_score_train)

         best_results_per_model[naam] = {
            'auc': auc_val,
            'fpr_test': fpr_test,
            'tpr_test': tpr_test,
            'fpr_train': fpr_train,
            'tpr_train': tpr_train,
            'pipeline': model
        }

    # Voeg de rij (met RF, LG en XGB scores) toe aan de tabel-lijst
    auc_tabel_data.append(fold_scores)

    # --- FEATURE ANALYSE ---
    # We pakken de features van het model dat in DEZE fold de hoogste AUC had
    # (Of je kunt dit aanpassen naar een specifiek model zoals RF)
    winnaar_naam = max(fold_scores, key=lambda k: fold_scores[k] if k != 'Fold' else -1)
    winnaar_model = getrainde_pipelines[winnaar_naam]
    
    rfecv = winnaar_model.named_steps['rfecv']
    filter_stap = winnaar_model.named_steps['feature_filter']
    namen_na_filter = filter_stap.columns_to_keep_
    winnende_features = [namen_na_filter[j] for j in range(len(namen_na_filter)) if rfecv.support_[j]]
    
    current_fold_df = pd.DataFrame({
        'Fold': i + 1,
        'Model': winnaar_naam,
        'Feature': winnende_features
    })

    alle_fold_features.append(current_fold_df)
    plt.figure(figsize=(8, 4))
    scores = rfecv.cv_results_['mean_test_score']
    x_range = np.arange(rfecv.min_features_to_select, 
                        rfecv.min_features_to_select + len(scores) * rfecv.step, 
                        rfecv.step)
    plt.plot(x_range, scores, marker='o', label='Mean AUC')
    plt.title(f"RFECV Curve - Fold {i+1} (Winnaar: {winnaar_naam})")
    plt.xlabel("Aantal Features"), plt.ylabel("AUC Score"), plt.legend(), plt.show()

# --- FINALE RESULTATEN ---

# 1. Print de tabel zoals op je whiteboard
df_auc_final = pd.DataFrame(auc_tabel_data)
print("\n📊 AUC PER MODEL PER FOLD (TEST SET):")
print("============================================")
print(df_auc_final.to_string(index=False))
print("============================================")

#%%
# --- FEATURE STABILITEIT ANALYSE ---

if len(alle_fold_features) > 0:
    df_eindrapport = pd.concat(alle_fold_features, ignore_index=True)

    print("\n--- OVERZICHT ALLE FOLDS ---")
    print(df_eindrapport)

    frequentie_tabel = df_eindrapport['Feature'].value_counts().reset_index()
    frequentie_tabel.columns = ['Radiomics Feature', 'Aantal keer gekozen']

    print("\n🏆 STABILITEITS-ANALYSE (Top 10):")
    print(frequentie_tabel.head(10))
else:
    print("⚠️ Geen features verzameld — check je loop (append ontbreekt?)")



# %%
import matplotlib.pyplot as plt

def plot_best_classifiers_roc(best_results_dict):
    plt.figure(figsize=(9, 7))
    
    # Definieer de kleuren om het oorspronkelijke plaatje te benaderen
    color_map = {
        'Gaussian': 'C0',          # Blauw
        'RandomForest': 'C1',      # Oranje
        'LogisticRegression': 'C2', # Groen
        'XGBoost': 'C4' ,            #paars
        'SVC': 'C3'                # Rood
    }
    
    # Loop door de beste resultaten per model
    for model_name, data in best_results_dict.items():
        # Bepaal de kleur, gebruik grijs als het model niet in de map staat
        color = color_map.get(model_name, 'grey')
        
        # Plot de trapvormige curve (best passende bij medische data)
        plt.step(data['fpr'], data['tpr'], where='post', color=color,
                 label=f"{model_name} (AUC = {data['auc']:.2f})")
    
    # Lay-out instellingen (zoals in je voorbeeldfoto)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', alpha=0.5) # Diagonale kanslijn
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.title('ROC Curve Comparison test set (Best Fold per Model)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- De functie aanroepen ---
# Doe dit helemaal aan het einde van je script
plot_best_classifiers_roc(best_results_per_model)
# %%
# 3. Stabiliteits-analyse (Welke features komen het vaakst voor?)
df_features = pd.concat(alle_fold_features, ignore_index=True)
frequentie = df_features['Feature'].value_counts().reset_index()
frequentie.columns = ['Radiomics Feature', 'Aantal keer gekozen']
print("\n🏆 FEATURE STABILITEIT (Top 10):")
print(frequentie.head(10))
# %% 

# Plak alle folds onder elkaar
df_eindrapport = pd.concat(alle_fold_features, ignore_index=True)

# 1. Toon de volledige lijst
print("\n--- OVERZICHT ALLE FOLDS ---")
print(df_eindrapport)

# 2. De "Gouden Tip": Welke features komen het vaakst voor?
frequentie_tabel = df_eindrapport['Feature'].value_counts().reset_index()
frequentie_tabel.columns = ['Radiomics Feature', 'Aantal keer gekozen (max 5)']

print("\n🏆 STABILITEITS-ANALYSE (Welke features zijn het meest robuust?):")
print(frequentie_tabel)
# %%
def plot_train_test_roc(best_results_dict):
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    for model_name, data in best_results_dict.items():
        
        # TRAIN
        ax1.plot(data['fpr_train'], data['tpr_train'],
                 label=f"{model_name}")
        
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
# %%
