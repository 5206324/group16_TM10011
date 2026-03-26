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

feature_names = data.drop(columns=['label']).columns
alle_analyse_data =[]
alle_fold_features = []
mijn_getrainde_modellen = {}
outer_results = []
best_results_per_model = {}

for i, pakketje in enumerate(folds):
    print(f"\n--- Fold {i+1}---")
    X_train_outer, X_test_outer = pakketje['X_train'], pakketje['X_test']
    y_train_outer, y_test_outer = pakketje['y_train'], pakketje['y_test']

    # 1. Training
    beste_model, naam = inner_loop(X_train_outer, y_train_outer)
    
   # KEY STAP: Sla op met unieke naam
    label = f"{naam} (Fold {i+1})"
    mijn_getrainde_modellen[label] = beste_model
    
    # 2. Resultaten verzamelen
    beste_model, naam = inner_loop(X_train_outer, y_train_outer)
    
    # --- 2. Resultaten verzamelen (AANGEPAST) ---
    
    # Bereken AUC voor deze fold
    y_score_outer = beste_model.predict_proba(X_test_outer)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test_outer, y_score_outer)
    auc_fold = metrics.auc(fpr, tpr)
    
    print(f"DEBUG Fold {i+1}: {naam} heeft AUC {auc_fold:.2f}")

    # Check of dit model al bekend is, en of deze AUC beter is
    if naam not in best_results_per_model or auc_fold > best_results_per_model[naam]['auc']:
        # Sla deze (betere) curve en AUC op
        best_results_per_model[naam] = {
            'auc': auc_fold,
            'fpr': fpr,
            'tpr': tpr,
            'fold_nr': i + 1
        }
        print(f"✅ Nieuwe beste curve voor {naam} gevonden in Fold {i+1}!")

    # --- FEATURE ANALYSE (Houd dit gescheiden!) ---
    rfecv_stap = beste_model.named_steps['rfecv']
    filter_stap = beste_model.named_steps['feature_filter']
    namen_na_filter = filter_stap.columns_to_keep_
    winnende_features = [namen_na_filter[j] for j in range(len(namen_na_filter)) if rfecv_stap.support_[j]]
    
    # Sla features op in de ANDERE lijst
    alle_fold_features.append(pd.DataFrame({
        'Fold': i + 1,
        'Model': naam,
        'Feature': winnende_features
    }))

    alle_fold_features.append(fold_df)

    # De data_fold (AUC, etc.) gaat in de ANDERE lijst:
    alle_analyse_data.append(data_fold)


# --- VISUALISATIE & FEATURE ANALYSE ---
    if 'rfecv' in beste_model.named_steps:
        rfecv = beste_model.named_steps['rfecv']
        filter_stap = beste_model.named_steps['feature_filter']
        
        print(f"\n✅ Fold {i+1} Klaar ({naam})")
        print(f"Optimaal aantal features: {rfecv.n_features_}")
        print(f"Winnende features: {winnende_features}")

        # 2. De RFECV Curve Plotten
        plt.figure(figsize=(8, 4))
        
        # De x-as: we berekenen de punten op basis van de scores die RFECV heeft opgeslagen
        scores = rfecv.cv_results_['mean_test_score']
        # x-as loopt van min_features tot totaal aantal in stappen van 'step'
        x_range = np.arange(rfecv.min_features_to_select, 
                            rfecv.min_features_to_select + len(scores) * rfecv.step, 
                            rfecv.step)

        plt.plot(x_range, scores, color='#2E86C1', marker='o', markersize=4, label='Mean AUC')
        plt.axvline(x=rfecv.n_features_, color='#E74C3C', linestyle='--', label=f'Optimum: {rfecv.n_features_}')
        
        plt.title(f"RFECV Curve - Fold {i+1} ({naam})")
        plt.xlabel("Aantal Features")
        plt.ylabel("AUC Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



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
import matplotlib.pyplot as plt

def plot_best_classifiers_roc(best_results_dict):
    plt.figure(figsize=(9, 7))
    
    # Definieer de kleuren om het oorspronkelijke plaatje te benaderen
    color_map = {
        'Gaussian': 'C0',          # Blauw
        'RandomForest': 'C1',      # Oranje
        'LogisticRegression': 'C2', # Groen
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
