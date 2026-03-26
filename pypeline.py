# %% --- Step 1: Data exploration ---
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import sys
import numpy as np

import matplotlib.pyplot as plt

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

for i, pakketje in enumerate(folds):
    print(f"\n--- Fold {i+1}---")
    X_train_outer, X_test_outer = pakketje['X_train'], pakketje['X_test']
    y_train_outer, y_test_outer = pakketje['y_train'], pakketje['y_test']

    # 1. Training
    beste_model, naam = inner_loop(X_train_outer, y_train_outer)

    # 2. Resultaten verzamelen
    data_fold = resultaten(beste_model, X_train_outer, y_train_outer, X_test_outer, y_test_outer, feature_names, naam)
    alle_analyse_data.append(data_fold)

    rfecv_stap = beste_model.named_steps['rfecv']
    filter_stap = beste_model.named_steps['feature_filter']
    
    # Gebruik de get_support() methode van de nieuwe filter
    masker_filter = filter_stap.get_support()
    namen_na_filter = X_train_outer.columns[masker_filter]
    
    # Pak de winnaars van deze specifieke fold
    winnende_features = namen_na_filter[rfecv_stap.support_].tolist()
    
    # Maak een tijdelijk tabelletje voor deze fold
    df_fold = pd.DataFrame({
        'Fold': i + 1,
        'Model': naam,
        'Feature': winnende_features
    })
    
    alle_fold_features.append(df_fold)

# --- VISUALISATIE & FEATURE ANALYSE ---
    if 'rfecv' in beste_model.named_steps:
        rfecv = beste_model.named_steps['rfecv']
        filter_stap = beste_model.named_steps['feature_filter']
        
        # 1. Namen ophalen (werkt nu direct door onze aanpassing in Stap_3B)
        namen_na_filter = filter_stap.columns_to_keep_
        winnende_features = [namen_na_filter[j] for j in range(len(namen_na_filter)) if rfecv.support_[j]]
        
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

# Optioneel: Sla het direct op voor je verslag
# df_eindrapport.to_csv("alle_features_per_fold.csv", index=False)


# import joblib

# # Sla het beste model van de allerlaatste fold op (of doe dit in de loop per fold)
# bestandsnaam = f"best_model_{naam}.pkl"
# joblib.dump(beste_model, bestandsnaam)

# print(f"✅ Model is opgeslagen als {bestandsnaam}. Je kunt nu de computer afsluiten!")

# # %%
# import pandas as pd

# # 1. Haal de onderdelen uit de pipeline die nu in je geheugen zit
# rfecv_stap = beste_model.named_steps['rfecv']
# filter_stap = beste_model.named_steps['feature_filter']

# # 2. Haal de namen op die de filter hebben overleefd (de 167 namen)
# namen_na_filter = filter_stap.columns_to_keep_

# # 3. Welke van die 167 heeft de RFECV aangevinkt als 'True'?
# # We maken een lijstje van de namen waar het masker 'True' is
# winnende_features = [namen_na_filter[i] for i in range(len(namen_na_filter)) if rfecv_stap.support_[i]]

# # 4. Maak er een mooi tabelletje van voor Fold 5
# df_top_features = pd.DataFrame({
#     'Rank': range(1, len(winnende_features) + 1),
#     'Radiomics Feature': winnende_features
# })

# print("\n--- RESULTAAT FOLD 5 ---")
# print(f"Gekozen model: {naam}")
# print(f"Aantal features: {len(winnende_features)}")
# print(df_top_features.to_string(index=False))

# Optioneel: Sla het direct op als CSV zodat je het in Excel kunt openen
# df_top_features.to_csv("top_features_fold5.csv", index=False)

# %% 

# %% Step 3: Imputation of missing values
#from Step_3_Imputation_of_missing_values.py import ...

# %% Step 4: Scaling
#from Step_4_Scaling.py import ...

# %% Step 5: Feature importance + selection
#from Step_5_Feature_importance_and_selection.py import ...

# %% Step 6: Machine learning modeling
#from Step_6_Machine_learning_modeling.py import ...

# %% Step 7: Post analysis
#from Step_7_Post_analysis.py import ...

# %%
