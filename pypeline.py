# %% === Step 1: Data exploration ===
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- Data inladen ---
sys.path.append(str(Path.cwd()))
from Stap_1_Data_inladen import data_lipo 
data = data_lipo("Lipo_radiomicFeatures.csv")
if 'label' in data.columns:                                         #Eerste kolom als index instellen (pt-ID)
    data = data.set_index(data.columns[0])

#%% === Step 1b: Data visualisatie ===
from Stap_1b_visualisatie import plot_baseline_comparison

X_baseline = data.select_dtypes(include=[np.number])                #Features (X) en labels (y) scheiden
if 'label' in X_baseline.columns:
    X_baseline = X_baseline.drop(columns=['label'])
y_baseline = data['label'].map({'lipoma': 0, 'liposarcoma': 1})     #Labels omzetten naar 0 (Lipoma) en 1 (Liposarcoma)

plot_baseline_comparison(X_baseline, y_baseline)                    #Plot maken om te bepalen welke classifiers potentie hebben

# %% === Step 2: k-fold cross validation ===
from sklearn.preprocessing import LabelEncoder
from Stap_2_kfolds_splitsing import kfold
from Stap_3_inner_loop import inner_loop
from Stap_7_Data_verzameling import analyse, resultaten

le = LabelEncoder()                                                 #Labels definitief encoderen voor modellen
data['label'] = le.fit_transform(data['label'])
print(f"Klassen succesvol omgezet: {dict(zip(le.classes_, le.transform(le.classes_)))}") #Vgm kan dit weg

folds = kfold(data, target_column='label', n_splits=5)              #Data in 5 folds verdelen voor cross-validation
feature_names = data.drop(columns=['label']).columns
alle_analyse_data =[]
alle_fold_features = []

for i, pakketje in enumerate(folds):
    print(f"\n--- Fold {i+1}---")
    X_train_outer, X_test_outer = pakketje['X_train'], pakketje['X_test']
    y_train_outer, y_test_outer = pakketje['y_train'], pakketje['y_test']

    #--- INNER LOOP ---
    # 1. Training (Scaling -> Feature Selection -> Hyperparameter Tuning)
    beste_model, naam = inner_loop(X_train_outer, y_train_outer)

    # 2. Resultaten verzamelen
    data_fold = resultaten(beste_model, X_train_outer, y_train_outer, X_test_outer, y_test_outer, feature_names, naam)
    alle_analyse_data.append(data_fold)

    rfecv_stap = beste_model.named_steps['rfecv']                   #Kijken welke features in RFECV zijn gekozen
    filter_stap = beste_model.named_steps['feature_filter']
    
    # Gebruik de get_support() methode van de nieuwe filter
    masker_filter = filter_stap.get_support()
    namen_na_filter = X_train_outer.columns[masker_filter]
    
    winnende_features = namen_na_filter[rfecv_stap.support_].tolist()       # Pak de winnaars van deze specifieke fold
    
    df_fold = pd.DataFrame({                                        # Tijdelijk tabelletje voor deze fold om later de stabiliteit te kunnen bepalen
        'Fold': i + 1,
        'Model': naam,
        'Feature': winnende_features
    })
    
    alle_fold_features.append(df_fold)


# === Step 3: VISUALISATIE & FEATURE ANALYSE PER FOLD===
# Draait aan het einde van elke fold in outer loop
    if 'rfecv' in beste_model.named_steps:
        rfecv = beste_model.named_steps['rfecv']
        filter_stap = beste_model.named_steps['feature_filter']
        
        # 1. Namen ophalen - terugkoppelen indiced RFECV aan kolomnamen 
        namen_na_filter = filter_stap.columns_to_keep_
        winnende_features = [namen_na_filter[j] for j in range(len(namen_na_filter)) if rfecv.support_[j]]
        
        print(f"\n -  Fold {i+1} Klaar ({naam})")
        print(f"Optimaal aantal features: {rfecv.n_features_}")
        print(f"Winnende features: {winnende_features}")

        # 2. De RFECV Curve Plotten
        plt.figure(figsize=(8, 4))
        
        # De x-as: aantal geteste features 
        scores = rfecv.cv_results_['mean_test_score']
        # x-as loopt van min_features tot totaal aantal in stappen van 'step'
        x_range = np.arange(rfecv.min_features_to_select, 
                            rfecv.min_features_to_select + len(scores) * rfecv.step, 
                            rfecv.step)

        plt.plot(x_range, scores, color='#2E86C1', marker='o', markersize=4, label='Mean AUC')
        plt.axvline(x=rfecv.n_features_, color='#E74C3C', linestyle='--', label=f'Optimum: {rfecv.n_features_}')    #Een rode stippelijn om an te geven vanaf welk punt extra features niet meer zorgen voor een betere score
        
        plt.title(f"RFECV Curve - Fold {i+1} ({naam})")
        plt.xlabel("Aantal Features")
        plt.ylabel("AUC Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



# %% === STEP 4: GLOBALE STABILITEITS-ANALYSE ===
df_eindrapport = pd.concat(alle_fold_features, ignore_index=True)   # Plak alle folds onder elkaar

print("\n--- OVERZICHT ALLE FOLDS ---")                             # Toon de volledige lijst
print(df_eindrapport)

frequentie_tabel = df_eindrapport['Feature'].value_counts().reset_index()   # 2. Welke features komen het vaakst voor?
frequentie_tabel.columns = ['Radiomics Feature', 'Aantal keer gekozen (max 5)']

print("\n - STABILITEITS-ANALYSE (Welke features zijn het meest robuust?):")
print(frequentie_tabel)


# %% Step 7: Post analysis
#from Step_7_Post_analysis.py import ...

#