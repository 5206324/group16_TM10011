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

for i, pakketje in enumerate(folds):
    print(f"\n--- Fold {i+1}---")
    X_train_outer, X_test_outer = pakketje['X_train'], pakketje['X_test']
    y_train_outer, y_test_outer = pakketje['y_train'], pakketje['y_test']

    # 1. Training
    beste_model, naam = inner_loop(X_train_outer, y_train_outer)

    # 2. Resultaten verzamelen
    data_fold = resultaten(beste_model, X_train_outer, y_train_outer, X_test_outer, y_test_outer, feature_names, naam)
    alle_analyse_data.append(data_fold)

#  --- VISUALISATIE & FEATURE ANALYSE ---
    if 'rfecv' in beste_model.named_steps:
        rfecv_stap = beste_model.named_steps['rfecv']
        filter_stap = beste_model.named_steps['feature_filter']
        
        # 1. Haal de namen op na de eerste filter stap
        # We kijken direct in de 'columns_to_keep_' van jouw VarianceCorrelationFilter
        if hasattr(filter_stap, 'columns_to_keep_'):
            namen_na_filter = filter_stap.columns_to_keep_
        else:
            # Als dat niet werkt, gebruiken we de originele X_train kolommen 
            # (Let op: dit werkt alleen als de filter niks heeft weggegooid, 
            #  anders krijg je een lengte-fout)
            namen_na_filter = X_train_outer.columns

        # 2. Welke zijn uiteindelijk door RFECV gekozen?
        rfecv_mask = rfecv_stap.support_
        
        # Match de namen (we zorgen dat de lengte klopt om errors te voorkomen)
        try:
            winnende_features = namen_na_filter[rfecv_mask]
            print(f"\n✅ Analyse Fold {i+1}:")
            print(f"Optimaal aantal features: {rfecv_stap.n_features_}")
            print(f"Geselecteerde features: {winnende_features.tolist()}")
        except Exception as e:
            print(f"⚠️ Kon namen niet matchen: {e}")
            print(f"Optimaal aantal features: {rfecv_stap.n_features_}")

        # 3. Het Plaatje (Score vs Aantal Features)
        scores = rfecv_stap.cv_results_['mean_test_score']
        stds = rfecv_stap.cv_results_['std_test_score']
        
        # Bereken de x-as op basis van de testpunten en de stapgrootte (10)
        n_punten = len(scores)
        # De x-as moet lopen van min_features_to_select tot het totaal aantal features
        # dat de RFECV heeft gezien (n_features_in_)
        x_axis = np.linspace(rfecv_stap.min_features_to_select, 
                             rfecv_stap.min_features_to_select + (n_punten-1) * rfecv_stap.step, 
                             n_punten)

        plt.figure(figsize=(10, 5))
        plt.plot(x_axis, scores, color='#2E86C1', lw=2, marker='o', markersize=4, label='Mean AUC')
        plt.fill_between(x_axis, scores - stds, scores + stds, alpha=0.15, color='#2E86C1')
        
        plt.axvline(x=rfecv_stap.n_features_, color='#E74C3C', linestyle='--', 
                    label=f'Optimum: {rfecv_stap.n_features_} f')
        
        plt.title(f"RFECV Curve - Fold {i+1} ({naam})")
        plt.xlabel("Aantal Features")
        plt.ylabel("Cross-Validation Score (AUC)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show() 
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
