# %% --- Step 1: Data exploration ---
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import sys
import numpy as np

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
from Stap_2_kfolds_splitsing import kfold
from Stap_3_inner_loop import inner_loop
from Stap_7_Data_verzameling import analyse, resultaten

folds = kfold(data, target_column='label', n_splits=5)
#alle_scores = []
#alle_importances = []
feature_names = data.drop(columns=['label']).columns
alle_analyse_data =[]

for i, pakketje in enumerate(folds):
    print (f"\n--- Fold {i+1}---")

    #Data ophalen
    X_train_outer = pakketje['X_train']
    X_test_outer = pakketje['X_test']
    y_train_outer = pakketje['y_train']
    y_test_outer = pakketje['y_test']

    print(f"Train n={len(X_train_outer)}, Test n={len(X_test_outer)}")

    # 4. Pak de trainingsdata (92 samples) en stuur naar de 'n training' fase
    beste_model = inner_loop(X_train_outer, y_train_outer)
    # Verander dit (regel 25-28 in je code):
    data_fold = resultaten(beste_model, 
                           X_train_outer, y_train_outer, # Deel 1: Train
                           X_test_outer, y_test_outer,   # Deel 2: Test
                           feature_names)                # Deel 3: Namen
    
    # Voeg toe aan onze grote lijst voor de finale analyse
    alle_analyse_data.append(data_fold)

    # HIER behoud je de zichtbare output per fold:
    print(f"Fold {i+1} - Train Acc: {data_fold['train_acc']:.2%}")
    print(f"Fold {i+1} - Test Acc: {data_fold['test_acc']:.2%}")
    #print(f"Fold {i+1} - AUC: {data_fold['auc']:.2f}")
    # Data verzamelen voor post-analyse
 #   score, imp = verzamel_resultaten(beste_model_fold, pakketje['X_test'], 
 #                                    pakketje['y_test'], feature_names)
 #   
 #   alle_scores.append(score)
 #   if imp is not None:
 #       alle_importances.append(imp)

  #  print(f"Score Fold {i+1}: {score:.3f}")
#if len(alle_analyse_data) == 5:
#    toon_vergelijking(alle_analyse_data)
#else:
#    print(f"WAARSCHUWING: Er zijn maar {len(alle_analyse_data)} folds verwerkt!")

# #toon_vergelijking(alle_analyse_data)
# for i, pakketje in enumerate(folds):
#     print (f"\n--- Fold {i+1}---")
    
#     # ... (je bestaande data ophaal code) ...

    # 4. Pak de trainingsdata en stuur naar de 'n training' fase
    beste_model = inner_loop(X_train_outer, y_train_outer)
    
#     # --- NIEUW: Haal de winnaar en zijn instellingen op ---
#     # 1. De naam van de classifier (bijv. 'RandomForestClassifier')
#     clf_naam = type(beste_model.named_steps['clf']).__name__
    
#     # 2. De hyperparameters van alleen de classifier-stap
#     # We filteren de hele pipeline-lijst op alles wat met 'clf__' begint
#     alle_params = beste_model.get_params()
#     best_params_clf = {k.split('__')[1]: v for k, v in alle_params.items() if k.startswith('clf__')}
    
#     print(f"🏆 Winnaar Fold {i+1}: {clf_naam}")
#     print(f"⚙️ Instellingen: {best_params_clf}")
#     # -----------------------------------------------------

#     # De rest van je code...
#     data_fold = resultaten(beste_model, X_train_outer, y_train_outer, X_test_outer, y_test_outer, feature_names)
#     alle_analyse_data.append(data_fold)

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
