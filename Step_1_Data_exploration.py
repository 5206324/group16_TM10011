#%% importeren
import os
import sys
import shutil
from sklearn.model_selection import StratifiedKFold
from streamline.runners.dataprocess_runner import DataProcessRunner
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import metrics

path = ("/Users/fenne/Documents/Technical Medicine/TM10011/Project/group16_TM10011/Lipo_radiomicFeatures.csv")
sys.path.append(os.getcwd())

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Lipo_radiomicFeatures.csv'), index_col=0)

    return data
data = load_data()


# %% STREAMLINE Phase 1 - Run parameters
demo_run = False
use_data_prompt = False

def setup_pipeline_folders(experiment_name, data_path):
    """Maakt de mappenstructuur aan zoals STREAMLINE dat verwacht."""
    base_path = os.path.dirname(data_path)
    
    # Maak UserOutput en UserData mappen aan
    output_folder = os.path.join(base_path, 'UserOutput', experiment_name)
    custom_data_path = os.path.join(base_path, 'UserData')

    for folder in [output_folder, custom_data_path]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Map aangemaakt: {folder}")
            
    return output_folder, custom_data_path

if not demo_run:
    # Paden aanmaken 
    base_folder = '/Users/fenne/Documents/Technical Medicine/TM10011/Project/group16_TM10011'
    output_path = os.path.join(base_folder, 'Lipo_Results')
    experiment_name = 'Lipo_Analysis_Step1_6'

    # Mappen aanmaken
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Succes! Map aangemaakt op: {output_path}")
    else:
        print(f"Map bestond al op: {output_path}")

    output_path, custom_data_path = setup_pipeline_folders(experiment_name, data_path)

    #Pipeline parameters
    class_label = 'label'
    instance_label = 'ID'
    match_label = None
    #applyToReplication = False
    #rep_data_path = None
    #dataset_for_rep = data_path

    #Leaving out features
    ignore_features = [] #als we kolommen willen gaan negeren
    categorical_feature_headers = None
    quantitiative_feature_headers = [
        col for col in data.columns 
        if col not in [class_label, instance_label] + ignore_features
    ]

    print(f"Aantal kwantitatieve features geïdentificeerd: {len(quantitiative_feature_headers)}")
    print(f"\n--- Pipeline Setup Gereed ---")
    print(f"Experiment: {experiment_name}")
    print(f"Resultaten worden opgeslagen in: {output_path}")

 #%% STREAMLINE Phase 1 - Cross Validation (CV)
n_splits = 3  # 
partition_method = 'Stratified' 

# Cutoffs
categorical_cutoff = 10 
sig_cutoff = 0.05 

# Set Random Seed for Reproducible Analysis
random_state = 42 

# EDA outpute file controls (None, outputs all files)
exclude_eda_output = None 
top_uni_features = 20 

# Data processing parameters (cleaning and feature engineering)
featureeng_missingness = 0.5 
cleaning_missingness = 0.5 
correlation_removal_threshold = 1 

# %% STREAMLINE Phase 1 - Uitvoering


# We maken de runner aan met alle parameters die je eerder hebt gedefinieerd
dpr = DataProcessRunner(data_path, output_path, experiment_name,
                exclude_eda_output=exclude_eda_output,
                class_label=class_label, instance_label=instance_label,
                match_label=match_label, n_splits=n_splits,
                partition_method=partition_method,
                ignore_features=ignore_features,
                categorical_features=categorical_feature_headers,
                quantitative_features=quantitiative_feature_headers,
                top_features=top_uni_features,
                categorical_cutoff=categorical_cutoff, sig_cutoff=sig_cutoff,
                featureeng_missingness=featureeng_missingness,
                cleaning_missingness=cleaning_missingness,
                correlation_removal_threshold=correlation_removal_threshold,
                random_state=random_state, show_plots=True)

# Start de analyse
print("Starten van STREAMLINE Stap 1")
dpr.run(run_parallel=False) 
print("Stap 1 voltooid")
# %

#%%
# %% STREAMLINE Phase 1 - Uitvoering (Gecorrigeerd)
import os
import pandas as pd
from streamline.runners.dataprocess_runner import DataProcessRunner

# 1. SCHONE PADEN
path = "/Users/fenne/Documents/Technical Medicine/TM10011/Project/group16_TM10011/Lipo_radiomicFeatures.csv"
# We maken één specifieke map voor ALLES
output_path = "/Users/fenne/Documents/Technical Medicine/TM10011/Project/group16_TM10011/STREAMLINE_OUTPUT"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 2. DATA CHECK (Even kijken of hij de data wel echt ziet)
data = pd.read_csv(path, index_col=0)
print(f"Data geladen! Aantal patiënten: {len(data)}")

# 3. DE RUNNER
experiment_name = 'Clean_Run_V1'

dpr = DataProcessRunner(
    path, 
    output_path, 
    experiment_name,
    class_label='label', 
    instance_label='ID',
    n_splits=3,
    quantitative_features=[col for col in data.columns if col not in ['label', 'ID']],
    show_plots=False # We zetten dit uit om de 'lege mappen' bug te omzeilen
)

print("Analyse start... even geduld...")
dpr.run(run_parallel=False)
print(f"Klaar! Kijk nu alleen in: {output_path}/{experiment_name}")
# %%
