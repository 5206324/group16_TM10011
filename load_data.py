#%% importeren

import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

path = ("/Users/fenne/Documents/Technical Medicine/TM10011/Project/group16_TM10011/Lipo_radiomicFeatures.csv")

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Lipo_radiomicFeatures.csv'), index_col=0)

    return data




# %% feature summary
feature_summary = {}

for col in data.columns:
    if "_sf_" in col:
        base = col.split("_sf_")[1].split("_")[0]
        stat = col.split(base + "_")[1].split("_")[0]
        
        if base not in feature_summary:
            feature_summary[base] = []
            
        feature_summary[base].append(stat)

for k, v in feature_summary.items():
    print(k, ":", sorted(v))
# %% printing verdeling
lipoma = data[data["label"] == "lipoma"]
liposarcoma = data[data["label"] == "liposarcoma"]

print("Lipoma:", lipoma.shape)
print("Liposarcoma:", liposarcoma.shape)



# %% Aanmaken van test set

# %%
# Features en labels
X = data.drop(columns=["label"])
y = data["label"]

# Stratified K-Fold met 5 splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Itereer over de folds
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    print(f"Fold {fold+1}")
    print("Train:", X_train.shape, y_train.value_counts().to_dict())
    print("Validation:", X_val.shape, y_val.value_counts().to_dict())
    print("-"*30)
# %%
# Splits dataset in train+val en test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# %%
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
# %%
# ik voeg dit nog een keer toe

#%% 
#Branch Fenne