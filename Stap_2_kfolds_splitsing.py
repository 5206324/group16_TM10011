#%% === STEP 2: K-FOLDS AANMAKEN ===
from sklearn.model_selection import StratifiedKFold

def kfold(data, target_column='label', n_splits=5):

    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    alle_folds = []
    
    for train_idx, test_idx in skf.split(X, y):
        pakketje = {
            'X_train': X.iloc[train_idx],
            'X_test': X.iloc[test_idx],
            'y_train': y.iloc[train_idx],
            'y_test': y.iloc[test_idx]
        }
        alle_folds.append(pakketje)
        print(f"DEBUG: {len(alle_folds)} folds succesvol aangemaakt.")
    return alle_folds
# %%
