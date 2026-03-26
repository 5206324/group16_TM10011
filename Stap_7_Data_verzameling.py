#%% === STEP 7: RESULTATEN ===
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def resultaten(beste_model, X_train, y_train, X_test, y_test, feature_names, model_naam):
    # 1. Scores berekenen
    train_acc = beste_model.score(X_train, y_train)
    test_acc = beste_model.score(X_test, y_test)
    
    # 2. AUC berekenen (met een 'try-except' voor het geval een model geen kansen geeft)
    try:
        y_probs = beste_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probs)
    except:
        roc_auc = 0.0
    
    # We geven de dictionary terug met exact de namen die je gewend bent
    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'auc': roc_auc,
        'model': model_naam 
    }

def analyse(alle_data):
    # We maken de DataFrame
    df = pd.DataFrame(alle_data)
    
    # Voeg een kolom toe voor het fold nummer (1 t/m 5)
    df.index = [f"Fold {i+1}" for i in range(len(df))]
    
    # We sorteren de kolommen even voor een mooie weergave in de print
    # Kolomvolgorde: Model, Train Acc, Test Acc, AUC
    kolommen = ['model', 'train_acc', 'test_acc', 'auc']
    df = df[kolommen]

    print("\n=== RESULTATEN PER FOLD ===")
    # We printen de tabel (round 4 is prima voor de decimalen)
    print(df.round(4))
    
    print("\n=== GEMIDDELDEN ===")
    print(f"Gemiddelde Train Acc: {df['train_acc'].mean():.2%}")
    print(f"Gemiddelde Test Acc:  {df['test_acc'].mean():.2%}")
    print(f"Gemiddelde AUC:       {df['auc'].mean():.4f}")
# %%
