#%% stap 7
import pandas as pd
import numpy as np

def resultaten(beste_model, X_train, y_train, X_test, y_test, feature_names):
    # We berekenen puur de accuracy scores
    train_acc = beste_model.score(X_train, y_train)
    test_acc = beste_model.score(X_test, y_test)
    
    return {
        'train_acc': train_acc,
        'test_acc': test_acc
    }

def analyse(alle_data):
    df = pd.DataFrame(alle_data)
    # Voeg een kolom toe voor het fold nummer (1 t/m 5)
    df.index = [f"Fold {i+1}" for i in range(len(df))]
    
    print("\n=== RESULTATEN PER FOLD ===")
    print(df.round(4))
    
    print("\n=== GEMIDDELDEN ===")
    print(f"Gemiddelde Train Acc: {df['train_acc'].mean():.2%}")
    print(f"Gemiddelde Test Acc:  {df['test_acc'].mean():.2%}")
# %%
