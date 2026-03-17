#%%
import pandas as pd
from pathlib import Path

def data_lipo(bestandsnaam):
    data_path = Path.cwd() / bestandsnaam
    
    if not data_path.exists():
        raise FileNotFoundError(f"Bestand niet gevonden")

    data = pd.read_csv(data_path)
    return data
    
# %%
