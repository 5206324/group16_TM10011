#%%

import pandas as pd
import os

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
# %%
