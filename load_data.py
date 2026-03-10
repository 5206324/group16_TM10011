#%% importeren

import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import metrics

path = ("/Users/fenne/Documents/Technical Medicine/TM10011/Project/group16_TM10011/Lipo_radiomicFeatures.csv")

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Lipo_radiomicFeatures.csv'), index_col=0)

    return data


data = load_data()

# %% Classifiers aanhalen
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# List to loop through later
models_to_test = [
    ('LogReg', LogisticRegression()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('RF', RandomForestClassifier()),
    ('KNN', KNeighborsClassifier(n_neighbors=5))
]
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
# %% print verdeling
lipoma = data[data["label"] == "lipoma"]
liposarcoma = data[data["label"] == "liposarcoma"]

print("Lipoma:", lipoma.shape)
print("Liposarcoma:", liposarcoma.shape)

 #%%Label verdeling
lipoma = data[data["label"] == "lipoma"]
liposarcoma = data[data["label"] == "liposarcoma"]

alle_kolommen = data.columns.tolist()
print(f"TOTALE FEATURES GEONDEN: {len(alle_kolommen)}")
print("-" * 40)

for i, col in enumerate(alle_kolommen, 1):
    # Print index en de volledige naam van de kolom
    print(f"{i:03}. {col}")

print("-" * 40)
# %% Feature exploration --- oppervlakte
area_cols = [
'PREDICT_original_sf_area_avg_2.5D',
'PREDICT_original_sf_area_max_2.5D',
'PREDICT_original_sf_area_min_2.5D',
'PREDICT_original_sf_area_std_2.5D'
]

graph_titles = {
    'PREDICT_original_sf_area_avg_2.5D': 'Average Tumor Area',
    'PREDICT_original_sf_area_max_2.5D': 'Maximum Tumor Area',
    'PREDICT_original_sf_area_min_2.5D': 'Minimum Tumor Area',
    'PREDICT_original_sf_area_std_2.5D': 'Area Standard Deviation'
}

fig, axes = plt.subplots(2, 2, figsize=(12,8))

for i, col in enumerate(area_cols):
    ax = axes[i//2, i%2]
    
    ax.hist(lipoma[col], bins=30, alpha=0.5, label="lipoma")
    ax.hist(liposarcoma[col], bins=30, alpha=0.5, label="liposarcoma")
    ax.set_title(graph_titles[col], fontsize=12, fontweight='bold')
    ax.set_xlabel("Area value")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout()
plt.show()

# %% Feature exploration --- visualisatie
#%% --- 1. CONFIGURATIE ---
USE_PCA = False 
f1 = 'PREDICT_original_sf_roughness_avg_2.5D'
f2 = 'PREDICT_original_sf_convexity_avg_2.5D'

#%% --- 2. DATA VOORBEREIDING ---
if USE_PCA:
    # Gebruik alle numerieke kolommen behalve 'label'
    X_raw = data.drop(columns=["label"])
    title_prefix = "PCA Componenten (Totaaloverzicht)"
else:
    # Controleer of de features bestaan
    if f1 in data.columns and f2 in data.columns:
        X_raw = data[[f1, f2]]
        title_prefix = f"Features: {f1.split('_')[-2]} vs {f2.split('_')[-2]}"
    else:
        print(f"FOUT: Een van de features ({f1} of {f2}) staat niet in de dataset!")
        X_raw = data.drop(columns=["label"])
        USE_PCA = True

# Schalen
X_scaled = StandardScaler().fit_transform(X_raw)
y_numeric = pd.factorize(data["label"])[0]

if USE_PCA:
    pca = PCA(n_components=2)
    X_final = pca.fit_transform(X_scaled)
else:
    X_final = X_scaled 

#%% --- 3. CLASSIFIERS INITIALISEREN ---
clsfs = [
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    GaussianNB(),
    LogisticRegression(),
    KNeighborsClassifier(n_neighbors=5)
]

# Grid instellen
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
results = []

#%% --- 4. DE BENCHMARK LOOP ---
for i, clf in enumerate(clsfs):
    ax = axes[i]
    
    # Model trainen
    clf.fit(X_final, y_numeric)
    y_pred = clf.predict(X_final)
    
    # Achtergrond inkleuren -- Beslissingsvlakken
    x_min, x_max = X_final[:, 0].min() - 1, X_final[:, 0].max() + 1
    y_min, y_max = X_final[:, 1].min() - 1, X_final[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), 
                         np.arange(y_min, y_max, 0.05))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot de grenzen en de data
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Paired)
    ax.scatter(X_final[:, 0], X_final[:, 1], c=y_numeric, 
               s=40, edgecolor='k', cmap=plt.cm.Paired, alpha=0.9)
    
    # Styling en Labels
    model_name = clf.__class__.__name__
    errors = (y_numeric != y_pred).sum()
    acc = (1 - errors/len(y_numeric)) * 100
    
    ax.set_title(f"{model_name}\nAcc: {acc:.1f}% ({errors} fouten)", fontsize=12)
    
    if USE_PCA:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    else:
        ax.set_xlabel(f1.split('_')[-2]) 
        ax.set_ylabel(f2.split('_')[-2])
    
    ax.grid(True, linestyle=':', alpha=0.6)
    results.append({'Model': model_name, 'Accuracy': acc, 'Errors': errors})

# Laatste subplot verwijderen en layout fixen
fig.delaxes(axes[5])
fig.suptitle(f"Benchmark: {title_prefix}", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# --- 5. RESULTATEN TABEL ---
print("\n" + "="*40)
print(f"EINDRESULTATEN: {title_prefix}")
print("="*40)
df_results = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
print(df_results.to_string(index=False))
# %%
