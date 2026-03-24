#%% Stap 1: Data exploratie
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import shapiro

def data_lipo(bestandsnaam):
    data_path = Path.cwd() / bestandsnaam
    if not data_path.exists():
        raise FileNotFoundError(f"Bestand niet gevonden")
    return pd.read_csv(data_path)

# 1. Laden
data = data_lipo('Lipo_radiomicFeatures.csv')
df = data.copy()
df['label_num'] = df['label'].map({'lipoma': 0, 'liposarcoma': 1})

# 2. SELECTIE: Pak alleen de getallen (negeer patiënt-ID's zoals 'Lipo-003_0')
feature_cols = df.select_dtypes(include=[np.number]).columns.drop(['label_num'])

# 3. Tellers
not_normal_dist0 = 0
not_normal_dist1 = 0
outliers_total = 0

print(f"Start analyse op {len(feature_cols)} numerieke features...")

# 4. De Loop
for column in feature_cols:
    # Check Normaliteit
    data0 = df[df['label_num'] == 0][column].dropna().values
    data1 = df[df['label_num'] == 1][column].dropna().values
    
    if len(data0) > 3:
        _, p0 = shapiro(data0)
        if p0 > 0.05: not_normal_dist0 += 1
    if len(data1) > 3:
        _, p1 = shapiro(data1)
        if p1 > 0.05: not_normal_dist1 += 1

    # Check Outliers (IQR)
    all_values = df[column].dropna().values
    if len(all_values) > 0:
        Q1, Q3 = np.percentile(all_values, [25, 75])
        IQR = Q3 - Q1
        if np.any((all_values < (Q1 - 1.5 * IQR)) | (all_values > (Q3 + 1.5 * IQR))):
            outliers_total += 1

# 5. Eindrapportage
num_features = len(feature_cols)
print(f"\nRatio Gaussian/Non-Gaussian in HEALTHY: {not_normal_dist0} - {num_features - not_normal_dist0}")
print(f"Ratio Gaussian/Non-Gaussian in SICK:    {not_normal_dist1} - {num_features - not_normal_dist1}")
print(f"Features met Outliers: {outliers_total} van de {num_features}")
# Advies voor je verslag
if outliers_total > (num_features / 2):
    print("\nADVIES: Gebruik de RobustScaler (meer dan 50% van de features heeft outliers).")
else:
    print("\nADVIES: StandardScaler is mogelijk, maar RobustScaler blijft veiliger.")
# %%