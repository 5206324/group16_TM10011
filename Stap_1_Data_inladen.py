#%% === STEP 1: DATA EXPLORATION & STATISTICAL ANALYSIS ===
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import shapiro

def data_lipo(bestandsnaam):
    data_path = Path.cwd() / bestandsnaam
    if not data_path.exists():
        raise FileNotFoundError(f"Bestand niet gevonden")
    return pd.read_csv(data_path)

# --- Data laden en labelen ---
data = data_lipo('Lipo_radiomicFeatures.csv')
df = data.copy()
df['label_num'] = df['label'].map({'lipoma': 0, 'liposarcoma': 1})

feature_cols = df.select_dtypes(include=[np.number]).columns.drop(['label_num'])    #Pak alleen de getallen (negeer patiënt-ID's zoals 'Lipo-003_0')

not_normal_dist0 = 0                                                                # Tellers
not_normal_dist1 = 0
outliers_total = 0
print(f"Start analyse op {len(feature_cols)} numerieke features...")

# --- Analyse Loop - per feature de verdeling en uitschieters bepalen --- 
for column in feature_cols:
    
    data0 = df[df['label_num'] == 0][column].dropna().values                        # Check Normaliteit
    data1 = df[df['label_num'] == 1][column].dropna().values
    
    if len(data0) > 3:
        _, p0 = shapiro(data0, nan_policy='raise')                                  # Check Normaliteit, betrouwbaar vanaf 3 metingen. nan_policy om te checken voor NaN's
        if p0 > 0.05: not_normal_dist0 += 1                                         # H0: De data is normaal verdeeld. Als p > 0.05, behouden we H0
    if len(data1) > 3:
        _, p1 = shapiro(data1, nan_policy='raise')                                  # Check Normaliteit, betrouwbaar vanaf 3 metingen. nan_policy om te checken voor NaN's
        if p1 > 0.05: not_normal_dist1 += 1                                         # H0: De data is normaal verdeeld. Als p > 0.05, behouden we H0

    all_values = df[column].dropna().values                                         # Check Outliers (IQR)
    if len(all_values) > 0:
        Q1, Q3 = np.percentile(all_values, [25, 75])
        IQR = Q3 - Q1
        if np.any((all_values < (Q1 - 1.5 * IQR)) | (all_values > (Q3 + 1.5 * IQR))):   # Tukey's Fences: Buiten [Q1 - 1.5*IQR, Q3 + 1.5*IQR] wordt gemarkeerd als outlier
            outliers_total += 1

# --- Uitkomsten voor in verslag ---
num_features = len(feature_cols)
print(f"\nRatio Gaussian/Non-Gaussian in HEALTHY: {not_normal_dist0} - {num_features - not_normal_dist0}")
print(f"Ratio Gaussian/Non-Gaussian in SICK:    {not_normal_dist1} - {num_features - not_normal_dist1}")
print(f"Features met Outliers: {outliers_total} van de {num_features}")

if outliers_total > (num_features / 2):                                             # Advies voor scalar gebruik
    print("\nADVIES: Gebruik de RobustScaler (meer dan 50% van de features heeft outliers).")
else:
    print("\nADVIES: StandardScaler is mogelijk, maar RobustScaler blijft veiliger.")
# %%