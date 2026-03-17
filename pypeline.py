# %% Step 1: Data exploration
#from Step_1_Data_exploration.py import ...
# Benodigde pakketten
import pandas as pd
from pathlib import Path
import sys

# Data inladen - vanuit folder ipv path
# Definitie van maken om zo weer te kunnen gebruiken in een ander

sys.path.append(str(Path.cwd()))
from Stap_1_Data_inladen import data_lipo 
data = data_lipo("Lipo_radiomicFeatures.csv")

print("Data succesvol ingeladen!")
print(data.head())


# %% Step 2: k-fold cross validation
#from Step_2_k_fold_cross_validation.py import ...

# %% Step 3: Imputation of missing values
#from Step_3_Imputation_of_missing_values.py import ...

# %% Step 4: Scaling
#from Step_4_Scaling.py import ...

# %% Step 5: Feature importance + selection
#from Step_5_Feature_importance_and_selection.py import ...

# %% Step 6: Machine learning modeling
#from Step_6_Machine_learning_modeling.py import ...

# %% Step 7: Post analysis
#from Step_7_Post_analysis.py import ...
