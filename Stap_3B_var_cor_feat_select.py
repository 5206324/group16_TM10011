#%%% === STEP 3B: variantie en correlatie van features berekenen
#  Featues met hoge correlatie en lage variantie weghalen

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold

class VarianceCorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.01, correlation_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)
        self.columns_to_keep_ = None  # Hier slaan we de winnende namen op
        self.feature_names_in_ = None # Originele namen

    def fit(self, X, y=None):
        # 1. Zorg dat we een DataFrame hebben met kolomnamen
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        
        self.feature_names_in_ = X_df.columns.tolist()

        # 2. Variantie Filter
        self.variance_selector.fit(X_df)
        # Welke namen overleven de variantie check?
        cols_after_var = X_df.columns[self.variance_selector.get_support()].tolist()
        data_var_df = X_df[cols_after_var]

        print(f"Features reduced from {len(self.feature_names_in_)} to {len(cols_after_var)} na variantie filtering")

        # 3. Correlatie Filter
        corr_matrix = data_var_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        
        # 4. De definitieve lijst met namen die we houden
        self.columns_to_keep_ = [col for col in cols_after_var if col not in to_drop]

        #print(f"Features reduced from {len(cols_after_var)} to {len(self.columns_to_keep_)} na correlatie filtering")
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        else:
            X_df = X
        
        # Geef alleen de kolommen terug die we in 'fit' hebben goedgekeurd
        return X_df[self.columns_to_keep_]

    def get_support(self, indices=False):
        """ Maakt deze filter compatibel met Scikit-learn tools zoals RFECV """
        mask = np.array([col in self.columns_to_keep_ for col in self.feature_names_in_])
        return np.where(mask)[0] if indices else mask