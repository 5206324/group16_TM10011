import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold


class VarianceCorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.01, correlation_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_selector = VarianceThreshold(threshold=self.variance_threshold)

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        data_var = self.variance_selector.fit_transform(X_df)

        print(
            f"Features reduced from {X_df.shape[1]} to {data_var.shape[1]} "
            "features after variation filtering"
        )

        data_var_df = pd.DataFrame(data_var)
        corr_matrix = data_var_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        self.to_drop_ = [
            column for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]

        print(
            f"Features reduced from {data_var.shape[1]} to "
            f"{data_var.shape[1] - len(self.to_drop_)} "
            "features after correlation filtering"
        )
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        data_var = self.variance_selector.transform(X_df)
        data_var_df = pd.DataFrame(data_var)
        data_cor = data_var_df.drop(columns=self.to_drop_, errors="ignore")
        return data_cor.to_numpy()
