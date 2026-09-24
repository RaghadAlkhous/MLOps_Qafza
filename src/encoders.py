import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping_ = None
        self.default_ = 0.0

    def fit(self, X, y=None):
        X_series = pd.Series(X).astype("string")
        frequencies = X_series.value_counts(normalize=True, dropna=False)
        self.mapping_ = frequencies.to_dict()
        return self

    def transform(self, X):
        if self.mapping_ is None:
            raise RuntimeError("FrequencyEncoder must be fitted before transform.")
        X_series = pd.Series(X).astype("string")
        encoded = X_series.map(self.mapping_).fillna(self.default_).astype(float)
        return encoded.to_numpy().reshape(-1, 1)
