from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class AgeCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_cleaned = X.copy()
        X_cleaned['Age'] = X_cleaned['Age'].astype(str).str.replace(' years', '', regex=False).astype(float)
        return X_cleaned