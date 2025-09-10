import pandas as pd
import numpy as np
import pickle
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline 
from xgboost import XGBClassifier

class AgeCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=['Age'])
        elif isinstance(X, pd.Series):
            X_df = X.to_frame()
        else:
            X_df = X.copy() 

        if 'Age' in X_df.columns:
            X_df['Age'] = X_df['Age'].astype(str).str.replace(' years', '', regex=False)
            X_df['Age'] = pd.to_numeric(X_df['Age'], errors='coerce')
        
        return X_df.values 

target_categories = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

feature_columns = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
    'CALC', 'MTRANS'
]


def load_model_and_encoder(model_path='final_model.pkl', encoder_path='target_encoder.pkl'):
    model_pipeline = None
    target_encoder = None

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please ensure it is saved correctly.")
    else:
        try:
            with open(model_path, 'rb') as f:
                model_pipeline = pickle.load(f)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")

    if not os.path.exists(encoder_path):
        print(f"Error: Target encoder file not found at {encoder_path}. Please ensure it is saved correctly.")
    else:
        try:
            with open(encoder_path, 'rb') as f:
                target_encoder = pickle.load(f)
            print(f"Successfully loaded target encoder from {encoder_path}")
        except Exception as e:
            print(f"Error loading target encoder from {encoder_path}: {e}")

    return model_pipeline, target_encoder

if __name__ == '__main__':
    print("Attempting to load model and encoder for testing...")
    loaded_model, loaded_encoder = load_model_and_encoder()

    if loaded_model and loaded_encoder:
        print("\nModel and encoder loaded successfully for testing.")
    
        dummy_data = {
            'Gender': 'Male',
            'Age': 31.0,
            'Height': 1.87,
            'Weight': 128.87,
            'family_history_with_overweight': 'yes',
            'FAVC': 'yes',
            'FCVC': 2.96,
            'NCP': 3.00,
            'CAEC': 'Sometimes',
            'SMOKE': 'yes',
            'CH2O': 1.28,
            'SCC': 'no',
            'FAF': 0.90,
            'TUE': 1.875,
            'CALC': 'Sometimes',
            'MTRANS': 'Automobile'
        }
        
        sample_df = pd.DataFrame([dummy_data], columns=feature_columns)
        print("\nSample Input DataFrame for testing:\n", sample_df)

        try:
            prediction_encoded = loaded_model.predict(sample_df)
            prediction_decoded = loaded_encoder.inverse_transform(prediction_encoded.reshape(-1, 1))
            print(f"Sample prediction (encoded): {prediction_encoded}")
            print(f"Sample prediction (decoded): {prediction_decoded[0][0]}")
        except Exception as e:
            print(f"Error during sample prediction: {e}")
    else:
        print("\nFailed to load model and/or encoder. Please ensure the .pkl files exist.")
