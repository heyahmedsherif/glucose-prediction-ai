#!/usr/bin/env python3
"""
Train and Save Glucose Prediction Models

This script trains the best performing models and saves them for use with the Streamlit app.
Uses A1C instead of baseline glucose as a core feature.

Author: Generated for CGMacros A1C-based glucose prediction
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(csv_file: str = "glucose_prediction_training_data.csv") -> pd.DataFrame:
    """Load and prepare the training data."""
    logger.info(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} records from {df['subject_id'].nunique()} subjects")
    return df


def prepare_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Prepare feature matrix, handling missing values appropriately."""
    X = df[feature_columns].copy()
    
    # Handle missing values
    # For core demographic features, fill with median
    demographic_cols = ['age', 'gender', 'bmi']
    for col in demographic_cols:
        if col in X.columns and X[col].isna().any():
            X.loc[:, col] = X[col].fillna(X[col].median())
    
    # For A1C (now core feature), use median imputation
    if 'a1c' in X.columns and X['a1c'].isna().any():
        X.loc[:, 'a1c'] = X['a1c'].fillna(X['a1c'].median())
    
    # For optional biomarker features, use median imputation
    biomarker_cols = ['fasting_glucose', 'fasting_insulin']
    for col in biomarker_cols:
        if col in X.columns and X[col].isna().any():
            X.loc[:, col] = X[col].fillna(X[col].median())
    
    # For activity/steps features, use 0 for missing steps data
    activity_cols = ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 'active_minutes']
    for col in activity_cols:
        if col in X.columns and X[col].isna().any():
            X.loc[:, col] = X[col].fillna(0)
    
    # For heart rate, use median imputation
    if 'hr_mean' in X.columns and X['hr_mean'].isna().any():
        X.loc[:, 'hr_mean'] = X['hr_mean'].fillna(X['hr_mean'].median())
    
    return X


def train_and_save_best_models():
    """Train and save the best performing models for each target variable."""
    
    # Load data
    df = load_and_prepare_data()
    
    # Best feature combinations based on model performance results
    best_combinations = {
        'glucose_30min': ['carbohydrates', 'protein', 'fat', 'fiber', 'calories', 
                         'age', 'gender', 'bmi', 'a1c', 'fasting_glucose', 'fasting_insulin'],  # core_plus_biomarkers
        'glucose_60min': ['carbohydrates', 'protein', 'fat', 'fiber', 'calories', 
                         'age', 'gender', 'bmi', 'a1c', 'fasting_glucose', 'fasting_insulin'],  # core_plus_biomarkers
        'glucose_90min': ['carbohydrates', 'protein', 'fat', 'fiber', 'calories', 
                         'age', 'gender', 'bmi', 'a1c', 'fasting_glucose', 'fasting_insulin'],  # core_plus_biomarkers
        'glucose_120min': ['carbohydrates', 'protein', 'fat', 'fiber', 'calories', 
                          'age', 'gender', 'bmi', 'a1c'],  # core_only
        'glucose_180min': ['carbohydrates', 'protein', 'fat', 'fiber', 'calories', 
                          'age', 'gender', 'bmi', 'a1c', 'steps_total', 'steps_mean_per_minute', 
                          'steps_max_per_minute', 'active_minutes', 'hr_mean']  # core_plus_activity
    }
    
    # Create models directory
    models_dir = "glucose_prediction_models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Store model information
    model_info = {}
    feature_sets = {}
    
    for target_var, features in best_combinations.items():
        logger.info(f"\nTraining {target_var} model...")
        logger.info(f"Features: {features}")
        
        # Prepare data
        X = prepare_features(df, features)
        y = df[target_var]
        groups = df['subject_id']
        
        # Remove any remaining NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        groups_clean = groups[valid_mask]
        
        logger.info(f"Training samples: {len(X_clean)}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # Create and train model (RandomForest performed best)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y_clean)
        
        # Calculate performance metrics using cross-validation
        cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y_clean, groups=groups_clean, 
                                   cv=cv, scoring='neg_mean_absolute_error')
        mae_scores = -cv_scores
        
        # Calculate R2 score
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y_clean, y_pred)
        
        logger.info(f"MAE: {mae_scores.mean():.2f} ± {mae_scores.std():.2f} mg/dL")
        logger.info(f"R²: {r2:.3f}")
        
        # Save model and scaler
        model_path = os.path.join(models_dir, f"{target_var}_model.joblib")
        scaler_path = os.path.join(models_dir, f"{target_var}_scaler.joblib")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"Saved model: {model_path}")
        logger.info(f"Saved scaler: {scaler_path}")
        
        # Store metadata
        feature_sets[target_var] = features
        model_info[target_var] = {
            'target': target_var,
            'n_samples': len(X_clean),
            'n_features': len(features),
            'feature_names': features,
            'mae': float(mae_scores.mean()),
            'mae_std': float(mae_scores.std()),
            'r2_score': float(r2),
            'training_date': datetime.now().isoformat()
        }
    
    # Create and save metadata
    metadata = {
        'feature_sets': feature_sets,
        'model_info': model_info,
        'pipeline_version': '2.0.0',  # Updated for A1C-based modeling
        'created_date': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\nSaved metadata: {metadata_path}")
    logger.info("All models saved successfully!")
    
    return metadata


def main():
    """Main execution function."""
    try:
        metadata = train_and_save_best_models()
        
        print("\n" + "="*60)
        print("MODEL TRAINING SUMMARY")
        print("="*60)
        
        for target_var, info in metadata['model_info'].items():
            print(f"\n{target_var}:")
            print(f"  Features: {info['n_features']}")
            print(f"  Training samples: {info['n_samples']}")
            print(f"  MAE: {info['mae']:.2f} ± {info['mae_std']:.2f} mg/dL")
            print(f"  R²: {info['r2_score']:.3f}")
        
        print(f"\nAll models saved to 'glucose_prediction_models/' directory")
        print("Ready for use with Streamlit app!")
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


if __name__ == "__main__":
    main()