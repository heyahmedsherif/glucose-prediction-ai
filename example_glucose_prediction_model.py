#!/usr/bin/env python3
"""
Example Glucose Prediction Model

This script demonstrates how to train models using the glucose prediction database
to predict blood glucose responses at 30, 60, 90, 120, and 180 minutes after meals.

Features used:
- Meal composition: carbohydrates, fiber, protein, fat, calories  
- Demographics: age, gender, BMI
- Optional biomarkers: A1C, fasting glucose, fasting insulin (when available)
- Wearable data: heart rate, activity, METs (when available)

Author: Example implementation for CGMacros glucose prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(csv_file: str = "glucose_prediction_training_data_steps.csv") -> pd.DataFrame:
    """Load and prepare the training data."""
    logger.info(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} records from {df['subject_id'].nunique()} subjects")
    return df


def create_feature_sets(df: pd.DataFrame) -> dict:
    """Create different feature sets for model training."""
    
    # Core features (always available)
    core_features = ['carbohydrates', 'protein', 'fat', 'fiber', 'calories', 
                    'age', 'gender', 'bmi', 'a1c']
    
    # Optional biomarker features  
    biomarker_features = ['fasting_glucose', 'fasting_insulin']
    
    # Optional activity/steps features
    activity_features = ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 
                        'active_minutes', 'hr_mean']
    
    feature_sets = {
        'core_only': core_features,
        'core_plus_biomarkers': core_features + biomarker_features,
        'core_plus_activity': core_features + activity_features,
        'all_features': core_features + biomarker_features + activity_features
    }
    
    return feature_sets


def prepare_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Prepare feature matrix, handling missing values appropriately."""
    X = df[feature_columns].copy()
    
    # Handle missing values
    # For core demographic features, fill with median
    demographic_cols = ['age', 'gender', 'bmi']
    for col in demographic_cols:
        if col in X.columns and X[col].isna().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    # For A1C (now core feature), use median imputation
    if 'a1c' in X.columns and X['a1c'].isna().any():
        X['a1c'].fillna(X['a1c'].median(), inplace=True)
    
    # For optional biomarker features, use median imputation (these are often missing)
    biomarker_cols = ['fasting_glucose', 'fasting_insulin']
    for col in biomarker_cols:
        if col in X.columns and X[col].isna().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    # For activity/steps features, use median imputation (or 0 for steps)
    activity_cols = ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 'active_minutes']
    for col in activity_cols:
        if col in X.columns and X[col].isna().any():
            X[col].fillna(0, inplace=True)  # Use 0 for missing steps data
    
    # For heart rate, use median imputation
    if 'hr_mean' in X.columns and X['hr_mean'].isna().any():
        X['hr_mean'].fillna(X['hr_mean'].median(), inplace=True)
    
    return X


def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series, groups: pd.Series, 
                            model_name: str, feature_set_name: str) -> dict:
    """Train and evaluate a model using group-based cross-validation."""
    
    # Remove any remaining NaN values
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[valid_mask]
    y_clean = y[valid_mask] 
    groups_clean = groups[valid_mask]
    
    if len(X_clean) == 0:
        logger.warning(f"No valid samples for {model_name} with {feature_set_name}")
        return None
    
    logger.info(f"Training {model_name} with {feature_set_name} features on {len(X_clean)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Create model
    if model_name == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        model = xgb.XGBRegressor(random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Group-based cross-validation (leave subjects out)
    cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    
    # Calculate cross-validation scores
    cv_scores = cross_val_score(model, X_scaled, y_clean, groups=groups_clean, 
                               cv=cv, scoring='neg_mean_absolute_error')
    
    mae_scores = -cv_scores
    
    results = {
        'model': model_name,
        'features': feature_set_name,
        'n_samples': len(X_clean),
        'n_features': X_clean.shape[1],
        'mae_mean': mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'feature_names': list(X_clean.columns)
    }
    
    logger.info(f"  MAE: {results['mae_mean']:.2f} ± {results['mae_std']:.2f} mg/dL")
    
    return results


def main():
    """Main execution function."""
    
    # Load data
    df = load_and_prepare_data()
    
    # Create feature sets
    feature_sets = create_feature_sets(df)
    
    # Target variables (glucose at different time points)
    target_variables = ['glucose_30min', 'glucose_60min', 'glucose_90min', 
                       'glucose_120min', 'glucose_180min']
    
    # Models to try
    models = ['RandomForest', 'XGBoost']
    
    # Store all results
    all_results = []
    
    # Train models for each target variable and feature set combination
    for target_var in target_variables:
        logger.info(f"\n{'='*50}")
        logger.info(f"PREDICTING {target_var.upper()}")
        logger.info(f"{'='*50}")
        
        for feature_set_name, feature_columns in feature_sets.items():
            logger.info(f"\nFeature set: {feature_set_name}")
            logger.info(f"Features: {feature_columns}")
            
            # Prepare features
            X = prepare_features(df, feature_columns)
            y = df[target_var]
            groups = df['subject_id']
            
            for model_name in models:
                result = train_and_evaluate_model(X, y, groups, model_name, feature_set_name)
                if result:
                    result['target'] = target_var
                    all_results.append(result)
    
    # Create summary report
    results_df = pd.DataFrame(all_results)
    
    print(f"\n{'='*80}")
    print("GLUCOSE PREDICTION MODEL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    # Best performance by target variable
    for target_var in target_variables:
        target_results = results_df[results_df['target'] == target_var]
        if len(target_results) > 0:
            best_result = target_results.loc[target_results['mae_mean'].idxmin()]
            print(f"\n{target_var}:")
            print(f"  Best Model: {best_result['model']} with {best_result['features']}")
            print(f"  MAE: {best_result['mae_mean']:.2f} ± {best_result['mae_std']:.2f} mg/dL")
            print(f"  Features used: {best_result['n_features']}")
            print(f"  Training samples: {best_result['n_samples']}")
    
    # Save detailed results
    results_df.to_csv('glucose_prediction_model_results.csv', index=False)
    logger.info(f"\nDetailed results saved to glucose_prediction_model_results.csv")
    
    print(f"\n{'='*80}")
    print("DATABASE SUMMARY")
    print(f"{'='*80}")
    print(f"Total meal records: {len(df):,}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print(f"Meal type distribution: {df['meal_type'].value_counts().to_dict()}")
    print(f"Target variable completeness:")
    for target_var in target_variables:
        valid_pct = (df[target_var].notna().sum() / len(df)) * 100
        print(f"  {target_var}: {valid_pct:.1f}% valid")
    
    print(f"\nCore features completeness:")
    core_features = ['carbohydrates', 'protein', 'fat', 'fiber', 'calories', 'age', 'gender', 'bmi', 'a1c']
    for feature in core_features:
        if feature in df.columns:
            valid_pct = (df[feature].notna().sum() / len(df)) * 100
            print(f"  {feature}: {valid_pct:.1f}% valid")
    
    print(f"\nOptional biomarkers completeness:")
    biomarker_features = ['fasting_glucose', 'fasting_insulin']
    for feature in biomarker_features:
        if feature in df.columns:
            valid_pct = (df[feature].notna().sum() / len(df)) * 100
            print(f"  {feature}: {valid_pct:.1f}% valid")
    
    print(f"\nActivity/Steps features completeness:")
    activity_features = ['steps_total', 'steps_mean_per_minute', 'active_minutes', 'hr_mean']
    for feature in activity_features:
        if feature in df.columns:
            valid_pct = (df[feature].notna().sum() / len(df)) * 100
            print(f"  {feature}: {valid_pct:.1f}% valid")


if __name__ == "__main__":
    main()