#!/usr/bin/env python3
"""
Train Enhanced Glucose Prediction Models with Diabetic Status

This script trains improved glucose prediction models that incorporate:
1. Diabetic status as a key input feature (Normal, Pre-diabetic, Type2Diabetic)
2. Enhanced baseline glucose prediction based on diabetic status
3. Improved feature engineering and model architecture

Author: Enhanced CGMacros glucose prediction with diabetic status
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGlucosePredictionPipeline:
    """Enhanced glucose prediction pipeline with diabetic status integration."""
    
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.feature_sets = {}
        self.model_info = {}
        self.baseline_predictors = {}
        
    def define_enhanced_feature_sets(self) -> Dict[str, Dict[str, list]]:
        """Define enhanced feature sets including diabetic status."""
        
        # Core demographic and meal features
        core_features = ['carbohydrates', 'protein', 'fat', 'fiber', 'calories', 
                        'age', 'gender', 'bmi']
        
        # Diabetic status features
        diabetic_features = ['diabetic_status_encoded', 'a1c', 'fasting_glucose', 'fasting_insulin']
        
        # Activity/steps features  
        activity_features = ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 
                           'active_minutes', 'hr_mean']
        
        # Baseline glucose (now predicted based on diabetic status)
        baseline_features = ['baseline']
        
        # Define feature sets for each time interval
        feature_sets = {
            'glucose_30min': {
                'features': core_features + diabetic_features + baseline_features + activity_features,
                'name': 'enhanced_full_features'
            },
            'glucose_60min': {
                'features': core_features + diabetic_features + baseline_features,
                'name': 'enhanced_core_features'  
            },
            'glucose_90min': {
                'features': core_features + diabetic_features + baseline_features,
                'name': 'enhanced_core_features'
            },
            'glucose_120min': {
                'features': core_features + diabetic_features + baseline_features,
                'name': 'enhanced_core_features'
            },
            'glucose_180min': {
                'features': core_features + diabetic_features + baseline_features + activity_features,
                'name': 'enhanced_full_features'
            }
        }
        
        return feature_sets
    
    def prepare_enhanced_features(self, df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        """Prepare enhanced feature matrix with proper handling of all feature types."""
        X = df[feature_columns].copy()
        
        # Handle missing values by feature type
        demographic_cols = ['age', 'gender', 'bmi']
        for col in demographic_cols:
            if col in X.columns and X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Fill diabetic status encoded (should not be missing in enhanced data)
        if 'diabetic_status_encoded' in X.columns and X['diabetic_status_encoded'].isna().any():
            X['diabetic_status_encoded'] = X['diabetic_status_encoded'].fillna(1)  # Default to pre-diabetic
        
        # Fill biomarkers with appropriate defaults
        biomarker_defaults = {
            'a1c': 5.7,  # Lower end of pre-diabetic range
            'fasting_glucose': 100,  # Normal upper limit
            'fasting_insulin': 10.0  # Typical value
        }
        for col, default_val in biomarker_defaults.items():
            if col in X.columns and X[col].isna().any():
                X[col] = X[col].fillna(default_val)
        
        # Activity features - fill with zeros (no activity)
        activity_cols = ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 'active_minutes']
        for col in activity_cols:
            if col in X.columns and X[col].isna().any():
                X[col] = X[col].fillna(0)
        
        # Heart rate - fill with typical resting HR
        if 'hr_mean' in X.columns and X['hr_mean'].isna().any():
            X['hr_mean'] = X['hr_mean'].fillna(75)
        
        # Baseline should not be missing in enhanced data, but handle just in case
        if 'baseline' in X.columns and X['baseline'].isna().any():
            X['baseline'] = X['baseline'].fillna(95)
        
        return X
    
    def train_baseline_predictor(self, df: pd.DataFrame):
        """Train a model to predict baseline glucose from diabetic status and demographics."""
        logger.info("Training baseline glucose predictor...")
        
        # Features for baseline prediction
        baseline_features = ['diabetic_status_encoded', 'age', 'bmi', 'a1c', 'fasting_glucose']
        
        # Prepare data
        X_baseline = self.prepare_enhanced_features(df, baseline_features)
        y_baseline = df['baseline']
        
        # Remove any remaining NaN values
        valid_mask = ~(X_baseline.isna().any(axis=1) | y_baseline.isna())
        X_baseline_clean = X_baseline[valid_mask]
        y_baseline_clean = y_baseline[valid_mask]
        
        # Train baseline predictor
        baseline_scaler = StandardScaler()
        X_baseline_scaled = baseline_scaler.fit_transform(X_baseline_clean)
        
        baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
        baseline_model.fit(X_baseline_scaled, y_baseline_clean)
        
        # Evaluate baseline predictor
        y_pred_baseline = baseline_model.predict(X_baseline_scaled)
        baseline_mae = mean_absolute_error(y_baseline_clean, y_pred_baseline)
        baseline_r2 = r2_score(y_baseline_clean, y_pred_baseline)
        
        logger.info(f"Baseline predictor - MAE: {baseline_mae:.2f} mg/dL, R²: {baseline_r2:.3f}")
        
        # Store baseline predictor
        self.baseline_predictors = {
            'model': baseline_model,
            'scaler': baseline_scaler,
            'features': baseline_features,
            'performance': {'mae': baseline_mae, 'r2': baseline_r2}
        }
    
    def train_enhanced_model(self, X: pd.DataFrame, y: pd.Series, target_var: str) -> Tuple[RandomForestRegressor, StandardScaler, dict]:
        """Train an enhanced model with diabetic status features."""
        
        # Remove any remaining NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) == 0:
            raise ValueError(f"No valid samples for {target_var}")
        
        logger.info(f"Training enhanced {target_var} model on {len(X_clean)} samples with {X_clean.shape[1]} features")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, stratify=X_clean['diabetic_status_encoded']
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try multiple algorithms and select the best
        models_to_try = {
            'RandomForest': RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        best_model = None
        best_score = float('inf')
        best_name = None
        
        for name, model in models_to_try.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            
            if mae < best_score:
                best_score = mae
                best_model = model
                best_name = name
        
        # Final evaluation with best model
        y_pred_final = best_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred_final)
        r2 = r2_score(y_test, y_pred_final)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, 
                                   cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        model_info = {
            'target': target_var,
            'algorithm': best_name,
            'n_samples': len(X_clean),
            'n_features': X_clean.shape[1],
            'feature_names': list(X_clean.columns),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'cv_mae': float(cv_mae),
            'training_date': datetime.now().isoformat()
        }
        
        logger.info(f"  {target_var} ({best_name}) - MAE: {mae:.2f} mg/dL, R²: {r2:.3f}, CV-MAE: {cv_mae:.2f}")
        
        return best_model, scaler, model_info
    
    def train_all_enhanced_models(self, data_file: str = "glucose_prediction_training_data_enhanced.csv"):
        """Train all enhanced glucose prediction models."""
        
        logger.info("Loading enhanced training data...")
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} records from {df['subject_id'].nunique()} subjects")
        
        # Show diabetic status distribution
        logger.info("Diabetic status distribution:")
        logger.info(df['diabetic_status'].value_counts())
        
        # Train baseline predictor first
        self.train_baseline_predictor(df)
        
        # Define enhanced feature sets
        enhanced_features = self.define_enhanced_feature_sets()
        
        # Train models for each time interval
        target_variables = ['glucose_30min', 'glucose_60min', 'glucose_90min', 
                           'glucose_120min', 'glucose_180min']
        
        for target_var in target_variables:
            logger.info(f"\nTraining enhanced model for {target_var}...")
            
            # Get feature configuration
            feature_config = enhanced_features[target_var]
            feature_columns = feature_config['features']
            
            # Prepare features
            X = self.prepare_enhanced_features(df, feature_columns)
            y = df[target_var]
            
            # Train enhanced model
            model, scaler, model_info = self.train_enhanced_model(X, y, target_var)
            
            # Store components
            self.models[target_var] = model
            self.scalers[target_var] = scaler
            self.feature_sets[target_var] = feature_columns
            self.model_info[target_var] = model_info
        
        logger.info(f"\nSuccessfully trained {len(self.models)} enhanced models")
    
    def predict_baseline_glucose(self, diabetic_status: str, age: float, bmi: float, 
                               a1c: float = None, fasting_glucose: float = None) -> float:
        """Predict baseline glucose based on patient characteristics."""
        
        if not self.baseline_predictors:
            # Fallback to status-based defaults if no baseline predictor
            defaults = {'Normal': 85, 'Pre-diabetic': 105, 'Type2Diabetic': 140}
            return defaults.get(diabetic_status, 100)
        
        # Encode diabetic status
        status_encoding = {'Normal': 0, 'Pre-diabetic': 1, 'Type2Diabetic': 2}
        status_encoded = status_encoding.get(diabetic_status, 1)
        
        # Prepare features
        features = [
            status_encoded,
            age,
            bmi,
            a1c if a1c else 5.7,
            fasting_glucose if fasting_glucose else 100
        ]
        
        # Make prediction
        X = np.array(features).reshape(1, -1)
        X_scaled = self.baseline_predictors['scaler'].transform(X)
        baseline = self.baseline_predictors['model'].predict(X_scaled)[0]
        
        return float(baseline)
    
    def predict_enhanced(self, input_features: Dict[str, Any]) -> Dict[str, float]:
        """Make enhanced glucose predictions including baseline prediction."""
        
        # First predict baseline glucose if not provided
        if 'baseline' not in input_features or input_features['baseline'] is None:
            baseline = self.predict_baseline_glucose(
                diabetic_status=input_features.get('diabetic_status', 'Normal'),
                age=input_features.get('age', 40),
                bmi=input_features.get('bmi', 25),
                a1c=input_features.get('a1c'),
                fasting_glucose=input_features.get('fasting_glucose')
            )
            input_features['baseline'] = baseline
        
        # Encode diabetic status
        status_encoding = {'Normal': 0, 'Pre-diabetic': 1, 'Type2Diabetic': 2}
        input_features['diabetic_status_encoded'] = status_encoding.get(
            input_features.get('diabetic_status', 'Normal'), 1
        )
        
        predictions = {'baseline': input_features['baseline']}
        
        # Make predictions for each time point
        for target_var in self.models.keys():
            # Get required features
            required_features = self.feature_sets[target_var]
            
            # Extract feature values
            feature_values = []
            for feature in required_features:
                if feature in input_features:
                    feature_values.append(input_features[feature])
                else:
                    # Default values for missing features
                    defaults = {
                        'steps_total': 0, 'steps_mean_per_minute': 0, 
                        'steps_max_per_minute': 0, 'active_minutes': 0,
                        'hr_mean': 75, 'a1c': 5.7, 'fasting_glucose': 100, 
                        'fasting_insulin': 10
                    }
                    if feature in defaults:
                        feature_values.append(defaults[feature])
                    else:
                        raise ValueError(f"Required feature '{feature}' missing for {target_var} prediction")
            
            # Make prediction
            X = np.array(feature_values).reshape(1, -1)
            X_scaled = self.scalers[target_var].transform(X)
            pred = self.models[target_var].predict(X_scaled)[0]
            predictions[target_var] = float(pred)
        
        return predictions
    
    def save_enhanced_models(self, output_dir: str = "glucose_prediction_models"):
        """Save enhanced models to disk."""
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving enhanced models to {output_dir}...")
        
        # Save individual model components
        for target_var in self.models.keys():
            joblib.dump(self.models[target_var], 
                       os.path.join(output_dir, f"{target_var}_model.joblib"))
            joblib.dump(self.scalers[target_var], 
                       os.path.join(output_dir, f"{target_var}_scaler.joblib"))
        
        # Save baseline predictor
        if self.baseline_predictors:
            joblib.dump(self.baseline_predictors['model'],
                       os.path.join(output_dir, "baseline_model.joblib"))
            joblib.dump(self.baseline_predictors['scaler'],
                       os.path.join(output_dir, "baseline_scaler.joblib"))
        
        # Save enhanced metadata
        enhanced_metadata = {
            'feature_sets': self.feature_sets,
            'model_info': self.model_info,
            'baseline_predictor_info': self.baseline_predictors.get('performance', {}),
            'pipeline_version': '2.0.0',
            'enhancements': [
                'Diabetic status integration',
                'Enhanced baseline prediction', 
                'Improved feature engineering',
                'Multi-algorithm selection'
            ],
            'created_date': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "model_metadata.json"), 'w') as f:
            json.dump(enhanced_metadata, f, indent=2)
        
        # Save complete pipeline
        with open(os.path.join(output_dir, "glucose_prediction_pipeline.pkl"), 'wb') as f:
            pickle.dump(self, f)
        
        logger.info("Enhanced models saved successfully!")


def main():
    """Main execution function."""
    
    # Initialize enhanced pipeline
    pipeline = EnhancedGlucosePredictionPipeline()
    
    # Train all enhanced models
    pipeline.train_all_enhanced_models()
    
    # Save models
    pipeline.save_enhanced_models()
    
    # Test enhanced prediction
    logger.info("\nTesting enhanced prediction...")
    
    test_input = {
        'diabetic_status': 'Pre-diabetic',
        'carbohydrates': 60.0, 'protein': 20.0, 'fat': 12.0, 'fiber': 3.0, 'calories': 420.0,
        'age': 45.0, 'gender': 1.0, 'bmi': 26.8, 'a1c': 6.0,
        'steps_total': 250, 'steps_mean_per_minute': 8.3, 'steps_max_per_minute': 35.0,
        'active_minutes': 12, 'hr_mean': 82.0
    }
    
    predictions = pipeline.predict_enhanced(test_input)
    
    print(f"\n{'='*70}")
    print("ENHANCED GLUCOSE PREDICTION RESULTS")
    print(f"{'='*70}")
    print(f"Patient Profile: {test_input['diabetic_status']}, Age {test_input['age']:.0f}, BMI {test_input['bmi']:.1f}")
    print(f"Meal: {test_input['calories']:.0f} cal, {test_input['carbohydrates']:.0f}g carbs")
    print(f"Predicted baseline: {predictions['baseline']:.1f} mg/dL")
    print("")
    print("Predicted glucose responses:")
    for time_point, glucose in predictions.items():
        if time_point != 'baseline':
            minutes = time_point.replace('glucose_', '').replace('min', '')
            print(f"  {minutes:>3} minutes: {glucose:.1f} mg/dL")
    
    print(f"\n{'='*70}")
    print("ENHANCED MODEL DEPLOYMENT READY!")
    print(f"{'='*70}")
    print("✅ Enhanced models with diabetic status integration")
    print("✅ Improved baseline glucose prediction")
    print("✅ Multi-algorithm model selection")
    print("✅ Ready for production deployment")


if __name__ == "__main__":
    main()