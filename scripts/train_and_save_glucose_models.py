#!/usr/bin/env python3
"""
Train and Save Glucose Prediction Models

This script trains the best-performing glucose prediction models and saves them
to disk for production deployment. It creates a complete prediction pipeline
with preprocessing and model artifacts.

Author: Production deployment for CGMacros glucose prediction
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlucosePredictionPipeline:
    """Complete glucose prediction pipeline with preprocessing and models."""
    
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.feature_sets = {}
        self.model_info = {}
        
    def define_feature_sets(self) -> Dict[str, Dict[str, list]]:
        """Define the optimal feature sets for each time interval."""
        
        # Core features (always available)
        core_features = ['carbohydrates', 'protein', 'fat', 'fiber', 'calories', 
                        'age', 'gender', 'bmi', 'baseline']
        
        # Activity/steps features
        activity_features = ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 
                           'active_minutes', 'hr_mean']
        
        # Based on our evaluation results, define optimal feature sets per time interval
        optimal_features = {
            'glucose_30min': {
                'features': core_features + activity_features,
                'name': 'core_plus_activity'
            },
            'glucose_60min': {
                'features': core_features,
                'name': 'core_only'  
            },
            'glucose_90min': {
                'features': core_features,
                'name': 'core_only'
            },
            'glucose_120min': {
                'features': core_features,
                'name': 'core_only'
            },
            'glucose_180min': {
                'features': core_features + activity_features,
                'name': 'core_plus_activity'
            }
        }
        
        return optimal_features
    
    def prepare_features(self, df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
        """Prepare feature matrix with proper missing value handling."""
        X = df[feature_columns].copy()
        
        # Handle missing values
        # For demographic features, fill with median
        demographic_cols = ['age', 'gender', 'bmi']
        for col in demographic_cols:
            if col in X.columns and X[col].isna().any():
                X.loc[:, col] = X[col].fillna(X[col].median())
        
        # For activity/steps features, use 0 for missing data
        activity_cols = ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 'active_minutes']
        for col in activity_cols:
            if col in X.columns and X[col].isna().any():
                X.loc[:, col] = X[col].fillna(0)
        
        # For heart rate, use median imputation
        if 'hr_mean' in X.columns and X['hr_mean'].isna().any():
            X.loc[:, col] = X['hr_mean'].fillna(X['hr_mean'].median())
        
        return X
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, target_var: str) -> Tuple[RandomForestRegressor, StandardScaler, dict]:
        """Train a single RandomForest model with preprocessing."""
        
        # Remove any remaining NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) == 0:
            raise ValueError(f"No valid samples for {target_var}")
        
        logger.info(f"Training {target_var} model on {len(X_clean)} samples with {X_clean.shape[1]} features")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train RandomForest model (best performing algorithm)
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_info = {
            'target': target_var,
            'n_samples': len(X_clean),
            'n_features': X_clean.shape[1],
            'feature_names': list(X_clean.columns),
            'mae': float(mae),
            'r2_score': float(r2),
            'training_date': datetime.now().isoformat()
        }
        
        logger.info(f"  {target_var} - MAE: {mae:.2f} mg/dL, R²: {r2:.3f}")
        
        return model, scaler, model_info
    
    def train_all_models(self, data_file: str = "glucose_prediction_training_data_steps.csv"):
        """Train all glucose prediction models."""
        
        logger.info("Loading training data...")
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} records from {df['subject_id'].nunique()} subjects")
        
        # Define optimal feature sets
        optimal_features = self.define_feature_sets()
        
        # Train models for each time interval
        target_variables = ['glucose_30min', 'glucose_60min', 'glucose_90min', 
                           'glucose_120min', 'glucose_180min']
        
        for target_var in target_variables:
            logger.info(f"\nTraining model for {target_var}...")
            
            # Get optimal features for this target
            feature_config = optimal_features[target_var]
            feature_columns = feature_config['features']
            
            # Prepare features
            X = self.prepare_features(df, feature_columns)
            y = df[target_var]
            
            # Train model
            model, scaler, model_info = self.train_model(X, y, target_var)
            
            # Store components
            self.models[target_var] = model
            self.scalers[target_var] = scaler  
            self.feature_sets[target_var] = feature_columns
            self.model_info[target_var] = model_info
        
        logger.info(f"\nSuccessfully trained {len(self.models)} models")
    
    def predict(self, input_features: Dict[str, Any]) -> Dict[str, float]:
        """Make glucose predictions for all time intervals."""
        
        predictions = {}
        
        for target_var in self.models.keys():
            # Get required features for this model
            required_features = self.feature_sets[target_var]
            
            # Extract and prepare features
            feature_values = []
            for feature in required_features:
                if feature in input_features:
                    feature_values.append(input_features[feature])
                else:
                    # Handle missing features with defaults
                    if feature in ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 'active_minutes']:
                        feature_values.append(0)  # Default for activity features
                    elif feature == 'hr_mean':
                        feature_values.append(75)  # Default resting heart rate
                    else:
                        raise ValueError(f"Required feature '{feature}' missing for {target_var} prediction")
            
            # Convert to numpy array and reshape
            X = np.array(feature_values).reshape(1, -1)
            
            # Scale features
            X_scaled = self.scalers[target_var].transform(X)
            
            # Make prediction
            pred = self.models[target_var].predict(X_scaled)[0]
            predictions[target_var] = float(pred)
        
        return predictions
    
    def save_models(self, output_dir: str = "glucose_prediction_models"):
        """Save all models and components to disk."""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving models to {output_dir}...")
        
        # Save individual model components
        for target_var in self.models.keys():
            # Save model
            model_file = os.path.join(output_dir, f"{target_var}_model.joblib")
            joblib.dump(self.models[target_var], model_file)
            
            # Save scaler
            scaler_file = os.path.join(output_dir, f"{target_var}_scaler.joblib")
            joblib.dump(self.scalers[target_var], scaler_file)
        
        # Save feature sets and model info
        metadata = {
            'feature_sets': self.feature_sets,
            'model_info': self.model_info,
            'pipeline_version': '1.0.0',
            'created_date': datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(output_dir, "model_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save complete pipeline object
        pipeline_file = os.path.join(output_dir, "glucose_prediction_pipeline.pkl")
        with open(pipeline_file, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info("Models saved successfully!")
        logger.info(f"Files created:")
        for target_var in self.models.keys():
            logger.info(f"  - {target_var}_model.joblib")
            logger.info(f"  - {target_var}_scaler.joblib")
        logger.info(f"  - model_metadata.json")
        logger.info(f"  - glucose_prediction_pipeline.pkl")
    
    def load_models(self, model_dir: str = "glucose_prediction_models"):
        """Load saved models from disk."""
        
        logger.info(f"Loading models from {model_dir}...")
        
        # Load metadata
        metadata_file = os.path.join(model_dir, "model_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.feature_sets = metadata['feature_sets']
        self.model_info = metadata['model_info']
        
        # Load models and scalers
        for target_var in self.feature_sets.keys():
            # Load model
            model_file = os.path.join(model_dir, f"{target_var}_model.joblib")
            self.models[target_var] = joblib.load(model_file)
            
            # Load scaler  
            scaler_file = os.path.join(model_dir, f"{target_var}_scaler.joblib")
            self.scalers[target_var] = joblib.load(scaler_file)
        
        logger.info(f"Loaded {len(self.models)} models successfully!")


def create_example_prediction_script(output_dir: str = "glucose_prediction_models"):
    """Create an example script showing how to use the saved models."""
    
    example_script = '''#!/usr/bin/env python3
"""
Example: Using Saved Glucose Prediction Models

This script demonstrates how to load and use the trained glucose prediction models
for making real-time predictions.
"""

import pickle
import numpy as np
from train_and_save_glucose_models import GlucosePredictionPipeline

def main():
    # Load the trained pipeline
    pipeline = GlucosePredictionPipeline()
    pipeline.load_models("glucose_prediction_models")
    
    # Example input: A meal with user characteristics
    example_input = {
        # Meal composition
        'carbohydrates': 50.0,      # grams
        'protein': 25.0,            # grams  
        'fat': 15.0,                # grams
        'fiber': 5.0,               # grams
        'calories': 400.0,          # total calories
        
        # Demographics
        'age': 35.0,                # years
        'gender': 0.0,              # 0=Female, 1=Male
        'bmi': 24.5,                # kg/m²
        'baseline': 95.0,           # mg/dL (current glucose)
        
        # Activity (optional - will use defaults if missing)
        'steps_total': 150,         # steps in 30-min window
        'steps_mean_per_minute': 5.0,   # average steps/min
        'steps_max_per_minute': 25.0,   # peak steps/min
        'active_minutes': 8,            # minutes with activity
        'hr_mean': 78.0             # average heart rate
    }
    
    # Make predictions
    predictions = pipeline.predict(example_input)
    
    # Display results
    print("\\n" + "="*50)
    print("GLUCOSE PREDICTION RESULTS")
    print("="*50)
    print(f"Baseline glucose: {example_input['baseline']:.1f} mg/dL")
    print(f"Meal: {example_input['calories']:.0f} cal, {example_input['carbohydrates']:.0f}g carbs")
    print("")
    print("Predicted glucose levels:")
    for time_point, glucose in predictions.items():
        minutes = time_point.replace('glucose_', '').replace('min', '')
        print(f"  {minutes:>3} minutes: {glucose:.1f} mg/dL")
    
    # Show model performance info
    print("\\n" + "="*50)
    print("MODEL PERFORMANCE (from training)")
    print("="*50)
    for target_var, info in pipeline.model_info.items():
        minutes = target_var.replace('glucose_', '').replace('min', '')
        print(f"{minutes:>3} min model: MAE = {info['mae']:.1f} mg/dL, R² = {info['r2_score']:.3f}")

if __name__ == "__main__":
    main()
'''
    
    example_file = os.path.join(output_dir, "example_prediction.py")
    with open(example_file, 'w') as f:
        f.write(example_script)
    
    logger.info(f"Created example script: {example_file}")


def main():
    """Main execution function."""
    
    # Initialize pipeline
    pipeline = GlucosePredictionPipeline()
    
    # Train all models
    pipeline.train_all_models()
    
    # Save models to disk
    output_dir = "glucose_prediction_models"
    pipeline.save_models(output_dir)
    
    # Create example prediction script
    create_example_prediction_script(output_dir)
    
    # Test loading and prediction
    logger.info("\nTesting model loading and prediction...")
    
    # Create test input
    test_input = {
        'carbohydrates': 60.0, 'protein': 20.0, 'fat': 12.0, 'fiber': 3.0, 'calories': 420.0,
        'age': 45.0, 'gender': 1.0, 'bmi': 26.8, 'baseline': 88.5,
        'steps_total': 250, 'steps_mean_per_minute': 8.3, 'steps_max_per_minute': 35.0,
        'active_minutes': 12, 'hr_mean': 82.0
    }
    
    # Make test prediction
    predictions = pipeline.predict(test_input)
    
    print(f"\n{'='*60}")
    print("TEST PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Input: {test_input['calories']:.0f} cal, {test_input['carbohydrates']:.0f}g carbs")
    print(f"Baseline glucose: {test_input['baseline']:.1f} mg/dL")
    print("")
    print("Predicted glucose responses:")
    for time_point, glucose in predictions.items():
        minutes = time_point.replace('glucose_', '').replace('min', '')
        print(f"  {minutes:>3} minutes: {glucose:.1f} mg/dL")
    
    print(f"\n{'='*60}")
    print("DEPLOYMENT READY!")
    print(f"{'='*60}")
    print(f"✅ Models trained and saved to: {output_dir}/")
    print(f"✅ Example usage script: {output_dir}/example_prediction.py")
    print(f"✅ Pipeline ready for production deployment")


if __name__ == "__main__":
    main()