#!/usr/bin/env python3
"""
Glucose Prediction Database Builder

This script creates a comprehensive training database for glucose prediction models.
It extracts glucose responses at 30, 60, 90, 120, and 180 minutes after meals,
along with meal composition, demographic, and wearable device features.

Author: Generated for CGMacros glucose prediction modeling
"""

import os
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_biomarker_data(bio_csv_path: str = "CGMacros/bio.csv") -> pd.DataFrame:
    """
    Load and process biomarker data from CSV file.
    
    Args:
        bio_csv_path: Path to biomarker CSV file
        
    Returns:
        pd.DataFrame: Processed biomarker data indexed by subject
    """
    logger.info("Loading biomarker data...")
    
    if not os.path.exists(bio_csv_path):
        raise FileNotFoundError(f"Biomarker file not found: {bio_csv_path}")
    
    df = pd.read_csv(bio_csv_path)
    
    # Process demographic and biomarker data
    processed_data = {}
    
    # Basic demographics
    processed_data['Age'] = df["Age"].to_numpy()
    processed_data['Gender'] = np.array([1 if x == 'M' else 0 for x in df["Gender"].tolist()])
    
    # Anthropometric data
    weights_lbs = df["Body weight "].dropna().to_numpy()
    heights_inches = df["Height "].dropna().to_numpy()
    
    # Convert to metric units
    weight_kg = weights_lbs * 0.453592
    height_m = heights_inches * 0.0254
    bmi = weight_kg / (height_m ** 2)
    processed_data['BMI'] = bmi
    
    # Biomarkers (optional - may have missing values)
    processed_data['A1C'] = df["A1c PDL (Lab)"].to_numpy()
    processed_data['Fasting_Glucose'] = df["Fasting GLU - PDL (Lab)"].to_numpy()
    
    # Process fasting insulin (handle special formatting)
    fasting_insulin_raw = df["Insulin "].to_numpy()
    fasting_insulin = []
    for x in fasting_insulin_raw:
        if pd.isna(x):
            fasting_insulin.append(np.nan)
        else:
            # Handle "(low)" suffix
            clean_value = str(x).strip(' (low)')
            try:
                fasting_insulin.append(float(clean_value))
            except (ValueError, TypeError):
                fasting_insulin.append(np.nan)
    processed_data['Fasting_Insulin'] = np.array(fasting_insulin)
    
    # Create DataFrame indexed by subject number (001, 002, etc.)
    subject_ids = [f"{i+1:03d}" for i in range(len(processed_data['Age']))]
    
    bio_df = pd.DataFrame(processed_data, index=subject_ids)
    
    logger.info(f"Loaded biomarker data for {len(bio_df)} subjects")
    return bio_df


def extract_glucose_response_at_intervals(glucose_series: pd.Series, timestamp_series: pd.Series, 
                                        meal_timestamp: pd.Timestamp, 
                                        intervals: List[int] = [30, 60, 90, 120, 180]) -> Dict[str, float]:
    """
    Extract glucose values at specific time intervals after a meal.
    
    Args:
        glucose_series: Series of glucose values
        timestamp_series: Series of timestamps
        meal_timestamp: Timestamp when meal occurred
        intervals: List of intervals in minutes after meal
        
    Returns:
        Dict with glucose values at each interval
    """
    glucose_responses = {}
    
    # Extract glucose at each interval
    for interval in intervals:
        target_time = meal_timestamp + timedelta(minutes=interval)
        target_idx = timestamp_series.searchsorted(target_time)
        
        if target_idx < len(glucose_series):
            glucose_responses[f'glucose_{interval}min'] = glucose_series.iloc[target_idx]
        else:
            glucose_responses[f'glucose_{interval}min'] = np.nan
    
    return glucose_responses


def extract_steps_features_around_meal(data: pd.DataFrame, timestamp_series: pd.Series,
                                     meal_timestamp: pd.Timestamp, 
                                     window_minutes: int = 30) -> Dict[str, float]:
    """
    Extract steps-based activity features around meal time.
    
    Args:
        data: DataFrame with wearable data columns
        timestamp_series: Series of timestamps
        meal_timestamp: Timestamp when meal occurred
        window_minutes: Minutes before/after meal to consider
        
    Returns:
        Dict with steps-based activity features
    """
    # Define time window around meal
    start_time = meal_timestamp - timedelta(minutes=window_minutes)
    end_time = meal_timestamp + timedelta(minutes=window_minutes)
    
    # Find indices within window
    start_idx = timestamp_series.searchsorted(start_time)
    end_idx = timestamp_series.searchsorted(end_time)
    
    # Extract features
    features = {}
    
    if start_idx < end_idx and end_idx <= len(data):
        window_data = data.iloc[start_idx:end_idx]
        
        # Try to get steps data
        steps_data = None
        
        # Option 1: Direct Steps column
        if 'Steps' in window_data.columns:
            steps_data = window_data['Steps'].dropna()
        
        # Option 2: Estimate steps from activity calories and METs
        elif 'Calories (Activity)' in window_data.columns and 'METs' in window_data.columns:
            # Rough estimation: 1 calorie ≈ 20 steps for average person
            # This is approximate but gives reasonable step estimates
            activity_cals = window_data['Calories (Activity)'].dropna()
            if len(activity_cals) > 0:
                estimated_steps = activity_cals * 20  # Approximation
                steps_data = estimated_steps
        
        # Calculate steps features
        if steps_data is not None and len(steps_data) > 0:
            features['steps_total'] = steps_data.sum()
            features['steps_mean_per_minute'] = steps_data.mean() 
            features['steps_max_per_minute'] = steps_data.max()
            features['active_minutes'] = (steps_data > 0).sum()  # Minutes with any steps
        else:
            # Fill with zeros if no activity data available
            features['steps_total'] = 0
            features['steps_mean_per_minute'] = 0
            features['steps_max_per_minute'] = 0
            features['active_minutes'] = 0
        
        # Basic heart rate if available (simplified)
        if 'HR' in window_data.columns:
            hr_data = window_data['HR'].dropna()
            if len(hr_data) > 0:
                features['hr_mean'] = hr_data.mean()
            else:
                features['hr_mean'] = np.nan
        else:
            features['hr_mean'] = np.nan
    
    else:
        # Fill with default values if window is invalid
        features['steps_total'] = 0
        features['steps_mean_per_minute'] = 0
        features['steps_max_per_minute'] = 0
        features['active_minutes'] = 0
        features['hr_mean'] = np.nan
    
    return features


def process_single_subject(subject_dir: str, subject_id: str, bio_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process a single subject's data to extract meal-based training records.
    
    Args:
        subject_dir: Path to subject directory
        subject_id: Subject identifier (e.g., "001")
        bio_data: Biomarker dataframe
        
    Returns:
        DataFrame with training records for this subject
    """
    csv_path = os.path.join(subject_dir, f"CGMacros-{subject_id}.csv")
    
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found for subject {subject_id}: {csv_path}")
        return pd.DataFrame()
    
    try:
        # Load subject data
        data = pd.read_csv(csv_path)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        
        # Find all meals (Breakfast, Lunch, Dinner)
        meal_types = ['Breakfast', 'breakfast', 'Lunch', 'lunch', 'Dinner', 'dinner']
        meal_mask = data['Meal Type'].isin(meal_types)
        meal_indices = data[meal_mask].index
        
        subject_records = []
        
        for meal_idx in meal_indices:
            meal_row = data.iloc[meal_idx]
            meal_timestamp = meal_row['Timestamp']
            
            # Skip if missing essential meal data
            if pd.isna(meal_row['Calories']) or pd.isna(meal_row['Carbs']):
                continue
            
            # Extract meal features
            meal_features = {
                'subject_id': subject_id,
                'meal_type': meal_row['Meal Type'].lower(),
                'meal_timestamp': meal_timestamp,
                'calories': meal_row['Calories'],
                'carbohydrates': meal_row['Carbs'],
                'protein': meal_row['Protein'],
                'fat': meal_row['Fat'], 
                'fiber': meal_row['Fiber'],
                'amount_consumed': meal_row['Amount Consumed']
            }
            
            # Extract glucose responses at target intervals
            glucose_responses = extract_glucose_response_at_intervals(
                data['Libre GL'], data['Timestamp'], meal_timestamp
            )
            meal_features.update(glucose_responses)
            
            # Extract steps-based activity features around meal time
            activity_features = extract_steps_features_around_meal(
                data, data['Timestamp'], meal_timestamp
            )
            meal_features.update(activity_features)
            
            # Add demographic and biomarker data
            if subject_id in bio_data.index:
                subject_bio = bio_data.loc[subject_id]
                meal_features.update({
                    'age': subject_bio['Age'],
                    'gender': subject_bio['Gender'],  # 1=Male, 0=Female
                    'bmi': subject_bio['BMI'],
                    'a1c': subject_bio['A1C'],
                    'fasting_glucose': subject_bio['Fasting_Glucose'],
                    'fasting_insulin': subject_bio['Fasting_Insulin']
                })
            else:
                logger.warning(f"No biomarker data found for subject {subject_id}")
                # Fill with NaN
                for col in ['age', 'gender', 'bmi', 'a1c', 'fasting_glucose', 'fasting_insulin']:
                    meal_features[col] = np.nan
            
            subject_records.append(meal_features)
        
        if subject_records:
            logger.info(f"Processed {len(subject_records)} meals for subject {subject_id}")
            return pd.DataFrame(subject_records)
        else:
            logger.warning(f"No valid meals found for subject {subject_id}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error processing subject {subject_id}: {e}")
        return pd.DataFrame()


def build_training_database(data_directory: str = "CGMacros", output_file: str = "glucose_prediction_training_data.csv") -> pd.DataFrame:
    """
    Build comprehensive training database from all subjects.
    
    Args:
        data_directory: Directory containing CGMacros subject data
        output_file: Output CSV filename
        
    Returns:
        DataFrame with complete training database
    """
    logger.info("Building glucose prediction training database...")
    
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory not found: {data_directory}")
    
    # Load biomarker data
    bio_data = load_biomarker_data(os.path.join(data_directory, "bio.csv"))
    
    # Process all subjects
    all_records = []
    
    for item in sorted(os.listdir(data_directory)):
        if item.startswith("CGMacros-") and os.path.isdir(os.path.join(data_directory, item)):
            subject_id = item.split("-")[1]  # Extract "001" from "CGMacros-001"
            subject_dir = os.path.join(data_directory, item)
            
            logger.info(f"Processing subject {subject_id}...")
            subject_df = process_single_subject(subject_dir, subject_id, bio_data)
            
            if not subject_df.empty:
                all_records.append(subject_df)
    
    if not all_records:
        raise ValueError("No valid subject data found")
    
    # Combine all records
    training_db = pd.concat(all_records, ignore_index=True)
    
    # Save to CSV
    training_db.to_csv(output_file, index=False)
    logger.info(f"Training database saved to {output_file}")
    logger.info(f"Total records: {len(training_db)}")
    logger.info(f"Subjects: {training_db['subject_id'].nunique()}")
    logger.info(f"Meal types: {training_db['meal_type'].value_counts().to_dict()}")
    
    # Print data quality summary
    logger.info("Data quality summary:")
    for col in ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']:
        missing_pct = (training_db[col].isna().sum() / len(training_db)) * 100
        logger.info(f"  {col}: {missing_pct:.1f}% missing")
    
    return training_db


def validate_training_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform data quality validation on the training database.
    
    Args:
        df: Training database DataFrame
        
    Returns:
        Dict with validation results
    """
    validation_results = {}
    
    # Basic statistics
    validation_results['total_records'] = len(df)
    validation_results['subjects'] = df['subject_id'].nunique()
    validation_results['meal_types'] = df['meal_type'].value_counts().to_dict()
    
    # Target variable completeness
    target_vars = ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']
    validation_results['target_completeness'] = {}
    for var in target_vars:
        valid_count = df[var].notna().sum()
        validation_results['target_completeness'][var] = {
            'valid_records': valid_count,
            'missing_pct': ((len(df) - valid_count) / len(df)) * 100
        }
    
    # Feature completeness  
    feature_vars = ['carbohydrates', 'protein', 'fat', 'fiber', 'calories', 'age', 'gender', 'bmi']
    validation_results['feature_completeness'] = {}
    for var in feature_vars:
        valid_count = df[var].notna().sum()
        validation_results['feature_completeness'][var] = {
            'valid_records': valid_count,
            'missing_pct': ((len(df) - valid_count) / len(df)) * 100
        }
    
    # Optional biomarker completeness
    optional_vars = ['a1c', 'fasting_glucose', 'fasting_insulin']
    validation_results['optional_completeness'] = {}
    for var in optional_vars:
        valid_count = df[var].notna().sum()
        validation_results['optional_completeness'][var] = {
            'valid_records': valid_count,
            'missing_pct': ((len(df) - valid_count) / len(df)) * 100
        }
    
    # Value ranges
    validation_results['value_ranges'] = {}
    for var in ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']:
        if var in df.columns:
            validation_results['value_ranges'][var] = {
                'min': df[var].min(),
                'max': df[var].max(),
                'mean': df[var].mean(),
                'std': df[var].std()
            }
    
    return validation_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Build Glucose Prediction Training Database')
    parser.add_argument('--data-dir', default='CGMacros', help='Directory containing CGMacros data')
    parser.add_argument('--output', default='glucose_prediction_training_data.csv', help='Output CSV file')
    parser.add_argument('--validate', action='store_true', help='Run validation after building database')
    args = parser.parse_args()
    
    try:
        # Build training database
        training_db = build_training_database(args.data_dir, args.output)
        
        if args.validate:
            # Run validation
            logger.info("Running data validation...")
            validation_results = validate_training_data(training_db)
            
            # Print validation summary
            print("\n" + "="*50)
            print("TRAINING DATABASE VALIDATION SUMMARY")
            print("="*50)
            print(f"Total records: {validation_results['total_records']}")
            print(f"Unique subjects: {validation_results['subjects']}")
            print(f"Meal types: {validation_results['meal_types']}")
            
            print(f"\nTarget Variable Completeness:")
            for var, stats in validation_results['target_completeness'].items():
                print(f"  {var}: {stats['valid_records']:,} valid ({stats['missing_pct']:.1f}% missing)")
                
            print(f"\nCore Feature Completeness:")
            for var, stats in validation_results['feature_completeness'].items():
                print(f"  {var}: {stats['valid_records']:,} valid ({stats['missing_pct']:.1f}% missing)")
                
            print(f"\nOptional Biomarker Completeness:")
            for var, stats in validation_results['optional_completeness'].items():
                print(f"  {var}: {stats['valid_records']:,} valid ({stats['missing_pct']:.1f}% missing)")
        
        logger.info("Database building completed successfully!")
        
    except Exception as e:
        logger.error(f"Database building failed: {e}")
        raise


if __name__ == "__main__":
    main()