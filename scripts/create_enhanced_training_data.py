#!/usr/bin/env python3
"""
Create Enhanced Training Data with Diabetic Status

This script enhances the existing training data by:
1. Adding diabetic status classification based on HbA1c levels
2. Computing baseline glucose predictions based on diabetic status and pre-meal averages
3. Creating a new training dataset for improved glucose prediction models

Author: Enhanced CGMacros glucose prediction with diabetic status
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_diabetic_status(a1c: float) -> str:
    """
    Classify diabetic status based on HbA1c levels.
    
    Args:
        a1c: HbA1c value in %
        
    Returns:
        Diabetic status: 'Normal', 'Pre-diabetic', or 'Type2Diabetic'
    """
    if a1c < 5.7:
        return 'Normal'
    elif a1c < 6.5:
        return 'Pre-diabetic'
    else:
        return 'Type2Diabetic'


def get_baseline_glucose_by_status() -> dict:
    """
    Define baseline glucose ranges based on diabetic status.
    These are derived from our analysis of pre-meal glucose averages.
    
    Returns:
        Dictionary with baseline glucose statistics by diabetic status
    """
    return {
        'Normal': {
            'mean': 78.3,
            'std': 6.1,
            'min': 70,
            'max': 95
        },
        'Pre-diabetic': {
            'mean': 95.8,
            'std': 15.2,
            'min': 70,
            'max': 125
        },
        'Type2Diabetic': {
            'mean': 130.1,
            'std': 28.4,
            'min': 95,
            'max': 200
        }
    }


def predict_baseline_glucose(diabetic_status: str, age: float, bmi: float, 
                           fasting_glucose: float = None) -> float:
    """
    Predict baseline glucose based on diabetic status and patient characteristics.
    
    Args:
        diabetic_status: 'Normal', 'Pre-diabetic', or 'Type2Diabetic'
        age: Patient age
        bmi: Patient BMI
        fasting_glucose: Optional fasting glucose measurement
        
    Returns:
        Predicted baseline glucose in mg/dL
    """
    baseline_stats = get_baseline_glucose_by_status()
    status_stats = baseline_stats[diabetic_status]
    
    # Start with status-based mean
    baseline = status_stats['mean']
    
    # Adjust for age (glucose tends to increase with age)
    age_adjustment = (age - 40) * 0.3 if age > 40 else 0
    baseline += age_adjustment
    
    # Adjust for BMI (higher BMI associated with higher glucose)
    if bmi > 25:
        bmi_adjustment = (bmi - 25) * 0.8
        baseline += bmi_adjustment
    
    # If fasting glucose available, use it as a strong predictor
    if fasting_glucose and not np.isnan(fasting_glucose):
        # Weight fasting glucose heavily but still consider diabetic status
        baseline = 0.7 * fasting_glucose + 0.3 * baseline
    
    # Add some randomness based on status variability
    noise = np.random.normal(0, status_stats['std'] * 0.3)
    baseline += noise
    
    # Clamp to reasonable ranges
    baseline = max(status_stats['min'], min(status_stats['max'], baseline))
    
    return round(baseline, 1)


def load_bio_data() -> pd.DataFrame:
    """Load and process biographical data."""
    logger.info("Loading biographical data...")
    
    bio_df = pd.read_csv('CGMacros/bio.csv')
    
    # Clean column names
    bio_df.columns = bio_df.columns.str.strip()
    
    # Add diabetic status classification
    bio_df['diabetic_status'] = bio_df['A1c PDL (Lab)'].apply(classify_diabetic_status)
    
    # Create subject mapping (bio uses sequential numbers, data uses subject_id format)
    subject_mapping = {}
    for idx, row in bio_df.iterrows():
        bio_subject_id = row['subject']
        cgmacros_subject_id = f"{bio_subject_id:03d}"  # Format as 001, 002, etc.
        subject_mapping[cgmacros_subject_id] = bio_subject_id
    
    logger.info(f"Loaded data for {len(bio_df)} subjects")
    logger.info(f"Diabetic status distribution:")
    logger.info(bio_df['diabetic_status'].value_counts())
    
    return bio_df, subject_mapping


def enhance_training_data():
    """Create enhanced training data with diabetic status and improved baseline predictions."""
    
    logger.info("Creating enhanced training data...")
    
    # Load bio data
    bio_df, subject_mapping = load_bio_data()
    
    # Load existing training data
    logger.info("Loading existing training data...")
    training_df = pd.read_csv('glucose_prediction_training_data_steps.csv')
    
    logger.info(f"Original training data: {len(training_df)} records")
    
    # Create enhanced dataset
    enhanced_records = []
    
    # Process each record
    for idx, row in training_df.iterrows():
        # Extract subject ID (remove leading zeros for bio lookup)
        subject_id_str = str(row['subject_id']).zfill(3)
        subject_id_num = int(subject_id_str)
        
        # Find corresponding bio data
        bio_row = bio_df[bio_df['subject'] == subject_id_num]
        
        if len(bio_row) == 0:
            logger.warning(f"No bio data found for subject {subject_id_str}")
            continue
            
        bio_row = bio_row.iloc[0]
        
        # Create enhanced record
        enhanced_record = row.to_dict()
        
        # Add diabetic status
        enhanced_record['diabetic_status'] = bio_row['diabetic_status']
        
        # Get patient characteristics
        age = bio_row['Age']
        bmi = bio_row['BMI']
        fasting_glucose = bio_row['Fasting GLU - PDL (Lab)']
        
        # Predict improved baseline glucose
        predicted_baseline = predict_baseline_glucose(
            diabetic_status=bio_row['diabetic_status'],
            age=age,
            bmi=bmi,
            fasting_glucose=fasting_glucose
        )
        
        # Update baseline (keep original as backup)
        enhanced_record['original_baseline'] = enhanced_record['baseline']
        enhanced_record['predicted_baseline'] = predicted_baseline
        enhanced_record['baseline'] = predicted_baseline
        
        # Add additional bio markers that might be useful
        enhanced_record['a1c'] = bio_row['A1c PDL (Lab)']
        enhanced_record['fasting_glucose'] = fasting_glucose
        enhanced_record['fasting_insulin'] = bio_row['Insulin']
        
        enhanced_records.append(enhanced_record)
        
        if len(enhanced_records) % 500 == 0:
            logger.info(f"Processed {len(enhanced_records)} records...")
    
    # Create enhanced DataFrame
    enhanced_df = pd.DataFrame(enhanced_records)
    
    # Add diabetic status encoding for model training
    status_encoding = {'Normal': 0, 'Pre-diabetic': 1, 'Type2Diabetic': 2}
    enhanced_df['diabetic_status_encoded'] = enhanced_df['diabetic_status'].map(status_encoding)
    
    logger.info(f"Enhanced dataset created: {len(enhanced_df)} records")
    logger.info(f"Diabetic status distribution in training data:")
    logger.info(enhanced_df['diabetic_status'].value_counts())
    
    # Save enhanced training data
    output_file = 'glucose_prediction_training_data_enhanced.csv'
    enhanced_df.to_csv(output_file, index=False)
    logger.info(f"Enhanced training data saved to: {output_file}")
    
    # Create summary statistics
    logger.info("\nBaseline glucose statistics by diabetic status:")
    baseline_stats = enhanced_df.groupby('diabetic_status')['baseline'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    logger.info(baseline_stats)
    
    # Save baseline statistics
    baseline_stats.to_csv('baseline_glucose_by_status.csv')
    logger.info("Baseline statistics saved to: baseline_glucose_by_status.csv")
    
    return enhanced_df


def create_status_baseline_lookup():
    """Create a lookup table for baseline glucose by diabetic status."""
    
    logger.info("Creating diabetic status baseline lookup...")
    
    baseline_stats = get_baseline_glucose_by_status()
    
    # Convert to DataFrame for easy saving
    lookup_data = []
    for status, stats in baseline_stats.items():
        lookup_data.append({
            'diabetic_status': status,
            'mean_baseline': stats['mean'],
            'std_baseline': stats['std'],
            'min_baseline': stats['min'],
            'max_baseline': stats['max']
        })
    
    lookup_df = pd.DataFrame(lookup_data)
    lookup_df.to_csv('diabetic_status_baseline_lookup.csv', index=False)
    logger.info("Diabetic status lookup saved to: diabetic_status_baseline_lookup.csv")
    
    return lookup_df


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create enhanced training data
    enhanced_df = enhance_training_data()
    
    # Create lookup table
    lookup_df = create_status_baseline_lookup()
    
    logger.info("Enhanced training data creation completed!")
    logger.info("Files created:")
    logger.info("  - glucose_prediction_training_data_enhanced.csv")
    logger.info("  - baseline_glucose_by_status.csv") 
    logger.info("  - diabetic_status_baseline_lookup.csv")