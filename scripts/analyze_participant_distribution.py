#!/usr/bin/env python3
"""
Analyze Participant Distribution and Glucose Response Patterns

This script analyzes the current participant distribution across glucose tolerance 
categories and examines if class imbalance is affecting glucose response patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def classify_diabetic_status(a1c: float) -> str:
    """Classify diabetic status based on HbA1c levels."""
    if a1c < 5.7:
        return 'Normal'
    elif a1c < 6.5:
        return 'Pre-diabetic'
    else:
        return 'Type2Diabetic'

def analyze_participant_distribution():
    """Analyze participant distribution across diabetic status categories."""
    
    print("=" * 60)
    print("PARTICIPANT DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Load bio data
    bio_df = pd.read_csv('CGMacros/bio.csv')
    bio_df.columns = bio_df.columns.str.strip()
    
    # Add diabetic status classification
    bio_df['diabetic_status'] = bio_df['A1c PDL (Lab)'].apply(classify_diabetic_status)
    
    print(f"\nTotal participants: {len(bio_df)}")
    print(f"\nDistribution by diabetic status:")
    status_counts = bio_df['diabetic_status'].value_counts()
    print(status_counts)
    
    print(f"\nPercentage distribution:")
    status_pct = bio_df['diabetic_status'].value_counts(normalize=True) * 100
    for status, pct in status_pct.items():
        print(f"  {status}: {pct:.1f}%")
    
    # A1c distribution
    print(f"\nA1c statistics by status:")
    a1c_stats = bio_df.groupby('diabetic_status')['A1c PDL (Lab)'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    print(a1c_stats)
    
    return bio_df, status_counts

def analyze_training_data_patterns():
    """Analyze glucose response patterns in training data."""
    
    print("\n" + "=" * 60)
    print("GLUCOSE RESPONSE PATTERN ANALYSIS")
    print("=" * 60)
    
    # Load enhanced training data if available
    try:
        training_df = pd.read_csv('glucose_prediction_training_data_enhanced.csv')
        print("Using enhanced training data with diabetic status")
    except FileNotFoundError:
        try:
            training_df = pd.read_csv('glucose_prediction_training_data_steps.csv')
            print("Using standard training data")
            
            # Add diabetic status if not present
            bio_df = pd.read_csv('CGMacros/bio.csv')
            bio_df.columns = bio_df.columns.str.strip()
            bio_df['diabetic_status'] = bio_df['A1c PDL (Lab)'].apply(classify_diabetic_status)
            
            # Merge with bio data
            training_df['subject_id_num'] = training_df['subject_id'].astype(str).str.zfill(3).astype(int)
            bio_df_merge = bio_df[['subject', 'diabetic_status']].rename(columns={'subject': 'subject_id_num'})
            training_df = training_df.merge(bio_df_merge, on='subject_id_num', how='left')
            
        except FileNotFoundError:
            print("No training data found. Please run data preparation first.")
            return None
    
    print(f"\nTotal training records: {len(training_df)}")
    
    if 'diabetic_status' in training_df.columns:
        print(f"\nTraining data distribution by diabetic status:")
        training_status = training_df['diabetic_status'].value_counts()
        print(training_status)
        
        # Analyze glucose response patterns
        glucose_cols = ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']
        available_glucose_cols = [col for col in glucose_cols if col in training_df.columns]
        
        if available_glucose_cols:
            print(f"\nGlucose response statistics by diabetic status:")
            for col in available_glucose_cols:
                minutes = col.replace('glucose_', '').replace('min', '')
                print(f"\n{minutes}-minute glucose response:")
                glucose_stats = training_df.groupby('diabetic_status')[col].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(1)
                print(glucose_stats)
                
                # Calculate glucose excursion (difference from baseline)
                if 'baseline' in training_df.columns:
                    excursion_col = f'excursion_{minutes}min'
                    training_df[excursion_col] = training_df[col] - training_df['baseline']
                    
                    print(f"\n{minutes}-minute glucose excursion (from baseline):")
                    excursion_stats = training_df.groupby('diabetic_status')[excursion_col].agg([
                        'mean', 'std', 'min', 'max'
                    ]).round(1)
                    print(excursion_stats)
    
    return training_df

def assess_class_imbalance_impact(training_df):
    """Assess if class imbalance is affecting glucose response visualization."""
    
    print("\n" + "=" * 60)
    print("CLASS IMBALANCE IMPACT ASSESSMENT")
    print("=" * 60)
    
    if training_df is None or 'diabetic_status' not in training_df.columns:
        print("Cannot assess class imbalance - diabetic status not available")
        return
    
    # Calculate class distribution
    status_counts = training_df['diabetic_status'].value_counts()
    total_records = len(training_df)
    
    print(f"\nClass distribution in training data:")
    for status, count in status_counts.items():
        pct = (count / total_records) * 100
        print(f"  {status}: {count} records ({pct:.1f}%)")
    
    # Identify imbalance
    majority_class_pct = (status_counts.iloc[0] / total_records) * 100
    minority_class_pct = (status_counts.iloc[-1] / total_records) * 100
    imbalance_ratio = status_counts.iloc[0] / status_counts.iloc[-1]
    
    print(f"\nImbalance analysis:")
    print(f"  Majority class: {majority_class_pct:.1f}%")
    print(f"  Minority class: {minority_class_pct:.1f}%")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    # Assess impact on glucose patterns
    glucose_cols = ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']
    available_glucose_cols = [col for col in glucose_cols if col in training_df.columns]
    
    if available_glucose_cols and 'baseline' in training_df.columns:
        print(f"\nImpact on glucose response patterns:")
        
        # Calculate average response by status
        for status in status_counts.index:
            status_data = training_df[training_df['diabetic_status'] == status]
            n_records = len(status_data)
            
            print(f"\n  {status} ({n_records} records):")
            baseline_mean = status_data['baseline'].mean()
            
            for col in available_glucose_cols:
                minutes = col.replace('glucose_', '').replace('min', '')
                glucose_mean = status_data[col].mean()
                excursion_mean = glucose_mean - baseline_mean
                print(f"    {minutes}min: {glucose_mean:.1f} mg/dL (+{excursion_mean:.1f} from baseline)")
        
        # Assess if minority classes are getting "averaged out"
        print(f"\nAssessment:")
        if imbalance_ratio > 3:
            print(f"  ⚠️  SEVERE CLASS IMBALANCE detected ({imbalance_ratio:.1f}:1)")
            print(f"     This could cause minority class patterns to be averaged out")
            print(f"     in aggregate statistics and visualizations.")
        elif imbalance_ratio > 2:
            print(f"  ⚠️  MODERATE CLASS IMBALANCE detected ({imbalance_ratio:.1f}:1)")
            print(f"     Minority class patterns may be less visible in aggregate analysis")
        else:
            print(f"  ✅ Classes are relatively balanced")

def evaluate_upsampling_solution():
    """Evaluate upsampling as a solution for better glucose spike visualization."""
    
    print("\n" + "=" * 60)
    print("UPSAMPLING SOLUTION EVALUATION")
    print("=" * 60)
    
    print("\nUpsampling for glucose response visualization:")
    print("\n✅ POTENTIAL BENEFITS:")
    print("  • Equal representation of all diabetic status groups")
    print("  • More prominent glucose spikes from diabetic participants")
    print("  • Better visibility of response patterns across categories")
    print("  • Improved model training for minority classes")
    
    print("\n⚠️  POTENTIAL DRAWBACKS:")
    print("  • Artificial inflation of minority class data")
    print("  • May not reflect real-world population distribution") 
    print("  • Could overemphasize rare response patterns")
    print("  • Risk of overfitting to minority class characteristics")
    
    print("\n🎯 RECOMMENDED APPROACH:")
    print("  1. Stratified analysis by diabetic status")
    print("  2. Separate visualizations for each group")
    print("  3. Weighted averaging instead of simple averaging")
    print("  4. Use confidence intervals to show uncertainty")
    print("  5. Consider SMOTE or similar techniques for balanced sampling")
    
    print("\n💡 ALTERNATIVE SOLUTIONS:")
    print("  • Stratified sampling for visualization")
    print("  • Separate models for each diabetic status")
    print("  • Weighted loss functions in modeling")
    print("  • Bootstrap sampling with replacement")

def main():
    """Main analysis function."""
    
    # Analyze participant distribution
    bio_df, status_counts = analyze_participant_distribution()
    
    # Analyze training data patterns  
    training_df = analyze_training_data_patterns()
    
    # Assess class imbalance impact
    assess_class_imbalance_impact(training_df)
    
    # Evaluate upsampling solution
    evaluate_upsampling_solution()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()