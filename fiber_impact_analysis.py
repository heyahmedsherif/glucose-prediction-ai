#!/usr/bin/env python3
"""
Fiber Impact Analysis on Glucose Spikes

Analyze how fiber intake correlates with glucose response in the CGMacros dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_fiber_impact():
    """Analyze fiber's impact on glucose spikes using CGMacros data."""
    
    print("🌾 Analyzing Fiber Impact on Glucose Spikes")
    print("=" * 50)
    
    # Load the training data which should have processed meal and glucose data
    try:
        data = pd.read_csv("glucose_prediction_training_data_enhanced.csv")
        print(f"✅ Loaded {len(data)} meal records")
    except FileNotFoundError:
        print("❌ Enhanced training data not found, trying basic version...")
        try:
            data = pd.read_csv("glucose_prediction_training_data.csv") 
            print(f"✅ Loaded {len(data)} meal records from basic dataset")
        except FileNotFoundError:
            print("❌ No training data found. Please run the model training first.")
            return
    
    print(f"📊 Columns available: {list(data.columns)}")
    
    # Check for required columns
    required_cols = ['fiber', 'carbohydrates']
    glucose_cols = [col for col in data.columns if 'glucose_' in col and 'min' in col]
    
    if not all(col in data.columns for col in required_cols):
        print(f"❌ Missing required columns. Need: {required_cols}")
        return
        
    if not glucose_cols:
        print("❌ No glucose time point columns found")
        return
    
    print(f"🎯 Found glucose time points: {glucose_cols}")
    
    # Clean extreme outliers in fiber (likely data entry errors)
    print(f"Raw fiber range: {data['fiber'].min():.1f} - {data['fiber'].max():.1f}g")
    
    # Remove extreme outliers (>95th percentile is likely data errors)
    fiber_95th = data['fiber'].quantile(0.95)
    print(f"95th percentile fiber: {fiber_95th:.1f}g")
    
    # Keep reasonable fiber values (0-50g is realistic)
    original_len = len(data)
    data = data[data['fiber'] <= 50]  # Reasonable upper limit
    print(f"Removed {original_len - len(data)} extreme outliers (>50g fiber)")
    
    # Basic fiber statistics
    print(f"\n📈 Fiber Intake Statistics (cleaned):")
    print(f"  • Mean: {data['fiber'].mean():.1f}g")
    print(f"  • Median: {data['fiber'].median():.1f}g") 
    print(f"  • Range: {data['fiber'].min():.1f} - {data['fiber'].max():.1f}g")
    print(f"  • Std Dev: {data['fiber'].std():.1f}g")
    
    # Calculate glucose excursions (peak - baseline)
    if 'baseline' in data.columns:
        baseline_col = 'baseline'
    else:
        # Use minimum glucose as proxy for baseline
        glucose_values = data[glucose_cols].values
        data['estimated_baseline'] = np.min(glucose_values, axis=1)
        baseline_col = 'estimated_baseline'
        print("⚠️  Using estimated baseline (minimum glucose value)")
    
    # Calculate peak glucose and excursion
    glucose_values = data[glucose_cols].values
    data['peak_glucose'] = np.max(glucose_values, axis=1)
    data['glucose_excursion'] = data['peak_glucose'] - data[baseline_col]
    
    print(f"\n🚀 Glucose Response Statistics:")
    print(f"  • Mean Peak: {data['peak_glucose'].mean():.1f} mg/dL")
    print(f"  • Mean Excursion: {data['glucose_excursion'].mean():.1f} mg/dL")
    
    # Fiber quartile analysis - handle duplicate values
    try:
        data['fiber_quartile'] = pd.qcut(data['fiber'], q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'], duplicates='drop')
    except ValueError:
        # If still having issues, use manual bins
        fiber_bins = [0, 2, 6, 10, 50]  # Manual reasonable bins
        data['fiber_quartile'] = pd.cut(data['fiber'], bins=fiber_bins, labels=['Low', 'Med-Low', 'Med-High', 'High'], include_lowest=True)
    
    print(f"\n🌾 Fiber Impact Analysis by Quartiles:")
    fiber_analysis = data.groupby('fiber_quartile').agg({
        'fiber': ['mean', 'min', 'max'],
        'carbohydrates': 'mean',
        'glucose_excursion': ['mean', 'std'],
        'peak_glucose': ['mean', 'std']
    }).round(1)
    
    print(fiber_analysis)
    
    # Calculate correlations
    print(f"\n📊 Fiber Correlations:")
    fiber_corr_excursion = stats.pearsonr(data['fiber'], data['glucose_excursion'])
    fiber_corr_peak = stats.pearsonr(data['fiber'], data['peak_glucose'])
    
    print(f"  • Fiber vs Glucose Excursion: r = {fiber_corr_excursion[0]:.3f}, p = {fiber_corr_excursion[1]:.3f}")
    print(f"  • Fiber vs Peak Glucose: r = {fiber_corr_peak[0]:.3f}, p = {fiber_corr_peak[1]:.3f}")
    
    # Controlled analysis: similar carb meals with different fiber
    print(f"\n🎯 Controlled Analysis: High-Carb Meals (>40g carbs)")
    high_carb_data = data[data['carbohydrates'] > 40]
    
    if len(high_carb_data) > 50:
        # Split into low vs high fiber for high-carb meals
        fiber_median = high_carb_data['fiber'].median()
        low_fiber = high_carb_data[high_carb_data['fiber'] <= fiber_median]
        high_fiber = high_carb_data[high_carb_data['fiber'] > fiber_median]
        
        print(f"  • Low Fiber Group (≤{fiber_median:.1f}g): {len(low_fiber)} meals")
        print(f"    - Mean fiber: {low_fiber['fiber'].mean():.1f}g")
        print(f"    - Mean carbs: {low_fiber['carbohydrates'].mean():.1f}g") 
        print(f"    - Mean excursion: {low_fiber['glucose_excursion'].mean():.1f} mg/dL")
        print(f"    - Mean peak: {low_fiber['peak_glucose'].mean():.1f} mg/dL")
        
        print(f"  • High Fiber Group (>{fiber_median:.1f}g): {len(high_fiber)} meals")
        print(f"    - Mean fiber: {high_fiber['fiber'].mean():.1f}g")
        print(f"    - Mean carbs: {high_fiber['carbohydrates'].mean():.1f}g")
        print(f"    - Mean excursion: {high_fiber['glucose_excursion'].mean():.1f} mg/dL")
        print(f"    - Mean peak: {high_fiber['peak_glucose'].mean():.1f} mg/dL")
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(low_fiber['glucose_excursion'], high_fiber['glucose_excursion'])
        excursion_diff = low_fiber['glucose_excursion'].mean() - high_fiber['glucose_excursion'].mean()
        
        print(f"\n📈 Fiber Effect on High-Carb Meals:")
        print(f"  • Excursion Difference: {excursion_diff:.1f} mg/dL (Low - High Fiber)")
        print(f"  • Statistical Significance: p = {p_value:.3f}")
        
        if p_value < 0.05:
            print(f"  • ✅ SIGNIFICANT: High fiber reduces glucose excursion by {abs(excursion_diff):.1f} mg/dL")
        else:
            print(f"  • ❌ NOT SIGNIFICANT: No clear fiber effect detected")
    
    # Diabetic status analysis
    if 'diabetic_status' in data.columns:
        print(f"\n🏥 Fiber Effect by Diabetic Status:")
        for status in ['Normal', 'Pre-diabetic', 'Type2Diabetic']:
            if status in data['diabetic_status'].values:
                status_data = data[data['diabetic_status'] == status]
                fiber_corr = stats.pearsonr(status_data['fiber'], status_data['glucose_excursion'])
                print(f"  • {status}: r = {fiber_corr[0]:.3f}, p = {fiber_corr[1]:.3f} ({len(status_data)} meals)")
    
    # Practical fiber recommendations
    print(f"\n💡 Practical Insights:")
    
    # Find optimal fiber ranges
    data['fiber_bin'] = pd.cut(data['fiber'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    fiber_effect = data.groupby('fiber_bin')['glucose_excursion'].mean()
    
    best_fiber_bin = fiber_effect.idxmin()
    worst_fiber_bin = fiber_effect.idxmax()
    
    print(f"  • Lowest glucose spikes with: {best_fiber_bin} fiber intake")
    print(f"  • Highest glucose spikes with: {worst_fiber_bin} fiber intake")
    print(f"  • Difference: {fiber_effect.max() - fiber_effect.min():.1f} mg/dL")
    
    # Fiber per carb ratio analysis
    data['fiber_carb_ratio'] = data['fiber'] / data['carbohydrates']
    fiber_ratio_corr = stats.pearsonr(data['fiber_carb_ratio'], data['glucose_excursion'])
    
    print(f"  • Fiber-to-Carb ratio correlation: r = {fiber_ratio_corr[0]:.3f}, p = {fiber_ratio_corr[1]:.3f}")
    
    if fiber_ratio_corr[1] < 0.05:
        print(f"  • ✅ Higher fiber-to-carb ratios significantly reduce glucose spikes")
    
    return data

if __name__ == "__main__":
    analyze_fiber_impact()