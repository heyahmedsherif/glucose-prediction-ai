#!/usr/bin/env python3
"""
Meal Pattern Analysis - Timing and First Meal Effects on Glucose Response

Analyzes how glucose spiking changes based on:
1. Meal timing (hour of day)
2. Meal type (breakfast, lunch, dinner)
3. First meal of the day effects
4. Sequential meal effects within the same day

Run with: python meal_pattern_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_meal_patterns():
    """Comprehensive analysis of meal timing patterns and glucose responses."""
    
    print("🍽️ MEAL PATTERN ANALYSIS")
    print("=" * 50)
    
    # Load data
    try:
        data = pd.read_csv("glucose_prediction_training_data_enhanced.csv")
        print(f"✅ Loaded {len(data)} meal records")
    except FileNotFoundError:
        print("❌ Enhanced training data not found")
        return
    
    # Parse timestamps and extract timing features
    data['meal_datetime'] = pd.to_datetime(data['meal_timestamp'])
    data['hour'] = data['meal_datetime'].dt.hour
    data['day'] = data['meal_datetime'].dt.date
    data['day_of_week'] = data['meal_datetime'].dt.day_name()
    
    # Calculate glucose metrics
    glucose_cols = ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']
    data['peak_glucose'] = data[glucose_cols].max(axis=1)
    data['glucose_excursion'] = data['peak_glucose'] - data['baseline']
    data['glucose_auc'] = data[glucose_cols].sum(axis=1) / len(glucose_cols)  # Simplified AUC
    
    print(f"\n📊 Dataset Overview:")
    print(f"  • Subjects: {data['subject_id'].nunique()}")
    print(f"  • Days: {data['day'].nunique()}")
    print(f"  • Meal types: {data['meal_type'].unique()}")
    print(f"  • Time range: {data['meal_datetime'].min()} to {data['meal_datetime'].max()}")
    
    # 1. MEAL TYPE ANALYSIS
    print(f"\n🍽️ ANALYSIS 1: Meal Type Effects")
    print("-" * 40)
    
    meal_analysis = data.groupby('meal_type').agg({
        'peak_glucose': ['mean', 'std'],
        'glucose_excursion': ['mean', 'std'],
        'glucose_auc': ['mean', 'std'],
        'carbohydrates': 'mean',
        'calories': 'mean',
        'hour': 'mean'
    }).round(1)
    
    print(meal_analysis)
    
    # Statistical test for meal type differences
    breakfast_excursion = data[data['meal_type'] == 'breakfast']['glucose_excursion']
    lunch_excursion = data[data['meal_type'] == 'lunch']['glucose_excursion']
    dinner_excursion = data[data['meal_type'] == 'dinner']['glucose_excursion']
    
    f_stat, p_val = stats.f_oneway(breakfast_excursion, lunch_excursion, dinner_excursion)
    print(f"\nMeal type effect on glucose excursion: F = {f_stat:.2f}, p = {p_val:.3f}")
    
    if p_val < 0.05:
        print("✅ SIGNIFICANT: Meal type affects glucose response")
        
        # Post-hoc pairwise comparisons
        meals = ['breakfast', 'lunch', 'dinner']
        for i, meal1 in enumerate(meals):
            for meal2 in meals[i+1:]:
                group1 = data[data['meal_type'] == meal1]['glucose_excursion']
                group2 = data[data['meal_type'] == meal2]['glucose_excursion']
                t_stat, p_val_pair = stats.ttest_ind(group1, group2)
                diff = group1.mean() - group2.mean()
                print(f"  {meal1} vs {meal2}: difference = {diff:.1f} mg/dL, p = {p_val_pair:.3f}")
    
    # 2. HOURLY TIMING ANALYSIS
    print(f"\n⏰ ANALYSIS 2: Hourly Timing Effects")
    print("-" * 40)
    
    # Create time bins for analysis
    data['time_bin'] = pd.cut(data['hour'], 
                              bins=[0, 8, 12, 17, 24], 
                              labels=['Early (0-8)', 'Morning (8-12)', 'Afternoon (12-17)', 'Evening (17-24)'],
                              include_lowest=True)
    
    time_analysis = data.groupby('time_bin').agg({
        'peak_glucose': ['mean', 'std'],
        'glucose_excursion': ['mean', 'std'],
        'glucose_auc': ['mean', 'std'],
        'hour': ['mean', 'min', 'max']
    }).round(1)
    
    print(time_analysis)
    
    # Test correlation between hour and glucose response
    hour_corr = stats.pearsonr(data['hour'], data['glucose_excursion'])
    print(f"\nHour vs Glucose Excursion: r = {hour_corr[0]:.3f}, p = {hour_corr[1]:.3f}")
    
    # 3. FIRST MEAL OF DAY ANALYSIS
    print(f"\n🌅 ANALYSIS 3: First Meal of Day Effects")
    print("-" * 40)
    
    # Identify first meal of each day for each subject
    daily_meals = data.groupby(['subject_id', 'day']).apply(
        lambda x: x.loc[x['meal_datetime'].idxmin()]
    ).reset_index(drop=True)
    
    # Mark first meals
    data['is_first_meal'] = False
    for idx, row in daily_meals.iterrows():
        mask = ((data['subject_id'] == row['subject_id']) & 
                (data['day'] == row['day']) & 
                (data['meal_datetime'] == row['meal_datetime']))
        data.loc[mask, 'is_first_meal'] = True
    
    first_meal_count = data['is_first_meal'].sum()
    print(f"First meals identified: {first_meal_count}")
    
    # Compare first vs subsequent meals
    first_meals = data[data['is_first_meal'] == True]
    subsequent_meals = data[data['is_first_meal'] == False]
    
    print(f"\n📈 First Meal vs Subsequent Meals:")
    print(f"First meals ({len(first_meals)}):")
    print(f"  • Mean excursion: {first_meals['glucose_excursion'].mean():.1f} mg/dL")
    print(f"  • Mean peak: {first_meals['peak_glucose'].mean():.1f} mg/dL")
    print(f"  • Mean AUC: {first_meals['glucose_auc'].mean():.1f}")
    
    print(f"Subsequent meals ({len(subsequent_meals)}):")
    print(f"  • Mean excursion: {subsequent_meals['glucose_excursion'].mean():.1f} mg/dL")
    print(f"  • Mean peak: {subsequent_meals['peak_glucose'].mean():.1f} mg/dL")
    print(f"  • Mean AUC: {subsequent_meals['glucose_auc'].mean():.1f}")
    
    # Statistical test
    t_stat, p_val = stats.ttest_ind(first_meals['glucose_excursion'], 
                                    subsequent_meals['glucose_excursion'])
    diff = first_meals['glucose_excursion'].mean() - subsequent_meals['glucose_excursion'].mean()
    
    print(f"\nFirst meal effect: difference = {diff:.1f} mg/dL, p = {p_val:.3f}")
    
    if p_val < 0.05:
        if diff > 0:
            print("✅ SIGNIFICANT: First meals cause HIGHER glucose response")
        else:
            print("✅ SIGNIFICANT: First meals cause LOWER glucose response")
    else:
        print("❌ NOT SIGNIFICANT: No clear first meal effect")
    
    # 4. MEAL SEQUENCE ANALYSIS
    print(f"\n📝 ANALYSIS 4: Meal Sequence Effects")
    print("-" * 40)
    
    # Create meal sequence within each day
    data_sorted = data.sort_values(['subject_id', 'day', 'meal_datetime'])
    data_sorted['meal_sequence'] = data_sorted.groupby(['subject_id', 'day']).cumcount() + 1
    
    # Analyze by meal sequence (1st, 2nd, 3rd+ meals)
    sequence_analysis = data_sorted.groupby('meal_sequence').agg({
        'glucose_excursion': ['mean', 'std', 'count'],
        'peak_glucose': ['mean', 'std'],
        'carbohydrates': 'mean',
        'hour': 'mean'
    }).round(1)
    
    print("Meal sequence analysis (1st, 2nd, 3rd+ meal of day):")
    print(sequence_analysis)
    
    # 5. DAWN PHENOMENON ANALYSIS
    print(f"\n🌅 ANALYSIS 5: Dawn Phenomenon Effects")
    print("-" * 40)
    
    # Analyze early morning meals (6-9 AM) vs other breakfast times
    early_breakfast = data[(data['meal_type'] == 'breakfast') & (data['hour'] >= 6) & (data['hour'] <= 9)]
    late_breakfast = data[(data['meal_type'] == 'breakfast') & ((data['hour'] < 6) | (data['hour'] > 9))]
    
    if len(early_breakfast) > 0 and len(late_breakfast) > 0:
        print(f"Early breakfast (6-9 AM): {len(early_breakfast)} meals")
        print(f"  • Mean excursion: {early_breakfast['glucose_excursion'].mean():.1f} mg/dL")
        print(f"  • Mean baseline: {early_breakfast['baseline'].mean():.1f} mg/dL")
        
        print(f"Late breakfast (other hours): {len(late_breakfast)} meals") 
        print(f"  • Mean excursion: {late_breakfast['glucose_excursion'].mean():.1f} mg/dL")
        print(f"  • Mean baseline: {late_breakfast['baseline'].mean():.1f} mg/dL")
        
        # Test for dawn phenomenon effect
        t_stat, p_val = stats.ttest_ind(early_breakfast['glucose_excursion'], 
                                        late_breakfast['glucose_excursion'])
        diff = early_breakfast['glucose_excursion'].mean() - late_breakfast['glucose_excursion'].mean()
        
        print(f"\nDawn phenomenon effect: difference = {diff:.1f} mg/dL, p = {p_val:.3f}")
        
        if p_val < 0.05 and diff > 0:
            print("✅ DAWN PHENOMENON DETECTED: Early breakfast shows higher glucose response")
        elif p_val < 0.05 and diff < 0:
            print("✅ REVERSE DAWN EFFECT: Early breakfast shows lower glucose response")
        else:
            print("❌ NO DAWN PHENOMENON: No significant difference")
    
    # 6. DIABETIC STATUS INTERACTION
    print(f"\n🏥 ANALYSIS 6: Timing Effects by Diabetic Status")
    print("-" * 40)
    
    for status in ['Normal', 'Pre-diabetic', 'Type2Diabetic']:
        if status in data['diabetic_status'].values:
            status_data = data[data['diabetic_status'] == status]
            print(f"\n{status} individuals ({len(status_data)} meals):")
            
            # Meal type effects within this diabetic status
            for meal in ['breakfast', 'lunch', 'dinner']:
                meal_data = status_data[status_data['meal_type'] == meal]
                if len(meal_data) > 0:
                    print(f"  {meal}: {meal_data['glucose_excursion'].mean():.1f} mg/dL (n={len(meal_data)})")
            
            # First meal effect within this status
            first_meals_status = status_data[status_data['is_first_meal'] == True]
            subsequent_meals_status = status_data[status_data['is_first_meal'] == False]
            
            if len(first_meals_status) > 0 and len(subsequent_meals_status) > 0:
                first_mean = first_meals_status['glucose_excursion'].mean()
                subsequent_mean = subsequent_meals_status['glucose_excursion'].mean()
                print(f"  First meal effect: {first_mean:.1f} vs {subsequent_mean:.1f} mg/dL")
    
    # 7. PRACTICAL INSIGHTS
    print(f"\n💡 PRACTICAL INSIGHTS & RECOMMENDATIONS")
    print("-" * 50)
    
    # Find optimal timing
    hour_effects = data.groupby('hour')['glucose_excursion'].mean()
    best_hour = hour_effects.idxmin()
    worst_hour = hour_effects.idxmax()
    
    print(f"🕐 Timing Recommendations:")
    print(f"  • Best hour for meals: {best_hour}:00 (avg excursion: {hour_effects[best_hour]:.1f} mg/dL)")
    print(f"  • Worst hour for meals: {worst_hour}:00 (avg excursion: {hour_effects[worst_hour]:.1f} mg/dL)")
    print(f"  • Timing difference: {hour_effects[worst_hour] - hour_effects[best_hour]:.1f} mg/dL")
    
    # Meal type recommendations
    meal_effects = data.groupby('meal_type')['glucose_excursion'].mean()
    best_meal = meal_effects.idxmin()
    worst_meal = meal_effects.idxmax()
    
    print(f"\n🍽️ Meal Type Recommendations:")
    print(f"  • Lowest response: {best_meal} ({meal_effects[best_meal]:.1f} mg/dL)")
    print(f"  • Highest response: {worst_meal} ({meal_effects[worst_meal]:.1f} mg/dL)")
    print(f"  • Meal type difference: {meal_effects[worst_meal] - meal_effects[best_meal]:.1f} mg/dL")
    
    # First meal insights
    if abs(diff) > 5:  # Only if meaningful difference
        print(f"\n🌅 First Meal Insights:")
        if diff > 0:
            print(f"  • First meals show {diff:.1f} mg/dL HIGHER glucose response")
            print(f"  • Consider lighter first meals or medication timing adjustment")
        else:
            print(f"  • First meals show {abs(diff):.1f} mg/dL LOWER glucose response")
            print(f"  • First meal may provide metabolic priming benefit")
    
    return data_sorted

if __name__ == "__main__":
    result_data = analyze_meal_patterns()