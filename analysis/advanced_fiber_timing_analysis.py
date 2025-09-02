#!/usr/bin/env python3
"""
Advanced Fiber-Timing Interaction Analysis

Explores sophisticated relationships between fiber intake and meal timing effects:
1. Fiber effectiveness by time of day
2. Fiber-timing interactions for different diabetic statuses
3. Optimal fiber-to-carb ratios by meal type
4. Fiber's role in mitigating dawn phenomenon
5. Sequential meal fiber strategies

Run with: python advanced_fiber_timing_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def advanced_fiber_timing_analysis():
    """Advanced analysis of fiber-timing interactions."""
    
    print("🌾⏰ ADVANCED FIBER-TIMING INTERACTION ANALYSIS")
    print("=" * 60)
    
    # Load data
    try:
        data = pd.read_csv("glucose_prediction_training_data_enhanced.csv")
        print(f"✅ Loaded {len(data)} meal records")
    except FileNotFoundError:
        print("❌ Enhanced training data not found")
        return
    
    # Parse timing and calculate metrics
    data['meal_datetime'] = pd.to_datetime(data['meal_timestamp'])
    data['hour'] = data['meal_datetime'].dt.hour
    data['day'] = data['meal_datetime'].dt.date
    
    # Calculate glucose metrics
    glucose_cols = ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']
    data['peak_glucose'] = data[glucose_cols].max(axis=1)
    data['glucose_excursion'] = data['peak_glucose'] - data['baseline']
    data['glucose_auc'] = data[glucose_cols].sum(axis=1) / len(glucose_cols)
    
    # Clean fiber data
    data = data[(data['fiber'] >= 0) & (data['fiber'] <= 50)]  # Remove outliers
    
    # Identify first meals
    daily_meals = data.groupby(['subject_id', 'day']).apply(
        lambda x: x.loc[x['meal_datetime'].idxmin()]
    ).reset_index(drop=True)
    
    data['is_first_meal'] = False
    for idx, row in daily_meals.iterrows():
        mask = ((data['subject_id'] == row['subject_id']) & 
                (data['day'] == row['day']) & 
                (data['meal_datetime'] == row['meal_datetime']))
        data.loc[mask, 'is_first_meal'] = True
    
    # Calculate fiber ratios
    data['fiber_carb_ratio'] = data['fiber'] / (data['carbohydrates'] + 0.1)  # Avoid division by zero
    data['fiber_per_100_cal'] = (data['fiber'] / (data['calories'] + 1)) * 100
    
    # Create fiber categories
    data['fiber_category'] = pd.cut(data['fiber'], 
                                   bins=[0, 3, 8, 15, 50], 
                                   labels=['Very Low (0-3g)', 'Low (3-8g)', 'Moderate (8-15g)', 'High (15g+)'])
    
    # Time categories
    data['time_category'] = pd.cut(data['hour'], 
                                  bins=[0, 8, 12, 17, 24],
                                  labels=['Early (0-8)', 'Morning (8-12)', 'Afternoon (12-17)', 'Evening (17-24)'],
                                  include_lowest=True)
    
    print(f"\n📊 Data Overview:")
    print(f"  • Fiber range: {data['fiber'].min():.1f} - {data['fiber'].max():.1f}g")
    print(f"  • Mean fiber: {data['fiber'].mean():.1f}g")
    print(f"  • Fiber-carb ratio: {data['fiber_carb_ratio'].mean():.2f}")
    
    # 1. FIBER EFFECTIVENESS BY TIME OF DAY
    print(f"\n🕐 ANALYSIS 1: Fiber Effectiveness by Time of Day")
    print("-" * 50)
    
    # Calculate fiber correlation by hour
    hourly_fiber_effectiveness = []
    for hour in range(6, 23):  # Reasonable meal hours
        hour_data = data[data['hour'] == hour]
        if len(hour_data) > 20:  # Need sufficient data
            corr, p_val = stats.pearsonr(hour_data['fiber'], hour_data['glucose_excursion'])
            hourly_fiber_effectiveness.append({
                'hour': hour,
                'correlation': corr,
                'p_value': p_val,
                'n_meals': len(hour_data),
                'mean_fiber': hour_data['fiber'].mean(),
                'mean_excursion': hour_data['glucose_excursion'].mean()
            })
    
    fiber_effectiveness_df = pd.DataFrame(hourly_fiber_effectiveness)
    
    print("Fiber-glucose correlation by hour:")
    print(fiber_effectiveness_df[['hour', 'correlation', 'p_value', 'n_meals']].round(3))
    
    # Find best and worst times for fiber effectiveness
    significant_hours = fiber_effectiveness_df[fiber_effectiveness_df['p_value'] < 0.05]
    if len(significant_hours) > 0:
        best_fiber_hour = significant_hours.loc[significant_hours['correlation'].idxmin()]
        worst_fiber_hour = significant_hours.loc[significant_hours['correlation'].idxmax()]
        
        print(f"\n💡 Fiber Timing Insights:")
        print(f"  • Most effective fiber hour: {best_fiber_hour['hour']}:00 (r = {best_fiber_hour['correlation']:.3f})")
        print(f"  • Least effective fiber hour: {worst_fiber_hour['hour']}:00 (r = {worst_fiber_hour['correlation']:.3f})")
    
    # 2. FIBER-TIMING INTERACTION BY MEAL TYPE
    print(f"\n🍽️ ANALYSIS 2: Fiber Effectiveness by Meal Type and Timing")
    print("-" * 55)
    
    meal_timing_fiber = data.groupby(['meal_type', 'time_category']).apply(
        lambda x: pd.Series({
            'fiber_glucose_corr': stats.pearsonr(x['fiber'], x['glucose_excursion'])[0] if len(x) > 10 else np.nan,
            'fiber_glucose_p': stats.pearsonr(x['fiber'], x['glucose_excursion'])[1] if len(x) > 10 else np.nan,
            'mean_fiber': x['fiber'].mean(),
            'mean_excursion': x['glucose_excursion'].mean(),
            'n_meals': len(x)
        })
    ).round(3)
    
    print("Fiber effectiveness by meal type and time:")
    print(meal_timing_fiber)
    
    # 3. DAWN PHENOMENON FIBER MITIGATION
    print(f"\n🌅 ANALYSIS 3: Fiber's Role in Dawn Phenomenon Mitigation")
    print("-" * 55)
    
    # Compare early breakfast with different fiber levels
    early_breakfast = data[(data['meal_type'] == 'breakfast') & 
                          (data['hour'] >= 6) & (data['hour'] <= 9)]
    
    if len(early_breakfast) > 50:
        # Split by fiber level
        fiber_median = early_breakfast['fiber'].median()
        low_fiber_dawn = early_breakfast[early_breakfast['fiber'] <= fiber_median]
        high_fiber_dawn = early_breakfast[early_breakfast['fiber'] > fiber_median]
        
        print(f"Dawn phenomenon analysis (6-9 AM breakfast):")
        print(f"  Low fiber group (≤{fiber_median:.1f}g): {len(low_fiber_dawn)} meals")
        print(f"    • Mean fiber: {low_fiber_dawn['fiber'].mean():.1f}g")
        print(f"    • Mean excursion: {low_fiber_dawn['glucose_excursion'].mean():.1f} mg/dL")
        print(f"    • Mean peak: {low_fiber_dawn['peak_glucose'].mean():.1f} mg/dL")
        
        print(f"  High fiber group (>{fiber_median:.1f}g): {len(high_fiber_dawn)} meals")
        print(f"    • Mean fiber: {high_fiber_dawn['fiber'].mean():.1f}g")
        print(f"    • Mean excursion: {high_fiber_dawn['glucose_excursion'].mean():.1f} mg/dL")
        print(f"    • Mean peak: {high_fiber_dawn['peak_glucose'].mean():.1f} mg/dL")
        
        # Statistical test for dawn phenomenon mitigation
        t_stat, p_val = stats.ttest_ind(low_fiber_dawn['glucose_excursion'], 
                                        high_fiber_dawn['glucose_excursion'])
        dawn_mitigation = low_fiber_dawn['glucose_excursion'].mean() - high_fiber_dawn['glucose_excursion'].mean()
        
        print(f"\n  Dawn phenomenon fiber mitigation: {dawn_mitigation:.1f} mg/dL, p = {p_val:.3f}")
        
        if p_val < 0.05 and dawn_mitigation > 0:
            print(f"  ✅ FIBER MITIGATES DAWN PHENOMENON: {dawn_mitigation:.1f} mg/dL reduction")
            
            # Calculate mitigation percentage
            mitigation_pct = (dawn_mitigation / low_fiber_dawn['glucose_excursion'].mean()) * 100
            print(f"  📊 Mitigation effectiveness: {mitigation_pct:.1f}% reduction in dawn response")
        else:
            print(f"  ❌ No significant dawn phenomenon mitigation detected")
    
    # 4. OPTIMAL FIBER-CARB RATIOS BY TIMING
    print(f"\n📊 ANALYSIS 4: Optimal Fiber-Carb Ratios by Timing")
    print("-" * 50)
    
    # Analyze fiber-carb ratio effectiveness by time
    for time_cat in data['time_category'].unique():
        if pd.isna(time_cat):
            continue
            
        time_data = data[data['time_category'] == time_cat]
        if len(time_data) > 50:
            
            # Find optimal fiber-carb ratio
            ratio_corr = stats.pearsonr(time_data['fiber_carb_ratio'], time_data['glucose_excursion'])
            
            print(f"\n{time_cat} timing ({len(time_data)} meals):")
            print(f"  • Fiber-carb ratio correlation: r = {ratio_corr[0]:.3f}, p = {ratio_corr[1]:.3f}")
            print(f"  • Mean fiber-carb ratio: {time_data['fiber_carb_ratio'].mean():.3f}")
            
            # Find quartiles and their effects
            quartiles = time_data['fiber_carb_ratio'].quantile([0.25, 0.5, 0.75])
            
            low_ratio = time_data[time_data['fiber_carb_ratio'] <= quartiles[0.25]]
            high_ratio = time_data[time_data['fiber_carb_ratio'] >= quartiles[0.75]]
            
            if len(low_ratio) > 5 and len(high_ratio) > 5:
                ratio_effect = low_ratio['glucose_excursion'].mean() - high_ratio['glucose_excursion'].mean()
                print(f"  • Low ratio effect: {low_ratio['glucose_excursion'].mean():.1f} mg/dL")
                print(f"  • High ratio effect: {high_ratio['glucose_excursion'].mean():.1f} mg/dL") 
                print(f"  • Ratio benefit: {ratio_effect:.1f} mg/dL reduction with high fiber-carb ratio")
    
    # 5. FIRST MEAL FIBER STRATEGY
    print(f"\n🥇 ANALYSIS 5: First Meal Fiber Strategy")
    print("-" * 45)
    
    first_meals = data[data['is_first_meal'] == True]
    subsequent_meals = data[data['is_first_meal'] == False]
    
    # Fiber effectiveness in first vs subsequent meals
    first_fiber_corr = stats.pearsonr(first_meals['fiber'], first_meals['glucose_excursion'])
    subsequent_fiber_corr = stats.pearsonr(subsequent_meals['fiber'], subsequent_meals['glucose_excursion'])
    
    print(f"Fiber effectiveness comparison:")
    print(f"  • First meals: r = {first_fiber_corr[0]:.3f}, p = {first_fiber_corr[1]:.3f} (n={len(first_meals)})")
    print(f"  • Subsequent meals: r = {subsequent_fiber_corr[0]:.3f}, p = {subsequent_fiber_corr[1]:.3f} (n={len(subsequent_meals)})")
    
    # High-fiber first meal strategy
    high_fiber_first = first_meals[first_meals['fiber'] > first_meals['fiber'].median()]
    low_fiber_first = first_meals[first_meals['fiber'] <= first_meals['fiber'].median()]
    
    if len(high_fiber_first) > 10 and len(low_fiber_first) > 10:
        first_meal_fiber_benefit = (low_fiber_first['glucose_excursion'].mean() - 
                                   high_fiber_first['glucose_excursion'].mean())
        
        print(f"\nFirst meal fiber strategy:")
        print(f"  • High fiber first meals: {high_fiber_first['glucose_excursion'].mean():.1f} mg/dL excursion")
        print(f"  • Low fiber first meals: {low_fiber_first['glucose_excursion'].mean():.1f} mg/dL excursion")
        print(f"  • First meal fiber benefit: {first_meal_fiber_benefit:.1f} mg/dL")
        
        if first_meal_fiber_benefit > 5:
            print(f"  ✅ HIGH-FIBER FIRST MEAL STRATEGY EFFECTIVE")
    
    # 6. DIABETIC STATUS FIBER-TIMING INTERACTIONS
    print(f"\n🏥 ANALYSIS 6: Diabetic Status Fiber-Timing Interactions")
    print("-" * 55)
    
    for status in ['Normal', 'Pre-diabetic', 'Type2Diabetic']:
        if status in data['diabetic_status'].values:
            status_data = data[data['diabetic_status'] == status]
            
            print(f"\n{status} individuals ({len(status_data)} meals):")
            
            # Overall fiber effectiveness
            status_fiber_corr = stats.pearsonr(status_data['fiber'], status_data['glucose_excursion'])
            print(f"  • Overall fiber correlation: r = {status_fiber_corr[0]:.3f}, p = {status_fiber_corr[1]:.3f}")
            
            # Time-specific fiber effectiveness
            for time_cat in ['Early (0-8)', 'Morning (8-12)', 'Afternoon (12-17)', 'Evening (17-24)']:
                time_status_data = status_data[status_data['time_category'] == time_cat]
                if len(time_status_data) > 15:
                    time_fiber_corr = stats.pearsonr(time_status_data['fiber'], 
                                                   time_status_data['glucose_excursion'])
                    print(f"    {time_cat}: r = {time_fiber_corr[0]:.3f}, p = {time_fiber_corr[1]:.3f} (n={len(time_status_data)})")
    
    # 7. PRACTICAL FIBER-TIMING RECOMMENDATIONS
    print(f"\n💡 PRACTICAL FIBER-TIMING RECOMMENDATIONS")
    print("-" * 50)
    
    print("🌅 **Dawn Phenomenon Mitigation:**")
    if 'dawn_mitigation' in locals() and dawn_mitigation > 5:
        print(f"  • Increase fiber by {high_fiber_dawn['fiber'].mean() - low_fiber_dawn['fiber'].mean():.1f}g for early breakfast")
        print(f"  • Target fiber-carb ratio >0.15 for dawn meals")
        print(f"  • Expect {dawn_mitigation:.1f} mg/dL reduction in glucose spike")
    else:
        print(f"  • Dawn phenomenon shows limited fiber responsiveness")
        print(f"  • Focus on meal timing and composition over fiber alone")
    
    print("\n⏰ **Time-Specific Fiber Strategies:**")
    if len(significant_hours) > 0:
        best_times = significant_hours[significant_hours['correlation'] < -0.2]  # Strong negative correlation
        if len(best_times) > 0:
            print(f"  • Fiber most effective at: {list(best_times['hour'])} hours")
            print(f"  • Consider high-fiber meals during these windows")
    
    print("\n🥇 **First Meal Strategy:**")
    if 'first_meal_fiber_benefit' in locals() and first_meal_fiber_benefit > 5:
        print(f"  • Prioritize fiber in first meal: {first_meal_fiber_benefit:.1f} mg/dL benefit")
        print(f"  • Target ≥{high_fiber_first['fiber'].mean():.0f}g fiber for first meal")
    
    print("\n🍽️ **Meal Type Fiber Allocation:**")
    meal_fiber_priority = meal_timing_fiber.reset_index()
    meal_fiber_priority = meal_fiber_priority[meal_fiber_priority['fiber_glucose_p'] < 0.05]
    if len(meal_fiber_priority) > 0:
        best_meal_timing = meal_fiber_priority.loc[meal_fiber_priority['fiber_glucose_corr'].idxmin()]
        print(f"  • Most fiber-responsive: {best_meal_timing['meal_type']} during {best_meal_timing['time_category']}")
        print(f"  • Correlation: r = {best_meal_timing['fiber_glucose_corr']:.3f}")
    
    return data

if __name__ == "__main__":
    result_data = advanced_fiber_timing_analysis()