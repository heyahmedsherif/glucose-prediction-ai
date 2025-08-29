#!/usr/bin/env python3
"""
Meal Timing Recovery Analysis

Analyzes why breakfast meals take longer to return to baseline than lunch/dinner meals
by examining circadian rhythm effects, dawn phenomenon, and first meal impacts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

def load_cgmacros_with_meal_timing(max_participants=15):
    """Load CGMacros data and identify meal timing patterns."""
    
    data_files = glob.glob("CGMacros/*/CGMacros-*.csv")[:max_participants]
    print(f"Analyzing meal timing effects from {len(data_files)} participants")
    
    all_data = []
    for file in data_files:
        try:
            df = pd.read_csv(file)
            participant_id = file.split('/')[-1].split('.')[0]
            df['participant_id'] = participant_id
            
            # Process glucose and timing data
            df['glucose_value'] = df['Libre GL'].fillna(df['Dexcom GL'])
            df['timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.sort_values('timestamp')
            df['time_minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
            
            # Extract hour of day for circadian analysis
            df['hour_of_day'] = df['timestamp'].dt.hour
            
            # Filter valid glucose data and sample for speed
            df = df.dropna(subset=['glucose_value']).iloc[::5].reset_index(drop=True)
            
            all_data.append(df)
            print(f"Loaded {participant_id}: {len(df)} records")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset: {len(combined_data)} total records")
    
    return combined_data

def classify_meal_timing(hour):
    """Classify meals by timing category."""
    if 6 <= hour <= 10:
        return 'breakfast'
    elif 11 <= hour <= 15:
        return 'lunch'  
    elif 18 <= hour <= 22:
        return 'dinner'
    else:
        return 'other'

def analyze_meal_timing_recovery(df):
    """Analyze recovery times by meal timing with circadian effects."""
    
    # Classify diabetic status
    participant_stats = df.groupby('participant_id').agg({
        'glucose_value': ['mean', 'std', 'min', 'max']
    }).round(2)
    participant_stats.columns = ['glucose_mean', 'glucose_std', 'glucose_min', 'glucose_max']
    participant_stats = participant_stats.reset_index()
    
    def classify_status(mean_glucose):
        if mean_glucose < 100:
            return 'Normal'
        elif mean_glucose < 126:
            return 'Pre-diabetic'
        else:
            return 'Type2Diabetic'
    
    participant_stats['diabetic_status'] = participant_stats['glucose_mean'].apply(classify_status)
    
    # Merge status back to main data
    df_with_status = df.merge(participant_stats[['participant_id', 'diabetic_status', 'glucose_mean']], 
                              on='participant_id', how='left')
    
    meal_episodes = []
    
    for participant_id in df_with_status['participant_id'].unique():
        participant_data = df_with_status[df_with_status['participant_id'] == participant_id].copy()
        participant_data = participant_data.sort_values('time_minutes')
        
        if len(participant_data) < 20:
            continue
        
        # Calculate baseline (rolling median)
        participant_data['baseline_glucose'] = participant_data['glucose_value'].rolling(
            window=30, min_periods=10).median().fillna(participant_data['glucose_value'].median())
        
        # Identify meal episodes (glucose rise > 15 mg/dL to catch more meals)
        glucose_rise = participant_data['glucose_value'] - participant_data['baseline_glucose']
        meal_starts = participant_data[glucose_rise > 15].copy()
        
        for _, meal_start in meal_starts.iterrows():
            start_time = meal_start['time_minutes']
            start_glucose = meal_start['glucose_value']
            baseline = meal_start['baseline_glucose']
            hour_of_day = meal_start['hour_of_day']
            meal_timing = classify_meal_timing(hour_of_day)
            diabetic_status = meal_start['diabetic_status']
            
            # Skip non-meal times
            if meal_timing == 'other':
                continue
            
            # Track recovery over 6 hours
            recovery_window = participant_data[
                (participant_data['time_minutes'] >= start_time) & 
                (participant_data['time_minutes'] <= start_time + 360)
            ].copy()
            
            if len(recovery_window) < 10:
                continue
            
            # Find peak glucose in first 3 hours
            peak_data = recovery_window[recovery_window['time_minutes'] <= start_time + 180]
            if peak_data.empty:
                continue
            
            peak_glucose = peak_data['glucose_value'].max()
            peak_time_idx = peak_data['glucose_value'].idxmax()
            peak_time = recovery_window.loc[peak_time_idx, 'time_minutes']
            time_to_peak = peak_time - start_time
            
            # Find baseline recovery time
            baseline_threshold = 10
            recovery_time = None
            baseline_recovered = False
            
            # Look for recovery after peak
            post_peak_data = recovery_window[recovery_window['time_minutes'] > peak_time]
            
            for _, row in post_peak_data.iterrows():
                if abs(row['glucose_value'] - baseline) <= baseline_threshold:
                    recovery_time = row['time_minutes'] - start_time
                    baseline_recovered = True
                    break
            
            # If no recovery, use final time
            if not baseline_recovered and not recovery_window.empty:
                recovery_time = recovery_window.iloc[-1]['time_minutes'] - start_time
            
            glucose_rise_amount = peak_glucose - baseline
            
            # Calculate circadian factors
            is_dawn_period = 6 <= hour_of_day <= 9
            is_morning = 6 <= hour_of_day <= 12
            is_first_meal_likely = hour_of_day <= 11  # Approximate first meal
            
            meal_episodes.append({
                'participant_id': participant_id,
                'diabetic_status': diabetic_status,
                'meal_timing': meal_timing,
                'hour_of_day': hour_of_day,
                'baseline_glucose': baseline,
                'peak_glucose': peak_glucose,
                'glucose_rise': glucose_rise_amount,
                'time_to_peak_minutes': time_to_peak,
                'recovery_time_minutes': recovery_time,
                'baseline_recovered': baseline_recovered,
                'is_dawn_period': is_dawn_period,
                'is_morning': is_morning,
                'is_first_meal_likely': is_first_meal_likely
            })
    
    return pd.DataFrame(meal_episodes)

def analyze_timing_effects(meal_episodes_df):
    """Analyze why breakfast takes longer to recover."""
    
    print("\n" + "="*80)
    print("📊 MEAL TIMING RECOVERY ANALYSIS")
    print("="*80)
    
    if meal_episodes_df.empty:
        print("No meal episodes found for analysis.")
        return
    
    print(f"Total meal episodes analyzed: {len(meal_episodes_df)}")
    
    # Recovery time by meal timing
    print(f"\n⏰ RECOVERY TIME BY MEAL TIMING:")
    print("-" * 80)
    
    timing_stats = meal_episodes_df[meal_episodes_df['baseline_recovered'] == True].groupby('meal_timing').agg({
        'recovery_time_minutes': ['count', 'mean', 'median', 'std'],
        'time_to_peak_minutes': ['mean', 'median'],
        'glucose_rise': ['mean', 'median'],
        'baseline_glucose': ['mean', 'median']
    }).round(1)
    
    timing_summary = []
    
    for meal_type in ['breakfast', 'lunch', 'dinner']:
        if meal_type not in timing_stats.index:
            continue
        
        count = int(timing_stats.loc[meal_type, ('recovery_time_minutes', 'count')])
        avg_recovery = timing_stats.loc[meal_type, ('recovery_time_minutes', 'mean')]
        median_recovery = timing_stats.loc[meal_type, ('recovery_time_minutes', 'median')]
        std_recovery = timing_stats.loc[meal_type, ('recovery_time_minutes', 'std')]
        
        avg_peak_time = timing_stats.loc[meal_type, ('time_to_peak_minutes', 'mean')]
        avg_glucose_rise = timing_stats.loc[meal_type, ('glucose_rise', 'mean')]
        avg_baseline = timing_stats.loc[meal_type, ('baseline_glucose', 'mean')]
        
        print(f"\n🔸 {meal_type.upper()} (n={count} episodes):")
        print(f"  Average recovery time: {avg_recovery:.1f} minutes ({avg_recovery/60:.1f} hours)")
        print(f"  Median recovery time: {median_recovery:.1f} minutes ({median_recovery/60:.1f} hours)")
        print(f"  Recovery time variability: ±{std_recovery:.1f} minutes")
        print(f"  Average time to peak: {avg_peak_time:.1f} minutes")
        print(f"  Average glucose rise: {avg_glucose_rise:.1f} mg/dL")
        print(f"  Average baseline glucose: {avg_baseline:.1f} mg/dL")
        
        timing_summary.append((meal_type, avg_recovery))
    
    # Ranking
    print(f"\n📈 RECOVERY TIME RANKING:")
    print("-" * 80)
    timing_summary.sort(key=lambda x: x[1])
    
    print("Ranking by recovery time (fastest to slowest):")
    for i, (meal_type, time) in enumerate(timing_summary, 1):
        print(f"  {i}. {meal_type.capitalize()}: {time:.1f} minutes ({time/60:.1f} hours)")
    
    # Calculate differences
    breakfast_time = next(t for m, t in timing_summary if m == 'breakfast')
    lunch_time = next((t for m, t in timing_summary if m == 'lunch'), None)
    dinner_time = next((t for m, t in timing_summary if m == 'dinner'), None)
    
    if lunch_time:
        diff_lunch = breakfast_time - lunch_time
        print(f"\n⚡ Breakfast takes {diff_lunch:.1f} minutes ({diff_lunch/60:.1f} hours) longer than lunch")
    
    if dinner_time:
        diff_dinner = breakfast_time - dinner_time
        print(f"⚡ Breakfast takes {diff_dinner:.1f} minutes ({diff_dinner/60:.1f} hours) longer than dinner")
    
    # Analyze underlying causes
    print(f"\n🔬 UNDERLYING PHYSIOLOGICAL CAUSES:")
    print("-" * 80)
    
    # Dawn phenomenon analysis
    dawn_episodes = meal_episodes_df[meal_episodes_df['is_dawn_period'] == True]
    non_dawn_episodes = meal_episodes_df[meal_episodes_df['is_dawn_period'] == False]
    
    if not dawn_episodes.empty and not non_dawn_episodes.empty:
        dawn_recovery = dawn_episodes[dawn_episodes['baseline_recovered']]['recovery_time_minutes'].mean()
        non_dawn_recovery = non_dawn_episodes[non_dawn_episodes['baseline_recovered']]['recovery_time_minutes'].mean()
        dawn_baseline = dawn_episodes['baseline_glucose'].mean()
        non_dawn_baseline = non_dawn_episodes['baseline_glucose'].mean()
        
        print(f"🌅 DAWN PHENOMENON EFFECT:")
        print(f"  Dawn period (6-9 AM) recovery: {dawn_recovery:.1f} minutes")
        print(f"  Non-dawn period recovery: {non_dawn_recovery:.1f} minutes")
        print(f"  Dawn phenomenon delay: +{dawn_recovery - non_dawn_recovery:.1f} minutes")
        print(f"  Dawn period baseline glucose: {dawn_baseline:.1f} mg/dL")
        print(f"  Non-dawn baseline glucose: {non_dawn_baseline:.1f} mg/dL")
        print(f"  Dawn phenomenon baseline elevation: +{dawn_baseline - non_dawn_baseline:.1f} mg/dL")
    
    # First meal effect
    first_meal_episodes = meal_episodes_df[meal_episodes_df['is_first_meal_likely'] == True]
    later_meal_episodes = meal_episodes_df[meal_episodes_df['is_first_meal_likely'] == False]
    
    if not first_meal_episodes.empty and not later_meal_episodes.empty:
        first_recovery = first_meal_episodes[first_meal_episodes['baseline_recovered']]['recovery_time_minutes'].mean()
        later_recovery = later_meal_episodes[later_meal_episodes['baseline_recovered']]['recovery_time_minutes'].mean()
        first_rise = first_meal_episodes['glucose_rise'].mean()
        later_rise = later_meal_episodes['glucose_rise'].mean()
        
        print(f"\n🍽️ FIRST MEAL EFFECT:")
        print(f"  First meal recovery: {first_recovery:.1f} minutes")
        print(f"  Later meal recovery: {later_recovery:.1f} minutes")
        print(f"  First meal delay: +{first_recovery - later_recovery:.1f} minutes")
        print(f"  First meal glucose rise: {first_rise:.1f} mg/dL")
        print(f"  Later meal glucose rise: {later_rise:.1f} mg/dL")
        print(f"  First meal higher spike: +{first_rise - later_rise:.1f} mg/dL")
    
    # Circadian insulin sensitivity
    print(f"\n🔄 CIRCADIAN INSULIN SENSITIVITY:")
    print("-" * 80)
    
    hourly_stats = meal_episodes_df.groupby('hour_of_day').agg({
        'recovery_time_minutes': 'mean',
        'glucose_rise': 'mean',
        'baseline_glucose': 'mean'
    }).round(1)
    
    print("Average recovery time by hour of day:")
    for hour in range(6, 23):
        if hour in hourly_stats.index:
            recovery = hourly_stats.loc[hour, 'recovery_time_minutes']
            rise = hourly_stats.loc[hour, 'glucose_rise']
            baseline = hourly_stats.loc[hour, 'baseline_glucose']
            print(f"  {hour:2d}:00 - Recovery: {recovery:5.1f} min, Rise: {rise:5.1f} mg/dL, Baseline: {baseline:5.1f} mg/dL")
    
    return meal_episodes_df

def create_timing_visualizations(meal_episodes_df):
    """Create visualizations for meal timing effects."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Meal Timing Effects on Glucose Recovery', fontsize=16, fontweight='bold')
    
    # 1. Recovery time by meal timing
    ax1 = axes[0, 0]
    recovered_episodes = meal_episodes_df[meal_episodes_df['baseline_recovered'] == True]
    
    meal_order = ['breakfast', 'lunch', 'dinner']
    recovery_data = []
    labels = []
    
    for meal_type in meal_order:
        data = recovered_episodes[recovered_episodes['meal_timing'] == meal_type]['recovery_time_minutes']
        if not data.empty:
            recovery_data.append(data)
            labels.append(meal_type.capitalize())
    
    if recovery_data:
        ax1.boxplot(recovery_data, labels=labels)
        ax1.set_ylabel('Recovery Time (minutes)')
        ax1.set_title('Recovery Time Distribution by Meal Timing')
        ax1.grid(True, alpha=0.3)
    
    # 2. Hourly recovery pattern
    ax2 = axes[0, 1]
    hourly_recovery = meal_episodes_df[meal_episodes_df['baseline_recovered'] == True].groupby('hour_of_day')['recovery_time_minutes'].mean()
    
    if not hourly_recovery.empty:
        ax2.plot(hourly_recovery.index, hourly_recovery.values, 'o-', linewidth=2, markersize=6)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Recovery Time (minutes)')
        ax2.set_title('Recovery Time by Hour of Day')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(6, 22)
    
    # 3. Glucose rise by meal timing
    ax3 = axes[1, 0]
    rise_data = []
    for meal_type in meal_order:
        data = meal_episodes_df[meal_episodes_df['meal_timing'] == meal_type]['glucose_rise']
        if not data.empty:
            rise_data.append(data)
    
    if rise_data:
        ax3.boxplot(rise_data, labels=labels)
        ax3.set_ylabel('Glucose Rise (mg/dL)')
        ax3.set_title('Glucose Rise Distribution by Meal Timing')
        ax3.grid(True, alpha=0.3)
    
    # 4. Dawn phenomenon effect
    ax4 = axes[1, 1]
    dawn_data = meal_episodes_df[meal_episodes_df['is_dawn_period'] == True]
    non_dawn_data = meal_episodes_df[meal_episodes_df['is_dawn_period'] == False]
    
    if not dawn_data.empty and not non_dawn_data.empty:
        dawn_recovery = dawn_data[dawn_data['baseline_recovered']]['recovery_time_minutes']
        non_dawn_recovery = non_dawn_data[non_dawn_data['baseline_recovered']]['recovery_time_minutes']
        
        ax4.boxplot([dawn_recovery, non_dawn_recovery], labels=['Dawn Period\n(6-9 AM)', 'Other Times'])
        ax4.set_ylabel('Recovery Time (minutes)')
        ax4.set_title('Dawn Phenomenon Effect on Recovery')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('meal_timing_recovery_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Visualization saved as 'meal_timing_recovery_analysis.png'")

def main():
    """Main analysis function."""
    
    print("🔬 Loading CGMacros Data for Meal Timing Recovery Analysis...")
    
    # Load data
    df = load_cgmacros_with_meal_timing()
    if df is None:
        return
    
    # Analyze meal timing effects
    print(f"\n🔍 Analyzing meal timing recovery patterns...")
    meal_episodes_df = analyze_meal_timing_recovery(df)
    
    if meal_episodes_df.empty:
        print("❌ No meal episodes found.")
        return
    
    # Generate timing analysis
    meal_episodes_df = analyze_timing_effects(meal_episodes_df)
    
    # Create visualizations
    try:
        create_timing_visualizations(meal_episodes_df)
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")
    
    # Save results
    meal_episodes_df.to_csv('meal_timing_recovery_episodes.csv', index=False)
    print(f"\n💾 Results saved to 'meal_timing_recovery_episodes.csv'")

if __name__ == "__main__":
    main()