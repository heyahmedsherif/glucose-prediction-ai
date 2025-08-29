#!/usr/bin/env python3
"""
Baseline Recovery Analysis
Analyze how likely each diabetic status group is to return to baseline by 180 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

def load_all_cgmacros_data(max_participants=15):
    """Load and combine CGMacros datasets (limited for faster analysis)."""
    
    # Find all CSV files in CGMacros directories
    data_files = glob.glob("CGMacros/*/CGMacros-*.csv")
    
    if not data_files:
        print("No CGMacros data files found. Please check the data directory.")
        return None
    
    # Limit to subset for faster analysis
    data_files = data_files[:max_participants]
    print(f"Analyzing {len(data_files)} participants for faster processing")
    
    all_data = []
    for file in data_files:
        try:
            df = pd.read_csv(file)
            participant_id = file.split('/')[-1].split('.')[0]  # Extract participant ID from filename
            df['participant_id'] = participant_id
            
            # Rename columns to standardized names and add time processing
            df['glucose_value'] = df['Libre GL'].fillna(df['Dexcom GL'])  # Use Libre GL primarily
            df['timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Calculate time in minutes from start of participant data
            df = df.sort_values('timestamp')
            df['time_minutes'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
            
            # Filter out rows with missing glucose values and sample for speed
            df = df.dropna(subset=['glucose_value'])
            
            # Sample every 5th row for faster processing (still 3-minute intervals)
            df = df.iloc[::5].reset_index(drop=True)
            
            all_data.append(df)
            print(f"Loaded {participant_id}: {len(df)} records")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        return None
    
    # Combine all datasets
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_data)} total records from {len(data_files)} participants")
    
    return combined_data

def classify_diabetic_status(df):
    """Classify participants by diabetic status based on glucose patterns."""
    
    # Calculate baseline glucose statistics for each participant
    participant_stats = df.groupby('participant_id').agg({
        'glucose_value': ['mean', 'std', 'min', 'max'],
        'time_minutes': 'count'
    }).round(2)
    
    participant_stats.columns = ['glucose_mean', 'glucose_std', 'glucose_min', 'glucose_max', 'record_count']
    participant_stats = participant_stats.reset_index()
    
    # Diabetic status classification based on mean glucose levels
    def classify_status(mean_glucose):
        if mean_glucose < 100:
            return 'Normal'
        elif mean_glucose < 126:
            return 'Pre-diabetic'
        else:
            return 'Type2Diabetic'
    
    participant_stats['diabetic_status'] = participant_stats['glucose_mean'].apply(classify_status)
    
    print(f"\nDiabetic Status Distribution:")
    status_counts = participant_stats['diabetic_status'].value_counts()
    for status, count in status_counts.items():
        percentage = (count / len(participant_stats)) * 100
        print(f"  {status}: {count} participants ({percentage:.1f}%)")
    
    return participant_stats

def analyze_baseline_recovery_with_timing(df, participant_stats):
    """Analyze baseline recovery patterns with actual recovery time tracking."""
    
    # Merge diabetic status into main dataset
    df_with_status = df.merge(participant_stats[['participant_id', 'diabetic_status', 'glucose_mean']], 
                              on='participant_id', how='left')
    
    # Focus on meal-related glucose episodes
    meal_episodes = []
    
    for participant_id in df_with_status['participant_id'].unique():
        participant_data = df_with_status[df_with_status['participant_id'] == participant_id].copy()
        participant_data = participant_data.sort_values('time_minutes')
        
        if len(participant_data) < 10:  # Skip participants with too little data
            continue
        
        # Calculate rolling baseline (median of previous 30 readings)
        participant_data['baseline_glucose'] = participant_data['glucose_value'].rolling(
            window=30, min_periods=10).median().fillna(participant_data['glucose_value'].median())
        
        # Identify potential meal start points (glucose rise > 20 mg/dL from baseline)
        glucose_rise = participant_data['glucose_value'] - participant_data['baseline_glucose']
        meal_starts = participant_data[glucose_rise > 20].copy()
        
        for _, meal_start in meal_starts.iterrows():
            start_time = meal_start['time_minutes']
            start_glucose = meal_start['glucose_value']
            baseline = meal_start['baseline_glucose']
            diabetic_status = meal_start['diabetic_status']
            
            # Track recovery over extended time periods (up to 6 hours)
            recovery_window = participant_data[
                (participant_data['time_minutes'] >= start_time) & 
                (participant_data['time_minutes'] <= start_time + 360)  # 6 hours
            ].copy()
            
            if len(recovery_window) < 5:
                continue
            
            # Find peak glucose in first 3 hours
            peak_data = recovery_window[recovery_window['time_minutes'] <= start_time + 180]
            if peak_data.empty:
                continue
            
            peak_glucose = peak_data['glucose_value'].max()
            peak_time_idx = peak_data['glucose_value'].idxmax()
            peak_time = recovery_window.loc[peak_time_idx, 'time_minutes']
            time_to_peak = peak_time - start_time
            
            # Find when glucose returns to baseline (±10 mg/dL)
            baseline_threshold = 10
            recovery_time = None
            recovery_glucose = None
            baseline_recovered = False
            
            # Look for recovery after peak
            post_peak_data = recovery_window[recovery_window['time_minutes'] > peak_time]
            
            for _, row in post_peak_data.iterrows():
                if abs(row['glucose_value'] - baseline) <= baseline_threshold:
                    recovery_time = row['time_minutes'] - start_time
                    recovery_glucose = row['glucose_value']
                    baseline_recovered = True
                    break
            
            # If no baseline recovery found, record the final glucose and time
            if not baseline_recovered and not recovery_window.empty:
                final_row = recovery_window.iloc[-1]
                recovery_time = final_row['time_minutes'] - start_time
                recovery_glucose = final_row['glucose_value']
            
            # Check recovery status at standard time points
            recovery_180min = None
            recovery_240min = None
            recovery_360min = None
            
            # 180 minutes
            data_180 = recovery_window[
                (recovery_window['time_minutes'] >= start_time + 165) & 
                (recovery_window['time_minutes'] <= start_time + 195)
            ]
            if not data_180.empty:
                closest_180_idx = (data_180['time_minutes'] - (start_time + 180)).abs().idxmin()
                glucose_180 = data_180.loc[closest_180_idx, 'glucose_value']
                recovery_180min = abs(glucose_180 - baseline) <= baseline_threshold
            
            # 240 minutes (4 hours)
            data_240 = recovery_window[
                (recovery_window['time_minutes'] >= start_time + 225) & 
                (recovery_window['time_minutes'] <= start_time + 255)
            ]
            if not data_240.empty:
                closest_240_idx = (data_240['time_minutes'] - (start_time + 240)).abs().idxmin()
                glucose_240 = data_240.loc[closest_240_idx, 'glucose_value']
                recovery_240min = abs(glucose_240 - baseline) <= baseline_threshold
            
            # 360 minutes (6 hours)
            data_360 = recovery_window[
                (recovery_window['time_minutes'] >= start_time + 345) & 
                (recovery_window['time_minutes'] <= start_time + 375)
            ]
            if not data_360.empty:
                closest_360_idx = (data_360['time_minutes'] - (start_time + 360)).abs().idxmin()
                glucose_360 = data_360.loc[closest_360_idx, 'glucose_value']
                recovery_360min = abs(glucose_360 - baseline) <= baseline_threshold
            
            glucose_rise_amount = peak_glucose - baseline
            recovery_amount = peak_glucose - (recovery_glucose or peak_glucose)
            recovery_percentage = (recovery_amount / glucose_rise_amount * 100) if glucose_rise_amount > 0 else 0
            
            meal_episodes.append({
                'participant_id': participant_id,
                'diabetic_status': diabetic_status,
                'baseline_glucose': baseline,
                'peak_glucose': peak_glucose,
                'time_to_peak_minutes': time_to_peak,
                'recovery_glucose': recovery_glucose,
                'glucose_rise': glucose_rise_amount,
                'recovery_amount': recovery_amount,
                'recovery_percentage': recovery_percentage,
                'baseline_recovered': baseline_recovered,
                'recovery_time_minutes': recovery_time,
                'recovery_180min': recovery_180min,
                'recovery_240min': recovery_240min,
                'recovery_360min': recovery_360min
            })
    
    return pd.DataFrame(meal_episodes)

def generate_recovery_analysis(meal_episodes_df):
    """Generate comprehensive recovery analysis."""
    
    print("\n" + "="*60)
    print("📊 BASELINE RECOVERY ANALYSIS (180-MINUTE POST-MEAL)")
    print("="*60)
    
    if meal_episodes_df.empty:
        print("No meal episodes found for analysis.")
        return
    
    # Overall statistics
    total_episodes = len(meal_episodes_df)
    print(f"Total meal episodes analyzed: {total_episodes}")
    
    # Recovery analysis by diabetic status
    recovery_stats = meal_episodes_df.groupby('diabetic_status').agg({
        'baseline_recovered': ['count', 'sum', 'mean'],
        'recovery_percentage': ['mean', 'median', 'std'],
        'glucose_rise': ['mean', 'median'],
        'recovery_glucose': ['mean', 'median']
    }).round(2)
    
    print(f"\n⏱️  AVERAGE RECOVERY TIMES BY DIABETIC STATUS:")
    print("-" * 70)
    
    recovery_summary = []
    
    for status in ['Normal', 'Pre-diabetic', 'Type2Diabetic']:
        status_data = meal_episodes_df[meal_episodes_df['diabetic_status'] == status]
        if status_data.empty:
            continue
        
        episode_count = len(status_data)
        
        # Calculate recovery statistics
        baseline_recovered = status_data['baseline_recovered'].sum()
        recovery_rate = (baseline_recovered / episode_count) * 100
        
        # Time to peak
        avg_peak_time = status_data['time_to_peak_minutes'].mean()
        
        # Recovery time for those who recovered
        recovered_episodes = status_data[status_data['baseline_recovered'] == True]
        if not recovered_episodes.empty:
            avg_recovery_time = recovered_episodes['recovery_time_minutes'].mean()
            median_recovery_time = recovered_episodes['recovery_time_minutes'].median()
            recovery_time_std = recovered_episodes['recovery_time_minutes'].std()
        else:
            avg_recovery_time = None
            median_recovery_time = None
            recovery_time_std = None
        
        # Recovery rates at different time points
        recovery_180 = status_data['recovery_180min'].sum() / status_data['recovery_180min'].count() * 100 if status_data['recovery_180min'].count() > 0 else 0
        recovery_240 = status_data['recovery_240min'].sum() / status_data['recovery_240min'].count() * 100 if status_data['recovery_240min'].count() > 0 else 0
        recovery_360 = status_data['recovery_360min'].sum() / status_data['recovery_360min'].count() * 100 if status_data['recovery_360min'].count() > 0 else 0
        
        print(f"\n🔸 {status.upper()} (n={episode_count} episodes):")
        print(f"  Overall baseline recovery rate: {recovery_rate:.1f}% ({baseline_recovered}/{episode_count})")
        print(f"  Average time to glucose peak: {avg_peak_time:.1f} minutes ({avg_peak_time/60:.1f} hours)")
        
        if avg_recovery_time is not None:
            print(f"  Average time to baseline recovery: {avg_recovery_time:.1f} minutes ({avg_recovery_time/60:.1f} hours)")
            print(f"  Median time to baseline recovery: {median_recovery_time:.1f} minutes ({median_recovery_time/60:.1f} hours)")
            print(f"  Recovery time std deviation: ±{recovery_time_std:.1f} minutes")
            recovery_summary.append((status, avg_recovery_time))
        else:
            print(f"  Average time to baseline recovery: No episodes achieved full recovery")
            recovery_summary.append((status, None))
        
        print(f"  Recovery rates by time point:")
        print(f"    • 3 hours (180 min): {recovery_180:.1f}%")
        print(f"    • 4 hours (240 min): {recovery_240:.1f}%") 
        print(f"    • 6 hours (360 min): {recovery_360:.1f}%")
    
    # Summary comparison
    print(f"\n📈 RECOVERY TIME RANKING:")
    print("-" * 70)
    
    # Sort by recovery time (fastest first)
    recovery_summary.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))
    
    print("Ranking by average time to return to baseline (fastest to slowest):")
    for i, (status, time) in enumerate(recovery_summary, 1):
        if time is not None:
            print(f"  {i}. {status}: {time:.1f} minutes ({time/60:.1f} hours) average")
        else:
            print(f"  {i}. {status}: Unable to achieve baseline recovery in observed timeframe")
    
    # Detailed recovery categorization
    print(f"\n📋 RECOVERY CATEGORIES:")
    print("-" * 60)
    
    def categorize_recovery(row):
        if row['baseline_recovered']:
            return 'Full Recovery (±10 mg/dL of baseline)'
        elif row['recovery_percentage'] >= 80:
            return 'Near Recovery (80%+ of rise recovered)'
        elif row['recovery_percentage'] >= 50:
            return 'Partial Recovery (50-79% recovered)'
        else:
            return 'Poor Recovery (<50% recovered)'
    
    meal_episodes_df['recovery_category'] = meal_episodes_df.apply(categorize_recovery, axis=1)
    
    recovery_breakdown = pd.crosstab(meal_episodes_df['diabetic_status'], 
                                    meal_episodes_df['recovery_category'], 
                                    normalize='index') * 100
    
    for status in recovery_breakdown.index:
        print(f"\n🔸 {status.upper()} Recovery Breakdown:")
        for category in recovery_breakdown.columns:
            percentage = recovery_breakdown.loc[status, category]
            print(f"  {category}: {percentage:.1f}%")
    
    # Statistical significance testing
    print(f"\n📊 STATISTICAL INSIGHTS:")
    print("-" * 60)
    
    normal_recovery = meal_episodes_df[meal_episodes_df['diabetic_status'] == 'Normal']['baseline_recovered'].mean() * 100
    prediabetic_recovery = meal_episodes_df[meal_episodes_df['diabetic_status'] == 'Pre-diabetic']['baseline_recovered'].mean() * 100
    diabetic_recovery = meal_episodes_df[meal_episodes_df['diabetic_status'] == 'Type2Diabetic']['baseline_recovered'].mean() * 100
    
    print(f"Baseline recovery likelihood ranking:")
    recovery_rates = [
        ('Normal', normal_recovery),
        ('Pre-diabetic', prediabetic_recovery), 
        ('Type2Diabetic', diabetic_recovery)
    ]
    
    # Sort by recovery rate
    recovery_rates.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 0, reverse=True)
    
    for i, (status, rate) in enumerate(recovery_rates, 1):
        if not np.isnan(rate):
            print(f"  {i}. {status}: {rate:.1f}% likely to return to baseline")
        else:
            print(f"  {i}. {status}: Insufficient data")
    
    # Generate summary recommendations
    print(f"\n🎯 KEY FINDINGS:")
    print("-" * 60)
    
    if not np.isnan(normal_recovery) and not np.isnan(diabetic_recovery):
        difference = normal_recovery - diabetic_recovery
        print(f"• Normal individuals are {difference:.1f} percentage points more likely")
        print(f"  to return to baseline than Type 2 diabetics")
    
    overall_recovery = meal_episodes_df['baseline_recovered'].mean() * 100
    print(f"• Overall baseline recovery rate: {overall_recovery:.1f}%")
    
    high_recovery = meal_episodes_df[meal_episodes_df['recovery_percentage'] >= 80]['diabetic_status'].value_counts(normalize=True) * 100
    print(f"• Participants with >80% recovery are predominantly:")
    for status, pct in high_recovery.head(2).items():
        print(f"  - {status}: {pct:.1f}%")
    
    return meal_episodes_df, recovery_stats

def create_visualizations(meal_episodes_df):
    """Create visualizations for recovery analysis."""
    
    if meal_episodes_df.empty:
        print("No data available for visualizations.")
        return
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Glucose Baseline Recovery Analysis by Diabetic Status', fontsize=16, fontweight='bold')
    
    # 1. Recovery likelihood by status
    ax1 = axes[0, 0]
    recovery_by_status = meal_episodes_df.groupby('diabetic_status')['baseline_recovered'].agg(['mean', 'count'])
    recovery_by_status['mean_pct'] = recovery_by_status['mean'] * 100
    
    bars = ax1.bar(recovery_by_status.index, recovery_by_status['mean_pct'], 
                   color=['#2E8B57', '#FF8C00', '#DC143C'], alpha=0.7)
    ax1.set_ylabel('Baseline Recovery Likelihood (%)')
    ax1.set_title('Likelihood of Returning to Baseline by 180 Minutes')
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, (status, row) in zip(bars, recovery_by_status.iterrows()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%\n(n={int(row["count"])})', 
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Recovery percentage distribution
    ax2 = axes[0, 1]
    for status in meal_episodes_df['diabetic_status'].unique():
        data = meal_episodes_df[meal_episodes_df['diabetic_status'] == status]['recovery_percentage']
        ax2.hist(data, alpha=0.6, label=status, bins=20)
    
    ax2.set_xlabel('Recovery Percentage (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Recovery Percentages')
    ax2.legend()
    ax2.axvline(x=80, color='red', linestyle='--', alpha=0.7, label='80% Recovery Line')
    
    # 3. Glucose rise vs recovery
    ax3 = axes[1, 0]
    colors = {'Normal': '#2E8B57', 'Pre-diabetic': '#FF8C00', 'Type2Diabetic': '#DC143C'}
    for status in meal_episodes_df['diabetic_status'].unique():
        data = meal_episodes_df[meal_episodes_df['diabetic_status'] == status]
        ax3.scatter(data['glucose_rise'], data['recovery_percentage'], 
                   alpha=0.6, c=colors.get(status, 'blue'), label=status)
    
    ax3.set_xlabel('Glucose Rise (mg/dL)')
    ax3.set_ylabel('Recovery Percentage (%)')
    ax3.set_title('Glucose Rise vs Recovery Percentage')
    ax3.legend()
    ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7)
    
    # 4. Recovery categories stacked bar
    ax4 = axes[1, 1]
    recovery_categories = pd.crosstab(meal_episodes_df['diabetic_status'], 
                                     meal_episodes_df['recovery_category'], 
                                     normalize='index') * 100
    
    recovery_categories.plot(kind='bar', stacked=True, ax=ax4, 
                            color=['#228B22', '#90EE90', '#FFD700', '#FF6347'])
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('Recovery Category Breakdown')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('baseline_recovery_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Visualization saved as 'baseline_recovery_analysis.png'")

def main():
    """Main analysis function."""
    
    print("🔬 Loading CGMacros Data for Baseline Recovery Analysis...")
    
    # Load data
    df = load_all_cgmacros_data()
    if df is None:
        return
    
    # Classify participants by diabetic status
    participant_stats = classify_diabetic_status(df)
    
    # Analyze baseline recovery patterns with timing
    print(f"\n🔍 Analyzing meal episodes and baseline recovery with timing...")
    meal_episodes_df = analyze_baseline_recovery_with_timing(df, participant_stats)
    
    if meal_episodes_df.empty:
        print("❌ No meal episodes found. This could be due to:")
        print("   • Insufficient glucose rise patterns in the data")
        print("   • Data time resolution issues")
        print("   • Participants not having clear meal patterns")
        return
    
    # Generate analysis
    meal_episodes_df, recovery_stats = generate_recovery_analysis(meal_episodes_df)
    
    # Create visualizations
    try:
        create_visualizations(meal_episodes_df)
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")
    
    # Save detailed results
    meal_episodes_df.to_csv('baseline_recovery_episodes.csv', index=False)
    recovery_stats.to_csv('recovery_statistics_by_status.csv')
    
    print(f"\n💾 Detailed results saved:")
    print(f"   • baseline_recovery_episodes.csv: Individual meal episodes")
    print(f"   • recovery_statistics_by_status.csv: Summary statistics")

if __name__ == "__main__":
    main()