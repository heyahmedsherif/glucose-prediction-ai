#!/usr/bin/env python3
"""
Meal Pattern Visualizations

Create comprehensive visualizations showing timing and sequence effects on glucose response.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_meal_pattern_visualizations():
    """Create comprehensive visualizations for meal pattern analysis."""
    
    print("📊 Creating Meal Pattern Visualizations")
    print("=" * 50)
    
    # Load and prepare data
    data = pd.read_csv("glucose_prediction_training_data_enhanced.csv")
    data['meal_datetime'] = pd.to_datetime(data['meal_timestamp'])
    data['hour'] = data['meal_datetime'].dt.hour
    data['day'] = data['meal_datetime'].dt.date
    
    # Calculate metrics
    glucose_cols = ['glucose_30min', 'glucose_60min', 'glucose_90min', 'glucose_120min', 'glucose_180min']
    data['peak_glucose'] = data[glucose_cols].max(axis=1)
    data['glucose_excursion'] = data['peak_glucose'] - data['baseline']
    
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
    
    # Create meal sequence
    data_sorted = data.sort_values(['subject_id', 'day', 'meal_datetime'])
    data_sorted['meal_sequence'] = data_sorted.groupby(['subject_id', 'day']).cumcount() + 1
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Hourly glucose response heatmap
    plt.subplot(3, 3, 1)
    hourly_response = data.groupby(['hour', 'meal_type'])['glucose_excursion'].mean().unstack()
    sns.heatmap(hourly_response.T, annot=True, fmt='.1f', cmap='RdYlBu_r', cbar_kws={'label': 'Glucose Excursion (mg/dL)'})
    plt.title('🕐 Glucose Response by Hour and Meal Type')
    plt.xlabel('Hour of Day')
    plt.ylabel('Meal Type')
    
    # 2. Meal type comparison
    plt.subplot(3, 3, 2)
    meal_data = []
    meal_labels = []
    for meal in ['breakfast', 'lunch', 'dinner']:
        meal_responses = data[data['meal_type'] == meal]['glucose_excursion']
        meal_data.append(meal_responses)
        meal_labels.append(f"{meal.title()}\n(n={len(meal_responses)})")
    
    bp = plt.boxplot(meal_data, labels=meal_labels, patch_artist=True)
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('🍽️ Glucose Excursion by Meal Type')
    plt.ylabel('Glucose Excursion (mg/dL)')
    plt.xticks(rotation=45)
    
    # 3. First meal vs subsequent meals
    plt.subplot(3, 3, 3)
    first_meal_data = data[data['is_first_meal'] == True]['glucose_excursion']
    subsequent_data = data[data['is_first_meal'] == False]['glucose_excursion']
    
    bp = plt.boxplot([first_meal_data, subsequent_data], 
                     labels=[f'First Meal\n(n={len(first_meal_data)})', 
                            f'Subsequent\n(n={len(subsequent_data)})'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('orange')
    bp['boxes'][1].set_facecolor('lightblue')
    
    plt.title('🌅 First Meal vs Subsequent Meals')
    plt.ylabel('Glucose Excursion (mg/dL)')
    
    # 4. Dawn phenomenon analysis
    plt.subplot(3, 3, 4)
    breakfast_data = data[data['meal_type'] == 'breakfast']
    breakfast_data['time_category'] = breakfast_data['hour'].apply(
        lambda x: 'Early (6-9 AM)' if 6 <= x <= 9 else 'Other Times'
    )
    
    dawn_categories = []
    dawn_data = []
    for cat in ['Early (6-9 AM)', 'Other Times']:
        cat_data = breakfast_data[breakfast_data['time_category'] == cat]['glucose_excursion']
        if len(cat_data) > 0:
            dawn_data.append(cat_data)
            dawn_categories.append(f'{cat}\n(n={len(cat_data)})')
    
    if len(dawn_data) > 1:
        bp = plt.boxplot(dawn_data, labels=dawn_categories, patch_artist=True)
        bp['boxes'][0].set_facecolor('gold')
        bp['boxes'][1].set_facecolor('lightblue')
    
    plt.title('🌅 Dawn Phenomenon Analysis')
    plt.ylabel('Glucose Excursion (mg/dL)')
    plt.xticks(rotation=45)
    
    # 5. Meal sequence effects
    plt.subplot(3, 3, 5)
    sequence_means = data_sorted.groupby('meal_sequence')['glucose_excursion'].agg(['mean', 'std'])
    sequence_means = sequence_means[sequence_means.index <= 5]  # Only first 5 meals
    
    x = sequence_means.index
    y = sequence_means['mean']
    yerr = sequence_means['std']
    
    plt.errorbar(x, y, yerr=yerr, marker='o', capsize=5, linewidth=2, markersize=8)
    plt.xlabel('Meal Sequence (within day)')
    plt.ylabel('Glucose Excursion (mg/dL)')
    plt.title('📝 Meal Sequence Effects')
    plt.grid(True, alpha=0.3)
    
    # 6. Diabetic status by meal type
    plt.subplot(3, 3, 6)
    diabetic_meal = data.pivot_table(values='glucose_excursion', 
                                   index='meal_type', 
                                   columns='diabetic_status', 
                                   aggfunc='mean')
    
    diabetic_meal.plot(kind='bar', ax=plt.gca(), width=0.8)
    plt.title('🏥 Response by Diabetic Status & Meal Type')
    plt.ylabel('Glucose Excursion (mg/dL)')
    plt.xlabel('Meal Type')
    plt.legend(title='Diabetic Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    # 7. Hourly pattern by diabetic status
    plt.subplot(3, 3, 7)
    for status in ['Normal', 'Pre-diabetic', 'Type2Diabetic']:
        status_data = data[data['diabetic_status'] == status]
        hourly_pattern = status_data.groupby('hour')['glucose_excursion'].mean()
        plt.plot(hourly_pattern.index, hourly_pattern.values, 
                marker='o', label=status, linewidth=2, markersize=4)
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Mean Glucose Excursion (mg/dL)')
    plt.title('⏰ Hourly Pattern by Diabetic Status')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. First meal effect by diabetic status
    plt.subplot(3, 3, 8)
    first_meal_effects = []
    statuses = ['Normal', 'Pre-diabetic', 'Type2Diabetic']
    
    for status in statuses:
        status_data = data[data['diabetic_status'] == status]
        first = status_data[status_data['is_first_meal'] == True]['glucose_excursion'].mean()
        subsequent = status_data[status_data['is_first_meal'] == False]['glucose_excursion'].mean()
        first_meal_effects.append(first - subsequent)
    
    bars = plt.bar(statuses, first_meal_effects, 
                   color=['lightgreen', 'orange', 'lightcoral'])
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('🌅 First Meal Effect by Diabetic Status')
    plt.ylabel('First Meal - Subsequent Meals\n(mg/dL)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, first_meal_effects):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 9. Time-based carbohydrate tolerance
    plt.subplot(3, 3, 9)
    
    # Create carb bins and calculate response by hour
    data['carb_bin'] = pd.cut(data['carbohydrates'], 
                              bins=[0, 40, 80, 200], 
                              labels=['Low (<40g)', 'Medium (40-80g)', 'High (>80g)'])
    
    for carb_level in ['Low (<40g)', 'Medium (40-80g)', 'High (>80g)']:
        carb_data = data[data['carb_bin'] == carb_level]
        if len(carb_data) > 0:
            hourly_carb = carb_data.groupby('hour')['glucose_excursion'].mean()
            plt.plot(hourly_carb.index, hourly_carb.values, 
                    marker='s', label=carb_level, linewidth=2, markersize=4)
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Mean Glucose Excursion (mg/dL)')
    plt.title('🍞 Carbohydrate Tolerance by Hour')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('meal_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved as 'meal_pattern_analysis.png'")
    
    # Create summary statistics table
    print("\n📋 SUMMARY STATISTICS TABLE")
    print("-" * 50)
    
    summary_stats = pd.DataFrame({
        'Metric': [
            'Breakfast Average Excursion',
            'Lunch Average Excursion', 
            'Dinner Average Excursion',
            'First Meal Average Excursion',
            'Subsequent Meals Average Excursion',
            'Early Morning (6-9 AM) Breakfast',
            'Other Time Breakfast',
            'Best Hour for Meals',
            'Worst Hour for Meals'
        ],
        'Value': [
            f"{data[data['meal_type'] == 'breakfast']['glucose_excursion'].mean():.1f} mg/dL",
            f"{data[data['meal_type'] == 'lunch']['glucose_excursion'].mean():.1f} mg/dL",
            f"{data[data['meal_type'] == 'dinner']['glucose_excursion'].mean():.1f} mg/dL",
            f"{data[data['is_first_meal'] == True]['glucose_excursion'].mean():.1f} mg/dL",
            f"{data[data['is_first_meal'] == False]['glucose_excursion'].mean():.1f} mg/dL",
            f"{data[(data['meal_type'] == 'breakfast') & (data['hour'] >= 6) & (data['hour'] <= 9)]['glucose_excursion'].mean():.1f} mg/dL",
            f"{data[(data['meal_type'] == 'breakfast') & ((data['hour'] < 6) | (data['hour'] > 9))]['glucose_excursion'].mean():.1f} mg/dL",
            f"{data.groupby('hour')['glucose_excursion'].mean().idxmin()}:00",
            f"{data.groupby('hour')['glucose_excursion'].mean().idxmax()}:00"
        ]
    })
    
    print(summary_stats.to_string(index=False))
    
    return data_sorted

if __name__ == "__main__":
    result_data = create_meal_pattern_visualizations()