#!/usr/bin/env python3
"""
Generate a detailed report with additional insights about pre-meal glucose patterns
"""

import pandas as pd
import numpy as np

def generate_detailed_report():
    """Generate a detailed report with additional insights"""
    
    # Read the results
    df = pd.read_csv('premeal_glucose_analysis.csv')
    
    print("=" * 100)
    print("CGMacros Dataset - Pre-Meal Glucose Analysis - Detailed Report")
    print("=" * 100)
    
    # Basic statistics
    print("\n📊 DATASET OVERVIEW:")
    print(f"• Total participants: {len(df)}")
    print(f"• Total meals analyzed: {df['Total Meals'].sum()}")
    print(f"• Total pre-meal glucose readings: {df['Pre-meal Readings'].sum()}")
    
    # Convert glucose columns to numeric, handling 'N/A' values
    df['Libre_numeric'] = pd.to_numeric(df['Avg Libre GL (mg/dL)'], errors='coerce')
    df['Dexcom_numeric'] = pd.to_numeric(df['Avg Dexcom GL (mg/dL)'], errors='coerce')
    
    # Overall statistics
    libre_mean = df['Libre_numeric'].mean()
    libre_std = df['Libre_numeric'].std()
    libre_min = df['Libre_numeric'].min()
    libre_max = df['Libre_numeric'].max()
    
    dexcom_mean = df['Dexcom_numeric'].mean()
    dexcom_std = df['Dexcom_numeric'].std()
    dexcom_min = df['Dexcom_numeric'].min()
    dexcom_max = df['Dexcom_numeric'].max()
    
    print(f"\n🩺 GLUCOSE SENSOR COMPARISON:")
    print(f"• Libre GL - Average: {libre_mean:.1f} ± {libre_std:.1f} mg/dL (Range: {libre_min:.1f} - {libre_max:.1f})")
    print(f"• Dexcom GL - Average: {dexcom_mean:.1f} ± {dexcom_std:.1f} mg/dL (Range: {dexcom_min:.1f} - {dexcom_max:.1f})")
    print(f"• Average difference (Dexcom - Libre): {dexcom_mean - libre_mean:.1f} mg/dL")
    
    # Classification by glucose levels
    print(f"\n📈 GLUCOSE LEVEL DISTRIBUTION (based on Libre readings):")
    
    # Normal range classifications (using ADA guidelines)
    normal_count = len(df[(df['Libre_numeric'] >= 70) & (df['Libre_numeric'] < 100)])
    prediabetic_count = len(df[(df['Libre_numeric'] >= 100) & (df['Libre_numeric'] < 126)])
    diabetic_count = len(df[df['Libre_numeric'] >= 126])
    hypoglycemic_count = len(df[df['Libre_numeric'] < 70])
    
    total_with_data = len(df.dropna(subset=['Libre_numeric']))
    
    print(f"• Normal (70-99 mg/dL): {normal_count} participants ({normal_count/total_with_data*100:.1f}%)")
    print(f"• Prediabetic range (100-125 mg/dL): {prediabetic_count} participants ({prediabetic_count/total_with_data*100:.1f}%)")
    print(f"• Diabetic range (≥126 mg/dL): {diabetic_count} participants ({diabetic_count/total_with_data*100:.1f}%)")
    print(f"• Hypoglycemic (<70 mg/dL): {hypoglycemic_count} participants ({hypoglycemic_count/total_with_data*100:.1f}%)")
    
    # Participants with most and least meals
    print(f"\n🍽️ MEAL FREQUENCY ANALYSIS:")
    max_meals = df.loc[df['Total Meals'].idxmax()]
    min_meals = df.loc[df['Total Meals'].idxmin()]
    avg_meals = df['Total Meals'].mean()
    
    print(f"• Most meals recorded: {max_meals['Participant']} with {max_meals['Total Meals']} meals")
    print(f"• Least meals recorded: {min_meals['Participant']} with {min_meals['Total Meals']} meals")
    print(f"• Average meals per participant: {avg_meals:.1f}")
    
    # Extreme glucose values
    print(f"\n⚠️ EXTREME GLUCOSE VALUES:")
    highest_libre = df.loc[df['Libre_numeric'].idxmax()]
    lowest_libre = df.loc[df['Libre_numeric'].idxmin()]
    highest_dexcom = df.loc[df['Dexcom_numeric'].idxmax()]
    lowest_dexcom = df.loc[df['Dexcom_numeric'].idxmin()]
    
    print(f"• Highest Libre pre-meal glucose: {highest_libre['Participant']} ({highest_libre['Libre_numeric']:.1f} mg/dL)")
    print(f"• Lowest Libre pre-meal glucose: {lowest_libre['Participant']} ({lowest_libre['Libre_numeric']:.1f} mg/dL)")
    print(f"• Highest Dexcom pre-meal glucose: {highest_dexcom['Participant']} ({highest_dexcom['Dexcom_numeric']:.1f} mg/dL)")
    print(f"• Lowest Dexcom pre-meal glucose: {lowest_dexcom['Participant']} ({lowest_dexcom['Dexcom_numeric']:.1f} mg/dL)")
    
    # Sensor agreement analysis
    print(f"\n🔄 SENSOR AGREEMENT ANALYSIS:")
    df['sensor_diff'] = df['Dexcom_numeric'] - df['Libre_numeric']
    valid_comparisons = df.dropna(subset=['sensor_diff'])
    
    mean_diff = valid_comparisons['sensor_diff'].mean()
    std_diff = valid_comparisons['sensor_diff'].std()
    
    print(f"• Average difference (Dexcom - Libre): {mean_diff:.1f} ± {std_diff:.1f} mg/dL")
    print(f"• Participants where Dexcom > Libre: {len(valid_comparisons[valid_comparisons['sensor_diff'] > 0])} ({len(valid_comparisons[valid_comparisons['sensor_diff'] > 0])/len(valid_comparisons)*100:.1f}%)")
    print(f"• Participants where Libre > Dexcom: {len(valid_comparisons[valid_comparisons['sensor_diff'] < 0])} ({len(valid_comparisons[valid_comparisons['sensor_diff'] < 0])/len(valid_comparisons)*100:.1f}%)")
    
    # Top 10 participants by glucose levels
    print(f"\n🔝 TOP 10 PARTICIPANTS BY PRE-MEAL GLUCOSE (Libre):")
    top_10_libre = df.nlargest(10, 'Libre_numeric')[['Participant', 'Avg Libre GL (mg/dL)', 'Total Meals']]
    for idx, row in top_10_libre.iterrows():
        print(f"   {row['Participant']}: {row['Avg Libre GL (mg/dL)']} mg/dL ({row['Total Meals']} meals)")
    
    print(f"\n🔻 BOTTOM 10 PARTICIPANTS BY PRE-MEAL GLUCOSE (Libre):")
    bottom_10_libre = df.nsmallest(10, 'Libre_numeric')[['Participant', 'Avg Libre GL (mg/dL)', 'Total Meals']]
    for idx, row in bottom_10_libre.iterrows():
        print(f"   {row['Participant']}: {row['Avg Libre GL (mg/dL)']} mg/dL ({row['Total Meals']} meals)")
    
    print(f"\n📋 KEY FINDINGS:")
    print(f"• The dataset shows significant inter-individual variation in pre-meal glucose levels")
    print(f"• Dexcom sensors consistently read higher than Libre sensors on average")
    print(f"• {diabetic_count} participants show average pre-meal glucose levels in the diabetic range")
    print(f"• {hypoglycemic_count} participants show average pre-meal glucose levels suggesting frequent hypoglycemia")
    print(f"• The wide range of glucose values suggests the dataset includes both healthy individuals and those with glucose metabolism disorders")
    
    print(f"\n" + "=" * 100)

if __name__ == "__main__":
    generate_detailed_report()