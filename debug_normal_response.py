#!/usr/bin/env python3
"""
Debug Normal Response Display Issue

This script investigates why the normal response might be showing as only one dot
instead of a continuous line in the glucose response plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def classify_diabetic_status(a1c):
    """Classify diabetic status based on A1c levels."""
    if a1c < 5.7:
        return "Normal"
    elif a1c <= 6.4:
        return "Pre-diabetic" 
    else:
        return "Type2Diabetic"

def debug_data_processing():
    """Debug the data processing to see what's happening with normal participants."""
    
    print("🔍 DEBUGGING NORMAL RESPONSE DISPLAY ISSUE")
    print("=" * 60)
    
    # Load bio data first
    if not os.path.exists("CGMacros/bio.csv"):
        print("❌ Bio data not found")
        return
    
    bio_df = pd.read_csv("CGMacros/bio.csv")
    bio_df.columns = bio_df.columns.str.strip()
    bio_df['diabetic_status'] = bio_df['A1c PDL (Lab)'].apply(classify_diabetic_status)
    
    print(f"📊 Bio data distribution:")
    status_counts = bio_df['diabetic_status'].value_counts()
    print(status_counts)
    print()
    
    # Process glucose data and check each step
    data_all_sub = pd.DataFrame(columns=["sub", "Libre GL", "Carb", "Protein", "Fat", "Fiber"])
    
    normal_subjects = []
    normal_glucose_data = []
    
    for sub_dir in sorted(os.listdir("CGMacros")):
        if sub_dir[:8] != "CGMacros":
            continue
            
        csv_path = os.path.join("CGMacros", sub_dir, sub_dir + '.csv')
        if not os.path.exists(csv_path):
            continue
            
        try:
            subject_num = int(sub_dir[-3:])
            
            # Check if this subject is normal
            bio_row = bio_df[bio_df['subject'] == subject_num]
            if len(bio_row) == 0:
                continue
                
            subject_status = bio_row.iloc[0]['diabetic_status']
            
            data = pd.read_csv(csv_path)
            breakfast_mask = (data["Meal Type"] == "Breakfast") | (data["Meal Type"] == "breakfast")
            breakfast_meals = data[breakfast_mask]
            
            print(f"Subject {sub_dir[-3:]}: {subject_status}, {len(breakfast_meals)} breakfast meals")
            
            meal_count = 0
            for index in breakfast_meals.index:
                # Extract glucose readings (every 15 minutes for 2 hours)
                glucose_readings = data["Libre GL"][index:index+135:15].to_list()
                
                print(f"  Meal {meal_count + 1}: {len(glucose_readings)} glucose points")
                
                if len(glucose_readings) >= 9:
                    glucose_curve = glucose_readings[:9]  # Take first 9 points
                    print(f"    Glucose curve: {[round(x, 1) if not pd.isna(x) else 'NaN' for x in glucose_curve]}")
                    
                    # Check for NaN values
                    nan_count = sum(1 for x in glucose_curve if pd.isna(x))
                    if nan_count > 0:
                        print(f"    ⚠️  {nan_count} NaN values found")
                    
                    if subject_status == "Normal":
                        normal_subjects.append(sub_dir[-3:])
                        normal_glucose_data.append(glucose_curve)
                        
                    meal_count += 1
                else:
                    print(f"    ❌ Insufficient data points ({len(glucose_readings)} < 9)")
            
            if meal_count > 0:
                print(f"    ✅ {meal_count} valid meals processed")
            print()
                    
        except Exception as e:
            print(f"❌ Error processing {sub_dir}: {e}")
            continue
    
    print(f"\n📊 NORMAL GROUP ANALYSIS:")
    print(f"Total normal subjects with data: {len(set(normal_subjects))}")
    print(f"Total normal meal records: {len(normal_glucose_data)}")
    
    if normal_glucose_data:
        print(f"\nFirst few normal glucose curves:")
        for i, curve in enumerate(normal_glucose_data[:5]):
            print(f"  Curve {i+1}: {[round(x, 1) if not pd.isna(x) else 'NaN' for x in curve]}")
        
        # Check for data quality issues
        all_valid = True
        for i, curve in enumerate(normal_glucose_data):
            if any(pd.isna(x) for x in curve):
                print(f"  ⚠️  Curve {i+1} has NaN values")
                all_valid = False
            if len(curve) != 9:
                print(f"  ⚠️  Curve {i+1} has {len(curve)} points (should be 9)")
                all_valid = False
        
        if all_valid:
            print("  ✅ All normal curves have valid data")
        
        # Calculate statistics
        normal_matrix = np.array([curve for curve in normal_glucose_data if len(curve) == 9 and not any(pd.isna(x) for x in curve)])
        
        if len(normal_matrix) > 0:
            print(f"\nValid normal curves for analysis: {len(normal_matrix)}")
            mean_curve = np.mean(normal_matrix, axis=0)
            std_curve = np.std(normal_matrix, axis=0)
            
            print(f"Mean glucose curve: {[round(x, 1) for x in mean_curve]}")
            print(f"Std deviation curve: {[round(x, 1) for x in std_curve]}")
            
            # Create a test plot
            time_points = [i * 15 for i in range(9)]
            
            plt.figure(figsize=(10, 6))
            plt.plot(time_points, mean_curve, 'b-o', linewidth=3, markersize=8, label='Normal Mean')
            plt.fill_between(time_points, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)
            plt.xlabel('Time (minutes)')
            plt.ylabel('Glucose (mg/dL)')
            plt.title('Normal Group Glucose Response - Debug Plot')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('debug_normal_response.png', dpi=150, bbox_inches='tight')
            print(f"\n📊 Debug plot saved as 'debug_normal_response.png'")
            
        else:
            print("❌ No valid normal curves found for analysis")
    
    else:
        print("❌ No normal glucose data found")
    
    print(f"\n🔍 POSSIBLE CAUSES OF SINGLE DOT ISSUE:")
    print("1. Normal participants might have mostly NaN glucose values")
    print("2. Glucose curves might be too short (< 9 time points)")  
    print("3. Plotting code might have a bug with the normal group specifically")
    print("4. Data processing might be filtering out normal participants")
    print("5. Line style or marker settings might be wrong for normal group")

def main():
    debug_data_processing()

if __name__ == "__main__":
    main()