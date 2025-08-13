#!/usr/bin/env python3
"""
CGMacros Pre-Meal Glucose Analysis

This script analyzes the CGMacros dataset to find pre-meal glucose readings for each participant.
It identifies meal entries and finds the glucose reading from just before the meal was consumed.
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

def analyze_participant(participant_folder, cgm_data_path):
    """
    Analyze a single participant's data to find pre-meal glucose readings.
    
    Args:
        participant_folder: Name of the participant folder (e.g., "CGMacros-001")
        cgm_data_path: Path to the CGMacros directory
    
    Returns:
        dict: Participant analysis results
    """
    csv_file = os.path.join(cgm_data_path, participant_folder, f"{participant_folder}.csv")
    
    # Check if CSV file exists
    if not os.path.exists(csv_file):
        return None
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Convert timestamp to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Sort by timestamp to ensure correct ordering
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        # Find meal entries (rows where Meal Type is not empty and Calories > 0)
        meal_mask = (~df['Meal Type'].isna()) & (df['Meal Type'] != '') & (df['Calories'] > 0)
        meal_entries = df[meal_mask].copy()
        
        if len(meal_entries) == 0:
            return {
                'participant': participant_folder,
                'total_meals': 0,
                'meals_with_premeal_glucose': 0,
                'average_premeal_glucose_libre': np.nan,
                'average_premeal_glucose_dexcom': np.nan,
                'error': 'No meal entries found'
            }
        
        premeal_libre_readings = []
        premeal_dexcom_readings = []
        meals_processed = []
        
        for idx, meal_row in meal_entries.iterrows():
            meal_time = meal_row['Timestamp']
            meal_type = meal_row['Meal Type']
            
            # Find glucose reading before the meal
            # Look for readings within 30 minutes before the meal
            time_window_start = meal_time - timedelta(minutes=30)
            
            # Get all readings before the meal within the time window
            before_meal = df[
                (df['Timestamp'] >= time_window_start) & 
                (df['Timestamp'] < meal_time)
            ].copy()
            
            if len(before_meal) == 0:
                continue
            
            # Get the closest reading before the meal
            closest_before = before_meal.iloc[-1]  # Last entry before meal
            
            # Extract glucose readings (prefer non-NaN values)
            libre_reading = closest_before['Libre GL']
            dexcom_reading = closest_before['Dexcom GL']
            
            # Store readings if they are not NaN
            if pd.notna(libre_reading) and libre_reading > 0:
                premeal_libre_readings.append(libre_reading)
            
            if pd.notna(dexcom_reading) and dexcom_reading > 0:
                premeal_dexcom_readings.append(dexcom_reading)
            
            # Store meal info for debugging
            meals_processed.append({
                'meal_time': meal_time,
                'meal_type': meal_type,
                'premeal_time': closest_before['Timestamp'],
                'libre_reading': libre_reading,
                'dexcom_reading': dexcom_reading,
                'time_diff_minutes': (meal_time - closest_before['Timestamp']).total_seconds() / 60
            })
        
        # Calculate averages
        avg_libre = np.mean(premeal_libre_readings) if premeal_libre_readings else np.nan
        avg_dexcom = np.mean(premeal_dexcom_readings) if premeal_dexcom_readings else np.nan
        
        return {
            'participant': participant_folder,
            'total_meals': len(meal_entries),
            'meals_with_premeal_glucose': len(meals_processed),
            'meals_with_libre_data': len(premeal_libre_readings),
            'meals_with_dexcom_data': len(premeal_dexcom_readings),
            'average_premeal_glucose_libre': avg_libre,
            'average_premeal_glucose_dexcom': avg_dexcom,
            'premeal_libre_readings': premeal_libre_readings,
            'premeal_dexcom_readings': premeal_dexcom_readings,
            'meals_processed': meals_processed
        }
        
    except Exception as e:
        return {
            'participant': participant_folder,
            'total_meals': 0,
            'meals_with_premeal_glucose': 0,
            'average_premeal_glucose_libre': np.nan,
            'average_premeal_glucose_dexcom': np.nan,
            'error': f'Error processing data: {str(e)}'
        }

def main():
    """Main analysis function"""
    cgm_data_path = "/Users/ahmedsherif/Library/CloudStorage/Dropbox/Upwork/ASPI_Glucose/cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0/CGMacros"
    
    # Get all participant folders
    participant_folders = [f for f in os.listdir(cgm_data_path) if f.startswith('CGMacros-') and os.path.isdir(os.path.join(cgm_data_path, f))]
    participant_folders.sort()
    
    print(f"Found {len(participant_folders)} participant folders")
    print("=" * 80)
    
    results = []
    
    for folder in participant_folders:
        print(f"Processing {folder}...")
        result = analyze_participant(folder, cgm_data_path)
        
        if result:
            results.append(result)
            
            # Print summary for this participant
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                libre_avg = result['average_premeal_glucose_libre']
                dexcom_avg = result['average_premeal_glucose_dexcom']
                
                print(f"  Total meals: {result['total_meals']}")
                print(f"  Meals with pre-meal glucose: {result['meals_with_premeal_glucose']}")
                
                if not pd.isna(libre_avg):
                    print(f"  Average pre-meal Libre glucose: {libre_avg:.1f} mg/dL ({result['meals_with_libre_data']} readings)")
                else:
                    print(f"  Average pre-meal Libre glucose: No data available")
                
                if not pd.isna(dexcom_avg):
                    print(f"  Average pre-meal Dexcom glucose: {dexcom_avg:.1f} mg/dL ({result['meals_with_dexcom_data']} readings)")
                else:
                    print(f"  Average pre-meal Dexcom glucose: No data available")
        
        print()
    
    # Generate overall summary
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    # Create summary table
    summary_data = []
    for result in results:
        if 'error' not in result:
            summary_data.append({
                'Participant': result['participant'],
                'Total Meals': result['total_meals'],
                'Pre-meal Readings': result['meals_with_premeal_glucose'],
                'Avg Libre GL (mg/dL)': f"{result['average_premeal_glucose_libre']:.1f}" if not pd.isna(result['average_premeal_glucose_libre']) else "N/A",
                'Libre Count': result['meals_with_libre_data'],
                'Avg Dexcom GL (mg/dL)': f"{result['average_premeal_glucose_dexcom']:.1f}" if not pd.isna(result['average_premeal_glucose_dexcom']) else "N/A",
                'Dexcom Count': result['meals_with_dexcom_data']
            })
    
    # Print summary table
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        # Calculate overall statistics
        libre_values = [result['average_premeal_glucose_libre'] for result in results 
                       if not pd.isna(result.get('average_premeal_glucose_libre', np.nan))]
        dexcom_values = [result['average_premeal_glucose_dexcom'] for result in results 
                        if not pd.isna(result.get('average_premeal_glucose_dexcom', np.nan))]
        
        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)
        print(f"Total participants analyzed: {len(results)}")
        print(f"Participants with Libre pre-meal data: {len(libre_values)}")
        print(f"Participants with Dexcom pre-meal data: {len(dexcom_values)}")
        
        if libre_values:
            print(f"Overall average pre-meal Libre glucose: {np.mean(libre_values):.1f} ± {np.std(libre_values):.1f} mg/dL")
            print(f"Libre glucose range: {np.min(libre_values):.1f} - {np.max(libre_values):.1f} mg/dL")
        
        if dexcom_values:
            print(f"Overall average pre-meal Dexcom glucose: {np.mean(dexcom_values):.1f} ± {np.std(dexcom_values):.1f} mg/dL")
            print(f"Dexcom glucose range: {np.min(dexcom_values):.1f} - {np.max(dexcom_values):.1f} mg/dL")
        
        # Save detailed results to CSV
        output_file = "premeal_glucose_analysis.csv"
        df_summary.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    results = main()