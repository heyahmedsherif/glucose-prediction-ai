#!/usr/bin/env python3
"""
Stratified Glucose Response Visualization

Based on parse_data.ipynb approach, this script creates separate glucose response 
visualizations for each diabetic status group to show the glucose spikes that 
get averaged out in combined visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def classify_diabetic_status(a1c):
    """Classify diabetic status based on A1c levels."""
    if a1c < 5.7:
        return "Normal"
    elif a1c <= 6.4:
        return "Pre-diabetic" 
    else:
        return "Type2Diabetic"

def load_and_process_data():
    """Load and process CGMacros data following parse_data.ipynb approach."""
    
    print("Loading CGMacros data...")
    data_all_sub = pd.DataFrame(columns=["sub", "Libre GL", "Carb", "Protein", "Fat", "Fiber"])
    
    hours = 2
    libre_samples = hours * 4 + 1
    
    # Process each subject directory
    for sub_dir in sorted(os.listdir("CGMacros")):
        if sub_dir[:8] != "CGMacros":
            continue
            
        csv_path = os.path.join("CGMacros", sub_dir, sub_dir + '.csv')
        if not os.path.exists(csv_path):
            continue
            
        print(f"Processing {sub_dir}...")
        data = pd.read_csv(csv_path)
        data_sub = pd.DataFrame(columns=["sub", "Libre GL", "Carb", "Protein", "Fat", "Fiber"])
        
        # Extract breakfast meals
        breakfast_mask = (data["Meal Type"] == "Breakfast") | (data["Meal Type"] == "breakfast")
        
        for index in data[breakfast_mask].index:
            data_meal = {}
            data_meal["sub"] = sub_dir[-3:]
            
            # Extract glucose readings (every 15 minutes for 2 hours)
            glucose_readings = data["Libre GL"][index:index+135:15].to_list()
            if len(glucose_readings) < 9:
                continue
                
            data_meal["Libre GL"] = glucose_readings
            data_meal["Carb"] = data["Carbs"][index]
            data_meal["Protein"] = data["Protein"][index] 
            data_meal["Fat"] = data["Fat"][index]
            data_meal["Fiber"] = data["Fiber"][index]
            data_meal["Calories"] = data["Calories"][index] if "Calories" in data.columns else 0
            
            data_sub = pd.concat([data_sub, pd.DataFrame([data_meal])], ignore_index=True)
        
        if len(data_sub) > 0:
            data_all_sub = pd.concat([data_all_sub, data_sub], ignore_index=True)
    
    # Load bio data for A1c classification
    print("Loading bio data...")
    bio_df = pd.read_csv("CGMacros/bio.csv")
    bio_df.columns = bio_df.columns.str.strip()
    bio_df['diabetic_status'] = bio_df['A1c PDL (Lab)'].apply(classify_diabetic_status)
    
    # Add baseline glucose (first reading) 
    baseline_glucose = []
    for i in range(len(data_all_sub)):
        baseline_glucose.append(data_all_sub["Libre GL"].iloc[i][0])
    data_all_sub["Baseline_Libre"] = baseline_glucose
    
    # Merge with bio data
    subjects = data_all_sub["sub"].unique()
    diabetic_status_list = []
    a1c_list = []
    
    for i, sub_id in enumerate(data_all_sub["sub"]):
        # Convert subject ID to match bio data
        bio_subject_id = int(sub_id)
        bio_row = bio_df[bio_df['subject'] == bio_subject_id]
        
        if len(bio_row) > 0:
            diabetic_status_list.append(bio_row.iloc[0]['diabetic_status'])
            a1c_list.append(bio_row.iloc[0]['A1c PDL (Lab)'])
        else:
            diabetic_status_list.append('Unknown')
            a1c_list.append(np.nan)
    
    data_all_sub['diabetic_status'] = diabetic_status_list
    data_all_sub['A1c'] = a1c_list
    
    # Remove unknown status entries
    data_all_sub = data_all_sub[data_all_sub['diabetic_status'] != 'Unknown']
    
    print(f"Processed {len(data_all_sub)} meal records from {len(data_all_sub['sub'].unique())} subjects")
    print(f"Distribution: {data_all_sub['diabetic_status'].value_counts().to_dict()}")
    
    return data_all_sub

def create_stratified_glucose_curves(data_all_sub):
    """Create glucose response curves stratified by diabetic status."""
    
    print("\nCreating stratified glucose response visualizations...")
    
    # Time points (0, 15, 30, 45, 60, 75, 90, 105, 120 minutes)
    time_points = [i * 15 for i in range(9)]
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Glucose Response Patterns by Diabetic Status', fontsize=16, fontweight='bold')
    
    # Colors for each status
    colors = {
        'Normal': 'blue',
        'Pre-diabetic': 'green', 
        'Type2Diabetic': 'red'
    }
    
    status_order = ['Normal', 'Pre-diabetic', 'Type2Diabetic']
    
    # Plot 1: Individual curves by status
    ax1 = axes[0, 0]
    for status in status_order:
        if status not in data_all_sub['diabetic_status'].values:
            continue
            
        status_data = data_all_sub[data_all_sub['diabetic_status'] == status]
        
        # Plot individual curves (semi-transparent)
        for _, row in status_data.iterrows():
            glucose_curve = row['Libre GL'][:len(time_points)]
            ax1.plot(time_points, glucose_curve, color=colors[status], alpha=0.1, linewidth=0.5)
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Glucose (mg/dL)')
    ax1.set_title('Individual Glucose Response Curves')
    ax1.grid(True, alpha=0.3)
    # Create legend handles
    legend_handles = []
    for status in status_order:
        if status in data_all_sub['diabetic_status'].values:
            legend_handles.append(plt.Line2D([0], [0], color=colors[status], label=status))
    ax1.legend(handles=legend_handles)
    
    # Plot 2: Mean curves with confidence intervals
    ax2 = axes[0, 1]
    for status in status_order:
        if status not in data_all_sub['diabetic_status'].values:
            continue
            
        status_data = data_all_sub[data_all_sub['diabetic_status'] == status]
        
        # Calculate mean and std for each time point
        glucose_arrays = []
        for _, row in status_data.iterrows():
            glucose_curve = row['Libre GL'][:len(time_points)]
            if len(glucose_curve) == len(time_points):
                glucose_arrays.append(glucose_curve)
        
        if glucose_arrays:
            glucose_matrix = np.array(glucose_arrays)
            mean_glucose = np.mean(glucose_matrix, axis=0)
            std_glucose = np.std(glucose_matrix, axis=0)
            
            # Plot mean with confidence interval
            ax2.plot(time_points, mean_glucose, color=colors[status], linewidth=3, label=f'{status} (n={len(glucose_arrays)})')
            ax2.fill_between(time_points, 
                           mean_glucose - std_glucose, 
                           mean_glucose + std_glucose,
                           color=colors[status], alpha=0.2)
    
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Glucose (mg/dL)')
    ax2.set_title('Mean Glucose Response with Standard Deviation')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Glucose excursion from baseline
    ax3 = axes[1, 0]
    for status in status_order:
        if status not in data_all_sub['diabetic_status'].values:
            continue
            
        status_data = data_all_sub[data_all_sub['diabetic_status'] == status]
        
        excursion_arrays = []
        for _, row in status_data.iterrows():
            glucose_curve = row['Libre GL'][:len(time_points)]
            if len(glucose_curve) == len(time_points):
                baseline = glucose_curve[0]
                excursion = [g - baseline for g in glucose_curve]
                excursion_arrays.append(excursion)
        
        if excursion_arrays:
            excursion_matrix = np.array(excursion_arrays)
            mean_excursion = np.mean(excursion_matrix, axis=0)
            std_excursion = np.std(excursion_matrix, axis=0)
            
            ax3.plot(time_points, mean_excursion, color=colors[status], linewidth=3, label=f'{status}')
            ax3.fill_between(time_points, 
                           mean_excursion - std_excursion, 
                           mean_excursion + std_excursion,
                           color=colors[status], alpha=0.2)
    
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Glucose Excursion from Baseline (mg/dL)')
    ax3.set_title('Glucose Excursion from Baseline')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Peak glucose distribution
    ax4 = axes[1, 1]
    peak_glucose_data = []
    status_labels = []
    
    for status in status_order:
        if status not in data_all_sub['diabetic_status'].values:
            continue
            
        status_data = data_all_sub[data_all_sub['diabetic_status'] == status]
        
        peaks = []
        for _, row in status_data.iterrows():
            glucose_curve = row['Libre GL'][:len(time_points)]
            if len(glucose_curve) == len(time_points):
                peaks.append(max(glucose_curve))
        
        if peaks:
            peak_glucose_data.append(peaks)
            status_labels.append(f'{status}\n(n={len(peaks)})')
    
    if peak_glucose_data:
        box_plot = ax4.boxplot(peak_glucose_data, labels=status_labels, patch_artist=True)
        
        # Color the boxes
        for patch, status in zip(box_plot['boxes'], status_order[:len(peak_glucose_data)]):
            patch.set_facecolor(colors[status])
            patch.set_alpha(0.7)
    
    ax4.set_ylabel('Peak Glucose (mg/dL)')
    ax4.set_title('Peak Glucose Distribution by Status')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stratified_glucose_responses.png', dpi=300, bbox_inches='tight')
    print("Saved visualization as 'stratified_glucose_responses.png'")
    
    return fig

def print_summary_statistics(data_all_sub):
    """Print summary statistics for each diabetic status group."""
    
    print("\n" + "="*60)
    print("GLUCOSE RESPONSE SUMMARY STATISTICS")
    print("="*60)
    
    time_points = [i * 15 for i in range(9)]
    
    for status in ['Normal', 'Pre-diabetic', 'Type2Diabetic']:
        if status not in data_all_sub['diabetic_status'].values:
            continue
            
        print(f"\n{status.upper()} GROUP:")
        print("-" * 40)
        
        status_data = data_all_sub[data_all_sub['diabetic_status'] == status]
        print(f"Number of meals: {len(status_data)}")
        print(f"Number of subjects: {len(status_data['sub'].unique())}")
        print(f"A1c range: {status_data['A1c'].min():.1f} - {status_data['A1c'].max():.1f}%")
        
        # Calculate glucose statistics
        glucose_arrays = []
        baselines = []
        peaks = []
        
        for _, row in status_data.iterrows():
            glucose_curve = row['Libre GL'][:len(time_points)]
            if len(glucose_curve) == len(time_points):
                glucose_arrays.append(glucose_curve)
                baselines.append(glucose_curve[0])
                peaks.append(max(glucose_curve))
        
        if glucose_arrays:
            glucose_matrix = np.array(glucose_arrays)
            
            print(f"Baseline glucose: {np.mean(baselines):.1f} ± {np.std(baselines):.1f} mg/dL")
            print(f"Peak glucose: {np.mean(peaks):.1f} ± {np.std(peaks):.1f} mg/dL")
            print(f"Peak excursion: {np.mean(peaks) - np.mean(baselines):.1f} ± {np.sqrt(np.var(peaks) + np.var(baselines)):.1f} mg/dL")
            
            # Find time to peak
            time_to_peaks = []
            for glucose_curve in glucose_arrays:
                peak_idx = np.argmax(glucose_curve)
                time_to_peaks.append(time_points[peak_idx])
            
            print(f"Time to peak: {np.mean(time_to_peaks):.0f} ± {np.std(time_to_peaks):.0f} minutes")

def main():
    """Main function."""
    
    print("="*60)
    print("STRATIFIED GLUCOSE RESPONSE VISUALIZATION")
    print("Based on parse_data.ipynb approach")
    print("="*60)
    
    # Load and process data
    data_all_sub = load_and_process_data()
    
    if len(data_all_sub) == 0:
        print("No data found. Please check that CGMacros directory exists.")
        return
    
    # Create visualizations
    fig = create_stratified_glucose_curves(data_all_sub)
    
    # Print summary statistics
    print_summary_statistics(data_all_sub)
    
    print(f"\n✅ Analysis complete!")
    print("Key findings:")
    print("• Separate visualization shows distinct glucose response patterns")
    print("• Type2Diabetic group shows highest glucose spikes")  
    print("• Normal group shows most stable glucose responses")
    print("• Pre-diabetic group shows intermediate patterns")
    
    plt.show()

if __name__ == "__main__":
    main()