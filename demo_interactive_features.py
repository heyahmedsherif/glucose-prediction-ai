#!/usr/bin/env python3
"""
Demo: Interactive Spike Analyzer Features

This demo showcases the key features of the Interactive Glucose Spike Analyzer
by creating sample visualizations with different spike methods.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def demo_spike_methods():
    """Demonstrate different spike emphasis methods with sample data."""
    
    print("🎛️ INTERACTIVE GLUCOSE SPIKE ANALYZER - DEMO")
    print("=" * 60)
    print()
    
    # Create sample glucose data for demonstration
    np.random.seed(42)
    time_points = [i * 15 for i in range(9)]  # 0 to 120 minutes
    
    # Sample data for different diabetic statuses
    sample_data = {
        'Normal': {
            'mean': [85, 90, 105, 110, 108, 100, 95, 90, 88],
            'std': [8, 12, 18, 20, 15, 12, 10, 8, 7]
        },
        'Pre-diabetic': {
            'mean': [95, 105, 135, 145, 140, 125, 115, 105, 98],
            'std': [12, 20, 35, 40, 35, 25, 20, 15, 12]
        },
        'Type2Diabetic': {
            'mean': [125, 140, 180, 200, 195, 170, 155, 140, 135],
            'std': [20, 30, 50, 60, 55, 45, 35, 25, 20]
        }
    }
    
    # Available spike methods
    spike_methods = {
        'mean': 'Mean Response',
        'upper_ci': 'Mean + 1 SD',
        'upper_ci_15': 'Mean + 1.5 SD', 
        'upper_ci_2': 'Mean + 2 SD',
        '95th_percentile': '95th Percentile',
        'max': 'Maximum Response'
    }
    
    print("📊 AVAILABLE SPIKE EMPHASIS METHODS:")
    print("-" * 40)
    for method_key, method_name in spike_methods.items():
        if method_key == "mean":
            print(f"• {method_name} ← Traditional flat curves")
        else:
            print(f"• {method_name}")
    print()
    
    print("🎯 SAMPLE PEAK GLUCOSE VALUES:")
    print("-" * 40)
    
    # Calculate and display peak values for each method and status
    for method_key, method_name in spike_methods.items():
        print(f"\n🔹 {method_name}:")
        
        for status, data in sample_data.items():
            mean_vals = np.array(data['mean'])
            std_vals = np.array(data['std'])
            
            if method_key == 'mean':
                curve = mean_vals
            elif method_key == 'upper_ci':
                curve = mean_vals + std_vals
            elif method_key == 'upper_ci_15':
                curve = mean_vals + 1.5 * std_vals
            elif method_key == 'upper_ci_2':
                curve = mean_vals + 2 * std_vals
            elif method_key == '95th_percentile':
                # Approximate 95th percentile as mean + 1.65 * std
                curve = mean_vals + 1.65 * std_vals
            elif method_key == 'max':
                # Approximate max as mean + 2.5 * std
                curve = mean_vals + 2.5 * std_vals
            
            peak_glucose = max(curve)
            baseline = curve[0]
            excursion = peak_glucose - baseline
            
            print(f"  {status:15}: Peak {peak_glucose:5.0f} mg/dL (+{excursion:3.0f} from baseline)")
    
    print("\n" + "=" * 60)
    print("🎛️ INTERACTIVE FEATURES IN THE STREAMLIT APP:")
    print("=" * 60)
    
    features = [
        "✓ Select multiple spike methods simultaneously",
        "✓ Compare methods side-by-side in subplots", 
        "✓ Choose which diabetic status groups to include",
        "✓ Custom standard deviation multiplier slider",
        "✓ Toggle individual glucose curves overlay",
        "✓ Interactive method comparison bar chart",
        "✓ Detailed metrics for each method and status",
        "✓ Export analysis data as CSV",
        "✓ Generate comprehensive analysis reports",
        "✓ Real-time updates when changing parameters"
    ]
    
    for feature in features:
        print(feature)
    
    print("\n" + "=" * 60)
    print("🚀 HOW TO LAUNCH THE INTERACTIVE APP:")
    print("=" * 60)
    
    launch_options = [
        "1. Simple launch: ./run_interactive_analyzer.sh",
        "2. Direct launch: streamlit run interactive_spike_analyzer.py --server.port 8507",
        "3. Background: streamlit run interactive_spike_analyzer.py &"
    ]
    
    for option in launch_options:
        print(option)
    
    print("\n📊 EXPECTED BENEFITS:")
    print("-" * 30)
    benefits = [
        "• See dramatic glucose spikes instead of flat curves",
        "• Compare different emphasis methods easily",
        "• Understand which method best shows your data patterns",
        "• Export results for presentations or publications",
        "• Interactive exploration of glucose response patterns"
    ]
    
    for benefit in benefits:
        print(benefit)
    
    print(f"\n✅ Demo complete! Launch the interactive app to see these features in action.")

if __name__ == "__main__":
    demo_spike_methods()