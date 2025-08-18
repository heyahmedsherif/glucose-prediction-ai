#!/usr/bin/env python3
"""
Test Normal Group Plotting Issue

Simple test to see if normal group data plots correctly as a line vs a single dot.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def test_normal_plotting():
    """Test plotting normal group data directly."""
    
    print("🧪 Testing Normal Group Plotting")
    print("=" * 40)
    
    # Create sample normal data from debug output
    time_points = [0, 15, 30, 45, 60, 75, 90, 105, 120]
    
    # Real normal glucose curves from debug output
    normal_curves = [
        [88.4, 98.0, 114.8, 114.0, 96.8, 93.8, 100.4, 92.6, 70.6],  # Subject 001, Meal 1
        [86.3, 96.5, 104.5, 100.3, 91.7, 86.6, 84.1, 82.5, 83.4],  # Subject 001, Meal 2
        [73.2, 77.6, 93.6, 109.0, 112.6, 105.0, 97.8, 94.4, 86.4], # Subject 002, Meal 1
        [99.7, 129.2, 156.7, 166.0, 156.4, 138.9, 127.3, 119.7, 102.1], # Subject 004, Meal 1
        [80.7, 87.5, 88.0, 88.0, 91.7, 110.7, 123.2, 102.5, 82.3]  # Subject 006, Meal 1
    ]
    
    print(f"Testing with {len(normal_curves)} normal glucose curves")
    
    # Calculate statistics
    normal_matrix = np.array(normal_curves)
    mean_curve = np.mean(normal_matrix, axis=0)
    std_curve = np.std(normal_matrix, axis=0)
    
    print(f"Mean curve: {[round(x, 1) for x in mean_curve]}")
    print(f"Time points: {time_points}")
    print(f"Curve lengths - Time: {len(time_points)}, Mean: {len(mean_curve)}")
    
    # Test matplotlib plotting
    print("\n📊 Testing matplotlib plotting...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual curves
    plt.subplot(2, 2, 1)
    for i, curve in enumerate(normal_curves):
        plt.plot(time_points, curve, 'b-', alpha=0.3, linewidth=1)
    plt.plot(time_points, mean_curve, 'b-', linewidth=3, marker='o', markersize=6, label='Normal Mean')
    plt.title('Individual Normal Curves')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot mean only
    plt.subplot(2, 2, 2)
    plt.plot(time_points, mean_curve, 'b-', linewidth=3, marker='o', markersize=8, label='Normal Mean')
    plt.title('Mean Normal Curve Only')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot spike methods
    plt.subplot(2, 2, 3)
    spike_curve = mean_curve + std_curve
    plt.plot(time_points, mean_curve, 'b-', linewidth=2, marker='o', label='Normal Mean')
    plt.plot(time_points, spike_curve, 'r-', linewidth=3, marker='D', markersize=6, label='Normal Spike (Mean+SD)')
    plt.title('Normal: Mean vs Spike')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test plotly plotting
    plt.subplot(2, 2, 4)
    plt.plot(time_points, mean_curve, 'b-', linewidth=3, marker='o', markersize=8)
    plt.title('Normal Curve - Single Line Test')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Glucose (mg/dL)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_normal_plotting.png', dpi=150, bbox_inches='tight')
    print("✅ Matplotlib test saved as 'test_normal_plotting.png'")
    
    # Test Plotly plotting
    print("\n📊 Testing Plotly plotting...")
    
    fig = go.Figure()
    
    # Add individual curves
    for i, curve in enumerate(normal_curves):
        fig.add_trace(go.Scatter(
            x=time_points, 
            y=curve,
            mode='lines',
            name=f'Normal Curve {i+1}',
            line=dict(color='lightblue', width=1),
            opacity=0.5
        ))
    
    # Add mean curve
    fig.add_trace(go.Scatter(
        x=time_points, 
        y=mean_curve,
        mode='lines+markers',
        name='Normal Mean',
        line=dict(color='blue', width=4),
        marker=dict(size=8, symbol='circle')
    ))
    
    # Add spike curve
    fig.add_trace(go.Scatter(
        x=time_points, 
        y=spike_curve,
        mode='lines+markers',
        name='Normal Spike (Mean+SD)',
        line=dict(color='red', width=4),
        marker=dict(size=8, symbol='diamond')
    ))
    
    fig.update_layout(
        title="Plotly Test: Normal Group Glucose Responses",
        xaxis_title="Time (minutes)",
        yaxis_title="Glucose (mg/dL)",
        height=600
    )
    
    fig.write_html('test_normal_plotly.html')
    print("✅ Plotly test saved as 'test_normal_plotly.html'")
    
    # Diagnose potential issues
    print(f"\n🔍 DIAGNOSTIC INFORMATION:")
    print(f"Data types:")
    print(f"  time_points type: {type(time_points[0])}")
    print(f"  mean_curve type: {type(mean_curve[0])}")
    print(f"  All finite values: {np.all(np.isfinite(mean_curve))}")
    print(f"  Any NaN values: {np.any(np.isnan(mean_curve))}")
    print(f"  Any infinite values: {np.any(np.isinf(mean_curve))}")
    
    # Check if there are duplicated points that might cause single dot
    duplicates = []
    for i in range(len(time_points)-1):
        if time_points[i] == time_points[i+1] and mean_curve[i] == mean_curve[i+1]:
            duplicates.append(i)
    
    if duplicates:
        print(f"  ⚠️  Found duplicate points at indices: {duplicates}")
    else:
        print(f"  ✅ No duplicate points found")
    
    # Check value ranges
    print(f"  Time range: {min(time_points)} - {max(time_points)}")
    print(f"  Glucose range: {min(mean_curve):.1f} - {max(mean_curve):.1f}")
    
    return mean_curve, time_points

if __name__ == "__main__":
    test_normal_plotting()