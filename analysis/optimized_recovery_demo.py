#!/usr/bin/env python3
"""
Optimized Recovery Timeline Bar Demo
Shows 4-hour scale that covers 86% of real recovery times from CGMacros data
"""

import matplotlib.pyplot as plt
import numpy as np
from fixed_glucose_prediction_logic import CorrectedGlucosePrediction

def create_optimized_timeline_bar(recovery_hours, title, meal_type, max_time=4.0):
    """Create optimized recovery timeline bar with 4-hour scale."""
    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Color coding based on data-driven thresholds
    if recovery_hours <= 1.8:  # 50th percentile
        color = '#2E8B57'  # Green - faster than median
        emoji = '🟢'
        speed_text = 'Fast'
    elif recovery_hours <= 3.2:  # 75th percentile  
        color = '#FF8C00'  # Orange - typical recovery
        emoji = '🟡'
        speed_text = 'Typical'
    elif recovery_hours <= 4.0:  # 85th percentile
        color = '#DC143C'  # Red - slower than typical
        emoji = '🔴'  
        speed_text = 'Slow'
    else:  # >85th percentile
        color = '#8B0000'  # Dark red - extreme cases
        emoji = '🔴'
        speed_text = 'Extreme'
    
    # Background bar (full 4-hour scale)
    ax.barh(0, max_time, height=0.6, color='#E8E8E8', alpha=0.3)
    
    # Progress bar (capped at scale limit)
    display_hours = min(recovery_hours, max_time)
    ax.barh(0, display_hours, height=0.6, color=color, alpha=0.8)
    
    # Add text with special handling for extreme cases
    if recovery_hours > max_time:
        text_content = f'{recovery_hours:.1f}h+'
        text_x = max_time - 0.3  # Position near end of scale
        text_color = 'white'
    else:
        text_content = f'{recovery_hours:.1f}h'
        text_x = display_hours / 2 if display_hours > 0.5 else 0.2
        text_color = 'white' if display_hours > 0.5 else 'black'
    
    ax.text(text_x, 0, text_content, 
           ha='center', va='center', fontweight='bold', 
           color=text_color, fontsize=11)
    
    # Add percentile indicator
    if recovery_hours <= 1.8:
        percentile_text = "(<50th %ile)"
    elif recovery_hours <= 3.2:
        percentile_text = "(<75th %ile)"
    elif recovery_hours <= 4.0:
        percentile_text = "(<85th %ile)"
    else:
        percentile_text = "(>85th %ile)"
    
    # Formatting
    ax.set_xlim(0, max_time)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['0h', '1h', '2h', '3h', '4h+'])
    ax.set_yticks([])
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_title(f'{emoji} {title} • {speed_text} Recovery {percentile_text}', 
                fontsize=12, pad=10)
    
    # Add coverage note
    ax.text(max_time + 0.1, 0, f'Peak: {max(glucose_curve):.0f} mg/dL\n4h scale covers\n86% of cases', 
           va='center', fontsize=8, 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    return fig

def demo_optimized_scale():
    """Demonstrate optimized 4-hour scale with real data coverage."""
    
    print("🔧 OPTIMIZED RECOVERY TIMELINE BAR (4-HOUR SCALE)")
    print("=" * 60)
    print("Scale covers 86% of real recovery times from CGMacros data")
    print("=" * 60)
    
    predictor = CorrectedGlucosePrediction()
    
    # Test scenarios including some extreme cases
    scenarios = [
        {
            'title': 'Normal - Fast Recovery',
            'meal': {'carbohydrates': 30, 'protein': 15, 'fat': 5, 'fiber': 8},
            'patient': {'diabetic_status': 'Normal', 'age': 25, 'bmi': 22, 'a1c': 5.0, 'fasting_glucose': 85},
            'timing': {'meal_type': 'lunch', 'meal_hour': 13, 'is_first_meal': False}
        },
        {
            'title': 'Normal - Typical Breakfast',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23, 'a1c': 5.2, 'fasting_glucose': 90},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True}
        },
        {
            'title': 'Pre-diabetic - Moderate Case',
            'meal': {'carbohydrates': 60, 'protein': 25, 'fat': 15, 'fiber': 4},
            'patient': {'diabetic_status': 'Pre-diabetic', 'age': 45, 'bmi': 28, 'a1c': 6.2, 'fasting_glucose': 110},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 7, 'is_first_meal': True}
        },
        {
            'title': 'Type2 - Slow Recovery',
            'meal': {'carbohydrates': 70, 'protein': 30, 'fat': 20, 'fiber': 3},
            'patient': {'diabetic_status': 'Type2Diabetic', 'age': 55, 'bmi': 32, 'a1c': 8.5, 'fasting_glucose': 150},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 7, 'is_first_meal': True}
        },
        {
            'title': 'Type2 - Extreme Dawn Case',
            'meal': {'carbohydrates': 80, 'protein': 20, 'fat': 15, 'fiber': 2},
            'patient': {'diabetic_status': 'Type2Diabetic', 'age': 65, 'bmi': 35, 'a1c': 9.2, 'fasting_glucose': 180},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 6, 'is_first_meal': True}
        }
    ]
    
    # Create subplot figure
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 12))
    fig.suptitle('⏱️ Optimized Recovery Timeline (4-Hour Scale)\nCovers 86% of Real CGMacros Recovery Times', 
                fontsize=16, fontweight='bold')
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n🧪 Testing: {scenario['title']}")
        
        # Get predictions
        predictions = predictor.predict_glucose_with_corrected_timing(
            scenario['meal'], scenario['patient'], scenario['timing']
        )
        
        # Create glucose curve
        time_points = [0, 30, 60, 90, 120, 180]
        global glucose_curve
        glucose_curve = [predictions['baseline']] + [predictions[f'glucose_{t}min'] for t in [30, 60, 90, 120, 180]]
        
        # Enhanced recovery estimation
        baseline = predictions['baseline']
        recovery_time = None
        
        # Look for recovery in observed points
        for j, glucose in enumerate(glucose_curve):
            if j > 0 and abs(glucose - baseline) <= 10:
                recovery_time = time_points[j]
                break
        
        # If not recovered by 180min, estimate based on final trend
        if recovery_time is None:
            final_glucose = glucose_curve[-1]
            if final_glucose > baseline + 10:
                # Enhanced estimation based on patient status
                if scenario['patient']['diabetic_status'] == 'Type2Diabetic':
                    # Type2 diabetics: slower clearance, add extra time
                    extra_time = (final_glucose - baseline - 10) * 2.0  # 2 min per mg/dL excess
                    recovery_time = 180 + extra_time
                elif scenario['patient']['diabetic_status'] == 'Pre-diabetic':
                    extra_time = (final_glucose - baseline - 10) * 1.5
                    recovery_time = 180 + extra_time
                else:  # Normal
                    extra_time = (final_glucose - baseline - 10) * 1.0
                    recovery_time = 180 + extra_time
                
                recovery_time = min(recovery_time, 400)  # Cap at realistic maximum
        
        recovery_hours = recovery_time / 60 if recovery_time else 6.5
        
        print(f"  Baseline: {baseline:.1f} mg/dL")
        print(f"  Peak: {max(glucose_curve):.1f} mg/dL")
        print(f"  Recovery: {recovery_hours:.1f} hours")
        
        # Create timeline bar
        ax = axes[i]
        max_time = 4.0
        
        # Color coding with data-driven thresholds
        if recovery_hours <= 1.8:
            color = '#2E8B57'  # Green - faster than median
            emoji = '🟢'
            speed_text = 'Fast'
        elif recovery_hours <= 3.2:
            color = '#FF8C00'  # Orange - typical
            emoji = '🟡'
            speed_text = 'Typical'
        elif recovery_hours <= 4.0:
            color = '#DC143C'  # Red - slow but within scale
            emoji = '🔴'
            speed_text = 'Slow'
        else:
            color = '#8B0000'  # Dark red - extreme
            emoji = '🔴'
            speed_text = 'Extreme'
        
        # Background bar
        ax.barh(0, max_time, height=0.6, color='#E8E8E8', alpha=0.3)
        
        # Progress bar (capped at scale)
        display_hours = min(recovery_hours, max_time)
        ax.barh(0, display_hours, height=0.6, color=color, alpha=0.8)
        
        # Add text with special handling for >4h cases
        if recovery_hours > max_time:
            text_content = f'{recovery_hours:.1f}h+'
            text_x = max_time - 0.4
            ax.text(text_x, 0, text_content, ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=10)
        else:
            text_content = f'{recovery_hours:.1f}h'
            text_x = display_hours / 2 if display_hours > 0.5 else 0.2
            ax.text(text_x, 0, text_content, ha='center', va='center', 
                   fontweight='bold', color='white', fontsize=10)
        
        # Percentile info
        if recovery_hours <= 1.8:
            percentile_text = "(<50th %ile)"
        elif recovery_hours <= 3.2:
            percentile_text = "(<75th %ile)"
        elif recovery_hours <= 4.0:
            percentile_text = "(<85th %ile)"
        else:
            percentile_text = "(>85th %ile)"
        
        # Formatting
        ax.set_xlim(0, max_time)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(['0h', '1h', '2h', '3h', '4h+'])
        ax.set_yticks([])
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_title(f'{emoji} {scenario["title"]} • {speed_text} Recovery {percentile_text}', 
                    fontsize=11, pad=8)
        
        # Add key info on the side
        peak = max(glucose_curve)
        ax.text(max_time + 0.15, 0, f'Peak: {peak:.0f} mg/dL\nBaseline: {baseline:.0f} mg/dL', 
               va='center', fontsize=8, 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        
        results.append({
            'title': scenario['title'],
            'recovery_hours': recovery_hours,
            'speed_category': speed_text,
            'percentile': percentile_text
        })
    
    # Add scale explanation
    plt.figtext(0.5, 0.02, 
               '4-Hour Scale Rationale: Covers 86% of real recovery times from CGMacros dataset\n' +
               'Green: <50th %ile (Fast) • Orange: 50-75th %ile (Typical) • Red: 75-85th %ile (Slow) • Dark Red: >85th %ile (Extreme)',
               ha='center', va='bottom', fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.xlabel('Time to Return to Baseline (hours)', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('optimized_recovery_timeline_4h_scale.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary with percentile context
    print(f"\n📊 OPTIMIZED 4-HOUR SCALE RESULTS:")
    print("-" * 60)
    for result in results:
        print(f"{result['title']:25s}: {result['recovery_hours']:4.1f}h • {result['speed_category']:7s} {result['percentile']}")
    
    print(f"\n💡 SCALE BENEFITS:")
    print(f"   ✓ Covers 86% of real recovery cases")
    print(f"   ✓ Clear visual distinction between fast/typical/slow")
    print(f"   ✓ Extreme cases (>4h) clearly marked with '+' indicator") 
    print(f"   ✓ Data-driven thresholds based on CGMacros percentiles")
    
    print(f"\n📁 Optimized timeline saved as 'optimized_recovery_timeline_4h_scale.png'")

if __name__ == "__main__":
    demo_optimized_scale()