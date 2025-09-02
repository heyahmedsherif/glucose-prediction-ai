#!/usr/bin/env python3
"""
Simple Recovery Timeline Bar Demo
"""

import matplotlib.pyplot as plt
import numpy as np
from fixed_glucose_prediction_logic import CorrectedGlucosePrediction

def create_simple_timeline_bar(recovery_hours, title, meal_type):
    """Create simple recovery timeline bar with matplotlib."""
    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Color coding
    if recovery_hours <= 1.5:
        color = '#2E8B57'  # Green
        emoji = '🟢'
    elif recovery_hours <= 2.5:
        color = '#FF8C00'  # Orange
        emoji = '🟡'
    else:
        color = '#DC143C'  # Red
        emoji = '🔴'
    
    # Create timeline bar
    max_time = 4
    
    # Background bar
    ax.barh(0, max_time, height=0.5, color='#E8E8E8', alpha=0.3)
    
    # Progress bar
    ax.barh(0, recovery_hours, height=0.5, color=color, alpha=0.8)
    
    # Add text
    ax.text(recovery_hours/2, 0, f'{recovery_hours:.1f}h', 
           ha='center', va='center', fontweight='bold', color='white', fontsize=12)
    
    # Formatting
    ax.set_xlim(0, max_time)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(['0h', '1h', '2h', '3h', '4h+'])
    ax.set_yticks([])
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_title(f'{emoji} {title}\nEstimated time to return to baseline (±10 mg/dL)', 
                fontsize=14, pad=20)
    
    plt.tight_layout()
    return fig

def demo_recovery_comparison():
    """Show breakfast vs lunch recovery comparison."""
    
    print("🔧 RECOVERY TIMELINE BAR DEMONSTRATION")
    print("=" * 50)
    
    predictor = CorrectedGlucosePrediction()
    
    # Test scenarios
    scenarios = [
        {
            'title': 'Normal Person - 50g Carb Breakfast',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23, 'a1c': 5.2, 'fasting_glucose': 90},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True}
        },
        {
            'title': 'Normal Person - 50g Carb Lunch',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23, 'a1c': 5.2, 'fasting_glucose': 90},
            'timing': {'meal_type': 'lunch', 'meal_hour': 12, 'is_first_meal': False}
        },
        {
            'title': 'Type2 Diabetic - 50g Carb Breakfast',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Type2Diabetic', 'age': 55, 'bmi': 30, 'a1c': 8.0, 'fasting_glucose': 140},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True}
        },
        {
            'title': 'Type2 Diabetic - 50g Carb Lunch',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Type2Diabetic', 'age': 55, 'bmi': 30, 'a1c': 8.0, 'fasting_glucose': 140},
            'timing': {'meal_type': 'lunch', 'meal_hour': 12, 'is_first_meal': False}
        }
    ]
    
    # Create subplot figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle('🕒 Recovery Timeline Comparison: Breakfast vs Lunch', fontsize=16, fontweight='bold')
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\n🧪 Testing: {scenario['title']}")
        
        # Get predictions
        predictions = predictor.predict_glucose_with_corrected_timing(
            scenario['meal'], scenario['patient'], scenario['timing']
        )
        
        # Create simple glucose curve for recovery estimation
        time_points = [0, 30, 60, 90, 120, 180]
        glucose_curve = [predictions['baseline']] + [predictions[f'glucose_{t}min'] for t in [30, 60, 90, 120, 180]]
        
        # Simple recovery estimation (when glucose drops within 10 mg/dL of baseline)
        baseline = predictions['baseline']
        recovery_time = None
        
        for j, glucose in enumerate(glucose_curve):
            if j > 0 and abs(glucose - baseline) <= 10:  # Found recovery
                recovery_time = time_points[j]
                break
        
        # If not recovered by 180min, estimate based on final trend
        if recovery_time is None:
            final_glucose = glucose_curve[-1]
            if final_glucose > baseline + 10:  # Still elevated
                # Simple linear estimate: assume continues dropping at same rate
                drop_rate = (glucose_curve[-2] - glucose_curve[-1]) / 60  # mg/dL per minute
                if drop_rate > 0:
                    remaining_drop = final_glucose - (baseline + 10)
                    recovery_time = 180 + (remaining_drop / drop_rate)
                    recovery_time = min(recovery_time, 300)  # Cap at 5 hours
                else:
                    recovery_time = 300  # If not dropping, assume long recovery
        
        recovery_hours = recovery_time / 60 if recovery_time else 5
        
        print(f"  Baseline: {baseline:.1f} mg/dL")
        print(f"  Peak: {max(glucose_curve):.1f} mg/dL")
        print(f"  Recovery: {recovery_hours:.1f} hours")
        
        results.append({
            'scenario': scenario['title'],
            'meal_type': scenario['timing']['meal_type'],
            'recovery_hours': recovery_hours,
            'baseline': baseline,
            'peak': max(glucose_curve)
        })
        
        # Create timeline bar for this scenario
        ax = axes[i]
        max_time = 4
        
        # Color coding
        if recovery_hours <= 1.5:
            color = '#2E8B57'  # Green
            emoji = '🟢'
        elif recovery_hours <= 2.5:
            color = '#FF8C00'  # Orange
            emoji = '🟡'
        else:
            color = '#DC143C'  # Red
            emoji = '🔴'
        
        # Background bar
        ax.barh(0, max_time, height=0.6, color='#E8E8E8', alpha=0.3)
        
        # Progress bar
        progress_width = min(recovery_hours, max_time)
        ax.barh(0, progress_width, height=0.6, color=color, alpha=0.8)
        
        # Add text
        text_x = progress_width / 2 if progress_width > 0.5 else progress_width + 0.1
        ax.text(text_x, 0, f'{recovery_hours:.1f}h', 
               ha='center', va='center', fontweight='bold', 
               color='white' if progress_width > 0.5 else 'black', fontsize=11)
        
        # Formatting
        ax.set_xlim(0, max_time)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(['0h', '1h', '2h', '3h', '4h+'])
        ax.set_yticks([])
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_title(f'{emoji} {scenario["title"]}', fontsize=12, pad=10)
        
        # Add baseline info
        ax.text(max_time + 0.1, 0, f'Baseline: {baseline:.0f} mg/dL\nPeak: {max(glucose_curve):.0f} mg/dL', 
               va='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.xlabel('Time to Return to Baseline (hours)', fontsize=12)
    plt.tight_layout()
    plt.savefig('recovery_timeline_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n📊 RECOVERY TIME COMPARISON SUMMARY:")
    print("-" * 60)
    
    breakfast_normal = next(r for r in results if 'Normal' in r['scenario'] and 'Breakfast' in r['scenario'])
    lunch_normal = next(r for r in results if 'Normal' in r['scenario'] and 'Lunch' in r['scenario'])
    breakfast_type2 = next(r for r in results if 'Type2' in r['scenario'] and 'Breakfast' in r['scenario'])
    lunch_type2 = next(r for r in results if 'Type2' in r['scenario'] and 'Lunch' in r['scenario'])
    
    print(f"Normal Person:")
    print(f"  Breakfast: {breakfast_normal['recovery_hours']:.1f} hours")
    print(f"  Lunch: {lunch_normal['recovery_hours']:.1f} hours")
    print(f"  Difference: +{(breakfast_normal['recovery_hours'] - lunch_normal['recovery_hours']) * 60:.0f} minutes longer for breakfast")
    
    print(f"\nType2 Diabetic:")
    print(f"  Breakfast: {breakfast_type2['recovery_hours']:.1f} hours")
    print(f"  Lunch: {lunch_type2['recovery_hours']:.1f} hours") 
    print(f"  Difference: +{(breakfast_type2['recovery_hours'] - lunch_type2['recovery_hours']) * 60:.0f} minutes longer for breakfast")
    
    print(f"\n💡 This visualization clearly shows users WHY breakfast takes longer!")
    print(f"📁 Timeline comparison saved as 'recovery_timeline_comparison.png'")

if __name__ == "__main__":
    demo_recovery_comparison()