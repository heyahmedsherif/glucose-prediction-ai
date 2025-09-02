#!/usr/bin/env python3
"""
Recovery Timeline Bar Demo

Demonstrates the visual recovery timeline bar for different meal scenarios.
Shows how breakfast vs lunch recovery times are displayed to end users.
"""

import plotly.graph_objects as go
import plotly.express as px
from fixed_glucose_prediction_logic import CorrectedGlucosePrediction
import webbrowser
import tempfile
import os

def estimate_recovery_time(spike_curve, time_points, baseline_glucose):
    """Estimate time to return to baseline (±10 mg/dL) based on glucose curve."""
    
    baseline_threshold = 10  # mg/dL tolerance
    recovery_time = None
    
    # Find peak time first
    peak_glucose = max(spike_curve)
    peak_idx = spike_curve.index(peak_glucose)
    peak_time = time_points[peak_idx]
    
    # Look for baseline recovery after peak
    for i in range(peak_idx + 1, len(spike_curve)):
        if abs(spike_curve[i] - baseline_glucose) <= baseline_threshold:
            recovery_time = time_points[i]
            break
    
    # If no recovery in observed window, estimate based on trend
    if recovery_time is None and len(spike_curve) >= 3:
        # Calculate slope from last 3 points
        last_points = spike_curve[-3:]
        last_times = time_points[-3:]
        
        if len(set(last_points)) > 1:  # Avoid division by zero
            # Linear extrapolation from trend
            slope = (last_points[-1] - last_points[0]) / (last_times[-1] - last_times[0])
            
            if slope < 0:  # Glucose is declining
                remaining_drop = last_points[-1] - (baseline_glucose + baseline_threshold)
                if remaining_drop > 0:
                    time_to_recovery = remaining_drop / abs(slope)
                    recovery_time = last_times[-1] + time_to_recovery
                    recovery_time = min(recovery_time, 300)  # Cap at 5 hours
    
    return recovery_time, peak_time

def create_recovery_timeline_bar(recovery_time_minutes, meal_type, diabetic_status, scenario_title):
    """Create a visual recovery timeline bar."""
    
    if recovery_time_minutes is None:
        return None
    
    # Convert to hours for display
    recovery_hours = recovery_time_minutes / 60
    
    # Create progress bar data
    max_time = 4  # 4 hour scale
    
    # Color coding based on recovery speed
    if recovery_hours <= 1.5:
        bar_color = "#2E8B57"  # Green - fast recovery
        status_emoji = "🟢"
    elif recovery_hours <= 2.5:
        bar_color = "#FF8C00"  # Orange - moderate recovery  
        status_emoji = "🟡"
    else:
        bar_color = "#DC143C"  # Red - slow recovery
        status_emoji = "🔴"
    
    # Create the timeline bar visualization
    fig = go.Figure()
    
    # Background bar (full timeline)
    fig.add_trace(go.Bar(
        x=[max_time], y=[scenario_title], 
        orientation='h', 
        marker_color='#E8E8E8',
        name='Timeline',
        text='', textposition='none',
        hoverinfo='none',
        opacity=0.3
    ))
    
    # Progress bar (recovery time)
    fig.add_trace(go.Bar(
        x=[recovery_hours], y=[scenario_title],
        orientation='h',
        marker_color=bar_color,
        name='Recovery Time',
        text=f'{recovery_hours:.1f}h',
        textposition='inside',
        textfont=dict(color='white', size=12, family='Arial Black'),
        hovertemplate=f'Recovery Time: {recovery_hours:.1f} hours<extra></extra>'
    ))
    
    # Add time markers
    for hour in range(0, 5):
        fig.add_vline(x=hour, line_width=1, line_color="gray", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{status_emoji} Recovery Timeline: {scenario_title}<br><sub>{meal_type.capitalize()} • {diabetic_status} • Time to baseline (±10 mg/dL)</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis=dict(
            title="Time (hours)",
            range=[0, max_time],
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['0h', '1h', '2h', '3h', '4h'],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showticklabels=True,
            range=[-0.5, 0.5]
        ),
        height=200,
        showlegend=False,
        margin=dict(l=150, r=50, t=100, b=50),
        barmode='overlay',
        plot_bgcolor='white'
    )
    
    return fig

def demo_recovery_timelines():
    """Demonstrate recovery timeline bars for different scenarios."""
    
    print("🔧 RECOVERY TIMELINE BAR DEMONSTRATION")
    print("=" * 50)
    
    predictor = CorrectedGlucosePrediction()
    
    # Test scenarios comparing breakfast vs lunch
    scenarios = [
        {
            'title': '50g Carb Breakfast (Normal Person)',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23, 'a1c': 5.2, 'fasting_glucose': 90},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True}
        },
        {
            'title': '50g Carb Lunch (Normal Person)', 
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Normal', 'age': 35, 'bmi': 23, 'a1c': 5.2, 'fasting_glucose': 90},
            'timing': {'meal_type': 'lunch', 'meal_hour': 12, 'is_first_meal': False}
        },
        {
            'title': '50g Carb Breakfast (Type2 Diabetic)',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Type2Diabetic', 'age': 55, 'bmi': 30, 'a1c': 8.0, 'fasting_glucose': 140},
            'timing': {'meal_type': 'breakfast', 'meal_hour': 8, 'is_first_meal': True}
        },
        {
            'title': '50g Carb Lunch (Type2 Diabetic)',
            'meal': {'carbohydrates': 50, 'protein': 20, 'fat': 10, 'fiber': 5},
            'patient': {'diabetic_status': 'Type2Diabetic', 'age': 55, 'bmi': 30, 'a1c': 8.0, 'fasting_glucose': 140},
            'timing': {'meal_type': 'lunch', 'meal_hour': 12, 'is_first_meal': False}
        }
    ]
    
    # Create HTML file with all timeline bars
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Glucose Recovery Timeline Demo</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }
            .scenario { margin: 30px 0; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .comparison { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>🕒 Glucose Recovery Timeline Bars</h1>
        <div class="comparison">
            <strong>Comparison Purpose:</strong> This demonstrates why breakfast meals take longer to return to baseline than lunch meals, 
            even with identical macronutrients. The visual timeline bars show recovery duration differences.
        </div>
    """
    
    timeline_charts = []
    
    for scenario in scenarios:
        print(f"\n🧪 Testing: {scenario['title']}")
        
        # Get predictions
        predictions = predictor.predict_glucose_with_corrected_timing(
            scenario['meal'], scenario['patient'], scenario['timing']
        )
        
        # Create glucose curve
        time_points = [0, 30, 60, 90, 120, 180]
        glucose_curve = [predictions['baseline']] + [predictions[f'glucose_{t}min'] for t in [30, 60, 90, 120, 180]]
        
        # Estimate recovery time
        recovery_time, peak_time = estimate_recovery_time(glucose_curve, time_points, predictions['baseline'])
        
        print(f"  Baseline: {predictions['baseline']:.1f} mg/dL")
        print(f"  Peak: {max(glucose_curve):.1f} mg/dL at {time_points[glucose_curve.index(max(glucose_curve))]} min")
        if recovery_time:
            print(f"  Recovery: {recovery_time:.1f} minutes ({recovery_time/60:.1f} hours)")
        else:
            print(f"  Recovery: No baseline recovery in observed timeframe")
        
        # Create timeline bar
        timeline_fig = create_recovery_timeline_bar(
            recovery_time, scenario['timing']['meal_type'], 
            scenario['patient']['diabetic_status'], scenario['title']
        )
        
        if timeline_fig:
            timeline_charts.append(timeline_fig.to_html(include_plotlyjs=False, div_id=f"timeline_{len(timeline_charts)}"))
    
    # Add all charts to HTML
    for i, chart_html in enumerate(timeline_charts):
        html_content += f'<div class="scenario">{chart_html}</div>'
    
    html_content += """
        <div class="comparison" style="margin-top: 30px;">
            <h3>🔍 Key Observations:</h3>
            <ul>
                <li><strong>Dawn Phenomenon:</strong> Morning insulin resistance makes breakfast recovery slower</li>
                <li><strong>First Meal Effect:</strong> Body needs time to activate glucose processing after overnight fast</li>
                <li><strong>Diabetic Status Impact:</strong> Type2 diabetics show consistently longer recovery times</li>
                <li><strong>Meal Timing Optimization:</strong> Lunch shows optimal insulin sensitivity window</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Save and open HTML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html_content)
        html_file = f.name
    
    print(f"\n📊 Opening recovery timeline demonstration...")
    webbrowser.open(f'file://{html_file}')
    print(f"📁 Timeline demo saved to: {html_file}")
    
    return html_file

if __name__ == "__main__":
    demo_recovery_timelines()