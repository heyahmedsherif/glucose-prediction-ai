#!/usr/bin/env python3
"""
Model Feature Importance Analysis

Analyze which features are most important for glucose prediction in the trained models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from collections import defaultdict

def analyze_feature_importance():
    """Analyze feature importance across all trained glucose prediction models."""
    
    print("🎯 Analyzing Feature Importance in Glucose Prediction Models")
    print("=" * 60)
    
    # Load model metadata
    metadata_path = "glucose_prediction_models/model_metadata.json"
    if not os.path.exists(metadata_path):
        print("❌ Model metadata not found")
        return
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load all models and analyze feature importance
    model_dir = "glucose_prediction_models"
    time_points = ['30min', '60min', '90min', '120min', '180min']
    
    all_importances = defaultdict(list)
    model_performance = {}
    
    for time_point in time_points:
        model_path = os.path.join(model_dir, f"glucose_{time_point}_model.joblib")
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                
                # Get feature names and importances
                if time_point in metadata['model_info']:
                    feature_names = metadata['model_info'][time_point]['feature_names']
                    r2_score = metadata['model_info'][time_point]['r2_score']
                    mae = metadata['model_info'][time_point]['mae']
                    
                    model_performance[time_point] = {
                        'r2': r2_score,
                        'mae': mae,
                        'n_features': len(feature_names)
                    }
                    
                    # Get feature importances (assuming RandomForest)
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        
                        print(f"\n📊 {time_point.upper()} MODEL FEATURE IMPORTANCE:")
                        print(f"   R² = {r2_score:.3f}, MAE = {mae:.1f} mg/dL")
                        print("   " + "-" * 45)
                        
                        # Sort features by importance
                        feature_importance = list(zip(feature_names, importances))
                        feature_importance.sort(key=lambda x: x[1], reverse=True)
                        
                        for i, (feature, importance) in enumerate(feature_importance[:10]):  # Top 10
                            print(f"   {i+1:2d}. {feature:<25} {importance:.3f} ({importance*100:.1f}%)")
                            all_importances[feature].append(importance)
                        
                        if len(feature_importance) > 10:
                            print(f"   ... and {len(feature_importance)-10} more features")
                            
                    else:
                        print(f"⚠️  {time_point} model doesn't have feature_importances_ attribute")
                        
            except Exception as e:
                print(f"❌ Error loading {time_point} model: {e}")
    
    # Calculate average feature importance across all models
    print(f"\n🏆 OVERALL FEATURE IMPORTANCE RANKINGS")
    print("=" * 50)
    
    avg_importances = {}
    for feature, importances in all_importances.items():
        if len(importances) > 0:
            avg_importances[feature] = {
                'mean': np.mean(importances),
                'std': np.std(importances),
                'models_used': len(importances),
                'max': np.max(importances),
                'min': np.min(importances)
            }
    
    # Sort by average importance
    sorted_features = sorted(avg_importances.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"{'Rank':<4} {'Feature':<25} {'Avg Imp':<8} {'±Std':<8} {'Models':<6} {'Range'}")
    print("-" * 65)
    
    for i, (feature, stats) in enumerate(sorted_features[:15]):  # Top 15
        print(f"{i+1:2d}.  {feature:<25} {stats['mean']:.3f}    {stats['std']:.3f}    {stats['models_used']}/5     {stats['min']:.3f}-{stats['max']:.3f}")
    
    # Categorize features for business insights
    print(f"\n💼 BUSINESS INSIGHTS - Feature Categories")
    print("=" * 50)
    
    categories = {
        'Meal Composition': ['carbohydrates', 'protein', 'fat', 'fiber', 'calories'],
        'Patient Baseline': ['diabetic_status_encoded', 'a1c', 'fasting_glucose', 'fasting_insulin', 'baseline'],
        'Demographics': ['age', 'gender', 'bmi'],
        'Physical Activity': ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 'active_minutes', 'hr_mean']
    }
    
    category_importance = defaultdict(list)
    
    for feature, stats in avg_importances.items():
        for category, features in categories.items():
            if feature in features:
                category_importance[category].append(stats['mean'])
                break
        else:
            category_importance['Other'].append(stats['mean'])
    
    # Calculate category averages
    category_averages = {}
    for category, importances in category_importance.items():
        if importances:
            category_averages[category] = {
                'mean': np.mean(importances),
                'total': np.sum(importances),
                'count': len(importances)
            }
    
    # Sort categories by total importance
    sorted_categories = sorted(category_averages.items(), key=lambda x: x[1]['total'], reverse=True)
    
    for category, stats in sorted_categories:
        print(f"{category:<20} Total: {stats['total']:.3f} | Avg: {stats['mean']:.3f} | Features: {stats['count']}")
    
    # Top insights
    print(f"\n🎯 KEY INSIGHTS FOR GLUCOSE PREDICTION")
    print("=" * 45)
    
    if sorted_features:
        top_feature = sorted_features[0]
        print(f"🥇 #1 Predictor: {top_feature[0]} (Importance: {top_feature[1]['mean']:.3f})")
        
        # Find top 3 from each category
        print(f"\n📊 Top Predictors by Category:")
        
        for category, features in categories.items():
            category_features = [(f, stats) for f, stats in sorted_features if f in features]
            if category_features:
                top_in_category = category_features[0]
                print(f"   • {category}: {top_in_category[0]} ({top_in_category[1]['mean']:.3f})")
    
    # Model performance summary
    print(f"\n⚡ MODEL PERFORMANCE SUMMARY")
    print("=" * 35)
    
    for time_point, perf in model_performance.items():
        print(f"{time_point:>8}: R² = {perf['r2']:.3f}, MAE = {perf['mae']:.1f} mg/dL ({perf['n_features']} features)")
    
    if model_performance:
        avg_r2 = np.mean([p['r2'] for p in model_performance.values()])
        avg_mae = np.mean([p['mae'] for p in model_performance.values()])
        print(f"{'Average':>8}: R² = {avg_r2:.3f}, MAE = {avg_mae:.1f} mg/dL")
    
    # Practical recommendations
    print(f"\n💡 PRACTICAL RECOMMENDATIONS")
    print("=" * 30)
    
    if sorted_features:
        # Get top 5 most important features
        top_5 = [f[0] for f in sorted_features[:5]]
        
        practical_advice = {
            'carbohydrates': 'Carb counting is crucial - biggest glucose driver',
            'baseline': 'Pre-meal glucose sets the foundation for response',
            'diabetic_status_encoded': 'Diabetic status fundamentally changes glucose handling',
            'a1c': 'Long-term glucose control (A1c) predicts meal responses', 
            'fasting_glucose': 'Fasting glucose indicates baseline metabolic health',
            'fiber': 'Fiber intake significantly moderates glucose spikes',
            'protein': 'Protein has delayed but measurable glucose impact',
            'fat': 'Fat affects glucose indirectly through delayed absorption',
            'bmi': 'Weight status correlates with insulin sensitivity',
            'age': 'Older adults typically have higher glucose responses'
        }
        
        print("For accurate glucose predictions, prioritize collecting:")
        for i, feature in enumerate(top_5):
            if feature in practical_advice:
                print(f"{i+1}. {feature}: {practical_advice[feature]}")
            else:
                print(f"{i+1}. {feature}: High predictive importance")
    
    return avg_importances, model_performance

if __name__ == "__main__":
    analyze_feature_importance()