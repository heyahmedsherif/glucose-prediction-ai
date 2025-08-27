#!/usr/bin/env python3

import joblib
import json
import numpy as np
from collections import defaultdict

# Load model metadata
with open('glucose_prediction_models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print('🎯 GLUCOSE PREDICTION MODEL - FEATURE IMPORTANCE ANALYSIS')
print('=' * 60)

all_importances = defaultdict(list)
time_points = ['30min', '60min', '90min', '120min', '180min']

for time_point in time_points:
    try:
        model = joblib.load(f'glucose_prediction_models/glucose_{time_point}_model.joblib')
        model_info = metadata['model_info'][f'glucose_{time_point}']
        feature_names = model_info['feature_names']
        r2_score = model_info['r2_score']
        mae = model_info['mae']
        
        print(f'\n📊 {time_point.upper()} MODEL (R² = {r2_score:.3f}, MAE = {mae:.1f} mg/dL)')
        print('-' * 55)
        
        importances = model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(feature_importance[:8]):
            print(f'{i+1:2d}. {feature:<25} {importance:.3f} ({importance*100:.1f}%)')
            all_importances[feature].append(importance)
            
    except Exception as e:
        print(f'❌ Error with {time_point}: {e}')

# Overall rankings
print(f'\n🏆 OVERALL FEATURE IMPORTANCE RANKINGS')
print('=' * 50)

avg_importances = {}
for feature, importances in all_importances.items():
    if len(importances) > 0:
        avg_importances[feature] = np.mean(importances)

sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)

print('Rank Feature                   Avg Importance   % of Total')
print('-' * 55)

total_importance = sum(avg_importances.values())
for i, (feature, importance) in enumerate(sorted_features):
    pct = (importance / total_importance) * 100 if total_importance > 0 else 0
    print(f'{i+1:2d}.  {feature:<25} {importance:.3f}         {pct:.1f}%')

print(f'\n💼 BUSINESS INSIGHTS')
print('=' * 30)

# Categorize top features
top_10 = dict(sorted_features[:10])
meal_features = ['carbohydrates', 'protein', 'fat', 'fiber', 'calories']
patient_features = ['diabetic_status_encoded', 'a1c', 'fasting_glucose', 'fasting_insulin', 'baseline', 'age', 'gender', 'bmi']
activity_features = ['steps_total', 'steps_mean_per_minute', 'steps_max_per_minute', 'active_minutes', 'hr_mean']

meal_importance = sum(imp for feat, imp in top_10.items() if feat in meal_features)
patient_importance = sum(imp for feat, imp in top_10.items() if feat in patient_features)
activity_importance = sum(imp for feat, imp in top_10.items() if feat in activity_features)

total_top10 = sum(top_10.values())
print(f'🍽️  Meal Factors:     {meal_importance:.3f} ({(meal_importance/total_top10)*100:.1f}%)')
print(f'👤 Patient Factors:  {patient_importance:.3f} ({(patient_importance/total_top10)*100:.1f}%)')  
print(f'🏃 Activity Factors: {activity_importance:.3f} ({(activity_importance/total_top10)*100:.1f}%)')

print(f'\n🎯 KEY TAKEAWAYS:')
if sorted_features:
    top_3 = sorted_features[:3]
    print(f'1. #{sorted_features[0][0]} is the strongest predictor ({sorted_features[0][1]:.3f})')
    print(f'2. #{sorted_features[1][0]} is second most important ({sorted_features[1][1]:.3f})')
    print(f'3. #{sorted_features[2][0]} rounds out the top 3 ({sorted_features[2][1]:.3f})')
    
    print(f'\n💡 For glucose prediction apps, prioritize collecting:')
    for i, (feature, importance) in enumerate(sorted_features[:5]):
        print(f'{i+1}. {feature}')