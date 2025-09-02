#!/bin/bash
# Run the main Enhanced Glucose Prediction App

echo "🚀 Starting Enhanced Glucose Prediction App with 3-Hour Baseline Return..."
echo "🌾 Features: Diabetic status-based personalization, realistic biological variation"
echo ""

cd "$(dirname "$0")"
streamlit run apps/enhanced_glucose_app_with_timing.py