#!/bin/bash
# Run the Simple Glucose Prediction App

echo "🎯 Starting Simple Glucose Prediction App..."
echo "📊 Streamlined interface with core prediction features"
echo ""

cd "$(dirname "$0")"
streamlit run apps/glucose_prediction_app_simple.py