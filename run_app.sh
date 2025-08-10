#!/bin/bash

echo "🚀 Starting Glucose Prediction App..."
echo "📱 The app will open in your default web browser"
echo "🛑 Press Ctrl+C to stop the app"
echo ""

# Run the Streamlit app
/opt/miniconda3/bin/conda run -n cgmacros streamlit run glucose_prediction_app.py --server.port 8501 --server.address 0.0.0.0