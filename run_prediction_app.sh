#!/bin/bash

# Glucose Prediction Spike App Launcher
# Quick launcher for the glucose prediction app with spike visualization

echo "🍽️ Launching Glucose Prediction with Spike Visualization App"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "glucose_prediction_spike_app.py" ]; then
    echo "❌ Error: glucose_prediction_spike_app.py not found in current directory"
    echo "Please run this script from the CGMacros project directory"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: streamlit not found"
    echo "Please install streamlit: pip install streamlit"
    exit 1
fi

# Check if required data directory exists
if [ ! -d "CGMacros" ]; then
    echo "⚠️  Warning: CGMacros directory not found"
    echo "The app will run with simplified prediction models only"
fi

# Check if trained models exist
if [ ! -d "glucose_prediction_models" ]; then
    echo "⚠️  Warning: glucose_prediction_models directory not found"
    echo "The app will use simplified prediction equations"
else
    echo "✅ Found trained prediction models"
fi

echo ""
echo "🚀 Starting Streamlit app..."
echo "📱 The app will open in your default web browser"
echo "🛑 Press Ctrl+C to stop the app"
echo ""

# Launch the app
streamlit run glucose_prediction_spike_app.py --server.port 8506 --server.headless false