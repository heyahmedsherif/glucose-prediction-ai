#!/bin/bash

echo "🚀 Starting Stratified Glucose Response Streamlit App"
echo "======================================================"

# Check if required files exist
if [ ! -d "CGMacros" ]; then
    echo "❌ Error: CGMacros directory not found"
    echo "Please ensure the CGMacros data directory is in the current location"
    exit 1
fi

if [ ! -f "CGMacros/bio.csv" ]; then
    echo "❌ Error: bio.csv not found in CGMacros directory"
    exit 1
fi

echo "✅ Data files found"
echo "📊 Launching Streamlit app..."
echo ""
echo "The app will be available at: http://localhost:8505"
echo "Press Ctrl+C to stop the app"
echo ""

# Run streamlit app
streamlit run stratified_glucose_app.py --server.port 8505