#!/bin/bash

echo "🚀 Starting ENHANCED Glucose Spike Streamlit App"
echo "================================================="

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
echo "🚀 Launching ENHANCED SPIKE app..."
echo ""
echo "🔥 This version emphasizes glucose SPIKES using upper confidence intervals"
echo "📈 Shows dramatic glucose spikes that were hidden in averaged curves"
echo ""
echo "The app will be available at: http://localhost:8506"
echo "Press Ctrl+C to stop the app"
echo ""

# Run enhanced spike app
streamlit run enhanced_spike_app.py --server.port 8506