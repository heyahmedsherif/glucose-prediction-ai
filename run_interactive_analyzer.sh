#!/bin/bash

echo "🎛️ Starting Interactive Glucose Spike Analyzer"
echo "=============================================="

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
echo "🎛️ Launching Interactive Spike Analyzer..."
echo ""
echo "🔥 FEATURES:"
echo "• Select multiple spike emphasis methods"
echo "• Compare methods side-by-side"
echo "• Custom standard deviation multipliers"
echo "• Interactive controls for all parameters"
echo "• Export analysis results"
echo ""
echo "The app will be available at: http://localhost:8507"
echo "Press Ctrl+C to stop the app"
echo ""

# Run interactive analyzer
streamlit run interactive_spike_analyzer.py --server.port 8507