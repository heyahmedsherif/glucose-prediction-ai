# CGMacros - Personalized Glucose Prediction Dataset & Applications

A comprehensive scientific dataset and application suite for personalized nutrition and diet monitoring using continuous glucose monitoring (CGM) data.

## 📁 Repository Organization

The repository has been organized into logical folders for better maintainability and navigation:

### Core Application Folders

- **`apps/`** - Streamlit applications for glucose prediction
  - `enhanced_glucose_app_with_timing.py` - Main enhanced app with 3-hour baseline return
  - `glucose_prediction_app_simple.py` - Simplified interface with core features
  - `glucose_prediction_app_enhanced.py` - Full-featured enhanced application

### Data & Models

- **`data/`** - Datasets and processed CSV files
  - Core baseline lookup tables and analysis results
  - Excludes large datasets (see .gitignore for details)

- **`models/`** - Machine learning models and artifacts
  - `glucose_prediction_models/` - Trained model files (.joblib, .pkl)
  - Model training outputs and serialized predictors

### Development & Analysis

- **`analysis/`** - Research scripts and data analysis notebooks
  - Exploratory data analysis and visualization scripts
  - Research and showcase demonstrations

- **`scripts/`** - Data processing and model training utilities
  - Preprocessing pipelines and feature engineering
  - Model training and evaluation scripts

- **`tests/`** - Test suites and validation scripts
  - Unit tests and integration tests
  - Model validation and performance testing

### Documentation & Support

- **`docs/`** - Documentation files and images
  - Technical documentation and guides
  - Image assets for documentation (preserved from .gitignore exclusions)

- **`backups/`** - Backup files and version history
  - Previous versions and backup copies
  - Development snapshots and recovery files

## 🚀 Quick Start

### Running the Applications

Use the provided shell scripts for easy application startup:

```bash
# Main Enhanced App (recommended)
./run_main_app.sh

# Simple Interface App
./run_simple_app.sh
```

Or run directly with Streamlit:

```bash
# Main Enhanced App
streamlit run apps/enhanced_glucose_app_with_timing.py

# Simple App  
streamlit run apps/glucose_prediction_app_simple.py
```

### Key Features

- **3-Hour Baseline Return**: All glucose predictions return to baseline within 3 hours (180 minutes) with ±10% biological variation
- **Diabetic Status Personalization**: Customized predictions for Normal, Pre-diabetic, and Type 2 Diabetic patients
- **Realistic Biological Variation**: Deterministic randomization ensures consistent results for same inputs
- **Comprehensive Food Analysis**: Detailed macronutrient impact modeling

## 🔬 Technical Details

### Glucose Prediction Model

The core prediction model implements:
- Time-based decay multipliers allowing complete baseline return at 180 minutes
- Diabetic status-specific response curves
- Deterministic randomization using patient characteristics as seed
- Biological variation within ±8% tolerance window

### Data Pipeline

- Preprocessed CGM data with meal timing analysis
- Baseline glucose lookup tables by diabetic status
- Feature engineering for personalized predictions

## 📋 Development

### File Structure Benefits

- **Separation of Concerns**: Apps, data, models, and utilities are clearly separated
- **Easy Navigation**: Logical grouping makes finding files intuitive
- **Scalability**: New components can be added to appropriate folders
- **Clean Root**: Reduced clutter in main directory

### Configuration

- `.gitignore` updated to reflect new folder structure
- Large datasets and model files excluded from version control
- Backup and temporary files properly ignored

## 📚 Documentation

For detailed technical documentation and development history, see:
- `CLAUDE_LOG.md` - Complete interaction and development log
- `BASELINE_RETURN_MODIFICATION_LOG.md` - Technical modification details

## 🧪 Testing

Run validation tests to ensure model accuracy:

```bash
python tests/test_baseline_return_fix.py
```

## 📊 Data Sources

Built on the **CGMacros scientific dataset** for personalized nutrition and diet monitoring, providing comprehensive glucose response data across different demographic groups and meal compositions.

### Dataset Citation

This project uses the CGMacros dataset developed by the Phenotype Science Initiative (PSI) at Texas A&M University.

**Original Dataset:** [CGMacros - A scientific dataset for personalized nutrition and diet monitoring](https://github.com/PSI-TAMU/CGMacros)

**Citation:** If you use this system or reference this work, please cite the original CGMacros dataset creators and their research.