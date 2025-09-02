# 🍎 Glucose Prediction Streamlit App

A user-friendly web application for predicting blood glucose responses after meals using machine learning models trained on continuous glucose monitoring data.

## 🚀 Features

- **Interactive Interface**: Easy-to-use sidebar for inputting personal and meal information
- **Real-time Predictions**: Glucose predictions at 30, 60, 90, 120, and 180 minutes after eating
- **Visual Results**: Interactive glucose response curve with normal ranges
- **Meal Presets**: Quick selection of common meal types
- **Activity Integration**: Optional inclusion of physical activity data
- **Risk Assessment**: Color-coded alerts based on predicted glucose levels
- **Model Transparency**: Information about model accuracy and training data

## 📋 Requirements

- Python 3.8+
- All dependencies listed in `requirements_streamlit.txt`
- Trained glucose prediction models (in `glucose_prediction_models/` directory)

## 🛠️ Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Ensure models are available:**
   - The app expects trained models in `glucose_prediction_models/` directory
   - Run `train_and_save_glucose_models.py` first if models don't exist

## 🎯 How to Run

### Method 1: Using the startup script
```bash
./run_app.sh
```

### Method 2: Direct Streamlit command
```bash
streamlit run glucose_prediction_app.py
```

### Method 3: With conda environment
```bash
conda run -n cgmacros streamlit run glucose_prediction_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 📱 How to Use the App

### 1. Personal Information
- Enter your age, gender, height, and weight
- BMI is automatically calculated

### 2. Current Glucose Level
- Input your current blood glucose reading (mg/dL)
- This serves as the baseline for predictions

### 3. Meal Information
- **Option A**: Choose from meal presets (Small/Large Breakfast, Light/Heavy Lunch, etc.)
- **Option B**: Enter custom macronutrients:
  - Carbohydrates (g)
  - Protein (g) 
  - Fat (g)
  - Fiber (g)
  - Total calories

### 4. Activity Data (Optional)
- Enable "Include activity data" checkbox
- Enter steps taken in 30-minute window around meal
- Enter average heart rate during that period

### 5. Get Predictions
- Click "🔮 Predict Glucose Response" button
- View interactive glucose curve
- See summary metrics and risk assessment
- Review detailed predictions table

## 📊 Understanding the Results

### Glucose Curve
- Shows predicted glucose levels over 3 hours
- Color-coded reference lines for normal ranges:
  - **Green dashed (70 mg/dL)**: Normal low
  - **Orange dashed (140 mg/dL)**: Normal high  
  - **Red dashed (180 mg/dL)**: High risk threshold

### Summary Metrics
- **Peak Glucose**: Highest predicted glucose level
- **Time to Peak**: When maximum glucose occurs
- **Return to Baseline**: Estimated time to return near starting glucose

### Risk Assessment
- **✅ Normal**: Peak < 140 mg/dL
- **⚠️ Elevated**: Peak 140-180 mg/dL
- **🚨 High**: Peak > 180 mg/dL

## 🤖 Model Information

### Training Data
- **1,269 meal records** from **42 individuals**
- Continuous glucose monitoring data from CGMacros dataset
- Includes breakfast, lunch, and dinner meals

### Model Performance
- **30-minute predictions**: 14.4 mg/dL average error
- **60-minute predictions**: 21.6 mg/dL average error
- **90-minute predictions**: 24.9 mg/dL average error
- **120-minute predictions**: 24.3 mg/dL average error
- **180-minute predictions**: 22.3 mg/dL average error

### Algorithm
- **RandomForest** models with optimal feature selection
- **Leave-one-subject-out** cross-validation for robust evaluation
- **Standardized preprocessing** for consistent predictions

## 🎛️ Customization

### Adding New Meal Presets
Edit the `meal_presets` dictionary in `glucose_prediction_app.py`:

```python
meal_presets = {
    "Your Meal Name": {
        "carbs": 45, 
        "protein": 20, 
        "fat": 12, 
        "fiber": 4, 
        "calories": 350
    }
}
```

### Modifying Display Options
- Adjust plot colors in `create_glucose_curve_plot()`
- Customize reference lines for different glucose targets
- Modify risk thresholds in the risk assessment logic

## ⚠️ Important Disclaimers

1. **Educational Purpose Only**: This app is for educational and research purposes
2. **Not Medical Advice**: Do not use for medical decision-making
3. **Consult Healthcare Providers**: Always work with medical professionals for diabetes management
4. **Individual Variation**: Actual glucose responses may vary significantly between individuals
5. **Model Limitations**: Predictions are based on population data and may not reflect individual physiology

## 🔧 Troubleshooting

### Common Issues

**"Models directory not found" error:**
- Ensure `glucose_prediction_models/` directory exists
- Run `train_and_save_glucose_models.py` to create models

**Import errors:**
- Install all dependencies: `pip install -r requirements_streamlit.txt`
- Check Python environment compatibility

**App won't start:**
- Try a different port: `streamlit run glucose_prediction_app.py --server.port 8502`
- Clear Streamlit cache: `streamlit cache clear`

**Predictions seem incorrect:**
- Verify input units (grams for macronutrients, mg/dL for glucose)
- Check that baseline glucose is reasonable (60-200 mg/dL range)

## 📞 Support

For technical issues or questions about the glucose prediction models:
1. Check this README file
2. Review model training scripts and documentation
3. Verify input data formats and ranges

## 🙏 Acknowledgments

- **CGMacros Dataset**: Scientific dataset for personalized nutrition research
- **Streamlit**: Open-source framework for data applications
- **Plotly**: Interactive visualization library
- **Scikit-learn**: Machine learning framework

---

**Built with ❤️ for diabetes research and education**