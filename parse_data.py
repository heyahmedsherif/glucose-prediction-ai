#!/usr/bin/env python3
"""
CGM Data Analysis Script

This script processes continuous glucose monitoring (CGM) data and performs 
machine learning analysis to predict glucose response metrics (iAUC and AUC).

Author: Converted from Jupyter notebook
"""

import os
import logging
import argparse
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import zscore, pearsonr
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import shap
import scipy


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def area_under_curve(time_points: List[float], glucose_values: List[float]) -> float:
    """
    Calculate incremental area under the curve using trapezoidal rule.
    
    Args:
        time_points: List of time points
        glucose_values: List of glucose values
        
    Returns:
        float: Incremental area under curve
    """
    total = 0
    baseline = glucose_values[0]
    
    for i in range(len(time_points) - 1):
        if (glucose_values[i + 1] - baseline >= 0) and (glucose_values[i] - baseline >= 0):
            temp = ((glucose_values[i] - baseline + glucose_values[i + 1] - baseline) / 2) * (time_points[i + 1] - time_points[i])
        elif (glucose_values[i + 1] - baseline < 0) and (glucose_values[i] - baseline >= 0):
            temp = (glucose_values[i] - baseline) * ((glucose_values[i] - baseline) / (glucose_values[i] - glucose_values[i + 1]) * (time_points[i + 1] - time_points[i]) / 2)
        elif (glucose_values[i + 1] - baseline >= 0) and (glucose_values[i] - baseline < 0):
            temp = (glucose_values[i + 1] - baseline) * ((glucose_values[i + 1] - baseline) / (glucose_values[i + 1] - glucose_values[i]) * (time_points[i + 1] - time_points[i]) / 2)
        elif (glucose_values[i] - baseline < 0) and (glucose_values[i + 1] - baseline < 0):
            temp = 0
        total = total + temp
    return total


def calc_iauc(cgm_values: List[float], sampling_interval: List[float]) -> float:
    """
    Calculate incremental area under curve for CGM data.
    
    Args:
        cgm_values: List of CGM glucose values
        sampling_interval: List of sampling intervals
        
    Returns:
        float: Incremental AUC
    """
    time_points = [i * sampling_interval[i] for i in range(len(cgm_values))]
    return area_under_curve(time_points, cgm_values)


def calc_auc(cgm_values: List[float], sampling_interval: float) -> float:
    """
    Calculate total area under curve using numpy trapezoidal integration.
    
    Args:
        cgm_values: List of CGM glucose values
        sampling_interval: Sampling interval in minutes
        
    Returns:
        float: Total AUC
    """
    return np.trapz(cgm_values, dx=sampling_interval)


def load_cgm_data(data_directory: str = ".") -> pd.DataFrame:
    """
    Load and process CGM data from subject directories.
    
    Args:
        data_directory: Directory containing CGM data subdirectories
        
    Returns:
        pd.DataFrame: Processed CGM data with meal information
    """
    logger.info("Loading CGM data...")
    
    # Check if data_directory exists
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory not found: {data_directory}")
    
    # Check if CGMacros subdirectory exists
    cgmacros_dir = os.path.join(data_directory, "CGMacros")
    if os.path.exists(cgmacros_dir):
        logger.info(f"Found CGMacros subdirectory, using: {cgmacros_dir}")
        search_directory = cgmacros_dir
    else:
        logger.info(f"Using directory directly: {data_directory}")
        search_directory = data_directory
    
    data_all_sub = pd.DataFrame(columns=["sub", "Libre GL", "Carb", "Protein", "Fat", "Fiber"])
    
    hours = 2
    libre_samples = hours * 4 + 1
    
    for sub in sorted(os.listdir(search_directory)):
        if not sub.startswith("CGMacros"):
            continue
            
        csv_path = os.path.join(search_directory, sub, f"{sub}.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found for subject {sub}")
            continue
            
        try:
            data = pd.read_csv(csv_path)
            data_sub = pd.DataFrame(columns=["sub", "Libre GL", "Carb", "Protein", "Fat", "Fiber"])
            
            breakfast_mask = (data["Meal Type"] == "Breakfast") | (data["Meal Type"] == "breakfast")
            
            for index in data[breakfast_mask].index:
                data_meal = {}
                data_meal["sub"] = sub[-3:]
                
                # Extract glucose values with 15-minute intervals
                glucose_slice = data["Libre GL"][index:index+135:15]
                data_meal["Libre GL"] = glucose_slice.tolist()
                
                if len(data_meal["Libre GL"]) < 9:
                    continue
                    
                # Calculate AUC metrics
                data_meal["iAUC"] = calc_iauc(data_meal["Libre GL"], [15 for _ in range(libre_samples)])
                data_meal["AUC"] = calc_auc(data_meal["Libre GL"], 15)
                
                # Extract macronutrient information
                data_meal["Carb"] = data["Carbs"][index] * 4
                data_meal["Protein"] = data["Protein"][index] * 4
                data_meal["Fat"] = data["Fat"][index] * 9
                data_meal["Fiber"] = data["Fiber"][index] * 2
                data_meal["Calories"] = data["Calories"][index]
                
                data_sub = pd.concat([data_sub, pd.DataFrame([data_meal])], ignore_index=True)
            
            # Remove standard meal if present
            if (len(data_sub) > 0 and 
                data_sub["Carb"].iloc[0] == 24 and 
                data_sub["Protein"].iloc[0] == 22 and 
                data_sub["Fat"].iloc[0] == 10.5 and 
                data_sub["Fiber"].iloc[0] == 0.0):
                data_sub = data_sub.iloc[1:]
                
            data_all_sub = pd.concat([data_all_sub, data_sub], ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error processing subject {sub}: {e}")
            continue
    
    # Filter positive iAUC values
    data_all_sub = data_all_sub[data_all_sub["iAUC"] > 0]
    data_all_sub.reset_index(drop=True, inplace=True)
    
    logger.info(f"Loaded data for {len(data_all_sub)} meals")
    return data_all_sub


def load_biomarker_data(bio_csv_path: str = "bio.csv") -> Dict[str, np.ndarray]:
    """
    Load and process biomarker data from CSV file.
    
    Args:
        bio_csv_path: Path to biomarker CSV file
        
    Returns:
        Dict containing processed biomarker arrays
    """
    logger.info("Loading biomarker data...")
    
    # Check for bio.csv in CGMacros directory if not found at specified path
    if not os.path.exists(bio_csv_path):
        cgm_bio_path = os.path.join("CGMacros", "bio.csv")
        if os.path.exists(cgm_bio_path):
            bio_csv_path = cgm_bio_path
            logger.info(f"Using biomarker file from CGMacros directory: {bio_csv_path}")
        else:
            raise FileNotFoundError(f"Biomarker file not found: {bio_csv_path}")
    else:
        logger.info(f"Using biomarker file: {bio_csv_path}")
    
    df = pd.read_csv(bio_csv_path)
    
    # Extract biomarker values
    a1c = df["A1c PDL (Lab)"].dropna().to_numpy()
    fasting_glucose = df["Fasting GLU - PDL (Lab)"].dropna().to_numpy()
    
    # Process fasting insulin (handle special formatting)
    fasting_insulin_raw = df["Insulin "].dropna().to_numpy()
    fasting_insulin = [float(str(x).strip(' (low)')) for x in fasting_insulin_raw]
    fasting_insulin = np.array(fasting_insulin)
    
    # Calculate HOMA-IR
    homa = (fasting_insulin * fasting_glucose) / 405
    
    # Extract lipid panel
    biomarkers = {
        'a1c': a1c,
        'fasting_glucose': fasting_glucose,
        'fasting_insulin': fasting_insulin,
        'homa': homa,
        'triglycerides': df["Triglycerides"].dropna().to_numpy(),
        'cholesterol': df["Cholesterol"].dropna().to_numpy(),
        'hdl': df["HDL"].dropna().to_numpy(),
        'non_hdl': df["Non HDL "].dropna().to_numpy(),
        'ldl': df["LDL (Cal)"].dropna().to_numpy(),
        'vldl': df["VLDL (Cal)"].dropna().to_numpy(),
        'cho_hdl_ratio': df["Cho/HDL Ratio"].dropna().to_numpy(),
    }
    
    # Process anthropometric data
    weights_lbs = df["Body weight "].dropna().to_numpy()
    heights_inches = df["Height "].dropna().to_numpy()
    
    # Convert to metric units
    weight_kg = weights_lbs * 0.453592
    height_m = heights_inches * 0.0254
    bmi = weight_kg / (height_m ** 2)
    
    biomarkers.update({
        'bmi': bmi,
        'age': df["Age"].to_numpy(),
        'gender': np.array([1 if x == 'M' else -1 for x in df["Gender"].tolist()])
    })
    
    logger.info(f"Loaded biomarker data for {len(a1c)} subjects")
    return biomarkers


def categorize_patients(a1c_values: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Categorize patients based on A1C levels.
    
    Args:
        a1c_values: Array of A1C values
        
    Returns:
        Tuple of patient categories and indices for each category
    """
    patients = []
    for a1c in a1c_values:
        if a1c < 5.7:
            patients.append("H")  # Healthy
        elif 5.7 <= a1c <= 6.4:
            patients.append("P")  # Pre-diabetic
        else:
            patients.append("T2D")  # Type 2 Diabetes
    
    patients = np.array(patients)
    
    indices = {
        'healthy': np.where(patients == "H")[0],
        'prediabetic': np.where(patients == "P")[0],
        'diabetic': np.where(patients == "T2D")[0]
    }
    
    return patients, indices


def merge_cgm_biomarker_data(cgm_data: pd.DataFrame, biomarkers: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Merge CGM data with biomarker data by subject.
    
    Args:
        cgm_data: DataFrame with CGM meal data
        biomarkers: Dictionary with biomarker arrays
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    logger.info("Merging CGM and biomarker data...")
    
    # Add baseline glucose
    baseline_glucose = []
    for i in range(len(cgm_data)):
        baseline_glucose.append(cgm_data["Libre GL"].iloc[i][0])
    cgm_data["Baseline_Libre"] = baseline_glucose
    
    # Get unique subjects
    subjects = cgm_data["sub"].unique()
    
    # Expand biomarker data to match meal records
    biomarker_columns = {}
    for key, values in biomarkers.items():
        expanded_values = []
        for i, subject in enumerate(subjects):
            if i < len(values):
                match_length = len(cgm_data[cgm_data["sub"] == subject])
                expanded_values.extend([values[i]] * match_length)
            else:
                logger.warning(f"Missing {key} data for subject {subject}")
        biomarker_columns[key] = expanded_values
    
    # Add biomarker columns to CGM data
    column_mapping = {
        'age': 'Age',
        'gender': 'Gender', 
        'bmi': 'BMI',
        'a1c': 'A1c',
        'homa': 'HOMA',
        'fasting_insulin': 'Insulin',
        'triglycerides': 'TG',
        'cholesterol': 'Cholesterol',
        'hdl': 'HDL',
        'non_hdl': 'Non HDL',
        'ldl': 'LDL',
        'vldl': 'VLDL',
        'cho_hdl_ratio': 'CHO/HDL ratio',
        'fasting_glucose': 'Fasting BG'
    }
    
    for key, column_name in column_mapping.items():
        if key in biomarker_columns:
            cgm_data[column_name] = biomarker_columns[key]
    
    # Add derived macronutrient columns
    cgm_data["Carbs"] = cgm_data["Carb"] * 4
    cgm_data["Fats"] = cgm_data["Fat"] * 9
    cgm_data["Net Carb"] = cgm_data["Carb"] - cgm_data["Fiber"]
    
    return cgm_data


def perform_cross_validation_analysis(data: pd.DataFrame, target: str, output_prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform leave-one-subject-out cross-validation analysis.
    
    Args:
        data: DataFrame with features and target
        target: Target variable name ('iAUC' or 'AUC')
        output_prefix: Prefix for output files
        
    Returns:
        Tuple of (predictions, ground_truth, colors)
    """
    logger.info(f"Performing cross-validation analysis for {target}...")
    
    feature_columns = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender', 
                      'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL', 
                      'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    
    x_data = data[['sub'] + feature_columns]
    subjects = data["sub"].unique()
    
    gt_list = []
    pred_list = []
    colors = []
    
    for i, sub in enumerate(subjects):
        # Split data by subject
        x_train = x_data[x_data["sub"] != sub].iloc[:, 1:]  # Exclude 'sub' column
        y_train = data[data["sub"] != sub][target]
        x_test = x_data[x_data["sub"] == sub].iloc[:, 1:]
        y_test = data[data["sub"] == sub][target]
        
        if len(x_test) == 0:
            continue
        
        # Scale features
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        # Assign colors based on A1C levels
        subject_a1c = data[data["sub"] == sub]["A1c"].iloc[0]
        for _ in range(len(x_test)):
            if subject_a1c < 5.7:
                colors.append('blue')
            elif 5.7 <= subject_a1c <= 6.4:
                colors.append('green')
            else:
                colors.append('red')
        
        # Train XGBoost model
        mdl = xgb.XGBRegressor(max_depth=1, n_estimators=80, learning_rate=0.2, reg_alpha=1, reg_lambda=0)
        mdl.fit(x_train_scaled, y_train)
        y_pred = mdl.predict(x_test_scaled)
        
        pred_list.append(y_pred)
        gt_list.append(y_test.values)
    
    # Combine results
    pred_array = np.concatenate(pred_list, axis=0)
    gt_array = np.concatenate(gt_list, axis=0)
    colors = np.array(colors)
    
    # Calculate correlation
    correlation, p_value = pearsonr(pred_array, gt_array)
    logger.info(f"{target} correlation: {correlation:.3f} (p={p_value:.2e})")
    
    # Save results
    status = ['healthy' if c == 'blue' else 'preD' if c == 'green' else 'T2D' for c in colors]
    results_df = pd.DataFrame({
        'ground truth': gt_array,
        'prediction': pred_array,
        'health status': status
    })
    results_df.to_csv(f"{output_prefix}.csv", index=False)
    
    return pred_array, gt_array, colors


def create_correlation_plot(gt_array: np.ndarray, pred_array: np.ndarray, colors: np.ndarray, 
                          target: str, output_prefix: str, save_plot: bool = True):
    """
    Create and save correlation scatter plot.
    
    Args:
        gt_array: Ground truth values
        pred_array: Predicted values  
        colors: Color array for health status
        target: Target variable name
        output_prefix: Prefix for output files
        save_plot: Whether to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Plot by health status
    for color, label in [('blue', 'healthy'), ('green', 'preD'), ('red', 'T2D')]:
        idx = np.where(colors == color)[0]
        if len(idx) > 0:
            plt.scatter(gt_array[idx], pred_array[idx], c=color, s=10, label=label)
    
    correlation = pearsonr(pred_array, gt_array)[0]
    
    plt.ylabel(f"Predicted {target} (mg/dl.h)", fontsize=13)
    plt.xlabel(f"Ground truth {target} (mg/dl.h)", fontsize=13)
    plt.title(f"Correlation: {correlation:.2f}")
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f"corr_{output_prefix}.svg", format='svg')
        plt.savefig(f"corr_{output_prefix}.png", dpi=300)
    
    plt.show()


def perform_shap_analysis(data: pd.DataFrame, target: str, output_prefix: str, save_plot: bool = True):
    """
    Perform SHAP analysis for feature importance.
    
    Args:
        data: DataFrame with features and target
        target: Target variable name
        output_prefix: Prefix for output files
        save_plot: Whether to save the plot
    """
    logger.info(f"Performing SHAP analysis for {target}...")
    
    feature_columns = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Baseline_Libre', 'Age', 'Gender',
                      'BMI', 'A1c', 'HOMA', 'Insulin', 'TG', 'Cholesterol', 'HDL', 'Non HDL',
                      'LDL', 'VLDL', 'CHO/HDL ratio', 'Fasting BG']
    
    # Prepare features with z-score normalization
    x = data[feature_columns].copy()
    for col in x.columns:
        x[col] = zscore(x[col])
    
    # Rename columns for better display
    x = x.rename(columns={
        'Baseline_Libre': 'Baseline gl.',
        'HOMA': 'HOMA-IR'
    })
    
    y_true = data[target]
    
    # Train model
    mdl = xgb.XGBRegressor()
    mdl.fit(x, y_true)
    
    # SHAP analysis
    explainer = shap.TreeExplainer(mdl, x)
    shap_values = explainer.shap_values(x, check_additivity=False)
    
    # Create SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, x, show=False)
    plt.title(target, fontsize=15)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(f"{output_prefix}.svg", format='svg')
        plt.savefig(f"{output_prefix}.png", dpi=300)
    
    plt.show()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='CGM Data Analysis Pipeline')
    parser.add_argument('--data-dir', default='.', help='Directory containing CGM data')
    parser.add_argument('--bio-file', default='bio.csv', help='Path to biomarker CSV file')
    parser.add_argument('--output-dir', default='.', help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    args = parser.parse_args()
    
    try:
        # Set pandas display options
        pd.set_option('display.max_rows', None)
        
        # Load and process data
        cgm_data = load_cgm_data(args.data_dir)
        biomarkers = load_biomarker_data(args.bio_file)
        
        # Merge datasets
        combined_data = merge_cgm_biomarker_data(cgm_data, biomarkers)
        
        # Remove the first column (index) if it exists
        if 'index' in combined_data.columns:
            combined_data = combined_data.drop('index', axis=1)
        
        logger.info(f"Final dataset shape: {combined_data.shape}")
        logger.info(f"Columns: {list(combined_data.columns)}")
        
        # Perform analysis for iAUC
        logger.info("=== iAUC Analysis ===")
        pred_iauc, gt_iauc, colors_iauc = perform_cross_validation_analysis(
            combined_data, 'iAUC', 'iAUC'
        )
        
        if not args.no_plots:
            create_correlation_plot(gt_iauc, pred_iauc, colors_iauc, 'iAUC', 'iAUC')
            perform_shap_analysis(combined_data, 'iAUC', 'iAUC')
        
        # Perform analysis for AUC  
        logger.info("=== AUC Analysis ===")
        pred_auc, gt_auc, colors_auc = perform_cross_validation_analysis(
            combined_data, 'AUC', 'AUC'
        )
        
        if not args.no_plots:
            create_correlation_plot(gt_auc, pred_auc, colors_auc, 'AUC', 'AUC')
            perform_shap_analysis(combined_data, 'AUC', 'AUC')
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()