#!/usr/bin/env python3
"""
Script to create a comprehensive combined dataset from all CGMacros data sources.
Combines individual CGMacros CSV files with bio, gut health test, and microbes data.
"""

import pandas as pd
import os
import glob
from pathlib import Path
import re
import numpy as np


def load_combined_cgmacros_data(base_path=None):
    """
    Load the combined CGMacros data (either from existing file or create it).

    Args:
        base_path (str): Path to the directory containing CGMacros folder

    Returns:
        pandas.DataFrame: Combined CGMacros DataFrame
    """

    if base_path is None:
        base_path = Path(__file__).parent
    else:
        base_path = Path(base_path)

    # Check if combined file already exists
    combined_file = base_path / "combined_cgmacros_data.csv"
    if combined_file.exists():
        print(f"Loading existing combined CGMacros data from: {combined_file}")
        return pd.read_csv(combined_file)

    # If not, create it
    print("Combined CGMacros file not found. Creating it...")
    return combine_cgmacros_data(base_path)


def combine_cgmacros_data(base_path):
    """
    Combine all CGMacros CSV files into a single DataFrame.

    Args:
        base_path (str): Path to the directory containing CGMacros folder

    Returns:
        pandas.DataFrame: Combined DataFrame with all CGMacros data
    """

    cgmacros_path = base_path / "CGMacros"

    if not cgmacros_path.exists():
        raise FileNotFoundError(f"CGMacros directory not found at: {cgmacros_path}")

    print(f"Looking for CGMacros folders in: {cgmacros_path}")

    # Find all CGMacros-XXX directories
    cgmacros_pattern = cgmacros_path / "CGMacros-*"
    cgmacros_dirs = [d for d in glob.glob(str(cgmacros_pattern)) if os.path.isdir(d)]
    cgmacros_dirs.sort()  # Sort to ensure consistent ordering

    print(f"Found {len(cgmacros_dirs)} CGMacros directories")

    combined_data = []

    for dir_path in cgmacros_dirs:
        dir_name = os.path.basename(dir_path)

        # Extract the number from CGMacros-XXX
        match = re.search(r"CGMacros-(\d+)", dir_name)
        if not match:
            print(f"Warning: Skipping directory with unexpected name: {dir_name}")
            continue

        folder_number = int(match.group(1))  # Convert to int for proper joining

        # Look for CSV file in the directory
        csv_files = glob.glob(os.path.join(dir_path, "*.csv"))

        if not csv_files:
            print(f"Warning: No CSV file found in {dir_name}")
            continue

        csv_file = csv_files[0]

        try:
            print(f"Processing {dir_name}...")
            df = pd.read_csv(csv_file)

            # Add the source identifier columns
            df["CGMacros_ID"] = folder_number
            df["Source_Folder"] = dir_name

            combined_data.append(df)
            print(f"  - Loaded {len(df)} rows from {os.path.basename(csv_file)}")

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

    if not combined_data:
        raise ValueError("No data was successfully loaded from any CGMacros files")

    # Combine all DataFrames
    print("\nCombining all CGMacros data...")
    combined_df = pd.concat(combined_data, ignore_index=True)

    return combined_df


def load_additional_data(base_path):
    """
    Load bio, gut health test, and microbes data.

    Args:
        base_path (str): Path to the directory containing data files

    Returns:
        tuple: (bio_df, gut_health_df, microbes_df)
    """

    cgmacros_path = base_path / "CGMacros"

    # Load bio data
    bio_file = cgmacros_path / "bio.csv"
    if not bio_file.exists():
        raise FileNotFoundError(f"Bio file not found at: {bio_file}")
    bio_df = pd.read_csv(bio_file)
    print(f"Loaded bio data: {len(bio_df)} rows, {len(bio_df.columns)} columns")

    # Load gut health test data
    gut_health_file = cgmacros_path / "gut_health_test.csv"
    if not gut_health_file.exists():
        raise FileNotFoundError(f"Gut health test file not found at: {gut_health_file}")
    gut_health_df = pd.read_csv(gut_health_file)
    print(
        f"Loaded gut health test data: {len(gut_health_df)} rows, {len(gut_health_df.columns)} columns"
    )

    # Load microbes data
    microbes_file = cgmacros_path / "microbes.csv"
    if not microbes_file.exists():
        raise FileNotFoundError(f"Microbes file not found at: {microbes_file}")
    microbes_df = pd.read_csv(microbes_file)
    print(
        f"Loaded microbes data: {len(microbes_df)} rows, {len(microbes_df.columns)} columns"
    )

    return bio_df, gut_health_df, microbes_df


def create_comprehensive_dataset(base_path=None):
    """
    Create a comprehensive dataset by joining all data sources.

    Args:
        base_path (str): Path to the directory containing data files

    Returns:
        pandas.DataFrame: Comprehensive combined dataset
    """

    if base_path is None:
        base_path = Path(__file__).parent
    else:
        base_path = Path(base_path)

    print("=== Creating Comprehensive CGMacros Dataset ===\n")

    # Load combined CGMacros data
    print("1. Loading CGMacros time-series data...")
    cgmacros_df = load_combined_cgmacros_data(base_path)
    print(
        f"   CGMacros data: {len(cgmacros_df)} rows, {len(cgmacros_df.columns)} columns"
    )
    print(f"   Unique subjects: {sorted(cgmacros_df['CGMacros_ID'].unique())}")

    # Load additional data
    print("\n2. Loading additional datasets...")
    bio_df, gut_health_df, microbes_df = load_additional_data(base_path)

    # Prepare data for joining
    print("\n3. Preparing data for joining...")

    # Rename columns to avoid conflicts and add prefixes for clarity
    bio_df_renamed = bio_df.copy()
    bio_df_renamed.columns = ["subject"] + [f"bio_{col}" for col in bio_df.columns[1:]]

    gut_health_df_renamed = gut_health_df.copy()
    gut_health_df_renamed.columns = ["subject"] + [
        f"gut_health_{col}" for col in gut_health_df.columns[1:]
    ]

    microbes_df_renamed = microbes_df.copy()
    microbes_df_renamed.columns = ["subject"] + [
        f"microbes_{col}" for col in microbes_df.columns[1:]
    ]

    # Join all datasets
    print("\n4. Joining all datasets...")

    # Start with CGMacros data
    comprehensive_df = cgmacros_df.copy()

    # Join with bio data
    comprehensive_df = comprehensive_df.merge(
        bio_df_renamed, left_on="CGMacros_ID", right_on="subject", how="left"
    )
    print(f"   After joining with bio data: {len(comprehensive_df)} rows")

    # Join with gut health data
    comprehensive_df = comprehensive_df.merge(
        gut_health_df_renamed, left_on="CGMacros_ID", right_on="subject", how="left"
    )
    print(f"   After joining with gut health data: {len(comprehensive_df)} rows")

    # Join with microbes data
    comprehensive_df = comprehensive_df.merge(
        microbes_df_renamed, left_on="CGMacros_ID", right_on="subject", how="left"
    )
    print(f"   After joining with microbes data: {len(comprehensive_df)} rows")

    # Clean up duplicate subject columns
    subject_cols = [
        col for col in comprehensive_df.columns if col.startswith("subject")
    ]
    if len(subject_cols) > 1:
        # Keep only the first subject column and drop duplicates
        for col in subject_cols[1:]:
            comprehensive_df = comprehensive_df.drop(columns=[col])

    # Reorder columns for better organization
    print("\n5. Organizing columns...")

    # Define column groups
    id_columns = ["CGMacros_ID", "Source_Folder", "subject"]
    time_columns = [
        col for col in comprehensive_df.columns if "Timestamp" in col or "Time" in col
    ]
    glucose_columns = [
        col for col in comprehensive_df.columns if "GL" in col and "Glucose" not in col
    ]
    activity_columns = [
        col
        for col in comprehensive_df.columns
        if any(x in col for x in ["HR", "Calories", "METs"])
    ]
    meal_columns = [
        col
        for col in comprehensive_df.columns
        if any(
            x in col
            for x in [
                "Meal",
                "Calories",
                "Carbs",
                "Protein",
                "Fat",
                "Fiber",
                "Amount",
                "Image",
            ]
        )
    ]
    bio_columns = [col for col in comprehensive_df.columns if col.startswith("bio_")]
    gut_health_columns = [
        col for col in comprehensive_df.columns if col.startswith("gut_health_")
    ]
    microbes_columns = [
        col for col in comprehensive_df.columns if col.startswith("microbes_")
    ]
    other_columns = [
        col
        for col in comprehensive_df.columns
        if col
        not in id_columns
        + time_columns
        + glucose_columns
        + activity_columns
        + meal_columns
        + bio_columns
        + gut_health_columns
        + microbes_columns
    ]

    # Reorder columns
    column_order = (
        id_columns
        + time_columns
        + glucose_columns
        + activity_columns
        + meal_columns
        + bio_columns
        + gut_health_columns
        + microbes_columns
        + other_columns
    )

    # Filter out columns that don't exist (in case of typos)
    column_order = [col for col in column_order if col in comprehensive_df.columns]
    comprehensive_df = comprehensive_df[column_order]

    print(f"\n=== Final Dataset Summary ===")
    print(f"Total rows: {len(comprehensive_df):,}")
    print(f"Total columns: {len(comprehensive_df.columns)}")
    print(f"Unique subjects: {len(comprehensive_df['CGMacros_ID'].unique())}")
    print(f"Subject IDs: {sorted(comprehensive_df['CGMacros_ID'].unique())}")

    # Check for missing joins
    missing_bio = comprehensive_df["bio_Age"].isna().sum()
    missing_gut = (
        comprehensive_df["gut_health_Gut Lining Health"].isna().sum()
        if "gut_health_Gut Lining Health" in comprehensive_df.columns
        else 0
    )
    missing_microbes = (
        comprehensive_df[
            [col for col in comprehensive_df.columns if col.startswith("microbes_")]
        ]
        .isna()
        .all(axis=1)
        .sum()
    )

    print(f"\nData completeness:")
    print(
        f"  Missing bio data: {missing_bio} rows ({missing_bio/len(comprehensive_df)*100:.1f}%)"
    )
    print(
        f"  Missing gut health data: {missing_gut} rows ({missing_gut/len(comprehensive_df)*100:.1f}%)"
    )
    print(
        f"  Missing microbes data: {missing_microbes} rows ({missing_microbes/len(comprehensive_df)*100:.1f}%)"
    )

    print(f"\nColumn breakdown:")
    print(f"  ID columns: {len(id_columns)}")
    print(f"  Time columns: {len(time_columns)}")
    print(f"  Glucose columns: {len(glucose_columns)}")
    print(f"  Activity columns: {len(activity_columns)}")
    print(f"  Meal columns: {len(meal_columns)}")
    print(f"  Bio columns: {len(bio_columns)}")
    print(f"  Gut health columns: {len(gut_health_columns)}")
    print(f"  Microbes columns: {len(microbes_columns)}")
    print(f"  Other columns: {len(other_columns)}")

    return comprehensive_df


def save_comprehensive_dataset(df, output_path=None):
    """
    Save the comprehensive dataset to a CSV file.

    Args:
        df (pandas.DataFrame): The comprehensive DataFrame
        output_path (str): Path where to save the file. If None, saves in script directory.
    """
    if output_path is None:
        script_dir = Path(__file__).parent
        output_path = script_dir / "comprehensive_cgmacros_dataset.csv"

    print(f"\nSaving comprehensive dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved successfully!")

    # Print file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")


def analyze_dataset(df):
    """
    Perform basic analysis of the comprehensive dataset.

    Args:
        df (pandas.DataFrame): The comprehensive DataFrame
    """

    print(f"\n=== Dataset Analysis ===")

    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Time range analysis
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        print(f"\nTime Range Analysis:")
        print(f"Overall date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        print(f"Total days: {(df['Timestamp'].max() - df['Timestamp'].min()).days}")

        # Per subject analysis
        subject_summary = (
            df.groupby("CGMacros_ID")["Timestamp"]
            .agg(["min", "max", "count"])
            .reset_index()
        )
        subject_summary["days"] = (
            subject_summary["max"] - subject_summary["min"]
        ).dt.days

        print(f"\nPer Subject Summary:")
        print(f"Average days per subject: {subject_summary['days'].mean():.1f}")
        print(f"Average records per subject: {subject_summary['count'].mean():.0f}")
        print(
            f"Records per day per subject: {(subject_summary['count'] / (subject_summary['days'] + 1)).mean():.1f}"
        )

    # Glucose data analysis
    glucose_cols = [
        col for col in df.columns if "GL" in col and col not in ["CGMacros_ID"]
    ]
    if glucose_cols:
        print(f"\nGlucose Data Analysis:")
        for col in glucose_cols:
            non_null = df[col].notna().sum()
            print(
                f"  {col}: {non_null:,} records ({non_null/len(df)*100:.1f}% coverage)"
            )
            if non_null > 0:
                print(f"    Range: {df[col].min():.1f} - {df[col].max():.1f}")
                print(f"    Mean: {df[col].mean():.1f}")

    # Bio data analysis
    bio_cols = [col for col in df.columns if col.startswith("bio_")]
    if bio_cols:
        print(f"\nBio Data Summary:")
        print(f"  Available for {df['bio_Age'].notna().sum()} subjects")
        if "bio_Age" in df.columns:
            ages = df.groupby("CGMacros_ID")["bio_Age"].first().dropna()
            print(
                f"  Age range: {ages.min():.0f} - {ages.max():.0f} years (mean: {ages.mean():.1f})"
            )
        if "bio_Gender" in df.columns:
            gender_dist = df.groupby("CGMacros_ID")["bio_Gender"].first().value_counts()
            print(f"  Gender distribution: {dict(gender_dist)}")


def main():
    """Main function to create and analyze the comprehensive dataset."""

    try:
        # Create comprehensive dataset
        comprehensive_df = create_comprehensive_dataset()

        # Save to CSV
        save_comprehensive_dataset(comprehensive_df)

        # Perform analysis
        analyze_dataset(comprehensive_df)

        # Display sample data
        print(f"\n=== Sample Data (first 3 rows) ===")
        print(comprehensive_df.head(3).to_string())

        return comprehensive_df

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    comprehensive_data = main()
