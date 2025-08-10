"""
Script to combine all CGMacros CSV files into a single pandas DataFrame.
Each record will be tagged with the source folder number (CGMacros-XXX).
"""

import pandas as pd
import os
import glob
from pathlib import Path
import re


def combine_cgmacros_data(base_path=None):
    """
    Combine all CGMacros CSV files into a single DataFrame.

    Args:
        base_path (str): Path to the CGMacros directory. If None, uses current script location.

    Returns:
        pandas.DataFrame: Combined DataFrame with all CGMacros data
    """

    if base_path is None:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        base_path = script_dir / "CGMacros"
    else:
        base_path = Path(base_path)

    if not base_path.exists():
        raise FileNotFoundError(f"CGMacros directory not found at: {base_path}")

    print(f"Looking for CGMacros folders in: {base_path}")

    # Find all CGMacros-XXX directories
    cgmacros_pattern = base_path / "CGMacros-*"
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

        folder_number = match.group(1)

        # Look for CSV file in the directory
        csv_files = glob.glob(os.path.join(dir_path, "*.csv"))

        if not csv_files:
            print(f"Warning: No CSV file found in {dir_name}")
            continue

        if len(csv_files) > 1:
            print(
                f"Warning: Multiple CSV files found in {dir_name}, using the first one"
            )

        csv_file = csv_files[0]

        try:
            # Read the CSV file
            print(f"Processing {dir_name}...")
            df = pd.read_csv(csv_file)

            # Add the source identifier column
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
    print("\nCombining all data...")
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Reorder columns to put identifiers first
    id_columns = ["CGMacros_ID", "Source_Folder"]
    other_columns = [col for col in combined_df.columns if col not in id_columns]
    combined_df = combined_df[id_columns + other_columns]

    print(f"\nCombined dataset summary:")
    print(f"  - Total rows: {len(combined_df):,}")
    print(f"  - Total columns: {len(combined_df.columns)}")
    print(f"  - CGMacros files processed: {len(combined_data)}")
    print(f"  - CGMacros IDs: {sorted(combined_df['CGMacros_ID'].unique())}")

    return combined_df


def save_combined_data(df, output_path=None):
    """
    Save the combined DataFrame to a CSV file.

    Args:
        df (pandas.DataFrame): The combined DataFrame
        output_path (str): Path where to save the file. If None, saves in script directory.
    """
    if output_path is None:
        script_dir = Path(__file__).parent
        output_path = script_dir / "combined_cgmacros_data.csv"

    print(f"\nSaving combined data to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"Data saved successfully!")

    # Print file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")


def main():
    """Main function to run the data combination process."""
    try:
        # Combine the data
        combined_df = combine_cgmacros_data()

        # Save to CSV
        save_combined_data(combined_df)

        # Display basic info about the combined dataset
        print(f"\nDataset info:")
        print(f"Shape: {combined_df.shape}")
        print(f"\nColumn names:")
        for i, col in enumerate(combined_df.columns, 1):
            print(f"  {i:2d}. {col}")

        print(f"\nData types:")
        print(combined_df.dtypes)

        print(f"\nFirst few rows:")
        print(combined_df.head())

        return combined_df

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    combined_data = main()
