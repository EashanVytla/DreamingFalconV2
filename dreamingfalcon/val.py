import pandas as pd
import numpy as np

def compute_l1_errors(file1_path, file2_path):
    # Read the CSV files
    df1 = pd.read_csv(file1_path, header=None)
    df2 = pd.read_csv(file2_path, header=None)
    
    # Get the columns we want to compare
    # Since there are no headers, use column indices 0-5
    columns = list(range(6))
    
    # Get the minimum length between the two files
    min_length = min(len(df1), len(df2))
    
    # Trim both dataframes to the minimum length
    df1 = df1.iloc[:min_length]
    df2 = df2.iloc[:min_length]
    
    # Calculate L1 error for each column
    errors = {}
    for col in columns:
        l1_error = np.mean(np.abs(df1[col].values - df2[col].values))
        errors[col] = l1_error
    
    # Print the errors
    print("L1 Errors for each component:")
    for col, error in errors.items():
        print(f"{col}: {error:.6f}")
    
    return errors

if __name__ == "__main__":
    # Replace these with your actual file paths
    file1_path = "data/1-31-2-Synthetic/val/forces.csv"
    file2_path = "output/1-31-2-Synthetic/forces.csv"
    
    errors = compute_l1_errors(file1_path, file2_path)