import pandas as pd
import numpy as np

def run_random_sample(df, num_samples):
    """
    Generates synthetic data. 
    - For numeric columns: samples from a uniform distribution (min to max).
    - For categorical/text columns: samples from existing unique values.
    """
    synthetic_data = {}
    
    for col in df.columns:
        # 1. Check if the column is numeric (int or float)
        if pd.api.types.is_numeric_dtype(df[col]):
            low = df[col].min()
            high = df[col].max()
            
            # Handle empty numeric columns or columns with only NaNs
            if pd.isna(low) or pd.isna(high):
                synthetic_data[col] = [np.nan] * num_samples
            else:
                # Generate random numbers between the observed min and max
                synthetic_data[col] = np.random.uniform(low, high, num_samples)
        
        # 2. Handle categorical / text columns (like "ERAS patient", "Yes/No", etc.)
        else:
            # Get all non-null values from the original column
            valid_values = df[col].dropna().values
            
            if len(valid_values) == 0:
                # If the column is entirely empty, fill with NaNs
                synthetic_data[col] = [np.nan] * num_samples
            else:
                # Randomly pick from existing text values to maintain data logic
                synthetic_data[col] = np.random.choice(valid_values, size=num_samples)
            
    # Return the generated data as a new DataFrame
    return pd.DataFrame(synthetic_data)