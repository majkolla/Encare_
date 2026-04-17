import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load the ERAS dataset and strip the trailing ghost column."""
    # Load the file
    df = pd.read_csv(filepath, sep=',', low_memory=False)
    
    return df

def preprocess_for_synthesis(df: pd.DataFrame) -> pd.DataFrame:
    
    "Placeholder for your data processing logic. This is where you can do feature selection, cleaning, and imputation before passing the data to the synthesis model."
    df_subset = df

    # Ideas
    # --- 1. Feature selection, choose specific columns for the model to target ---
    # --- 2. Data processing and cleaning (Replace with your own logic) ---
    # --- 3. Handling null-values ---

    return df_subset