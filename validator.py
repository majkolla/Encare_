import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict

def run_evaluation_report(original_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    """
    Executes a comprehensive comparison between the source dataset and the generated one.
    """
    print("\n" + "="*60)
    print("EVALUATION REPORT: SYNTHETIC DATA FIDELITY")
    print("="*60)

    # 1. Distribution Similarity (Numeric)
    # Compares the CDFs of two samples using Kolmogorov-Smirnov test.
    similarities = _compare_distributions(original_df, synthetic_df)
    
    print("\nStatistical Similarity Scores (1.0 = Identical):")
    # Show a mix of columns, filtering out NaNs (empty columns in original)
    valid_scores = {k: v for k, v in similarities.items() if not np.isnan(v)}
    for col, score in list(valid_scores.items())[:10]:
        print(f"  {col[:30]:30s}: {score:.3f}")

    # 2. Clinical Logic: Range and Relationship Checks
    # Ensures the AI generated biologically plausible 'patients'
    validations = _validate_clinical_logic(synthetic_df)
    
    print("\nClinical Sanity Check Results:")
    for check, passed in validations.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {check:30s}: {status}")

def _compare_distributions(orig: pd.DataFrame, synth: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates 1 minus the KS-statistic for each numeric column.
    A score of 1.0 indicates identical cumulative distributions.
    """
    results = {}
    common_cols = list(set(orig.columns) & set(synth.columns))
    
    for col in common_cols:
        # Clean data for comparison: Convert to numeric, ignore strings like "Unknown"
        o_clean = pd.to_numeric(orig[col], errors='coerce').dropna()
        s_clean = pd.to_numeric(synth[col], errors='coerce').dropna()
        
        # If the original column was empty, we can't calculate similarity
        if len(o_clean) == 0:
            results[col] = np.nan
            continue
            
        # Perform two-sample Kolmogorov-Smirnov test
        stat, _ = stats.ks_2samp(o_clean, s_clean)
        results[col] = round(1 - stat, 3)
        
    return results

def _validate_clinical_logic(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Verifies that generated values obey known clinical relationships and bounds.
    """
    results = {}
    
    # --- Check 1: Age Boundaries ---
    age_col = 'Age::40'
    if age_col in df.columns:
        # errors='coerce' turns "Unknown" into NaN so it doesn't crash the logic
        ages = pd.to_numeric(df[age_col], errors='coerce').dropna()
        if len(ages) > 0:
            # Check if ages are within a realistic human lifespan
            results['Age Within Range (18-110)'] = ((ages >= 18) & (ages <= 110)).all()

    # --- Check 2: BMI Mathematical Relationship (BMI = kg / m^2) ---
    weight_col = 'Preoperative body weight (kg)::20'
    height_col = 'Height (cm)::23'
    bmi_col = 'BMI::24'
    
    if all(c in df.columns for c in [weight_col, height_col, bmi_col]):
        # Convert all to numeric safely
        w = pd.to_numeric(df[weight_col], errors='coerce')
        h = pd.to_numeric(df[height_col], errors='coerce') / 100 # cm to meters
        bmi = pd.to_numeric(df[bmi_col], errors='coerce')
        
        # Avoid division by zero if height is 0 or NaN
        valid_mask = (h > 0) & h.notna() & w.notna() & bmi.notna()
        
        if valid_mask.any():
            expected_bmi = w[valid_mask] / (h[valid_mask]**2)
            actual_bmi = bmi[valid_mask]
            
            # Check if 90% of rows are within 2.0 BMI units of the expected calculation
            # (Using 2.0 to allow for slight rounding differences in the AI model)
            diff = abs(actual_bmi - expected_bmi)
            is_consistent = (diff < 2.0).mean() > 0.90
            results['BMI-Weight-Height Logic'] = is_consistent

    # --- Check 3: Categorical Sanity (Gender) ---
    gender_col = 'Gender::5'
    if gender_col in df.columns:
        allowed = ['Male', 'Female', 'Unknown']
        # Check if any generated values are outside the known categories
        invalid_genders = df[~df[gender_col].isin(allowed)]
        results['Valid Gender Categories'] = len(invalid_genders) == 0

    return results