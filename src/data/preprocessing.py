import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Basic cleaning for Telco churn.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - simple NA handling
    """
     
    df_encoded = df.copy()
    
    df_encoded.columns = df_encoded.columns.str.strip()
    
    if "customerID" in df_encoded:
        df_encoded = df_encoded.drop(columns=["customerID"])

    if target_col in df_encoded.columns and df_encoded[target_col].dtype == "object":
        df_encoded[target_col] = df_encoded[target_col].str.strip().map({"No": 0, "Yes": 1})

    if "TotalCharges" in df_encoded.columns:
        df_encoded["TotalCharges"] = pd.to_numeric(df_encoded["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df_encoded.columns:
        df_encoded["SeniorCitizen"] = df_encoded["SeniorCitizen"].fillna(0).astype(int)

    # simple NA strategy:
    # - numeric: fill with 0
    # - others: leave for encoders to handle (get_dummies ignores NaN safely)
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df_encoded