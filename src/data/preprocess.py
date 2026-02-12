import pandas as pd
from src.utils import get_logger

logger = get_logger("PREPROCESS")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    #Czyści surowe dane
    clean = df.copy()

    if "customerID" in clean.columns:
        clean = clean.drop(columns=["customerID"])
    
    clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce").fillna(0)

    clean.columns = clean.columns.str.lower().str.replace(" ", "_")

    if "churn" in clean.columns and clean["churn"].dtype == "object":
        clean["churn"] = clean["churn"].map({"Yes":1, "No":0})

    logger.info(f'Preprocessing zakończony')
    return clean