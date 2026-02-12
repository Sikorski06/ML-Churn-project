import pandas as pd
from src.utils import get_logger

logger = get_logger("VALIDATE_DATA")

def validate_data(df: pd.DataFrame) -> bool:
    """
    Sprawdza schemat danych i podstawowe reguły biznesowe.
    Zwraca True, jeśli dane są poprawne, w przeciwnym razie rzuca błąd.
    """
    # 1. Sprawdź wymagane kolumny (to tylko przykład kilku kluczowych)
    required_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                     'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                     'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
                     'MonthlyCharges', 'TotalCharges']
    
    # Konwertujemy na lowercase, bo w preprocess robimy to samo
    df_cols = set(df.columns)
    missing_cols = [col for col in required_cols if col not in df_cols and col != 'Churn']
    
    if missing_cols:
        logger.error(f"❌ Brakuje kolumn w danych wejściowych: {missing_cols}")
        raise ValueError(f"Błędny schemat danych. Brakuje: {missing_cols}")

    # 2. Sprawdź typy danych (np. czy tenure nie jest ujemne)
    if 'tenure' in df.columns:
        if (df['tenure'] < 0).any():
            logger.error("❌ Wykryto ujemną wartość w 'tenure'!")
            raise ValueError("Tenure nie może być ujemne.")

    logger.info("✅ Walidacja danych zakończona sukcesem.")
    return True