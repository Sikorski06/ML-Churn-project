import pandas as pd
import xgboost as xgb
import os
from src.utils import get_logger

# importuje logikę:
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

logger = get_logger("PREDICT_MODEL")

def make_batch_predictions(input_file: str, output_file: str):
    """
    Wczytuje surowe dane CSV, przepuszcza przez ten sam pipeline co trening,
    i generuje predykcje.
    """
    # 1. Ładowanie danych
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Brak pliku: {input_file}")
    
    df_raw = pd.read_csv(input_file)
    logger.info(f"Wczytano {len(df_raw)} wierszy.")

    # Zachowaj ID do wyników, jeśli istnieją
    customer_ids = df_raw['customerID'] if 'customerID' in df_raw.columns else df_raw.index

    # 2. REUŻYCIE PIPELINE'U (Preprocessing)
    # To samo czyszczenie co przy treningu!
    df_clean = preprocess_data(df_raw)
    
    # 3. REUŻYCIE PIPELINE'U (Feature Engineering)
    # train_mode=False -> ładuje zapisany encoder z folderu models/
    try:
        df_features = build_features(df_clean, train_mode=False)
    except Exception as e:
        logger.error(f"Błąd podczas budowania cech. Czy uruchomiłeś trening? {e}")
        return

    # 4. Ładowanie modelu
    model_path = "models/xgb_model.json"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Brak modelu! Uruchom 'run_pipeline.py' najpierw.")
    
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Upewnij się, że nie ma kolumny target w danych do predykcji (xgboost by zgłupiał)
    if 'churn' in df_features.columns:
        df_features = df_features.drop(columns=['churn'])

    # 5. Predykcja
    preds = model.predict(df_features)
    probs = model.predict_proba(df_features)[:, 1]
    
    # 6. Zapis
    results = pd.DataFrame({
        'customerID': customer_ids,
        'prediction': preds,
        'probability': probs
    })
    
    results.to_csv(output_file, index=False)
    logger.info(f"✅ Wyniki zapisane w {output_file}")

if __name__ == "__main__":
    make_batch_predictions("data/Telco-Customer-Churn.csv", "data/predictions.csv")