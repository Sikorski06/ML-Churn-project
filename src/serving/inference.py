import pandas as pd
import xgboost as xgb
import joblib
import json
import os
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils import get_logger

logger = get_logger("INFERENCE_SERVICE")

class ChurnModel:
    def __init__(self):
        self.model_path = "models/xgb_model.json"
        self.encoder_path = "models/encoder.joblib"
        self.threshold_path = "models/threshold.json"
        
        self.model = None
        self.encoder = None
        self.threshold = 0.5 # Domyślna wartość, nadpiszemy ją
        
        self._load_artifacts()

    def _load_artifacts(self):
        #Ładuje Model, Encoder i Threshold przy starcie.
        try:
            # 1. Model XGBoost
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            
            # 2. Encoder
            self.encoder = joblib.load(self.encoder_path)
            
            # 3. Threshold 
            if os.path.exists(self.threshold_path):
                with open(self.threshold_path, "r") as f:
                    data = json.load(f)
                    self.threshold = data.get("threshold", 0.5)
            
            logger.info(f"Artefakty załadowane. Próg decyzji: {self.threshold:.3f}")
            
        except Exception as e:
            logger.error(f"Błąd ładowania artefaktów: {e}")
            raise e

    def predict(self, data: dict) -> dict:
        """
        Główna funkcja predykcyjna.
        Przyjmuje słownik danych klienta -> Zwraca wynik Churn/No Churn.
        """
        try:
            # 1. Dict -> DataFrame
            df_raw = pd.DataFrame([data])
            
            # 2. Pipeline
            df_clean = preprocess_data(df_raw)
            df_features = build_features(df_clean, train_mode=False)
            
            # Usunięcie targetu (jeśli przeszedł w danych)
            if 'churn' in df_features.columns:
                df_features = df_features.drop(columns=['churn'])
            
            # 3. Predykcja Prawdopodobieństwa
            # Zwraca np. [0.2, 0.8] -> bierzemy drugą wartość (prawd. Churn)
            prob = self.model.predict_proba(df_features)[0][1]
            
            # 4. Decyzja w oparciu o Twój Threshold (np. 0.62)
            prediction = 1 if prob >= self.threshold else 0
            
            result = {
                "churn_prediction": int(prediction),
                "churn_probability": float(prob),
                "threshold_used": self.threshold,
                "risk_level": "Critical" if prob > 0.8 else ("High" if prob > self.threshold else "Low")
            }
            return result
            
        except Exception as e:
            logger.error(f"Błąd podczas predykcji: {e}")
            raise e