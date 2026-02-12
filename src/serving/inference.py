import pandas as pd
import xgboost as xgb
import os
# IMPORTUJEMY LOGIKĘ:
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

class ChurnModel:
    def __init__(self):
        self.model_path = "models/xgb_model.json"
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
        else:
            # W produkcji lepiej nie rzucać błędu w init, tylko w predict, ale tu dla uproszczenia:
            raise FileNotFoundError(f"Model nie znaleziony: {self.model_path}")

    def predict(self, data: dict) -> dict:
        """
        Przyjmuje pojedynczy rekord (słownik), przetwarza go i zwraca wynik.
        """
        # 1. Dict -> DataFrame
        # Tworzymy listę z jednego słownika ([data]), żeby pandas zrozumiał to jako wiersz
        df_raw = pd.DataFrame([data])
        
        # 2. Preprocessing (To samo co w treningu!)
        # Usuwa ID, naprawia TotalCharges, zmienia nazwy kolumn na małe
        df_clean = preprocess_data(df_raw)
        
        # 3. Features (To samo co w treningu!)
        # Ładuje encoder z dysku i transformuje zmienne kategoryczne
        df_features = build_features(df_clean, train_mode=False)
        
        # Usunięcie targetu (jeśli jakimś cudem przeszedł w danych wejściowych)
        if 'churn' in df_features.columns:
            df_features = df_features.drop(columns=['churn'])
            
        # 4. Predykcja
        prediction = self.model.predict(df_features)[0]
        probability = self.model.predict_proba(df_features)[0][1]
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "risk_level": "High" if probability > 0.6 else "Low"
        }