import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from src.utils import get_logger

logger = get_logger("TRAIN_MODEL")

def train_model(df: pd.DataFrame, target_col: str = 'churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Stratify jest ważne przy niezbalansowanych danych
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    mlflow.set_experiment("Telco_Churn_Pro_Edition")
    
    with mlflow.start_run() as run:
        # Ładowanie parametrów
        params_path = "models/best_params.json"
        default_params = {
            "n_estimators": 100, 
            "max_depth": 6, 
            "learning_rate": 0.1,
            "scale_pos_weight": 3.0,  # Domyślna waga
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        }

        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                loaded_params = json.load(f)
            params = {**default_params, **loaded_params}
            logger.info("Załadowano parametry z Optuny.")
        else:
            params = default_params
            logger.warning("Używam parametrów domyślnych.")

        mlflow.log_params(params)
        
        # Trening
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # THRESHOLD TUNING (Program sam decyduje)
        # Pobieramy prawdopodobieństwo (od 0.0 do 1.0) zamiast sztywnej decyzji (0 lub 1)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        f2_scores = []
        recalls = []
        precisions = []
        
        for t in thresholds:
            y_pred_temp = (y_probs >= t).astype(int)
            
            # Obliczamy składowe
            rec = recall_score(y_test, y_pred_temp, zero_division=0)
            prec = precision_score(y_test, y_pred_temp, zero_division=0)
            recalls.append(rec)
            precisions.append(prec)
            
            # F2-Score (kłada nacisk na Recall)
            if (4 * prec + rec) == 0:
                f2 = 0
            else:
                f2 = (5 * prec * rec) / (4 * prec + rec)
            f2_scores.append(f2)
            
        # Wybieramy próg, który daje najlepszy F2 (czyli promuje Recall)
        best_idx = np.argmax(f2_scores)
        best_threshold = float(thresholds[best_idx])
        best_f2 = f2_scores[best_idx]
        
        logger.info(f"Znaleziono optymalny Threshold (F2): {best_threshold:.2f} (Recall w tym punkcie: {recalls[best_idx]:.4f})")
        
        # Finalne predykcje z nowym progiem
        y_final_pred = (y_probs >= best_threshold).astype(int)
        
        # Metryki
        metrics = {
            "accuracy": accuracy_score(y_test, y_final_pred),
            "recall": recall_score(y_test, y_final_pred),
            "precision": precision_score(y_test, y_final_pred),
            "f1": f1_score(y_test, y_final_pred),
            "threshold": best_threshold
        }
        mlflow.log_metrics(metrics)
        logger.info(f"Wyniki po optymalizacji progu: {metrics}")
        
        # Wykres Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_final_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (Thresh: {best_threshold:.2f})")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Zapis modelu i PROGU
        mlflow.xgboost.log_model(xgb_model=model, name="model")
        model.save_model("models/xgb_model.json")
        
        # Zapisujemy próg do pliku, żeby API wiedziało, jak decydować
        with open("models/threshold.json", "w") as f:
            json.dump({"threshold": float(best_threshold)}, f)
            
        logger.info("Model zapisany. Próg zapisany w models/threshold.json")
        
        return model