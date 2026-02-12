import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from src.utils import get_logger

logger = get_logger("TRAIN_MODEL")

def train_model(df: pd.DataFrame, target_col: str = 'churn'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Podział 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    mlflow.set_experiment("Telco_Churn_Model")
    
    with mlflow.start_run() as run:
        logger.info(f"Rozpoczynam trening MLflow Run ID: {run.info.run_id}")
        
        # Parametry
        params = {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "use_label_encoder": False
        }
        mlflow.log_params(params)
        
        # Model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predykcje
        y_pred = model.predict(X_test)
        
        # Metryki
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        logger.info(f"Wyniki: {metrics}")
        
        # Generowanie wykresu Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (Recall: {metrics['recall']:.2f})")
        plt.ylabel('Prawdziwa etykieta')
        plt.xlabel('Przewidziana etykieta')
        
        # Zapisz wykres i wyślij do MLflow
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        logger.info("Wykres Confusion Matrix wysłany do MLflow")
        
        # Zapis modelu
        mlflow.xgboost.log_model(model, "model")
        
        # Zapis lokalny dla API (jako backup)
        model.save_model("models/xgb_model.json")
        logger.info("Model zapisany lokalnie w models/xgb_model.json")

        return model