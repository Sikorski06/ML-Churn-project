import optuna
import xgboost as xgb
import pandas as pd
import json
import os
import functools
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from src.utils import get_logger
# Importujemy nasze moduły
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

logger = get_logger("TUNING")

def objective(trial, X_train, y_train, X_test, y_test):
    # Parametry do optymalizacji
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        
        # scale_pos_weight balansuje wagi. 
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
        
        # Stałe
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': -1
    }
    
    # Trening
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Ewaluacja (Optymalizujemy pod Recall!)
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    
    return recall

def run_tuning():
    logger.info("Rozpoczynam tuning...")
    
    data_path = 'data/raw/Telco-Customer-Churn.csv'
    df = load_data(data_path)
    df = preprocess_data(df)
    df = build_features(df, train_mode=True)
    
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    # Obliczmy wstępny balans (liczba negatywnych / liczba pozytywnych)
    ratio = float(y.value_counts()[0]) / y.value_counts()[1]
    logger.info(f"Balans klas (Ratio): {ratio:.2f}.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Dzięki temu Optuna widzi tylko argument 'trial', a dane są "zamrożone"
    objective_with_data = functools.partial(objective, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_with_data, n_trials=30) 
    
    logger.info(f"Najlepsze parametry: {study.best_params}")
    logger.info(f"Najlepszy Recall: {study.best_value}")
    
    os.makedirs("models", exist_ok=True)
    with open("models/best_params.json", "w") as f:
        json.dump(study.best_params, f)
        
    logger.info("Zapisano najlepsze parametry.")

if __name__ == "__main__":
    run_tuning()