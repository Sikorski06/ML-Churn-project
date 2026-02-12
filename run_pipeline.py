from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.model.train_model import train_model
from src.model.tune_model import run_tuning
from src.utils import get_logger
import sys

# Inicjalizacja loggera
logger = get_logger("PIPELINE")

def main():
    try:
        logger.info("START: Uruchamiam Pipeline ML")
        
        # Wczytywanie Danych
        data_path = 'data/raw/Telco-Customer-Churn.csv'
        logger.info(f"Wczytywanie danych z {data_path}")
        df = load_data(data_path)
        
        # Czyszczenie (Preprocessing)
        logger.info("Preprocessing...")
        df_clean = preprocess_data(df)
        
        # Inżynieria Cech (Feature Engineeiring)
        logger.info("Budowanie cech i zapisywanie Encodera...")
        df_features = build_features(df_clean, train_mode=True)
        
        # Tuning Hiperparametrów (Optuna)
        logger.info("Optymalizacja Hyperparametrów (Optuna)...")
        run_tuning()  #Ta funkcja zapisze plik 'models/best_params.json'
        
        # Trening Modelu 
        logger.info("Trening finalnego modelu XGBoost")
        trained_model = train_model(df_features, target_col='churn')
        
        logger.info("SUKCES: Pipeline zakończony! Model jest zoptymalizowany.")

    except Exception as e:
        logger.error(f"BŁĄD KRYTYCZNY: Pipeline przerwany.")
        logger.error(f"Szczegóły błędu: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()