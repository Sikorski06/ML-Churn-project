import pandas as pd
import os
from src.utils import get_logger

logger = get_logger("LOAD_DATA")

def load_data(file_path: str) -> pd.DataFrame:
    #Wczytuje dane z CSV z obsługą błędów
    if not os.path.exists(file_path):
        logger.error(f"Plik nie istnieje: {file_path}")
        raise FileNotFoundError(f"Brak pliku: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Wczytano dane pomyślnie. Rozmiar: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Błąd podczas odczytu CSV: {e}")
        raise e