import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report
from src.utils import get_logger

# Ten skrypt zakłada, że masz osobny zbiór testowy, 
# ale dla uproszczenia tutoriala często go pomijamy,
# bo MLflow robi robotę w train_model.py.
# Zostawmy go jako placeholder na przyszłość.

def evaluate_model():
    print("Ewaluacja jest wykonywana bezpośrednio w train_model.py i logowana do MLflow.")

if __name__ == "__main__":
    evaluate_model()