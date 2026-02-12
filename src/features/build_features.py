import pandas as pd
import joblib
import os
from sklearn.preprocessing import OneHotEncoder
from src.utils import get_logger

logger = get_logger("BUILD_FEATURES")

def build_features(df: pd.DataFrame, train_mode: bool = True) -> pd.DataFrame:
    """
    Transformuje dane.
    W trybie train: uczy encoder i go zapisuje.
    W trybie predict (train_mode=False): ładuje encoder i tylko transformuje.
    """
    df_featured = df.copy()
    artifacts_dir = "models"
    encoder_path = os.path.join(artifacts_dir, "encoder.joblib")
    
    target_col = 'churn' # Zakładamy lowercase po preprocessingu
    y = None
    if target_col in df_featured.columns:
        y = df_featured[target_col]
        df_featured = df_featured.drop(columns=[target_col])
    
    # Wykrywanie kolumn (musimy być pewni, że typy są ok)
    cat_cols = df_featured.select_dtypes(include=['object']).columns.tolist()
    num_cols = df_featured.select_dtypes(include=['number']).columns.tolist()
    
    if train_mode:
        os.makedirs(artifacts_dir, exist_ok=True)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_array = encoder.fit_transform(df_featured[cat_cols])
        joblib.dump(encoder, encoder_path)
    else:
        if not os.path.exists(encoder_path):
            raise FileNotFoundError("Brak encodera! Uruchom najpierw trening.")
        encoder = joblib.load(encoder_path)
        
        # Weryfikacja czy w nowych danych są te same kolumny kategoryczne co przy treningu
        # Encoder wymaga dokładnie tych samych kolumn wejściowych.
        encoded_array = encoder.transform(df_featured[cat_cols])
    
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_featured.index)
    
    # Składanie całości
    df_final = pd.concat([df_featured[num_cols], df_encoded], axis=1)
    
    # Jeśli mieliśmy target (np. w zbiorze testowym), doklejamy go
    if y is not None:
        df_final[target_col] = y
        
    return df_final