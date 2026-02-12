from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.serving.inference import ChurnModel

app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")

# Inicjalizacja modelu (ładuje się raz przy starcie aplikacji)
model_service = ChurnModel()

# Definicja schematu danych wejściowych (Validation)
class CustomerData(BaseModel):
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int
    phoneservice: str
    multipleLines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: str  # Czasem przychodzi jako string

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Model is ready"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        # Konwersja Pydantic modelu na zwykły słownik
        input_data = data.dict()
        
        # Wywołanie logiki biznesowej
        result = model_service.predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))