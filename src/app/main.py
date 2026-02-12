from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.serving.inference import ChurnModel

app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")

# Inicjalizacja modelu (ładuje się raz przy starcie aplikacji)
model_service = ChurnModel()

# Definicja schematu danych wejściowych (Validation)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str  # Czasem przychodzi jako string

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