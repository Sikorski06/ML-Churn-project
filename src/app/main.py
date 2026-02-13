from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import ChurnModel

# Inicjalizacja Aplikacji i Modelu
app = FastAPI(title="Telco Churn Prediction API", version="1.0.0")
model_service = ChurnModel()

# Definicja Danych Wejściowych (Pydantic)
# Używamy małych liter, żeby pasowało do preprocessingu
class CustomerData(BaseModel):
    gender: str = "Male"
    seniorcitizen: int = 0
    partner: str = "No"
    dependents: str = "No"
    tenure: int = 1
    phoneservice: str = "Yes"
    multiplelines: str = "No"
    internetservice: str = "DSL"
    onlinesecurity: str = "No"
    onlinebackup: str = "No"
    deviceprotection: str = "No"
    techsupport: str = "No"
    streamingtv: str = "No"
    streamingmovies: str = "No"
    contract: str = "Month-to-month"
    paperlessbilling: str = "Yes"
    paymentmethod: str = "Electronic check"
    monthlycharges: float = 29.85
    totalcharges: str = "29.85"

# ENDPOINTY FASTAPI

@app.get("/")
def health_check():
    return {"status": "ok", "threshold": model_service.threshold}

@app.post("/predict")
def predict_churn_api(data: CustomerData):
    try:
        return model_service.predict(data.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# GRADIO UI
# Funkcja wrapper dla Gradio (przyjmuje argumenty pozycyjne)
def gradio_predict(gender, seniorcitizen, partner, dependents, tenure, phoneservice, 
                   multiplelines, internetservice, onlinesecurity, onlinebackup, 
                   deviceprotection, techsupport, streamingtv, streamingmovies, 
                   contract, paperlessbilling, paymentmethod, monthlycharges, totalcharges):
    
    safe_gender = "Female" if gender == "Other" else gender
    # Budujemy słownik z danych z formularza
    data_dict = {
        "gender": safe_gender,
        "seniorcitizen": int(seniorcitizen),
        "partner": partner,
        "dependents": dependents,
        "tenure": int(tenure),
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "monthlycharges": float(monthlycharges),
        "totalcharges": str(totalcharges)
    }
    
    result = model_service.predict(data_dict)
    
    # Formatowanie wyniku dla użytkownika
    emoji = "🚨" if result['churn_prediction'] == 1 else "✅"
    message = f"{emoji} Wynik: {'ODEJDZIE' if result['churn_prediction'] == 1 else 'ZOSTANIE'}"
    details = f"Prawdopodobieństwo: {result['churn_probability']:.2%}\nTwój próg decyzji: {result['threshold_used']:.2f}"
    
    return message, details

# Definicja Interfejsu
demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Dropdown(["Female", "Male", "Other"], label="Gender", value="Female"),
        gr.Radio(["0", "1"], label="Senior Citizen"),
        gr.Radio(["Yes", "No"], label="Partner"),
        gr.Radio(["Yes", "No"], label="Dependents"),
        gr.Slider(0, 72, label="Tenure (Miesiące)"),
        gr.Radio(["Yes", "No"], label="Phone Service"),
        gr.Radio(["No phone service", "No", "Yes"], label="Multiple Lines"),
        gr.Radio(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Radio(["No internet service", "No", "Yes"], label="Online Security"),
        gr.Radio(["No internet service", "No", "Yes"], label="Online Backup"),
        gr.Radio(["No internet service", "No", "Yes"], label="Device Protection"),
        gr.Radio(["No internet service", "No", "Yes"], label="Tech Support"),
        gr.Radio(["No internet service", "No", "Yes"], label="Streaming TV"),
        gr.Radio(["No internet service", "No", "Yes"], label="Streaming Movies"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Radio(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges")
    ],
    outputs=[gr.Text(label="Decyzja"), gr.Text(label="Szczegóły")],
    title="Telco Churn Predictor",
    description="Wprowadź dane klienta, aby sprawdzić ryzyko odejścia.",
    flagging_mode="never"
)

# Montujemy Gradio w FastAPI pod ścieżką /ui
app = gr.mount_gradio_app(app, demo, path="/ui")