import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
modelo_salvo = pickle.load(open("pipe_lr_model.pkl", "rb"))

class HealthData(BaseModel):
    age: int
    sbp: float
    hba1c: float
    bmi: float
    gender: float
    married: float
    high_bp: float
    chf: float
    occupation: float
    smoking: float


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(data: HealthData):
    
    # Colunas numéricas e categóricas (MESMA ORDEM DO TREINAMENTO)
    input_data = pd.DataFrame({
        'RIDAGEYR_age': [data.age],
        'BPXSY1_sbp': [data.sbp],
        'LBXGH_hba1c': [data.hba1c],
        'BMXBMI_bmi': [data.bmi],
        'RIAGENDR_gender_bin': [data.gender],
        'DMDMARTL_married_bin': [data.married],
        'BPQ020_high_bp_bin': [data.high_bp],
        'MCQ160B_chf_bin': [data.chf],
        'OCQ260_occupation': [data.occupation],
        'SMQ020_smoking_bin': [data.smoking]
    })
    
    # Fazer predição e obter probabilidade
    prediction = modelo_salvo.predict(input_data)
    prediction_proba = modelo_salvo.predict_proba(input_data)
    
    return {
        "prediction_stroke": int(prediction[0]), # converter numpy.int64 para int
        "probability_no_stroke": round(float(prediction_proba[0][0]), 4), # converter numpy.float64 para float
        "probability_stroke": round(float(prediction_proba[0][1]), 4), # converter numpy.float64 para float
        "input": data # retornar os dados de entrada para verificação
    }