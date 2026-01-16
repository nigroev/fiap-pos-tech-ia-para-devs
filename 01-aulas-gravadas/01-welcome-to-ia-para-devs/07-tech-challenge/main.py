import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
modelo_salvo = pickle.load(open("pipe_rf_model.pkl", "rb"))

class HealthData(BaseModel):
    id: int
    age: int
    sbp: float
    hba1c: float
    bmi: float
    gender: float
    married: float
    high_bp: float
    chf: float
    smoking: float


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def predict(data: HealthData):
    # Criar DataFrame com os nomes das colunas esperadas pelo modelo
    num_cols = ['RIDAGEYR_age', 'BPXSY1_sbp', 'LBXGH_hba1c', 'BMXBMI_bmi']
    cat_cols = ['RIAGENDR_gender_bin', 'DMDMARTL_married_bin', 'BPQ020_high_bp_bin', 'MCQ160B_chf_bin', 'SMQ020_smoking_bin']
    
    input_data = pd.DataFrame({
        'RIDAGEYR_age': [data.age],
        'BPXSY1_sbp': [data.sbp],
        'LBXGH_hba1c': [data.hba1c],
        'BMXBMI_bmi': [data.bmi],
        'RIAGENDR_gender_bin': [data.gender],
        'DMDMARTL_married_bin': [data.married],
        'BPQ020_high_bp_bin': [data.high_bp],
        'MCQ160B_chf_bin': [data.chf],
        'SMQ020_smoking_bin': [data.smoking]
    })
    
    prediction = modelo_salvo.predict(input_data)
    return {"prediction_stroke": int(prediction[0]), "input": data}