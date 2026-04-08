"""
inference.py — Script de inferência para o SageMaker Endpoint
==============================================================
Funções model_fn, input_fn, predict_fn e output_fn usadas pelo
SageMaker SKLearn Serving Container.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd


def model_fn(model_dir):
    """Carrega o modelo serializado do diretório."""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    """Deserializa o payload de entrada."""
    if request_content_type == "application/json":
        data = json.loads(request_body)

        # Mapeamento de nomes amigáveis para nomes do modelo
        rename_map = {
            "age": "RIDAGEYR_age",
            "sbp": "BPXSY1_sbp",
            "hba1c": "LBXGH_hba1c",
            "bmi": "BMXBMI_bmi",
            "gender": "RIAGENDR_gender_bin",
            "married": "DMDMARTL_married_bin",
            "high_bp": "BPQ020_high_bp_bin",
            "chf": "MCQ160B_chf_bin",
            "occupation": "OCQ260_occupation",
            "smoking": "SMQ020_smoking_bin",
        }

        # Suporta entrada única (dict) ou lista de registros
        if isinstance(data, dict):
            data = [data]

        # Renomear chaves se necessário
        renamed_data = []
        for record in data:
            renamed = {rename_map.get(k, k): v for k, v in record.items()}
            renamed_data.append(renamed)

        return pd.DataFrame(renamed_data)

    elif request_content_type == "text/csv":
        from io import StringIO

        columns = [
            "RIDAGEYR_age",
            "BPXSY1_sbp",
            "LBXGH_hba1c",
            "BMXBMI_bmi",
            "RIAGENDR_gender_bin",
            "DMDMARTL_married_bin",
            "BPQ020_high_bp_bin",
            "MCQ160B_chf_bin",
            "OCQ260_occupation",
            "SMQ020_smoking_bin",
        ]
        return pd.read_csv(StringIO(request_body), header=None, names=columns)

    raise ValueError(f"Content type não suportado: {request_content_type}")


def predict_fn(input_data, model):
    """Executa a predição usando o modelo carregado."""
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    return {"predictions": predictions.tolist(), "probabilities": probabilities.tolist()}


def output_fn(prediction, accept):
    """Serializa a resposta da predição."""
    if accept == "application/json":
        response = []
        for i, pred in enumerate(prediction["predictions"]):
            response.append(
                {
                    "prediction_stroke": int(pred),
                    "probability_no_stroke": round(float(prediction["probabilities"][i][0]), 4),
                    "probability_stroke": round(float(prediction["probabilities"][i][1]), 4),
                }
            )
        return json.dumps(response)

    raise ValueError(f"Accept type não suportado: {accept}")
