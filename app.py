from fastapi import FastAPI
from pydantic import BaseModel, Field

from inference_pipeline import Preprocessor, Model, Postprocessor, InferenceService

import joblib

model = joblib.load("imdb_tfidf_lr.joblib")

app = FastAPI(title="NLP Deploy Skeleton")

svc = InferenceService(Preprocessor(), Model(), Postprocessor())

class PredictRequest(BaseModel):
    text: str = Field(min_length=1, description="Input text to classify")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": "0.1.0"}

@app.post("/predict")
def predict(req: PredictRequest):
    prob = model.predict_proba([req.text])[0]
    label = 'positive' if prob[1] > 0.5 else 'negative'
    score = float(prob[1])  # probability of positive class
    return {"label": label, "score": score}
