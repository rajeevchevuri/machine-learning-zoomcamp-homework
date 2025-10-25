import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn


# datapoint = {
#     'lead_source': 'paid_ads',
#     'number_of_courses_viewed': 2,
#     'annual_income': 79276.0
# }

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


app = FastAPI(title="customer-churn-prediction")

class DataPoint(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(datapoint: DataPoint):
    # Convert to the dict the pipeline expects
    d = datapoint.model_dump()
    result = pipeline.predict_proba([d])[0, 1]  # list of dicts!
    return {"churn_probability": float(result)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)