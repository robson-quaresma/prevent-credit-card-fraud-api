import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from pydantic import BaseModel
from .model import PCCFModelSimple, Predictor


app = FastAPI()

class Transaction(BaseModel):
    distance_from_home: float | int
    distance_from_last_transaction: float | int
    ratio_to_median_purchase_price: float | int
    repeat_retailer: float | int
    used_chip: float | int
    used_pin_number: float | int
    online_order: float | int

model_path = "pccf_model.pth"

# Load ML model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def parse_data(transaction: Transaction):
    return np.array([
        transaction.distance_from_home,
        transaction.distance_from_last_transaction,
        transaction.ratio_to_median_purchase_price,
        transaction.repeat_retailer,
        transaction.used_chip,
        transaction.used_pin_number,
        transaction.online_order
    ]).reshape(1, -1)


@app.get('/')
async def root():
    return {'message': 'Welcome to Prevent Credit Card Fraud API. To see how use this API, go to /docs'}


@app.post('/ann-predict')
async def ann_predict(transaction: Transaction):
    data = torch.tensor(parse_data(transaction)).float()
    predictor = Predictor()
    predictor.load_model(PCCFModelSimple, model_path)
    return predictor.make_prediction(data)    


@app.post('/predict')
async def predict(transaction: Transaction):
    data = parse_data(transaction)
    return int(model.predict(data))

