from __future__ import annotations

import sys
import os
import torch
from fastapi import FastAPI
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from pydantic import BaseModel

# Add the subfolder to the Python path
subfolder_path = os.path.join(os.path.dirname(__file__), 'ann_model')
sys.path.append(subfolder_path)


from pccf_model_simple import PCCFModelSimple

app = FastAPI()

class Transaction(BaseModel):
    distance_from_home: float | int
    distance_from_last_transaction: float | int
    ratio_to_median_purchase_price: float | int
    repeat_retailer: float | int
    used_chip: float | int
    used_pin_number: float | int
    online_order: float | int


# load ANN Model
pccf_model = PCCFModelSimple()
pccf_model.load_state_dict(torch.load('pccf_model.pth'))
pccf_model.eval()

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
    with torch.no_grad():
        prediction = pccf_model(data)

    probability = torch.sigmoid(prediction)
    return int((probability > 0.5).float())


@app.post('/predict')
async def predict(transaction: Transaction):
    data = parse_data(transaction)

    return int(model.predict(data)[0])


