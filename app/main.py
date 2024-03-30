from __future__ import annotations

from fastapi import FastAPI
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from pydantic import BaseModel

app = FastAPI()


class Transaction(BaseModel):
    distance_from_home: float | int
    distance_from_last_transaction: float | int
    ratio_to_median_purchase_price: float | int
    repeat_retailer: float | int
    used_chip: float | int
    used_pin_number: float | int
    online_order: float | int


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.get('/')
async def root():
    return {'message': 'Welcome to Prevent Credit Card Fraud API. To see how use this API, go to /docs'}


@app.post('/predict')
async def predict(transaction: Transaction):

    data = np.array([
        transaction.distance_from_home,
        transaction.distance_from_last_transaction,
        transaction.ratio_to_median_purchase_price,
        transaction.repeat_retailer,
        transaction.used_chip,
        transaction.used_pin_number,
        transaction.online_order
    ]).reshape(1, -1)

    return int(model.predict(data)[0])


