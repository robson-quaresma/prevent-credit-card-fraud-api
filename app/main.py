import torch
from fastapi import FastAPI
from .model import PCCFModel, PCCFPredictor, LRPredictor
from .utils import parse_data, Transaction

app = FastAPI()

# model path
model_path = "./app/artifacts"
ann_model_path = f"{model_path}/pccf_model.pth"
ml_model_path =  f"{model_path}/model.pkl"


@app.get('/')
async def root():
    return {'message': 'Welcome to Prevent Credit Card Fraud API. To see how to use this API, go to /docs'}


@app.post('/ann-predict')
async def ann_predict(transaction: Transaction):
    data = torch.tensor(parse_data(transaction)).float()
    predict = PCCFPredictor().predict(PCCFModel, ann_model_path, data)
    return {"result": predict}


@app.post('/predict')
async def predict(transaction: Transaction):
    data = parse_data(transaction)
    predict = LRPredictor().predict(ml_model_path, data)
    return {"result": predict}

