# Prevent Credit Card Fraud API 

## Overview
A Simple FastAPI API to provide an API to expose the model created with scikit-learn LogisticRegression algorithm based on [Kaggle Global Hack Week](https://www.kaggle.com/competitions/global-hack-week-detect-credit-card-fraud)

## Installation
Make sure to have python installed on your machine.
After clone this repository if you want change the code, create your virtual environment, or just move for Running section:
```
python -m venv environment_name
```
Then activate:
__Windows__:
```
environment_name\\Scripts\\activate
```
__MacOS or Linux__:
```
source environment_name/bin/activate
```

## Running
Install the dependencies:
```
pip install -r requirements.txt
```
### Run the code with uvicorn:
```
uvicorn app.main:app --reload
```
### Run with Docker
Create the docker image
```
docker build -t image_name .
```
Run:
```
docker run -d --name container_name -p 80:80 image_name
```

## Render Deployment:
[Link](https://prevent-credit-card-fraud-api.onrender.com)

### JSON:
```
{
	"distance_from_home": 5000.25,
    "distance_from_last_transaction": 3.894,
    "ratio_to_median_purchase_price": 10.0845876,
    "repeat_retailer": 0.0,
    "used_chip": 0.0,
    "used_pin_number": 1.0,
    "online_order": 1.0
}
```
