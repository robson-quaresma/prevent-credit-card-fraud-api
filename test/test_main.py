import json
import os
import sys
from fastapi.testclient import TestClient

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from app import main

client = TestClient(main.app)

input_data = {
	"distance_from_home": 5000.25,
    "distance_from_last_transaction": 1,
    "ratio_to_median_purchase_price": 11515150.0845876,
    "repeat_retailer": 1.0,
    "used_chip": 1.0,
    "used_pin_number": 1.0,
    "online_order": 1.0
}

def test_predict():
    response = client.post(
        '/predict',
        json=input_data
    )

    assert response.status_code == 200
    assert response.json() == {"result": 1}

def test_ann_predict():
    response = client.post(
        '/ann-predict',
        json=input_data
    )

    assert response.status_code == 200
    assert response.json() == {"result": 0}
