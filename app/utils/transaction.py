from pydantic import BaseModel

class Transaction(BaseModel):
    distance_from_home: float | int
    distance_from_last_transaction: float | int
    ratio_to_median_purchase_price: float | int
    repeat_retailer: float | int
    used_chip: float | int
    used_pin_number: float | int
    online_order: float | int