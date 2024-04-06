import numpy as np
from .transaction import Transaction

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