import pandera.pandas as pa
from pandera.typing import Series


class TransactionSchema(pa.DataFrameModel):
    transaction_id: Series[str]
    customer_id: Series[str]
    timestamp: Series[pa.DateTime]
    amount: Series[float]
    currency: Series[str]
    merchant_category: Series[str]
    country: Series[str]
    payment_method: Series[str]
    device_id: Series[str]
    is_international: Series[bool]
    customer_age_days: Series[int]
    transactions_last_24h: Series[int]
    amount_last_24h: Series[float]
    is_fraud: Series[int]

    class Config:
        strict = True
        coerce = True
