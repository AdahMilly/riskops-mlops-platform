import pandas as pd


def build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    features = dataframe.copy()
    features["hour"] = features["timestamp"].dt.hour
    features["day_of_week"] = features["timestamp"].dt.dayofweek
    features["amount_to_24h_ratio"] = features["amount"] / features["amount_last_24h"].clip(lower=1)
    features["velocity_risk"] = (features["transactions_last_24h"] >= 10).astype(int)
    features["high_value_transaction"] = (features["amount"] >= 1000).astype(int)

    return features
