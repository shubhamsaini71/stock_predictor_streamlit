import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_features(df: pd.DataFrame, n_lags: int = 5):
    df2 = df.copy()
    for lag in range(1, n_lags+1):
        df2[f'lag_{lag}'] = df2['Close'].shift(lag)
    df2['roll_mean_5'] = df2['Close'].rolling(window=5).mean()
    df2['roll_std_5'] = df2['Close'].rolling(window=5).std()
    df2 = df2.dropna()
    return df2

def train_test_split_by_date(df: pd.DataFrame, test_size: float = 0.2):
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1.')
    split = int(len(df) * (1 - test_size))
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {'mse': mse, 'rmse': rmse, 'mae': mae}
