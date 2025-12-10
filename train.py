import argparse
import pandas as pd
from data_loader import fetch_data
from utils import create_features, train_test_split_by_date
from model import train_and_save

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--start', default=None)
    parser.add_argument('--end', default=None)
    parser.add_argument('--model', default='random_forest', choices=['random_forest','linear_regression'])
    parser.add_argument('--model_id', default='run1')
    args = parser.parse_args()

    df = fetch_data(args.ticker, start=args.start, end=args.end)
    df_feat = create_features(df, n_lags=5)
    df_feat['target'] = df_feat['Close'].shift(-1)
    df_feat = df_feat.dropna()
    X = df_feat[[c for c in df_feat.columns if c.startswith('lag_') or c.startswith('roll_')]]
    y = df_feat['target']

    split = int(len(X)*0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    path = train_and_save(X_train, y_train, model_name=args.model, model_id=args.model_id)
    print('Saved model to', path)

if __name__ == '__main__':
    main()
