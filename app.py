import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import fetch_data
from utils import create_features, train_test_split_by_date, evaluate
from model import build_model, train_and_save, load_model, predict, MODEL_DIR
import os, glob

st.set_page_config(page_title='Stock Price Predictor', layout='wide')
st.title('Stock Price Predictor (Streamlit)')
st.write('Fetch historical data with yfinance, train a model, and predict next-day closing price.')

with st.sidebar:
    ticker = st.text_input('Ticker (Yahoo Finance format)', value='AAPL')
    start_date = st.date_input('Start date', value=pd.to_datetime('2019-01-01'))
    end_date = st.date_input('End date', value=pd.to_datetime(pd.Timestamp.today().date()))
    n_lags = st.number_input('Number of lag days (features)', min_value=1, max_value=30, value=5)
    test_size = st.slider('Test set fraction', min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    model_choice = st.selectbox('Model', ['random_forest', 'linear_regression'])
    n_estimators = None
    if model_choice == 'random_forest':
        n_estimators = st.number_input('n_estimators (RF)', min_value=10, max_value=1000, value=100, step=10)
    train_button = st.button('Fetch & Train')

def list_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    files = glob.glob(os.path.join(MODEL_DIR, '*.pkl'))
    return files

st.write('Saved models:')
for p in list_models():
    st.write('-', p)

if train_button:
    try:
        with st.spinner('Fetching data...'):
            df = fetch_data(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        st.success(f'Fetched {len(df)} rows for {ticker}')
        st.write(df.tail())

        df_feat = create_features(df, n_lags=n_lags)
        df_feat['target'] = df_feat['Close'].shift(-1)
        df_feat = df_feat.dropna()

        X = df_feat[[c for c in df_feat.columns if c.startswith('lag_') or c.startswith('roll_')]]
        y = df_feat['target']

        split = int(len(df_feat) * (1 - test_size))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        st.write('Training set size:', X_train.shape)
        st.write('Test set size:', X_test.shape)

        kwargs = {}
        if model_choice == 'random_forest':
            kwargs['n_estimators'] = int(n_estimators)
        model_id = f"{ticker}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = train_and_save(X_train, y_train, model_name=model_choice, model_id=model_id, **kwargs)
        st.success('Model trained and saved to ' + model_path)

        model = load_model(model_path)
        preds = predict(model, X_test)
        metrics = evaluate(y_test, preds)
        st.write('Evaluation on test set:', metrics)

        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(y_test.index, y_test.values, label='Actual')
        ax.plot(y_test.index, preds, label='Predicted')
        ax.set_title(f'Actual vs Predicted Close Price ({ticker})')
        ax.legend()
        st.pyplot(fig)

        latest_row = df_feat.iloc[[-1]]
        X_latest = latest_row[[c for c in latest_row.columns if c.startswith('lag_') or c.startswith('roll_')]]
        next_pred = predict(model, X_latest)[0]
        st.write(f'Predicted next day Close price: **{next_pred:.4f}**')

    except Exception as e:
        st.error('Error: ' + str(e))

st.markdown('---')
st.write('You can also train offline using `train.py` and then place the model in the `models/` folder.')
