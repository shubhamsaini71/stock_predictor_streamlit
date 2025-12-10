# Stock Predictor (Streamlit)
A complete, ready-to-run Streamlit project for basic stock price prediction using historical data from Yahoo Finance (via `yfinance`) and a scikit-learn model (RandomForest or Linear Regression).

## Features
- Fetch historical stock data using `yfinance`.
- Train a model (RandomForestRegressor or LinearRegression) on past `n` days features to predict next-day closing price.
- Save / load trained model with `joblib`.
- Interactive Streamlit UI to train, predict, and visualize results.
- Error-handling and clear instructions.

## Files
- `app.py` - Streamlit application (main entrypoint).
- `data_loader.py` - Fetches and prepares data from yfinance.
- `model.py` - Training, saving, loading, and prediction utilities.
- `utils.py` - Helper functions (feature engineering, evaluation).
- `train.py` - CLI training script (optional standalone use).
- `requirements.txt` - Python dependencies.
- `.gitignore` - files to ignore in git.

## Quick start (local)
1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\\Scripts\\activate   # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
