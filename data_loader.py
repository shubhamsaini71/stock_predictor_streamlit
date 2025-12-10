# data_loader.py  (replace the file with this)
import yfinance as yf
import pandas as pd

def _flatten_columns(cols):
    """
    Turn MultiIndex column tuples into single-level names.
    For tuple like ('Close','AAPL') -> 'Close'
    If every level is empty, join with '_'.
    """
    new = []
    for c in cols:
        if isinstance(c, tuple):
            # choose the first non-empty / non-None element (usually 'Close' etc.)
            chosen = next((str(x) for x in c if x not in (None, "") ), None)
            if chosen is None:
                # fallback: join all parts with '_'
                chosen = "_".join(str(x) for x in c)
            new.append(chosen)
        else:
            new.append(c)
    return new

def fetch_data(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for `ticker` from Yahoo Finance and normalize column names.
    Returns a DataFrame with columns: ['Open','High','Low','Close','Adj_Close','Volume'] and datetime index.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data fetched for ticker {ticker}. Check the ticker symbol and your internet connection.")

    # If columns are MultiIndex or tuples, flatten them to single-level names
    if hasattr(df.columns, "levels") and getattr(df.columns, "nlevels", 1) > 1:
        df.columns = _flatten_columns(df.columns)
    else:
        # also handle rare case of tuple-like column elements
        if any(isinstance(c, tuple) for c in df.columns):
            df.columns = _flatten_columns(df.columns)

    # Try to normalize possible 'adjusted close' column name variants to 'Adj_Close'
    # Common variants: 'Adj Close', 'Adj_Close', 'AdjClose', 'Adj'
    adj_candidates = ['Adj Close', 'Adj_Close', 'AdjClose', 'Adj']
    for candidate in adj_candidates:
        if candidate in df.columns:
            df = df.rename(columns={candidate: 'Adj_Close'})
            break

    # Some Yahoo returns 'Close' as already adjusted; keep that but prefer Adj_Close if exists
    # Required column names expected by the app:
    required_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']

    # If Adj_Close is missing but 'Close' exists, we can create Adj_Close = Close (best-effort)
    if 'Adj_Close' not in df.columns and 'Close' in df.columns:
        df['Adj_Close'] = df['Close']

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from data for {ticker}: {missing}. Available columns: {list(df.columns)}")

    # Keep only required columns in this normalized order
    df = df[required_cols].copy()

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    return df
