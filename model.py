import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def build_model(model_name='random_forest', **kwargs):
    if model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators=kwargs.get('n_estimators', 100),
                                      random_state=kwargs.get('random_state', 42),
                                      max_depth=kwargs.get('max_depth', None))
    elif model_name == 'linear_regression':
        model = LinearRegression()
    else:
        raise ValueError('Unknown model_name: ' + model_name)
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    return pipe

def train_and_save(X_train, y_train, model_name='random_forest', model_id='default', **kwargs):
    pipe = build_model(model_name=model_name, **kwargs)
    pipe.fit(X_train, y_train)
    path = os.path.join(MODEL_DIR, f"{model_id}_{model_name}.pkl")
    joblib.dump(pipe, path)
    return path

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

def predict(model, X):
    return model.predict(X)
