# model_utils.py
"""
 This File Shows all the functions and codes for making a model using ensemble apporach,
 Which can be used by other files to call the model to use enter data in to make analysis.

 Author: Vishank
Created: 6 December 2025
"""
import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor

# Try import TCN; provide a Conv1D fallback if not present
try:
    from tcn import TCN
    TCN_AVAILABLE = True
except Exception:
    TCN_AVAILABLE = False

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1)).replace([np.inf, -np.inf], np.nan)
    df["Volatility5"] = df["Returns"].rolling(5).std()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["Momentum"] = df["Close"].diff()
    # RSI14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    df["RSI14"] = 100 - (100 / (1 + rs))
    # ATR14
    df["TR"] = np.maximum(df["High"] - df["Low"], np.maximum(abs(df["High"] - df["Close"].shift(1)), abs(df["Low"] - df["Close"].shift(1))))
    df["ATR14"] = df["TR"].rolling(14).mean()
    # Bollinger
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_STD"] = df["Close"].rolling(20).std()
    df["BB_UP"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOW"] = df["BB_MID"] - 2 * df["BB_STD"]
    df["BB_PCTB"] = (df["Close"] - df["BB_LOW"]) / (df["BB_UP"] - df["BB_LOW"] + 1e-12)
    # OBV
    df["OBV"] = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def build_lag_features(df: pd.DataFrame, lags: List[int] = [1,2,3,5,10,20]) -> pd.DataFrame:
    lagged = pd.DataFrame(index=df.index)
    for lag in lags:
        lagged[f"Close_lag_{lag}"] = df["Close"].shift(lag)
        lagged[f"High_lag_{lag}"] = df["High"].shift(lag)
        lagged[f"Low_lag_{lag}"] = df["Low"].shift(lag)
        lagged[f"Vol_lag_{lag}"] = df["Volume"].shift(lag)
    techs = ["Returns","Log_Returns","Volatility5","MA10","MA20","RSI14","ATR14","BB_PCTB","OBV"]
    for t in techs:
        if t in df.columns:
            lagged[t] = df[t]
    lagged.replace([np.inf,-np.inf], np.nan, inplace=True)
    lagged.dropna(inplace=True)
    return lagged

def create_sequences(arr: np.ndarray, labels: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(labels[i+seq_len])
    return np.array(X), np.array(y)

# Model builders & trainers
def build_tcn_model(input_shape, output_size=4):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, GlobalAveragePooling1D, Dropout
    if TCN_AVAILABLE:
        model = Sequential([
            TCN(nb_filters=64, kernel_size=4, dilations=[1,2,4,8], dropout_rate=0.1, activation="relu", padding="causal"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(output_size)
        ])
    else:
        model = Sequential([
            Conv1D(64, kernel_size=4, padding="causal", activation="relu", input_shape=input_shape),
            Conv1D(64, kernel_size=4, padding="causal", activation="relu"),
            GlobalAveragePooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(output_size)
        ])
    model.compile(optimizer="adam", loss="mae")
    return model

def build_blstm_model(input_shape, output_size=4):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense
    inp = Input(shape=input_shape)
    x = Bidirectional(LSTM(64, return_sequences=False))(inp)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(output_size)(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mae")
    return model

def train_sequence_models(X_train, y_train, X_val, y_val, epochs=55, batch_size=32, seed=42):
    import tensorflow as tf
    tf.random.set_seed(seed)
    input_shape = X_train.shape[1:]
    tcn = build_tcn_model(input_shape, output_size=y_train.shape[1])
    blstm = build_blstm_model(input_shape, output_size=y_train.shape[1])
    # modest epochs so app remains responsive; user can increase later
    tcn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    blstm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    return tcn, blstm

def train_gb_models(X_tab_train: pd.DataFrame, y_tab_train: pd.DataFrame, n_estimators=200, learning_rate=0.05, random_state=42):
    models = []
    for col in range(y_tab_train.shape[1]):
        g = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=4, random_state=random_state)
        g.fit(X_tab_train, y_tab_train.iloc[:, col])
        models.append(g)
    return models

def inverse_y(scaler_y, arr_scaled):
    try:
        return scaler_y.inverse_transform(arr_scaled)
    except Exception:
        return arr_scaled
