# app.py
"""
 This File Shows all the functions and codes for making a fully interactive Streamlit dashboard ,
 Which can be used by users who cannot understand coding languages.

 Author: Vishank
Created: 8 December 2025
"""
import os, datetime, numpy as np, pandas as pd, streamlit as st
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

from ticker_utils import name_to_ticker_safe, normalize_columns, get_col
from model_utils import add_indicators, build_lag_features, create_sequences, train_sequence_models, train_gb_models, inverse_y
from chart_utils import plot_candlestick, plot_clean_combined, plot_forecast_candles

# determinism
import random, tensorflow as tf
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seed(42)

st.set_page_config(layout="wide")
st.title("Ensemble OHLC Forecast — TCN + BiLSTM + GBoost")

# Sidebar
st.sidebar.markdown("### Stock ForeCasting And Predictive Analysis")
name = st.sidebar.text_input("Company name or ticker (e.g. Tesla or TSLA)", value="TSLA")
today = datetime.date.today()
end_date = st.sidebar.date_input("End date (data up to this date)", value=today, max_value=today)
periods = st.sidebar.selectbox("Recent period", ["1wk","1mo","3mo","6mo","1y","5y"], index=3)
forecast_options = {"Next Day":1, "1 Week (5 days)":5, "2 Weeks (10 days)":10, "1 Month (20 days)":20}
choice = st.sidebar.selectbox("Forecast horizon", list(forecast_options.keys()))
forecast_days = forecast_options[choice]
run_button = st.sidebar.button("Run Analysis")

if not run_button:
    st.info("Configure inputs and click Run Analysis")
    st.stop()

# Resolve ticker
ticker = name_to_ticker_safe(name)
st.sidebar.success(f"Using ticker: {ticker}")

# Download data
import yfinance as yf
df_raw = yf.download(ticker, start="2015-01-01", end=str(end_date), progress=False)
if df_raw is None or df_raw.empty:
    st.error("No data returned. Check ticker/date.")
    st.stop()

# Normalize columns
df_norm = normalize_columns(df_raw, ticker)
try:
    open_col = get_col(df_norm, "Open", ticker)
    high_col = get_col(df_norm, "High", ticker)
    low_col = get_col(df_norm, "Low", ticker)
    close_col = get_col(df_norm, "Close", ticker)
    vol_col = get_col(df_norm, "Volume", ticker)
except KeyError as e:
    st.error(str(e))
    st.stop()

df_work = df_norm[[open_col, high_col, low_col, close_col, vol_col]].copy()
df_work.columns = ["Open","High","Low","Close","Volume"]

# Show recent candlestick
st.write("### Recent Candles")
try:
    st.plotly_chart(plot_candlestick(df_work.tail(250), "Open","High","Low","Close", title="Recent candles"), width='stretch')
except Exception:
    st.line_chart(df_work["Close"].tail(250))

# Feature engineering
df_work = add_indicators(df_work)
df_work.dropna(inplace=True)
if df_work.shape[0] < 80:
    st.warning("Limited history; results may be less accurate.")

# Target and features
target = df_work[["High","Low","Close"]].shift(-1).dropna()
features = df_work.iloc[:-1].copy()
feature_cols = features.columns.tolist()

# Lagged tabular
lagged = build_lag_features(df_work)
lagged = lagged.reindex(target.index).dropna()
common_idx = target.index.intersection(lagged.index).intersection(features.index)
features = features.loc[common_idx]
target = target.loc[common_idx]
lagged = lagged.loc[common_idx]
if len(features) < 50:
    st.error("Not enough aligned data after feature engineering. Increase date range.")
    st.stop()

# Scaling (fit on train rows only)
SEQ_LEN = min(30, max(10, int(len(features) * 0.2)))
train_row_cut = int(len(features) * 0.8)
if train_row_cut <= SEQ_LEN:
    train_row_cut = SEQ_LEN + 1

scaler_X = RobustScaler()
scaler_y = RobustScaler()
scaler_X.fit(features.iloc[:train_row_cut])
scaler_y.fit(target.iloc[:train_row_cut])

features_scaled = pd.DataFrame(scaler_X.transform(features), index=features.index, columns=feature_cols)
target_scaled = pd.DataFrame(scaler_y.transform(target), index=target.index, columns=target.columns)

# Sequences
X_seq, y_seq = create_sequences(features_scaled.values, target_scaled.values, SEQ_LEN)
seq_dates = list(target_scaled.index[SEQ_LEN:])
if len(X_seq) == 0:
    st.error("Not enough sequence rows. Increase history.")
    st.stop()

# Train/test splits (time-ordered)
seq_cut = int(len(X_seq) * 0.8)
X_train_seq, X_test_seq = X_seq[:seq_cut], X_seq[seq_cut:]
y_train_seq, y_test_seq = y_seq[:seq_cut], y_seq[seq_cut:]
dates_test_seq = seq_dates[seq_cut:]

X_tab = lagged.copy()
y_tab = target.loc[lagged.index]
tab_train_cut = int(len(X_tab) * 0.8)
X_train_tab, X_test_tab = X_tab.iloc[:tab_train_cut], X_tab.iloc[tab_train_cut:]
y_train_tab, y_test_tab = y_tab.iloc[:tab_train_cut], y_tab.iloc[tab_train_cut:]

# Train models (cached to avoid re-training each interaction)
@st.cache_resource
def train_all(X_tr_seq, y_tr_seq, X_val_seq, y_val_seq, X_tr_tab, y_tr_tab):
    tcn_model, blstm_model = train_sequence_models(X_tr_seq, y_tr_seq, X_val_seq, y_val_seq, epochs=10, batch_size=32)
    gbr_models = train_gb_models(X_tr_tab, y_tr_tab, n_estimators=150)
    return tcn_model, blstm_model, gbr_models

tcn_model, blstm_model, gbr_models = train_all(X_train_seq, y_train_seq, X_test_seq, y_test_seq, X_train_tab, y_train_tab)

# Predict test
tcn_pred_s = tcn_model.predict(X_test_seq)
blstm_pred_s = blstm_model.predict(X_test_seq)
tcn_pred = scaler_y.inverse_transform(tcn_pred_s)
blstm_pred = scaler_y.inverse_transform(blstm_pred_s)
y_test_unscaled = scaler_y.inverse_transform(y_test_seq)
ensemble = (0.55 * tcn_pred + 0.45 * blstm_pred)  # weighted average (sequence models dominate)

# Test metrics
df_test_plot = pd.DataFrame({
    "Date": dates_test_seq,
    "Actual_High": y_test_unscaled[:,0],
    "Actual_Low": y_test_unscaled[:,1],
    "Actual_Close": y_test_unscaled[:,2],
    "Ensemble_High": ensemble[:,0],
    "Ensemble_Low": ensemble[:,1],
    "Ensemble_Close": ensemble[:,2]
})
df_test_plot.set_index("Date", inplace=True)
mae_h = mean_absolute_error(df_test_plot["Actual_High"], df_test_plot["Ensemble_High"])
mae_l = mean_absolute_error(df_test_plot["Actual_Low"], df_test_plot["Ensemble_Low"])
mae_c = mean_absolute_error(df_test_plot["Actual_Close"], df_test_plot["Ensemble_Close"])
st.write("### Test MAE (Ensemble)")
st.write(f"High MAE: {mae_h:.4f}  |  Low MAE: {mae_l:.4f}  |  Close MAE: {mae_c:.4f}")

# Clean test plot (combined)
fig_test = plot_clean_combined(df_work[["High","Low","Close"]], pd.DataFrame({
    "Predicted_High": df_test_plot["Ensemble_High"],
    "Predicted_Low": df_test_plot["Ensemble_Low"],
    "Predicted_Close": df_test_plot["Ensemble_Close"]
}, index=df_test_plot.index).iloc[-60:], lookback=60, title="Test: Actual (recent) vs Ensemble")
st.plotly_chart(fig_test, width='stretch')

# Forecast multi-day autoregressive with lightweight GB adjustment
st.write(f"### 🔮 Forecast for next {forecast_days} business days")
raw_features_df = features.copy().reset_index(drop=True)
current_seq = features_scaled.values[-SEQ_LEN:].copy()
future_preds = []

for step in range(forecast_days):
    seq_in = current_seq.reshape(1, SEQ_LEN, current_seq.shape[1])
    p_tcn_s = tcn_model.predict(seq_in)
    p_blstm_s = blstm_model.predict(seq_in)
    p_tcn = scaler_y.inverse_transform(p_tcn_s)[0]
    p_blstm = scaler_y.inverse_transform(p_blstm_s)[0]
    p_ensemble = (0.55 * p_tcn + 0.45 * p_blstm)

    # GB tabular correction
    latest_lags = build_lag_features(raw_features_df).iloc[[-1]]
    if not latest_lags.empty:
        gpreds = []
        for g in gbr_models:
            try:
                gpreds.append(g.predict(latest_lags)[0])
            except Exception:
                gpreds.append(np.nan)
        gpred = np.array(gpreds)
        if not np.isnan(gpred).any():
            p_final = 0.6 * p_ensemble + 0.4 * gpred  # weighted ensemble
        else:
            p_final = p_ensemble
    else:
        p_final = p_ensemble

    future_preds.append(p_final.tolist())

    # Build next raw row and append (minimal indicator updates)
    last_row = raw_features_df.iloc[-1].copy()
    ph, pl, pc = float(p_final[0]), float(p_final[1]), float(p_final[2])
    new_row = {}
    new_row["Open"] = last_row["Close"]
    new_row["High"] = ph
    new_row["Low"] = pl
    new_row["Close"] = pc
    new_row["Volume"] = last_row["Volume"]
    new_row["Returns"] = (pc - last_row["Close"]) / (last_row["Close"] + 1e-12)
    returns_series = pd.concat([raw_features_df["Returns"], pd.Series([new_row["Returns"]])], ignore_index=True)
    new_row["Volatility5"] = returns_series.tail(5).std()
    close_series = pd.concat([raw_features_df["Close"], pd.Series([pc])], ignore_index=True)
    new_row["MA10"] = close_series.tail(10).mean()
    new_row["MA20"] = close_series.tail(20).mean()
    try:
        new_row["EMA9"] = close_series.ewm(span=9, adjust=False).mean().iloc[-1]
    except Exception:
        new_row["EMA9"] = new_row["MA10"]
    new_row["Momentum"] = pc - last_row["Close"]
    # RSI approx
    delta = close_series.diff().fillna(0)
    gain = delta.clip(lower=0).tail(14).mean()
    loss = (-delta.clip(upper=0)).tail(14).mean()
    new_row["RSI14"] = 100 - (100 / (1 + (gain / (loss + 1e-12))))
    new_row["ATR14"] = raw_features_df["ATR14"].iloc[-1] if "ATR14" in raw_features_df.columns else raw_features_df["Close"].diff().abs().rolling(14).mean().iloc[-1]
    new_row["BB_MID"] = close_series.tail(20).mean()
    new_row["BB_STD"] = close_series.tail(20).std()
    new_row["BB_UP"] = new_row["BB_MID"] + 2 * new_row["BB_STD"]
    new_row["BB_LOW"] = new_row["BB_MID"] - 2 * new_row["BB_STD"]
    new_row["BB_PCTB"] = (pc - new_row["BB_LOW"]) / (new_row["BB_UP"] - new_row["BB_LOW"] + 1e-12)
    new_row["OBV"] = raw_features_df["OBV"].iloc[-1] + np.sign(pc - last_row["Close"]) * new_row["Volume"]

    raw_features_df = pd.concat([raw_features_df, pd.DataFrame([new_row])], ignore_index=True)

    # fill missing feature columns and scale the new row
    for c in features.columns:
        if c not in raw_features_df.columns:
            raw_features_df[c] = raw_features_df[c].ffill().bfill()
    new_row_df = raw_features_df.iloc[[-1]][features.columns]
    new_row_scaled = RobustScaler().fit(features).transform(new_row_df) if False else scaler_X.transform(new_row_df)
    current_seq = np.vstack([current_seq[1:], new_row_scaled[0]])

# Build forecast DataFrame
future_idx = pd.date_range(start=df_work.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
forecast_df = pd.DataFrame(future_preds, columns=["Predicted_High","Predicted_Low","Predicted_Close"], index=future_idx)
last_close = df_work["Close"].iloc[-1]
opens = []
prev = last_close
for r in future_preds:
    opens.append(prev)
    prev = r[2]
forecast_df["Predicted_Open"] = opens

st.write("### Forecast (OHLC)")
st.dataframe(forecast_df[["Predicted_Open","Predicted_High","Predicted_Low","Predicted_Close"]])

# Plots
fig_comb = plot_clean_combined(df_work[["High","Low","Close"]], forecast_df[["Predicted_High","Predicted_Low","Predicted_Close"]], lookback=60)
st.plotly_chart(fig_comb,width='stretch')

forecast_ohlc = forecast_df[["Predicted_Open","Predicted_High","Predicted_Low","Predicted_Close"]].copy()
forecast_ohlc.columns = ["Open","High","Low","Close"]
fig_candle = plot_forecast_candles(forecast_ohlc, title="Forecast Candles")
st.plotly_chart(fig_candle, width='stretch')

st.success("Forecast complete. To improve accuracy: increase epochs in model training, or enable GPU.")
