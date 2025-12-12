# chart_utils.py
"""
 This File Shows all the functions and codes for making callable functions,
 Which can be used in other application file to make visual plots for the data.

 Author: Vishank
Created: 7 December 2025
"""
import plotly.graph_objs as go
import pandas as pd

def plot_candlestick(df, open_col, high_col, low_col, close_col, title="OHLC"):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df[open_col],
        high=df[high_col],
        low=df[low_col],
        close=df[close_col]
    )])
    fig.update_layout(title=title, height=380, margin=dict(l=10,r=10,t=40,b=20))
    return fig

def plot_clean_combined(actual_df, forecast_df, lookback=60, title="Actual vs Forecast — H/L/C"):
    lookback = min(lookback, len(actual_df))
    recent = actual_df.iloc[-lookback:]
    fig = go.Figure()
    # Actual thin
    fig.add_trace(go.Scatter(x=recent.index, y=recent["High"], name="Actual High", line=dict(width=1.2, color="rgba(76,175,80,0.7)")))
    fig.add_trace(go.Scatter(x=recent.index, y=recent["Low"], name="Actual Low", line=dict(width=1.2, color="rgba(244,67,54,0.7)")))
    fig.add_trace(go.Scatter(x=recent.index, y=recent["Close"], name="Actual Close", line=dict(width=1.4, color="rgba(33,150,243,0.9)")))
    # Forecast dashed with markers
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Predicted_High"], name="Forecast High", mode="lines+markers", line=dict(width=2.6, dash="dash", color="#2E7D32"), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Predicted_Low"], name="Forecast Low", mode="lines+markers", line=dict(width=2.6, dash="dash", color="#C62828"), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Predicted_Close"], name="Forecast Close", mode="lines+markers", line=dict(width=2.8, dash="dash", color="#0277BD"), marker=dict(size=6)))
    fig.add_vrect(x0=forecast_df.index[0], x1=forecast_df.index[-1], fillcolor="rgba(200,200,200,0.08)", line_width=0)
    fig.add_vline(x=forecast_df.index[0], line_dash="dot", line_color="gray")
    fig.update_layout(title=title, height=520, legend=dict(orientation="h"), margin=dict(l=10,r=10,t=40,b=30))
    return fig

def plot_forecast_candles(forecast_ohlc_df, title="Forecast Candles"):
    fig = go.Figure(data=[go.Candlestick(
        x=forecast_ohlc_df.index,
        open=forecast_ohlc_df["Open"],
        high=forecast_ohlc_df["High"],
        low=forecast_ohlc_df["Low"],
        close=forecast_ohlc_df["Close"]
    )])
    fig.update_layout(title=title, height=420, margin=dict(l=10,r=10,t=40,b=20))
    return fig
