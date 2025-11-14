"""Forecasting and metric helpers for ZenStock."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from prophet import Prophet  # type: ignore

    PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet not installed. Using simple moving average for forecasting.")


__all__ = [
    "forecast_stock",
    "predict_stock_zero_date",
    "get_status_color",
    "calculate_reorder_date",
    "calculate_loss_per_day",
    "create_forecast_chart",
]


def _forecast_prophet(product_data: pd.DataFrame, days_ahead: int) -> float:
    if not PROPHET_AVAILABLE or len(product_data) < 10:
        return _forecast_simple(product_data)
    try:
        prophet_data = product_data.groupby('Date')['Quantity Sold'].sum().reset_index()
        prophet_data.columns = ['ds', 'y']
        model = Prophet(daily_seasonality=True, yearly_seasonality=False)
        model.fit(prophet_data)
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        future_sales = forecast[forecast['ds'] > prophet_data['ds'].max()]['yhat'].values
        avg_predicted_sales = max(future_sales.mean(), 0.1)
        return avg_predicted_sales
    except Exception:
        return _forecast_simple(product_data)


def _forecast_simple(product_data: pd.DataFrame) -> float:
    recent_sales = product_data['Quantity Sold'].tail(7).mean()
    return max(recent_sales, 0.1)


def forecast_stock(product_data: pd.DataFrame, days_ahead: int = 30) -> float:
    if PROPHET_AVAILABLE and len(product_data) >= 10:
        return _forecast_prophet(product_data, days_ahead)
    return _forecast_simple(product_data)


def predict_stock_zero_date(current_stock: float, daily_sales_rate: float):
    if daily_sales_rate <= 0:
        return None
    days_until_zero = current_stock / daily_sales_rate
    zero_date = datetime.now() + timedelta(days=days_until_zero)
    return zero_date, days_until_zero


def get_status_color(days_until_zero: float):
    if days_until_zero < 7:
        return "🔴 Critical", "status-red"
    if days_until_zero < 14:
        return "🟡 Warning", "status-yellow"
    return "🟢 Safe", "status-green"


def calculate_reorder_date(stock_zero_date: datetime, lead_time_days: int):
    try:
        return stock_zero_date - timedelta(days=lead_time_days)
    except Exception:
        return None


def calculate_loss_per_day(avg_daily_sales: float, profit_per_unit: float) -> float:
    return max(0.0, avg_daily_sales) * max(0.0, profit_per_unit)


def create_forecast_chart(df: pd.DataFrame, product_name: str, demand_multiplier: float = 1.0):
    product_data = df[df['Product'] == product_name].copy()
    if len(product_data) == 0:
        return None

    historical = product_data.groupby('Date')['Quantity Sold'].sum().reset_index()

    if PROPHET_AVAILABLE and len(historical) >= 10:
        try:
            prophet_data = historical.copy()
            prophet_data.columns = ['ds', 'y']
            model = Prophet(daily_seasonality=True, yearly_seasonality=False)
            model.fit(prophet_data)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            forecast.loc[forecast['ds'] > prophet_data['ds'].max(), 'yhat'] *= demand_multiplier
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=historical['Date'], y=historical['Quantity Sold'], mode='lines+markers', name='Historical Sales', line=dict(color='blue')))
            future_forecast = forecast[forecast['ds'] > historical['Date'].max()]
            fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name=f'Forecast (Demand x{demand_multiplier})', line=dict(color='red', dash='dash')))
            fig.update_layout(title=f'Sales Forecast for {product_name}', xaxis_title='Date', yaxis_title='Quantity Sold', hovermode='x unified')
            return fig
        except Exception:
            pass

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical['Date'], y=historical['Quantity Sold'], mode='lines+markers', name='Historical Sales', line=dict(color='blue')))
    avg_sales = historical['Quantity Sold'].mean() * demand_multiplier
    future_dates = pd.date_range(start=historical['Date'].max() + timedelta(days=1), periods=30)
    future_sales = [avg_sales] * 30
    fig.add_trace(go.Scatter(x=future_dates, y=future_sales, mode='lines', name=f'Simple Forecast (Demand x{demand_multiplier})', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f'Sales Forecast for {product_name}', xaxis_title='Date', yaxis_title='Quantity Sold', hovermode='x unified')
    return fig
