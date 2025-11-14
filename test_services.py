"""Unit tests for reusable service helpers."""
from datetime import datetime

import pandas as pd

from services import (
    calculate_loss_per_day,
    calculate_reorder_date,
    create_forecast_chart,
    forecast_stock,
    generate_sample_data,
    get_status_color,
    predict_stock_zero_date,
)


def sample_df() -> pd.DataFrame:
    df = generate_sample_data()
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def test_forecast_stock_positive():
    df = sample_df()
    for product in df['Product'].unique():
        rate = forecast_stock(df[df['Product'] == product])
        assert rate >= 0.1


def test_predict_stock_zero_date():
    zero_date, days_until_zero = predict_stock_zero_date(100, 5)
    assert days_until_zero == 20
    assert isinstance(zero_date, datetime)


def test_get_status_color_thresholds():
    assert get_status_color(5)[0].startswith('🔴')
    assert get_status_color(10)[0].startswith('🟡')
    assert get_status_color(20)[0].startswith('🟢')


def test_calculate_reorder_and_loss():
    zero_date = datetime(2024, 1, 31)
    reorder = calculate_reorder_date(zero_date, 7)
    assert reorder.strftime('%Y-%m-%d') == '2024-01-24'
    assert calculate_loss_per_day(12.5, 50) == 625


def test_create_forecast_chart_returns_figure():
    df = sample_df()
    chart = create_forecast_chart(df, df['Product'].iloc[0])
    assert chart is not None
    assert len(chart.data) >= 1
