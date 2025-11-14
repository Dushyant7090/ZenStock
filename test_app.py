"""Pytest sanity checks for ZenStock core helpers."""

from datetime import datetime, timedelta

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return pd.read_csv('sales_data.csv')


@pytest.fixture(scope="module")
def df(raw_df: pd.DataFrame) -> pd.DataFrame:
    processed = raw_df.copy()
    processed['Date'] = pd.to_datetime(processed['Date'])
    processed['Quantity Sold'] = pd.to_numeric(processed['Quantity Sold'], errors='coerce')
    processed['Current Stock'] = pd.to_numeric(processed['Current Stock'], errors='coerce')
    return processed.dropna().sort_values('Date')


def test_data_loading(raw_df: pd.DataFrame):
    required_columns = {'Date', 'Product', 'Quantity Sold', 'Current Stock'}
    assert not raw_df.empty
    assert required_columns.issubset(raw_df.columns)
    assert raw_df['Product'].nunique() > 0


def test_data_preprocessing(df: pd.DataFrame):
    assert pd.api.types.is_datetime64_any_dtype(df['Date'])
    assert pd.api.types.is_numeric_dtype(df['Quantity Sold'])
    assert pd.api.types.is_numeric_dtype(df['Current Stock'])
    assert df.isna().sum().sum() == 0


def test_forecasting(df: pd.DataFrame):
    products = df['Product'].unique()
    assert len(products) > 0

    for product in products:
        product_data = df[df['Product'] == product]
        current_stock = product_data['Current Stock'].iloc[-1]
        avg_daily_sales = product_data['Quantity Sold'].mean()
        assert avg_daily_sales >= 0

        if avg_daily_sales > 0:
            days_until_zero = current_stock / avg_daily_sales
            zero_date = datetime.now() + timedelta(days=days_until_zero)
            assert isinstance(zero_date, datetime)
            assert days_until_zero >= 0
