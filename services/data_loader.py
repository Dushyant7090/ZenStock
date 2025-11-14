"""Data loading utilities for ZenStock."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def load_csv(uploaded_file) -> pd.DataFrame | None:
    """Load and validate CSV file with sales data."""
    try:
        df = pd.read_csv(uploaded_file)
        norm = {c: c.strip().lower().replace(' ', '_') for c in df.columns}
        std_cols = {k: None for k in ('date', 'product', 'quantity_sold', 'current_stock')}
        for orig, normalized in norm.items():
            if normalized == 'date':
                std_cols['date'] = orig
            elif normalized == 'product':
                std_cols['product'] = orig
            elif normalized in ('quantity_sold', 'sales', 'qty_sold', 'quantity'):
                std_cols['quantity_sold'] = orig
            elif normalized in ('current_stock', 'current__stock', 'stock', 'stock_on_hand'):
                std_cols['current_stock'] = orig
        missing = [k for k, v in std_cols.items() if v is None]
        if missing:
            st.error(f"Missing required columns (after normalization): {missing}.")
            return None
        df = df.rename(columns={
            std_cols['date']: 'Date',
            std_cols['product']: 'Product',
            std_cols['quantity_sold']: 'Quantity Sold',
            std_cols['current_stock']: 'Current Stock',
        })
        return df
    except Exception as exc:
        st.error(f"Error loading CSV: {exc}")
        return None


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame | None:
    """Clean and preprocess the sales data."""
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Quantity Sold'] = pd.to_numeric(df['Quantity Sold'], errors='coerce')
        df['Current Stock'] = pd.to_numeric(df['Current Stock'], errors='coerce')
        df = df.dropna().sort_values('Date')
        return df
    except Exception as exc:
        st.error(f"Error preprocessing data: {exc}")
        return None


def generate_sample_data() -> pd.DataFrame:
    """Generate demo sales data."""
    products = ['Widget A', 'Widget B', 'Widget C']
    dates = pd.date_range(start='2024-08-01', end='2024-08-30', freq='D')
    data = []
    current_stocks = {'Widget A': 150, 'Widget B': 75, 'Widget C': 200}
    for date in dates:
        for product in products:
            base_sales = {'Widget A': 8, 'Widget B': 12, 'Widget C': 5}[product]
            weekend_factor = 0.7 if date.weekday() >= 5 else 1.0
            random_factor = np.random.normal(1.0, 0.3)
            quantity_sold = max(0, int(base_sales * weekend_factor * random_factor))
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Product': product,
                'Quantity Sold': quantity_sold,
                'Current Stock': current_stocks[product]
            })
    return pd.DataFrame(data)
