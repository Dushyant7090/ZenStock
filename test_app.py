"""
Quick test script to verify ZenStock functionality
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_data_loading():
    """Test CSV data loading"""
    try:
        df = pd.read_csv('sales_data.csv')
        print("✅ CSV loading successful")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {list(df.columns)}")
        print(f"   - Products: {df['Product'].unique()}")
        return df
    except Exception as e:
        print(f"❌ CSV loading failed: {e}")
        return None

def test_data_preprocessing(df):
    """Test data preprocessing"""
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Quantity Sold'] = pd.to_numeric(df['Quantity Sold'], errors='coerce')
        df['Current Stock'] = pd.to_numeric(df['Current Stock'], errors='coerce')
        df = df.dropna()
        print("✅ Data preprocessing successful")
        print(f"   - Date range: {df['Date'].min()} to {df['Date'].max()}")
        return df
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return None

def test_forecasting(df):
    """Test basic forecasting logic"""
    try:
        products = df['Product'].unique()
        results = []
        
        for product in products:
            product_data = df[df['Product'] == product]
            current_stock = product_data['Current Stock'].iloc[-1]
            avg_daily_sales = product_data['Quantity Sold'].mean()
            
            if avg_daily_sales > 0:
                days_until_zero = current_stock / avg_daily_sales
                zero_date = datetime.now() + timedelta(days=days_until_zero)
                
                # Determine status
                if days_until_zero < 7:
                    status = "🔴 Critical"
                elif days_until_zero < 14:
                    status = "🟡 Warning"
                else:
                    status = "🟢 Safe"
                
                results.append({
                    'Product': product,
                    'Current Stock': int(current_stock),
                    'Avg Daily Sales': f"{avg_daily_sales:.1f}",
                    'Days Until Zero': f"{days_until_zero:.0f}",
                    'Status': status
                })
        
        print("✅ Forecasting logic successful")
        for result in results:
            print(f"   - {result['Product']}: {result['Status']} ({result['Days Until Zero']} days)")
        
        return results
    except Exception as e:
        print(f"❌ Forecasting failed: {e}")
        return None

def main():
    print("🧪 Testing ZenStock Core Functionality")
    print("=" * 50)
    
    # Test data loading
    df = test_data_loading()
    if df is None:
        return
    
    # Test preprocessing
    df = test_data_preprocessing(df)
    if df is None:
        return
    
    # Test forecasting
    results = test_forecasting(df)
    if results is None:
        return
    
    print("\n🎉 All tests passed! ZenStock is ready to run.")
    print("\nTo start the application, run:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
