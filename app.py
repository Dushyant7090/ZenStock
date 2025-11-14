"""
ZenStock - Smart Inventory Management Tool
A hackathon-winning AI-powered inventory management system that predicts stock depletion 
and provides intelligent restock recommendations.

Author: AI Assistant
Created for: Hackathon Project
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import smtplib
from email.mime.text import MIMEText
import requests
from io import BytesIO
import os
from dotenv import load_dotenv
import json
try:
    # Optional PDF support
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False
load_dotenv()  # Load variables from .env if present
warnings.filterwarnings('ignore')

# Try to import Prophet, fallback to simple forecasting if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet not installed. Using simple moving average for forecasting.")

# Try to import Supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="ZenStock - Smart Inventory Management",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .status-red {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .status-yellow {
        background-color: #fff8e1;
        color: #f57f17;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .status-green {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Move Supabase recovery tokens from URL hash to query string so Streamlit can read them
def ingest_hash_into_query():
    components.html(
        """
        <script>
        (function() {
          try {
            // Try to operate on the outermost accessible window
            var ctx = window.top || window.parent || window;
            var url = new URL(ctx.location.href);
            if (url.searchParams.get('hash_parsed') === '1') { return; }
            var hash = ctx.location.hash || '';
            if (hash && hash.length > 1) {
              var hp = new URLSearchParams(hash.substring(1));
              if (hp.has('type') || hp.has('access_token') || hp.has('refresh_token') || hp.has('token') || hp.has('token_hash')) {
                hp.forEach((v, k) => url.searchParams.set(k, v));
                url.searchParams.set('hash_parsed', '1');
                try { ctx.location.hash = ''; } catch(e) {}
                // Attempt multiple navigation strategies for reliability
                try { ctx.location.replace(url.toString()); } catch(e) {}
                try { ctx.history.replaceState(null, '', url.toString()); ctx.location.reload(); } catch(e) {}
              }
            }
          } catch (e) { /* ignore */ }
        })();
        </script>
        """,
        height=1,
    )


def load_csv(uploaded_file):
    """
    Load and validate CSV file with sales data
    Accepts either original or MVP naming and normalizes to:
      Date, Product, Quantity Sold, Current Stock
    """
    try:
        df = pd.read_csv(uploaded_file)

        # Normalize column names: lower, strip, replace spaces/underscores
        col_map = {c: c for c in df.columns}
        norm = {c: c.strip().lower().replace(' ', '_') for c in df.columns}

        # Determine mappings to standard names
        std_cols = {
            'date': None,
            'product': None,
            'quantity_sold': None,
            'current_stock': None,
        }
        for orig, n in norm.items():
            if n in ('date',):
                std_cols['date'] = orig
            elif n in ('product',):
                std_cols['product'] = orig
            elif n in ('quantity_sold', 'sales', 'qty_sold', 'quantity'):
                std_cols['quantity_sold'] = orig
            elif n in ('current_stock', 'current__stock', 'stock', 'stock_on_hand'):
                std_cols['current_stock'] = orig

        missing = [k for k, v in std_cols.items() if v is None]
        if missing:
            st.error(f"Missing required columns (after normalization): {missing}. Expected one of the aliases for each.")
            return None

        # Rename to standard headers used across the app
        df = df.rename(columns={
            std_cols['date']: 'Date',
            std_cols['product']: 'Product',
            std_cols['quantity_sold']: 'Quantity Sold',
            std_cols['current_stock']: 'Current Stock',
        })

        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def preprocess_data(df):
    """
    Clean and preprocess the sales data
    """
    try:
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure numeric columns are properly typed
        df['Quantity Sold'] = pd.to_numeric(df['Quantity Sold'], errors='coerce')
        df['Current Stock'] = pd.to_numeric(df['Current Stock'], errors='coerce')
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Sort by date
        df = df.sort_values('Date')
        
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

def calculate_daily_sales(df):
    """
    Calculate average daily sales for each product
    """
    daily_sales = df.groupby(['Product', 'Date'])['Quantity Sold'].sum().reset_index()
    avg_daily_sales = daily_sales.groupby('Product')['Quantity Sold'].mean().reset_index()
    avg_daily_sales.columns = ['Product', 'Avg Daily Sales']
    
    return avg_daily_sales

def forecast_stock_prophet(product_data, days_ahead=30):
    """
    Use Prophet to forecast stock depletion for a specific product
    """
    if not PROPHET_AVAILABLE or len(product_data) < 10:
        return forecast_stock_simple(product_data, days_ahead)
    
    try:
        # Prepare data for Prophet
        prophet_data = product_data.groupby('Date')['Quantity Sold'].sum().reset_index()
        prophet_data.columns = ['ds', 'y']
        
        # Create and fit the model
        model = Prophet(daily_seasonality=True, yearly_seasonality=False)
        model.fit(prophet_data)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        
        # Get the predicted daily sales
        future_sales = forecast[forecast['ds'] > prophet_data['ds'].max()]['yhat'].values
        avg_predicted_sales = max(future_sales.mean(), 0.1)  # Ensure positive
        
        return avg_predicted_sales
    except:
        return forecast_stock_simple(product_data, days_ahead)

def forecast_stock_simple(product_data, days_ahead=30):
    """
    Simple moving average forecasting as fallback
    """
    recent_sales = product_data['Quantity Sold'].tail(7).mean()
    return max(recent_sales, 0.1)  # Ensure positive

def forecast_stock(product_data, days_ahead=30):
    """
    API-aligned wrapper to estimate daily sales rate for a product.
    Chooses Prophet when available and data is sufficient; otherwise falls back to simple average.
    Returns a positive daily sales rate (float).
    """
    if PROPHET_AVAILABLE and len(product_data) >= 10:
        return forecast_stock_prophet(product_data, days_ahead)
    return forecast_stock_simple(product_data, days_ahead)

def predict_stock_zero_date(current_stock, daily_sales_rate):
    """
    Predict when stock will reach zero
    """
    if daily_sales_rate <= 0:
        return None
    
    days_until_zero = current_stock / daily_sales_rate
    zero_date = datetime.now() + timedelta(days=days_until_zero)
    
    return zero_date, days_until_zero

def get_status_color(days_until_zero):
    """
    Determine status color based on days until stock zero
    """
    if days_until_zero < 7:
        return "🔴 Critical", "status-red"
    elif days_until_zero < 14:
        return "🟡 Warning", "status-yellow"
    else:
        return "🟢 Safe", "status-green"

def calculate_reorder_date(stock_zero_date: datetime, lead_time_days: int) -> datetime:
    """
    Reorder Date = Stock-Out Date - Lead Time days
    """
    try:
        return stock_zero_date - timedelta(days=lead_time_days)
    except Exception:
        return None

def calculate_loss_per_day(avg_daily_sales: float, profit_per_unit: float) -> float:
    """
    Potential revenue loss per day if stockout happens.
    """
    return max(0.0, avg_daily_sales) * max(0.0, profit_per_unit)

def demand_simulator(multiplier: float) -> float:
    """
    Simple demand simulator API. Stores the multiplier in session state and returns it.
    The rest of the app reads this multiplier to scale forecasts and update visuals.
    """
    st.session_state['demand_multiplier'] = multiplier
    return multiplier

def send_email(smtp_server: str, smtp_port: int, username: str, password: str,
               to_email: str, subject: str, body: str, use_tls: bool = True) -> bool:
    """
    Send an email using basic SMTP. Returns True on success, False otherwise.
    """
    try:
        msg = MIMEText(body, 'plain')
        msg['Subject'] = subject
        msg['From'] = username
        msg['To'] = to_email

        server = smtplib.SMTP(smtp_server, smtp_port, timeout=15)
        if use_tls:
            server.starttls()
        if username:
            server.login(username, password)
        server.sendmail(username, [to_email], msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email send failed: {e}")
        return False

def create_forecast_chart(df, product_name, demand_multiplier=1.0):
    """
    Create a forecast visualization chart for a specific product
    """
    product_data = df[df['Product'] == product_name].copy()
    
    if len(product_data) == 0:
        return None
    
    # Historical data
    historical = product_data.groupby('Date')['Quantity Sold'].sum().reset_index()
    
    # Forecast future sales
    if PROPHET_AVAILABLE and len(historical) >= 10:
        try:
            prophet_data = historical.copy()
            prophet_data.columns = ['ds', 'y']
            
            model = Prophet(daily_seasonality=True, yearly_seasonality=False)
            model.fit(prophet_data)
            
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # Apply demand multiplier to forecast
            forecast.loc[forecast['ds'] > prophet_data['ds'].max(), 'yhat'] *= demand_multiplier
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical['Date'],
                y=historical['Quantity Sold'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='blue')
            ))
            
            # Forecast
            future_forecast = forecast[forecast['ds'] > historical['Date'].max()]
            fig.add_trace(go.Scatter(
                x=future_forecast['ds'],
                y=future_forecast['yhat'],
                mode='lines',
                name=f'Forecast (Demand x{demand_multiplier})',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'Sales Forecast for {product_name}',
                xaxis_title='Date',
                yaxis_title='Quantity Sold',
                hovermode='x unified'
            )
            
            return fig
        except Exception:
            # Fall back to simple method below
            pass

    # Simple forecast fallback
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical['Date'],
        y=historical['Quantity Sold'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='blue')
    ))
    
    # Simple trend line
    avg_sales = historical['Quantity Sold'].mean() * demand_multiplier
    future_dates = pd.date_range(start=historical['Date'].max() + timedelta(days=1), periods=30)
    future_sales = [avg_sales] * 30
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_sales,
        mode='lines',
        name=f'Simple Forecast (Demand x{demand_multiplier})',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'Sales Forecast for {product_name}',
        xaxis_title='Date',
        yaxis_title='Quantity Sold',
        hovermode='x unified'
    )
    
    return fig

def generate_purchase_order(dashboard_data: pd.DataFrame,
                            supplier_name: str,
                            days_coverage: int = 30):
    """
    Create a purchase order DataFrame with columns:
      Product, Qty to Order, Supplier, Reorder Date
    Qty to Order ~= Avg Daily Sales * days_coverage, rounded up
    Returns (csv_bytes, optional_pdf_bytes)
    """
    po_rows = []
    for _, row in dashboard_data.iterrows():
        avg_sales = float(row['Avg Daily Sales']) if isinstance(row['Avg Daily Sales'], str) else row['Avg Daily Sales']
        qty = int(max(0, np.ceil(avg_sales * days_coverage)))
        po_rows.append({
            'Product': row['Product'],
            'Qty to Order': qty,
            'Supplier': supplier_name,
            'Reorder Date': row.get('Reorder By', ''),
        })
    po_df = pd.DataFrame(po_rows)

    # CSV bytes
    csv_bytes = po_df.to_csv(index=False).encode('utf-8')

    pdf_bytes = None
    if REPORTLAB_AVAILABLE:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 50
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "ZenStock - Purchase Order")
        y -= 20
        c.setFont("Helvetica", 10)
        c.drawString(50, y, f"Supplier: {supplier_name}")
        y -= 20
        headers = ["Product", "Qty to Order", "Supplier", "Reorder Date"]
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y, "  |  ".join(headers))
        y -= 15
        c.setFont("Helvetica", 10)
        for _, r in po_df.iterrows():
            line = f"{r['Product']}  |  {r['Qty to Order']}  |  {r['Supplier']}  |  {r['Reorder Date']}"
            c.drawString(50, y, line[:100])
            y -= 15
            if y < 50:
                c.showPage()
                y = height - 50
        c.save()
        pdf_bytes = buffer.getvalue()
        buffer.close()

    return csv_bytes, pdf_bytes

def generate_sample_data():
    """
    Generate sample sales data for demo purposes
    """
    products = ['Widget A', 'Widget B', 'Widget C']
    dates = pd.date_range(start='2024-08-01', end='2024-08-30', freq='D')
    
    data = []
    current_stocks = {'Widget A': 150, 'Widget B': 75, 'Widget C': 200}
    
    for date in dates:
        for product in products:
            # Simulate realistic sales patterns
            base_sales = {'Widget A': 8, 'Widget B': 12, 'Widget C': 5}[product]
            
            # Add some randomness and weekly patterns
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

def display_dashboard(df, demand_multiplier=1.0, lead_time_days: int = 7, profit_per_unit: float = 50.0):
    """
    Display the main dashboard with product status
    """
    # Calculate metrics for each product
    products = df['Product'].unique()
    dashboard_data = []
    
    for product in products:
        product_data = df[df['Product'] == product]
        current_stock = product_data['Current Stock'].iloc[-1]
        
        # Calculate average daily sales via unified forecast_stock API
        daily_sales = forecast_stock(product_data) * demand_multiplier

        # Predict stock zero date
        zero_date, days_until_zero = predict_stock_zero_date(current_stock, daily_sales)
        status, status_class = get_status_color(days_until_zero)

        # Reorder Date
        reorder_date = calculate_reorder_date(zero_date, lead_time_days) if zero_date else None

        # Revenue impact (loss per day)
        loss_per_day = calculate_loss_per_day(daily_sales, profit_per_unit)

        dashboard_data.append({
            'Product': product,
            'Current Stock': int(current_stock),
            'Avg Daily Sales': f"{daily_sales:.1f}",
            'Days Until Zero': f"{days_until_zero:.0f}",
            'Stock Zero Date': zero_date.strftime('%Y-%m-%d') if zero_date else 'N/A',
            'Reorder By': reorder_date.strftime('%Y-%m-%d') if reorder_date else 'N/A',
            'Status': status,
            '💰 Loss/Day': f"₹{loss_per_day:,.0f}"
        })
    
    # Create dashboard DataFrame
    dashboard_df = pd.DataFrame(dashboard_data)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        critical_products = len([d for d in dashboard_data if '🔴' in d['Status']])
        st.metric("Critical Products", critical_products, delta=None)
    
    with col2:
        warning_products = len([d for d in dashboard_data if '🟡' in d['Status']])
        st.metric("Warning Products", warning_products, delta=None)
    
    with col3:
        safe_products = len([d for d in dashboard_data if '🟢' in d['Status']])
        st.metric("Safe Products", safe_products, delta=None)
    
    with col4:
        total_products = len(products)
        st.metric("Total Products", total_products, delta=None)
    
    # Display the dashboard table
    st.subheader("📊 Product Status Dashboard")
    
    # Style the dataframe
    def style_status(val):
        if '🔴' in val:
            return 'background-color: #ffebee; color: #c62828; font-weight: bold'
        elif '🟡' in val:
            return 'background-color: #fff8e1; color: #f57f17; font-weight: bold'
        elif '🟢' in val:
            return 'background-color: #e8f5e8; color: #2e7d32; font-weight: bold'
        return ''
    
    # Reorder columns for the required view
    view_cols = ['Product', 'Current Stock', 'Avg Daily Sales', 'Days Until Zero', 'Stock Zero Date', 'Reorder By', 'Status', '💰 Loss/Day']
    existing_cols = [c for c in view_cols if c in dashboard_df.columns]
    dashboard_df = dashboard_df[existing_cols]

    styled_df = dashboard_df.style.applymap(style_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)
    
    return dashboard_df

def main():
    """
    Main application function
    """
    # Ingest Supabase recovery tokens from URL hash into query string ASAP
    ingest_hash_into_query()

    

    # Header
    st.markdown("""
    <h1 style="
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 2.5em;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        background: linear-gradient(90deg, #2563eb, #1f77b4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    ">
        📦 ZenStock
    </h1>
    <p style="
        text-align: center;
        font-size: 1.1em;
        color: #4b5563;
        margin-top: 0;
    ">
        Smart Inventory Management Tool, Powered by AI
    </p>
""", unsafe_allow_html=True)

    
    # Sidebar
    st.sidebar.title("🔧 Controls")

    # --- Auth: Supabase Login/Signup Gate ---
    st.sidebar.subheader("👤 Account")

    def get_supabase_client():
        # Prefer environment variables (from .env), then optionally st.secrets
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_ANON_KEY", "")
        try:
            if not url:
                url = st.secrets["SUPABASE_URL"]  # may raise if secrets.toml missing
            if not key:
                key = st.secrets["SUPABASE_ANON_KEY"]
        except Exception:
            # No secrets.toml configured; rely on env only
            pass
        if not SUPABASE_AVAILABLE:
            st.info("Supabase client not installed. Run: pip install supabase")
            return None
        if not url or not key:
            st.warning("Supabase not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in .env or Streamlit secrets.")
            return None
        try:
            client: Client = create_client(url, key)
            # Restore session across reruns if tokens are cached
            try:
                at = st.session_state.get('sb_access_token')
                rt = st.session_state.get('sb_refresh_token')
                if at and rt:
                    # supabase-py v2 supports set_session(access_token, refresh_token)
                    client.auth.set_session(at, rt)
            except Exception:
                pass
            return client
        except Exception as e:
            st.error(f"Failed to initialize Supabase: {e}")
            return None

    def auth_ui(sb: 'Client'):
        if 'sb_user' not in st.session_state:
            st.session_state['sb_user'] = None

        if st.session_state['sb_user']:
            st.sidebar.markdown(f"🟢 Logged in as {st.session_state['sb_user'].get('email','user')}")
            if st.sidebar.button("Logout"):
                try:
                    sb.auth.sign_out()
                except Exception:
                    pass
                # Clear cached tokens on logout
                st.session_state['sb_user'] = None
                st.session_state.pop('sb_access_token', None)
                st.session_state.pop('sb_refresh_token', None)
                st.rerun()
            return True

        st.markdown("### 🔐 Login to continue")
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
        with tab_login:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login"):
                try:
                    res = sb.auth.sign_in_with_password({"email": email, "password": password})
                    user = getattr(res, 'user', None)
                    # Attempt to read tokens from res.session (supabase-py v2)
                    try:
                        session = getattr(res, 'session', None)
                        if session:
                            at = getattr(session, 'access_token', None)
                            rt = getattr(session, 'refresh_token', None)
                            if at and rt:
                                st.session_state['sb_access_token'] = at
                                st.session_state['sb_refresh_token'] = rt
                    except Exception:
                        pass
                    if user and getattr(user, 'email', None):
                        # Cache email and user id for Supabase writes
                        st.session_state['sb_user'] = {"email": user.email, "id": getattr(user, 'id', None)}
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
                        # On failed login, reveal the 'Forgot password?' button (not form yet)
                        st.session_state['show_forgot_btn'] = True
                        st.session_state['show_forgot'] = False
                except Exception as e:
                    st.error(f"Login failed: {e}")
                    # On error, reveal the 'Forgot password?' button (not form yet)
                    st.session_state['show_forgot_btn'] = True
                    st.session_state['show_forgot'] = False
            # Forgot password flow (hidden by default; only appears after failed login)
            if 'show_forgot' not in st.session_state:
                st.session_state['show_forgot'] = False
            if 'show_forgot_btn' not in st.session_state:
                st.session_state['show_forgot_btn'] = False

            if st.session_state['show_forgot']:
                st.caption("Forgot your password? Send a reset link to your email.")
                fp_email = st.text_input("Reset Email", value=email or "", key="forgot_email")
                redirect_url = os.environ.get("SUPABASE_REDIRECT_URL", "")
                if not redirect_url:
                    try:
                        redirect_url = st.secrets.get("SUPABASE_REDIRECT_URL", "")
                    except Exception:
                        redirect_url = ""
                cols = st.columns([1,1])
                with cols[0]:
                    if st.button("Send reset link"):
                        if not fp_email:
                            st.warning("Please enter your email to receive the reset link.")
                        else:
                            try:
                                # supabase-py v2 method name: reset_password_for_email
                                # Provide redirect_to if configured in Supabase Auth settings
                                if redirect_url:
                                    sb.auth.reset_password_for_email(fp_email, options={"redirect_to": redirect_url})
                                else:
                                    sb.auth.reset_password_for_email(fp_email)
                                st.success("Reset link sent successfully, check your email")
                                st.session_state['show_forgot'] = False
                                st.session_state['show_forgot_btn'] = False
                                st.rerun()
                            except Exception as e1:
                                # Fallback to older naming if library version differs
                                try:
                                    sb.auth.reset_password_email(fp_email)
                                    st.success("Reset link sent successfully, check your email")
                                    st.session_state['show_forgot'] = False
                                    st.session_state['show_forgot_btn'] = False
                                    st.rerun()
                                except Exception as e2:
                                    st.error(f"Failed to send reset email: {e1 or e2}")
                with cols[1]:
                    if st.button("Back to login"):
                        st.session_state['show_forgot'] = False
                        # Hide the button again unless you want it to persist
                        st.session_state['show_forgot_btn'] = False
                        st.rerun()
            elif st.session_state['show_forgot_btn']:
                st.divider()
                if st.button("Forgot password?"):
                    st.session_state['show_forgot'] = True
                    st.rerun()
        with tab_signup:
            s_email = st.text_input("Email", key="signup_email")
            s_password = st.text_input("Password", type="password", key="signup_password")
            if st.button("Create Account"):
                try:
                    res = sb.auth.sign_up({"email": s_email, "password": s_password})
                    if getattr(res, 'user', None):
                        st.success("Account created! Check your email to confirm if required, then login.")
                    else:
                        st.info("Signup initiated. Check your email to confirm.")
                except Exception as e:
                    st.error(f"Signup failed: {e}")
        return False

    def supabase_insert_alerts(sb: 'Client', user_id: str, records: list):
        try:
            if not records:
                return True
            payload = [{
                'user_id': user_id,
                'level': r.get('level'),
                'message': r.get('message'),
                'stock_zero_date': r.get('stock_zero_date'),
                'reorder_by': r.get('reorder_by'),
                'loss_per_day': r.get('loss_per_day')
            } for r in records]
            sb.table('alerts').insert(payload).execute()
            return True
        except Exception as e:
            st.error(f"Failed to log alerts: {e}")
            return False

    def supabase_log_notification(sb: 'Client', user_id: str, channel: str, payload: dict, status: str = 'sent'):
        try:
            sb.table('notification_logs').insert({
                'user_id': user_id,
                'channel': channel,
                'payload': json.dumps(payload),
                'status': status
            }).execute()
        except Exception as e:
            st.error(f"Failed to log notification: {e}")

    def supabase_sync_products_and_sales(sb: 'Client', user_id: str, df: pd.DataFrame):
        """
        Upsert products for this user (by name) and insert sales rows referencing product IDs.
        Assumes df has columns: Date, Product, Quantity Sold, Current Stock.
        """
        try:
            # Derive the effective user id from the authenticated session to satisfy RLS.
            # Fall back to the provided user_id if session lookup is unavailable.
            try:
                _usr = sb.auth.get_user()
                session_uid = getattr(getattr(_usr, 'user', None), 'id', None)
            except Exception:
                session_uid = None
            effective_user_id = session_uid or user_id
            if session_uid and user_id and str(session_uid) != str(user_id):
                st.warning("Supabase session user id differs from cached user id. Using session user id for syncing to satisfy RLS.")

            # Get existing products for user
            existing = sb.table('products').select('id,name').eq('user_id', effective_user_id).execute()
            existing_rows = getattr(existing, 'data', existing.data) if hasattr(existing, 'data') else existing
            name_to_id = {row['name']: row['id'] for row in (existing_rows or [])}

            # Determine missing products
            products = sorted(df['Product'].unique().tolist())
            to_insert = []
            for name in products:
                if name not in name_to_id:
                    current_stock = int(df[df['Product'] == name]['Current Stock'].iloc[-1])
                    to_insert.append({
                        'user_id': effective_user_id,
                        'name': name,
                        'current_stock': current_stock,
                        'profit_per_unit': 0,
                        'supplier': None
                    })
            if to_insert:
                ins = sb.table('products').insert(to_insert).execute()
                ins_rows = getattr(ins, 'data', ins.data) if hasattr(ins, 'data') else ins
                for r in (ins_rows or []):
                    name_to_id[r['name']] = r['id']

            # Insert sales
            sales_payload = []
            for _, r in df.iterrows():
                pname = r['Product']
                pid = name_to_id.get(pname)
                if not pid:
                    # In case product just created but not returned, fetch again
                    ref = sb.table('products').select('id').eq('user_id', effective_user_id).eq('name', pname).single().execute()
                    pid = (getattr(ref, 'data', ref.data) or {}).get('id') if hasattr(ref, 'data') else ref.get('id')
                    if pid:
                        name_to_id[pname] = pid
                if pid:
                    sales_payload.append({
                        'user_id': effective_user_id,
                        'product_id': pid,
                        'date': pd.to_datetime(r['Date']).date().isoformat(),
                        'quantity_sold': float(r['Quantity Sold'])
                    })
            if sales_payload:
                # Insert in chunks to avoid payload limits
                chunk = 500
                for i in range(0, len(sales_payload), chunk):
                    sb.table('sales').insert(sales_payload[i:i+chunk]).execute()
            return True
        except Exception as e:
            st.error(f"Failed to sync products/sales: {e}")
            return False

    def supabase_storage_upload(sb: 'Client', bucket: str, path: str, data: bytes, content_type: str) -> str:
        """Upload bytes to Supabase Storage and return public URL if available."""
        try:
            storage = sb.storage.from_(bucket)
            # Ensure bucket exists (ignore errors if already exists)
            try:
                sb.storage.create_bucket(bucket)
            except Exception:
                pass
            # Upload with upsert
            storage.upload(path, data, {
                'contentType': content_type,
                'upsert': True
            })
            # Get public URL (if bucket is public). You may need to set the bucket to public in Supabase UI.
            res = storage.get_public_url(path)
            if isinstance(res, dict):
                return res.get('publicUrl') or res.get('public_url') or ''
            # Some client versions return an object with .public_url / .publicUrl
            return getattr(res, 'public_url', '') or getattr(res, 'publicUrl', '')
        except Exception as e:
            st.error(f"Storage upload failed: {e}")
            return ''

    sb_client = get_supabase_client()

    # Password Reset Page: triggered by Supabase recovery redirect with tokens in URL
    def handle_password_reset(sb: 'Client') -> bool:
        """Render password reset UI when recovery params are present. Returns True if handled (page rendered)."""
        # Read query params compatibly across Streamlit versions
        try:
            params = dict(st.query_params)
        except Exception:
            params = st.experimental_get_query_params()

        # Normalize param accessor
        def pget(name: str):
            v = params.get(name)
            return v if isinstance(v, str) else (v[0] if isinstance(v, list) and v else None)

        q_type = (pget('type') or '').lower()
        access_token = pget('access_token')
        refresh_token = pget('refresh_token')
        token = pget('token') or pget('token_hash')  # some templates use token_hash
        email_param = pget('email')  # may be present in some templates

        # Decide whether to show reset UI: if explicit recovery type OR any recovery token is present
        if q_type != 'recovery' and not (access_token or token):
            return False

        st.markdown('<h2 style="text-align:center;">Reset your password</h2>', unsafe_allow_html=True)
        st.write("Create a new password for your account.")

        with st.form("password_reset_form"):
            new_pwd = st.text_input("New Password", type="password")
            confirm_pwd = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Set new password")
        if submitted:
            if not new_pwd or not confirm_pwd:
                st.warning("Please enter and confirm your new password.")
                return True
            if new_pwd != confirm_pwd:
                st.error("Passwords do not match.")
                return True
            try:
                # Establish a valid session using either access_token/refresh_token
                # or by verifying the recovery OTP token (token/token_hash + email).
                established_session = False
                if access_token and refresh_token:
                    try:
                        sb.auth.set_session(access_token, refresh_token)
                        established_session = True
                        # Cache tokens for subsequent requests
                        st.session_state['sb_access_token'] = access_token
                        st.session_state['sb_refresh_token'] = refresh_token
                    except Exception:
                        pass
                if not established_session and token:
                    try:
                        # Attempt verification with available email; if email absent, prompt the user once
                        v_email = email_param or st.text_input("Confirm your account email", value="", key="recovery_email_confirm")
                        if not v_email:
                            st.info("Enter your account email to continue password reset.")
                            return True
                        res = sb.auth.verify_otp({
                            "email": v_email,
                            "token": token,
                            "type": "recovery",
                        })
                        # Try to persist tokens if returned
                        sess = getattr(res, 'session', None)
                        if sess:
                            at = getattr(sess, 'access_token', None)
                            rt = getattr(sess, 'refresh_token', None)
                            if at and rt:
                                try:
                                    sb.auth.set_session(at, rt)
                                except Exception:
                                    pass
                                st.session_state['sb_access_token'] = at
                                st.session_state['sb_refresh_token'] = rt
                            established_session = True
                    except Exception:
                        # Some SDKs may require token_hash field name
                        try:
                            v_email = email_param or st.text_input("Confirm your account email", value="", key="recovery_email_confirm_alt")
                            if not v_email:
                                st.info("Enter your account email to continue password reset.")
                                return True
                            res = sb.auth.verify_otp({
                                "email": v_email,
                                "token_hash": token,
                                "type": "recovery",
                            })
                            sess = getattr(res, 'session', None)
                            if sess:
                                at = getattr(sess, 'access_token', None)
                                rt = getattr(sess, 'refresh_token', None)
                                if at and rt:
                                    try:
                                        sb.auth.set_session(at, rt)
                                    except Exception:
                                        pass
                                    st.session_state['sb_access_token'] = at
                                    st.session_state['sb_refresh_token'] = rt
                                established_session = True
                        except Exception:
                            pass

                # Update the password for the authenticated (recovery) user
                sb.auth.update_user({"password": new_pwd})
                st.success("Password reset successfully. You can now log in with your new password.")
                if st.button("Go to login"):
                    # Best-effort clear of the recovery view
                    try:
                        # Streamlit >= 1.30 supports assignment
                        st.query_params.clear()
                    except Exception:
                        pass
                    st.rerun()
            except Exception as e:
                st.error(f"Password reset failed: {e}")
            return True
        return True

    # If recovery page is triggered, render it and exit early (no auth gate required)
    if sb_client is not None and handle_password_reset(sb_client):
        return

    if sb_client is None:
        st.info("Proceeding without authentication (dev mode). Set Supabase secrets to enable login.")
    else:
        authed = auth_ui(sb_client)
        if not authed:
            return
    
    # File upload
    st.sidebar.subheader("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Sales Data CSV",
        type=['csv'],
        help="CSV should contain: Date, Product, Quantity Sold, Current Stock"
    )
    
    # Load data
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if df is not None:
            df = preprocess_data(df)
    else:
        # Use sample data
        st.sidebar.info("💡 Using sample data. Upload your CSV to analyze real data.")
        df = generate_sample_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check your CSV format.")
        return

    # Sync uploaded data to Supabase (products + sales)
    sync_to_supabase = False
    if sb_client and st.session_state.get('sb_user', {}).get('id'):
        st.sidebar.subheader("🗄️ Data Persistence")
        sync_to_supabase = st.sidebar.checkbox("Sync uploads to Supabase (products + sales)", value=(uploaded_file is not None))
        if sync_to_supabase and uploaded_file is not None:
            ok_sync = supabase_sync_products_and_sales(sb_client, st.session_state['sb_user']['id'], df)
            if ok_sync:
                st.sidebar.success("Data synced to Supabase.")
            else:
                st.sidebar.error("Failed to sync data to Supabase.")
    
    # Lead time and profit per unit
    st.sidebar.subheader("⏱️ Lead Time & 💹 Profit")
    lead_time_days = st.sidebar.number_input("Supplier Lead Time (days)", min_value=0, max_value=90, value=7, step=1)
    profit_per_unit = st.sidebar.number_input("Profit per Unit (₹)", min_value=0.0, max_value=1_000_000.0, value=50.0, step=10.0)

    # What-If Demand Simulator
    st.sidebar.subheader("🎯 What-If Demand Simulator")
    increase_pct = st.sidebar.slider("Simulate Sales Increase (%)", min_value=0, max_value=200, value=0, step=5,
                                     help="0% = normal, 50% = 1.5x, 100% = 2x")
    demand_multiplier = 1.0 + (increase_pct / 100.0)
    
    if demand_multiplier != 1.0:
        st.sidebar.success(f"📈 Simulating {(demand_multiplier-1)*100:+.0f}% demand change")
    
    # Persist simulator setting (API-aligned function call)
    demand_simulator(demand_multiplier)

    # Automatic Email Notifications (no UI input)
    # SMTP settings are read from environment/secrets. Alerts will be emailed to the logged-in user's email once per session.
    def get_smtp_settings():
        server = os.environ.get("SMTP_SERVER", "")
        port = os.environ.get("SMTP_PORT", "")
        username = os.environ.get("SMTP_USERNAME", "")
        password = os.environ.get("SMTP_PASSWORD", "")
        from_email = os.environ.get("SMTP_FROM", "")
        # Try st.secrets if any are missing
        try:
            if not server:
                server = st.secrets["SMTP_SERVER"]
            if not port:
                port = st.secrets["SMTP_PORT"]
            if not username:
                username = st.secrets["SMTP_USERNAME"]
            if not password:
                password = st.secrets["SMTP_PASSWORD"]
            if not from_email:
                from_email = st.secrets.get("SMTP_FROM", "")
        except Exception:
            pass
        try:
            port = int(port) if str(port).strip() else 587
        except Exception:
            port = 587
        if not from_email:
            from_email = username
        return {
            'server': server,
            'port': port,
            'username': username,
            'password': password,
            'from_email': from_email,
        }
    
    # Main dashboard
    dashboard_df = display_dashboard(df, demand_multiplier, lead_time_days, profit_per_unit)
    
    # Forecast visualization section
    st.subheader("📈 Forecast Visualization")
    
    # Product selection for detailed view
    selected_product = st.selectbox(
        "Select Product for Detailed Forecast",
        df['Product'].unique()
    )
    
    if selected_product:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create and display forecast chart
            chart = create_forecast_chart(df, selected_product, demand_multiplier)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning("Unable to generate forecast chart for this product.")
        
        with col2:
            # Product details
            product_info = dashboard_df[dashboard_df['Product'] == selected_product].to_dict('records')
            product_info = product_info[0] if product_info else None
            if product_info:
                st.markdown("### Product Details")
                st.markdown(f"**Current Stock:** {product_info['Current Stock']}")
                st.markdown(f"**Avg Daily Sales:** {product_info['Avg Daily Sales']}")
                st.markdown(f"**Days Until Zero:** {product_info['Days Until Zero']}")
                st.markdown(f"**Stock Zero Date:** {product_info['Stock Zero Date']}")
                st.markdown(f"**Reorder By:** {product_info['Reorder By']}")
                st.markdown(f"**Status:** {product_info['Status']}")
                st.markdown(f"**Loss/Day:** {product_info['💰 Loss/Day']}")
    
    # Restock Reminders
    st.subheader("🔔 Restock Reminders")
    
    critical_products = [item for item in dashboard_df.to_dict('records') if '🔴' in item['Status']]
    warning_products = [item for item in dashboard_df.to_dict('records') if '🟡' in item['Status']]
    
    if critical_products:
        st.error("🚨 **URGENT: Critical Stock Levels**")
        for product in critical_products:
            st.error(f"• **{product['Product']}**: Only {product['Days Until Zero']} days left! Stock out by {product['Stock Zero Date']} | Reorder by {product['Reorder By']} | Loss/Day: {product['💰 Loss/Day']}")
    
    if warning_products:
        st.warning("⚠️ **Warning: Low Stock Levels**")
        for product in warning_products:
            st.warning(f"• **{product['Product']}**: {product['Days Until Zero']} days remaining. Stock out by {product['Stock Zero Date']} | Reorder by {product['Reorder By']} | Loss/Day: {product['💰 Loss/Day']}")
    
    if not critical_products and not warning_products:
        st.success("✅ All products have sufficient stock levels!")

    # Auto-send email alerts once per session when critical/warning items are detected
    alerts_present = bool(critical_products or warning_products)
    if alerts_present and sb_client and st.session_state.get('sb_user', {}).get('email'):
        if not st.session_state.get('auto_email_sent', False):
            user_email = st.session_state['sb_user']['email']
            smtp = get_smtp_settings()
            if all([smtp['server'], smtp['port'], smtp['username'], smtp['password'], user_email]):
                lines = []
                for p in critical_products:
                    lines.append(f"CRITICAL - {p['Product']}: {p['Days Until Zero']} days left (Zero by {p['Stock Zero Date']}) | Reorder by {p['Reorder By']} | Loss/Day {p['💰 Loss/Day']}")
                for p in warning_products:
                    lines.append(f"WARNING - {p['Product']}: {p['Days Until Zero']} days left (Zero by {p['Stock Zero Date']}) | Reorder by {p['Reorder By']} | Loss/Day {p['💰 Loss/Day']}")
                subject = "ZenStock Alerts: Inventory Notifications"
                body = "\n".join(lines)
                ok = send_email(smtp['server'], int(smtp['port']), smtp['username'], smtp['password'],
                                user_email, subject, body, use_tls=True)
                if ok:
                    st.session_state['auto_email_sent'] = True
                    st.info("📧 Restock alerts emailed to your account (auto).")
                    # Log notification to Supabase
                    if sb_client and st.session_state.get('sb_user', {}).get('id'):
                        supabase_log_notification(sb_client, st.session_state['sb_user']['id'], 'email', {
                            'subject': subject,
                            'body': body,
                            'to': user_email,
                            'auto': True
                        }, status='sent')
                else:
                    if sb_client and st.session_state.get('sb_user', {}).get('id'):
                        supabase_log_notification(sb_client, st.session_state['sb_user']['id'], 'email', {
                            'subject': subject,
                            'body': body,
                            'to': user_email,
                            'auto': True
                        }, status='failed')
            else:
                # Show a gentle one-time hint if SMTP is not configured
                if not st.session_state.get('smtp_missing_notice_shown', False):
                    st.session_state['smtp_missing_notice_shown'] = True
                    st.info("To enable auto-email alerts, set SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD (and optional SMTP_FROM) in your environment or Streamlit secrets.")

    # Push Notifications
    st.subheader("📲 Push Notifications")
    with st.expander("Enable Push Notifications"):
        channel = st.selectbox("Channel", ["WhatsApp (Twilio)", "Slack Webhook", "Telegram Bot"]) 
        msg_preview = "\n".join([
            (f"⚠️ Restock Alert: {p['Product']} will be out in {p['Days Until Zero']} days. "
             f"Order by {p['Reorder By']}. Loss/day: {p['💰 Loss/Day']}.") for p in (critical_products + warning_products)
        ]) or "No current alerts."
        st.text_area("Message Preview", value=msg_preview, height=150)

        colA, colB = st.columns(2)
        with colA:
            if channel == 'WhatsApp (Twilio)':
                tw_sid = st.text_input("Twilio Account SID", value="")
                tw_token = st.text_input("Twilio Auth Token", value="", type="password")
                tw_from = st.text_input("From (whatsapp:+1415..)", value="")
                tw_to = st.text_input("To (whatsapp:+91..)", value="")
            elif channel == 'Slack Webhook':
                slack_hook = st.text_input("Slack Webhook URL", value="")
            else:
                tg_token = st.text_input("Telegram Bot Token", value="")
                tg_chat = st.text_input("Telegram Chat ID", value="")
        with colB:
            send_push = st.button("Send Push Now")

        if 'send_push' in locals() and send_push:
            if channel == 'WhatsApp (Twilio)':
                ok = send_push_notification(channel, msg_preview, twilio_sid=tw_sid, twilio_token=tw_token,
                                            twilio_from=tw_from, twilio_to=tw_to)
            elif channel == 'Slack Webhook':
                ok = send_push_notification(channel, msg_preview, slack_webhook=slack_hook)
            else:
                ok = send_push_notification(channel, msg_preview, telegram_bot_token=tg_token, telegram_chat_id=tg_chat)
            if ok:
                st.success("Push notification sent!")
                if sb_client and st.session_state.get('sb_user', {}).get('id'):
                    supabase_log_notification(sb_client, st.session_state['sb_user']['id'],
                                              'whatsapp' if channel.startswith('WhatsApp') else ('slack' if 'Slack' in channel else 'telegram'),
                                              {'message': msg_preview}, status='sent')
            else:
                st.error("Failed to send push notification.")
                if sb_client and st.session_state.get('sb_user', {}).get('id'):
                    supabase_log_notification(sb_client, st.session_state['sb_user']['id'],
                                              'whatsapp' if channel.startswith('WhatsApp') else ('slack' if 'Slack' in channel else 'telegram'),
                                              {'message': msg_preview}, status='failed')

    # Manual save of current alerts into Supabase (without sending)
    if sb_client and st.session_state.get('sb_user', {}).get('id'):
        if st.button("💾 Save Alerts to Supabase"):
            records = []
            for p in critical_products:
                records.append({
                    'level': 'critical',
                    'message': f"{p['Product']} critical: {p['Days Until Zero']} days left",
                    'stock_zero_date': p['Stock Zero Date'],
                    'reorder_by': p['Reorder By'],
                    'loss_per_day': p['💰 Loss/Day'].replace('₹','').replace(',','')
                })
            for p in warning_products:
                records.append({
                    'level': 'warning',
                    'message': f"{p['Product']} warning: {p['Days Until Zero']} days left",
                    'stock_zero_date': p['Stock Zero Date'],
                    'reorder_by': p['Reorder By'],
                    'loss_per_day': p['💰 Loss/Day'].replace('₹','').replace(',','')
                })
            ok_alerts = supabase_insert_alerts(sb_client, st.session_state['sb_user']['id'], records)
            if ok_alerts:
                st.success("Alerts saved to Supabase.")

    # Purchase Order Generation
    st.subheader("📦 Generate Purchase Order")
    supplier_name = st.text_input("Supplier Name", value="Acme Supplies")
    days_coverage = st.number_input("Days of Coverage", min_value=7, max_value=120, value=30, step=1)
    if st.button("Generate Purchase Order"):
        csv_bytes, pdf_bytes = generate_purchase_order(dashboard_df, supplier_name, int(days_coverage))
        st.download_button("Download PO (CSV)", data=csv_bytes, file_name="purchase_order.csv", mime="text/csv")
        if REPORTLAB_AVAILABLE and pdf_bytes:
            st.download_button("Download PO (PDF)", data=pdf_bytes, file_name="purchase_order.pdf", mime="application/pdf")

        # Upload to Supabase Storage and insert purchase_orders
        if sb_client and st.session_state.get('sb_user', {}).get('id'):
            user_id = st.session_state['sb_user']['id']
            bucket = 'zenstock'
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_path = f"{user_id}/purchase_orders/po_{ts}"
            csv_path = base_path + ".csv"
            pdf_path = base_path + ".pdf"
            csv_url = supabase_storage_upload(sb_client, bucket, csv_path, csv_bytes, 'text/csv') if csv_bytes else ''
            pdf_url = ''
            if REPORTLAB_AVAILABLE and pdf_bytes:
                pdf_url = supabase_storage_upload(sb_client, bucket, pdf_path, pdf_bytes, 'application/pdf')
            # Insert record
            try:
                sb_client.table('purchase_orders').insert({
                    'user_id': user_id,
                    'supplier': supplier_name,
                    'coverage_days': int(days_coverage),
                    'csv_url': csv_url,
                    'pdf_url': pdf_url
                }).execute()
                st.success("Purchase order saved to Supabase.")
            except Exception as e:
                st.error(f"Failed to save purchase order to Supabase: {e}")
    
    # Footer removed for cleaner UI

if __name__ == "__main__":
    main()
