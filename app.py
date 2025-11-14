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
from datetime import datetime, timedelta
import warnings
import smtplib
from email.mime.text import MIMEText
import requests
from io import BytesIO
import os
from dotenv import load_dotenv
import json

from services import (
    load_csv,
    preprocess_data,
    generate_sample_data,
    forecast_stock,
    predict_stock_zero_date,
    get_status_color,
    calculate_reorder_date,
    calculate_loss_per_day,
    create_forecast_chart,
)
from supabase_client import (
    get_supabase_client,
    auth_ui,
    supabase_insert_alerts,
    supabase_log_notification,
    supabase_sync_products_and_sales,
    supabase_storage_upload,
    handle_password_reset,
)

try:
    # Optional PDF support
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False
load_dotenv()  # Load variables from .env if present
warnings.filterwarnings('ignore')

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


def send_push_notification(
    channel: str,
    message: str,
    *,
    twilio_sid: str | None = None,
    twilio_token: str | None = None,
    twilio_from: str | None = None,
    twilio_to: str | None = None,
    slack_webhook: str | None = None,
    telegram_bot_token: str | None = None,
    telegram_chat_id: str | None = None,
) -> bool:
    """Dispatch push notifications to the configured channel."""
    try:
        if channel == 'WhatsApp (Twilio)':
            if not all([twilio_sid, twilio_token, twilio_from, twilio_to]):
                st.error("Provide Twilio SID, token, from, and to numbers.")
                return False
            url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
            data = {
                'From': twilio_from,
                'To': twilio_to,
                'Body': message,
            }
            resp = requests.post(url, data=data, auth=(twilio_sid, twilio_token), timeout=15)
            if resp.status_code >= 400:
                st.error(f"Twilio send failed: {resp.text}")
                return False
            return True

        if channel == 'Slack Webhook':
            if not slack_webhook:
                st.error("Provide Slack webhook URL.")
                return False
            resp = requests.post(slack_webhook, json={'text': message}, timeout=15)
            if resp.status_code >= 400:
                st.error(f"Slack send failed: {resp.text}")
                return False
            return True

        # Telegram
        if not all([telegram_bot_token, telegram_chat_id]):
            st.error("Provide Telegram bot token and chat ID.")
            return False
        url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
        resp = requests.post(url, data={'chat_id': telegram_chat_id, 'text': message}, timeout=15)
        if resp.status_code >= 400:
            st.error(f"Telegram send failed: {resp.text}")
            return False
        return True
    except requests.RequestException as exc:
        st.error(f"Push notification failed: {exc}")
        return False

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
    st.dataframe(styled_df, width="stretch")
    
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

    sb_client = get_supabase_client()

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
                st.plotly_chart(chart, width="stretch")
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
