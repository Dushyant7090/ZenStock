"""Supabase helpers for ZenStock Streamlit app."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import time

try:  # pragma: no cover - optional dependency
    from supabase import Client, create_client

    SUPABASE_AVAILABLE = True
except Exception:  # pragma: no cover - handled gracefully at runtime
    Client = None  # type: ignore
    create_client = None  # type: ignore
    SUPABASE_AVAILABLE = False


AUTH_CSS = """
<style>
body, .stApp {
    background-color: #0d0f15 !important;
    color: #e6e6e6 !important;
}
.auth-box {
    padding: 28px;
    background: #161a23;
    border-radius: 14px;
    box-shadow: 0 0 25px rgba(0,0,0,0.45);
    margin-top: 20px;
}
input {
    background-color: #0f1219 !important;
    color: #e6e6e6 !important;
    border-radius: 8px !important;
}
.stButton>button {
    background: linear-gradient(135deg, #3a3f4b, #282c34);
    color: white;
    transition: all 0.25s ease;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #4f5a6e, #313843);
    transform: translateY(-2px);
}
.stTabs [data-baseweb="tab"] {
    color: #9ea3b5 !important;
    font-size: 16px !important;
}
.stTabs [aria-selected="true"] {
    color: white !important;
    border-bottom: 2px solid #4c82ff !important;
}
.success-anim {
    font-size: 23px;
    text-align: center;
    animation: pop 0.45s ease-out forwards;
}
@keyframes pop {
    0% { transform: scale(0.2); opacity: 0; }
    100% { transform: scale(1); opacity: 1; }
}
</style>
"""


__all__ = [
    "SUPABASE_AVAILABLE",
    "get_supabase_client",
    "auth_ui",
    "supabase_insert_alerts",
    "supabase_log_notification",
    "supabase_sync_products_and_sales",
    "supabase_storage_upload",
    "handle_password_reset",
]


def _read_secret(key: str, default: str = "") -> str:
    value = os.environ.get(key, default)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        return default


def _ensure_supabase_import() -> None:
    """Best-effort lazy import so newly installed packages work without restart."""
    global Client, create_client, SUPABASE_AVAILABLE
    if Client is not None and create_client is not None:
        return
    try:
        from supabase import Client as _Client, create_client as _create_client  # type: ignore

        Client = _Client
        create_client = _create_client
        SUPABASE_AVAILABLE = True
    except Exception:
        SUPABASE_AVAILABLE = False
        Client = None
        create_client = None


def get_supabase_client() -> Optional['Client']:
    """Return an initialized Supabase client or None if unavailable."""
    _ensure_supabase_import()
    if not SUPABASE_AVAILABLE:
        st.info("Supabase client not installed. Run: pip install supabase")
        return None

    url = _read_secret("SUPABASE_URL")
    key = _read_secret("SUPABASE_ANON_KEY")
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
                client.auth.set_session(at, rt)
        except Exception:
            pass
        return client
    except Exception as exc:
        st.error(f"Failed to initialize Supabase: {exc}")
        return None


def auth_ui(sb: 'Client') -> bool:
    """Render Supabase auth controls with dark theme. Returns True if user is authenticated."""
    st.markdown(AUTH_CSS, unsafe_allow_html=True)

    if 'sb_user' not in st.session_state:
        st.session_state['sb_user'] = None

    user = st.session_state['sb_user']
    if user:
        with st.sidebar:
            st.markdown(f"🟢 **Logged in as {user.get('email', 'user')}**")
            if st.button("Logout"):
                try:
                    sb.auth.sign_out()
                except Exception:
                    pass
                st.session_state['sb_user'] = None
                st.session_state.pop('sb_access_token', None)
                st.session_state.pop('sb_refresh_token', None)
                st.rerun()
        return True

    st.markdown("<div class='auth-box'>", unsafe_allow_html=True)
    auth_mode_default = st.session_state.get('auth_mode', 'Login')
    auth_mode = st.radio(
        "",
        options=["Login", "Sign Up"],
        index=0 if auth_mode_default != "Sign Up" else 1,
        horizontal=True,
        key="auth_mode_selector",
    )
    st.session_state['auth_mode'] = auth_mode
    heading = "## 🔐 Login to Continue" if auth_mode == "Login" else "## ✨ Create an Account"
    st.markdown(heading)

    if auth_mode == "Login":
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            try:
                res = sb.auth.sign_in_with_password({"email": email, "password": password})
                user = getattr(res, 'user', None)
                session = getattr(res, 'session', None)
                if session:
                    st.session_state['sb_access_token'] = getattr(session, 'access_token', None)
                    st.session_state['sb_refresh_token'] = getattr(session, 'refresh_token', None)
                if user and getattr(user, 'email', None):
                    st.markdown("<div class='success-anim'>🎉 Login Successful!</div>", unsafe_allow_html=True)
                    time.sleep(0.8)
                    st.session_state['sb_user'] = {"email": user.email, "id": getattr(user, 'id', None)}
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
                    st.session_state['show_forgot_btn'] = True
                    st.session_state['show_forgot'] = False
            except Exception as exc:
                st.error(f"Login failed: {exc}")
                st.session_state['show_forgot_btn'] = True
                st.session_state['show_forgot'] = False

        if 'show_forgot' not in st.session_state:
            st.session_state['show_forgot'] = False
        if 'show_forgot_btn' not in st.session_state:
            st.session_state['show_forgot_btn'] = False

        if st.session_state['show_forgot']:
            st.caption("Enter your email to receive a password reset link.")
            fp_email = st.text_input("Reset Email", value=email or "", key="forgot_email")
            redirect_url = _read_secret("SUPABASE_REDIRECT_URL")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Send Reset Link"):
                    if not fp_email:
                        st.warning("Please enter your email.")
                    else:
                        try:
                            options = {"redirect_to": redirect_url} if redirect_url else None
                            if options:
                                sb.auth.reset_password_for_email(fp_email, options=options)
                            else:
                                sb.auth.reset_password_for_email(fp_email)
                            st.success("Reset link sent successfully. Check your email.")
                            st.session_state['show_forgot'] = False
                            st.session_state['show_forgot_btn'] = False
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed to send reset email: {exc}")
            with col2:
                if st.button("Back to Login"):
                    st.session_state['show_forgot'] = False
                    st.session_state['show_forgot_btn'] = False
                    st.rerun()
        elif st.session_state['show_forgot_btn']:
            st.divider()
            if st.button("Forgot password?"):
                st.session_state['show_forgot'] = True
                st.rerun()

    else:
        s_email = st.text_input("Email", key="signup_email")
        s_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Create Account"):
            try:
                res = sb.auth.sign_up({"email": s_email, "password": s_password})
                if getattr(res, 'user', None):
                    st.success("Account created! Check your email to confirm, then login.")
                else:
                    st.info("Signup initiated. Verify via email.")
            except Exception as exc:
                st.error(f"Signup failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)
    return False


def supabase_insert_alerts(sb: 'Client', user_id: str, records: List[Dict[str, Any]]) -> bool:
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
    except Exception as exc:
        st.error(f"Failed to log alerts: {exc}")
        return False


def supabase_log_notification(sb: 'Client', user_id: str, channel: str, payload: Dict[str, Any], status: str = 'sent') -> None:
    try:
        sb.table('notification_logs').insert({
            'user_id': user_id,
            'channel': channel,
            'payload': json.dumps(payload),
            'status': status
        }).execute()
    except Exception as exc:
        st.error(f"Failed to log notification: {exc}")


def supabase_sync_products_and_sales(sb: 'Client', user_id: str, df: pd.DataFrame) -> bool:
    """Upsert products and sales rows tied to the authenticated user."""
    try:
        try:
            _usr = sb.auth.get_user()
            session_uid = getattr(getattr(_usr, 'user', None), 'id', None)
        except Exception:
            session_uid = None
        effective_user_id = session_uid or user_id
        if session_uid and user_id and str(session_uid) != str(user_id):
            st.warning("Supabase session user id differs from cached user id. Using session user id for syncing to satisfy RLS.")

        existing = sb.table('products').select('id,name').eq('user_id', effective_user_id).execute()
        existing_rows = getattr(existing, 'data', existing.data) if hasattr(existing, 'data') else existing
        name_to_id = {row['name']: row['id'] for row in (existing_rows or [])}

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
            for row in (ins_rows or []):
                name_to_id[row['name']] = row['id']

        sales_payload: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            pname = r['Product']
            pid = name_to_id.get(pname)
            if not pid:
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
            chunk = 500
            for i in range(0, len(sales_payload), chunk):
                sb.table('sales').insert(sales_payload[i:i + chunk]).execute()
        return True
    except Exception as exc:
        st.error(f"Failed to sync products/sales: {exc}")
        return False


def supabase_storage_upload(sb: 'Client', bucket: str, path: str, data: bytes, content_type: str) -> str:
    """Upload bytes to Supabase Storage and return public URL if available."""
    try:
        storage = sb.storage.from_(bucket)
        try:
            sb.storage.create_bucket(bucket)
        except Exception:
            pass
        storage.upload(path, data, {
            'contentType': content_type,
            'upsert': True
        })
        res = storage.get_public_url(path)
        if isinstance(res, dict):
            return res.get('publicUrl') or res.get('public_url') or ''
        return getattr(res, 'public_url', '') or getattr(res, 'publicUrl', '')
    except Exception as exc:
        st.error(f"Storage upload failed: {exc}")
        return ''


def handle_password_reset(sb: 'Client') -> bool:
    """Render password reset UI when recovery params are present. Returns True if handled."""
    try:
        params = dict(st.query_params)
    except Exception:
        params = st.experimental_get_query_params()

    def pget(name: str) -> Optional[str]:
        value = params.get(name)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            return value[0]
        return None

    q_type = (pget('type') or '').lower()
    access_token = pget('access_token')
    refresh_token = pget('refresh_token')
    token = pget('token') or pget('token_hash')
    email_param = pget('email')

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
            established_session = False
            if access_token and refresh_token:
                try:
                    sb.auth.set_session(access_token, refresh_token)
                    established_session = True
                    st.session_state['sb_access_token'] = access_token
                    st.session_state['sb_refresh_token'] = refresh_token
                except Exception:
                    pass
            if not established_session and token:
                try:
                    v_email = email_param or st.text_input("Confirm your account email", value="", key="recovery_email_confirm")
                    if not v_email:
                        st.info("Enter your account email to continue password reset.")
                        return True
                    res = sb.auth.verify_otp({
                        "email": v_email,
                        "token": token,
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

            sb.auth.update_user({"password": new_pwd})
            st.success("Password reset successfully. You can now log in with your new password.")
            if st.button("Go to login"):
                try:
                    st.query_params.clear()
                except Exception:
                    pass
                st.rerun()
        except Exception as exc:
            st.error(f"Password reset failed: {exc}")
        return True
    return True
