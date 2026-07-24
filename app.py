"""
Student Stress Monitor — Streamlit entry point.
Run: streamlit run app.py

All actual logic lives in src/. This file only wires together:
  landing page -> auth page -> main app
"""

import streamlit as st

from src.db.schema import init_all
from src.ui.styles import inject_global_css
from src.ui.landing_page import show_landing_page
from src.ui.auth_page import show_auth_page
from src.ui.main_app import show_main_app

st.set_page_config(
    page_title="Student Stress Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()
init_all()

if "show_landing" not in st.session_state:
    st.session_state["show_landing"] = True

if st.session_state.get("show_landing", True):
    show_landing_page()
elif "user" not in st.session_state or st.session_state["user"] is None:
    show_auth_page()
else:
    show_main_app(st.session_state["user"])
