import streamlit as st
import logging

logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize Streamlit session state variables if they don't exist."""
    defaults = {
        "logged_in": False,
        "email": "",
        "auth_tab": "Sign In",
        "signup_email": "",
        "pending_verification": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
