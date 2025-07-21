import streamlit as st
import logging

logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    logger.debug("Initializing session state variables")
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "email" not in st.session_state:
        st.session_state.email = ""
    if "auth_tab" not in st.session_state:
        st.session_state.auth_tab = "Sign In"
    if "signup_email" not in st.session_state:
        st.session_state.signup_email = ""
    if "pending_verification" not in st.session_state:
        st.session_state.pending_verification = False
