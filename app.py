# app.py
import streamlit as st
from session_state import initialize_session_state
from utils import local_css
from ui_components import display_login_page, display_otp_page, display_main_app
from extractor import load_all_models
from config import MONGO_URI, EMAIL_USER

def main():
    # Set page config once at the beginning
    st.set_page_config(
        page_title="Smart Answer Sheet Scanner",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state and apply CSS
    initialize_session_state()
    local_css()

    # --- Pre-computation Check ---
    # Check if secrets are configured before doing anything else
    if not MONGO_URI or not EMAIL_USER:
        st.error("🚨 Application is not configured correctly!")
        st.info("Please ask the application administrator to set up the required secrets (MongoDB URI and Email Credentials) in the Streamlit Cloud settings.")
        st.stop()
    
    # --- Page Routing ---
    if not st.session_state.get('logged_in', False):
        if st.session_state.page == 'login':
            display_login_page()
        elif st.session_state.page == 'otp':
            display_otp_page()
    else:
        # Load models only after the user is logged in
        extractor = load_all_models()
        if extractor:
            display_main_app(extractor)
        else:
            # Error is already shown inside load_all_models
            st.stop()


if __name__ == "__main__":
    main()