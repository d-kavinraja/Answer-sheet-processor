import streamlit as st
from config import load_secrets
from database import MongoManager
from services import EmailService
from session_state import initialize_session_state
from ui_components import (local_css, display_header, display_signup, 
                           display_signin, display_scan_tab, display_history_tab, 
                           display_about_tab)
from extractor import load_extractor
from streamlit_option_menu import option_menu
import logging
import sys

# Set up logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Smart Answer Sheet Scanner",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    """Main function to run the Streamlit application."""
    logger.debug("Entering main() function")
    try:
        # Load secrets from config
        secrets = load_secrets()

        # Initialize core services
        mongo_manager = MongoManager(secrets["MONGO_URI"])
        email_service = EmailService(
            secrets["SMTP_SERVER"],
            secrets["SMTP_PORT"],
            secrets["EMAIL_USER"],
            secrets["EMAIL_PASSWORD"]
        )

        # Initialize session state
        initialize_session_state()

        # Apply custom CSS and display header
        local_css()
        display_header()

        # Authentication Flow: Show sign-in/sign-up if not logged in
        if not st.session_state.get("logged_in", False):
            auth_tabs = option_menu(
                menu_title=None,
                options=["Sign In", "Sign Up"],
                icons=["box-arrow-in-right", "person-plus"],
                default_index=0 if st.session_state.auth_tab == "Sign In" else 1,
                orientation="horizontal",
                styles={
                    "container": {"padding": "0!important", "background-color": "#f8f9fa", "border-radius": "8px"},
                    "icon": {"color": "#2E86AB", "font-size": "16px"},
                    "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "color": "#333"},
                    "nav-link-selected": {"background-color": "#2E86AB", "color": "#fff"}
                }
            )
            st.session_state.auth_tab = auth_tabs

            if auth_tabs == "Sign Up":
                display_signup(mongo_manager, email_service)
            else:
                display_signin(mongo_manager, email_service)
            return

        # Main App for Logged-in Users
        selected_tab = option_menu(
            menu_title=None,
            options=["Scan", "History", "About"],
            icons=["camera", "clock-history", "info-circle"],
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#f8f9fa", "border-radius": "8px"},
                "icon": {"color": "#2E86AB", "font-size": "16px"},
                "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "color": "#333"},
                "nav-link-selected": {"background-color": "#2E86AB", "color": "#fff"}
            }
        )

        # Lazily load the ML model extractor only when needed
        extractor = load_extractor()

        # Display content based on the selected tab
        if selected_tab == "Scan":
            display_scan_tab(extractor, mongo_manager)
        elif selected_tab == "History":
            display_history_tab(mongo_manager)
        else:
            display_about_tab()

    except Exception as e:
        logger.error(f"Critical error in main app execution: {str(e)}", exc_info=True)
        st.error(f"A critical error occurred. Please check the logs or contact support.")

if __name__ == "__main__":
    main()
