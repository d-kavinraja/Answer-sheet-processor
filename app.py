import streamlit as st
from config import load_secrets
from database import MongoManager
from services import EmailService
from session_state import initialize_session_state
from ui_components import local_css, display_header, display_signup, display_signin, display_scan_tab, display_history_tab, display_about_tab
from extractor import load_extractor
from streamlit_option_menu import option_menu
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Set page configuration
logger.debug("Setting Streamlit page configuration")
st.set_page_config(
    page_title="Smart Answer Sheet Scanner",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    logger.debug("Entering main() function")
    try:
        # Load secrets
        logger.debug("Loading secrets")
        secrets = load_secrets()
        
        # Initialize MongoDB and EmailService
        logger.debug("Initializing MongoManager and EmailService")
        mongo_manager = MongoManager(secrets["MONGO_URI"])
        email_service = EmailService(
            secrets["SMTP_SERVER"],
            secrets["SMTP_PORT"],
            secrets["EMAIL_USER"],
            secrets["EMAIL_PASSWORD"]
        )
        
        # Initialize session state
        logger.debug("Initializing session state")
        initialize_session_state()
        
        # Apply CSS and display header
        logger.debug("Applying CSS and displaying header")
        local_css()
        display_header()
        
        # Authentication flow
        if not st.session_state.logged_in:
            logger.debug("Displaying authentication tabs")
            auth_tabs = option_menu(
                menu_title=None,
                options=["Sign In", "Sign Up"],
                icons=["box-arrow-in-right", "person-plus"],
                default_index=0 if st.session_state.auth_tab == "Sign In" else 1,
                orientation="horizontal",
                key="auth_menu",
                styles={
                    "container": {"padding": "0!important", "background-color": "#f8f9fa", "border-radius": "8px"},
                    "icon": {"color": "#2E86AB", "font-size": "16px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "center",
                        "margin": "0px",
                        "padding": "10px",
                        "color": "#333"
                    },
                    "nav-link-selected": {"background-color": "#2E86AB", "color": "#fff"}
                }
            )
            st.session_state.auth_tab = auth_tabs
            if auth_tabs == "Sign Up":
                logger.debug("Displaying signup UI")
                display_signup(mongo_manager, email_service)
            else:
                logger.debug("Displaying signin UI")
                display_signin(mongo_manager, email_service)
            return
        
        # Main app tabs for logged-in users
        logger.debug("Displaying main app tabs")
        selected_tab = option_menu(
            menu_title=None,
            options=["Scan", "History", "About"],
            icons=["camera", "clock-history", "info-circle"],
            default_index=0,
            orientation="horizontal",
            key="main_menu",
            styles={
                "container": {"padding": "0!important", "background-color": "#f8f9fa", "border-radius": "8px"},
                "icon": {"color": "#2E86AB", "font-size": "16px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "0px",
                    "padding": "10px",
                    "color": "#333"
                },
                "nav-link-selected": {"background-color": "#2E86AB", "color": "#fff"}
            }
        )
        
        # Load extractor
        logger.debug("Loading extractor")
        extractor = load_extractor()
        
        # Display selected tab
        if selected_tab == "Scan":
            logger.debug("Displaying scan tab")
            display_scan_tab(extractor, mongo_manager)
        elif selected_tab == "History":
            logger.debug("Displaying history tab")
            display_history_tab(mongo_manager)
        else:
            logger.debug("Displaying about tab")
            display_about_tab()
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
