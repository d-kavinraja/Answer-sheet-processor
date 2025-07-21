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
import traceback

# Set up logging with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
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
    
    # Load secrets and initialize services
    logger.debug("Loading secrets")
    try:
        secrets = load_secrets()
        logger.debug(f"Secrets loaded: {list(secrets.keys())}")
    except Exception as e:
        logger.error(f"Error loading secrets: {str(e)}")
        st.error(f"Failed to load secrets: {str(e)}")
        traceback.print_exc()
        st.stop()

    logger.debug("Initializing MongoManager")
    try:
        mongo_manager = MongoManager(secrets["MONGO_URI"])
        logger.debug("MongoManager initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing MongoManager: {str(e)}")
        st.error(f"Database connection failed: {str(e)}")
        traceback.print_exc()
        st.stop()

    logger.debug("Initializing EmailService")
    try:
        email_service = EmailService(
            secrets["SMTP_SERVER"],
            secrets["SMTP_PORT"],
            secrets["EMAIL_USER"],
            secrets["EMAIL_PASSWORD"]
        )
        logger.debug("EmailService initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing EmailService: {str(e)}")
        st.error(f"Email service setup failed: {str(e)}")
        traceback.print_exc()
        st.stop()

    logger.debug("Initializing session state")
    try:
        initialize_session_state()
        logger.debug("Session state initialized")
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        st.error(f"Session state initialization failed: {str(e)}")
        traceback.print_exc()
        st.stop()

    logger.debug("Applying local CSS")
    try:
        local_css()
        logger.debug("Local CSS applied")
    except Exception as e:
        logger.error(f"Error applying local CSS: {str(e)}")
        st.error(f"CSS application failed: {str(e)}")
        traceback.print_exc()
        st.stop()

    logger.debug("Displaying header")
    try:
        display_header()
        logger.debug("Header displayed")
    except Exception as e:
        logger.error(f"Error displaying header: {str(e)}")
        st.error(f"Header rendering failed: {str(e)}")
        traceback.print_exc()
        st.stop()

    if not st.session_state.logged_in:
        logger.debug("User not logged in, displaying auth tabs")
        if 'auth_tab' not in st.session_state:
            st.session_state.auth_tab = "Sign In"
            logger.debug("Set auth_tab to Sign In")
        
        try:
            auth_tabs = option_menu(
                menu_title=None,
                options=["Sign In", "Sign Up"],
                icons=["box-arrow-in-right", "person-plus"],
                default_index=0 if st.session_state.auth_tab == "Sign In" else 1,
                orientation="horizontal",
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
            logger.debug(f"Auth tabs rendered, selected: {auth_tabs}")
        except Exception as e:
            logger.error(f"Error rendering auth tabs: {str(e)}")
            st.error(f"Auth tabs rendering failed: {str(e)}")
            traceback.print_exc()
            st.stop()

        if auth_tabs == "Sign Up":
            logger.debug("Displaying signup UI")
            try:
                display_signup(mongo_manager, email_service)
                logger.debug("Signup UI displayed")
            except Exception as e:
                logger.error(f"Error displaying signup UI: {str(e)}")
                st.error(f"Signup UI rendering failed: {str(e)}")
                traceback.print_exc()
                st.stop()
        else:
            logger.debug("Displaying signin UI")
            try:
                display_signin(mongo_manager, email_service)
                logger.debug("Signin UI displayed")
            except Exception as e:
                logger.error(f"Error displaying signin UI: {str(e)}")
                st.error(f"Signin UI rendering failed: {str(e)}")
                traceback.print_exc()
                st.stop()
        return

    # Load extractor
    logger.debug("Loading extractor")
    try:
        with st.spinner("Loading models..."):
            extractor = load_extractor()
            if not extractor:
                logger.error("Extractor failed to load")
                st.error("Failed to load models. Check model files and try again.")
                st.stop()
            logger.debug("Extractor loaded successfully")
            st.success("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading extractor: {str(e)}")
        st.error(f"Model loading failed: {str(e)}")
        traceback.print_exc()
        st.stop()

    # Main navigation
    logger.debug("Rendering main navigation")
    try:
        selected_tab = option_menu(
            menu_title=None,
            options=["Scan", "History", "About"],
            icons=["camera", "clock-history", "info-circle"],
            default_index=0,
            orientation="horizontal",
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
        logger.debug(f"Main navigation rendered, selected: {selected_tab}")
    except Exception as e:
        logger.error(f"Error rendering main navigation: {str(e)}")
        st.error(f"Navigation rendering failed: {str(e)}")
        traceback.print_exc()
        st.stop()

    if selected_tab == "Scan":
        logger.debug("Displaying scan tab")
        try:
            display_scan_tab(extractor, mongo_manager)
            logger.debug("Scan tab displayed")
        except Exception as e:
            logger.error(f"Error displaying scan tab: {str(e)}")
            st.error(f"Scan tab rendering failed: {str(e)}")
            traceback.print_exc()
            st.stop()
    elif selected_tab == "History":
        logger.debug("Displaying history tab")
        try:
            display_history_tab(mongo_manager)
            logger.debug("History tab displayed")
        except Exception as e:
            logger.error(f"Error displaying history tab: {str(e)}")
            st.error(f"History tab rendering failed: {str(e)}")
            traceback.print_exc()
            st.stop()
    else:
        logger.debug("Displaying about tab")
        try:
            display_about_tab()
            logger.debug("About tab displayed")
        except Exception as e:
            logger.error(f"Error displaying about tab: {str(e)}")
            st.error(f"About tab rendering failed: {str(e)}")
            traceback.print_exc()
            st.stop()

if __name__ == "__main__":
    logger.debug("Starting Smart Answer Sheet Scanner")
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        traceback.print_exc()
        st.stop()