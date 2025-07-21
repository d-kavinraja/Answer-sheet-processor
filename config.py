import streamlit as st

def load_secrets():
    """
    Loads secrets from Streamlit's secrets manager.
    Returns a dictionary of secrets or stops the app if critical keys are missing.
    """
    try:
        secrets = {
            "MONGO_URI": st.secrets["MONGO_URI"],
            "SMTP_SERVER": st.secrets["SMTP_SERVER"],
            "SMTP_PORT": st.secrets["SMTP_PORT"],
            "EMAIL_USER": st.secrets["EMAIL_USER"],
            "EMAIL_PASSWORD": st.secrets["EMAIL_PASSWORD"],
        }
        return secrets
    except (FileNotFoundError, KeyError) as e:
        st.error("🚨 Critical Error: API keys or secrets are missing. Please configure your .streamlit/secrets.toml file.")
        st.error(f"Missing configuration: {e}")
        st.stop()