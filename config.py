import toml
import os
import logging

logger = logging.getLogger(__name__)

def load_secrets():
    """Load secrets from .streamlit/secrets.toml or environment variables."""
    logger.debug("Loading secrets")
    try:
        if os.path.exists(".streamlit/secrets.toml"):
            secrets = toml.load(".streamlit/secrets.toml")
            logger.debug("Secrets loaded from secrets.toml")
            return secrets
        else:
            secrets = {
                "MONGO_URI": os.environ.get("MONGO_URI"),
                "SMTP_SERVER": os.environ.get("SMTP_SERVER"),
                "SMTP_PORT": int(os.environ.get("SMTP_PORT", 587)),
                "EMAIL_USER": os.environ.get("EMAIL_USER"),
                "EMAIL_PASSWORD": os.environ.get("EMAIL_PASSWORD")
            }
            logger.debug("Secrets loaded from environment variables")
            return secrets
    except Exception as e:
        logger.error(f"Error loading secrets: {str(e)}")
        raise Exception(f"Failed to load secrets: {str(e)}")
