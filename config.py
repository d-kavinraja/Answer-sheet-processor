import toml
import os
import logging

logger = logging.getLogger(__name__)

def load_secrets():
    """Load secrets from .streamlit/secrets.toml or environment variables."""
    try:
        # Prefer local secrets.toml for development
        if os.path.exists(".streamlit/secrets.toml"):
            secrets = toml.load(".streamlit/secrets.toml")
            logger.debug("Secrets loaded from secrets.toml")
            return secrets
        else:
            # Fallback to environment variables for deployment
            secrets = {
                "MONGO_URI": os.environ.get("MONGO_URI"),
                "SMTP_SERVER": os.environ.get("SMTP_SERVER"),
                "SMTP_PORT": int(os.environ.get("SMTP_PORT", 587)),
                "EMAIL_USER": os.environ.get("EMAIL_USER"),
                "EMAIL_PASSWORD": os.environ.get("EMAIL_PASSWORD")
            }
            logger.debug("Secrets loaded from environment variables")
            # Ensure all required secrets are present
            if not all(secrets.values()):
                raise ValueError("One or more required secrets are missing from environment variables.")
            return secrets
    except Exception as e:
        logger.error(f"Error loading secrets: {e}")
        raise
