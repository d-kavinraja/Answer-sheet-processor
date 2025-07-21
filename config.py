# config.py
import streamlit as st
import os

# --- PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "user_data.db")
UPLOADS_DIR = os.path.join(SCRIPT_DIR, "uploads")
CAPTURES_DIR = os.path.join(SCRIPT_DIR, "captures")
CROPPED_REG_DIR = os.path.join(SCRIPT_DIR, "cropped_register_numbers")
CROPPED_SUB_DIR = os.path.join(SCRIPT_DIR, "cropped_subject_codes")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# --- MODEL WEIGHTS ---
YOLO_IMPROVED_PATH = os.path.join(SCRIPT_DIR, "improved_weights.pt")
YOLO_FALLBACK_PATH = os.path.join(SCRIPT_DIR, "weights.pt")
REGISTER_CRNN_PATH = os.path.join(SCRIPT_DIR, "best_crnn_model.pth")
SUBJECT_CRNN_PATH = os.path.join(SCRIPT_DIR, "best_subject_code_model.pth")

# --- EMAIL CREDENTIALS ---
# Load from Streamlit secrets
try:
    EMAIL_USER = st.secrets["email_credentials"]["user"]
    EMAIL_PASSWORD = st.secrets["email_credentials"]["password"]
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
except (KeyError, FileNotFoundError):
    st.error("Email credentials not found in st.secrets. Please configure .streamlit/secrets.toml")
    EMAIL_USER = None
    EMAIL_PASSWORD = None

# --- CONSTANTS ---
OTP_VALIDITY_MINUTES = 10