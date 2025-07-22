# utils.py
import smtplib
import random
from email.mime.text import MIMEText
import streamlit as st

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_email(recipient_email, otp):
    subject = "Smart Scanner - OTP Verification"
    body = f"Your OTP is: {otp}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = st.secrets["EMAIL_USER"]
    msg["To"] = recipient_email

    with smtplib.SMTP(st.secrets["SMTP_SERVER"], st.secrets["SMTP_PORT"]) as server:
        server.starttls()
        server.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASSWORD"])
        server.send_message(msg)
