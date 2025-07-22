# auth.py
import streamlit as st
import hashlib
from db import get_user_collection
from utils import generate_otp, send_otp_email

def hash_password(password):
    return hashlib.sha256((password + st.secrets["HASH_SECRET_KEY"]).encode()).hexdigest()

def signup(username, email, password):
    users = get_user_collection()
    if users.find_one({"email": email}):
        return "Email already exists."

    otp = generate_otp()
    send_otp_email(email, otp)
    st.session_state.temp_signup = {
        "username": username,
        "email": email,
        "password": hash_password(password),
        "otp": otp
    }
    return "OTP sent to your email."

def verify_otp(entered_otp):
    data = st.session_state.get("temp_signup", {})
    if not data:
        return "Session expired. Please sign up again."
    if data["otp"] == entered_otp:
        users = get_user_collection()
        users.insert_one({
            "username": data["username"],
            "email": data["email"],
            "password": data["password"]
        })
        del st.session_state.temp_signup
        return "Account created!"
    return "Incorrect OTP."

def login(email, password):
    users = get_user_collection()
    user = users.find_one({"email": email})
    if not user or user["password"] != hash_password(password):
        return None
    return user
