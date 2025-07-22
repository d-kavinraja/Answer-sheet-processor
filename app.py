# app.py
import streamlit as st
from auth import signup, login, verify_otp

st.set_page_config(page_title="Smart Answer Sheet Scanner", layout="wide")

def main():
    if "authenticated_user" not in st.session_state:
        st.title("🔐 Welcome to Smart Scanner")

        tab = st.radio("Choose Action", ["Sign In", "Sign Up"], horizontal=True)

        if tab == "Sign Up":
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Send OTP"):
                msg = signup(username, email, password)
                st.info(msg)

            if "temp_signup" in st.session_state:
                entered_otp = st.text_input("Enter OTP to verify")
                if st.button("Verify OTP"):
                    result = verify_otp(entered_otp)
                    st.success(result)
                    if result.startswith("Account created"):
                        st.session_state.authenticated_user = username
                        st.rerun()

        elif tab == "Sign In":
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                user = login(email, password)
                if user:
                    st.success("Login successful!")
                    st.session_state.authenticated_user = user["username"]
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        st.stop()

    # 🔓 Authenticated user
    st.sidebar.success(f"Logged in as: {st.session_state.authenticated_user}")
    st.sidebar.button("Logout", on_click=lambda: st.session_state.pop("authenticated_user"))

    # 🔁 Now load your main app
    from answer_sheet_streamlit import main as scanner_main
    scanner_main()

if __name__ == "__main__":
    main()
