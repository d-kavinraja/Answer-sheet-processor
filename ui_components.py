import streamlit as st
from streamlit_webrtc import webrtc_streamer
from video_processor import VideoProcessor
from utils import st_success, st_error, st_info, st_warning, get_image_download_button, save_results_to_file
import os
from datetime import datetime
import base64

def local_css():
    """Apply custom CSS for consistent styling."""
    st.markdown("""
    <style>
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stTextInput>div>input {
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the app header."""
    st.markdown("""
    <h1 style='text-align: center; color: #2E86AB;'>Smart Answer Sheet Scanner</h1>
    <p style='text-align: center; color: #666;'>Upload or capture answer sheets to extract register numbers and subject codes.</p>
    """, unsafe_allow_html=True)

def display_signup(mongo_manager, email_service):
    """Display the signup form."""
    st.subheader("Create an Account")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        if username and email and password:
            try:
                user_data = {"username": username, "email": email, "password": password, "verified": False}
                result = mongo_manager.create_user(user_data)
                if result:
                    otp = email_service.generate_otp()
                    mongo_manager.save_otp(email, otp)
                    email_service.send_otp(email, otp)
                    st.session_state.auth_tab = "Sign In"
                    st.session_state.signup_email = email
                    st_success("Account created! Check your email for the OTP to verify.")
                else:
                    st_error("Username or email already exists.")
            except Exception as e:
                st_error(f"Error during signup: {str(e)}")
        else:
            st_warning("Please fill in all fields.")

def display_signin(mongo_manager, email_service):
    """Display the signin form."""
    st.subheader("Sign In")
    email = st.text_input("Email", key="signin_email")
    password = st.text_input("Password", type="password", key="signin_password")
    if st.button("Sign In"):
        if email and password:
            user = mongo_manager.verify_user(email, password)
            if user:
                if user.get("verified", False):
                    st.session_state.logged_in = True
                    st.session_state.email = email
                    st_success("Successfully signed in!")
                    st.experimental_rerun()
                else:
                    otp = email_service.generate_otp()
                    mongo_manager.save_otp(email, otp)
                    email_service.send_otp(email, otp)
                    st.session_state.signup_email = email
                    st_info("Please verify your email with the OTP sent.")
            else:
                st_error("Invalid email or password.")
        else:
            st_warning("Please fill in all fields.")

def display_scan_tab(extractor, mongo_manager):
    """Display the scan tab with upload and camera options."""
    st.subheader("Scan Answer Sheet")
    option = st.radio("Choose input method:", ["Upload File", "Use Camera"])
    
    if option == "Upload File":
        uploaded_file = st.file_uploader("Upload an answer sheet (JPG, PNG, or PDF)", type=["jpg", "png", "pdf"])
        if uploaded_file:
            with st.spinner("Processing..."):
                file_path = os.path.join("uploads", uploaded_file.name)
                os.makedirs("uploads", exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    results = extractor.extract(file_path)
                    if results:
                        st.image(file_path, caption="Processed Answer Sheet", use_column_width=True)
                        for label, value in results:
                            st.markdown(f"**{label}:** {value}")
                        results_file = save_results_to_file(results, uploaded_file.name.split('.')[0])
                        get_image_download_button(file_path, uploaded_file.name, "Download Processed Image")
                        if results_file:
                            get_image_download_button(results_file, f"{uploaded_file.name.split('.')[0]}_results.txt", "Download Results")
                        mongo_manager.save_scan(st.session_state.email, file_path, results)
                        st_success("Results extracted and saved!")
                    else:
                        st_error("No text detected in the image.")
                except Exception as e:
                    st_error(f"Error processing file: {str(e)}")

    else:
        webrtc_ctx = webrtc_streamer(
            key="answer-sheet-scanner",
            video_processor_factory=VideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        if webrtc_ctx.video_processor:
            if st.button("Capture and Process"):
                with st.spinner("Processing..."):
                    try:
                        frame = webrtc_ctx.video_processor.get_frame()
                        if frame:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            file_path = f"captures/capture_{timestamp}.jpg"
                            os.makedirs("captures", exist_ok=True)
                            with open(file_path, "wb") as f:
                                f.write(frame)
                            results = extractor.extract(file_path)
                            if results:
                                st.image(file_path, caption="Captured Answer Sheet", use_column_width=True)
                                for label, value in results:
                                    st.markdown(f"**{label}:** {value}")
                                results_file = save_results_to_file(results, f"capture_{timestamp}")
                                get_image_download_button(file_path, f"capture_{timestamp}.jpg", "Download Captured Image")
                                if results_file:
                                    get_image_download_button(results_file, f"capture_{timestamp}_results.txt", "Download Results")
                                mongo_manager.save_scan(st.session_state.email, file_path, results)
                                st_success("Results extracted and saved!")
                            else:
                                st_error("No text detected in the image.")
                    except Exception as e:
                        st_error(f"Error processing capture: {str(e)}")

def display_history_tab(mongo_manager):
    """Display the history tab with past scans."""
    st.subheader("Scan History")
    scans = mongo_manager.get_user_scans(st.session_state.email)
    if scans:
        for scan in scans:
            st.markdown(f"**Date:** {scan['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.image(scan["image_path"], caption="Scanned Answer Sheet", use_column_width=True)
            for label, value in scan["results"]:
                st.markdown(f"**{label}:** {value}")
            get_image_download_button(scan["image_path"], os.path.basename(scan["image_path"]), "Download Image")
            results_file = save_results_to_file(scan["results"], f"history_{scan['_id']}")
            if results_file:
                get_image_download_button(results_file, f"history_{scan['_id']}_results.txt", "Download Results")
            st.markdown("---")
    else:
        st_info("No scan history found.")

def display_about_tab():
    """Display the about tab with app information."""
    st.subheader("About Smart Answer Sheet Scanner")
    st.markdown("""
    This application allows users to scan answer sheets to extract register numbers and subject codes using advanced computer vision techniques.
    
    **Features:**
    - Upload answer sheets in JPG, PNG, or PDF format.
    - Capture answer sheets using a webcam.
    - Extract register numbers and subject codes with YOLO and CRNN models.
    - Save scan history to MongoDB Atlas.
    - User authentication with email verification.
    
    **Developed by:** Your Name
    **Version:** 1.0.0
    """)
