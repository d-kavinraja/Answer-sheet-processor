import streamlit as st
from streamlit_webrtc import webrtc_streamer
from video_processor import VideoProcessor
from utils import st_success, st_error, st_info, st_warning, get_image_download_button
import os
from datetime import datetime
import logging
import time
import cv2

logger = logging.getLogger(__name__)

def local_css():
    """Apply custom CSS for consistent styling."""
    st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #2E86AB; color: white; }
    .st-emotion-cache-18ni7ap, .st-emotion-cache-10trblm { background-color: #ffffff; border-radius: 10px; padding: 25px; }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the app header and a sign-out button if logged in."""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
            <h1 style="color: #2E86AB;">📝 Smart Answer Sheet Scanner</h1>
            <p style="color: #555;">Automated extraction of student and subject information from answer sheets.</p>
        """, unsafe_allow_html=True)
    if st.session_state.get("logged_in", False):
        with col2:
            st.write("")
            st.write("")
            if st.button("Sign Out"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

def display_signup(mongo_manager, email_service):
    """Display the signup form and handle new user registration."""
    with st.form("signup_form"):
        st.subheader("Create a New Account")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign Up")

        if submitted:
            if not (username and email and password):
                st_warning("Please fill in all fields.")
                return
            
            user_data = {"username": username, "email": email, "password": password}
            if not mongo_manager.create_user(user_data):
                st_error("Username or email already exists. Please try another.")
                return
            
            try:
                otp = email_service.generate_otp()
                mongo_manager.save_otp(email, otp)
                email_service.send_otp(email, otp, username)
                
                st.session_state.auth_tab = "Sign In"
                st.session_state.signup_email = email
                st.session_state.pending_verification = True
                
                st_success("Account created! Check your email for the OTP to verify.")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                logger.error(f"Error sending OTP: {e}")
                st_error("Account created, but we failed to send a verification email. Please check the system's email configuration.")

def display_signin(mongo_manager, email_service):
    """Display the sign-in form, including OTP verification."""
    if st.session_state.get("pending_verification", False):
        email = st.session_state.get("signup_email", "")
        with st.form("otp_form"):
            st.subheader("Verify Your Email")
            st.markdown(f"An OTP has been sent to **{email}**. Please enter it below.")
            otp = st.text_input("One-Time Password (OTP)", max_chars=6)
            submitted = st.form_submit_button("Verify OTP")
            if submitted:
                verified_user = mongo_manager.verify_otp(email, otp)
                if verified_user:
                    email_service.send_welcome_email(email, verified_user['username'])
                    st.session_state.logged_in = True
                    st.session_state.email = email
                    st.session_state.pending_verification = False
                    st_success("Email verified successfully! You are now logged in.")
                    time.sleep(2)
                    st.rerun()
                else:
                    st_error("Invalid or expired OTP. Please try again.")
    else:
        with st.form("signin_form"):
            st.subheader("Sign In")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")
            if submitted:
                user = mongo_manager.verify_user(email, password)
                if user:
                    if user.get("verified", False):
                        st.session_state.logged_in = True
                        st.session_state.email = email
                        st_success("Successfully signed in!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st_info("Your account isn't verified. Sending a new OTP...")
                        try:
                            otp = email_service.generate_otp()
                            mongo_manager.save_otp(email, otp)
                            email_service.send_otp(email, otp, user['username'])
                            st.session_state.signup_email = email
                            st.session_state.pending_verification = True
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Failed to send OTP on sign-in: {e}")
                            st_error("Failed to send verification email. Please check server configuration.")
                else:
                    st_error("Invalid email or password.")

def display_scan_tab(extractor, mongo_manager):
    """Display the main scanning interface."""
    st.subheader("Process a New Answer Sheet")
    if extractor is None:
        st_error("Extractor models not loaded. Please check configuration and model files.")
        return

    option = st.radio("Choose input:", ["Upload Image", "Use Camera"], horizontal=True, label_visibility="collapsed")
    
    def process_and_display(file_path):
        with st.spinner("Analyzing answer sheet... This may take a moment."):
            try:
                results, reg_path, sub_path, overlay_path, _, history_item = extractor.process_answer_sheet(file_path)
                if not results:
                    st_error("Could not detect any scannable regions. Please use a clearer image.")
                    return
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(overlay_path, caption="Processed Image with Detections", use_column_width=True)
                with col2:
                    st.subheader("Extraction Results")
                    for label, value in results:
                        st.markdown(f"**{label}:** `{value}`")
                    if reg_path and os.path.exists(reg_path):
                        st.image(reg_path, caption="Cropped Register Number")
                    if sub_path and os.path.exists(sub_path):
                        st.image(sub_path, caption="Cropped Subject Code")

                mongo_manager.save_scan(st.session_state.email, history_item)
                st_success("Analysis complete and results saved to your history.")
            except Exception as e:
                logger.error(f"Error processing sheet: {e}", exc_info=True)
                st_error(f"A critical error occurred during processing: {e}")

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an answer sheet (JPG or PNG)", type=["jpg", "png"])
        if uploaded_file:
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            process_and_display(file_path)
    else: # Use Camera
        webrtc_ctx = webrtc_streamer(key="scanner-camera", video_processor_factory=VideoProcessor)
        if webrtc_ctx.video_processor and st.button("Capture & Process"):
            frame = webrtc_ctx.video_processor.get_frame()
            if frame is not None:
                capture_dir = "captures"
                os.makedirs(capture_dir, exist_ok=True)
                file_path = os.path.join(capture_dir, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(file_path, frame)
                process_and_display(file_path)
            else:
                st_warning("No frame captured. Please try again.")

def display_history_tab(mongo_manager):
    """Display the user's scan history."""
    st.subheader("Your Scan History")
    scans = mongo_manager.get_user_scans(st.session_state.email)
    if not scans:
        st_info("You have no past scans. Use the 'Scan' tab to process a new sheet.")
        return

    for i, scan in enumerate(scans):
        history_item = scan.get("history_item", {})
        scan_time = history_item.get("timestamp", "Unknown time")
        with st.expander(f"Scan from {scan_time}", expanded=(i == 0)):
            overlay_path = history_item.get("overlay_image_path")
            results = history_item.get("results", [])
            col1, col2 = st.columns(2)
            with col1:
                if overlay_path and os.path.exists(overlay_path):
                    st.image(overlay_path, caption="Processed Sheet", use_column_width=True)
                else:
                    st_warning("Processed image file not found.")
            with col2:
                st.markdown("**Extracted Data:**")
                for label, value in results:
                    st.markdown(f"- **{label}:** `{value}`")
                if overlay_path and os.path.exists(overlay_path):
                    get_image_download_button(overlay_path, f"scan_{scan['_id']}.jpg", "Download Image")

def display_about_tab():
    """Display static information about the application."""
    st.subheader("About the Smart Answer Sheet Scanner")
    st.markdown("""
    This application uses computer vision to automate answer sheet processing.
    - **UI Framework:** Streamlit
    - **Object Detection:** YOLOv8
    - **Text Recognition:** CRNN (PyTorch)
    - **Database:** MongoDB
    **Developed by:** Kavin Raja | **Version:** 1.2.0
    """)
