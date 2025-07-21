import streamlit as st
from streamlit_webrtc import webrtc_streamer
from video_processor import VideoProcessor
from utils import st_success, st_error, st_info, st_warning, get_image_download_button
import os
from datetime import datetime
import logging
import time
import cv2 # Required for camera capture saving

logger = logging.getLogger(__name__)

def local_css():
    """Apply custom CSS for consistent styling."""
    logger.debug("Applying local CSS")
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #2E86AB;
        color: white;
    }
    .st-emotion-cache-18ni7ap, .st-emotion-cache-10trblm {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the app header and a sign-out button if logged in."""
    logger.debug("Displaying header")
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
            <h1 style="color: #2E86AB;">📝 Smart Answer Sheet Scanner</h1>
            <p style="color: #555;">Upload or capture answer sheets to automatically extract register numbers and subject codes.</p>
        """, unsafe_allow_html=True)
    
    if st.session_state.get("logged_in", False):
        with col2:
            st.write("") # Spacer
            st.write("") # Spacer
            if st.button("Sign Out"):
                logger.debug("User signing out")
                # Clear all session state keys to log out
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

def display_signup(mongo_manager, email_service):
    """Display the signup form and handle new user registration."""
    logger.debug("Rendering signup form")
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

            logger.debug(f"Signup attempt: username={username}, email={email}")
            user_data = {"username": username, "email": email, "password": password, "verified": False}
            
            # Step 1: Create user in the database
            if not mongo_manager.create_user(user_data):
                st_error("Username or email already exists. Please try another.")
                return

            # Step 2: Try to send OTP email
            try:
                otp = email_service.generate_otp()
                mongo_manager.save_otp(email, otp)
                email_service.send_otp(email, otp) # Correct call based on services.py
                
                # Set session state to move to OTP verification
                st.session_state.auth_tab = "Sign In"
                st.session_state.signup_email = email
                st.session_state.pending_verification = True
                
                logger.debug(f"Signup successful, OTP sent to {email}")
                st_success("Account created! Check your email for the OTP to verify.")
                time.sleep(2) # Allow user to read the message
                st.rerun()

            except Exception as e:
                logger.error(f"Error sending OTP during signup: {str(e)}")
                # This provides a more specific error message to the user
                st_error("Account created, but we failed to send a verification email. Please check the system's email configuration or try signing in to receive a new OTP.")

def display_signin(mongo_manager, email_service):
    """Display the sign-in form, including the OTP verification step."""
    # State for pending OTP verification
    if st.session_state.get("pending_verification", False):
        email_to_verify = st.session_state.get("signup_email", "")
        logger.debug(f"Displaying OTP verification for email: {email_to_verify}")
        
        with st.form("otp_form"):
            st.subheader("Verify Your Email")
            st.markdown(f"An OTP has been sent to **{email_to_verify}**. Please enter it below.")
            otp = st.text_input("One-Time Password (OTP)", max_chars=6)
            submitted = st.form_submit_button("Verify OTP")

            if submitted:
                if not otp:
                    st_warning("Please enter the OTP.")
                    return
                
                if mongo_manager.verify_otp(email_to_verify, otp):
                    st.session_state.logged_in = True
                    st.session_state.email = email_to_verify
                    st.session_state.pending_verification = False
                    logger.debug("OTP verified, user logged in")
                    st_success("Email verified successfully! You are now logged in.")
                    time.sleep(2)
                    st.rerun()
                else:
                    st_error("Invalid OTP. Please try again.")

    # Standard email/password sign-in
    else:
        with st.form("signin_form"):
            st.subheader("Sign In")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")

            if submitted:
                if not (email and password):
                    st_warning("Please enter both email and password.")
                    return
                
                user = mongo_manager.verify_user(email, password)
                if user:
                    if user.get("verified", False):
                        st.session_state.logged_in = True
                        st.session_state.email = email
                        logger.debug("User signed in successfully")
                        st_success("Successfully signed in!")
                        time.sleep(1)
                        st.rerun()
                    else: # User exists but is not verified
                        st_info("Your account is not verified. Sending a new OTP...")
                        try:
                            otp = email_service.generate_otp()
                            mongo_manager.save_otp(email, otp)
                            email_service.send_otp(email, otp)
                            
                            st.session_state.signup_email = email
                            st.session_state.pending_verification = True
                            logger.debug(f"Unverified user tried to sign in. New OTP sent to {email}")
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Failed to send OTP on sign-in attempt: {e}")
                            st_error("Failed to send verification email. Please check server configuration.")
                else:
                    st_error("Invalid email or password.")

def display_scan_tab(extractor, mongo_manager):
    """Display the main scanning interface with upload and camera options."""
    logger.debug("Rendering scan tab")
    st.subheader("Process a New Answer Sheet")
    
    # Ensure extractor is loaded
    if extractor is None:
        st_error("The Answer Sheet Extractor could not be loaded. Please check the model files and configuration.")
        return

    option = st.radio("Choose input method:", ["Upload an Image", "Use Camera"], horizontal=True, label_visibility="collapsed")
    
    def process_and_display(file_path):
        with st.spinner("Analyzing answer sheet... This may take a moment."):
            try:
                # Correctly call the main processing function and unpack all results
                results, reg_path, sub_path, overlay_path, _, history_item = extractor.process_answer_sheet(file_path)
                
                if not results:
                    st_error("Could not detect any scannable regions. Please try a clearer or better-aligned image.")
                    return

                col1, col2 = st.columns(2)
                with col1:
                    if overlay_path and os.path.exists(overlay_path):
                        st.image(overlay_path, caption="Processed Image with Detections", use_column_width=True)
                with col2:
                    st.subheader("Extraction Results")
                    for label, value in results:
                        st.markdown(f"**{label}:** `{value}`")
                    
                    if reg_path and os.path.exists(reg_path):
                        st.image(reg_path, caption="Cropped Register Number")
                    if sub_path and os.path.exists(sub_path):
                        st.image(sub_path, caption="Cropped Subject Code")
                
                # Save the results to the database using the history item
                mongo_manager.save_scan(st.session_state.email, history_item)
                logger.debug("Scan results saved to database")
                st_success("Analysis complete and results saved to your history.")

            except Exception as e:
                logger.error(f"Error during sheet processing: {str(e)}", exc_info=True)
                st_error(f"A critical error occurred during processing. Please check the logs.")

    if option == "Upload an Image":
        uploaded_file = st.file_uploader("Upload an answer sheet (JPG or PNG)", type=["jpg", "png"])
        if uploaded_file:
            upload_dir = "uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            process_and_display(file_path)

    elif option == "Use Camera":
        webrtc_ctx = webrtc_streamer(key="answer-sheet-scanner", video_processor_factory=VideoProcessor)
        
        if webrtc_ctx.video_processor and st.button("Capture & Process"):
            frame = webrtc_ctx.video_processor.get_frame()
            if frame is not None:
                capture_dir = "captures"
                os.makedirs(capture_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(capture_dir, f"capture_{timestamp}.jpg")
                
                # Save the captured frame as an image file
                cv2.imwrite(file_path, frame)
                process_and_display(file_path)
            else:
                st_warning("No frame captured from the camera. Please try again.")

def display_history_tab(mongo_manager):
    """Display the user's scan history from the database."""
    logger.debug("Rendering history tab")
    st.subheader("Your Scan History")
    scans = mongo_manager.get_user_scans(st.session_state.email)
    
    if not scans:
        st_info("You have no past scans. Use the 'Scan' tab to process a new answer sheet.")
        return

    for i, scan in enumerate(scans):
        # The 'scan' object is now the full history item from the DB
        scan_time = scan.get("history_item", {}).get("timestamp", "Unknown time")
        with st.expander(f"Scan from {scan_time}", expanded=(i == 0)):
            history_item = scan.get("history_item", {})
            overlay_path = history_item.get("overlay_image_path")
            results = history_item.get("results", [])

            col1, col2 = st.columns(2)
            with col1:
                if overlay_path and os.path.exists(overlay_path):
                    st.image(overlay_path, caption="Processed Answer Sheet", use_column_width=True)
                else:
                    st_warning("Processed image file not found.")
            with col2:
                st.markdown("**Extracted Data:**")
                if results:
                    for label, value in results:
                        st.markdown(f"- **{label}:** `{value}`")
                else:
                    st.markdown("No data was extracted for this scan.")
                
                if overlay_path and os.path.exists(overlay_path):
                    get_image_download_button(overlay_path, f"scan_{scan['_id']}.jpg", "Download Image")

def display_about_tab():
    """Display static information about the application."""
    logger.debug("Rendering about tab")
    st.subheader("About the Smart Answer Sheet Scanner")
    st.markdown("""
    This application leverages advanced computer vision to automate the processing of student answer sheets.
    
    **Core Technologies:**
    - **UI Framework:** Streamlit
    - **Object Detection:** YOLOv8 for locating register number and subject code fields.
    - **Text Recognition:** A custom Convolutional Recurrent Neural Network (CRNN) for extracting text.
    - **Backend:** Python, OpenCV, PyTorch
    - **Database:** MongoDB for storing user data and scan history.
    
    **Developed by:** Kavin Raja
    **Version:** 1.2.0
    """)
