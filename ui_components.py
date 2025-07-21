import streamlit as st
from streamlit_webrtc import webrtc_streamer
from video_processor import VideoProcessor
from utils import st_success, st_error, st_info, st_warning, get_image_download_button, save_results_to_file
import os
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

def local_css():
    """Apply custom CSS for consistent styling."""
    logger.debug("Applying local CSS")
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-18ni7ap, .st-emotion-cache-10trblm {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 25px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #2E86AB;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display the app header and a sign-out button."""
    logger.debug("Displaying header")
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        st.markdown("""
        <div style="padding: 1rem; border-radius: 10px; background-color: #ffffff;">
            <h1 style="color: #2E86AB;">📝 Smart Answer Sheet Scanner</h1>
            <p>Upload or capture answer sheets to automatically extract register numbers and subject codes.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.session_state.get("logged_in"):
            if st.button("Sign Out"):
                logger.debug("User signing out")
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

def display_signup(mongo_manager, email_service):
    """Display the signup form and handle new user registration."""
    logger.debug("Rendering signup form")
    with st.form("signup_form"):
        st.subheader("Create a New Account")
        username = st.text_input("Username", key="signup_username")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        submitted = st.form_submit_button("Sign Up")

        if submitted:
            logger.debug(f"Signup attempt: username={username}, email={email}")
            if username and email and password:
                try:
                    user_data = {"username": username, "email": email, "password": password, "verified": False}
                    result = mongo_manager.create_user(user_data)
                    if result:
                        otp = email_service.generate_otp()
                        mongo_manager.save_otp(email, otp)
                        email_service.send_otp(email, otp, username) # Pass username for email template

                        st.session_state.auth_tab = "Sign In"
                        st.session_state.pending_verification = True
                        st.session_state.signup_email = email # Store email to pre-fill on sign-in
                        
                        logger.debug(f"Signup successful, OTP sent to {email}")
                        st_success("Account created! Check your email for the OTP to verify.")
                        time.sleep(2) # Give user time to read the message
                        st.rerun()
                    else:
                        logger.warning("Username or email already exists")
                        st_error("Username or email already exists.")
                except Exception as e:
                    logger.error(f"Error during signup: {str(e)}")
                    st_error(f"An unexpected error occurred during signup.")
            else:
                logger.warning("Incomplete signup form")
                st_warning("Please fill in all fields.")

def display_signin(mongo_manager, email_service):
    """Display the sign-in form, including the OTP verification step."""
    logger.debug("Rendering signin form")
    
    # OTP Verification Stage
    if st.session_state.get("pending_verification", False):
        email_to_verify = st.session_state.get("signup_email", "")
        logger.debug(f"Displaying OTP verification for email: {email_to_verify}")
        with st.form("otp_form"):
            st.subheader("Verify Your Email")
            st.markdown(f"Enter the OTP sent to **{email_to_verify}**")
            otp = st.text_input("Enter OTP", key="otp_input", max_chars=6)
            submitted = st.form_submit_button("Verify OTP")
            
            if submitted:
                if otp:
                    try:
                        verified_user = mongo_manager.verify_otp(email_to_verify, otp)
                        if verified_user:
                            # Send welcome email upon successful verification
                            email_service.send_welcome_email(email_to_verify, verified_user['username'])

                            st.session_state.logged_in = True
                            st.session_state.email = email_to_verify
                            st.session_state.pending_verification = False
                            
                            logger.debug("OTP verified, user logged in")
                            st_success("Email verified successfully! You are now logged in.")
                            time.sleep(2)
                            st.rerun()
                        else:
                            logger.warning("Invalid OTP entered")
                            st_error("Invalid OTP. Please try again or sign up again for a new one.")
                    except Exception as e:
                        logger.error(f"Error verifying OTP: {str(e)}")
                        st_error("An unexpected error occurred during OTP verification.")
                else:
                    st_warning("Please enter the OTP.")

    # Standard Password Sign-in Stage
    else:
        with st.form("signin_form"):
            st.subheader("Sign In")
            email = st.text_input("Email", key="signin_email")
            password = st.text_input("Password", type="password", key="signin_password")
            submitted = st.form_submit_button("Sign In")

            if submitted:
                if email and password:
                    try:
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
                                otp = email_service.generate_otp()
                                mongo_manager.save_otp(email, otp)
                                email_service.send_otp(email, otp, user['username'])
                                
                                st.session_state.signup_email = email
                                st.session_state.pending_verification = True
                                
                                logger.debug(f"Unverified user tried to sign in. New OTP sent to {email}")
                                st_info("Your account is not verified. A new OTP has been sent to your email.")
                                time.sleep(2)
                                st.rerun()
                        else:
                            logger.warning("Invalid email or password")
                            st_error("Invalid email or password.")
                    except Exception as e:
                        logger.error(f"Error during signin: {str(e)}")
                        st_error("An unexpected error occurred during sign-in.")
                else:
                    st_warning("Please enter both email and password.")

def display_scan_tab(extractor, mongo_manager):
    """Display the main scanning interface with upload and camera options."""
    logger.debug("Rendering scan tab")
    st.subheader("Scan a New Answer Sheet")
    option = st.radio("Choose input method:", ["Upload an Image", "Use Camera"], horizontal=True)

    def process_and_display(file_path):
        with st.spinner("Analyzing answer sheet... This may take a moment."):
            try:
                # Correctly call the main processing function from the extractor
                results, reg_path, sub_path, overlay_path, _, history_item = extractor.process_answer_sheet(file_path)
                
                if not results:
                    st_error("Could not detect any regions of interest. Please try a clearer image.")
                    return

                col1, col2 = st.columns(2)
                with col1:
                    st.image(overlay_path, caption="Detected Regions", use_column_width=True)
                with col2:
                    st.subheader("Extraction Results")
                    for label, value in results:
                        st.markdown(f"**{label}:** `{value}`")
                    if reg_path:
                        st.image(reg_path, caption="Cropped Register Number")
                    if sub_path:
                        st.image(sub_path, caption="Cropped Subject Code")
                
                # Save the results to the database
                mongo_manager.save_scan(st.session_state.email, overlay_path, results)
                logger.debug("Scan results saved to database")
                st_success("Analysis complete and results saved to your history.")

            except Exception as e:
                logger.error(f"Error during sheet processing: {str(e)}")
                st_error(f"A critical error occurred during processing: {e}")

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
                with open(file_path, "wb") as f:
                    f.write(cv2.imencode('.jpg', frame)[1].tobytes())
                process_and_display(file_path)
            else:
                st_warning("No frame captured from the camera. Please try again.")

def display_history_tab(mongo_manager):
    """Display the user's scan history."""
    logger.debug("Rendering history tab")
    st.subheader("Your Scan History")
    scans = mongo_manager.get_user_scans(st.session_state.email)
    
    if not scans:
        st_info("You have no past scans. Use the 'Scan' tab to process a new answer sheet.")
        return

    for i, scan in enumerate(scans):
        with st.expander(f"Scan from {scan['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}", expanded=(i == 0)):
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists(scan["image_path"]):
                    st.image(scan["image_path"], caption="Processed Answer Sheet", use_column_width=True)
                else:
                    st_warning("Image file not found.")
            with col2:
                st.markdown("**Extracted Data:**")
                for label, value in scan.get("results", []):
                    st.markdown(f"- **{label}:** `{value}`")
                
                # Provide a download button for the image if it exists
                if os.path.exists(scan["image_path"]):
                    get_image_download_button(scan["image_path"], f"scan_{scan['_id']}.jpg", "Download Image")

def display_about_tab():
    """Display static information about the application."""
    logger.debug("Rendering about tab")
    st.subheader("About the Smart Answer Sheet Scanner")
    st.markdown("""
    This application leverages advanced computer vision to automate the processing of answer sheets.
    
    **Core Technologies:**
    - **UI Framework:** Streamlit
    - **Object Detection:** YOLOv8 for locating register number and subject code fields.
    - **Text Recognition:** A custom Convolutional Recurrent Neural Network (CRNN) for extracting text from the detected regions.
    - **Backend:** Python, OpenCV
    - **Database:** MongoDB for storing user data and scan history.
    
    **Developed by:** Kavin Raja
    **Version:** 1.1.0
    """)
