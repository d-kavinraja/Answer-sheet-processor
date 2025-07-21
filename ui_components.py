# ui_components.py
import streamlit as st
import os
import io
import uuid
import time
from datetime import datetime
import cv2
import pypdfium2 as pdfium
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison

import database as db
import services
from models import verify_password
from video_processor import VideoProcessor
from utils import st_success, st_error, st_info, st_warning, get_image_download_button, save_results_to_file, fallback_extract_text
from config import UPLOADS_DIR, CAPTURES_DIR, RESULTS_DIR

# --- AUTHENTICATION UI ---

def display_login_page():
    st.markdown('<div class="header-container"><h1>Welcome to the Smart Scanner</h1></div>', unsafe_allow_html=True)
    
    login_tab, signup_tab = st.tabs(["🔒 Sign In", "✍️ Sign Up"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign In")

            if submitted:
                user = db.find_user_by_email(email)
                if user and verify_password(user['salt'], user['password_hash'], password):
                    if user['is_verified']:
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.rerun()
                    else:
                        st_warning("Account not verified. Please check your email for an OTP or sign up again to receive a new one.")
                        otp = services.generate_otp()
                        db.update_otp_for_user(email, otp)
                        services.send_verification_email(email, otp)
                        st.session_state.user_email = email
                        st.session_state.page = 'otp'
                        st.rerun()
                else:
                    st_error("Invalid email or password.")
    
    with signup_tab:
        with st.form("signup_form"):
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm_password")
            submitted = st.form_submit_button("Sign Up")

            if submitted:
                if new_password != confirm_password:
                    st_error("Passwords do not match.")
                elif db.find_user_by_email(new_email):
                    st_error("An account with this email already exists.")
                else:
                    otp = services.generate_otp()
                    if db.add_user(new_email, new_password, otp):
                        success, msg = services.send_verification_email(new_email, otp)
                        if success:
                            st.session_state.user_email = new_email
                            st.session_state.page = 'otp'
                            st.rerun()
                        else:
                            st_error(f"Could not send verification email. {msg}")
                    else:
                        st_error("Failed to create account. The email might already be in use.")

def display_otp_page():
    st.info(f"An OTP has been sent to **{st.session_state.user_email}**. Please enter it below.")
    with st.form("otp_form"):
        otp_code = st.text_input("Enter OTP")
        submitted = st.form_submit_button("Verify")
        if submitted:
            if db.verify_user_otp(st.session_state.user_email, otp_code):
                st_success("Verification successful! You are now logged in.")
                st.session_state.logged_in = True
                st.session_state.page = 'main_app'
                time.sleep(2)
                st.rerun()
            else:
                st_error("Invalid or expired OTP. Please try again.")

# --- MAIN APPLICATION UI ---

def display_main_app(extractor):
    with st.sidebar:
        st.info(f"Logged in as:\n**{st.session_state.user_email}**")
        if st.button("Log Out"):
            # Clear session state on logout
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    display_header()
    
    selected_tab = option_menu(
        menu_title=None, options=["Scan", "History", "About"],
        icons=["camera", "clock-history", "info-circle"],
        default_index=0, orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "var(--secondary-background-color)", "border-radius": "10px"},
            "icon": {"color": "var(--primary-color)", "font-size": "16px"},
            "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "padding": "10px", "color": "var(--text-color)"},
            "nav-link-selected": {"background-color": "var(--primary-color)", "color": "white"}
        }
    )

    if selected_tab == "Scan":
        display_scan_tab(extractor)
    elif selected_tab == "History":
        display_history_tab()
    elif selected_tab == "About":
        display_about_tab()
    
    display_footer()

def display_header():
    with st.container():
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown('<div style="font-size: 60px; text-align: center;">📝</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<h1>Smart Answer Sheet Scanner</h1>', unsafe_allow_html=True)
            st.markdown('<p>Automatically extract register numbers and subject codes</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_footer():
    st.markdown('<div class="footer"><p>© 2025 Smart Scanner Project. Built with Streamlit.</p></div>', unsafe_allow_html=True)

def display_scan_tab(extractor):
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.markdown("<h3>Choose input method:</h3>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("⬆️ Upload", key="upload_image_btn"):
            st.session_state.input_method = "Upload Image"
            st.rerun()
    with col2:
        if st.button("📸 Camera", key="use_camera_btn"):
            st.session_state.input_method = "Use Camera"
            st.rerun()
    with col3:
        if st.button("🔄 Reset", key="reset_btn_scan"):
            st.session_state.image_path = None
            st.session_state.image_captured = False
            st.session_state.selected_history_item_index = None
            st.session_state.webrtc_key = f"webrtc_{uuid.uuid4().hex}"
            st.rerun()

    if st.session_state.input_method == "Upload Image":
        handle_upload()
    else:
        handle_camera()

    if st.session_state.image_path and st.session_state.image_captured and st.session_state.selected_history_item_index is None:
        if st.button("🔍 Extract Information", key="extract_btn", type="primary"):
            run_extraction(extractor)
    st.markdown('</div>', unsafe_allow_html=True)

def handle_upload():
    uploaded_file = st.file_uploader(
        "Upload Answer Sheet Image or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        temp_path = os.path.join(UPLOADS_DIR, f"upload_{uuid.uuid4().hex}.jpg")
        
        try:
            if file_extension == 'pdf':
                pdf_file = io.BytesIO(uploaded_file.getvalue())
                pdf = pdfium.PdfDocument(pdf_file)
                page = pdf[0]
                pil_image = page.render(scale=300/72).to_pil()
                pil_image.save(temp_path, 'JPEG')
            else:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            st.session_state.image_path = temp_path
            st.session_state.image_captured = True
            st.session_state.selected_history_item_index = None
            st.rerun()
        except Exception as e:
            st_error(f"Error processing file: {e}")
    
    if st.session_state.image_path and st.session_state.image_captured:
        st.image(st.session_state.image_path, caption="Uploaded/Converted Image", use_container_width=True)
    else:
        st.info("Drag and drop or click to upload your file.")


def handle_camera():
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    if not st.session_state.image_captured:
        ctx = webrtc_streamer(
            key=st.session_state.webrtc_key, mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=VideoProcessor, async_processing=True
        )
        if ctx.state.playing and st.button("📸 Capture Image"):
            if ctx.video_processor and ctx.video_processor.frame is not None:
                os.makedirs(CAPTURES_DIR, exist_ok=True)
                temp_path = os.path.join(CAPTURES_DIR, f"capture_{uuid.uuid4().hex}.jpg")
                cv2.imwrite(temp_path, ctx.video_processor.frame)
                st.session_state.image_path = temp_path
                st.session_state.image_captured = True
                st.session_state.selected_history_item_index = None
                st.rerun()
            else:
                st_warning("Camera is not ready. Please wait a moment.")
    
    if st.session_state.image_path and st.session_state.image_captured:
        st.image(st.session_state.image_path, caption="Captured Image", use_container_width=True)
        if st.button("🔄 Recapture"):
            st.session_state.image_captured = False
            st.session_state.image_path = None
            st.session_state.webrtc_key = f"webrtc_{uuid.uuid4().hex}" # Reset key to restart component
            st.rerun()

def run_extraction(extractor):
    progress_bar = st.progress(0, text="Initializing...")
    try:
        results, register_cropped, subject_cropped, overlay_path, processing_time = extractor.process_answer_sheet(st.session_state.image_path)
        progress_bar.progress(100, text="Extraction Complete!")
        time.sleep(1)
        progress_bar.empty()

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("📋 Extracted Information")
        if results:
            for label, value in results:
                st.markdown(f"**{label}:** `{value}`")
            # Provide download for text results
        else:
            st_warning("Could not extract any information.")
        st.markdown(f"<p style='text-align: right; font-size: 0.9em;'>Processing time: {processing_time:.2f} seconds</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("🔍 Visual Results")
        img_cols = st.columns(2)
        with img_cols[0]:
            st.markdown("<h6>Original vs. Detections</h6>", unsafe_allow_html=True)
            if overlay_path and os.path.exists(overlay_path):
                image_comparison(img1=st.session_state.image_path, img2=overlay_path, label1="Original", label2="Detections")
        with img_cols[1]:
            st.markdown("<h6>Cropped Regions for OCR</h6>", unsafe_allow_html=True)
            if register_cropped and os.path.exists(register_cropped):
                st.image(register_cropped, caption="Register Number", use_container_width=True)
            if subject_cropped and os.path.exists(subject_cropped):
                st.image(subject_cropped, caption="Subject Code", use_container_width=True)
    except Exception as e:
        progress_bar.empty()
        st_error(f"An error occurred during processing: {e}")

def display_history_tab():
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("📜 Processing History")
    if not st.session_state.results_history:
        st_info("No history yet. Scan an answer sheet to begin.")
    else:
        # Display summary and detail view logic
        for i, item in enumerate(st.session_state.results_history):
            with st.expander(f"Scan from {item.get('timestamp', 'N/A')}"):
                st.markdown(f"**Processing Time:** {item.get('processing_time', 0):.2f} seconds")
                st.markdown("**Results:**")
                if item.get('results'):
                    for label, value in item['results']:
                        st.markdown(f"- **{label}:** `{value}`")
                else:
                    st.write("No information extracted.")

                cols = st.columns(2)
                with cols[0]:
                    if item.get('original_image_path') and os.path.exists(item['original_image_path']):
                        st.image(item['original_image_path'], caption="Original Image")
                with cols[1]:
                    if item.get('overlay_image_path') and os.path.exists(item['overlay_image_path']):
                        st.image(item['overlay_image_path'], caption="Detections Overlay")
    st.markdown('</div>', unsafe_allow_html=True)

def display_about_tab():
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("ℹ️ About the Smart Answer Sheet Scanner")
    st.markdown("""
    This application leverages computer vision to automatically extract Register Numbers and Subject Codes from answer sheets.
    - **Object Detection:** YOLOv8 identifies the locations of the relevant fields.
    - **Text Recognition (OCR):** Specialized CRNN models read characters within detected regions.
    - **Web Interface:** Built with Streamlit for an interactive experience.
    """)
    st.markdown("---")
    st.markdown("<h6>Disclaimer:</h6>", unsafe_allow_html=True)
    st_warning("This tool is for demonstration purposes. Extracted results should always be verified for accuracy.")
    st.markdown('</div>', unsafe_allow_html=True)