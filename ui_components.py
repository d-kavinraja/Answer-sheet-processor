import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_image_comparison import image_comparison
from video_processor import VideoProcessor
import os
import uuid
import io
import time
import PyPDF2
import pypdfium2 as pdfium
import logging
from datetime import datetime
from utils import st_success, st_error, st_info, st_warning, get_image_download_button, save_results_to_file

logger = logging.getLogger(__name__)

def local_css():
    st.markdown("""
    <style>
        :root {
            --primary-color: #2E86AB;
            --secondary-color: #6B7280;
            --background-color: #F9FAFB;
            --secondary-background-color: #FFFFFF;
            --text-color: #1F2937;
            --text-color-inverse: #FFFFFF;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            background-color: var(--background-color);
            font-family: 'Inter', sans-serif;
        }
        [data-testid="stHeader"] button {
            display: none !important;
        }
        .stButton>button {
            background-color: var(--primary-color);
            color: var(--text-color-inverse);
            font-weight: 500;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            font-size: 1rem;
            border: none;
        }
        .stButton>button:hover {
            filter: brightness(90%);
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .success-box {
            background-color: #D1FAE5;
            border: 1px solid #10B981;
            color: #065F46 !important;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .error-box {
            background-color: #FEE2E2;
            border: 1px solid #EF4444;
            color: #991B1B !important;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #DBEAFE;
            border: 1px solid #3B82F6;
            color: #1E40AF !important;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .warning-box {
            background-color: #FEF3C7;
            border: 1px solid #F59E0B;
            color: #92400E !important;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .result-card {
            background-color: var(--secondary-background-color);
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        .header-container {
            background: linear-gradient(90deg, #2E86AB 0%, #1E3A8A 100%);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            color: var(--text-color-inverse);
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .camera-container {
            border: 2px dashed #D1D5DB;
            border-radius: 8px;
            padding: 1.5rem;
            background-color: var(--secondary-background-color);
            margin-bottom: 1.5rem;
        }
        .image-container {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .tab-content {
            padding: 1.5rem;
            border-radius: 8px;
            background-color: var(--secondary-background-color);
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .history-item {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            background-color: var(--secondary-background-color);
            cursor: pointer;
            transition: all 0.3s ease;
            border-left: 4px solid var(--primary-color);
        }
        .history-item:hover {
            filter: brightness(95%);
            transform: translateY(-2px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .footer {
            margin-top: 2rem;
            padding: 1rem;
            text-align: center;
            font-size: 0.9rem;
            background-color: var(--secondary-background-color);
            border-radius: 8px;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.05);
        }
        .footer a {
            color: var(--primary-color) !important;
            text-decoration: none;
            transition: color 0.3s;
        }
        .footer a:hover {
            filter: brightness(85%);
            text-decoration: underline;
        }
        .camera-controls {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-top: 1rem;
        }
        .input-buttons-col {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-bottom: 1.5rem;
            max-width: 300px;
            margin-left: auto;
            margin-right: auto;
        }
        .extracted-output {
            background-color: #F3F4F6;
            border: 1px solid var(--primary-color);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            font-family: 'Courier New', Courier, monospace;
            color: var(--text-color);
        }
        .image-comparison-container {
            width: 100%;
            max-width: 600px;
            margin: 0 auto;
        }
        .auth-form {
            max-width: 400px;
            margin: 0 auto;
            padding: 1.5rem;
            background-color: var(--secondary-background-color);
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .stTextInput>div>input {
            border-radius: 8px;
            border: 1px solid #D1D5DB;
            padding: 0.75rem;
        }
        .stProgress > div > div > div > div {
            background-color: var(--primary-color) !important;
        }
        @media (max-width: 768px) {
            .stApp { padding: 0 1rem; }
            .header-container { flex-direction: column; text-align: center; }
            .camera-controls { flex-direction: column; gap: 0.75rem; }
            .input-buttons-col { max-width: 100%; }
            .auth-form { max-width: 100%; }
        }
        @media (max-width: 480px) {
            .stButton>button { font-size: 0.9rem; padding: 0.5rem 1rem; }
            .footer { font-size: 0.8rem; }
        }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    with st.container():
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown('<div style="font-size: 48px; text-align: center;">📝</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<h1 style="margin: 0; font-size: 2rem;">Smart Answer Sheet Scanner</h1>', unsafe_allow_html=True)
            st.markdown('<p style="margin: 0; font-size: 1rem;">Automatically extract register numbers and subject codes</p>', unsafe_allow_html=True)
        if st.session_state.logged_in:
            st.markdown(f'<p style="margin: 0; font-size: 0.9rem; text-align: right;">Welcome, {st.session_state.username} | <a href="#" onclick="st.session_state.logged_in=False;st.session_state.username=\'\';st.experimental_rerun()">Logout</a></p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_signup(mongo_manager, email_service):
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("📝 Sign Up")
    with st.form("signup_form", border=False):
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        username = st.text_input("Username", max_chars=50, placeholder="Enter your username")
        email = st.text_input("Email", max_chars=100, placeholder="Enter your email")
        password = st.text_input("Password", type="password", max_chars=50, placeholder="Enter your password")
        confirm_password = st.text_input("Confirm Password", type="password", max_chars=50, placeholder="Confirm your password")
        submit = st.form_submit_button("Sign Up", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submit:
            if not all([username, email, password, confirm_password]):
                st_error("All fields are required.")
            elif password != confirm_password:
                st_error("Passwords do not match.")
            elif len(password) < 8:
                st_error("Password must be at least 8 characters long.")
            else:
                result = mongo_manager.add_user(username, email, password)
                if result["success"]:
                    otp = email_service.generate_otp()
                    if email_service.send_otp_email(email, otp, username) and mongo_manager.store_otp(email, otp):
                        st.session_state.temp_user_data = {"username": username, "email": email}
                        st.session_state.otp_stage = True
                        st.rerun()
                    else:
                        st_error("Failed to send OTP. Please try again.")
                else:
                    st_error(result["message"])
    
    if st.session_state.otp_stage and st.session_state.temp_user_data:
        st.markdown("---")
        st.subheader("🔐 Verify OTP")
        with st.form("otp_form", border=False):
            st.markdown('<div class="auth-form">', unsafe_allow_html=True)
            otp = st.text_input("Enter OTP", max_chars=6, placeholder="6-digit OTP")
            verify = st.form_submit_button("Verify", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if verify:
                if not otp:
                    st_error("Please enter the OTP.")
                else:
                    result = mongo_manager.verify_otp(st.session_state.temp_user_data["email"], otp)
                    if result["success"]:
                        st_success("Email verified successfully!")
                        email_service.send_welcome_email(st.session_state.temp_user_data["email"], st.session_state.temp_user_data["username"])
                        st.session_state.logged_in = True
                        st.session_state.username = st.session_state.temp_user_data["username"]
                        st.session_state.email = st.session_state.temp_user_data["email"]
                        st.session_state.otp_stage = False
                        st.session_state.temp_user_data = {}
                        mongo_manager.update_last_login(st.session_state.username)
                        st.rerun()
                    else:
                        st_error(result["message"])
    
    st.markdown('<p style="text-align: center;">Already have an account? <a href="#" onclick="st.session_state.auth_tab=\'Sign In\';st.experimental_rerun()">Sign In</a></p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_signin(mongo_manager, email_service):
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("🔑 Sign In")
    with st.form("signin_form", border=False):
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        username = st.text_input("Username", max_chars=50, placeholder="Enter your username")
        password = st.text_input("Password", type="password", max_chars=50, placeholder="Enter your password")
        submit = st.form_submit_button("Sign In", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if submit:
            if not all([username, password]):
                st_error("All fields are required.")
            else:
                user = mongo_manager.find_user(username)
                if user and mongo_manager.check_password(password, user["password_hash"]):
                    if user["email_verified"]:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.email = user["email"]
                        mongo_manager.update_last_login(username)
                        st.rerun()
                    else:
                        otp = email_service.generate_otp()
                        if email_service.send_otp_email(user["email"], otp, username) and mongo_manager.store_otp(user["email"], otp):
                            st.session_state.temp_user_data = {"username": username, "email": user["email"]}
                            st.session_state.otp_stage = True
                            st.rerun()
                        else:
                            st_error("Failed to send OTP. Please try again.")
                else:
                    st_error("Invalid username or password.")
    
    if st.session_state.otp_stage and st.session_state.temp_user_data:
        st.markdown("---")
        st.subheader("🔐 Verify OTP")
        with st.form("otp_form_signin", border=False):
            st.markdown('<div class="auth-form">', unsafe_allow_html=True)
            otp = st.text_input("Enter OTP", max_chars=6, placeholder="6-digit OTP")
            verify = st.form_submit_button("Verify", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if verify:
                if not otp:
                    st_error("Please enter the OTP.")
                else:
                    result = mongo_manager.verify_otp(st.session_state.temp_user_data["email"], otp)
                    if result["success"]:
                        st_success("Email verified successfully!")
                        st.session_state.logged_in = True
                        st.session_state.username = st.session_state.temp_user_data["username"]
                        st.session_state.email = st.session_state.temp_user_data["email"]
                        st.session_state.otp_stage = False
                        st.session_state.temp_user_data = {}
                        mongo_manager.update_last_login(st.session_state.username)
                        st.rerun()
                    else:
                        st_error(result["message"])
    
    st.markdown('<p style="text-align: center;">Need an account? <a href="#" onclick="st.session_state.auth_tab=\'Sign Up\';st.experimental_rerun()">Sign Up</a></p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_scan_tab(extractor, mongo_manager):
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("📷 Scan Answer Sheet")
    
    st.markdown('<div class="input-buttons-col">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬆️ Upload Image or PDF", key="upload_image_btn"):
            st.session_state.input_method = "Upload Image"
            st.session_state.image_path = None
            st.session_state.image_captured = False
            st.session_state.selected_history_item_index = None
            st.rerun()
    with col2:
        if st.button("📸 Use Camera", key="use_camera_btn"):
            st.session_state.input_method = "Use Camera"
            st.session_state.image_path = None
            st.session_state.image_captured = False
            st.session_state.selected_history_item_index = None
            st.session_state.webrtc_key = f"webrtc_{uuid.uuid4().hex}"
            st.rerun()
    with col3:
        if st.button("🔄 Reset Scan", key="reset_btn_scan"):
            st.session_state.image_path = None
            st.session_state.image_captured = False
            st.session_state.selected_history_item_index = None
            st.session_state.webrtc_key = f"webrtc_{uuid.uuid4().hex}"
            st.session_state.input_method = "Upload Image"
            st_info("Scan reset. Upload an image or PDF or use the camera.")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    
    if st.session_state.input_method == "Upload Image":
        with st.container():
            st.markdown('<div class="camera-container">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload Answer Sheet Image or PDF",
                type=["png", "jpg", "jpeg", "pdf"],
                key="uploader",
                label_visibility="collapsed"
            )
            if uploaded_file:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                uploads_dir = os.path.join(script_dir, "uploads")
                os.makedirs(uploads_dir, exist_ok=True)
                temp_path = os.path.join(uploads_dir, f"image_{uuid.uuid4().hex}.jpg")
                
                try:
                    file_size = uploaded_file.size / (1024 * 1024)
                    logger.info(f"Uploaded file: {uploaded_file.name}, Size: {file_size:.2f} MB")
                    if file_size > 50:
                        st_error("File size exceeds 50MB. Please upload a smaller file.")
                        st.session_state.image_path = None
                        st.session_state.image_captured = False
                        st.rerun()
                    
                    if file_extension == 'pdf':
                        pdf_file = io.BytesIO(uploaded_file.getvalue())
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        num_pages = len(pdf_reader.pages)
                        if num_pages == 0:
                            st_error("The uploaded PDF is empty.")
                            st.session_state.image_path = None
                            st.session_state.image_captured = False
                            st.rerun()
                        else:
                            st_info(f"PDF uploaded with {num_pages} page(s). Processing the first page.")
                            try:
                                pdf_file.seek(0)
                                pdf = pdfium.PdfDocument(pdf_file)
                                if len(pdf) == 0:
                                    raise Exception("PDF has no pages.")
                                page = pdf[0]
                                pil_image = page.render(scale=300/72).to_pil()
                                pil_image.save(temp_path, 'JPEG')
                                page.close()
                                pdf.close()
                                logger.info(f"PDF page converted to image: {temp_path}")
                            except Exception as e:
                                logger.error(f"pypdfium2 error: {e}")
                                st_error(f"Error converting PDF to image: {e}")
                                st_warning("Trying fallback text extraction...")
                                pdf_file.seek(0)
                                fallback_text = fallback_extract_text(pdf_file)
                                st.markdown('<div class="extracted-output">', unsafe_allow_html=True)
                                st.markdown(f"**Fallback Extracted Text (First Page):** `{fallback_text}`")
                                st.markdown('</div>', unsafe_allow_html=True)
                                st_info("Note: Fallback text extraction may not detect register numbers or subject codes accurately.")
                                st.session_state.image_path = None
                                st.session_state.image_captured = False
                                st.rerun()
                    else:
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        logger.info(f"Image saved: {temp_path}")

                    if os.path.exists(temp_path):
                        st.session_state.image_path = temp_path
                        st.session_state.image_captured = True
                        st.session_state.selected_history_item_index = None
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(st.session_state.image_path, caption="Uploaded Image" if file_extension != 'pdf' else "Converted PDF Page", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st_error("Failed to save the uploaded file.")
                        st.session_state.image_path = None
                        st.session_state.image_captured = False
                except Exception as e:
                    logger.error(f"Error processing uploaded file: {e}")
                    st_error(f"Error processing uploaded file: {e}")
                    st.session_state.image_path = None
                    st.session_state.image_captured = False
            elif not st.session_state.image_path or not st.session_state.image_captured:
                st.markdown("""
                <div style="border: 2px dashed #D1D5DB; border-radius: 8px; padding: 2rem; text-align: center;">
                    <h3 style="margin: 0; color: var(--text-color);">Drag & drop or click to upload</h3>
                    <p style="margin: 0.5rem 0 0 0; color: var(--secondary-color);">Supported formats: JPG, PNG, JPEG, PDF (first page processed)</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:  # Use Camera
        with st.container():
            st.markdown('<div class="camera-container">', unsafe_allow_html=True)
            if not st.session_state.image_captured:
                st.markdown("<h4>📸 Live Camera Feed</h4>", unsafe_allow_html=True)
                st_info("Position the answer sheet within the frame and click 'Capture Image'.")

                RTC_CONFIGURATION = {
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]}
                    ]
                }
                media_constraints = {
                    "video": {
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720},
                        "frameRate": {"ideal": 30}
                    },
                    "audio": False
                }

                ctx = webrtc_streamer(
                    key=st.session_state.webrtc_key,
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    media_stream_constraints=media_constraints,
                    video_processor_factory=VideoProcessor,
                    async_processing=True
                )

                st.markdown('<div class="camera-controls">', unsafe_allow_html=True)
                capture_btn_disabled = not (ctx.state.playing and ctx.video_processor)
                if st.button("📸 Capture Image", key="capture_btn", disabled=capture_btn_disabled, type="primary"):
                    if ctx.video_processor and hasattr(ctx.video_processor, 'frame') and ctx.video_processor.frame is not None:
                        frame_to_save = ctx.video_processor.frame
                        captures_dir = os.path.join(script_dir, "captures")
                        os.makedirs(captures_dir, exist_ok=True)
                        temp_path = os.path.join(captures_dir, f"image_{uuid.uuid4().hex}.jpg")
                        try:
                            cv2.imwrite(temp_path, frame_to_save)
                            if not os.path.exists(temp_path):
                                raise IOError("Failed to save captured image file.")
                            st.session_state.image_path = temp_path
                            st.session_state.image_captured = True
                            st.session_state.selected_history_item_index = None
                            st_success("Image captured successfully!")
                            st.rerun()
                        except Exception as e:
                            st_error(f"Error saving captured image: {e}")
                            st.session_state.image_path = None
                            st.session_state.image_captured = False
                    else:
                        st_warning("No frame available yet. Please wait a moment and try again.")
                st.markdown('</div>', unsafe_allow_html=True)

            elif st.session_state.image_path and os.path.exists(st.session_state.image_path):
                st.markdown("<h4>Captured Image</h4>", unsafe_allow_html=True)
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(st.session_state.image_path, caption="Captured Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                if st.button("🔄 Recapture Image", key="recapture_btn"):
                    st.session_state.image_captured = False
                    st.session_state.image_path = None
                    st.session_state.webrtc_key = f"webrtc_{uuid.uuid4().hex}"
                    st.rerun()
            else:
                st_error("Captured image file missing. Please capture again.")
                st.session_state.image_captured = False
                st.session_state.image_path = None
                if st.button("Go back to Camera", key="back_to_camera_btn"):
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.image_path and st.session_state.image_captured and st.session_state.selected_history_item_index is None:
        st.markdown("---")
        if st.button("🔍 Extract Information", key="extract_btn", type="primary"):
            status_placeholder = st.empty()
            status_placeholder.info("🚀 Starting extraction process...")
            progress_bar = st.progress(0, text="Initializing...")

            try:
                progress_bar.progress(10, text="Processing image...")
                results, register_cropped, subject_cropped, overlay_path, processing_time, history_item = extractor.process_answer_sheet(st.session_state.image_path)
                mongo_manager.store_scan_result(st.session_state.username, history_item)
                progress_bar.progress(100, text="Extraction Complete!")
                time.sleep(1)
                progress_bar.empty()
                status_placeholder.empty()

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("📋 Extracted Information")
                if results:
                    st.markdown('<div class="extracted-output">', unsafe_allow_html=True)
                    for label, value in results:
                        st.markdown(f"**{label}:** `{value}`")
                    st.markdown('</div>', unsafe_allow_html=True)
                    results_file = save_results_to_file(results, f"results_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                    if results_file and os.path.exists(results_file):
                        with open(results_file, "rb") as file:
                            st.download_button(
                                label="📥 Download Results (.txt)",
                                data=file,
                                file_name="extracted_data.txt",
                                mime="text/plain",
                                key=f"download_results_{uuid.uuid4().hex}"
                            )
                else:
                    st_warning("Could not extract any information.")
                st.markdown(f"<p style='text-align: right; font-size: 0.9em;'>Processing time: {processing_time:.2f} seconds</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.subheader("🔍 Visual Results")
                img_cols = st.columns(2)
                with img_cols[0]:
                    st.markdown("<h6>Original vs. Detections</h6>", unsafe_allow_html=True)
                    if st.session_state.image_path and overlay_path and os.path.exists(st.session_state.image_path) and os.path.exists(overlay_path):
                        st.markdown('<div class="image-comparison-container">', unsafe_allow_html=True)
                        image_comparison(
                            img1=st.session_state.image_path,
                            img2=overlay_path,
                            label1="Original",
                            label2="Detections"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        get_image_download_button(overlay_path, "detections_overlay.jpg", "Download Detections Image")
                    else:
                        st_warning("Could not display image comparison.")
                with img_cols[1]:
                    st.markdown("<h6>Cropped Regions</h6>", unsafe_allow_html=True)
                    if register_cropped and os.path.exists(register_cropped):
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(register_cropped, caption="Register Number", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        get_image_download_button(register_cropped, "register_number_crop.jpg", "Download Register Number Cropped Image")
                    if subject_cropped and os.path.exists(subject_cropped):
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(subject_cropped, caption="Subject Code", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        get_image_download_button(subject_cropped, "subject_code_crop.jpg", "Download Subject Code Cropped Image")
                    if not register_cropped and not subject_cropped:
                        st_info("No regions cropped.")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                progress_bar.empty()
                status_placeholder.empty()
                st_error(f"An unexpected error occurred during processing: {e}")
                st_info("Please try again with a different image or PDF.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown('<p>© 2025 Smart Answer Sheet Scanner. Built with Streamlit.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_history_tab(mongo_manager):
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("📜 Processing History")
    
    scans = mongo_manager.get_user_scans(st.session_state.username)
    if not scans:
        st_info("No processing history yet. Scan an answer sheet on the 'Scan' tab to populate history.")
    else:
        st.markdown("Click 'View Details' to see the images and full results for a past scan.")
        for i, item in enumerate(scans):
            timestamp = item.get("timestamp", "N/A")
            results_summary = ", ".join([f"{label}: `{value}`" for label, value in item.get("results", [])]) or "N/A"
            processing_time = item.get("processing_time", 0)
            
            hist_cols = st.columns([3, 1])
            with hist_cols[0]:
                st.markdown(f"""
                <div class="history-item">
                    <p><strong>Scan Time:</strong> {timestamp}</p>
                    <p><strong>Results:</strong> {results_summary}</p>
                    <p><strong>Processing Time:</strong> {processing_time:.2f} sec</p>
                </div>
                """, unsafe_allow_html=True)
            with hist_cols[1]:
                if st.button("View Details", key=f"view_history_{i}"):
                    st.session_state.selected_history_item_index = i
                    st.rerun()

        st.markdown("---")
        
        if st.session_state.selected_history_item_index is not None:
            st.subheader("📜 Detailed History View")
            try:
                selected_item = scans[st.session_state.selected_history_item_index]
            except IndexError:
                st_error("Selected history item not found. It might have been cleared.")
                st.session_state.selected_history_item_index = None
                st.rerun()
                return

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"<h6>Scan Timestamp: {selected_item.get('timestamp', 'N/A')}</h6>", unsafe_allow_html=True)
            st.markdown(f"<p>Processing Time: {selected_item.get('processing_time', 0):.2f} seconds</p>", unsafe_allow_html=True)

            st.markdown("<h6>Extracted Information:</h6>", unsafe_allow_html=True)
            if selected_item.get("results"):
                st.markdown('<div class="extracted-output">', unsafe_allow_html=True)
                for label, value in selected_item["results"]:
                    st.markdown(f"**{label}:** `{value}`")
                st.markdown('</div>', unsafe_allow_html=True)
                results_file = save_results_to_file(selected_item["results"], f"results_{selected_item.get('timestamp', '').replace(' ', '_').replace(':', '-')}")
                if results_file and os.path.exists(results_file):
                    with open(results_file, "rb") as file:
                        st.download_button(
                            label="📥 Download Results (.txt)",
                            data=file,
                            file_name="extracted_data.txt",
                            mime="text/plain",
                            key=f"download_history_results_{i}_{uuid.uuid4().hex}"
                        )
            else:
                st_info("No results were extracted in this scan.")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<h6>Images from Scan:</h6>", unsafe_allow_html=True)
            hist_img_cols = st.columns(2)
            original_image_path = selected_item.get("original_image_path")
            overlay_image_path = selected_item.get("overlay_image_path")
            register_cropped_path = selected_item.get("register_cropped_path")
            subject_cropped_path = selected_item.get("subject_cropped_path")

            with hist_img_cols[0]:
                st.markdown("<u>Original vs. Detections:</u>", unsafe_allow_html=True)
                if original_image_path and overlay_image_path and os.path.exists(original_image_path) and os.path.exists(overlay_image_path):
                    st.markdown('<div class="image-comparison-container">', unsafe_allow_html=True)
                    image_comparison(
                        img1=original_image_path,
                        img2=overlay_image_path,
                        label1="Original",
                        label2="Detections"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                    get_image_download_button(overlay_image_path, "detections_overlay.jpg", "Download Detections Image")
                else:
                    st_warning("Original or detection overlay image not found.")

            with hist_img_cols[1]:
                st.markdown("<u>Cropped Regions:</u>", unsafe_allow_html=True)
                if register_cropped_path and os.path.exists(register_cropped_path):
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(register_cropped_path, caption="Register Number (Cropped)", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    get_image_download_button(register_cropped_path, "register_number_crop.jpg", "Download Register Number Cropped Image")
                else:
                    st.markdown("<p>No Register Number crop.</p>", unsafe_allow_html=True)

                if subject_cropped_path and os.path.exists(subject_cropped_path):
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(subject_cropped_path, caption="Subject Code (Cropped)", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    get_image_download_button(subject_cropped_path, "subject_code_crop.jpg", "Download Subject Code Cropped Image")
                else:
                    st.markdown("<p>No Subject Code crop.</p>", unsafe_allow_html=True)

            if st.button("Hide Details", key="hide_history_details"):
                st.session_state.selected_history_item_index = None
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="footer">', unsafe_allow_html=True)