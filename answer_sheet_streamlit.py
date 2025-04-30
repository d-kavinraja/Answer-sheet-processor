import streamlit as st
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import torch.nn as nn
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import uuid
import time
from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison
from datetime import datetime
import json

# Set page configuration
st.set_page_config(
    page_title="Smart Answer Sheet Scanner",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            color: #000000;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            color: #000000;
        }
        h1, h2, h3, h4, h5, h6, p, div, span, li, a {
            color: #000000 !important;
        }
        [data-testid="stHeader"] button {
            display: none !important;
        }
        .stButton>button {
            background-color: #d3d3d3; /* Light gray background */
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #b0b0b0; /* Slightly darker light gray on hover */
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        .stButton>button:active, .stButton>button:focus {
            background-color: #4CAF50 !important; /* Green when clicked */
        }
        .success-box {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724 !important;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .error-box {
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24 !important;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: #cce5ff;
            border-color: #b8daff;
            color: #004085 !important;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .result-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .header-container {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            padding: 20px;
            border-radius: 10px;
            color: white !important;
            margin-bottom: 30px;
        }
        .header-container h1, .header-container p {
            color: white !important;
        }
        .camera-container {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 10px;
            background-color: #f8f9fa;
        }
        .image-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .tab-content {
            padding: 20px;
            border-radius: 0 0 10px 10px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .history-item {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            background-color: #f1f1f1;
            cursor: pointer;
            transition: all 0.3s;
            border-left: 5px solid #4CAF50;
        }
        .history-item:hover {
            background-color: #e1e1e1;
            transform: translateY(-2px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .history-item p {
            margin: 5px 0;
            color: #333 !important;
        }
        .footer {
            margin-top: 50px;
            padding: 20px;
            text-align: center;
            color: #333333 !important;
            font-size: 0.9rem;
            background-color: #e9ecef;
            border-radius: 10px;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.05);
            width: 100%;
        }
        .footer p {
            margin: 5px 0;
            color: #333333 !important;
        }
        .footer a {
            color: #4CAF50 !important;
            text-decoration: none;
            transition: color 0.3s;
        }
        .footer a:hover {
            color: #388E3C !important;
            text-decoration: underline;
        }
        .footer-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }
        @media (max-width: 768px) {
            .footer {
                padding: 15px;
                font-size: 0.8rem;
            }
            .footer-content {
                flex-direction: column;
                gap: 8px;
            }
        }
        @media (max-width: 480px) {
            .footer {
                padding: 10px;
                font-size: 0.7rem;
            }
            .footer-content {
                gap: 6px;
            }
        }
        .camera-controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 10px;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50 !important;
        }
        .input-buttons-col {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 20px;
            max-width: 200px;
            margin-left: auto;
            margin-right: auto;
        }
        .extracted-output {
            background-color: #e6f3ff;
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-family: 'Courier New', Courier, monospace;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()
# Initialize session state
if 'image_path' not in st.session_state:
    st.session_state.image_path = None
if 'image_captured' not in st.session_state:
    st.session_state.image_captured = False
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'processing_start_time' not in st.session_state:
    st.session_state.processing_start_time = None
if 'selected_history_item' not in st.session_state:
    st.session_state.selected_history_item = None
if 'webrtc_key' not in st.session_state:
    st.session_state.webrtc_key = uuid.uuid4().hex
if 'input_method' not in st.session_state:
    st.session_state.input_method = "Upload Image"

# Define CRNN model
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout2d(0.3),
            nn.Conv2d(512, 512, kernel_size=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Cache model loading
@st.cache_resource
def load_extractor():
    try:
        extractor = AnswerSheetExtractor(
            "improved_weights.pt",
            "best_crnn_model.pth",
            "best_subject_code_model.pth"
        )
        return extractor
    except Exception as e:
        st.error(f"Failed to initialize extractor: {e}")
        st.info("Ensure model files (improved_weights.pt, best_crnn_model.pth, best_subject_code_model.pth) are present.")
        return None

# AnswerSheetExtractor class
class AnswerSheetExtractor:
    def __init__(self, yolo_weights_path, register_crnn_model_path, subject_crnn_model_path):
        for dir in ["cropped_register_numbers", "cropped_subject_codes", "results", "uploads", "captures"]:
            os.makedirs(dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(yolo_weights_path):
            raise FileNotFoundError(f"YOLO weights not found at: {yolo_weights_path}")
        try:
            self.yolo_model = YOLO(yolo_weights_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

        self.register_crnn_model = CRNN(num_classes=11)
        self.register_crnn_model.to(self.device)
        if not os.path.exists(register_crnn_model_path):
            raise FileNotFoundError(f"Register CRNN model not found at: {register_crnn_model_path}")
        try:
            checkpoint = torch.load(register_crnn_model_path, map_location=self.device)
            self.register_crnn_model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load register CRNN model: {e}")
        self.register_crnn_model.eval()

        self.subject_crnn_model = CRNN(num_classes=37)
        self.subject_crnn_model.to(self.device)
        if not os.path.exists(subject_crnn_model_path):
            raise FileNotFoundError(f"Subject CRNN model not found at: {subject_crnn_model_path}")
        try:
            self.subject_crnn_model.load_state_dict(torch.load(subject_crnn_model_path, map_location=self.device))
        except Exception as e:
            raise RuntimeError(f"Failed to load subject CRNN model: {e}")
        self.subject_crnn_model.eval()

        self.register_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.subject_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.char_map = {i: str(i-1) for i in range(1, 11)}
        self.char_map.update({i: chr(i - 11 + ord('A')) for i in range(11, 37)})

    def detect_regions(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            st.error(f"Could not load image from {image_path}")
            return [], [], None

        results = self.yolo_model(image)
        detections = results[0].boxes
        classes = results[0].names

        register_regions = []
        subject_regions = []
        overlay = image.copy()

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = classes[class_id]

            h, w = image.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if x1 >= x2 or y1 >= y2:
                continue

            color = (0, 255, 0) if label == "RegisterNumber" else (0, 0, 255) if label == "SubjectCode" else (255, 0, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            text_y = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(overlay, f"{label} {confidence:.2f}", (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == "RegisterNumber" and confidence > 0.2:
                padding = 10
                padded_x1, padded_y1 = max(0, x1 - padding), max(0, y1 - padding)
                padded_x2, padded_y2 = min(w, x2 + padding), min(h, y2 + padding)
                cropped_region = image[padded_y1:padded_y2, padded_x1:padded_x2]
                save_path = f"cropped_register_numbers/register_number_{uuid.uuid4().hex}.jpg"
                cv2.imwrite(save_path, cropped_region)
                register_regions.append((save_path, confidence))
            elif label == "SubjectCode" and confidence > 0.2:
                padding = 10
                padded_x1, padded_y1 = max(0, x1 - padding), max(0, y1 - padding)
                padded_x2, padded_y2 = min(w, x2 + padding), min(h, y2 + padding)
                cropped_region = image[padded_y1:padded_y2, padded_x1:padded_x2]
                save_path = f"cropped_subject_codes/subject_code_{uuid.uuid4().hex}.jpg"
                cv2.imwrite(save_path, cropped_region)
                subject_regions.append((save_path, confidence))

        overlay_path = f"results/detection_overlay_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(overlay_path, overlay)
        return register_regions, subject_regions, overlay_path

    def extract_text(self, image_path, model, img_transform, char_map):
        try:
            if not os.path.exists(image_path):
                st.error(f"Cropped image not found: {image_path}")
                return "FILE_MISSING"
            image = Image.open(image_path).convert('L')
            image_tensor = img_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = model(image_tensor).squeeze(1)
                output = output.softmax(1).argmax(1)
                seq = output.cpu().numpy()
                prev = 0
                result = []
                for s in seq:
                    if s != 0 and s != prev:
                        result.append(char_map.get(s, ''))
                    prev = s
            return ''.join(result)
        except Exception as e:
            st.error(f"Failed to extract text from {image_path}: {e}")
            return "ERROR"

    def extract_register_number(self, image_path):
        register_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}}
        return self.extract_text(image_path, self.register_crnn_model, self.register_transform, register_char_map)

    def extract_subject_code(self, image_path):
        subject_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}, **{i: chr(i - 11 + ord('A')) for i in range(11, 37)}}
        return self.extract_text(image_path, self.subject_crnn_model, self.subject_transform, subject_char_map)

    def process_answer_sheet(self, image_path):
        st.session_state.processing_start_time = time.time()
        with st.spinner("Detecting regions..."):
            register_regions, subject_regions, overlay_path = self.detect_regions(image_path)

        results = []
        best_register_cropped_path = None
        best_subject_cropped_path = None

        if register_regions:
            best_region = max(register_regions, key=lambda x: x[1])
            best_register_cropped_path = best_region[0]
            with st.spinner("Extracting Register Number..."):
                register_number = self.extract_register_number(best_register_cropped_path)
            results.append(("Register Number", register_number))
            st_success(f"Register Number detected with confidence {best_region[1]:.2f}.")
        else:
            st.warning("No RegisterNumber regions detected.")

        if subject_regions:
            best_subject = max(subject_regions, key=lambda x: x[1])
            best_subject_cropped_path = best_subject[0]
            with st.spinner("Extracting Subject Code..."):
                subject_code = self.extract_subject_code(best_subject_cropped_path)
            results.append(("Subject Code", subject_code))
            st_success(f"SubjectCode detected with confidence {best_subject[1]:.2f}.")
        else:
            st.warning("No SubjectCode regions detected.")

        processing_time = time.time() - st.session_state.processing_start_time
        if results or overlay_path:
            history_item = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "original_image_path": image_path,
                "overlay_image_path": overlay_path,
                "register_cropped_path": best_register_cropped_path,
                "subject_cropped_path": best_subject_cropped_path,
                "results": results,
                "processing_time": processing_time
            }
            st.session_state.results_history.append(history_item)

        return results, best_register_cropped_path, best_subject_cropped_path, overlay_path, processing_time

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ]}
)

# Video processor class
class VideoProcessor:
    def __init__(self):
        self.frame = None
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_count = 0
        self.last_processed = 0
        self.process_interval = 0.1

    def recv(self, frame):
        current_time = time.time()
        if current_time - self.last_processed < self.process_interval:
            return av.VideoFrame.from_ndarray(self.frame, format="bgr24") if self.frame is not None else frame

        self.frame = frame.to_ndarray(format="bgr24")
        self.last_processed = current_time
        self.frame_count += 1
        if current_time - self.last_frame_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_frame_time)
            self.last_frame_time = current_time
            self.frame_count = 0

        cv2.putText(self.frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        h, w = self.frame.shape[:2]
        center_x, center_y = w//2, h//2
        cv2.line(self.frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 1)
        cv2.line(self.frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 1)
        cv2.putText(self.frame, "Scanning...", (w//2 - 50, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return av.VideoFrame.from_ndarray(self.frame, format="bgr24")

# Colored text boxes
def st_success(text):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def st_error(text):
    st.markdown(f'<div class="error-box">{text}</div>', unsafe_allow_html=True)

def st_info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def st_warning(text):
    st.markdown(f'<div class="error-box" style="background-color: #fff3cd; border-color: #ffeeba; color: #856404 !important;">{text}</div>', unsafe_allow_html=True)

# Header display
def display_header():
    with st.container():
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image("https://img.icons8.com/ios-filled/100/ffffff/scanner.png", width=80)
        with col2:
            st.markdown('<h1>Smart Answer Sheet Scanner</h1>', unsafe_allow_html=True)
            st.markdown('<p>Automatically extract register numbers and subject codes</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Download button
def get_image_download_button(image_path, filename, button_text):
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as file:
                return st.download_button(
                    label=button_text,
                    data=file,
                    file_name=filename,
                    mime="image/jpeg",
                    key=f"download_{filename}_{uuid.uuid4().hex}"
                )
        except Exception as e:
            st_error(f"Failed to create download button for {filename}: {e}")
    return None

# Save results
def save_results_to_file(results, filename_prefix="results"):
    try:
        filename = f"{filename_prefix}_{uuid.uuid4().hex}.txt"
        filepath = os.path.join("results", filename)
        with open(filepath, "w") as f:
            for label, value in results:
                f.write(f"{label}: {value}\n")
        return filepath
    except Exception as e:
        st_error(f"Failed to save results: {e}")
        return None

# Main app
def main():
    display_header()

    model_files = ["improved_weights.pt", "best_crnn_model.pth", "best_subject_code_model.pth"]
    if not all(os.path.exists(file) for file in model_files):
        st_error("Missing model files: " + ", ".join(file for file in model_files if not os.path.exists(file)))
        st_info("Ensure YOLOv8 and CRNN weights are in the script directory.")
        st.stop()

    with st.spinner("Loading models..."):
        extractor = load_extractor()
        if extractor:
            st_success("Models loaded successfully!")
        else:
            st.stop()

    selected_tab = option_menu(
        menu_title=None,
        options=["Scan", "History", "About"],
        icons=["camera", "clock-history", "info-circle"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6", "border-radius": "10px"},
            "icon": {"color": "orange", "font-size": "16px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "padding": "10px",
                "--hover-color": "#eee",
                "color": "#000000"
            },
            "nav-link-selected": {"background-color": "#4CAF50", "color": "white"},
        }
    )

    if selected_tab == "Scan":
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("<h3>Choose input method:</h3>", unsafe_allow_html=True)

        # Arrange buttons in a single column
        st.markdown('<div class="input-buttons-col">', unsafe_allow_html=True)
        if st.button("⬆️ Upload Image", key="upload_image_btn"):
            st.session_state.input_method = "Upload Image"
            st.session_state.image_path = None
            st.session_state.image_captured = False
            st.session_state.selected_history_item = None
            st.rerun()
        if st.button("📸 Use Camera", key="use_camera_btn"):
            st.session_state.input_method = "Use Camera"
            st.session_state.image_path = None
            st.session_state.image_captured = False
            st.session_state.selected_history_item = None
            st.session_state.webrtc_key = uuid.uuid4().hex
            st.rerun()
        if st.button("🔄 Reset Scan", key="reset_btn_scan"):
            st.session_state.image_path = None
            st.session_state.image_captured = False
            st.session_state.results_history = []
            st.session_state.selected_history_item = None
            st.session_state.webrtc_key = uuid.uuid4().hex
            st.session_state.input_method = "Upload Image"
            st_info("Reset complete. Capture or upload a new image.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.input_method == "Upload Image":
            with st.container():
                st.markdown('<div class="camera-container">', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("Upload Answer Sheet Image", type=["png", "jpg", "jpeg"], key="uploader")
                if uploaded_file:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    if file_extension not in ["png", "jpg", "jpeg"]:
                        st_error("Unsupported file type. Use PNG, JPG, or JPEG.")
                        st.session_state.image_path = None
                        st.session_state.image_captured = False
                    else:
                        st.session_state.image_path = f"uploads/image_{uuid.uuid4().hex}.{file_extension}"
                        with open(st.session_state.image_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.session_state.image_captured = True
                        st.session_state.selected_history_item = None
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(st.session_state.image_path, caption="Uploaded Image", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                elif not st.session_state.image_path or not st.session_state.image_captured:
                    st.markdown("""
                    <div style="border: 2px dashed #ccc; border-radius: 5px; padding: 20px; text-align: center;">
                        <h3>Drag and drop your answer sheet image here</h3>
                        <p>Supported formats: JPG, PNG, JPEG</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        else:  # Use Camera
            with st.container():
                st.markdown('<div class="camera-container">', unsafe_allow_html=True)
                if not st.session_state.image_captured:
                    st.markdown("<h3>📸 Live Camera Feed</h3>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <p>Position answer sheet in frame, ensuring register number and subject code are visible.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    ctx = webrtc_streamer(
                        key=st.session_state.webrtc_key,
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration=RTC_CONFIGURATION,
                        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                        video_processor_factory=VideoProcessor,
                        async_processing=False,
                    )

                    st.markdown('<div class="camera-controls">', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        capture_btn_disabled = not (ctx.state.playing and ctx.video_transformer and hasattr(ctx.video_transformer, 'frame') and ctx.video_transformer.frame is not None)
                        st.button("📸 Capture Image", key="capture_btn")
                    with col2:
                        st.button("🔄 Restart Camera", key="restart_camera_btn")
                    st.markdown('</div>', unsafe_allow_html=True)

                    if ctx.video_transformer and st.session_state.get('capture_btn', False):
                        frame = None
                        timeout = 5
                        start_time = time.time()
                        while time.time() - start_time < timeout:
                            if ctx.video_transformer.frame is not None:
                                frame = ctx.video_transformer.frame
                                break
                            time.sleep(0.1)
                        if frame is not None:
                            st.session_state.image_path = f"captures/image_{uuid.uuid4().hex}.jpg"
                            try:
                                cv2.imwrite(st.session_state.image_path, frame)
                                if not os.path.exists(st.session_state.image_path):
                                    raise IOError("Failed to save captured image")
                                st.session_state.image_captured = True
                                st.session_state.selected_history_item = None
                                st.session_state.webrtc_key = uuid.uuid4().hex
                                st_success("Image captured successfully!")
                                st.rerun()
                            except Exception as e:
                                st_error(f"Error saving captured image: {e}")
                                st.session_state.image_path = None
                                st.session_state.image_captured = False
                        else:
                            st_error("No frame available. Ensure camera is active.")

                    if st.session_state.get('restart_camera_btn', False):
                        st.session_state.webrtc_key = uuid.uuid4().hex
                        st.session_state.image_captured = False
                        st.session_state.image_path = None
                        st.rerun()

                else:
                    if st.session_state.image_path and os.path.exists(st.session_state.image_path):
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(st.session_state.image_path, caption="Captured Image", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st_error("Captured image file not found.")
                        st.session_state.image_captured = False

                st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.image_path and st.session_state.image_captured and st.session_state.selected_history_item is None:
            process_btn = st.button("🔍 Extract Information", key="extract_btn")
            if process_btn:
                st.session_state.selected_history_item = None
                status_placeholder = st.empty()
                status_placeholder.info("Starting processing...")
                try:
                    results, register_cropped, subject_cropped, overlay_path, processing_time = extractor.process_answer_sheet(st.session_state.image_path)
                    status_placeholder.empty()
                    st_success(f"Extraction completed in {processing_time:.2f} seconds!")

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
                                    label="📥 Download Results as Text",
                                    data=file,
                                    file_name="extracted_data.txt",
                                    mime="text/plain",
                                    key=f"download_results_{uuid.uuid4().hex}"
                                )
                    else:
                        st_error("No information extracted.")
                        st_info("Try adjusting the image or check model training.")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.subheader("🔍 Detection Results")
                    if st.session_state.image_path and overlay_path and os.path.exists(st.session_state.image_path) and os.path.exists(overlay_path):
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        image_comparison(
                            img1=st.session_state.image_path,
                            img2=overlay_path,
                            label1="Original",
                            label2="Detections",
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st_warning("Could not display image comparison.")

                    if register_cropped or subject_cropped:
                        st.subheader("✂️ Cropped Regions")
                        col1, col2 = st.columns(2)
                        if register_cropped and os.path.exists(register_cropped):
                            with col1:
                                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                                st.image(register_cropped, caption="Register Number", width=200)
                                get_image_download_button(register_cropped, "register_number.jpg", "📥 Download Register Number Image")
                                st.markdown('</div>', unsafe_allow_html=True)
                        if subject_cropped and os.path.exists(subject_cropped):
                            with col2:
                                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                                st.image(subject_cropped, caption="Subject Code", width=200)
                                get_image_download_button(subject_cropped, "subject_code.jpg", "📥 Download Subject Code Image")
                                st.markdown('</div>', unsafe_allow_html=True)
                        if not register_cropped and not subject_cropped:
                            st_info("No cropped images saved.")
                except Exception as e:
                    status_placeholder.empty()
                    st_error(f"Error during processing: {e}")
                    st_info("Try again or check the image.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif selected_tab == "History":
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader(" Processing History")
        if not st.session_state.results_history:
            st_info("No processing history. Scan an answer sheet to see results.")
        else:
            for i, item in enumerate(reversed(st.session_state.results_history)):
                timestamp = item.get("timestamp", "N/A")
                results_summary = ", ".join([f"{label}: {value}" for label, value in item.get("results", [])])
                processing_time = item.get("processing_time", 0)
                history_item_index = len(st.session_state.results_history) - 1 - i
                st.markdown(f"""
                <div class="history-item">
                    <p><strong>Timestamp:</strong> {timestamp}</p>
                    <p><strong>Results:</strong> {results_summary}</p>
                    <p><strong>Processing Time:</strong> {processing_time:.2f} seconds</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("View Details", key=f"view_history_{history_item_index}"):
                    st.session_state.selected_history_item = item
                    st.rerun()

            if st.session_state.selected_history_item:
                st.markdown("---")
                st.subheader("Detailed History View")
                selected_item = st.session_state.selected_history_item
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"<p><strong>Timestamp:</strong> {selected_item.get('timestamp', 'N/A')}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><strong>Processing Time:</strong> {selected_item.get('processing_time', 0):.2f} seconds</p>", unsafe_allow_html=True)
                st.markdown("<h4>Extracted Information:</h4>", unsafe_allow_html=True)
                if selected_item.get("results"):
                    st.markdown('<div class="extracted-output">', unsafe_allow_html=True)
                    for label, value in selected_item["results"]:
                        st.markdown(f"**{label}:** `{value}`")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st_info("No extracted results found.")
                st.markdown('</div>', unsafe_allow_html=True)

                st.subheader("Images")
                original_image_path = selected_item.get("original_image_path")
                overlay_image_path = selected_item.get("overlay_image_path")
                if original_image_path and overlay_image_path and os.path.exists(original_image_path) and os.path.exists(overlay_image_path):
                    st.markdown("<h4>Original vs. Detections:</h4>", unsafe_allow_html=True)
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    image_comparison(
                        img1=original_image_path,
                        img2=overlay_image_path,
                        label1="Original",
                        label2="Detections",
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st_warning("Original or overlay image not found.")

                st.markdown("<h4>Cropped Regions:</h4>", unsafe_allow_html=True)
                col_cropped1, col_cropped2 = st.columns(2)
                register_cropped_path = selected_item.get("register_cropped_path")
                subject_cropped_path = selected_item.get("subject_cropped_path")
                if register_cropped_path and os.path.exists(register_cropped_path):
                    with col_cropped1:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(register_cropped_path, caption="Register Number (Cropped)", width=200)
                        get_image_download_button(register_cropped_path, "history_register_number.jpg", "📥 Download Image")
                        st.markdown('</div>', unsafe_allow_html=True)
                if subject_cropped_path and os.path.exists(subject_cropped_path):
                    with col_cropped2:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(subject_cropped_path, caption="Subject Code (Cropped)", width=200)
                        get_image_download_button(subject_cropped_path, "history_subject_code.jpg", "📥 Download Image")
                        st.markdown('</div>', unsafe_allow_html=True)
                if not register_cropped_path and not subject_cropped_path:
                    st_info("No cropped images saved.")
                if st.button("Hide Details", key="hide_history_details"):
                    st.session_state.selected_history_item = None
        st.markdown('</div>', unsafe_allow_html=True)

    elif selected_tab == "About":
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("ℹ️ About the Smart Answer Sheet Scanner")
        col_about1, col_about2 = st.columns([2, 3])
        with col_about1:
            st.image("https://img.icons8.com/ios-filled/200/000000/upload.png", width=200)
        with col_about2:
            st.markdown("""
            <p>This app uses computer vision to extract register numbers and subject codes from answer sheets.</p>
            <h4>Key Features:</h4>
            <ul>
                <li><strong>Object Detection:</strong> YOLOv8 detects Register Number and Subject Code regions.</li>
                <li><strong>Text Recognition:</strong> CRNN models recognize digits and alphanumeric characters.</li>
                <li><strong>Flexible Input:</strong> Upload images or use camera.</li>
                <li><strong>Visual Feedback:</strong> Shows detection overlays and cropped regions.</li>
                <li><strong>History:</strong> Tracks past scans.</li>
            </ul>
            <h4>How it Works:</h4>
            <ol>
                <li>Upload or capture an answer sheet image.</li>
                <li>YOLO detects regions.</li>
                <li>Regions are cropped.</li>
                <li>CRNN extracts text.</li>
                <li>Results are displayed with overlays.</li>
            </ol>
            """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h4>Model Details:</h4>", unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li><strong>YOLO:</strong> Custom-trained for RegisterNumber and SubjectCode.</li>
            <li><strong>Register CRNN:</strong> Digit recognition (0-9).</li>
            <li><strong>Subject CRNN:</strong> Alphanumeric recognition (0-9, A-Z).</li>
        </ul>
        <p>Ensure model weights are in the script directory.</p>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h4>Disclaimer:</h4>", unsafe_allow_html=True)
        st.markdown("<p>Accuracy depends on image quality and model training.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown('<div class="footer-content">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()