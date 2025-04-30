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
import streamlit_lottie as st_lottie
import requests
from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison
from datetime import datetime
import json

# Set page configuration
st.set_page_config(
    page_title="Smart Answer Sheet Scanner",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light theme with black text and button alignment
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
        /* Ensure all text is black */
        h1, h2, h3, h4, h5, h6, p, div, span, li, a {
            color: #000000 !important;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            border: none;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        .reset-btn > button {
            background-color: #f44336;
        }
        .reset-btn > button:hover {
            background-color: #d32f2f;
        }
        .capture-btn > button {
            background-color: #2196F3;
            width: 100%;
            margin: 0 auto;
        }
        .capture-btn > button:hover {
            background-color: #0b7dda;
        }
        .refresh-btn > button {
            background-color: #ff9800;
            width: 100%;
            margin: 0 auto;
        }
        .refresh-btn > button:hover {
            background-color: #e68a00;
        }
        .extract-btn > button {
            background-color: #ff9800;
        }
        .extract-btn > button:hover {
            background-color: #e68a00;
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
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            background-color: #f1f1f1;
            cursor: pointer;
            transition: all 0.3s;
        }
        .history-item:hover {
            background-color: #e1e1e1;
            transform: translateY(-2px);
        }
        .footer {
            margin-top: 50px;
            padding: 20px;
            text-align: center;
            color: #000000 !important;
            font-size: 0.8rem;
        }
        /* Center buttons in camera controls */
        .camera-controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 10px;
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
if 'camera_filters' not in st.session_state:
    st.session_state.camera_filters = "None"
if 'enhance_contrast' not in st.session_state:
    st.session_state.enhance_contrast = False

# Function to load lottie animations with fallback
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=3)  # Reduced timeout for faster response
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"Failed to load Lottie animation from {url}. Status code: {r.status_code}")
            return None
    except Exception as e:
        st.warning(f"Error loading Lottie animation: {e}")
        return None

# Load lottie animations (optimized for speed)
lottie_scanning = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_upload = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ydo1amjm.json")
lottie_camera = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_rumq6s.json")
lottie_success = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_jbrw3hsa.json")  # Professional checkmark
lottie_error = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_qpwbiyxf.json")
lottie_processing = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_kkhqcsbh.json")

# Define the CRNN model class
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

# Define custom image processing filters
def apply_filter(frame, filter_type):
    if filter_type == "None":
        return frame
    elif filter_type == "Grayscale":
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif filter_type == "Adaptive Threshold":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Canny Edge":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_type == "High Contrast":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)
        return cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
    elif filter_type == "Document Scan":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return frame

# Function to enhance contrast
def enhance_image_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# Cache model loading for performance
@st.cache_resource
def load_extractor():
    return AnswerSheetExtractor(
        "improved_weights.pt",
        "best_crnn_model.pth",
        "best_subject_code_model.pth"
    )

# Define the AnswerSheetExtractor class
class AnswerSheetExtractor:
    def __init__(self, yolo_weights_path, register_crnn_model_path, subject_crnn_model_path):
        os.makedirs("cropped_register_numbers", exist_ok=True)
        os.makedirs("cropped_subject_codes", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.yolo_model = YOLO(yolo_weights_path)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load YOLO model from {yolo_weights_path}: {e}")
        
        self.register_crnn_model = CRNN(num_classes=11)  # 10 digits + blank
        self.register_crnn_model.to(self.device)
        try:
            checkpoint = torch.load(register_crnn_model_path, map_location=self.device)
            self.register_crnn_model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise FileNotFoundError(f"Failed to load register CRNN model from {register_crnn_model_path}: {e}")
        self.register_crnn_model.eval()
        
        self.subject_crnn_model = CRNN(num_classes=37)  # blank + 0-9 + A-Z
        self.subject_crnn_model.to(self.device)
        try:
            self.subject_crnn_model.load_state_dict(torch.load(subject_crnn_model_path, map_location=self.device))
        except Exception as e:
            raise FileNotFoundError(f"Failed to load subject CRNN model from {subject_crnn_model_path}: {e}")
        self.subject_crnn_model.eval()
        
        self.register_transform = transforms.Compose([
            transforms.Resize((32, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.subject_transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.char_map = {i: str(i-1) for i in range(1, 11)}
        self.char_map.update({i: chr(i - 11 + ord('A')) for i in range(11, 37)})

    def detect_regions(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        results = self.yolo_model(image)
        detections = results[0].boxes
        classes = results[0].names
        
        register_regions = []
        subject_regions = []
        overlay = image.copy()
        
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = classes[class_id]
            cropped_region = image[y1:y2, x1:x2]
            
            if label == "RegisterNumber":
                color = (0, 255, 0)
            elif label == "SubjectCode":
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
                
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, f"{label} {confidence:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if label == "RegisterNumber" and confidence > 0.2:
                save_path = f"cropped_register_numbers/register_number_{i}.jpg"
                cv2.imwrite(save_path, cropped_region)
                register_regions.append((save_path, confidence))
            elif label == "SubjectCode" and confidence > 0.2:
                save_path = f"cropped_subject_codes/subject_code_{i}.jpg"
                cv2.imwrite(save_path, cropped_region)
                subject_regions.append((save_path, confidence))
        
        overlay_path = f"results/detection_overlay_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(overlay_path, overlay)
        
        return register_regions, subject_regions, overlay_path
    
    def extract_register_number(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            image_tensor = self.register_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.register_crnn_model(image_tensor).squeeze(1)
                output = output.softmax(1).argmax(1)
                seq = output.cpu().numpy()
                prev = -1
                result = []
                for s in seq:
                    if s != 0 and s != prev:
                        result.append(s - 1)
                    prev = s
            extracted = ''.join(map(str, result))
            return extracted
        except Exception as e:
            st.error(f"Failed to extract Register Number from {image_path}: {e}")
            return "ERROR"
    
    def extract_subject_code(self, image_path):
        try:
            image = Image.open(image_path).convert('L')
            image_tensor = self.subject_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.subject_crnn_model(image_tensor).squeeze(1)
                output = output.softmax(1).argmax(1)
                seq = output.cpu().numpy()
                prev = 0
                result = []
                for s in seq:
                    if s != 0 and s != prev:
                        result.append(self.char_map.get(s, ''))
                    prev = s
            extracted = ''.join(result)
            return extracted
        except Exception as e:
            st.error(f"Failed to extract Subject Code from {image_path}: {e}")
            return "ERROR"
    
    def process_answer_sheet(self, image_path):
        st.session_state.processing_start_time = time.time()
        
        register_regions, subject_regions, overlay_path = self.detect_regions(image_path)
        results = []
        register_cropped_path = None
        subject_cropped_path = None
        
        if register_regions:
            best_region = max(register_regions, key=lambda x: x[1])
            register_cropped_path = best_region[0]
            register_number = self.extract_register_number(register_cropped_path)
            results.append(("Register Number", register_number))
        else:
            st.warning("No RegisterNumber regions detected.")
        
        if subject_regions:
            best_subject = max(subject_regions, key=lambda x: x[1])
            subject_cropped_path = best_subject[0]
            subject_code = self.extract_subject_code(subject_cropped_path)
            results.append(("Subject Code", subject_code))
        else:
            st.warning("No SubjectCode regions detected.")
            
        processing_time = time.time() - st.session_state.processing_start_time
        
        if results:
            history_item = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": image_path,
                "overlay_path": overlay_path,
                "register_cropped_path": register_cropped_path,
                "subject_cropped_path": subject_cropped_path,
                "results": results,
                "processing_time": processing_time
            }
            st.session_state.results_history.append(history_item)
        
        return results, register_cropped_path, subject_cropped_path, overlay_path, processing_time

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video processor class for WebRTC
class VideoProcessor:
    def __init__(self):
        self.frame = None
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_count = 0
    
    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        self.frame = apply_filter(self.frame, st.session_state.camera_filters)
        
        if st.session_state.enhance_contrast:
            self.frame = enhance_image_contrast(self.frame)
            
        current_time = time.time()
        self.frame_count += 1
        if (current_time - self.last_frame_time) > 1.0:
            self.fps = self.frame_count / (current_time - self.last_frame_time)
            self.last_frame_time = current_time
            self.frame_count = 0
            
        cv2.putText(self.frame, f"FPS: {self.fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        h, w = self.frame.shape[:2]
        grid_color = (255, 255, 255)
        grid_thickness = 1
        grid_alpha = 0.3
        overlay = self.frame.copy()
        
        for y in range(0, h, h//10):
            cv2.line(overlay, (0, y), (w, y), grid_color, grid_thickness)
        for x in range(0, w, w//10):
            cv2.line(overlay, (x, 0), (x, h), grid_color, grid_thickness)
            
        center_x, center_y = w//2, h//2
        cv2.line(overlay, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
        cv2.line(overlay, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
        
        cv2.addWeighted(overlay, grid_alpha, self.frame, 1 - grid_alpha, 0, self.frame)
        
        scan_pos = int((time.time() % 2) / 2 * h)
        cv2.line(self.frame, (0, scan_pos), (w, scan_pos), (0, 255, 0), 2)
        
        border_color = (0, 120, 255)
        border_width = 5
        cv2.rectangle(self.frame, (border_width, border_width), 
                     (w - border_width, h - border_width), 
                     border_color, border_width)
        
        cv2.putText(self.frame, "Scanning...", (w//2 - 60, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(self.frame, format="bgr24")

# Function to display colored text boxes
def st_success(text):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def st_error(text):
    st.markdown(f'<div class="error-box">{text}</div>', unsafe_allow_html=True)

def st_info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

# Function to display header
def display_header():
    with st.container():
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 5])
        with col1:
            if lottie_scanning:
                st_lottie.st_lottie(lottie_scanning, height=100, key="header_animation", quality="low")  # Optimized
            else:
                st.image("https://img.icons8.com/ios-filled/100/ffffff/scanner.png", width=80)
        with col2:
            st.markdown('<h1>Smart Answer Sheet Scanner</h1>', unsafe_allow_html=True)
            st.markdown('<p>Automatically extract register numbers and subject codes from answer sheets</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Function to add a download button for a file
def get_image_download_button(image_path, filename, button_text):
    try:
        with open(image_path, "rb") as file:
            btn = st.download_button(
                label=button_text,
                data=file,
                file_name=filename,
                mime="image/jpeg"
            )
        return btn
    except Exception as e:
        st_error(f"Failed to create download button: {e}")
        return None

# Function to save results to a file
def save_results_to_file(results, filename="results.txt"):
    try:
        with open(f"results/{filename}", "w") as f:
            for label, value in results:
                f.write(f"{label}: {value}\n")
        return f"results/{filename}"
    except Exception as e:
        st_error(f"Failed to save results: {e}")
        return None

# Streamlit app
def main():
    display_header()
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/scanner.png", width=80)
        st.title("Settings")
        
        st.subheader("Camera Settings")
        st.session_state.camera_filters = st.selectbox(
            "Camera Filter", 
            ["None", "Grayscale", "Adaptive Threshold", "Canny Edge", "High Contrast", "Document Scan"]
        )
        st.session_state.enhance_contrast = st.checkbox("Enhance Contrast", value=st.session_state.enhance_contrast)
        
        st.subheader("About")
        st.info("This application uses computer vision and deep learning to extract information from answer sheets.")
        
        with st.expander("Help"):
            st.markdown("""
            ### How to use:
            1. Select input method (upload or camera)
            2. Capture or upload an image
            3. Click "Extract Information"
            4. View results
            
            ### Tips:
            - Ensure good lighting
            - Keep the answer sheet flat
            - Make sure register number and subject code are clearly visible
            """)
    
    model_files = ["improved_weights.pt", "best_crnn_model.pth", "best_subject_code_model.pth"]
    if not all(os.path.exists(file) for file in model_files):
        st_error("Missing model files: " + ", ".join(file for file in model_files if not os.path.exists(file)))
        st_info("Ensure YOLOv8 weights (.pt) and CRNN weights (.pth) are in the same directory.")
        st.stop()
    
    with st.spinner("Loading models..."):
        try:
            extractor = load_extractor()
            st_success("Models loaded successfully!")
        except Exception as e:
            st_error(f"Failed to load models: {e}")
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
        col1, col2 = st.columns(2)
        with col1:
            if lottie_upload:
                st_lottie.st_lottie(lottie_upload, height=80, key="upload_animation", quality="low")
            else:
                st.markdown("<p style='text-align: center;'>Upload Image</p>", unsafe_allow_html=True)
        with col2:
            if lottie_camera:
                st_lottie.st_lottie(lottie_camera, height=80, key="camera_animation", quality="low")
            else:
                st.markdown("<p style='text-align: center;'>Use Camera</p>", unsafe_allow_html=True)
        
        input_method = st.radio("Choose input method:", ("Upload Image", "Use Camera"), horizontal=True)
        
        reset_col1, reset_col2, reset_col3 = st.columns([1, 1, 1])
        with reset_col2:
            if st.button("Reset", key="reset_btn"):
                st.session_state.image_path = None
                st.session_state.image_captured = False
                st_info("Reset complete. Please capture or upload a new image.")
                st.rerun()  # Replaced experimental_rerun
        
        if input_method == "Upload Image":
            with st.container():
                st.markdown('<div class="camera-container">', unsafe_allow_html=True)
                uploaded_file = st.file_uploader("Upload Answer Sheet Image", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    st.session_state.image_path = f"uploads/image_{uuid.uuid4().hex}.jpg"
                    os.makedirs("uploads", exist_ok=True)
                    with open(st.session_state.image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.image_captured = True
                    
                    if lottie_success:
                        st_lottie.st_lottie(lottie_success, height=80, key="upload_success", quality="low")
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(st.session_state.image_path, caption="Uploaded Image", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="border: 2px dashed #ccc; border-radius: 5px; padding: 20px; text-align: center;">
                        <h3>Drag and drop your answer sheet image here</h3>
                        <p>Supported formats: JPG, PNG, JPEG</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            with st.container():
                st.markdown('<div class="camera-container">', unsafe_allow_html=True)
                if not st.session_state.image_captured:
                    st.markdown("<h3>📸 Live Camera Feed</h3>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                        <p>Position your answer sheet in the frame, ensuring the register number and subject code are clearly visible.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    ctx = webrtc_streamer(
                        key="camera",
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration=RTC_CONFIGURATION,
                        media_stream_constraints={"video": True, "audio": False},
                        video_processor_factory=VideoProcessor,
                        async_processing=True,
                    )
                    
                    # Centered button layout
                    st.markdown('<div class="camera-controls">', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="capture-btn">', unsafe_allow_html=True)
                        capture_btn = st.button("📸 Capture Image", key="capture_btn")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="refresh-btn">', unsafe_allow_html=True)
                        refresh_btn = st.button("🔄 Refresh Camera", key="refresh_btn")
                        st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if ctx.video_processor and capture_btn:
                        if ctx.video_processor.frame is not None:
                            st.session_state.image_path = f"captures/image_{uuid.uuid4().hex}.jpg"
                            os.makedirs("captures", exist_ok=True)
                            cv2.imwrite(st.session_state.image_path, ctx.video_processor.frame)
                            st.session_state.image_captured = True
                            st_success("Image captured successfully!")
                            st.rerun()  # Replaced experimental_rerun
                        else:
                            st_error("No frame captured. Ensure camera is active.")
                    if refresh_btn:
                        st.rerun()  # Replaced experimental_rerun
                else:
                    if lottie_success:
                        st_lottie.st_lottie(lottie_success, height=80, key="capture_success", quality="low")
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(st.session_state.image_path, caption="Captured Image", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.image_path:
            st.markdown('<div class="extract-btn">', unsafe_allow_html=True)
            process_btn = st.button("🔍 Extract Information", key="extract_btn")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if process_btn:
                with st.spinner("Processing image..."):
                    if lottie_processing:
                        st_lottie.st_lottie(lottie_processing, height=100, key="processing_animation", quality="low")
                    
                    try:
                        results, register_cropped, subject_cropped, overlay_path, processing_time = extractor.process_answer_sheet(st.session_state.image_path)
                        
                        # Professional success animation
                        if lottie_success:
                            st_lottie.st_lottie(lottie_success, height=80, key="success_animation", quality="low")
                        st_success(f"Extraction completed in {processing_time:.2f} seconds!")
                        
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.subheader("📋 Extracted Information")
                        
                        if results:
                            for label, value in results:
                                st.markdown(f"**{label}:** `{value}`")
                            
                            results_file = save_results_to_file(results, f"results_{uuid.uuid4().hex}.txt")
                            if results_file:
                                with open(results_file, "rb") as file:
                                    st.download_button(
                                        label="📥 Download Results as Text",
                                        data=file,
                                        file_name="extracted_data.txt",
                                        mime="text/plain"
                                    )
                        else:
                            st_error("No information extracted.")
                            st_info("Try adjusting the image or lowering the confidence threshold.")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.subheader("🔍 Detection Results")
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        image_comparison(
                            img1=st.session_state.image_path,
                            img2=overlay_path,
                            label1="Original",
                            label2="Detections"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        if register_cropped or subject_cropped:
                            st.subheader("✂️ Cropped Regions")
                            col1, col2 = st.columns(2)
                            if register_cropped:
                                with col1:
                                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                                    st.image(register_cropped, caption="Register Number", width=200)
                                    get_image_download_button(register_cropped, "register_number.jpg", "📥 Download")
                                    st.markdown('</div>', unsafe_allow_html=True)
                            if subject_cropped:
                                with col2:
                                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                                    st.image(subject_cropped, caption="Subject Code", width=200)
                                    get_image_download_button(subject_cropped, "subject_code.jpg", "📥 Download")
                                    st.markdown('</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        if lottie_error:
                            st_lottie.st_lottie(lottie_error, height=80, key="error_animation", quality="low")
                        st_error(f"Failed to process image: {e}")
                        st_info("Ensure the image is valid and models are compatible.")
    
    elif selected_tab == "History":
        st.subheader("📜 Extraction History")
        
        if not st.session_state.results_history:
            st.markdown("""
            <div style="text-align: center; padding: 50px; background-color: #f9f9f9; border-radius: 10px;">
                <img src="https://img.icons8.com/ios/100/000000/empty-box.png" width="80">
                <h3>No history yet</h3>
                <p>Previous extraction results will appear here</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, item in enumerate(reversed(st.session_state.results_history)):
                with st.expander(f"Result {i+1} - {item['timestamp']}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if os.path.exists(item['overlay_path']):
                            st.image(item['overlay_path'], caption="Detected Regions", width=300)
                    
                    with col2:
                        st.subheader("Extracted Information")
                        for label, value in item['results']:
                            st.markdown(f"**{label}:** `{value}`")
                        
                        st.markdown(f"**Processing Time:** {item['processing_time']:.2f} seconds")
                        
                        if item['register_cropped_path'] and os.path.exists(item['register_cropped_path']):
                            st.image(item['register_cropped_path'], caption="Register Number", width=150)
                        
                        if item['subject_cropped_path'] and os.path.exists(item['subject_cropped_path']):
                            st.image(item['subject_cropped_path'], caption="Subject Code", width=150)
            
            if st.button("🗑️ Clear History"):
                st.session_state.results_history = []
                st_success("History cleared!")
                st.rerun()  # Replaced experimental_rerun
    
    elif selected_tab == "About":
        st.subheader("📖 About This Application")
        
        st.markdown("""
        <div class="info-box">
        <h3>Smart Answer Sheet Scanner</h3>
        <p>This application uses state-of-the-art computer vision and deep learning techniques to automatically extract information from answer sheets, including register numbers and subject codes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="result-card">
            <h3>🔍 Detection Technology</h3>
            <ul>
                <li>YOLOv8 object detection for locating register numbers and subject codes</li>
                <li>CRNN (Convolutional Recurrent Neural Network) for text recognition</li>
                <li>Advanced image preprocessing for optimal recognition</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="result-card">
            <h3>🛠️ Features</h3>
            <ul>
                <li>Real-time camera capture with visual guides</li>
                <li>Multiple image enhancement filters</li>
                <li>Detailed detection visualization</li>
                <li>Historical tracking of previous scans</li>
                <li>Result download capabilities</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="result-card">
        <h3>📋 How It Works</h3>
        <ol>
            <li><strong>Image Acquisition:</strong> Upload an image or capture one using the camera</li>
            <li><strong>Object Detection:</strong> The YOLOv8 model locates regions containing register numbers and subject codes</li>
            <li><strong>Region Extraction:</strong> These regions are cropped from the original image</li>
            <li><strong>Text Recognition:</strong> CRNN models recognize the text within these regions</li>
            <li><strong>Result Display:</strong> The extracted information is displayed and can be downloaded</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="result-card">
        <h3>🔧 Tips for Best Results</h3>
        <ul>
            <li>Ensure good lighting conditions</li>
            <li>Keep the answer sheet flat and aligned</li>
            <li>Make sure register numbers and subject codes are clearly visible</li>
            <li>Try different camera filters if detection is difficult</li>
            <li>Use the enhance contrast option for better text recognition</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        <p>© 2025 Smart Answer Sheet Scanner | Made with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()