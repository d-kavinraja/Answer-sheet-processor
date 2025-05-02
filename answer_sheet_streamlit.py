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

# Custom CSS for styling with theme compatibility and mobile-friendly buttons
def local_css():
    st.markdown("""
    <style>
        /* Theme compatibility: Use Streamlit theme variables */
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Hide header button */
        [data-testid="stHeader"] button {
            display: none !important;
        }

        /* Button styling */
        .stButton>button {
            font-weight: bold;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: 100%;
            font-size: 1.1rem;
        }

        /* Status boxes */
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
        .warning-box {
            background-color: #fff3cd;
            border-color: #ffeeba;
            color: #856404 !important;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }

        /* Result card */
        .result-card {
            background-color: var(--secondary-background-color);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        /* Header container */
        .header-container {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            color: var(--text-color-inverse);
        }
        .header-container h1, .header-container p {
            color: var(--text-color-inverse);
        }

        /* Camera container */
        .camera-container {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 15px;
            background-color: var(--secondary-background-color);
        }

        .image-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Tab content */
        .tab-content {
            padding: 20px;
            border-radius: 0 0 10px 10px;
            background-color: var(--background-color);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* History item */
        .history-item {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            background-color: var(--secondary-background-color);
            cursor: pointer;
            transition: all 0.3s;
            border-left: 5px solid var(--primary-color);
        }
        .history-item:hover {
            filter: brightness(95%);
            transform: translateY(-2px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Footer */
        .footer {
            margin-top: 50px;
            padding: 20px;
            text-align: center;
            font-size: 0.9rem;
            background-color: var(--secondary-background-color);
            border-radius: 10px;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.05);
            width: 100%;
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
        .footer-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }

        /* Camera controls */
        .camera-controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
        }
        .camera-controls .stButton>button {
            padding: 1rem 2rem;
            font-size: 1.2rem;
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: var(--primary-color) !important;
        }

        /* Input buttons column */
        .input-buttons-col {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-bottom: 20px;
            max-width: 250px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Extracted output */
        .extracted-output {
            background-color: var(--secondary-background-color);
            border: 2px solid var(--primary-color);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-family: 'Courier New', Courier, monospace;
            color: var(--text-color);
        }

        /* Image comparison width control */
        .image-comparison-container {
            width: 100%;
            max-width: 600px;
            margin: 0 auto;
        }

        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .footer { padding: 15px; font-size: 0.8rem; }
            .footer-content { flex-direction: column; gap: 8px; }
            .camera-controls { flex-direction: column; gap: 10px; }
            .camera-controls .stButton>button { padding: 0.75rem 1.5rem; font-size: 1rem; }
            .input-buttons-col { max-width: 100%; }
        }
        @media (max-width: 480px) {
            .footer { padding: 10px; font-size: 0.7rem; }
            .camera-controls .stButton>button { padding: 0.5rem 1rem; font-size: 0.9rem; }
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
if 'selected_history_item_index' not in st.session_state:
    st.session_state.selected_history_item_index = None
if 'webrtc_key' not in st.session_state:
    st.session_state.webrtc_key = f"webrtc_{uuid.uuid4().hex}"
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
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
        yolo_improved_path = os.path.join(script_dir, "improved_weights.pt")
        yolo_fallback_path = os.path.join(script_dir, "weights.pt")
        register_crnn_path = os.path.join(script_dir, "best_crnn_model.pth")
        subject_crnn_path = os.path.join(script_dir, "best_subject_code_model_fulldataset.pth")

        # Check for model files
        missing_files = []
        for p in [yolo_improved_path, register_crnn_path, subject_crnn_path]:
            if not os.path.exists(p):
                missing_files.append(p)
        # weights.pt is optional; we'll handle it in AnswerSheetExtractor
        if missing_files:
            st.error(f"Required model files missing: {', '.join(missing_files)}")
            st.info("""
            To deploy on Streamlit Community Cloud:
            1. Ensure all model files (improved_weights.pt, best_crnn_model.pth, best_subject_code_model_fulldataset.pth) are in your GitHub repository's root directory.
            2. Verify that the file paths in the code match the repository structure.
            3. If weights.pt is unavailable, the app will use improved_weights.pt only.
            4. Create a requirements.txt file with all dependencies (e.g., streamlit, torch, opencv-python, ultralytics, etc.).
            5. Redeploy the app after adding the files.
            """)
            return None

        # Check for CRNN model files and create dummy files if missing (for testing)
        for p in [register_crnn_path, subject_crnn_path]:
            if not os.path.exists(p):
                st.warning(f"CRNN model file {p} not found. Creating dummy file for testing. Replace with actual model weights for production use!")
                try:
                    dummy_model = CRNN(num_classes=11 if 'register' in p else 37)
                    torch.save({'model_state_dict': dummy_model.state_dict()}, p)
                except Exception as e:
                    st.error(f"Failed to create dummy CRNN file {p}: {e}")
                    open(p, 'a').close()

        extractor = AnswerSheetExtractor(
            yolo_improved_path,
            yolo_fallback_path if os.path.exists(yolo_fallback_path) else None,
            register_crnn_path,
            subject_crnn_path
        )
        return extractor
    except Exception as e:
        st.error(f"Failed to initialize extractor: {e}")
        st.info("""
        Ensure the following:
        - Model files (improved_weights.pt, best_crnn_model.pth, best_subject_code_model_fulldataset.pth) are in the script's directory.
        - If weights.pt is unavailable, the app will proceed with improved_weights.pt only.
        - All dependencies are listed in requirements.txt.
        - The ultralytics library version is compatible with the YOLO model weights.
        """)
        return None

# AnswerSheetExtractor class
class AnswerSheetExtractor:
    def __init__(self, yolo_improved_weights_path, yolo_fallback_weights_path, register_crnn_model_path, subject_crnn_model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
        for dir_name in ["cropped_register_numbers", "cropped_subject_codes", "results", "uploads", "captures"]:
            os.makedirs(os.path.join(script_dir, dir_name), exist_ok=True)
        self.script_dir = script_dir

        # Robust device selection
        try:
            cuda_available = torch.cuda.is_available()
            self.device = torch.device('cuda' if cuda_available else 'cpu')
            if cuda_available:
                st.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                st.info("CUDA not available. Using CPU.")
        except Exception as e:
            st.warning(f"Error checking CUDA availability: {e}. Falling back to CPU.")
            self.device = torch.device('cpu')

        # Load YOLO models
        if not os.path.exists(yolo_improved_weights_path):
            raise FileNotFoundError(f"Improved YOLO weights not found at: {yolo_improved_weights_path}")
        try:
            self.yolo_improved_model = YOLO(yolo_improved_weights_path)
            self.yolo_improved_model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load improved YOLO model: {e}")
        
        self.yolo_fallback_model = None
        if yolo_fallback_weights_path and os.path.exists(yolo_fallback_weights_path):
            try:
                self.yolo_fallback_model = YOLO(yolo_fallback_weights_path)
                self.yolo_fallback_model.to(self.device)
                st.info("Fallback YOLO model (weights.pt) loaded successfully.")
            except Exception as e:
                st.warning(f"Failed to load fallback YOLO model: {e}. Proceeding with improved_weights.pt only.")
        else:
            st.warning("Fallback YOLO weights (weights.pt) not found. Proceeding with improved_weights.pt only.")

        # Load Register CRNN model
        self.register_crnn_model = CRNN(num_classes=11)
        self.register_crnn_model.to(self.device)
        if not os.path.exists(register_crnn_model_path):
            raise FileNotFoundError(f"Register CRNN model not found at: {register_crnn_model_path}")
        try:
            checkpoint = torch.load(register_crnn_model_path, map_location=self.device)
            self.register_crnn_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        except Exception as e:
            raise RuntimeError(f"Failed to load register CRNN model: {e}")
        self.register_crnn_model.eval()

        # Load Subject CRNN model
        self.subject_crnn_model = CRNN(num_classes=37)
        self.subject_crnn_model.to(self.device)
        if not os.path.exists(subject_crnn_model_path):
            raise FileNotFoundError(f"Subject CRNN model not found at: {subject_crnn_model_path}")
        try:
            checkpoint = torch.load(subject_crnn_model_path, map_location=self.device)
            self.subject_crnn_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
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

        self.register_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}}
        self.subject_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}, **{i: chr(i - 11 + ord('A')) for i in range(11, 37)}}

    def detect_regions(self, image_path, model, model_name):
        image = cv2.imread(image_path)
        if image is None:
            st.error(f"Could not load image from {image_path}")
            return [], [], None

        try:
            results = model(image)
        except Exception as e:
            st.error(f"YOLO detection error with {model_name}: {e}")
            return [], [], None

        detections = results[0].boxes
        classes = results[0].names
        register_regions = []
        subject_regions = []
        overlay = image.copy()

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[

System: It looks like your message was cut off, and the artifact content is incomplete. However, I can still address the issue based on the error message and the context provided. The error indicates that the application fails to initialize the `AnswerSheetExtractor` due to missing model files (`weights.pt`) and issues loading YOLO models. Additionally, you're deploying on Streamlit Community Cloud, which has specific requirements for file inclusion and dependency management.

Since this is an update to the previous artifact, I'll use the same `artifact_id` (`9221469c-96a5-4d7d-a951-6c37249b309c`) and provide a complete, updated version of the code that:
1. Handles missing model files gracefully, especially `weights.pt`, by proceeding with only `improved_weights.pt` if `weights.pt` is unavailable.
2. Improves error handling to prevent crashes and display user-friendly messages with deployment instructions.
3. Maintains the logic to try `improved_weights.pt` first and fall back to `weights.pt` (if available) when detections fail.
4. Includes guidance for Streamlit Community Cloud deployment, such as ensuring model files are in the repository and dependencies are listed in `requirements.txt`.
5. Keeps the CPU fallback logic, as Streamlit Community Cloud does not support CUDA.
6. Preserves all other functionality (e.g., camera input, history tab, UI styling) unchanged.

Below is the updated code. I've modified the `load_extractor` function, `AnswerSheetExtractor` class, and `main` function to address the issues. I've also ensured the artifact is complete and follows the provided guidelines.

<xaiArtifact artifact_id="9221469c-96a5-4d7d-a951-6c37249b309c" artifact_version_id="904031bc-d46d-4d2c-bcc0-869b6362fe9c" title="Smart Answer Sheet Scanner" contentType="text/python">
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

# Custom CSS for styling with theme compatibility and mobile-friendly buttons
def local_css():
    st.markdown("""
    <style>
        /* Theme compatibility: Use Streamlit theme variables */
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }

        /* Hide header button */
        [data-testid="stHeader"] button {
            display: none !important;
        }

        /* Button styling */
        .stButton>button {
            font-weight: bold;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: 100%;
            font-size: 1.1rem;
        }

        /* Status boxes */
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
        .warning-box {
            background-color: #fff3cd;
            border-color: #ffeeba;
            color: #856404 !important;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }

        /* Result card */
        .result-card {
            background-color: var(--secondary-background-color);
            border-radius: 10 interrogate px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        /* Header container */
        .header-container {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            color: var(--text-color-inverse);
        }
        .header-container h1, .header-container p {
            color: var(--text-color-inverse);
        }

        /* Camera container */
        .camera-container {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 15px;
            background-color: var(--secondary-background-color);
        }

        .image-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* Tab content */
        .tab-content {
            padding: 20px;
            border-radius: 0 0 10px 10px;
            background-color: var(--background-color);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        /* History item */
        .history-item {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            background-color: var(--secondary-background-color);
            cursor: pointer;
            transition: all 0.3s;
            border-left: 5px solid var(--primary-color);
        }
        .history-item:hover {
            filter: brightness(95%);
            transform: translateY(-2px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Footer */
        .footer {
            margin-top: 50px;
            padding: 20px;
            text-align: center;
            font-size: 0.9rem;
            background-color: var(--secondary-background-color);
            border-radius: 10px;
            box-shadow: 0 -2px 4px rgba(0,0,0,0.05);
            width: 100%;
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
        .footer-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }

        /* Camera controls */
        .camera-controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
        }
        .camera-controls .stButton>button {
            padding: 1rem 2rem;
            font-size: 1.2rem;
        }

        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: var(--primary-color) !important;
        }

        /* Input buttons column */
        .input-buttons-col {
            display: flex;
            flex-direction: column;
            gap: 15px;
            margin-bottom: 20px;
            max-width: 250px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Extracted output */
        .extracted-output {
            background-color: var(--secondary-background-color);
            border: 2px solid var(--primary-color);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-family: 'Courier New', Courier, monospace;
            color: var(--text-color);
        }

        /* Image comparison width control */
        .image-comparison-container {
            width: 100%;
            max-width: 600px;
            margin: 0 auto;
        }

        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .footer { padding: 15px; font-size: 0.8rem; }
            .footer-content { flex-direction: column; gap: 8px; }
            .camera-controls { flex-direction: column; gap: 10px; }
            .camera-controls .stButton>button { padding: 0.75rem 1.5rem; font-size: 1rem; }
            .input-buttons-col { max-width: 100%; }
        }
        @media (max-width: 480px) {
            .footer { padding: 10px; font-size: 0.7rem; }
            .camera-controls .stButton>button { padding: 0.5rem 1rem; font-size: 0.9rem; }
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
if 'selected_history_item_index' not in st.session_state:
    st.session_state.selected_history_item_index = None
if 'webrtc_key' not in st.session_state:
    st.session_state.webrtc_key = f"webrtc_{uuid.uuid4().hex}"
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
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
        yolo_improved_path = os.path.join(script_dir, "improved_weights.pt")
        yolo_fallback_path = os.path.join(script_dir, "weights.pt")
        register_crnn_path = os.path.join(script_dir, "best_crnn_model.pth")
        subject_crnn_path = os.path.join(script_dir, "best_subject_code_model_fulldataset.pth")

        # Check for required model files
        required_files = [yolo_improved_path, register_crnn_path, subject_crnn_path]
        missing_files = [p for p in required_files if not os.path.exists(p)]
        if missing_files:
            st.error(f"Required model files missing: {', '.join(missing_files)}")
            st.info("""
            To deploy on Streamlit Community Cloud:
            1. Ensure all required model files (improved_weights.pt, best_crnn_model.pth, best_subject_code_model_fulldataset.pth) are in your GitHub repository's root directory.
            2. Verify that the file paths in the code match the repository structure.
            3. weights.pt is optional; the app will proceed with improved_weights.pt if weights.pt is missing.
            4. Create a requirements.txt file with all dependencies (e.g., streamlit, torch, opencv-python, ultralytics, etc.).
            5. Redeploy the app after adding the files.
            """)
            return None

        # Check for optional weights.pt
        if not os.path.exists(yolo_fallback_path):
            st.warning("Optional YOLO weights file (weights.pt) not found. The app will proceed using only improved_weights.pt.")

        # Create dummy CRNN files for testing if missing
        for p in [register_crnn_path, subject_crnn_path]:
            if not os.path.exists(p):
                st.warning(f"CRNN model file {p} not found. Creating dummy file for testing. Replace with actual model weights for production use!")
                try:
                    dummy_model = CRNN(num_classes=11 if 'register' in p else 37)
                    torch.save({'model_state_dict': dummy_model.state_dict()}, p)
                except Exception as e:
                    st.error(f"Failed to create dummy CRNN file {p}: {e}")
                    open(p, 'a').close()

        extractor = AnswerSheetExtractor(
            yolo_improved_path,
            yolo_fallback_path if os.path.exists(yolo_fallback_path) else None,
            register_crnn_path,
            subject_crnn_path
        )
        return extractor
    except Exception as e:
        st.error(f"Failed to initialize extractor: {e}")
        st.info("""
        Ensure the following:
        - Required model files (improved_weights.pt, best_crnn_model.pth, best_subject_code_model_fulldataset.pth) are in the script's directory.
        - If weights.pt is unavailable, the app will proceed with improved_weights.pt only.
        - All dependencies are listed in requirements.txt.
        - The ultralytics library version is compatible with the YOLO model weights.
        """)
        return None

# AnswerSheetExtractor class
class AnswerSheetExtractor:
    def __init__(self, yolo_improved_weights_path, yolo_fallback_weights_path, register_crnn_model_path, subject_crnn_model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
        for dir_name in ["cropped_register_numbers", "cropped_subject_codes", "results", "uploads", "captures"]:
            os.makedirs(os.path.join(script_dir, dir_name), exist_ok=True)
        self.script_dir = script_dir

        # Robust device selection
        try:
            cuda_available = torch.cuda.is_available()
            self.device = torch.device('cuda' if cuda_available else 'cpu')
            if cuda_available:
                st.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                st.info("CUDA not available. Using CPU.")
        except Exception as e:
            st.warning(f"Error checking CUDA availability: {e}. Falling back to CPU.")
            self.device = torch.device('cpu')

        # Load YOLO models
        if not os.path.exists(yolo_improved_weights_path):
            raise FileNotFoundError(f"Improved YOLO weights not found at: {yolo_improved_weights_path}")
        try:
            self.yolo_improved_model = YOLO(yolo_improved_weights_path)
            self.yolo_improved_model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load improved YOLO model: {e}")

        self.yolo_fallback_model = None
        if yolo_fallback_weights_path:
            if not os.path.exists(yolo_fallback_weights_path):
                st.warning("Fallback YOLO weights (weights.pt) not found. Proceeding with improved_weights.pt only.")
            else:
                try:
                    self.yolo_fallback_model = YOLO(yolo_fallback_weights_path)
                    self.yolo_fallback_model.to(self.device)
                    st.info("Fallback YOLO model (weights.pt) loaded successfully.")
                except Exception as e:
                    st.warning(f"Failed to load fallback YOLO model: {e}. Proceeding with improved_weights.pt only.")
        else:
            st.info("No fallback YOLO weights provided. Proceeding with improved_weights.pt only.")

        # Load Register CRNN model
        self.register_crnn_model = CRNN(num_classes=11)
        self.register_crnn_model.to(self.device)
        if not os.path.exists(register_crnn_model_path):
            raise FileNotFoundError(f"Register CRNN model not found at: {register_crnn_model_path}")
        try:
            checkpoint = torch.load(register_crnn_model_path, map_location=self.device)
            self.register_crnn_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        except Exception as e:
            raise RuntimeError(f"Failed to load register CRNN model: {e}")
        self.register_crnn_model.eval()

        # Load Subject CRNN model
        self.subject_crnn_model = CRNN(num_classes=37)
        self.subject_crnn_model.to(self.device)
        if not os.path.exists(subject_crnn_model_path):
            raise FileNotFoundError(f"Subject CRNN model not found at: {subject_crnn_model_path}")
        try:
            checkpoint = torch.load(subject_crnn_model_path, map_location=self.device)
            self.subject_crnn_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
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

        self.register_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}}
        self.subject_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}, **{i: chr(i - 11 + ord('A')) for i in range(11, 37)}}

    def detect_regions(self, image_path, model, model_name):
        image = cv2.imread(image_path)
        if image is None:
            st.error(f"Could not load image from {image_path}")
            return [], [], None

        try:
            results = model(image)
        except Exception as e:
            st.error(f"YOLO detection error with {model_name}: {e}")
            return [], [], None

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

            padding = 10
            padded_x1, padded_y1 = max(0, x1 - padding), max(0, y1 - padding)
            padded_x2, padded_y2 = min(w, x2 + padding), min(h, y2 + padding)
            cropped_region = image[padded_y1:padded_y2, padded_x1:padded_x2]
            save_dir = os.path.join(self.script_dir, "cropped_register_numbers" if label == "RegisterNumber" else "cropped_subject_codes")
            save_path = os.path.join(save_dir, f"{label.lower()}_{model_name}_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(save_path, cropped_region)
            if label == "RegisterNumber" and confidence > 0.2:
                register_regions.append((save_path, confidence))
            elif label == "SubjectCode" and confidence > 0.2:
                subject_regions.append((save_path, confidence))

        overlay_path = os.path.join(self.script_dir, "results", f"detection_overlay_{model_name}_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(overlay_path, overlay)
        return register_regions, subject_regions, overlay_path

    def select_best_detections(self, improved_results, fallback_results=None):
        improved_registers, improved_subjects, improved_overlay = improved_results
        best_register = None
        best_subject = None
        best_overlay = improved_overlay

        # If no fallback model, use improved model results directly
        if not fallback_results:
            if improved_registers:
                best_register = max(improved_registers, key=lambda x: x[1])
            if improved_subjects:
                best_subject = max(improved_subjects, key=lambda x: x[1])
            return best_register, best_subject, best_overlay

        # Compare with fallback model
        fallback_registers, fallback_subjects, fallback_overlay = fallback_results

        # Select best register number
        if improved_registers:
            best_register = max(improved_registers, key=lambda x: x[1])
        if fallback_registers and (not best_register or best_register[1] < max(fallback_registers, key=lambda x: x[1])[1]):
            best_register = max(fallback_registers, key=lambda x: x[1])
            best_overlay = fallback_overlay

        # Select best subject code
        if improved_subjects:
            best_subject = max(improved_subjects, key=lambda x: x[1])
        if fallback_subjects and (not best_subject or best_subject[1] < max(fallback_subjects, key=lambda x: x[1])[1]):
            best_subject = max(fallback_subjects, key=lambda x: x[1])
            best_overlay = fallback_overlay

        return best_register, best_subject, best_overlay

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
                        result.append(char_map.get(s, '?'))
                    prev = s
            return ''.join(result)
        except Exception as e:
            st.error(f"Failed to extract text from {image_path}: {e}")
            return "ERROR"

    def extract_register_number(self, image_path):
        return self.extract_text(image_path, self.register_crnn_model, self.register_transform, self.register_char_map)

    def extract_subject_code(self, image_path):
        return self.extract_text(image_path, self.subject_crnn_model, self.subject_transform, self.subject_char_map)

    def process_answer_sheet(self, image_path):
        st.session_state.processing_start_time = time.time()

        # Step 1: Try improved model
        with st.spinner("Detecting regions with improved model..."):
            improved_results = self.detect_regions(image_path, self.yolo_improved_model, "improved")
            improved_registers, improved_subjects, improved_overlay = improved_results

        # Step 2: If either register or subject is not detected and fallback model exists, try fallback model
        fallback_results = None
        if self.yolo_fallback_model and not (improved_registers and improved_subjects):
            with st.spinner("Detecting regions with fallback model..."):
                fallback_results = self.detect_regions(image_path, self.yolo_fallback_model, "fallback")

        # Step 3: Select best detections
        best_register, best_subject, best_overlay = self.select_best_detections(improved_results, fallback_results)

арда        results = []
        best_register_cropped_path = best_register[0] if best_register else None
        best_subject_cropped_path = best_subject[0] if best_subject else None

        # Step 4: Proceed with extraction for the best detections
        if best_register:
            with st.spinner("Extracting Register Number..."):
                register_number = self.extract_register_number(best_register_cropped_path)
            results.append(("Register Number", register_number))
            st_success(f"Register Number detected (Confidence: {best_register[1]:.2f}). Extracted: '{register_number}'")
        else:
            st_warning("No RegisterNumber regions detected with either model.")

        if best_subject:
            with st.spinner("Extracting Subject Code..."):
                subject_code = self.extract_subject_code(best_subject_cropped_path)
            results.append(("Subject Code", subject_code))
            st_success(f"Subject Code detected (Confidence: {best_subject[1]:.2f}). Extracted: '{subject_code}'")
        else:
            st_warning("No SubjectCode regions detected with either model.")

        processing_time = time.time() - st.session_state.processing_start_time
        if results or best_overlay:
            history_item = {
                "timestamp": datetime.now().strftime("%Y-%m-d %H:%M:%S"),
                "original_image_path": image_path,
                "overlay_image_path": best_overlay,
                "register_cropped_path": best_register_cropped_path,
                "subject_cropped_path": best_subject_cropped_path,
                "results": results,
                "processing_time": processing_time
            }
            st.session_state.results_history.insert(0, history_item)

        return results, best_register_cropped_path, best_subject_cropped_path, best_overlay, processing_time

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
        self.process_interval = 0.05

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        current_time = time.time()
        img = frame.to_ndarray(format="bgr24")
        self.frame = img  # Always update frame to ensure it's not None
        self.last_processed = current_time
        self.frame_count += 1
        if current_time - self.last_frame_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_frame_time)
            self.last_frame_time = current_time
            self.frame_count = 0

        cv2.putText(img, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        h, w = img.shape[:2]
        center_x, center_y = w//2, h//2
        cv2.line(img, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
        cv2.line(img, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)
        cv2.putText(img, "Align Sheet & Capture", (center_x - 100, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Colored text boxes
def st_success(text):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def st_error(text):
    st.markdown(f'<div class="error-box">{text}</div>', unsafe_allow_html=True)

def st_info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def st_warning(text):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)

# Header display
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

# Download button helper
def get_image_download_button(image_path, filename, button_text):
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as file:
                return st.download_button(
                    label=button_text,
                    data=file,
                    file_name=filename,
                    mime="image/jpeg",
                    key=f"download_{filename.replace('.', '_')}_{uuid.uuid4().hex}"
                )
        except Exception as e:
            st_error(f"Failed to create download button for {filename}: {e}")
    return None

# Save results helper
def save_results_to_file(results, filename_prefix="results"):
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    try:
        filename = f"{filename_prefix}_{uuid.uuid4().hex}.txt"
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "w") as f:
            for label, value in results:
                f.write(f"{label}: {value}\n")
        return filepath
    except Exception as e:
        st_error(f"Failed to save results to {filepath}: {e}")
        return None

# Main app
def main():
    display_header()

    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    model_files = ["improved_weights.pt", "best_crnn_model.pth", "best_subject_code_model_fulldataset.pth"]
    model_paths = [os.path.join(script_dir, f) for f in model_files]

    with st.spinner("Loading models..."):
        extractor = load_extractor()
        if extractor:
            st_success("Models loaded successfully!")
        else:
            st_error("Failed to load models. Please check the error messages above for deployment instructions.")
            st.stop()

    selected_tab = option_menu(
        menu_title=None,
        options=["Scan", "History", "About"],
        icons=["camera", "clock-history", "info-circle"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "var(--secondary-background-color)", "border-radius": "10px"},
            "icon": {"color": "var(--primary-color)", "font-size": "16px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "padding": "10px",
                "color": "var(--text-color)"
            },
            "nav-link-selected": {"background-color": "var(--primary-color)", "color": "var(--text-color-inverse)"}
        }
    )

    if selected_tab == "Scan":
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.markdown("<h3>Choose input method:</h3>", unsafe_allow_html=True)

        st.markdown('<div class="input-buttons-col">', unsafe_allow_html=True)
        if st.button("⬆️ Upload Image", key="upload_image_btn"):
            st.session_state.input_method = "Upload Image"
            st.session_state.image_path = None
            st.session_state.image_captured = False
            st.session_state.selected_history_item_index = None
            st.rerun()
        if st.button("📸 Use Camera", key="use_camera_btn"):
            st.session_state.input_method = "Use Camera"
            st.session_state.image_path = None
            st.session_state.image_captured = False
            st.session_state.selected_history_item_index = None
            st.session_state.webrtc_key = f"webrtc_{uuid.uuid4().hex}"
            st.rerun()
        if st.button("🔄 Reset Scan", key она="reset_btn_scan"):
            st.session_state.image_path = None
            st.session_state.image_captured = False
            st.session_state.selected_history_item_index = None
            st.session_state.webrtc_key = f"webrtc_{uuid.uuid4().hex}"
            st.session_state.input_method = "Upload Image"
            st_info("Scan reset. Upload an image or use the camera.")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.input_method == "Upload Image":
            with st.container():
                st.markdown('<div class="camera-container">', unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Upload Answer Sheet Image",
                    type=["png", "jpg", "jpeg"],
                    key="uploader",
                    label_visibility="collapsed"
                )
                if uploaded_file:
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    uploads_dir = os.path.join(script_dir, "uploads")
                    os.makedirs(uploads_dir, exist_ok=True)
                    temp_path = os.path.join(uploads_dir, f"image_{uuid.uuid4().hex}.{file_extension}")
                    try:
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.session_state.image_path = temp_path
                        st.session_state.image_captured = True
                        st.session_state.selected_history_item_index = None
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(st.session_state.image_path, caption="Uploaded Image", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st_error(f"Error saving uploaded file: {e}")
                        st.session_state.image_path = None
                        st.session_state.image_captured = False
                elif not st.session_state.image_path or not st.session_state.image_captured:
                    st.markdown("""
                    <div style="border: 2px dashed #ccc; border-radius: 5px; padding: 40px 20px; margin-top: 10px;">
                        <h3>Drag & drop or click to upload</h3>
                        <p>Supported formats: JPG, PNG, JPEG</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        else:  # Use Camera
            with st.container():
                st.markdown('<div class="camera-container">', unsafe_allow_html=True)
                if not st.session_state.image_captured:
                    st.markdown("<h4>📸 Live Camera Feed</h4>", unsafe_allow_html=True)
                    st_info("Position the answer sheet within the frame and click 'Capture Image'.")

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
                    if st.button("📸 Capture Image", key="capture_btn", disabled=capture_btn_disabled):
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
                    results, register_cropped, subject_cropped, overlay_path, processing_time = extractor.process_answer_sheet(st.session_state.image_path)
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
                            get_image_download_button(register_cropped, "register_number_crop.jpg", "Download Register Crop")
                        if subject_cropped and os.path.exists(subject_cropped):
                            st.markdown('<div class="image-container">', unsafe_allow_html=True)
                            st.image(subject_cropped, caption="Subject Code", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            get_image_download_button(subject_cropped, 'subject_code_crop.jpg', 'Download Subject Crop')
                        if not register_cropped and not subject_cropped:
                            st_info("No regions cropped.")
                except Exception as e:
                    progress_bar.empty()
                    status_placeholder.empty()
                    st_error(f"An unexpected error occurred during processing: {e}")
                    st_info("Please try again with a different image.")
        st.markdown('</div>', unsafe_allow_html=True)

    elif selected_tab == "History":
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("📜 Processing History")

        if not st.session_state.results_history:
            st_info("No processing history yet. Scan an answer sheet on the 'Scan' tab to populate history.")
        else:
            st.markdown("Click 'View Details' to see the images and full results for a past scan.")
            for i, item in enumerate(st.session_state.results_history):
                timestamp = item.get("timestamp", "N/A")
                results_summary = ", ".join([f"{label}: `{value}`" for label, value in item.get("results", [])])
                if not results_summary:
                    results_summary = "N/A"
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

            if 'selected_history_item_index' in st.session_state and st.session_state.selected_history_item_index is not None:
                st.subheader("📜 Detailed History View")
                try:
                    selected_item = st.session_state.results_history[st.session_state.selected_history_item_index]
                except IndexError:
                    st_error("Selected history item not found. It might have been cleared.")
                    st.session_state.selected_history_item_index = None
                    st.rerun()

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"<h6>Scan Timestamp: {selected_item.get('timestamp', 'N/A')}</h6>", unsafe_allow_html=True)
                st.markdown(f"<p>Processing Time: {selected_item.get('processing_time', 0):.2f} seconds</p>", unsafe_allow_html=True)

                st.markdown("<h6>Extracted Information:</h6>", unsafe_allow_html=True)
                if selected_item.get("results"):
                    st.markdown('<div class="extracted-output">', unsafe_allow_html=True)
                    for label, value in selected_item["results"]:
                        st.markdown(f"**{label}:** `{value}`")
                    st.markdown('</div>', unsafe_allow_html=True)
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
                    else:
                        st_warning("Original or detection overlay image not found.")

                with hist_img_cols[1]:
                    st.markdown("<u>Cropped Regions:</u>", unsafe_allow_html=True)
                    if register_cropped_path and os.path.exists(register_cropped_path):
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(register_cropped_path, caption="Register Number (Cropped)", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("<p>No Register Number crop.</p>", unsafe_allow_html=True)

                    if subject_cropped_path and os.path.exists(subject_cropped_path):
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(subject_cropped_path, caption="Subject Code (Cropped)", use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("<p>No Subject Code crop.</p>", unsafe_allow_html=True)

                if st.button("Hide Details", key="hide_history_details"):
                    st.session_state.selected_history_item_index = None
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    elif selected_tab == "About":
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("ℹ️ About the Smart Answer Sheet Scanner")

        col_about1, col_about2 = st.columns([1, 3])
        with col_about1:
            st.markdown('<div style="font-size: 100px; text-align: center; padding-top: 20px;">🧐</div>', unsafe_allow_html=True)
        with col_about2:
            st.markdown("""
            <p>This application leverages computer vision models to automatically detect and extract Register Numbers and Subject Codes from scanned or photographed answer sheets.</p>
            <h6>Key Technologies Used:</h6>
            <ul>
                <li><b>Object Detection:</b> A custom-trained YOLOv8 model identifies the locations of the relevant fields (Register Number, Subject Code) on the sheet.</li>
                <li><b>Text Recognition (OCR):</b> Convolutional Recurrent Neural Network (CRNN) models are employed to read the characters within the detected regions. Separate CRNN models are optimized for recognizing digits (Register Number) and alphanumeric characters (Subject Code).</li>
                <li><b>Web Interface:</b> Built with Streamlit, providing an interactive user interface for image upload, camera capture, and results visualization.</li>
            </ul>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h6>How to Use:</h6>", unsafe_allow_html=True)
        st.markdown("""
        <ol>
            <li>Navigate to the <b>Scan</b> tab.</li>
            <li>Choose your input method: <b>Upload Image</b> or <b>Use Camera</b>.</li>
            <li>If uploading, select a clear image of the answer sheet.</li>
            <li>If using the camera, position the sheet clearly and click <b>Capture Image</b>.</li>
            <li>Once an image is loaded or captured, click <b>Extract Information</b>.</li>
            <li>View the extracted text, detection overlays, and cropped regions.</li>
            <li>Check the <b>History</b> tab to review past scans.</li>
        </ol>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h6>Model Information:</h6>", unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li>The models require specific weights files (<code>improved_weights.pt</code>, <code>best_crnn_model.pth</code>, <code>best_subject_code_model_fulldataset.pth</code>) to be present in the same directory as the script.</li>
            <li><code>weights.pt</code> is optional; the app will use <code>improved_weights.pt</code> if <code>weights.pt</code> is missing.</li>
            <li>Accuracy is dependent on the quality of the input image (clarity, lighting, angle) and the training data used for the models.</li>
            <li>If CRNN model files are missing, dummy files are created for testing. Replace them with trained model weights for production use.</li>
        </ul>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h6>Disclaimer:</h6>", unsafe_allow_html=True)
        st_warning("This tool is for demonstration or assistive purposes. Extracted results should always be verified for accuracy, especially in critical applications.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown('<div class="footer-content">', unsafe_allow_html=True)
    st.markdown("<p>© 2025 Smart Scanner Project. Built with Streamlit.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()