# extractor.py
import streamlit as st
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import uuid
import time
from datetime import datetime
from utils import st_success, st_warning, st_error, st_info
from config import (
    YOLO_IMPROVED_PATH, YOLO_FALLBACK_PATH, REGISTER_CRNN_PATH, SUBJECT_CRNN_PATH,
    UPLOADS_DIR, CAPTURES_DIR, CROPPED_REG_DIR, CROPPED_SUB_DIR, RESULTS_DIR
)

# Define CRNN model
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)), nn.Dropout2d(0.3),
            nn.Conv2d(512, 512, kernel_size=(2, 1)), nn.BatchNorm2d(512), nn.ReLU(),
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

class AnswerSheetExtractor:
    def __init__(self, yolo_improved_weights, yolo_fallback_weights, register_crnn_model, subject_crnn_model):
        for dir_path in [CROPPED_REG_DIR, CROPPED_SUB_DIR, RESULTS_DIR, UPLOADS_DIR, CAPTURES_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda_available else 'cpu')
        
        self.yolo_improved_model = YOLO(yolo_improved_weights)
        self.yolo_fallback_model = YOLO(yolo_fallback_weights)
        
        self.register_crnn_model = register_crnn_model
        self.subject_crnn_model = subject_crnn_model
        
        self.register_crnn_model.to(self.device)
        self.subject_crnn_model.to(self.device)
        self.yolo_improved_model.to(self.device)
        self.yolo_fallback_model.to(self.device)

        self.register_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 256)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])
        self.subject_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 128)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])
        self.register_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}}
        self.subject_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}, **{i: chr(i - 11 + ord('A')) for i in range(11, 37)}}

    def detect_regions(self, image_path, model, model_name):
        image = cv2.imread(image_path)
        if image is None: return [], [], None
        
        results = model(image)
        detections = results[0].boxes
        classes = results[0].names
        register_regions, subject_regions = [], []
        overlay = image.copy()

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = classes[class_id]
            
            color = (0, 255, 0) if label == "RegisterNumber" else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cropped_region = image[y1:y2, x1:x2]
            save_dir = CROPPED_REG_DIR if label == "RegisterNumber" else CROPPED_SUB_DIR
            save_path = os.path.join(save_dir, f"{label.lower()}_{model_name}_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(save_path, cropped_region)
            
            if label == "RegisterNumber":
                register_regions.append((save_path, confidence))
            elif label == "SubjectCode":
                subject_regions.append((save_path, confidence))

        overlay_path = os.path.join(RESULTS_DIR, f"detection_overlay_{model_name}_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(overlay_path, overlay)
        return register_regions, subject_regions, overlay_path

    def select_best_detections(self, improved_results, fallback_results):
        improved_registers, improved_subjects, improved_overlay = improved_results
        fallback_registers, fallback_subjects, fallback_overlay = fallback_results
        
        best_register = max(improved_registers, key=lambda x: x[1]) if improved_registers else None
        if fallback_registers and (not best_register or best_register[1] < max(fallback_registers, key=lambda x: x[1])[1]):
            best_register = max(fallback_registers, key=lambda x: x[1])

        best_subject = max(improved_subjects, key=lambda x: x[1]) if improved_subjects else None
        if fallback_subjects and (not best_subject or best_subject[1] < max(fallback_subjects, key=lambda x: x[1])[1]):
            best_subject = max(fallback_subjects, key=lambda x: x[1])
            
        final_overlay = improved_overlay if improved_registers and improved_subjects else fallback_overlay or improved_overlay
        return best_register, best_subject, final_overlay

    def extract_text(self, image_path, model, img_transform, char_map):
        image = Image.open(image_path).convert('L')
        image_tensor = img_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = model(image_tensor).squeeze(1)
            output = output.softmax(1).argmax(1)
            seq = output.cpu().numpy()
            prev, result = 0, []
            for s in seq:
                if s != 0 and s != prev: result.append(char_map.get(s, '?'))
                prev = s
        return ''.join(result)

    def extract_register_number(self, image_path):
        return self.extract_text(image_path, self.register_crnn_model, self.register_transform, self.register_char_map)

    def extract_subject_code(self, image_path):
        return self.extract_text(image_path, self.subject_crnn_model, self.subject_transform, self.subject_char_map)

    def process_answer_sheet(self, image_path):
        st.session_state.processing_start_time = time.time()
        
        with st.spinner("Detecting regions with improved model..."):
            improved_results = self.detect_regions(image_path, self.yolo_improved_model, "improved")
        
        fallback_results = ([], [], None)
        if not (improved_results[0] and improved_results[1]):
            with st.spinner("Detecting regions with fallback model..."):
                fallback_results = self.detect_regions(image_path, self.yolo_fallback_model, "fallback")
        
        best_register, best_subject, best_overlay = self.select_best_detections(improved_results, fallback_results)

        results = []
        best_register_cropped_path = best_register[0] if best_register else None
        best_subject_cropped_path = best_subject[0] if best_subject else None

        if best_register:
            with st.spinner("Extracting Register Number..."):
                register_number = self.extract_register_number(best_register_cropped_path)
            results.append(("Register Number", register_number))
            st_success(f"Register Number detected. Extracted: '{register_number}'")
        else:
            st_warning("No RegisterNumber regions detected.")

        if best_subject:
            with st.spinner("Extracting Subject Code..."):
                subject_code = self.extract_subject_code(best_subject_cropped_path)
            results.append(("Subject Code", subject_code))
            st_success(f"Subject Code detected. Extracted: '{subject_code}'")
        else:
            st_warning("No SubjectCode regions detected.")

        processing_time = time.time() - st.session_state.processing_start_time
        history_item = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_image_path": image_path,
            "overlay_image_path": best_overlay,
            "register_cropped_path": best_register_cropped_path,
            "subject_cropped_path": best_subject_cropped_path,
            "results": results,
            "processing_time": processing_time
        }
        st.session_state.results_history.insert(0, history_item)
        
        return results, best_register_cropped_path, best_subject_cropped_path, best_overlay, processing_time


# Cached function to load all models
@st.cache_resource
def load_all_models():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load Register CRNN Model
        register_crnn = CRNN(num_classes=11)
        checkpoint_reg = torch.load(REGISTER_CRNN_PATH, map_location=device)
        register_crnn.load_state_dict(checkpoint_reg.get('model_state_dict', checkpoint_reg))
        register_crnn.eval()

        # Load Subject CRNN Model
        subject_crnn = CRNN(num_classes=37)
        checkpoint_sub = torch.load(SUBJECT_CRNN_PATH, map_location=device)
        subject_crnn.load_state_dict(checkpoint_sub.get('model_state_dict', checkpoint_sub))
        subject_crnn.eval()
        
        # Initialize the main extractor class with loaded models
        extractor = AnswerSheetExtractor(
            YOLO_IMPROVED_PATH,
            YOLO_FALLBACK_PATH,
            register_crnn,
            subject_crnn
        )
        return extractor
    except FileNotFoundError as e:
        st_error(f"A required model file was not found: {e}. Please ensure all .pt and .pth files are in the root directory.")
        return None
    except Exception as e:
        st_error(f"An error occurred while loading models: {e}")
        return None