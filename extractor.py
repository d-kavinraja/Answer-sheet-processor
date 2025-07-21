import streamlit as st
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from models import CRNN
from utils import st_success, st_warning, st_error
import uuid
import time
from datetime import datetime

@st.cache_resource
def load_extractor():
    """Load and cache the YOLO and CRNN models."""
    try:
        return AnswerSheetExtractor("improved_weights.pt", "weights.pt", "best_crnn_model.pth", "best_subject_code_model.pth")
    except Exception as e:
        st_error(f"Failed to initialize extractor: {e}")
        return None

class AnswerSheetExtractor:
    def __init__(self, yolo_improved_path, yolo_fallback_path, register_crnn_path, subject_crnn_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.yolo_improved_model = YOLO(yolo_improved_path)
        self.yolo_fallback_model = YOLO(yolo_fallback_path)
        
        self.register_crnn_model = self._load_crnn(register_crnn_path, 11)
        self.subject_crnn_model = self._load_crnn(subject_crnn_path, 37)

        self.register_transform = self._get_transform(size=(32, 256))
        self.subject_transform = self._get_transform(size=(32, 128))

        self.register_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}}
        self.subject_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}, **{i: chr(i - 11 + ord('A')) for i in range(11, 37)}}

        for dir_name in ["cropped_register_numbers", "cropped_subject_codes", "results"]:
            os.makedirs(dir_name, exist_ok=True)

    def _load_crnn(self, path, num_classes):
        model = CRNN(num_classes=num_classes).to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        model.eval()
        return model

    def _get_transform(self, size):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def detect_regions(self, image_path, model):
        """Detect regions using a YOLO model."""
        image = cv2.imread(image_path)
        detections = model(image)[0].boxes
        classes = model(image)[0].names
        
        register_regions, subject_regions = [], []
        overlay = image.copy()

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf, cls_id = float(box.conf[0]), int(box.cls[0])
            label = classes[cls_id]
            
            color = (0, 255, 0) if "Register" in label else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cropped_region = image[y1:y2, x1:x2]
            save_path = f"cropped_{label.lower()}_{uuid.uuid4().hex}.jpg"
            if "Register" in label:
                cv2.imwrite(os.path.join("cropped_register_numbers", save_path), cropped_region)
                register_regions.append((os.path.join("cropped_register_numbers", save_path), conf))
            else:
                cv2.imwrite(os.path.join("cropped_subject_codes", save_path), cropped_region)
                subject_regions.append((os.path.join("cropped_subject_codes", save_path), conf))

        overlay_path = os.path.join("results", f"overlay_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(overlay_path, overlay)
        return register_regions, subject_regions, overlay_path
    
    def extract_text(self, image_path, model, transform, char_map):
        """Extract text from a cropped image using a CRNN model."""
        image = Image.open(image_path).convert('L')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = model(image_tensor).squeeze(1).softmax(1).argmax(1).cpu().numpy()
        
        prev = 0
        result = []
        for s in output:
            if s != 0 and s != prev:
                result.append(char_map.get(s, '?'))
            prev = s
        return ''.join(result)

    def process_answer_sheet(self, image_path):
        """Full pipeline to process an answer sheet image."""
        start_time = time.time()
        
        # Try improved model first
        reg, subj, overlay = self.detect_regions(image_path, self.yolo_improved_model)
        # If detection is poor, try fallback model
        if not (reg and subj):
            reg_fb, subj_fb, overlay_fb = self.detect_regions(image_path, self.yolo_fallback_model)
            if len(reg_fb) > len(reg): reg, overlay = reg_fb, overlay_fb
            if len(subj_fb) > len(subj): subj, overlay = subj_fb, overlay_fb

        results = []
        reg_path, subj_path = None, None
        
        if reg:
            reg_path, conf = max(reg, key=lambda item: item[1])
            text = self.extract_text(reg_path, self.register_crnn_model, self.register_transform, self.register_char_map)
            results.append(("Register Number", text))
            st_success(f"Register Number Extracted: '{text}' (Confidence: {conf:.2f})")
        
        if subj:
            subj_path, conf = max(subj, key=lambda item: item[1])
            text = self.extract_text(subj_path, self.subject_crnn_model, self.subject_transform, self.subject_char_map)
            results.append(("Subject Code", text))
            st_success(f"Subject Code Extracted: '{text}' (Confidence: {conf:.2f})")
        
        history_item = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overlay_image_path": overlay,
            "results": results
        }
        return results, reg_path, subj_path, overlay, time.time() - start_time, history_item
