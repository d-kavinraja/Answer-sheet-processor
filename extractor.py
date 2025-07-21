# extractor.py

import streamlit as st
import os
import torch
import cv2
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from models import CRNN
from utils import st_success, st_warning
import uuid
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)

@st.cache_resource
def load_extractor():
    """Load and cache the YOLO and CRNN models. This function runs only once."""
    try:
        return AnswerSheetExtractor(
            "improved_weights.pt",
            "weights.pt",
            "best_crnn_model.pth",
            "best_subject_code_model.pth"
        )
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}", exc_info=True)
        st.error(f"Failed to initialize extractor: {e}")
        return None

class AnswerSheetExtractor:
    def __init__(self, yolo_improved_path, yolo_fallback_path, register_crnn_path, subject_crnn_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Load YOLO models
        self.yolo_improved_model = YOLO(yolo_improved_path)
        self.yolo_fallback_model = YOLO(yolo_fallback_path)

        # Load CRNN models with the corrected architecture
        self.register_crnn_model = self._load_crnn(register_crnn_path, 11)
        self.subject_crnn_model = self._load_crnn(subject_crnn_path, 37)

        # Define image transforms
        self.register_transform = self._get_transform(size=(32, 256))
        self.subject_transform = self._get_transform(size=(32, 128))

        # Define character maps for decoding
        self.register_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}}
        self.subject_char_map = {0: '', **{i: str(i-1) for i in range(1, 11)}, **{i: chr(i - 11 + ord('A')) for i in range(11, 37)}}

        # Ensure output directories exist
        for dir_name in ["cropped_register_numbers", "cropped_subject_codes", "results"]:
            os.makedirs(dir_name, exist_ok=True)

    def _load_crnn(self, path, num_classes):
        """Helper to load a CRNN model and its weights."""
        model = CRNN(num_classes=num_classes).to(self.device)
        # Load weights, assuming they might be directly the state_dict or nested
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _get_transform(self, size):
        return transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _detect_regions(self, image_path, model):
        """Detect regions using a given YOLO model."""
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
            
            cropped_region = image[y1:y2, x1:x2]
            if "Register" in label:
                register_regions.append((cropped_region, conf))
            else:
                subject_regions.append((cropped_region, conf))

        return register_regions, subject_regions, overlay

    def _extract_text(self, image_array, model, transform, char_map):
        """Extract text from a cropped image array using a CRNN model."""
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)).convert('L')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = model(image_tensor).squeeze(1).softmax(2).argmax(2).permute(1, 0)
        
        prev = 0
        result = []
        for s in output[0]:
            if s != 0 and s != prev:
                result.append(char_map.get(s.item(), '?'))
            prev = s
        return ''.join(result)

    def process_answer_sheet(self, image_path):
        """Main pipeline to process an answer sheet image."""
        start_time = time.time()
        
        reg, subj, overlay = self._detect_regions(image_path, self.yolo_improved_model)
        if not (reg and subj):
            reg_fb, subj_fb, overlay_fb = self._detect_regions(image_path, self.yolo_fallback_model)
            if len(reg_fb) > len(reg): reg, overlay = reg_fb, overlay_fb
            if len(subj_fb) > len(subj): subj, overlay = subj_fb, overlay_fb

        results, reg_path, subj_path = [], None, None
        
        if reg:
            best_reg_crop, conf = max(reg, key=lambda item: item[1])
            reg_path = os.path.join("cropped_register_numbers", f"reg_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(reg_path, best_reg_crop)
            text = self._extract_text(best_reg_crop, self.register_crnn_model, self.register_transform, self.register_char_map)
            results.append(("Register Number", text))
            st_success(f"Register Number Extracted: '{text}'")
        
        if subj:
            best_subj_crop, conf = max(subj, key=lambda item: item[1])
            subj_path = os.path.join("cropped_subject_codes", f"subj_{uuid.uuid4().hex}.jpg")
            cv2.imwrite(subj_path, best_subj_crop)
            text = self._extract_text(best_subj_crop, self.subject_crnn_model, self.subject_transform, self.subject_char_map)
            results.append(("Subject Code", text))
            st_success(f"Subject Code Extracted: '{text}'")

        if not results:
            st_warning("No scannable regions were detected. Please try a different image.")

        overlay_path = os.path.join("results", f"overlay_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(overlay_path, overlay)

        history_item = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overlay_image_path": overlay_path,
            "results": results,
            "processing_time": time.time() - start_time
        }
        return results, reg_path, subj_path, overlay_path, time.time() - start_time, history_item
