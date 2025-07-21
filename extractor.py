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

@st.cache_resource
def load_extractor():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
        yolo_improved_path = os.path.join(script_dir, "improved_weights.pt")
        yolo_fallback_path = os.path.join(script_dir, "weights.pt")
        register_crnn_path = os.path.join(script_dir, "best_crnn_model.pth")
        subject_crnn_path = os.path.join(script_dir, "best_subject_code_model.pth")

        for p in [yolo_improved_path, yolo_fallback_path, register_crnn_path, subject_crnn_path]:
            if not os.path.exists(p):
                st_warning(f"Model file {p} not found. Creating dummy file for testing. Replace with actual model weights for production use!")
                if p.endswith('.pt'):
                    try:
                        dummy_state = {'model': torch.nn.Module()}
                        torch.save(dummy_state, p)
                    except Exception as e:
                        st_error(f"Failed to create dummy YOLO file {p}: {e}")
                        open(p, 'a').close()
                elif p.endswith('.pth'):
                    try:
                        dummy_model = CRNN(num_classes=11 if 'register' in p else 37)
                        torch.save({'model_state_dict': dummy_model.state_dict()}, p)
                    except Exception as e:
                        st_error(f"Failed to create dummy CRNN file {p}: {e}")
                        open(p, 'a').close()

        extractor = AnswerSheetExtractor(
            yolo_improved_path,
            yolo_fallback_path,
            register_crnn_path,
            subject_crnn_path
        )
        return extractor
    except Exception as e:
        st_error(f"Failed to initialize extractor: {e}")
        st_info("Ensure model files are in the script's directory.")
        return None

class AnswerSheetExtractor:
    def __init__(self, yolo_improved_weights_path, yolo_fallback_weights_path, register_crnn_model_path, subject_crnn_model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
        for dir_name in ["cropped_register_numbers", "cropped_subject_codes", "results", "uploads", "captures"]:
            os.makedirs(os.path.join(script_dir, dir_name), exist_ok=True)
        self.script_dir = script_dir

        try:
            cuda_available = torch.cuda.is_available()
            self.device = torch.device('cuda' if cuda_available else 'cpu')
            if cuda_available:
                st_info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                st_info("CUDA not available. Using CPU.")
        except Exception as e:
            st_warning(f"Error checking CUDA availability: {e}. Falling back to CPU.")
            self.device = torch.device('cpu')

        if not os.path.exists(yolo_improved_weights_path):
            raise FileNotFoundError(f"Improved YOLO weights not found at: {yolo_improved_weights_path}")
        if not os.path.exists(yolo_fallback_weights_path):
            raise FileNotFoundError(f"Fallback YOLO weights not found at: {yolo_fallback_weights_path}")
        try:
            self.yolo_improved_model = YOLO(yolo_improved_weights_path)
            self.yolo_improved_model.to(self.device)
            self.yolo_fallback_model = YOLO(yolo_fallback_weights_path)
            self.yolo_fallback_model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO models: {e}")

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
            st_error(f"Could not load image from {image_path}")
            return [], [], None

        try:
            results = model(image)
        except Exception as e:
            st_error(f"YOLO detection error with {model_name}: {e}")
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

    def select_best_detections(self, improved_results, fallback_results):
        improved_registers, improved_subjects, improved_overlay = improved_results
        fallback_registers, fallback_subjects, fallback_overlay = fallback_results

        best_register = None
        best_subject = None
        best_overlay = improved_overlay

        if improved_registers:
            best_register = max(improved_registers, key=lambda x: x[1])
        if fallback_registers and (not best_register or best_register[1] < max(fallback_registers, key=lambda x: x[1])[1]):
            best_register = max(fallback_registers, key=lambda x: x[1])
            best_overlay = fallback_overlay

        if improved_subjects:
            best_subject = max(improved_subjects, key=lambda x: x[1])
        if fallback_subjects and (not best_subject or best_subject[1] < max(fallback_subjects, key=lambda x: x[1])[1]):
            best_subject = max(fallback_subjects, key=lambda x: x[1])
            best_overlay = fallback_overlay

        return best_register, best_subject, best_overlay

    def extract_text(self, image_path, model, img_transform, char_map):
        try:
            if not os.path.exists(image_path):
                st_error(f"Cropped image not found: {image_path}")
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
            st_error(f"Failed to extract text from {image_path}: {e}")
            return "ERROR"

    def extract_register_number(self, image_path):
        return self.extract_text(image_path, self.register_crnn_model, self.register_transform, self.register_char_map)

    def extract_subject_code(self, image_path):
        return self.extract_text(image_path, self.subject_crnn_model, self.subject_transform, self.subject_char_map)

    def process_answer_sheet(self, image_path):
        st.session_state.processing_start_time = time.time()

        with st.spinner("Detecting regions with improved model..."):
            improved_results = self.detect_regions(image_path, self.yolo_improved_model, "improved")
            improved_registers, improved_subjects, improved_overlay = improved_results

        if not (improved_registers and improved_subjects):
            with st.spinner("Detecting regions with fallback model..."):
                fallback_results = self.detect_regions(image_path, self.yolo_fallback_model, "fallback")
        else:
            fallback_results = ([], [], None)

        best_register, best_subject, best_overlay = self.select_best_detections(improved_results, fallback_results)

        results = []
        best_register_cropped_path = best_register[0] if best_register else None
        best_subject_cropped_path = best_subject[0] if best_subject else None

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
        history_item = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_image_path": image_path,
            "overlay_image_path": best_overlay,
            "register_cropped_path": best_register_cropped_path,
            "subject_cropped_path": best_subject_cropped_path,
            "results": results,
            "processing_time": processing_time
        }
        return results, best_register_cropped_path, best_subject_cropped_path, best_overlay, processing_time, history_item
