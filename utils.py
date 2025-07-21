# utils.py
import streamlit as st
import os
import uuid
import PyPDF2
import logging

logger = logging.getLogger(__name__)

# --- STYLING AND UI HELPERS ---

def local_css():
    st.markdown("""
    <style>
        .stApp { max-width: 1200px; margin: 0 auto; }
        [data-testid="stHeader"] { visibility: hidden; }
        .stButton>button {
            font-weight: 500; border-radius: 10px; padding: 0.75rem 1.5rem; transition: all 0.3s;
            cursor: pointer; display: inline-flex; align-items: center; justify-content: center;
            gap: 8px; width: 100%; font-size: 1.1rem;
        }
        .success-box { background-color: #d4edda; border-color: #c3e6cb; color: #155724 !important; padding: 1rem; border-radius: 0.25rem; margin-bottom: 1rem; }
        .error-box { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24 !important; padding: 1rem; border-radius: 0.25rem; margin-bottom: 1rem; }
        .info-box { background-color: #cce5ff; border-color: #b8daff; color: #004085 !important; padding: 1rem; border-radius: 0.25rem; margin-bottom: 1rem; }
        .warning-box { background-color: #fff3cd; border-color: #ffeeba; color: #856404 !important; padding: 1rem; border-radius: 0.25rem; margin-bottom: 1rem; }
        .result-card { background-color: var(--secondary-background-color); border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .header-container { background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%); padding: 20px; border-radius: 10px; margin-bottom: 30px; color: white; }
        .header-container h1, .header-container p { color: white; }
        .camera-container { border: 2px dashed #ccc; border-radius: 10px; padding: 15px; background-color: var(--secondary-background-color); }
        .image-container { border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .tab-content { padding: 20px; border-radius: 0 0 10px 10px; background-color: var(--background-color); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .history-item { padding: 15px; border-radius: 8px; margin-bottom: 15px; background-color: var(--secondary-background-color); cursor: pointer; transition: all 0.3s; border-left: 5px solid var(--primary-color); }
        .history-item:hover { filter: brightness(95%); transform: translateY(-2px); box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .footer { margin-top: 50px; padding: 20px; text-align: center; font-size: 0.9rem; background-color: var(--secondary-background-color); border-radius: 10px; box-shadow: 0 -2px 4px rgba(0,0,0,0.05); width: 100%; }
    </style>
    """, unsafe_allow_html=True)

def st_success(text):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)

def st_error(text):
    st.markdown(f'<div class="error-box">{text}</div>', unsafe_allow_html=True)

def st_info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def st_warning(text):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)


# --- FILE HANDLING HELPERS ---

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

def save_results_to_file(results, filename_prefix="results", results_dir="results"):
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

def fallback_extract_text(pdf_file):
    try:
        pdf_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        if len(pdf_reader.pages) > 0:
            page = pdf_reader.pages[0]
            text = page.extract_text() or ""
            return text.strip()
        return "No text extracted"
    except Exception as e:
        logger.error(f"Fallback text extraction failed: {e}")
        return f"Text extraction error: {e}"