import streamlit as st
import os
from datetime import datetime

def st_success(message: str):
    """Display a success message with consistent styling."""
    st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)

def st_error(message: str):
    """Display an error message with consistent styling."""
    st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)

def st_info(message: str):
    """Display an info message with consistent styling."""
    st.markdown(f'<div class="info-box">{message}</div>', unsafe_allow_html=True)

def st_warning(message: str):
    """Display a warning message with consistent styling."""
    st.markdown(f'<div class="warning-box">{message}</div>', unsafe_allow_html=True)

def get_image_download_button(file_path: str, download_filename: str, button_text: str) -> None:
    """Create a download button for an image file."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            st.download_button(
                label=button_text,
                data=file,
                file_name=download_filename,
                mime="image/jpeg",
                key=f"download_{download_filename}_{datetime.now().timestamp()}"
            )
    else:
        st_warning(f"File not found: {file_path}")

def save_results_to_file(results: list, filename: str) -> str:
    """Save extraction results to a text file and return the file path."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"{filename}.txt")
    try:
        with open(file_path, "w") as f:
            for label, value in results:
                f.write(f"{label}: {value}\n")
        return file_path
    except Exception as e:
        st_error(f"Failed to save results to file: {str(e)}")
        return ""
