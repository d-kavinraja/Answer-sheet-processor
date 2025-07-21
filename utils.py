import streamlit as st
import os

def st_success(message: str):
    """Display a success message with consistent styling."""
    st.markdown(f'<div style="background-color:#d4edda;color:#155724;padding:10px;border-radius:5px;margin-bottom:10px;">✅ {message}</div>', unsafe_allow_html=True)

def st_error(message: str):
    """Display an error message with consistent styling."""
    st.markdown(f'<div style="background-color:#f8d7da;color:#721c24;padding:10px;border-radius:5px;margin-bottom:10px;">❌ {message}</div>', unsafe_allow_html=True)

def st_warning(message: str):
    """Display a warning message with consistent styling."""
    st.markdown(f'<div style="background-color:#fff3cd;color:#856404;padding:10px;border-radius:5px;margin-bottom:10px;">⚠️ {message}</div>', unsafe_allow_html=True)

def st_info(message: str):
    """Display an info message with consistent styling."""
    st.markdown(f'<div style="background-color:#d1ecf1;color:#0c5460;padding:10px;border-radius:5px;margin-bottom:10px;">ℹ️ {message}</div>', unsafe_allow_html=True)

def get_image_download_button(file_path: str, filename: str, button_text: str):
    """Create a download button for an image file."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            st.download_button(
                label=button_text,
                data=file,
                file_name=filename,
                mime="image/jpeg"
            )
