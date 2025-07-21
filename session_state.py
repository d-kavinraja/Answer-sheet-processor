# session_state.py
import streamlit as st
import uuid

def initialize_session_state():
    """Initializes all required keys in the Streamlit session state."""
    # --- Authentication State ---
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None

    # --- Scanner App State ---
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