import streamlit as st
import uuid

def initialize_session_state():
    """Initializes Streamlit session state variables if they don't exist."""
    defaults = {
        'logged_in': False,
        'username': "",
        'email': "",
        'otp_stage': False,
        'temp_user_data': {},
        'input_method': "Upload Image",
        'image_path': None,
        'image_captured': False,
        'selected_history_item_index': None,
        'webrtc_key': f"webrtc_{uuid.uuid4().hex}",
        'processing_start_time': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value