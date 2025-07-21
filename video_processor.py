import cv2
from streamlit_webrtc import VideoProcessorBase
import logging

logger = logging.getLogger(__name__)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        """Receive and store the latest video frame."""
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

    def get_frame(self):
        """Return the latest captured frame."""
        return self.frame
