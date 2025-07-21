import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase
import logging

logger = logging.getLogger(__name__)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        """Initialize video processor."""
        logger.debug("Initializing VideoProcessor")
        self.frame = None

    def recv(self, frame):
        """Receive and process video frame."""
        logger.debug("Receiving video frame")
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

    def get_frame(self):
        """Return the latest frame."""
        logger.debug("Getting latest frame")
        return self.frame if self.frame is not None else None
