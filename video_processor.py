# video_processor.py
import av
import cv2
import time

class VideoProcessor:
    def __init__(self):
        self.frame = None
        self.last_frame_time = time.time()
        self.fps = 0
        self.frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        current_time = time.time()
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        self.frame_count += 1
        
        if current_time - self.last_frame_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_frame_time)
            self.last_frame_time = current_time
            self.frame_count = 0

        cv2.putText(img, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        h, w = img.shape[:2]
        center_x, center_y = w//2, h//2
        cv2.line(img, (center_x - 15, center_y), (center_x + 15, center_y), (0, 0, 255), 2)
        cv2.line(img, (center_x, center_y - 15), (center_x, center_y + 15), (0, 0, 255), 2)
        cv2.putText(img, "Align Sheet & Capture", (center_x - 100, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")