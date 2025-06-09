import cv2

MODEL_NAME = "mnasnet_small.lamb_in1k"
MODEL_PATH = "model.onnx"
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"
CLASS_NAMES = [
    "anger", "contempt", "disgust", "embarrass", "fear",
    "joy", "neutral", "pride", "sadness", "surprise"
]
WINDOW_TITLE = "Deteksi Emosi (Timm Preprocessing)"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
VIDEO_FPS = 30
DETECTION_INTERVAL_SECONDS = 0.5
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 10
HAAR_MIN_SIZE = (30, 30)
# --- Drawing Configuration ---
BOX_COLOR = (0, 255, 0)      # Green
TEXT_COLOR = (255, 255, 255)  # White
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
