import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from tensorflow.keras.models import load_model

from src.config import CLASS_NAMES, MODEL_PATH
from src.utils import predict_label, preprocess_image_array


REALTIME_STATE = {
    "last_update_time": 0.0,
    "last_detections": [],
}

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
}


def resize_with_aspect_ratio(frame: np.ndarray, target_long_edge: int = 480):
    """Resize a frame without distorting portrait or landscape camera input."""
    height, width = frame.shape[:2]
    long_edge = max(height, width)
    if long_edge <= target_long_edge:
        return frame.copy(), 1.0

    scale = target_long_edge / float(long_edge)
    resized = cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


st.set_page_config(
    page_title="Human Emotion Detection",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles():
    """Apply a custom modern visual theme to the app."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

        :root {
            --bg: #07111f;
            --panel: rgba(9, 18, 31, 0.86);
            --panel-strong: rgba(11, 23, 38, 0.96);
            --panel-soft: rgba(255, 255, 255, 0.04);
            --text: #edf4ff;
            --muted: #96a9c5;
            --line: rgba(255, 255, 255, 0.08);
            --accent: #67f0c2;
            --accent-2: #ff9a62;
            --accent-3: #72b5ff;
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 10%, rgba(103, 240, 194, 0.16), transparent 28%),
                radial-gradient(circle at 92% 12%, rgba(126, 184, 255, 0.18), transparent 26%),
                radial-gradient(circle at 48% 85%, rgba(255, 138, 91, 0.12), transparent 32%),
                linear-gradient(180deg, #050c15 0%, #091320 48%, #040a13 100%);
            color: var(--text);
            font-family: "DM Sans", sans-serif;
        }

        .main .block-container {
            padding-top: 0.9rem;
            padding-bottom: 2rem;
            max-width: 1250px;
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }

        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        #MainMenu,
        footer {
            display: none !important;
        }

        section[data-testid="stSidebar"] {
            display: none !important;
        }

        h1, h2, h3, h4 {
            font-family: "Space Grotesk", sans-serif;
            color: var(--text);
            letter-spacing: -0.03em;
        }

        [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }

        [data-testid="stImage"] img {
            border-radius: 22px;
            border: 1px solid var(--line);
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stAlert"],
        .st-emotion-cache-1r6slb0,
        .st-emotion-cache-12w0qpk,
        .st-emotion-cache-1jicfl2 {
            border-radius: 22px;
        }

        .nav-shell {
            position: sticky;
            top: 0.8rem;
            z-index: 20;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            background: rgba(6, 14, 24, 0.74);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 24px;
            padding: 0.95rem 1.2rem;
            backdrop-filter: blur(20px);
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.22);
            margin-bottom: 1.1rem;
        }

        .nav-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .nav-badge {
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(103, 240, 194, 0.18), rgba(114, 181, 255, 0.2));
            border: 1px solid rgba(103, 240, 194, 0.2);
            font-size: 1.2rem;
        }

        .nav-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text);
        }

        .nav-subtitle {
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 0.08rem;
        }

        .nav-links {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .nav-link {
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            padding: 0.62rem 0.9rem;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.03);
            color: var(--muted);
            font-size: 0.88rem;
            transition: all 0.2s ease;
            cursor: pointer;
        }

        .nav-link:hover {
            color: var(--text);
            border-color: rgba(103, 240, 194, 0.28);
            transform: translateY(-1px);
        }

        .nav-link.active {
            color: #08111d;
            background: linear-gradient(135deg, var(--accent), #90ffd9);
            border-color: transparent;
            font-weight: 700;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(8, 18, 31, 0.98), rgba(10, 20, 35, 0.88));
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 2.2rem 2.2rem 1.8rem 2.2rem;
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
        }

        .hero-shell::before {
            content: "";
            position: absolute;
            inset: auto -10% -35% auto;
            width: 340px;
            height: 340px;
            background: radial-gradient(circle, rgba(103, 240, 194, 0.22), transparent 65%);
            pointer-events: none;
        }

        .eyebrow {
            display: inline-block;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #d7fff3;
            background: rgba(103, 240, 194, 0.12);
            border: 1px solid rgba(103, 240, 194, 0.24);
            margin-bottom: 1rem;
        }

        .hero-title {
            font-size: clamp(2.3rem, 5vw, 4.2rem);
            line-height: 0.95;
            margin: 0;
            max-width: 8.5em;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 1.02rem;
            max-width: 55rem;
            margin: 1rem 0 1.4rem 0;
            line-height: 1.7;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 0.35rem;
        }

        .chip {
            padding: 0.6rem 0.95rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--line);
            color: var(--text);
            font-size: 0.92rem;
        }

        .section-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.35rem 1.35rem 1.15rem 1.35rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
            height: 100%;
            overflow: hidden;
        }

        .section-kicker {
            color: var(--accent);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
            font-weight: 700;
        }

        .section-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.45rem;
            margin-bottom: 0.35rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.65;
            margin-bottom: 1rem;
        }

        .metric-card {
            background: var(--panel-soft);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
            min-height: 112px;
            height: 100%;
            overflow: hidden;
        }

        .metric-label {
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.45rem;
        }

        .metric-value {
            font-family: "Space Grotesk", sans-serif;
            font-size: clamp(1.05rem, 2vw, 1.45rem);
            color: var(--text);
            margin-bottom: 0.25rem;
            line-height: 1.15;
            word-break: break-word;
            overflow-wrap: anywhere;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
            word-break: break-word;
            overflow-wrap: anywhere;
        }

        .prediction-card {
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.75rem;
        }

        .prediction-face {
            color: var(--muted);
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .prediction-emotion {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.2rem;
            margin: 0.2rem 0;
            color: var(--text);
        }

        .prediction-confidence {
            color: var(--accent);
            font-weight: 700;
            font-size: 0.95rem;
        }

        .footer-note {
            color: var(--muted);
            text-align: center;
            font-size: 0.92rem;
            margin-top: 1rem;
        }

        .uploader-shell {
            margin-top: 0.55rem;
        }

        .preview-shell {
            margin-top: 1rem;
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(135deg, rgba(10, 22, 38, 0.96), rgba(14, 28, 45, 0.92));
            border: 1px dashed rgba(103, 240, 194, 0.28) !important;
            padding: 1rem;
            color: var(--text) !important;
        }

        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploaderDropzoneInstructions"] {
            font-family: "DM Sans", sans-serif;
            color: var(--text) !important;
        }

        [data-testid="stAlert"] {
            background: rgba(17, 38, 63, 0.72);
            border: 1px solid rgba(114, 181, 255, 0.18);
            color: var(--text);
        }

        [data-testid="stAlert"] * {
            color: var(--text) !important;
        }

        [data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, rgba(103, 240, 194, 0.18), rgba(114, 181, 255, 0.18)) !important;
            border: 1px solid rgba(103, 240, 194, 0.26) !important;
            color: var(--text) !important;
            border-radius: 16px !important;
        }

        .st-emotion-cache-1kyxreq,
        .st-emotion-cache-7ym5gk,
        button[kind="secondary"],
        button[kind="secondaryFormSubmit"] {
            color: var(--text) !important;
        }

        .webcam-shell {
            margin-top: 0.75rem;
            padding: 1rem;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(9, 18, 31, 0.88), rgba(10, 21, 35, 0.72));
            border: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }

        iframe[title^="streamlit_webrtc"],
        iframe[title^="component"] {
            border-radius: 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            background: #0d1726 !important;
        }

        @media (max-width: 900px) {
            .nav-shell {
                position: static;
                flex-direction: column;
                align-items: flex-start;
            }

            .nav-links {
                justify-content: flex-start;
            }

            .hero-title {
                max-width: none;
            }
        }

        .stButton>button, .stDownloadButton>button {
            border-radius: 14px;
            border: 1px solid rgba(103, 240, 194, 0.30);
            background: linear-gradient(135deg, rgba(103, 240, 194, 0.18), rgba(126, 184, 255, 0.18));
            color: var(--text);
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    """Load the trained model and Haar cascade once per app session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run `python train.py` first."
        )

    model = load_model(MODEL_PATH)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_detector.empty():
        raise RuntimeError("Unable to load Haar cascade face detector.")
    return model, face_detector


def analyze_frame(frame: np.ndarray, model, face_detector):
    """Detect faces, predict emotions, and draw overlays on the frame."""
    annotated = frame.copy()
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        grayscale_frame,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
    )

    detections = []
    for (x, y, w, h) in faces:
        face_region = grayscale_frame[y : y + h, x : x + w]
        processed_face = preprocess_image_array(face_region)
        label, confidence = predict_label(model, processed_face)
        detections.append({"emotion": label, "confidence": confidence, "box": (x, y, w, h)})

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (103, 240, 194), 2)
        cv2.putText(
            annotated,
            f"{label} ({confidence:.2f})",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (103, 240, 194),
            2,
        )

    return annotated, detections


def analyze_frame_realtime(frame: np.ndarray, model, face_detector, state):
    """Run a lighter real-time detection pass for smoother webcam performance."""
    display_frame, resize_scale = resize_with_aspect_ratio(frame, target_long_edge=480)
    annotated = display_frame.copy()

    detection_scale = 0.5
    now = time.perf_counter()
    detect_this_frame = (now - state.last_update_time >= 0.9) or not state.last_detections

    if detect_this_frame:
        small_frame = cv2.resize(
            display_frame,
            (
                max(1, int(display_frame.shape[1] * detection_scale)),
                max(1, int(display_frame.shape[0] * detection_scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        grayscale_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        grayscale_small = cv2.equalizeHist(grayscale_small)
        faces = face_detector.detectMultiScale(
            grayscale_small,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(24, 24),
        )

        detections = []
        full_gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        full_gray = cv2.equalizeHist(full_gray)
        faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)[:1]
        for (x, y, w, h) in faces:
            x = int(x / detection_scale)
            y = int(y / detection_scale)
            w = int(w / detection_scale)
            h = int(h / detection_scale)

            face_region = full_gray[y : y + h, x : x + w]
            if face_region.size == 0:
                continue

            processed_face = preprocess_image_array(face_region)
            label, confidence = predict_label_fast(model, processed_face)
            detections.append({"emotion": label, "confidence": confidence, "box": (x, y, w, h)})

        state.last_detections = detections
        state.last_update_time = now

    for detection in state.last_detections:
        x, y, w, h = detection["box"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (103, 240, 194), 2)
        cv2.putText(
            annotated,
            f"{detection['emotion']} ({detection['confidence']:.2f})",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (103, 240, 194),
            2,
        )

    return annotated


class EmotionProcessor:
    """Process webcam frames for live Streamlit emotion detection."""

    def __init__(self):
        self.model, self.face_detector = load_artifacts()
        self.lock = threading.Lock()
        self.last_update_time = 0.0
        self.last_detections = []

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        with self.lock:
            annotated = analyze_frame_realtime(
                image,
                self.model,
                self.face_detector,
                self,
            )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def predict_label_fast(model, processed_image: np.ndarray):
    """Run direct inference without Keras predict() overhead."""
    predictions = model(processed_image, training=False).numpy()[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    return CLASS_NAMES[predicted_index], confidence


def render_navbar():
    """Render the top navigation bar."""
    st.markdown(
        """
        <div class="nav-shell">
            <div class="nav-brand">
                <div class="nav-badge">AI</div>
                <div>
                    <div class="nav-title">Emotion Vision Studio</div>
                    <div class="nav-subtitle">Dark mode facial emotion analytics</div>
                </div>
            </div>
            <div class="nav-links">
                <a class="nav-link active" href="#dashboard">Dashboard</a>
                <a class="nav-link" href="#image-scan">Image Scan</a>
                <a class="nav-link" href="#live-camera">Live Camera</a>
                <a class="nav-link" href="#model-info">Model Info</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    """Render the app hero section."""
    st.markdown(
        """
        <div id="dashboard" class="hero-shell">
            <div class="eyebrow">Computer Vision + Deep Learning</div>
            <div class="hero-title">Human Emotion Detection using CNN</div>
            <div class="hero-copy">
                Run emotion recognition from uploaded photos or switch into live browser webcam mode
                for real-time facial analysis across the seven FER2013 classes.
            </div>
            <div class="chip-row">
                <div class="chip">7 Emotion Classes</div>
                <div class="chip">48x48 Grayscale Input</div>
                <div class="chip">Live Webcam Inference</div>
                <div class="chip">Image Upload Support</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_panel():
    """Render model and app details."""
    st.markdown(
        """
        <div id="model-info" class="section-card">
            <div class="section-kicker">System Overview</div>
            <div class="section-title">Inference Control Panel</div>
            <div class="section-copy">
                The app loads the trained CNN once, detects faces with Haar cascades, and predicts the
                visible emotion for each detected face region.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model File</div>
                <div class="metric-value">{MODEL_PATH.name}</div>
                <div class="metric-note">Loaded from local project output</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Input Format</div>
                <div class="metric-value">48x48 Gray</div>
                <div class="metric-note">Face crop normalized to 0-1</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Emotion Classes</div>
            <div class="metric-note">{", ".join(name.title() for name in CLASS_NAMES)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_cards(detections):
    """Render prediction cards for detected faces."""
    if not detections:
        st.warning("No face detected in the uploaded image.")
        return

    st.markdown(
        """
        <div class="section-kicker" style="margin-top: 0.5rem;">Detections</div>
        <div class="section-title" style="font-size:1.15rem; margin-bottom: 0.9rem;">Prediction Summary</div>
        """,
        unsafe_allow_html=True,
    )

    for index, detection in enumerate(detections, start=1):
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-face">Face {index}</div>
                <div class="prediction-emotion">{detection['emotion'].title()}</div>
                <div class="prediction-confidence">{detection['confidence']:.2%} confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def uploaded_image_ui(model, face_detector):
    """Render uploaded-image inference UI."""
    st.markdown(
        """
        <div id="image-scan" class="section-card">
            <div class="section-kicker">Static Input</div>
            <div class="section-title">Image Detection</div>
            <div class="section-copy">
                Upload a front-facing image to scan visible faces and highlight the predicted emotion
                for each detected subject.
            </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"],
        help="Best results come from a clear, front-facing face image.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Upload an image to run emotion detection.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Unable to decode the uploaded image.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    annotated, detections = analyze_frame(image, model, face_detector)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.markdown('<div class="preview-shell">', unsafe_allow_html=True)
    preview_left, preview_right = st.columns(2, gap="medium")
    with preview_left:
        st.image(original_rgb, caption="Original image", use_column_width=True)
    with preview_right:
        st.image(display_image, caption="Detected emotions", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    render_prediction_cards(detections)
    st.markdown("</div>", unsafe_allow_html=True)


def live_webcam_ui():
    """Render live webcam inference UI using WebRTC."""
    st.markdown(
        """
        <div id="live-camera" class="section-card">
            <div class="section-kicker">Realtime Mode</div>
            <div class="section-title">Live Webcam Detection</div>
            <div class="section-copy">
                Allow browser camera access, start the stream, and watch the model classify facial
                emotion in real time with on-frame overlays.
            </div>
        """,
        unsafe_allow_html=True,
    )

    st.info("Allow camera access in your browser, then press start to begin live detection.")
    st.caption(
        "If live camera takes too long to connect on Streamlit Cloud, try Chrome on a normal network. "
        "Some mobile networks, office Wi-Fi setups, and in-app browsers can block WebRTC."
    )

    st.markdown('<div class="webcam-shell">', unsafe_allow_html=True)
    webrtc_streamer(
        key="emotion-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "facingMode": "user",
                "width": {"ideal": 640, "min": 320},
                "height": {"ideal": 480, "min": 240},
                "frameRate": {"ideal": 8, "max": 10},
            },
            "audio": False,
        },
        async_processing=False,
        sendback_audio=False,
        video_html_attrs={
            "autoPlay": True,
            "playsInline": True,
            "controls": False,
            "width": "100%",
            "height": 420,
            "style": {
                "width": "100%",
                "height": "420px",
                "objectFit": "contain",
                "borderRadius": "20px",
                "backgroundColor": "#0d1726",
            }
        },
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    inject_styles()
    render_navbar()
    render_hero()

    model, face_detector = load_artifacts()

    top_left, top_right = st.columns([1.45, 0.85], gap="large")
    with top_left:
        uploaded_image_ui(model, face_detector)
    with top_right:
        render_info_panel()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    live_webcam_ui()
    st.markdown(
        '<div class="footer-note">Designed for portfolio demos, browser inference, and real-time emotion visualization.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
