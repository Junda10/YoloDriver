import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import joblib
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import base64
import os
import tempfile
import io
import asyncio
import sys
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
# ---------------- Model Definitions & Loading ----------------

# Define CNN Model for Behavior Classification
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# Class Labels Mapping for Behavior Classification
class_labels = {
    0: "Normal Driving",
    1: "Operating the Radio",
    2: "Reaching Behind"
}

# Load YOLO Model (used solely for person detection)
yolo_model = YOLO("yolov8m.pt")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Pretrained CNN Model
cnn_model = ImprovedCNN(num_classes=3).to(device)
cnn_model.load_state_dict(torch.load("best_model_CNN_99.33.pth", map_location=device))
cnn_model.eval()

# Create Feature Extractor from CNN (using conv_layers)
feature_extractor = nn.Sequential(
    cnn_model.conv_layers,
    nn.Flatten()
).to(device)

# Load Trained SVM Model
svm_model = joblib.load("svm_classifier_gridsearch_yolo_c057.pkl")

# ---------------- Image Preprocessing ----------------
# Using the full image for classification (since your training data is full image)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ---------------- Video Functions ----------------
def _overlay_text(img, texts):
    """Overlay multiple lines of text on the image."""
    y0, dy = 50, 20
    for i, text in enumerate(texts):
        y = y0 + i * dy
        cv2.putText(img, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def process_frame(img):
    """
    Process the frame using YOLO for person detection.
    Then, use the full image for classification.
    Returns the annotated image and the predicted label.
    """
    debug_text = [f"Frame shape: {img.shape}"]
    try:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        debug_text.append(f"cvtColor error: {e}")
        _overlay_text(img, debug_text)
        return img, "Error"

    try:
        results = yolo_model(image_rgb)
    except Exception as e:
        debug_text.append(f"YOLO error: {e}")
        _overlay_text(img, debug_text)
        return img, "Error"

    detected_label = "Normal Driving"
    person_boxes = []
    # Use YOLO solely for person detection (class 0)
    for result in results:
        for box in result.boxes:
            try:
                cls = int(box.cls.item())
                if cls == 0:
                    person_boxes.append(box)
            except Exception as e:
                debug_text.append(f"Box processing error: {e}")
    debug_text.append(f"Person boxes: {len(person_boxes)}")

    # Draw bounding box for person detection (if any)
    if person_boxes:
        best_box = max(person_boxes, key=lambda b: b.conf.item())
        try:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except Exception as e:
            debug_text.append(f"Coordinate error: {e}")
    else:
        detected_label = "No person detected"
        debug_text.append("No person detected")

    # Use the full image for classification
    try:
        tensor = transform(image_rgb).unsqueeze(0).to(device)
    except Exception as e:
        debug_text.append(f"Transform error: {e}")
        tensor = None
    if tensor is not None:
        try:
            with torch.no_grad():
                features = feature_extractor(tensor)
        except Exception as e:
            debug_text.append(f"Feature extraction error: {e}")
            features = None
        if features is not None:
            features = features.view(features.size(0), -1).cpu().numpy()
            try:
                prediction = svm_model.predict(features)[0]
                detected_label = class_labels[prediction]
            except Exception as e:
                debug_text.append(f"SVM error: {e}")
                detected_label = "Error predicting"
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            cv2.putText(img, detected_label, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(img, f"Status: {detected_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _overlay_text(img, debug_text)
    return img, detected_label

# ---------------- Custom Video Processor for Live Tracking ----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        debug_text = [f"Frame: {self.frame_count}, Shape: {img.shape}"]
        try:
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            debug_text.append(f"cvtColor error: {e}")
            self._overlay_text(img, debug_text)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        try:
            results = yolo_model(image_rgb)
        except Exception as e:
            debug_text.append(f"YOLO error: {e}")
            self._overlay_text(img, debug_text)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        detected_label = "Normal Driving"
        person_boxes = []
        for result in results:
            for box in result.boxes:
                try:
                    cls = int(box.cls.item())
                    if cls == 0:
                        person_boxes.append(box)
                except Exception as e:
                    debug_text.append(f"Box processing error: {e}")
        debug_text.append(f"Person boxes: {len(person_boxes)}")
        if person_boxes:
            best_box = max(person_boxes, key=lambda b: b.conf.item())
            try:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            except Exception as e:
                debug_text.append(f"Coordinate error: {e}")
                x1 = y1 = x2 = y2 = 0
        else:
            detected_label = "No person detected"
            debug_text.append("No person detected")
        # Classify using the full image
        try:
            tensor = transform(image_rgb).unsqueeze(0).to(device)
        except Exception as e:
            debug_text.append(f"Transform error: {e}")
            tensor = None
        if tensor is not None:
            try:
                with torch.no_grad():
                    features = feature_extractor(tensor)
            except Exception as e:
                debug_text.append(f"Feature extraction error: {e}")
                features = None
            if features is not None:
                features = features.view(features.size(0), -1).cpu().numpy()
                try:
                    prediction = svm_model.predict(features)[0]
                    detected_label = class_labels[prediction]
                except Exception as e:
                    debug_text.append(f"SVM error: {e}")
                    detected_label = "Error predicting"
                color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                cv2.putText(img, detected_label, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Status: {detected_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self._overlay_text(img, debug_text)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    def _overlay_text(self, img, texts):
        y0, dy = 50, 20
        for i, text in enumerate(texts):
            y = y0 + i * dy
            cv2.putText(img, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# ---------------- Streamlit App Layout & Navigation ----------------

st.set_page_config(page_title="Driver Monitoring System", page_icon="🚗", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Main Page", "Detection Page"])

if page == "Main Page":
    st.header("Project Overview")
    st.write("""
        **Driver Behavior Monitoring System**  
        This project uses YOLOv8 for person detection combined with a custom CNN-SVM pipeline to identify driver distractions in real time.  
        The system alerts the driver when unsafe behavior is detected.
    """)
    st.write("Use the **Detection Page** from the sidebar to access live tracking, photo, or video detection.")

elif page == "Detection Page":
    st.header("Driver Distraction Detection")
    detection_mode = st.selectbox("Select Detection Mode", ["Live Tracking", "Photos", "Video Detection"])

    if detection_mode == "Live Tracking":
        st.write("### Live Tracking")
        st.write("This mode uses your webcam for real-time driver behavior monitoring.")
        webrtc_streamer(
            key="driver-monitoring",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            frontend_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

    elif detection_mode == "Photos":
        st.write("### Photo Detection")
        st.subheader("Upload an Image for Driver Behavior Analysis")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_rgb = np.array(image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            # YOLO detection on the uploaded image
            results = yolo_model(image_rgb)
            person_boxes = []
            for result in results:
                for box in result.boxes:
                    try:
                        cls = int(box.cls.item())
                        if cls == 0:
                            person_boxes.append(box)
                    except Exception as e:
                        st.write(f"Box processing error: {e}")
            if person_boxes:
                best_box = max(person_boxes, key=lambda b: b.conf.item())
                person_crop = image_rgb
                if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                    image_tensor = transform(person_crop).unsqueeze(0).to(device)
                    transformed_img = image_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
                    transformed_img = np.clip(transformed_img * 255, 0, 255).astype(np.uint8)
                    with torch.no_grad():
                        features = feature_extractor(image_tensor)
                    features = features.view(features.size(0), -1).cpu().numpy()
                    prediction = svm_model.predict(features)[0]
                    predicted_label = class_labels[prediction]
                    st.write(f"### 🚦 Predicted Activity: {predicted_label}")
                    st.image(transformed_img, caption="Processed Image", use_container_width=True)
                    if prediction != 0:  # Alert if not normal driving
                        with open("preview.mp3", "rb") as f:
                            audio_bytes = f.read()
                        audio_b64 = base64.b64encode(audio_bytes).decode()
                        audio_html = f"""
                        <audio autoplay>
                            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                        </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)
            else:
                st.write("No person detected.")
                
    elif detection_mode == "Video Detection":
        st.write("### Video Detection")
        st.subheader("Upload a Video for Driver Behavior Analysis")
        video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if video_file is not None:
            # Save the video to a temporary file
            g = io.BytesIO(video_file.read())
            with open("ultralytics.mp4", "wb") as out:
                out.write(g.read())
            vid_file_name = "ultralytics.mp4"
            cap = cv2.VideoCapture(vid_file_name)
            if not cap.isOpened():
                st.error("Could not open video file.")
            else:
                # Create placeholders for real-time display and log
                org_frame = st.empty()
                ann_frame = st.empty()
                log_ph = st.empty()
                stop_button = st.button("Stop")
                detection_log = []   # Each entry: {"Time (s)": timestamp, "Activity": prediction}
                last_second = -1
                last_pred = "No detection"
                last_processed_frame = None
                # Process video frames in a loop
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("End of video or cannot read frame.")
                        break
                    # Get current timestamp (in seconds) from the video
                    timestamp_sec = int(cap.get(cv2.CAP_PROP_POS_MSEC) // 1000)
                    # Process only when a new second is reached
                    if timestamp_sec > last_second:
                        annotated_frame, pred = process_frame(frame)
                        last_pred = pred
                        last_processed_frame = annotated_frame
                        last_second = timestamp_sec
                        detection_log.append({"Time (s)": timestamp_sec, "Activity": last_pred})
                    else:
                        if last_processed_frame is not None:
                            annotated_frame = last_processed_frame.copy()
                            cv2.putText(annotated_frame, f"Status: {last_pred}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        else:
                            annotated_frame, pred = process_frame(frame)
                            last_pred = pred
                            last_processed_frame = annotated_frame
                            last_second = timestamp_sec
                    # Display the original and annotated frames, and update the log table
                    org_frame.image(frame, channels="BGR")
                    ann_frame.image(annotated_frame, channels="BGR")
                    log_ph.table(detection_log)
                    if stop_button:
                        break
                cap.release()
                cv2.destroyAllWindows()
            # Summarize detection log
            def summarize_detection_log(detection_log):
                if not detection_log:
                    return []
                summarized = []
                current_activity = detection_log[0]["Activity"]
                start_time = detection_log[0]["Time (s)"]
                end_time = start_time
                for entry in detection_log[1:]:
                    if entry["Activity"] == current_activity:
                        end_time = entry["Time (s)"]
                    else:
                        summarized.append({"Time Range": f"{start_time} - {end_time}", "Activity": current_activity})
                        current_activity = entry["Activity"]
                        start_time = entry["Time (s)"]
                        end_time = start_time
                summarized.append({"Time Range": f"{start_time} - {end_time}", "Activity": current_activity})
                return summarized
            summarized_log = summarize_detection_log(detection_log)
            st.write("### Detection Log Summary:")
            st.table(summarized_log)

# ---------------- Sidebar: Class Labels ----------------
st.sidebar.subheader("Class Labels for Driver Behavior Classification:")
for key, value in class_labels.items():
    st.sidebar.write(f"**{key}: {value}**")
