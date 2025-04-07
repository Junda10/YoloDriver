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
import io
import asyncio
import sys

# Setup asyncio policy for Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# ---------------- Model Definitions & Loading ----------------
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
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
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class_labels = {0: "Normal Driving", 1: "Operating the Radio", 2: "Reaching Behind"}

# Load models
yolo_model = YOLO("yolov8m.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model = ImprovedCNN(num_classes=3).to(device)
cnn_model.load_state_dict(torch.load("best_model_CNN_99.33.pth", map_location=device))
cnn_model.eval()
feature_extractor = nn.Sequential(cnn_model.conv_layers, nn.Flatten()).to(device)
svm_model = joblib.load("svm_classifier_gridsearch_yolo_c057.pkl")

# ---------------- Preprocessing ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def overlay_text(img, texts, start_y=50, dy=20):
    for i, text in enumerate(texts):
        y = start_y + i * dy
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

def process_frame(img):
    debug_text = [f"Frame shape: {img.shape}"]
    try:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        debug_text.append(f"cvtColor error: {e}")
        overlay_text(img, debug_text)
        return img, "Error"

    try:
        results = yolo_model(image_rgb)
    except Exception as e:
        debug_text.append(f"YOLO error: {e}")
        overlay_text(img, debug_text)
        return img, "Error"

    detected_label = "Normal Driving"
    person_boxes = []
    for result in results:
        for box in result.boxes:
            try:
                if int(box.cls.item()) == 0:
                    person_boxes.append(box)
            except Exception as e:
                debug_text.append(f"Box error: {e}")
    debug_text.append(f"Person boxes: {len(person_boxes)}")

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
                detected_label = class_labels.get(prediction, "Unknown")
            except Exception as e:
                debug_text.append(f"SVM error: {e}")
                detected_label = "Error predicting"
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            cv2.putText(img, detected_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(img, f"Status: {detected_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    overlay_text(img, debug_text)
    return img, detected_label

# ---------------- Video Processor ----------------
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
                    if int(box.cls.item()) == 0:
                        person_boxes.append(box)
                except Exception as e:
                    debug_text.append(f"Box error: {e}")
        debug_text.append(f"Person boxes: {len(person_boxes)}")
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
                    detected_label = class_labels.get(prediction, "Unknown")
                except Exception as e:
                    debug_text.append(f"SVM error: {e}")
                    detected_label = "Error predicting"
                color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                cv2.putText(img, detected_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"Status: {detected_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self._overlay_text(img, debug_text)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    def _overlay_text(self, img, texts, start_y=50, dy=20):
        for i, text in enumerate(texts):
            y = start_y + i * dy
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# ---------------- Streamlit App Layout ----------------
st.set_page_config(page_title="Driver Monitoring System", page_icon="🚗", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Main Page", "Detection Page"])

if page == "Main Page":
    st.header("Project Overview")
    st.markdown("""
    **Driver Behavior Monitoring System**  
    This project uses YOLOv8 for person detection combined with a custom CNN-SVM pipeline to identify driver distractions in real time.  
    The system alerts the driver when unsafe behavior is detected.
    """)
    st.write("Use the **Detection Page** from the sidebar for live tracking, photo, or video detection.")

elif page == "Detection Page":
    st.header("Driver Distraction Detection")
    detection_mode = st.selectbox("Select Detection Mode", ["Live Tracking", "Photos", "Video Detection"])
    
    if detection_mode == "Live Tracking":
        st.subheader("Live Tracking")
        st.write("Using your webcam for real-time driver behavior monitoring.")
        webrtc_streamer(
            key="driver-monitoring",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            frontend_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
    
    elif detection_mode == "Photos":
        st.subheader("Photo Detection")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_rgb = np.array(image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            results = yolo_model(image_rgb)
            person_boxes = []
            for result in results:
                for box in result.boxes:
                    try:
                        if int(box.cls.item()) == 0:
                            person_boxes.append(box)
                    except Exception as e:
                        st.write(f"Box error: {e}")
            if person_boxes:
                try:
                    image_tensor = transform(image_rgb).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = feature_extractor(image_tensor)
                    features = features.view(features.size(0), -1).cpu().numpy()
                    prediction = svm_model.predict(features)[0]
                    predicted_label = class_labels.get(prediction, "Unknown")
                    st.markdown(f"### 🚦 Predicted Activity: {predicted_label}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.info("No person detected.")
                
    elif detection_mode == "Video Detection":
        st.subheader("Video Detection")
        video_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
        if video_file:
            video_bytes = io.BytesIO(video_file.read())
            with open("uploaded_video.mp4", "wb") as out:
                out.write(video_bytes.read())
            cap = cv2.VideoCapture("uploaded_video.mp4")
            if not cap.isOpened():
                st.error("Could not open video file.")
            else:
                org_frame = st.empty()
                ann_frame = st.empty()
                log_ph = st.empty()
                stop_button = st.button("Stop")
                detection_log = []
                last_second = -1
                last_pred = "No detection"
                last_processed_frame = None
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("End of video or frame read error.")
                        break
                    timestamp_sec = int(cap.get(cv2.CAP_PROP_POS_MSEC) // 1000)
                    if timestamp_sec > last_second:
                        annotated_frame, pred = process_frame(frame)
                        last_pred = pred
                        last_processed_frame = annotated_frame
                        last_second = timestamp_sec
                        detection_log.append({"Time (s)": timestamp_sec, "Activity": last_pred})
                    else:
                        annotated_frame = last_processed_frame.copy() if last_processed_frame is not None else frame.copy()
                        cv2.putText(annotated_frame, f"Status: {last_pred}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    org_frame.image(frame, channels="BGR")
                    ann_frame.image(annotated_frame, channels="BGR")
                    log_ph.table(detection_log)
                    if stop_button:
                        break
                cap.release()
                cv2.destroyAllWindows()
                def summarize_detection_log(log):
                    summarized = []
                    if log:
                        current_activity = log[0]["Activity"]
                        start_time = log[0]["Time (s)"]
                        end_time = start_time
                        for entry in log[1:]:
                            if entry["Activity"] == current_activity:
                                end_time = entry["Time (s)"]
                            else:
                                summarized.append({"Time Range": f"{start_time} - {end_time}", "Activity": current_activity})
                                current_activity = entry["Activity"]
                                start_time = entry["Time (s)"]
                                end_time = start_time
                        summarized.append({"Time Range": f"{start_time} - {end_time}", "Activity": current_activity})
                    return summarized
                st.markdown("### Detection Log Summary:")
                st.table(summarize_detection_log(detection_log))

st.sidebar.subheader("Class Labels:")
for key, value in class_labels.items():
    st.sidebar.write(f"**{key}: {value}**")
