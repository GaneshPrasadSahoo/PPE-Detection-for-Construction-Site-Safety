import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

# Load YOLO model (Force CPU)
model_path = r"E:\ppe detection\best.pt"
model = YOLO(model_path).to("cpu")

# Define PPE items
PPE_CLASSES = ['Boots', 'Ear-protection', 'Glass', 'Glove', 'Helmet', 'Mask', 'Person', 'Vest']

st.title("🦺 PPE Detection Web App")

# Sidebar for options
st.sidebar.title("Options")
option = st.sidebar.radio("Choose an input method:", ("Upload Image", "Upload Video", "Use Camera"))

def process_frame(frame):
    """Runs YOLO detection and processes results."""
    results = model(frame)
    
    detected_classes = set()
    missing_ppe = set(PPE_CLASSES)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Track detected classes
            detected_classes.add(label)

            # Remove detected items from missing list
            if label in missing_ppe:
                missing_ppe.remove(label)

            # Draw bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate safety percentage
    total_ppe = len(PPE_CLASSES) - 1  # Excluding 'Person'
    detected_ppe = len(detected_classes - {'Person'})  # Excluding 'Person'
    safety_percent = int((detected_ppe / total_ppe) * 100)

    return frame, safety_percent, missing_ppe

# Upload Image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to OpenCV format
        image_np = np.array(image)
        image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Process frame
        image_cv2, safety_percent, missing_ppe = process_frame(image_cv2)

        # Display processed image
        st.image(image_cv2, caption="Detected Objects", use_column_width=True)

        # Show safety message
        if safety_percent == 100:
            st.success("✅ You are **100% Safe!**")
        elif safety_percent >= 60:
            st.warning(f"⚠️ You are **{safety_percent}% Safe**. Please wear: {', '.join(missing_ppe)}")
        else:
            st.error(f"❌ **You are NOT Safe!** Missing: {', '.join(missing_ppe)}")

# Upload Video
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        st.video(uploaded_video)

        # Save video temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            frame, safety_percent, missing_ppe = process_frame(frame)

            # Display frame
            stframe.image(frame, channels="BGR")

            # Display safety message
            if safety_percent == 100:
                st.success("✅ You are **100% Safe!**")
            elif safety_percent >= 60:
                st.warning(f"⚠️ You are **{safety_percent}% Safe**. Please wear: {', '.join(missing_ppe)}")
            else:
                st.error(f"❌ **You are NOT Safe!** Missing: {', '.join(missing_ppe)}")

        cap.release()

# Real-Time Camera Detection
elif option == "Use Camera":
    st.write("📷 **Starting Camera...**")
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        frame, safety_percent, missing_ppe = process_frame(frame)

        # Display frame
        stframe.image(frame, channels="BGR")

        # Display safety message
        if safety_percent == 100:
            st.success("✅ You are **100% Safe!**")
        elif safety_percent >= 60:
            st.warning(f"⚠️ You are **{safety_percent}% Safe**. Please wear: {', '.join(missing_ppe)}")
        else:
            st.error(f"❌ **You are NOT Safe!** Missing: {', '.join(missing_ppe)}")

    cap.release()

