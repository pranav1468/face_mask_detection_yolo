import streamlit as st
import numpy as np
import cv2
import time
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
<<<<<<< HEAD
from collections import Counter
=======
>>>>>>> 0245f7f012bd24bf9a8a6307b624562dc072f348

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    "<h1>😷 Face Mask Detection System</h1>"
    "<p style='color:#6b7280;'>YOLO-based Surveillance Detection</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Model (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best_face_mask.onnx")

model = load_model()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5, 0.05
)

mode = st.sidebar.radio(
    "Detection Mode",
    [
        "📤 Upload Image",
        "🎥 Upload Video",
        "💻 Webcam (Local OpenCV)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 Image & Video work everywhere.\n"
    "Local webcam works only on your PC."
)

# ==================================================
# IMAGE MODE
# ==================================================
if mode == "📤 Upload Image":
    st.subheader("📤 Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        if st.button("🚀 Run Detection"):
            with st.spinner("Detecting faces and masks..."):
                start = time.time()
                results = model(img_np, conf=conf_threshold)[0]
                end = time.time()

            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(annotated, caption="Detected Output", use_column_width=True)

            # ---------------- SUMMARY ----------------
            st.markdown("### 📊 Detection Summary")

            if len(results.boxes) == 0:
                st.info("No faces detected.")
            else:
                labels = [model.names[int(b.cls[0])] for b in results.boxes]
                counts = Counter(labels)

                m1, m2, m3 = st.columns(3)
                m1.metric("😷 With Mask", counts.get("with_mask", 0))
                m2.metric("⚠️ Incorrect Mask", counts.get("mask_weared_incorrect", 0))
                m3.metric("🚨 No Mask", counts.get("without_mask", 0))

                st.markdown("#### 🔍 Detailed Detections")
                for box in results.boxes:
                    label = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])

                    if label == "without_mask":
                        st.error(f"🚨 No Mask — {conf:.2f}")
                    elif label == "mask_weared_incorrect":
                        st.warning(f"⚠️ Incorrect Mask — {conf:.2f}")
                    else:
                        st.success(f"✅ Mask Detected — {conf:.2f}")

            st.caption(f"⏱ Inference Time: {end - start:.3f} seconds")

# ==================================================
# VIDEO MODE
# ==================================================
elif mode == "🎥 Upload Video":
    st.subheader("🎥 Upload a Video")

    video_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf_threshold)[0]
            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            stframe.image(annotated, channels="RGB")

        cap.release()
        os.remove(tfile.name)

<<<<<<< HEAD
=======

>>>>>>> 0245f7f012bd24bf9a8a6307b624562dc072f348
# ==================================================
# WEBCAM (LOCAL – OPENCV)
# ==================================================
elif mode == "💻 Webcam (Local OpenCV)":
    st.subheader("💻 Live Webcam (Local OpenCV)")
    st.warning("⚠️ Works only on your local machine")

    run = st.checkbox("▶️ Start Webcam")

    if run:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf_threshold)[0]
            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(annotated, channels="RGB")

        cap.release()
        cv2.destroyAllWindows()

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "Academic + Real-World Surveillance Pipeline</p>",
    unsafe_allow_html=True
)
