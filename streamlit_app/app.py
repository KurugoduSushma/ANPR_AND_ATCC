import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import Counter
import matplotlib.pyplot as plt

# =============================
# PAGE CONFIG
# =============================
st.set_page_config("Smart Traffic AI", layout="wide")
st.title("🚦 Smart Traffic AI System")

# =============================
# LOAD MODELS
# =============================
atcc_model = YOLO("yolov8n.pt")   # vehicle detector
anpr_model = YOLO("yolov8n.pt")   # plate detector (demo)
reader = easyocr.Reader(['en'])

# =============================
# SIDEBAR CONTROLS
# =============================
st.sidebar.header("⚙ Detection Settings")

mode = st.sidebar.radio(
    "Select Detection Mode",
    ["ATCC – Vehicle Counting", "ANPR – License Plate Detection"]
)

confidence = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.4
)

st.sidebar.success(f"Confidence Set: {confidence}")

# =============================
# FILE UPLOAD + ANALYZE BUTTON
# =============================
uploaded_file = st.file_uploader(
    "📤 Upload Image (ANPR) or Video (ATCC)",
    type=["jpg", "png", "jpeg", "mp4", "avi", "mov"]
)

analyze = st.button("🚀 Start Analysis")

# =============================
# ✅ ATCC MODE (VIDEO) — FINAL STABLE + CLEAN UI
# =============================
if analyze and mode.startswith("ATCC") and uploaded_file:

    st.subheader("🎥 Input Video (Full Width)")

    # ✅ Save uploaded video safely
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_file.read())
    temp.close()

    cap = cv2.VideoCapture(temp.name)

    stframe = st.empty()
    vehicle_counts = Counter()

    # =============================
    # ✅ FULL WIDTH VIDEO STREAM
    # =============================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = atcc_model(frame, conf=confidence)[0]

        for box in results.boxes:
            cls = int(box.cls)
            name = results.names[cls]
            vehicle_counts[name] += 1

        annotated = results.plot()
        stframe.image(annotated, channels="BGR", width=1200)

    cap.release()

    # =============================
    # ✅ VEHICLE COUNT TABLE
    # =============================
    st.markdown("---")
    st.subheader("📊 Vehicle Count Table")

    if len(vehicle_counts) > 0:
        import pandas as pd
        df = pd.DataFrame(vehicle_counts.items(), columns=["Vehicle Type", "Count"])
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No vehicles detected.")

    # =============================
    # ✅ CENTERED + PROFESSIONAL ANALYTICS
    # =============================
    st.markdown("---")
    st.subheader("📈 Vehicle Analytics")

    if len(vehicle_counts) > 0:
        labels = list(vehicle_counts.keys())
        values = list(vehicle_counts.values())

        # ✅ CENTERED BAR + PIE
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 📊 Distribution Overview")

            fig1, ax1 = plt.subplots(figsize=(7, 4))
            ax1.bar(labels, values)
            ax1.set_ylabel("Count")
            ax1.set_xlabel("Vehicle Type")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            st.pyplot(fig2)

    # ✅ CENTERED LINE + SCATTER
    st.markdown("### 📉 Traffic Trend & Scatter")
    col4, col5, col6 = st.columns([1, 2, 1])

    with col5:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(labels, values, marker="o")
        ax3.set_ylabel("Count")
        ax3.set_xlabel("Vehicle Type")
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots(figsize=(7, 4))
        ax4.scatter(labels, values)
        ax4.set_ylabel("Count")
        ax4.set_xlabel("Vehicle Type")
        st.pyplot(fig4)

# =============================
# ANPR MODE (IMAGE) — OCR ONLY (NO BOX)
# =============================
elif analyze and mode.startswith("ANPR") and uploaded_file:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼 Input Image")

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, channels="BGR", caption="Original Image")

    with col2:
        st.subheader("🔢 Detected License Plate Text")

        # ✅ Run OCR directly on full image (NO BOX)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        results = reader.readtext(thresh)

        if len(results) > 0:
            detected_plates = []

            for res in results:
                detected_plates.append(res[-2])

            st.success("✅ Detected Plate Characters:")
            for plate in detected_plates:
                st.write(f"🔢 {plate}")

        else:
            st.warning("⚠ No license plate text detected.")
