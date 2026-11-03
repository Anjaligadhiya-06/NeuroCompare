# app.py
import streamlit as st
import cv2
import numpy as np
from engine import align_and_register, compute_progression

st.set_page_config(layout="wide", page_title="NeuroCompare AI")

st.title("🧠 NeuroCompare AI")
st.subheader("Human Impact Track: HI-01-Medical Image Diagnostic Assistant")
st.write("Longitudinal Image Alignment & Progression Tracking System")

st.divider()

# File Upload columns
col1, col2 = st.columns(2)
with col1:
    file_base = st.file_uploader("Upload Baseline Scan (T1)", type=["png", "jpg", "jpeg"])
with col2:
    file_follow = st.file_uploader("Upload Follow-up Scan (T2)", type=["png", "jpg", "jpeg"])

if file_base and file_follow:
    # Decode uploaded files into OpenCV format
    file_bytes_1 = np.asarray(bytearray(file_base.read()), dtype=np.uint8)
    file_bytes_2 = np.asarray(bytearray(file_follow.read()), dtype=np.uint8)
    
    img_base = cv2.imdecode(file_bytes_1, cv2.IMREAD_COLOR)
    img_follow = cv2.imdecode(file_bytes_2, cv2.IMREAD_COLOR)
    
    # Execution Column Layout
    st.success("Scans uploaded successfully! Ready for analysis.")
    
    if st.button("⚡ Run Longitudinal Alignment & Compare"):
        with st.spinner("Executing spatial registration algorithms..."):
            # Run Backend Logic
            aligned_img = align_and_register(img_base, img_follow)
            overlay_map, growth_metric = compute_progression(img_base, aligned_img)
            
        # Display Results Side-by-Side
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.image(img_base, caption="1. Baseline Scan (Original)", use_container_width=True)
        with res_col2:
            st.image(img_follow, caption="2. Raw Follow-up (Misaligned/Tilted)", use_container_width=True)
        with res_col3:
            st.image(overlay_map, caption="3. AI Progression Map (Red = Structural Change)", use_container_width=True)
            
        # Display Analytics Cards
        st.divider()
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="Volumetric Tissue Alteration", value=f"{growth_metric:.2f}%")
        with metric_col2:
            status = "Significant Progression Detected" if growth_metric > 2.0 else "Stable Disease Structure"
            st.info(f"Clinical Insight: {status}")
