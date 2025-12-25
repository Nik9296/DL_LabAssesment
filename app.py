import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
from datetime import datetime
from PIL import Image
import tempfile
import os

# --------------------------------------------------
# Advanced Configuration & State
# --------------------------------------------------
st.set_page_config(page_title="GuardianAI Pro", layout="wide", page_icon="🛡️")

if 'violation_log' not in st.session_state:
    st.session_state.violation_log = pd.DataFrame(columns=['Timestamp', 'Event', 'Confidence'])

# --------------------------------------------------
# CSS Injection for "Dark Mode" Enterprise UI
# --------------------------------------------------
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .metric-card {
        background: #1e2130; padding: 20px; border-radius: 10px;
        border: 1px solid #31333f; transition: 0.3s;
    }
    .metric-card:hover { border-color: #4f46e5; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Optimized Model Loading
# --------------------------------------------------
@st.cache_resource
def get_model():
    # Load model and verify classes
    model = YOLO("best_face_mask.pt") 
    return model

model = get_model()

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
with st.sidebar:
    st.title("🛡️ GuardianAI v3.0")
    st.info("Enterprise Mask Compliance System")
    
    app_mode = st.selectbox("Switch Workspace", 
                           ["Live Surveillance", "Violation Analytics", "Batch Audit"])
    
    conf = st.slider("Detection Confidence", 0.25, 1.0, 0.5)
    st.divider()
    
    if st.button("Export Violation Log (.csv)"):
        csv = st.session_state.violation_log.to_csv(index=False)
        st.download_button("Download Report", csv, "mask_violations.csv", "text/csv")

# --------------------------------------------------
# Workspace 1: Live Surveillance (with Tracking)
# --------------------------------------------------
if app_mode == "Live Surveillance":
    st.subheader("📡 Real-Time Intelligence Feed")
    
    col_vid, col_stat = st.columns([3, 1])
    
    source = st.radio("Input Source", ["Webcam", "Video File"], horizontal=True)
    
    run = st.checkbox("Initialize Stream", value=True)
    frame_placeholder = col_vid.empty()
    
    if source == "Video File":
        v_file = st.file_uploader("Upload MP4", type=['mp4', 'avi'])
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(v_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            cap = None
    else:
        cap = cv2.VideoCapture(0)

    # Metrics Placeholders
    m_count = col_stat.empty()
    m_safe = col_stat.empty()
    m_risk = col_stat.empty()

    

    while run and cap is not None and cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Advanced Tracking (BoT-SORT)
        # persistence=True keeps the ID even if person is briefly occluded
        results = model.track(frame, persist=True, conf=conf, iou=0.5, tracker="bytetrack.yaml")[0]
        
        annotated_frame = results.plot()

        # Logic for Loggin Violations
        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "without_mask":
                new_entry = pd.DataFrame([[datetime.now().strftime("%H:%M:%S"), "No Mask", float(box.conf[0])]], 
                                         columns=['Timestamp', 'Event', 'Confidence'])
                st.session_state.violation_log = pd.concat([st.session_state.violation_log, new_entry], ignore_index=True)

        # Update Metrics UI
        counts = Counter([model.names[int(c)] for c in results.boxes.cls]) if results.boxes.cls is not None else {}
        
        m_count.metric("Total People", len(results.boxes))
        m_safe.metric("Compliant", counts.get('with_mask', 0), delta_color="normal")
        m_risk.metric("Violations", counts.get('without_mask', 0), delta="- Critical", delta_color="inverse")

        frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

    if cap: cap.release()

# --------------------------------------------------
# Workspace 2: Violation Analytics (Plotly BI)
# --------------------------------------------------
elif app_mode == "Violation Analytics":
    st.subheader("📊 Compliance Trends & Historical Data")
    
    if st.session_state.violation_log.empty:
        st.warning("No data logged yet. Please run Surveillance mode first.")
    else:
        log_df = st.session_state.violation_log
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("Recent Violation Log")
            st.dataframe(log_df.tail(10), use_container_width=True)
            
        with c2:
            # Time-series of violations
            fig = px.line(log_df, x='Timestamp', title="Violation Frequency Over Time",
                         template="plotly_dark", color_discrete_sequence=['#ef4444'])
            st.plotly_chart(fig, use_container_width=True)

        # Compliance Distribution
        counts = log_df['Event'].value_counts().reset_index()
        fig_pie = px.pie(counts, values='count', names='Event', hole=0.5, title="Event Distribution")
        st.plotly_chart(fig_pie)

# --------------------------------------------------
# Workspace 3: Batch Audit (Drag & Drop)
# --------------------------------------------------
elif app_mode == "Batch Audit":
    st.subheader(" Mass Audit System")
    files = st.file_uploader("Upload Audit Images", accept_multiple_files=True)
    
    if files:
        cols = st.columns(3)
        for idx, file in enumerate(files):
            img = Image.open(file)
            res = model(img, conf=conf)[0]
            with cols[idx % 3]:
                st.image(res.plot(), caption=f"Audit: {file.name}", use_container_width=True)