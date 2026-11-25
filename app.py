import tempfile
from pathlib import Path
from ultralytics import YOLO
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")
st.set_page_config(
    page_title="Bird-Aware UAV",
    page_icon="🛩️",
    layout="wide"
)

st.sidebar.title("🛩️ Bird-Aware UAV")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Bird Detection", "Bird Tracking",
     "Path Planning", "Route & Energy Analysis"]
)

st.title("AI-Based Bird-Aware UAV System")
st.caption("Bird Detection | Tracking | Path Optimization")

if page == "Home":
    st.header("Project Overview")
    st.write("An AI-based software system for bird-aware UAV path planning and energy analysis.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Video Frames", "488")
    c2.metric("Bird Detection Frames", "268")
    c3.metric("Total Detections", "982")

    st.info("Welcome to the Bird-Aware UAV decision-support dashboard.")


elif page == "Bird Detection":
    st.header("Bird Detection")

    uploaded_video = st.file_uploader(
        "Upload a new bird video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        st.success("Video uploaded successfully!")

        if st.button("Run Bird Detection"):
            model = load_model()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
                temp.write(uploaded_video.read())
                video_path = temp.name

            st.info("Detecting birds... Please wait.")

            results = model.predict(
                source=video_path,
                classes=[14],
                conf=0.25,
                save=True,
                project="runs/detect",
                name="bird_result",
                exist_ok=True,
                stream=True
            )

            total_frames = 0
            total_birds = 0

            for result in results:
                total_frames += 1
                total_birds += len(result.boxes)

            st.success("Detection completed!")
            st.metric("Frames Processed", total_frames)
            st.metric("Bird Detections", total_birds)

    st.subheader("Previous Detection Result")

    if os.path.exists("bird_detection_final.mp4"):
        st.video("bird_detection_final.mp4")
elif page == "Bird Tracking":
    st.header("Bird Tracking")
    if os.path.exists("bird_tracking_final.mp4"):
        st.video("bird_tracking_final.mp4")
    else:
        st.warning("Tracking video not found.")

elif page == "Path Planning":
    st.header("UAV Path Planning")

    birds = [(35,38), (55,48), (72,80), (45,65)]

    routes = {
        "Original": [(0,0), (100,100)],
        "Manual": [(0,0), (20,90), (100,100)],
        "Optimized": [(0,0), (60,35), (100,100)]
    }

    fig, ax = plt.subplots()

    for name, points in routes.items():
        x, y = zip(*points)
        ax.plot(x, y, marker="o", label=name)

    bx, by = zip(*birds)
    ax.scatter(bx, by, color="red", marker="x", label="Simulated birds")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("UAV Route Comparison")
    ax.legend()
    ax.grid()

    st.pyplot(fig)

elif page == "Route & Energy Analysis":
    st.header("Route and Energy Analysis")

    if os.path.exists("route_comparison.csv"):
        data = pd.read_csv("route_comparison.csv")
        st.dataframe(data, use_container_width=True)

        st.subheader("Energy Comparison")
        st.bar_chart(data.set_index("Route")["Energy"])
    else:
        st.warning("Route comparison file not found.")

st.sidebar.divider()
st.sidebar.caption("UAV Path Planning Project | 2026")
