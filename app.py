
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

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
        st.video(uploaded_video)

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
