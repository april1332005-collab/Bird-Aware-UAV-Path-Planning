
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Bird-Aware UAV", layout="wide")

st.title("🛩️ Bird-Aware UAV Path Planning")
st.caption("AI-Based Bird Detection, Tracking and Path Optimization")

folder = "."

st.header("1. Bird Detection and Tracking")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Bird Detection")
    if os.path.exists("bird_detection_final.mp4"):
        st.video("bird_detection_final.mp4")

with c2:
    st.subheader("Bird Tracking")
    if os.path.exists("bird_tracking_final.mp4"):
        st.video("bird_tracking_final.mp4")

st.header("2. UAV Path Optimization")

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

st.header("3. Route Comparison")

if os.path.exists("route_comparison.csv"):
    data = pd.read_csv("route_comparison.csv")
    st.dataframe(data, use_container_width=True)

st.info("Bird positions in the path-planning simulation are illustrative 2D coordinates, separate from video detections.")
