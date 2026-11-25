import subprocess
import imageio_ffmpeg
import cv2
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

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_video.name).suffix
            ) as temp:
                temp.write(uploaded_video.getvalue())
                video_path = temp.name

            st.info("Detecting birds... Please wait.")

            cap = cv2.VideoCapture(video_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap.release()

            if width <= 0 or height <= 0:
                st.error("Unable to read the uploaded video.")
            else:
                output_path = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ).name

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps if fps > 0 else 25,
                    (width, height)
                )

                results = model.predict(
                    source=video_path,
                    classes=[14],
                    conf=0.25,
                    stream=True,
                    save=False
                )

                total_frames = 0
                total_birds = 0

                for result in results:
                    total_frames += 1
                    total_birds += len(result.boxes)
                    writer.write(result.plot())

                writer.release()

                st.success("Detection completed!")

                st.metric("Frames Processed", total_frames)
                st.metric("Bird Detections", total_birds)

                st.subheader("Converting video...")

                converted_path = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ).name

                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

                conversion = subprocess.run(
                    [
                        ffmpeg_exe,
                        "-y",
                        "-i", output_path,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        converted_path
                    ],
                    capture_output=True,
                    text=True
                )

                if conversion.returncode == 0:
                    st.success("Video conversion successful!")
                    st.subheader("Annotated Bird Detection Video")
                    st.video(converted_path)
                else:
                    st.error("Video conversion failed.")
                    st.code(conversion.stderr[-1500:])

    st.subheader("Previous Detection Result")

    if os.path.exists("bird_detection_final.mp4"):
        st.video("bird_detection_final.mp4")


elif page == "Bird Tracking":
    st.header("Bird Tracking")

    st.write("Track detected birds across video frames using YOLO and ByteTrack.")

    tracking_video = st.file_uploader(
        "Upload video for bird tracking",
        type=["mp4", "avi", "mov"],
        key="tracking_video"
    )

    if tracking_video:
        if st.button("Run Bird Tracking"):
            model = load_model()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(tracking_video.name).suffix
            ) as temp:
                temp.write(tracking_video.getvalue())
                video_path = temp.name

            st.info("Tracking birds... Please wait.")

            results = model.track(
                source=video_path,
                classes=[14],
                conf=0.25,
                tracker="bytetrack.yaml",
                stream=True,
                persist=True,
                save=False
            )

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            output_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ).name

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps if fps > 0 else 25,
                (width, height)
            )

            total_frames = 0
            unique_ids = set()

            for result in results:
                total_frames += 1

                if result.boxes.id is not None:
                    ids = result.boxes.id.int().cpu().tolist()
                    unique_ids.update(ids)

                writer.write(result.plot())

            writer.release()

            st.success("Tracking completed!")

            st.metric("Frames Processed", total_frames)
            st.metric("Unique Tracking IDs", len(unique_ids))

            converted_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ).name

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            conversion = subprocess.run(
                [
                    ffmpeg_exe, "-y",
                    "-i", output_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    converted_path
                ],
                capture_output=True,
                text=True
            )

            if conversion.returncode == 0:
                st.subheader("Bird Tracking Video")
                st.video(converted_path)
            else:
                st.error("Tracking video conversion failed.")
                st.code(conversion.stderr[-1500:])


elif page == "Path Planning":
    st.header("UAV Path Planning")

    st.write("Adjust bird positions and clearance to compare UAV routes.")

    st.subheader("1. Bird Positions")

    default_birds = [(35, 38), (55, 48), (72, 80), (45, 65)]
    birds = []

    for i, (default_x, default_y) in enumerate(default_birds):
        col1, col2 = st.columns(2)

        with col1:
            bx = st.number_input(
                f"Bird {i+1} X coordinate",
                min_value=0.0,
                max_value=100.0,
                value=float(default_x),
                key=f"bird_x_{i}"
            )

        with col2:
            by = st.number_input(
                f"Bird {i+1} Y coordinate",
                min_value=0.0,
                max_value=100.0,
                value=float(default_y),
                key=f"bird_y_{i}"
            )

        birds.append((bx, by))

    clearance_limit = st.slider(
        "Minimum assumed clearance (m)",
        min_value=1,
        max_value=30,
        value=10
    )

    original = [(0, 0), (100, 100)]
    manual = [(0, 0), (20, 90), (100, 100)]

    def point_segment_distance(point, start, end):
        px, py = point
        x1, y1 = start
        x2, y2 = end

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)

        t = ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))

        closest_x = x1 + t*dx
        closest_y = y1 + t*dy

        return math.hypot(px-closest_x, py-closest_y)

    def route_distance(route):
        distance = 0
        for i in range(len(route)-1):
            distance += math.dist(route[i], route[i+1])
        return distance

    def route_clearance(route, bird_positions):
        distances = []

        for bird in bird_positions:
            for i in range(len(route)-1):
                distances.append(
                    point_segment_distance(
                        bird, route[i], route[i+1]
                    )
                )

        return min(distances) if distances else float("inf")

    st.subheader("2. Automatic Route Optimization")

    candidates = []

    for x in range(5, 100, 5):
        for y in range(5, 100, 5):
            candidate = [(0, 0), (x, y), (100, 100)]

            distance = route_distance(candidate)
            clearance = route_clearance(candidate, birds)

            if clearance >= clearance_limit:
                candidates.append((distance, candidate, clearance))

    routes = {
        "Original": original,
        "Manual": manual
    }

    if candidates:
        best = min(candidates, key=lambda item: item[0])
        routes["Optimized"] = best[1]
        st.success("A feasible optimized route was found!")
    else:
        st.warning(
            "No feasible route found with the current clearance. "
            "Try adjusting bird positions or the clearance threshold."
        )

    st.subheader("3. Route Comparison")

    for name, route in routes.items():
        distance = route_distance(route)
        clearance = route_clearance(route, birds)
        feasible = clearance >= clearance_limit

        st.write(f"**{name}**")
        st.write(f"Distance: {distance:.2f} m")
        st.write(f"Minimum clearance: {clearance:.2f} m")
        st.write(f"Meets assumed clearance: {'Yes' if feasible else 'No'}")

    st.subheader("4. Dynamic UAV Route Graph")

    fig, ax = plt.subplots(figsize=(9, 6))

    for name, route in routes.items():
        x, y = zip(*route)
        ax.plot(x, y, marker="o", label=name)

    bx, by = zip(*birds)
    ax.scatter(bx, by, color="red", marker="x",
               s=100, label="Simulated Birds")

    ax.scatter(0, 0, color="green", s=100, label="Start")
    ax.scatter(100, 100, color="black", s=100, label="Destination")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Dynamic UAV Path Planning")
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
