import math
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
    st.title("AI-Based Bird-Aware UAV Path Planning")

    st.write(
        "An AI-based software prototype for bird detection, "
        "tracking and UAV route planning."
    )

    st.divider()

    # COMMON VIDEO UPLOAD
    st.subheader("Upload Project Video")

    uploaded_video = st.file_uploader(
        "Upload your video here (only once)",
        type=["mp4", "avi", "mov"],
        key="common_video_uploader"
    )

    if uploaded_video is not None:
        st.session_state["uploaded_video_bytes"] = (
            uploaded_video.getvalue()
        )
        st.session_state["uploaded_video_name"] = (
            uploaded_video.name
        )

    if "uploaded_video_bytes" in st.session_state:
        st.success(
            f"Video ready: "
            f"{st.session_state['uploaded_video_name']}"
        )
        st.info(
            "This video will be used by both Bird Detection "
            "and Bird Tracking."
        )
    else:
        st.warning("Upload a video to begin the project.")

    st.divider()

    # PROJECT OVERVIEW
    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Bird Tracking IDs",
            len(st.session_state.get("tracked_birds", []))
        )

    with col2:
        route_data = st.session_state.get(
            "route_comparison_data", []
        )
        st.metric("Routes Compared", len(route_data))

    with col3:
        route_available = any(
            r["Route"] in ["Optimized", "Best Available"]
            for r in route_data
        )

        st.metric(
            "Optimized Route",
            "Available" if route_available else "Pending"
        )

    st.divider()

    # WORKFLOW
    st.subheader("System Workflow")

    st.markdown("""
    1. **Video Upload** – Upload the video once.
    2. **Bird Detection** – YOLOv8 identifies birds.
    3. **Bird Tracking** – ByteTrack assigns IDs and extracts coordinates.
    4. **Path Planning** – Alternative UAV routes are evaluated.
    5. **Energy Analysis** – Estimated route energy is compared.
    """)

    st.divider()

    st.subheader("Project Technology")

    st.write(
        "Python | Streamlit | YOLOv8 | ByteTrack | "
        "OpenCV | Matplotlib | Pandas"
    )

    st.caption(
        "Software simulation prototype – "
        "not connected to a physical UAV."
    )





elif page == "Bird Detection":
    st.header("Bird Detection")

    st.write(
        "YOLOv8 identifies birds in the video uploaded on Home."
    )

    video_bytes = st.session_state.get(
        "uploaded_video_bytes"
    )

    video_name = st.session_state.get(
        "uploaded_video_name"
    )

    if video_bytes is None:
        st.warning("Please upload a video on the Home page first.")

    else:
        st.success(f"Using video: {video_name}")

        if st.button("Run Bird Detection"):

            model = load_model()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(video_name).suffix
            ) as temp:
                temp.write(video_bytes)
                video_path = temp.name

            st.info("Detecting birds... Please wait.")

            results = model.predict(
                source=video_path,
                classes=[14],
                conf=0.25,
                stream=True,
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
            total_birds = 0

            for result in results:
                total_frames += 1

                if result.boxes is not None:
                    total_birds += len(result.boxes)

                writer.write(result.plot())

            writer.release()

            st.success("Bird detection completed!")

            st.metric("Frames Processed", total_frames)
            st.metric("Total Bird Detections", total_birds)

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
                st.subheader("Bird Detection Output")
                st.video(converted_path)
            else:
                st.error("Video conversion failed.")
                st.code(conversion.stderr[-1500:])



elif page == "Bird Tracking":
    st.header("Bird Tracking")

    st.write(
        "YOLOv8 and ByteTrack identify and track birds "
        "using the video uploaded on Home."
    )

    video_bytes = st.session_state.get(
        "uploaded_video_bytes"
    )

    video_name = st.session_state.get(
        "uploaded_video_name"
    )

    if video_bytes is None:
        st.warning("Please upload a video on the Home page first.")

    else:
        st.success(f"Using video: {video_name}")

        if st.button("Run Bird Tracking"):

            model = load_model()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(video_name).suffix
            ) as temp:
                temp.write(video_bytes)
                video_path = temp.name

            st.info("Tracking birds... Please wait.")

            cap = cv2.VideoCapture(video_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap.release()

            if width <= 0 or height <= 0:
                st.error("Unable to read uploaded video.")

            else:
                results = model.track(
                    source=video_path,
                    classes=[14],
                    conf=0.25,
                    tracker="bytetrack.yaml",
                    stream=True,
                    persist=True,
                    save=False
                )

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
                latest_positions = {}

                for result in results:
                    total_frames += 1

                    if result.boxes.id is not None:
                        ids = result.boxes.id.int().cpu().tolist()
                        boxes = result.boxes.xywh.cpu().tolist()

                        unique_ids.update(ids)

                        for track_id, box in zip(ids, boxes):
                            center_x = box[0]
                            center_y = box[1]

                            normalized_x = (
                                center_x / width
                            ) * 100

                            normalized_y = (
                                center_y / height
                            ) * 100

                            latest_positions[track_id] = (
                                normalized_x,
                                normalized_y
                            )

                    writer.write(result.plot())

                writer.release()

                st.session_state["tracked_birds"] = list(
                    latest_positions.values()
                )

                st.success("Tracking completed!")

                st.metric("Frames Processed", total_frames)
                st.metric("Unique Tracking IDs", len(unique_ids))

                st.subheader("Extracted Bird Coordinates")

                st.write(
                    st.session_state["tracked_birds"]
                )

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
                    st.subheader("Bird Tracking Video")
                    st.video(converted_path)

                else:
                    st.error("Tracking video conversion failed.")
                    st.code(conversion.stderr[-1500:])






elif page == "Path Planning":
    st.header("UAV Path Planning")

    st.write(
        "Bird-aware UAV route planning using "
        "coordinates extracted from the tracking video."
    )

    # 1. Check whether tracking has been completed
    if "tracked_birds" not in st.session_state:
        st.warning("Bird tracking has not been completed.")
        st.info(
            "Go to Bird Tracking and run the uploaded video first."
        )
        st.stop()

    birds = st.session_state["tracked_birds"]

    if birds:
        st.success(f"{len(birds)} tracked bird positions loaded!")
    else:
        st.success(
            "No birds detected! Proceeding with normal UAV route planning."
        )

    # 2. Display bird coordinates
    st.subheader("1. Extracted Bird Positions")

    if birds:
        bird_data = []

        for i, bird in enumerate(birds):
            bird_data.append({
                "Bird ID": i + 1,
                "X Coordinate": round(bird[0], 2),
                "Y Coordinate": round(bird[1], 2)
            })

        st.dataframe(bird_data, use_container_width=True)

    else:
        st.info("No bird coordinates available for this video.")

    # 3. Clearance threshold
    clearance_limit = st.slider(
        "Minimum assumed clearance (map units)",
        min_value=1,
        max_value=30,
        value=10
    )

    # 4. Route definitions
    original = [(0, 0), (100, 100)]
    manual = [(0, 0), (20, 90), (100, 100)]

    # 5. Distance from point to route segment
    def point_segment_distance(point, start, end):
        px, py = point
        x1, y1 = start
        x2, y2 = end

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)

        t = (
            (px - x1) * dx + (py - y1) * dy
        ) / (dx * dx + dy * dy)

        t = max(0, min(1, t))

        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return math.hypot(
            px - closest_x,
            py - closest_y
        )

    # 6. Route distance
    def route_distance(route):
        distance = 0

        for i in range(len(route) - 1):
            distance += math.dist(
                route[i],
                route[i + 1]
            )

        return distance

    # 7. Minimum bird clearance
    def route_clearance(route, bird_positions):
        if not bird_positions:
            return None

        distances = []

        for bird in bird_positions:
            for i in range(len(route) - 1):
                distances.append(
                    point_segment_distance(
                        bird,
                        route[i],
                        route[i + 1]
                    )
                )

        return min(distances) if distances else None

    # 8. Automatic route optimization
    st.subheader("2. Automatic Route Optimization")

    candidates = []

    waypoints = [
        (x, y)
        for x in range(10, 100, 10)
        for y in range(10, 100, 10)
    ]

    total_combinations = 0

    for wp1 in waypoints:
        for wp2 in waypoints:

            if wp1 == wp2:
                continue

            candidate = [
                (0, 0),
                wp1,
                wp2,
                (100, 100)
            ]

            total_combinations += 1

            distance = route_distance(candidate)
            clearance = route_clearance(candidate, birds)

            candidates.append(
                (distance, candidate, clearance)
            )

    routes = {
        "Original": original,
        "Manual": manual
    }

    if not birds:
        # No bird obstacles: choose shortest tested route
        best = min(
            candidates,
            key=lambda item: item[0]
        )

        routes["Optimized"] = best[1]

        st.success(
            "Optimized route generated successfully!"
        )

        st.info(
            "No bird obstacles were detected. "
            "The shortest tested route has been selected."
        )

    else:
        feasible_candidates = [
            item for item in candidates
            if item[2] >= clearance_limit
        ]

        if feasible_candidates:
            best = min(
                feasible_candidates,
                key=lambda item: item[0]
            )

            routes["Optimized"] = best[1]

            st.success("Feasible optimized route found!")

            st.write(
                f"Best route clearance: {best[2]:.2f} map units"
            )

        else:
            best = max(
                candidates,
                key=lambda item: (item[2], -item[0])
            )

            routes["Best Available"] = best[1]

            st.warning(
                "No tested route meets the assumed clearance. "
                "Showing the route with greatest clearance."
            )

            st.write(
                f"Best available clearance: {best[2]:.2f} map units"
            )

    st.write(
        f"Waypoint combinations tested: {total_combinations}"
    )

    # 9. Route comparison
    st.subheader("3. Route Comparison")

    comparison_data = []

    for name, route in routes.items():

        distance = route_distance(route)
        clearance = route_clearance(route, birds)

        if clearance is None:
            clearance_display = "N/A - No birds detected"
            feasibility = "No bird obstacles"
        else:
            clearance_display = round(clearance, 2)
            feasibility = (
                "Yes" if clearance >= clearance_limit else "No"
            )

        comparison_data.append({
            "Route": name,
            "Distance (map units)": round(distance, 2),
            "Minimum Clearance (map units)": clearance_display,
            "Meets Assumed Clearance": feasibility
        })

    st.session_state["route_comparison_data"] = comparison_data

    st.dataframe(
        comparison_data,
        use_container_width=True
    )

    # 10. Route graph
    st.subheader("4. Dynamic UAV Route Graph")

    fig, ax = plt.subplots(figsize=(9, 6))

    for name, route in routes.items():
        x, y = zip(*route)
        ax.plot(x, y, marker="o", label=name)

    if birds:
        bx, by = zip(*birds)

        ax.scatter(
            bx, by,
            color="red",
            marker="x",
            s=100,
            label="Tracked Birds"
        )

    else:
        ax.text(
            50, 90,
            "No birds detected",
            ha="center",
            fontsize=11
        )

    ax.scatter(
        0, 0,
        color="green",
        s=100,
        label="Start"
    )

    ax.scatter(
        100, 100,
        color="black",
        s=100,
        label="Destination"
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.set_xlabel("Normalized X Coordinate")
    ax.set_ylabel("Normalized Y Coordinate")

    ax.set_title("Bird-Aware UAV Path Planning")

    ax.legend()
    ax.grid()

    st.pyplot(fig)

    st.caption(
        "2D simulation using normalized image coordinates. "
        "No-bird results indicate no bird obstacles were detected "
        "in the processed video, not guaranteed real-world safety."
    )




elif page == "Route & Energy Analysis":
    st.header("Route and Energy Analysis")

    comparison_data = st.session_state.get(
        "route_comparison_data",
        []
    )

    if comparison_data:

        data = pd.DataFrame(comparison_data)

        # Illustrative energy model
        energy_factor = 1.0

        data["Estimated Energy (units)"] = (
            data["Distance (map units)"] * energy_factor
        ).round(2)

        st.subheader("1. Route Comparison")

        st.dataframe(
            data,
            use_container_width=True
        )

        st.subheader("2. Estimated Energy Comparison")

        energy_chart = data.set_index("Route")[
            "Estimated Energy (units)"
        ]

        st.bar_chart(energy_chart)

        st.subheader("3. Energy Summary")

        for _, row in data.iterrows():
            st.metric(
                f"{row['Route']} Route",
                f"{row['Estimated Energy (units)']:.2f} units"
            )

        if "Optimized" in data["Route"].values:
            original_energy = data.loc[
                data["Route"] == "Original",
                "Estimated Energy (units)"
            ].iloc[0]

            optimized_energy = data.loc[
                data["Route"] == "Optimized",
                "Estimated Energy (units)"
            ].iloc[0]

            if original_energy > 0:
                saving = (
                    (original_energy - optimized_energy)
                    / original_energy
                ) * 100

                st.subheader("4. Energy Saving")

                st.metric(
                    "Estimated Energy Reduction",
                    f"{saving:.2f}%"
                )

    else:
        st.warning(
            "No route data available. "
            "Run Bird Tracking and Path Planning first."
        )

    st.caption(
        "Note: Energy is estimated using 1 illustrative "
        "energy unit per map unit. This is not measured "
        "UAV power or battery consumption."
    )

st.sidebar.divider()
st.sidebar.caption("UAV Path Planning Project | 2026")
