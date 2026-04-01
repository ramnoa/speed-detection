import streamlit as st
import cv2
import tempfile
import os
from ultralytics import solutions

st.set_page_config(page_title="Real-Time Traffic Speed Monitor", layout="wide")
st.title("🚦 Real-Time Multi-Car Speed Detection")
st.caption("Powered by AI Computer Vision | YOLO Model")

# ------------------ SIDEBAR ------------------
st.sidebar.title("Control Panel")
speed_limit = st.sidebar.slider("Speed Limit (km/h)", 40, 120, 80)

# ------------------ VIDEO UPLOAD ------------------
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_file:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.video(video_path)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_file = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

    # ------------------ INIT SPEED ESTIMATOR ------------------
    model_file = "yolo26n.pt"
    if not os.path.exists(model_file):
        st.error(f"YOLO model not found at {model_file}")
        st.stop()

    speedestimator = solutions.SpeedEstimator(
        show=False,
        model=model_file,
        fps=fps,
        max_speed=150,
        max_hist=10,
        meter_per_pixel=0.05,
        classes=[2],  # cars only
        line_width=3,
    )

    st.info("Processing video... This may take a while.")
    prev_speeds = {}
    stframe = st.empty()
    progress_bar = st.progress(0)
    processed_frames = 0

    # Sidebar overspeed alerts container
    alert_container = st.sidebar.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame_resized = cv2.resize(frame, (640, int(640 * h / w)))
        results = speedestimator(frame_resized)

        overspeeding_cars = []

        if hasattr(results, "speed"):
            # Use idx for vertical positioning, independent of obj_id format
            for idx, (obj_id, speed) in enumerate(results.speed.items()):

                # Smooth speed
                if obj_id in prev_speeds:
                    speed = (prev_speeds[obj_id] + speed) / 2
                prev_speeds[obj_id] = speed

                # Box color
                color = (0, 255, 0) if speed <= speed_limit else (0, 0, 255)
                if speed > speed_limit:
                    overspeeding_cars.append(f"Car {obj_id}: {speed:.1f} km/h")

                # Overlay speed and ID using idx for Y-position
                cv2.putText(
                    results.plot_im,
                    f"Car {obj_id}: {speed:.1f} km/h",
                    (50, 50 + 30 * idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )

        # Update sidebar alerts
        if overspeeding_cars:
            alert_container.warning("🚨 Overspeeding Cars:\n" + "\n".join(overspeeding_cars))
        else:
            alert_container.info("No overspeeding cars")

        # Write frame to output video
        out.write(results.plot_im)

        # Display live frame
        stframe.image(results.plot_im, channels="BGR")

        processed_frames += 1
        progress_bar.progress(min(processed_frames / frame_count, 1.0))

    cap.release()
    out.release()

    st.success("✅ Video processing complete!")
    st.video(output_file)
    st.download_button(
        label="Download Processed Video",
        data=open(output_file, "rb"),
        file_name="processed_video.mp4",
        mime="video/mp4",
    )