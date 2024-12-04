import streamlit as st
from ultralytics import YOLO
import os
import shutil
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import numpy as np

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
            {"urls": "stun:stun2.l.google.com:19302"},
            {"urls": "stun:stun3.l.google.com:19302"},
            {"urls": "stun:stun4.l.google.com:19302"}
        ]
    }
)

# Updated YOLOVideoTransformer class
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self, model, confidence, selected_classes):
        self.model = model
        self.confidence = confidence
        self.selected_classes = selected_classes
        

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to NumPy array

        # Run YOLO inference
        results = self.model.predict(
            source=img,
            conf=self.confidence,
            classes=self.selected_classes
        )

        # Annotate frame with detection results
        annotated_frame = results[0].plot()
        return annotated_frame


def display_credits():
    st.markdown("---")
    st.markdown("### About this Web App")
    st.write("This web application was developed by [Fatih Karahan](https://portfolio-app-fatihkarahan.streamlit.app/).")
    
    # Contact and social links
    st.markdown("""
    **Contact Details:**

    - 📧 Email: [sekanti02@gmail.com](mailto:sekanti02@gmail.com)
    - 💼 LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/fatih-karahan-717931193/)
    - 🐙 GitHub: [GitHub Repository](https://github.com/Fatih0234/yolo_molo_showcase)
    - 🌐 Portfolio: [Portfolio Website](https://portfolio-app-fatihkarahan.streamlit.app/)
    """)
    st.write("Feel free to reach out for feedback, suggestions, or collaborations!")
    st.markdown("---")

# Helper function to save uploaded file
def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def run_yolo_and_process(model_path, input_file_path, output_dir, confidence, selected_classes, file_type):
    # Define the run folder path
    run_folder = os.path.join(output_dir, "run")
    
    # Remove the run folder if it exists
    if os.path.exists(run_folder):
        shutil.rmtree(run_folder)
    
    # Load YOLO model
    model = YOLO(model_path)

    # Run detection and save results
    results = model.predict(
        source=input_file_path,
        conf=confidence,
        save=True,
        project=output_dir,
        name="run",  # Always use the same folder name
        classes=selected_classes
    )

    # Handle different file types
    if file_type == "video":
        # Process video
        output_avi_path = os.path.abspath(os.path.join(run_folder, os.path.basename(input_file_path).replace(".mp4", ".avi")))
        output_mp4_path = output_avi_path.replace(".avi", ".mp4")

        if not os.path.exists(output_avi_path):
            raise FileNotFoundError(f"AVI file not found: {output_avi_path}")

        try:
            # Convert to MP4
            import subprocess
            subprocess.run(
                [
                    "ffmpeg",
                    "-i", output_avi_path,
                    "-vcodec", "libx264",
                    "-preset", "fast",
                    "-crf", "22",
                    output_mp4_path
                ],
                check=True
            )
            os.remove(output_avi_path)  # Cleanup AVI file
            return output_mp4_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg conversion failed: {e}")

    elif file_type == "image":
        # Process image
        output_image_path = os.path.abspath(os.path.join(run_folder, os.path.basename(input_file_path)))
        if not os.path.exists(output_image_path):
            raise FileNotFoundError(f"Processed image not found: {output_image_path}")
        return output_image_path


# Initialize Streamlit session state
if "processed_file" not in st.session_state:
    st.session_state["processed_file"] = None
if "uploaded_file_path" not in st.session_state:
    st.session_state["uploaded_file_path"] = None

# App Introduction
st.title("YOLO Ultralytics Object Detection Playground 🎯")

st.write("""
Welcome to the **YOLO Ultralytics Object Detection Playground**! This web app allows you to:
1. **Select YOLO Models**: Choose from lightweight models for small devices or powerful models for high accuracy.
2. **Filter Classes**: Specify the object classes you want to detect from the COCO dataset.
3. **Set Confidence Levels**: Adjust the detection confidence threshold to fine-tune the results.
4. **Upload Files**: Upload images or videos to see YOLO in action under different settings.
5. **Live Camera Detection**: Use your device's camera to see real-time object detection (browser compatibility required).

📝 **Modes Available:**
- **File Upload Mode**: Upload an image or video (up to 50MB) to run object detection and download the annotated results.
- **Real-Time Camera Mode**: Use your camera for live detection. Please note that this mode might not work on certain browsers or devices. If you face issues, consider switching to a different browser or device and be patient while the camera initializes.

**Pro Tip:** Start by selecting a model, then choose classes, adjust confidence, and finally upload your file or switch to live camera mode for real-time detection.
""")


# Step 1: Model selection
st.write("### Step 1: Choose a YOLO Model 🔍")
model_choices = [
    "yolov8n.pt - Small model, recommended ✅",
    "yolov8s.pt - Medium model",
    "yolov8m.pt - Larger model",
    "yolov8l.pt - High accuracy, resource-intensive",
    "yolov8x.pt - Very high accuracy, requires significant processing power"
]
model_name = st.selectbox(
    "Select a YOLO model to begin:", 
    model_choices,
    help="Select a model based on your device capabilities and accuracy needs."
)

model_path = model_name.split(" - ")[0].strip()  # Extract the actual model name
# Load the YOLO model once
yolo_model = YOLO(model_path)
# Step 2: Class selection
st.write("### Step 2: Select Classes to Filter 🏷️")
model = YOLO(model_path)  # Load model temporarily to get classes
all_classes = {int(k): v for k, v in model.names.items()}
selected_class_names = st.multiselect(
    "Select the classes you want to detect:",
    list(all_classes.values()),
    help="Choose specific object classes to narrow down the detection."
)
selected_class_ids = [k for k, v in all_classes.items() if v in selected_class_names]

st.write("### Detection Parameters")

# Confidence slider on the main page
confidence_threshold = st.slider(
    "Confidence Threshold:",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Adjust the confidence level for object detection."
)


mode = st.sidebar.radio(
    "Select Mode:",
    options=["File Upload", "Real-Time Camera"],
    help="Choose between uploading files for detection or using your camera in real-time."
)

if mode == "File Upload":
    st.write("### File Upload Mode 📂")
    uploaded_file = st.file_uploader(
        "Upload an Image or Video (Max 50MB):",
        type=["jpg", "jpeg", "png", "mp4"],
        help="Upload a photo or video to see the YOLO model in action."
    )
    # Check if a new file is uploaded
    if uploaded_file and (st.session_state["uploaded_file_path"] != uploaded_file.name):
        # Save the uploaded file temporarily
        input_path = os.path.join("temp_input", uploaded_file.name)
        os.makedirs("temp_input", exist_ok=True)
        save_uploaded_file(uploaded_file, input_path)
        st.session_state["uploaded_file_path"] = uploaded_file.name

        # Output directory for annotated files
        output_dir = "temp_output"
        os.makedirs(output_dir, exist_ok=True)

        # Determine file type (image or video)
        file_ext = uploaded_file.name.split(".")[-1].lower()
        file_type = "image" if file_ext in ["jpg", "jpeg", "png"] else "video"

        # Run YOLO and process the file
        st.write("### Running detection... 🚀")
        with st.spinner("Detecting objects. This might take a while..."):
            processed_file = run_yolo_and_process(
                model_path=model_path,
                input_file_path=input_path,
                output_dir=output_dir,
                confidence=confidence_threshold,
                selected_classes=selected_class_ids,
                file_type=file_type
            )
            st.session_state["processed_file"] = processed_file

    # Display results only if a file has been uploaded and processed
    if st.session_state["processed_file"]:
        if os.path.exists(st.session_state["processed_file"]):
            file_ext = st.session_state["processed_file"].split(".")[-1].lower()
            if file_ext in ["mp4"]:
                st.write("### Annotated Output Video 🎥")
                st.video(st.session_state["processed_file"])
            else:
                st.write("### Annotated Output Image 🖼️")
                st.image(st.session_state["processed_file"])

            with open(st.session_state["processed_file"], "rb") as file:
                st.download_button(
                    label="Download Annotated File 📥",
                    data=file,
                    file_name=f"annotated_{st.session_state['uploaded_file_path']}",
                    mime="video/mp4" if file_ext == "mp4" else "image/jpeg"
                )
        else:
            st.error("Processed file not found. Please try again.")
    elif uploaded_file is None:
        pass

    # Cleanup temporary files
    if st.session_state["processed_file"] is None and st.session_state["uploaded_file_path"] is None:
        if os.path.exists("temp_input"):
            shutil.rmtree("temp_input", ignore_errors=True)
        if os.path.exists("temp_output"):
            shutil.rmtree("temp_output", ignore_errors=True)

elif mode == "Real-Time Camera":
    st.write("### Real-Time Camera Detection 🎥")
    # Warning message for users
    st.warning("""
    ⚠️ **Important Note:**
    - The live camera mode may not work on certain browsers or devices due to browser compatibility or permissions.
    - If you experience issues, try switching to a different browser or device.
    - Please be patient while the camera initializes, as it may take a moment to load.
    """)

    # Proceed with WebRTC setup
    webrtc_streamer(
        key="realtime-yolo",
        video_processor_factory=lambda: YOLOVideoTransformer(
            model=yolo_model,
            confidence=confidence_threshold,
            selected_classes=selected_class_ids,
        ),
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )


# Display credits at the bottom
display_credits()
