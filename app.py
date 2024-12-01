import streamlit as st
from ultralytics import YOLO
import os
import subprocess
import ffmpeg

# Helper function to save uploaded file
def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


# Function to run YOLO and convert output to .mp4
def run_yolo_and_convert_to_mp4(model_path, input_video_path, output_dir, confidence, selected_classes):
    # Load the YOLO model
    model = YOLO(model_path)

    # Predict and save the results
    results = model.predict(
        source=input_video_path,
        conf=confidence,
        save=True,
        project=output_dir,
        name="run",
        classes=selected_classes
    )

    # Define the paths
    output_avi_path = os.path.join(output_dir, "run", os.path.basename(input_video_path).replace(".mp4", ".avi"))
    output_mp4_path = output_avi_path.replace(".avi", ".mp4")

    # Convert .avi to .mp4
    if os.path.exists(output_avi_path):
        try:
            # Use ffmpeg-python to convert .avi to .mp4
            (
                ffmpeg
                .input(output_avi_path)
                .output(output_mp4_path, vcodec='libx264', preset='fast', crf=22)
                .run()
            )
            os.remove(output_avi_path)  # Remove the .avi file
        except ffmpeg.Error as e:
            st.error(f"Error during video conversion: {e}")
            return None

    return output_mp4_path

# App Introduction
st.title("YOLO Ultralytics Object Detection Web App")

st.write("""
This interactive web application demonstrates the capabilities of the **COCO dataset** in conjunction with **YOLO models**. 
Users can explore and compare the performance of various YOLO Ultralytics models (e.g., yolov8n, yolov8s, yolov8m) 
by uploading images or videos and selecting specific classes for object detection.
""")

# Step 1: Model selection
model_choices = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
model_name = st.selectbox("Choose a YOLO model:", model_choices)

# Step 2: Class selection
st.write("### Select Classes to Filter")
model = YOLO(model_name)  # Load model temporarily to get classes
all_classes = {int(k): v for k, v in model.names.items()}
selected_class_names = st.multiselect("Select the classes you want to detect:", list(all_classes.values()))
selected_class_ids = [k for k, v in all_classes.items() if v in selected_class_names]

# Step 3: Confidence threshold
confidence_threshold = st.slider("Confidence Threshold:", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# Step 4: Upload a video or image
uploaded_file = st.file_uploader("Upload an Image or Video (Max 50MB):", type=["jpg", "jpeg", "png", "mp4"])

# Ensure file size limits
if uploaded_file and uploaded_file.size > 50 * 1024 * 1024:
    st.error("File size exceeds 50MB limit. Please upload a smaller file.")
    uploaded_file = None

# Step 5: Perform detection
if uploaded_file and selected_class_names:
    # Save the uploaded file temporarily
    input_path = os.path.join("temp_input", uploaded_file.name)
    os.makedirs("temp_input", exist_ok=True)
    save_uploaded_file(uploaded_file, input_path)

    # Output directory for annotated files
    output_dir = "temp_output"
    os.makedirs(output_dir, exist_ok=True)

    # Run YOLO and convert output to .mp4
    st.write("### Running detection...")
    with st.spinner("Detecting objects. This might take a while..."):
        output_video = run_yolo_and_convert_to_mp4(
            model_path=model_name,
            input_video_path=input_path,
            output_dir=output_dir,
            confidence=confidence_threshold,
            selected_classes=selected_class_ids
        )

    # Display and provide a download button for the output
    if output_video:
        st.write("### Annotated Output")
        st.video(output_video)

        with open(output_video, "rb") as file:
            st.download_button(
                label="Download Annotated Video",
                data=file,
                file_name=f"annotated_{uploaded_file.name.replace('.avi', '.mp4')}",
                mime="video/mp4"
            )
    else:
        st.error("Failed to generate the annotated video. Please try again.")

    import shutil

    # Cleanup temporary files
    if os.path.exists("temp_input"):
        shutil.rmtree("temp_input", ignore_errors=True)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

