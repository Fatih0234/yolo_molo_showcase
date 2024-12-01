import streamlit as st
from ultralytics import YOLO
import os
import shutil

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
    st.write("### Running detection...")
    with st.spinner("Detecting objects. This might take a while..."):
        processed_file = run_yolo_and_process(
            model_path=model_name,
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
            st.write("### Annotated Output Video")
            st.video(st.session_state["processed_file"])
        else:
            st.write("### Annotated Output Image")
            st.image(st.session_state["processed_file"])

        with open(st.session_state["processed_file"], "rb") as file:
            st.download_button(
                label="Download Annotated File",
                data=file,
                file_name=f"annotated_{st.session_state['uploaded_file_path']}",
                mime="video/mp4" if file_ext == "mp4" else "image/jpeg"
            )
    else:
        st.error("Processed file not found. Please try again.")
elif uploaded_file is None:
    # Do not show any error message if no file has been uploaded
    pass


# Cleanup temporary files
if st.session_state["processed_file"] is None and st.session_state["uploaded_file_path"] is None:
    if os.path.exists("temp_input"):
        shutil.rmtree("temp_input", ignore_errors=True)

    if os.path.exists("temp_output"):
        shutil.rmtree("temp_output", ignore_errors=True)
        
        
