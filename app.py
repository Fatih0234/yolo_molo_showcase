import streamlit as st
from ultralytics import YOLO
import os
import shutil

# Function to display author credits
def display_credits():
    st.markdown("---")
    st.markdown("### About this Web App")
    st.write("This web application was developed by [Fatih Karahan].")
    
    # Contact and social links
    st.markdown("""
    **Contact Details:**

    - 📧 Email: [sekanti02@gmail.com](mailto:sekanti02@gmail.com)
    - 💼 LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/fatih-karahan-717931193/)
    - 🐙 GitHub: [GitHub Repository](https://github.com/Fatih0234/yolo_molo_showcase)
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

📝 **Pro Tip:** Start by selecting a model, then choose classes, adjust confidence, and finally upload your file.
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

# Step 3: Confidence threshold
st.write("### Step 3: Set Confidence Threshold 🎛️")
confidence_threshold = st.slider(
    "Confidence Threshold:",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Adjust the confidence level for object detection."
)

# Step 4: Upload a video or image
st.write("### Step 4: Upload Your File 📂")
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

# Display credits at the bottom
display_credits()
