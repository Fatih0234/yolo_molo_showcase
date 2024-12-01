**WebApp**: [WebApp](https://yolomoloshowcase-upwbkf6a3pnycxubxfabxx.streamlit.app/)

# YOLO Ultralytics Object Detection Playground 🎯

Welcome to the **YOLO Ultralytics Object Detection Playground**! This web application allows users to experiment with the YOLO object detection models from Ultralytics. The app is designed to be an interactive playground where users can filter classes, adjust confidence levels, and explore the performance of different YOLO models by uploading images or videos.

---

## 🚀 Features

- **Model Selection:** Choose from lightweight to high-performance YOLO models based on your processing needs.
- **Class Filtering:** Detect specific object classes from the COCO dataset.
- **Confidence Level Adjustment:** Fine-tune the detection threshold to customize results.
- **File Uploads:** Supports both image (`.jpg`, `.jpeg`, `.png`) and video (`.mp4`) formats.
- **Interactive Results:** View annotated outputs directly in the app and download them for further analysis.

---

## 🛠️ Setup and Installation

Follow the steps below to set up the application in your environment:

### 1. Clone the Repository
```bash
git clone https://github.com/Fatih0234/yolo_molo_showcase.git
cd yolo_molo_showcase
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
```
### Activate the Virtual Environment

- **On Windows**:
    ```bash
    venv\Scripts\activate
    ```

- **On Linux/Mac**:
    ```bash
    source venv/bin/activate
    ```

### 3. Install Dependencies

Install the required Python packages from `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Download YOLO Pretrained Models

This app relies on YOLO pretrained models. The default models are automatically downloaded by the `ultralytics` package during the first run.


### 🖥️ Running the Application

Start the Streamlit application by running:

```bash
streamlit run app.py
```

### 🧩 Requirements
## Python
Ensure you have Python 3.8 or higher installed. You can check your Python version using:


```bash
python --version
```

### Required Packages

- `streamlit`
- `ultralytics`
- `shutil`
- `subprocess`
- `ffmpeg` (Required for video processing. See installation instructions below.)

### 🛠️ Installing FFmpeg

FFmpeg is required for video processing in the application. Install it using the following commands:

- **On Windows**:
    1. Download FFmpeg from the [official website](https://ffmpeg.org/download.html).
    2. Add the FFmpeg `bin` folder to your system's PATH.

- **On Linux**:
    ```bash
    sudo apt update
    sudo apt install ffmpeg
    ```

- **On Mac**:
    ```bash
    brew install ffmpeg
    ```

### Verify Installation
Run the following command to verify that FFmpeg is installed correctly:
```bash
ffmpeg -version
```

## 🛡️ Troubleshooting

### Common Issues

1. **Port Conflict:**
   - If port `8501` is busy, specify a different port when running the app:
     ```bash
     streamlit run app.py --server.port 8502
     ```

2. **Missing Dependencies:**
   - Ensure all required dependencies are installed using:
     ```bash
     pip install -r requirements.txt
     ```

3. **FFmpeg Errors:**
   - Ensure FFmpeg is correctly installed and added to your system's PATH.

## 📜 About the Author

This web application was developed by **[Fatih Karahan](https://www.linkedin.com/in/fatih-karahan-717931193/)**. If you have any questions, suggestions, or feedback, feel free to reach out!

### 📬 Contact

- **Email**: [sekanti02@gmail.com](mailto:sekanti02@gmail.com)
- **LinkedIn**: [Fatih Karahan](https://www.linkedin.com/in/fatih-karahan-717931193/)
