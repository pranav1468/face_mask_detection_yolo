# 😷 Face Mask Detection System (YOLOv8 + Streamlit)

An end-to-end **Face Mask Detection System** built using **YOLOv8**, deployed with **Streamlit**, and integrated with **GitHub Actions CI** for automated validation.

This project demonstrates **computer vision**, **deep learning deployment**, and **CI pipeline practices** in a real-world–style setup.

---

## 🚀 Features

* ✅ Face detection with mask classification:

  * `with_mask`
  * `without_mask`
  * `mask_weared_incorrect`
* 🖼 Image-based inference
* 🎥 Video-based inference
* 📁 Batch image processing
* 📊 Visual detection summary with confidence scores
* 🧪 GitHub Actions CI pipeline
* 🧠 ONNX-optimized inference for deployment

---

## 🧠 Model Overview

* **Architecture:** YOLOv8
* **Framework:** Ultralytics
* **Formats Supported:**

  * `.pt` (training & experimentation)
  * `.onnx` (deployment & inference)
* **Task:** Object Detection + Mask Classification

> ⚠️ **Note:**
> The model performs best on **close-up faces** similar to the training distribution.
> In dense crowd scenes, predictions may suffer due to **distribution shift** and dataset bias.

---

## 📂 Project Structure

```text
face_mask_detection_yolo/
│
├── app/
│   └── streamlit_app.py          # Streamlit-based web application
│
├── inference/
│   ├── predict_image.py          # Image inference logic
│   ├── predict_video.py          # Video inference logic
│   ├── visualize.py              # Bounding box & label visualization
│   └── webcam_local.py           # Local webcam inference (OpenCV)
│
├── model/
│   ├── train.py                  # YOLOv8 training script
│   ├── evaluate.py               # Model evaluation utilities
│   └── export.py                 # Export model to ONNX
│
├── utils/
│   ├── data_loader.py            # Dataset loading utilities
│   ├── preprocess.py             # Image preprocessing helpers
│   ├── voc_to_yolo.py            # VOC → YOLO annotation conversion
│   └── metrics.py                # Custom evaluation metrics
│
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI pipeline
│
├── .devcontainer/                # Development container configuration
│
├── best_face_mask.onnx           # ONNX model for deployment
├── best_face_mask.pt             # Trained YOLOv8 model
├── last_mask_detection.pt        # Training checkpoint
│
├── export.py                     # Standalone model export script
├── face-mask-detection.ipynb     # Training & experimentation notebook
├── main.py                       # Entry point / experimentation
├── test.py                       # Sanity & quick tests
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## 🖥 Streamlit Application

### Modes Available

* **📤 Upload Image** – Single image inference
* **📁 Batch Processing** – Multiple images at once
* **🎥 Upload Video** – Frame-by-frame detection

### UI Highlights

* Side-by-side original vs detected output
* Confidence-based alerts
* Per-class counters
* Inference time display
* Clean, light UI for readability

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/pranav1468/face_mask_detection_yolo.git
cd face_mask_detection_yolo
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Then open the browser at:

```
http://localhost:8501
```

---

## 🔄 CI Pipeline (GitHub Actions)

This project includes a **Continuous Integration (CI)** pipeline using GitHub Actions.

### What the pipeline does:

* ✅ Installs dependencies
* ✅ Verifies required files (model + app)
* ✅ Checks Python imports
* ✅ Performs syntax validation
* ✅ Runs automatically on:

  * `push` to `main`
  * `pull_request` to `main`

📁 Workflow file:

```
.github/workflows/ci.yml
```

---

## 📊 Limitations & Future Improvements

### Current Limitations

* Reduced accuracy in dense crowd scenes
* Model may rely on visual shortcuts due to dataset bias
* Single-stage detection (face + mask together)

### Future Improvements

* Two-stage pipeline (Face detector + Mask classifier)
* More diverse training dataset
* Improved hard-negative mining
* Real-time webcam inference (WebRTC)
* Dockerized deployment

---

## 🧪 Technologies Used

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* ONNX Runtime
* Streamlit
* Plotly
* GitHub Actions

---

## 👤 Author

**Pranav Baghare**
GitHub: [@pranav1468](https://github.com/pranav1468)
