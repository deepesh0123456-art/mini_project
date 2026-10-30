# AI-Generated Multimedia Content Detection System

An AI-powered multimedia detection application that analyzes **Text, Images, Audio, and Videos** to identify potentially AI-generated content using pretrained Machine Learning and Deep Learning models.

## 🚀 Project Overview

With the rapid advancement of Generative AI, creating realistic synthetic content has become easier than ever. AI-generated images, voices, videos, and text can sometimes be difficult to distinguish from authentic human-created content.

This project aims to provide a unified platform that analyzes different types of multimedia content and predicts whether the submitted content is likely AI-generated or human-created.

The application uses pretrained deep learning models and provides an interactive interface built using Streamlit.

## ✨ Features

The application supports four types of content detection:

### 1. Image Detection
- Upload images for AI-generated content analysis.
- Uses a pretrained image classification model.
- Displays AI-generated and human classification scores.

### 2. Audio Detection
- Upload audio files for synthetic voice detection.
- Uses a Wav2Vec2-based audio classification model.
- Identifies potentially AI-generated and human voices.
- Displays prediction scores.

### 3. Text Detection
- Enter text for AI-generated content analysis.
- Uses a pretrained Transformer-based text classification model.
- Predicts whether the text resembles AI-generated or human-written content.

### 4. Video Detection
- Upload videos for AI-generated content analysis.
- Uses OpenCV to extract frames from videos.
- Analyzes sampled frames using the image detection model.
- Aggregates frame-level predictions to generate a video-level score.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | Pretrained model integration |
| OpenCV | Video processing and frame extraction |
| Librosa | Audio preprocessing |
| NumPy | Numerical computations |
| Pillow | Image processing |

## 🧠 Pretrained Models

The project integrates the following pretrained models:

| Modality | Model |
|---|---|
| Image | [Organika/sdxl-detector](https://huggingface.co/Organika/sdxl-detector) |
| Audio | [Hemgg/Deepfake-audio-detection](https://huggingface.co/Hemgg/Deepfake-audio-detection) |
| Text | [ahmediqbal/ai-text-detector-model](https://huggingface.co/ahmediqbal/ai-text-detector-model) |
| Video | Image detector applied to sampled video frames |

## 🏗️ Project Architecture

```text
AI-Multimedia-Detector/
│
├── app.py
│
├── detectors/
│   ├── __init__.py
│   ├── image_detector.py
│   ├── audio_detector.py
│   ├── text_detector.py
│   └── video_detector.py
│
├── requirements.txt
├── README.md
└── venv/
```

### System Workflow

1. User selects a detection category.
2. User uploads multimedia content or enters text.
3. The application preprocesses the input.
4. The corresponding pretrained model performs inference.
5. The system processes model predictions.
6. Results are displayed through the Streamlit interface.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git
- Internet connection for downloading pretrained models

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL

cd AI-Multimedia-Detector
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 📊 Detection Methodology

- **Image:** Image classification using a pretrained vision model.
- **Audio:** Audio classification after waveform preprocessing and resampling.
- **Text:** Transformer-based text classification.
- **Video:** Frame extraction using OpenCV followed by image classification and score aggregation.

## ⚠️ Limitations

- Detection results depend on the training data and capabilities of the selected pretrained models.
- AI-generated content detection is not always 100% accurate.
- The text detector may produce false positives and false negatives.
- The audio detector may have limitations across different languages and recording conditions.
- Video detection currently relies on sampled frames rather than dedicated temporal video analysis.
- Model prediction scores should not be treated as definitive proof of content authenticity.

## 🔮 Future Improvements

- Improve detection performance using larger and more diverse datasets.
- Integrate dedicated deepfake video detection models.
- Add temporal analysis for video content.
- Support additional AI image and audio generators.
- Implement multimodal feature fusion.
- Add model evaluation using accuracy, precision, recall and F1-score.
- Deploy the application as a publicly accessible web application.

## 👨‍💻 Author

**Deepesh Pandey**

B.Tech – Computer Science and Engineering

Motilal Nehru National Institute of Technology (MNNIT), Allahabad

## 📌 Disclaimer

This project is developed for educational and research purposes. The predictions generated by the application are model-based estimates and should not be considered conclusive evidence of whether content is AI-generated or authentic.

---

⭐ If you find this project interesting, consider giving the repository a star!
