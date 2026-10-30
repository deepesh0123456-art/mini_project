
import streamlit as st
from PIL import Image
from detectors.image_detector import detect_image
from detectors.audio_detector import detect_audio
from detectors.text_detector import detect_text
from detectors.video_detector import detect_video

st.set_page_config(
    page_title="AI Multimedia Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Generated Multimedia Detector")

st.markdown(
    "Analyze images, audio, video and text "
    "to identify potentially AI-generated content."
)

option = st.sidebar.selectbox(
    "Select Detection Type",
    [
        "Overview",
        "Image",
        "Audio",
        "Video",
        "Text"
    ]
)

st.subheader(f"{option} Detection")

if option == "Overview":

    st.title("AI Multimedia Detection System")

    st.markdown(
        "An intelligent system for analyzing "
        "potentially AI-generated multimedia content."
    )

    st.divider()

    st.subheader("Detection Capabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Detection Modules", "4")
        st.metric("Image Detection", "Available")
        st.metric("Audio Detection", "Available")

    with col2:
        st.metric("Video Detection", "Available")
        st.metric("Text Detection", "Available")
        st.metric("ML Framework", "PyTorch")

    st.divider()

    st.subheader("How the System Works")

    st.markdown("""
    1. User uploads multimedia content or enters text.
    2. The system identifies the content modality.
    3. The corresponding pretrained ML model processes it.
    4. The model generates classification scores.
    5. The application displays the prediction and scores.

    **Video processing:** Selected frames are extracted
    using OpenCV and analyzed using the image classifier.
    """)

    st.info(
        "Select a detection module from the sidebar "
        "to start analyzing content."
    )
elif option == "Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=400
        )

        if st.button("Analyze Image", type="primary"):

            with st.spinner(
                "Loading model and analyzing image..."
            ):
                try:
                    result = detect_image(image)

                    st.divider()
                    st.subheader("Detection Results")

                    st.success(result["prediction"])

                    ai = result["ai_probability"]
                    human = result["human_probability"]

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "AI-generated score",
                        f"{ai * 100:.2f}%"
                    )

                    col2.metric(
                        "Human-created score",
                        f"{human * 100:.2f}%"
                    )

                    st.write("AI-generated score")
                    st.progress(ai)

                    st.write("Human-created score")
                    st.progress(human)

                    st.caption(
                        "These are model classification scores, "
                        "not calibrated probabilities or proof "
                        "of an image's origin."
                    )

                except Exception as e:
                    st.error(f"Detection failed: {e}")

elif option == "Audio":

    uploaded_file = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "flac", "ogg"]
    )

    if uploaded_file is not None:

        st.audio(uploaded_file)

        if st.button("Analyze Audio", type="primary"):

            with st.spinner("Analyzing audio..."):

                try:
                    result = detect_audio(
                        uploaded_file.getvalue()
                    )

                    st.divider()
                    st.subheader("Audio Detection Results")

                    ai = result["ai_probability"]
                    human = result["human_probability"]

                    if ai is not None and human is not None:

                        prediction = (
                            "Likely AI-generated audio"
                            if ai > human
                            else "Likely authentic human audio"
                        )

                        st.success(prediction)

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "AI-generated score",
                            f"{ai * 100:.2f}%"
                        )

                        col2.metric(
                            "Human audio score",
                            f"{human * 100:.2f}%"
                        )

                    else:
                        st.warning(
                            "Model labels could not be mapped "
                            "to AI/real automatically."
                        )

                    st.write("Raw model predictions:")

                    st.json(result["results"])

                    st.caption(
                        "Predictions are model scores, not proof "
                        "of audio authenticity. The model was "
                        "trained on English speech data."
                    )

                except Exception as e:
                    st.error(f"Audio detection failed: {e}")


elif option == "Video":

    uploaded_file = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_file is not None:

        st.video(uploaded_file)

        if st.button("Analyze Video", type="primary"):

            with st.spinner(
                "Extracting and analyzing video frames..."
            ):

                try:
                    result = detect_video(
                        uploaded_file.getvalue()
                    )

                    st.divider()
                    st.subheader("Video Detection Results")

                    ai = result["ai_probability"]
                    human = result["human_probability"]

                    if ai >= human:
                        st.success(
                            "Likely AI-generated video content"
                        )
                    else:
                        st.info(
                            "Likely authentic video content"
                        )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "AI score",
                        f"{ai * 100:.2f}%"
                    )

                    col2.metric(
                        "Human score",
                        f"{human * 100:.2f}%"
                    )

                    col3.metric(
                        "Frames analyzed",
                        result["frames_analyzed"]
                    )

                    st.write("Frame-level AI scores")

                    for index, frame in enumerate(
                        result["frame_results"]
                    ):
                        st.write(
                            f"Frame {index + 1} "
                            f"({frame['timestamp']} sec)"
                        )

                        st.progress(
                            frame["ai_probability"]
                        )

                        st.caption(
                            f"AI score: "
                            f"{frame['ai_probability'] * 100:.2f}%"
                        )

                    st.caption(
                        "Video prediction is based on sampled "
                        "frames. Audio and temporal consistency "
                        "are not analyzed in this baseline."
                    )

                except Exception as e:
                    st.error(f"Video detection failed: {e}")
else:

    text_input = st.text_area(
        "Enter text to analyze",
        height=200,
        placeholder="Paste your text here..."
    )

    if st.button("Analyze Text", type="primary"):

        if not text_input.strip():
            st.warning("Please enter some text.")

        else:
            with st.spinner("Analyzing text..."):

                try:
                    results = detect_text(text_input)

                    st.subheader("Text Detection Results")

                    for item in results:
                        st.write(
                            f"{item['label']}: "
                            f"{item['score'] * 100:.2f}%"
                        )

                    st.json(results)

                    st.caption(
                        "These are model classification scores, "
                        "not proof of authorship."
                    )

                except Exception as e:
                    st.error(f"Text detection failed: {e}")

