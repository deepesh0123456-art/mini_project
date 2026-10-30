
import cv2
import tempfile
import os
import numpy as np
from PIL import Image

from detectors.image_detector import detect_image


def detect_video(video_bytes):

    # Save uploaded video temporarily
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )

    try:
        temp_file.write(video_bytes)
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)

        if not cap.isOpened():
            raise ValueError("Unable to open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or total_frames <= 0:
            raise ValueError("Invalid or unsupported video.")

        duration = total_frames / fps

        # Analyze up to 8 frames across the video
        number_of_frames = min(8, max(1, int(duration) + 1))

        timestamps = np.linspace(
            0,
            max(0, duration - 0.1),
            number_of_frames
        )

        frame_results = []

        for timestamp in timestamps:

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                float(timestamp * 1000)
            )

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(frame)

            result = detect_image(image)

            ai_score = result["ai_probability"]

            if ai_score is not None:
                frame_results.append({
                    "timestamp": round(float(timestamp), 2),
                    "ai_probability": float(ai_score)
                })

        cap.release()

        if not frame_results:
            raise ValueError("No frames could be analyzed.")

        average_ai_score = float(
            np.mean([
                item["ai_probability"]
                for item in frame_results
            ])
        )

        return {
            "ai_probability": average_ai_score,
            "human_probability": 1 - average_ai_score,
            "frames_analyzed": len(frame_results),
            "frame_results": frame_results,
            "duration": round(duration, 2)
        }

    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
