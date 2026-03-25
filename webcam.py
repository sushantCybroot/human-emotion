"""Real-time webcam emotion detection using OpenCV."""

import cv2
from tensorflow.keras.models import load_model

from src.config import MODEL_PATH
from src.utils import predict_label, preprocess_image_array


def main():
    model = load_model(MODEL_PATH)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        raise RuntimeError("Unable to access the webcam.")

    while True:
        success, frame = webcam.read()
        if not success:
            break

        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            grayscale_frame,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
        )

        for (x, y, w, h) in faces:
            face_region = grayscale_frame[y : y + h, x : x + w]
            processed_face = preprocess_image_array(face_region)
            label, confidence = predict_label(model, processed_face)
            display_text = f"{label} ({confidence:.2f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                display_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

