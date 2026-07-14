from callbacks.recognition import app_callback, UserData

from hailo_apps.hailo_app_python.apps.face_recognition.face_recognition_pipeline import (
    GStreamerFaceRecognitionApp,
)


def main():
    user_data = UserData()

    app = GStreamerFaceRecognitionApp(
        app_callback,
        user_data,
    )

    app.run()


if __name__ == "__main__":
    main()