import argparse

from callbacks.registration import (
    UserData,
    app_callback,
)

from hailo_apps.hailo_app_python.apps.face_recognition.face_recognition_pipeline import (
    GStreamerFaceRecognitionApp,
)


def main():

    parser = argparse.ArgumentParser(
        description="Register a new employee into LanceDB"
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Employee name",
    )

    parser.add_argument(
        "--input",
        default="rpi",
        help="Input source (default: rpi)",
    )

    args = parser.parse_args()

    user_data = UserData(args.name)

    app = GStreamerFaceRecognitionApp(
        app_callback,
        user_data,
    )

    print()
    print("===================================")
    print(f"Registering: {args.name}")
    print("Look at the camera...")
    print("===================================")
    print()

    try:
        app.run()

    except KeyboardInterrupt:
        print("\nRegistration cancelled.")


if __name__ == "__main__":
    main()