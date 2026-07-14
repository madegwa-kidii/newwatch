import argparse
import cv2
import os
import time

from picamera2 import Picamera2


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def create_employee_folder(employee_name: str) -> str:
    """
    Creates data/employees/<employee_name>/ if it does not exist.
    """

    folder = os.path.join("data/employees", employee_name)

    os.makedirs(folder, exist_ok=True)

    return folder


def next_image_number(folder: str) -> int:
    """
    Returns the next image number.

    Example:
        1.jpg
        2.jpg
        3.jpg

    returns 4
    """

    numbers = []

    for filename in os.listdir(folder):

        if filename.endswith(".jpg"):

            try:
                numbers.append(int(filename.split(".")[0]))
            except ValueError:
                pass

    if not numbers:
        return 1

    return max(numbers) + 1


def open_camera():

    camera = Picamera2()

    camera.configure(
        camera.create_preview_configuration(
            main={
                "format": "XRGB8888",
                "size": (IMAGE_WIDTH, IMAGE_HEIGHT),
            }
        )
    )

    camera.start()

    time.sleep(2)

    return camera


def draw_ui(frame, employee_name, count):

    h, w = frame.shape[:2]

    box_size = 260

    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2

    x2 = x1 + box_size
    y2 = y1 + box_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Employee : {employee_name}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Captured : {count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "SPACE = Capture    Q = Quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def capture_images(employee_name: str):

    folder = create_employee_folder(employee_name)

    image_number = next_image_number(folder)

    camera = open_camera()

    print(f"\nCapturing images for {employee_name}")
    print("Press SPACE to capture")
    print("Press Q to finish\n")

    while True:

        frame = camera.capture_array()

        frame = cv2.flip(frame, 1)

        draw_ui(frame, employee_name, image_number - 1)

        cv2.imshow("NewWatch Registration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):

            filename = os.path.join(folder, f"{image_number}.jpg")

            cv2.imwrite(filename, frame)

            print(f"Saved {filename}")

            image_number += 1

        elif key == ord("q"):
            break

    camera.stop()

    cv2.destroyAllWindows()

    print(
        f"\nFinished capturing {image_number-1} images for {employee_name}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        required=True,
        help="Employee name",
    )

    args = parser.parse_args()

    capture_images(args.name)


if __name__ == "__main__":
    main()