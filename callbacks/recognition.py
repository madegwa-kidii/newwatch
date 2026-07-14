import gi
gi.require_version("Gst", "1.0")

from gi.repository import Gst

import hailo
import numpy as np

from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    app_callback_class,
)

from database.lancedb import EmployeeDatabase
from services.attendance import AttendanceService


class UserData(app_callback_class):

    def __init__(self):
        super().__init__()

        self.database = EmployeeDatabase()
        self.attendance = AttendanceService()

        # Disable Hailo's Telegram integration
        self.telegram_enabled = False

    def send_notification(self, *args, **kwargs):
        pass


def app_callback(pad, info, user_data):

    buffer = info.get_buffer()

    if buffer is None:
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)

    detections = roi.get_objects_typed(
        hailo.HAILO_DETECTION
    )

    for detection in detections:

        if detection.get_label() != "face":
            continue

        print(
            "Detected:",
            detection.get_label(),
            detection.get_confidence(),
        )

        embeddings = detection.get_objects_typed(
            hailo.HAILO_MATRIX
        )

        if len(embeddings) != 1:
            print("No embedding available.")
            continue

        embedding = np.array(
            embeddings[0].get_data(),
            dtype=np.float32,
        )

        print("\n===== LIVE EMBEDDING =====")
        print("Query embedding length:", len(embedding))
        print("Norm:", np.linalg.norm(embedding))
        print("First 10:", embedding[:10])

        records = (
            user_data.database.table
            .search()
            .limit(1)
            .to_list()
        )

        if records:
            stored = records[0]["embedding"]

            print("\n===== DATABASE EMBEDDING =====")
            print("Stored embedding length:", len(stored))
            print("Stored first 10 values:", stored[:10])

        results = (
            user_data.database.table
            .search(
                embedding.tolist(),
                vector_column_name="embedding",
            )
            .metric("cosine")
            .limit(5)
            .to_list()
        )

        print("\nNearest neighbours")

        for r in results:
            print(
                r["name"],
                r["_distance"],
            )

        print()
        
        person = user_data.database.search(embedding)

        if person is None:

            print("Recognized: Unknown")

            user_data.attendance.process_detection(
                label="Unknown",
                confidence=0.0,
            )

            continue

        print(
            f"Recognized: {person['name']} "
            f"(distance={person['_distance']:.4f})"
        )

        confidence = max(
            0.0,
            1.0 - float(person["_distance"]),
        )

        user_data.attendance.process_detection(
            label=person["name"],
            confidence=confidence,
        )

    return Gst.PadProbeReturn.OK