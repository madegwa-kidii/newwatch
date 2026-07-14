from pathlib import Path

import gi
gi.require_version("Gst", "1.0")

from gi.repository import Gst

import hailo
import numpy as np

from hailo_apps.hailo_app_python.apps.face_recognition.face_recognition_pipeline import (
    GStreamerFaceRecognitionApp,
)

from database.lancedb import EmployeeDatabase


class EmbeddingPipeline(GStreamerFaceRecognitionApp):

    """
    Uses Hailo's complete training pipeline but stores
    embeddings inside our own LanceDB instead of Hailo's DB.
    """

    def __init__(self, app_callback, user_data):

        super().__init__(app_callback, user_data)

        self.database = EmployeeDatabase()

    def train_vector_db_callback(self, pad, info, user_data):

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

            embeddings = detection.get_objects_typed(
                hailo.HAILO_MATRIX
            )

            if len(embeddings) != 1:
                continue

            embedding = np.array(
                embeddings[0].get_data(),
                dtype=np.float32,
            )

            #
            # Current file:
            #
            # data/employees/Ossy/1.jpg
            #
            # Parent folder = employee name
            #

            employee_name = Path(
                self.current_file
            ).parent.name

            #
            # Avoid duplicates.
            # Hailo loops each image ~30 frames.
            #

            if self.current_file in self.processed_files:
                continue

            self.processed_files.add(
                self.current_file
            )

            self.database.add_employee(
                employee_name,
                embedding,
            )

            print(
                f"✓ Added embedding "
                f"for {employee_name}"
            )

        return Gst.PadProbeReturn.OK