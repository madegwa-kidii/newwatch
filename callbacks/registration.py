import gi
gi.require_version("Gst", "1.0")

from gi.repository import Gst

import hailo
import numpy as np

from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    app_callback_class,
)

from database.lancedb import EmployeeDatabase


class UserData(app_callback_class):

    def __init__(self, employee_name):

        super().__init__()

        self.employee_name = employee_name

        self.db = EmployeeDatabase()

        self.registered = False


def app_callback(pad, info, user_data):

    if user_data.registered:
        return Gst.PadProbeReturn.OK

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

        user_data.db.add_employee(
            user_data.employee_name,
            embedding,
        )

        print()
        print("===================================")
        print("Employee registered successfully!")
        print(f"Name : {user_data.employee_name}")
        print("===================================")
        print()

        user_data.registered = True

        break

    return Gst.PadProbeReturn.OK