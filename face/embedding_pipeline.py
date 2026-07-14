from pathlib import Path
import time
import threading

import gi
gi.require_version("Gst", "1.0")

from gi.repository import Gst

import hailo
import numpy as np

from hailo_apps.hailo_app_python.apps.face_recognition.face_recognition_pipeline import (
    GStreamerFaceRecognitionApp,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import (
    app_callback_class,
)


class EmbeddingUserData(app_callback_class):
    """
    Stores the embedding extracted from one image.
    """

    def __init__(self):
        super().__init__()

        self.embedding = None
        self.finished = threading.Event()


class EmbeddingPipeline(GStreamerFaceRecognitionApp):
    """
    Reuses Hailo's entire training pipeline but intercepts the
    generated embedding instead of writing to Hailo's LanceDB.
    """

    def __init__(self):

        self.user_data = EmbeddingUserData()

        super().__init__(
            app_callback=lambda *args: Gst.PadProbeReturn.OK,
            user_data=self.user_data,
        )

        print("\nTRAIN PIPELINE")
        print("Recognition HEF:", self.hef_path_recognition)
        print("Detection HEF :", self.hef_path_detection)

    def train_vector_db_callback(self, pad, info, user_data):

        buffer = info.get_buffer()

        if buffer is None:
            return Gst.PadProbeReturn.OK

        roi = hailo.get_roi_from_buffer(buffer)

        for detection in (
            d
            for d in roi.get_objects_typed(hailo.HAILO_DETECTION)
            if d.get_label() == "face"
        ):

            embeddings = detection.get_objects_typed(
                hailo.HAILO_MATRIX
            )

            if len(embeddings) != 1:
                continue

            embedding = np.array(
                embeddings[0].get_data(),
                dtype=np.float32,
            )

            print("\nTRAIN EMBEDDING")
            print("Norm:", np.linalg.norm(embedding))
            print("First 10:", embedding[:10])

            user_data.embedding = embedding
            user_data.finished.set()

            return Gst.PadProbeReturn.OK

            break

        return Gst.PadProbeReturn.OK

    def get_embedding(self, image_path):

        self.options_menu.mode = "train"

        self.user_data.embedding = None
        self.user_data.finished.clear()

        self.current_file = str(Path(image_path))

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        self.create_pipeline()

        self.connect_train_vector_db_callback()

        self.pipeline.set_state(Gst.State.PLAYING)

        found = self.user_data.finished.wait(timeout=5)

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        if not found:
            return None

        return self.user_data.embedding