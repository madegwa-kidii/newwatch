import numpy as np
import hailo

import gi
gi.require_version("Gst", "1.0")

from gi.repository import Gst


class EmbeddingCallback:

    def __init__(self, on_embedding):

        self.on_embedding = on_embedding

    def __call__(self, pad, info, user_data):

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

            self.on_embedding(
                embedding,
                detection,
            )

        return Gst.PadProbeReturn.OK