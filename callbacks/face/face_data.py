from dataclasses import dataclass

import numpy as np


@dataclass
class FaceData:
    """
    Represents one detected face.
    """

    embedding: np.ndarray

    confidence: float

    bbox: tuple

    track_id: int | None

    frame: np.ndarray | None = None