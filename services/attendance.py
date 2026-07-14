import time


class AttendanceService:
    """
    Handles attendance events from the face recognition callback.
    Prevents duplicate detections within a cooldown period.
    """

    COOLDOWN_SECONDS = 10

    def __init__(self):
        # employee -> last detection timestamp
        self.last_seen = {}

    def process_detection(self, label: str, confidence: float):

        # Ignore unknown people
        if label == "Unknown":
            return

        now = time.time()

        last = self.last_seen.get(label)

        # Ignore repeated detections
        if last and (now - last) < self.COOLDOWN_SECONDS:
            return

        self.last_seen[label] = now

        print(
            f"[Attendance] "
            f"{label} "
            f"({confidence:.2f}) "
            f"checked in."
        )