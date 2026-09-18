import supervision as sv


class VehicleTracker:

    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.35,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

    def update(self, result):
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)
        return detections