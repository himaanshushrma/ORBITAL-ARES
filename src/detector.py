from ultralytics import YOLO


class AerialDetector:
    def __init__(self, model_path="yolo11n.pt"):

        # Load YOLO model
        self.model = YOLO(model_path)

        # COCO class mapping
        self.classes = {
            0: "Person",
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=0.35,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                cls = int(box.cls[0])

                if cls not in self.classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "label": self.classes[cls],
                    "class_id": cls,
                    "confidence": round(confidence, 2)
                })

        return detections