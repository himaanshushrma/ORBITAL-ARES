import cv2
import torch
from ultralytics import YOLO
from src.tracker import VehicleTracker

# ==========================================================
# ORBITAL ARES V2
# GPU + YOLO11 + ByteTrack + Full Resolution Video
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# Load model
model = YOLO("models/yolo11s.pt")
model.to(DEVICE)

tracker = VehicleTracker()

# COCO vehicle classes
VEHICLE_CLASSES = [2, 3, 5, 7]

CLASS_NAMES = {
    2: "CAR",
    3: "BIKE",
    5: "BUS",
    7: "TRUCK"
}

VIDEO_PATH = "input/demo.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception("Cannot open input/demo.mp4")

# -------- Original video properties --------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps}")

# -------- Output writer --------
writer = cv2.VideoWriter(
    "output/orbital_v2.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# -------- Full size window --------
cv2.namedWindow("ORBITAL ARES", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ORBITAL ARES", width, height)

frame_count = 0

# ==========================================================
# MAIN LOOP
# ==========================================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Keep original resolution
    frame = cv2.resize(frame, (width, height))

    # ---------------- YOLO on GPU ----------------
    result = model.predict(
        source=frame,
        device=DEVICE,
        conf=0.45,
        verbose=False
    )[0]

    # ---------------- Tracking ----------------
    detections = tracker.update(result)

    if len(detections) > 0:

        for box, cls, track_id in zip(
                detections.xyxy,
                detections.class_id,
                detections.tracker_id
        ):

            if cls not in VEHICLE_CLASSES:
                continue

            if track_id is None:
                continue

            x1, y1, x2, y2 = map(int, box)

            label = f"{CLASS_NAMES[int(cls)]} #{int(track_id)}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    # ---------------- HUD ----------------
    cv2.rectangle(frame, (10, 10), (270, 90), (25, 25, 25), -1)

    cv2.putText(
        frame,
        "ORBITAL ARES",
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"GPU : {DEVICE.upper()}",
        (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        f"FRAME : {frame_count}",
        (20, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # -------- Save + Display --------
    writer.write(frame)
    cv2.imshow("ORBITAL ARES", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

# ==========================================================
# CLEANUP
# ==========================================================
cap.release()
writer.release()
cv2.destroyAllWindows()

print("\nMission Complete")
print("Saved -> output/orbital_v2.mp4")