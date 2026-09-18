import cv2
from src.detector import AerialDetector

WINDOW_W = 1280
WINDOW_H = 720

def resize_keep_ratio(frame, max_w, max_h):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))

def main():

    detector = AerialDetector()

    cap = cv2.VideoCapture("input/demo.mp4")

    if not cap.isOpened():
        print("Cannot open video")
        return

    cv2.namedWindow("ORBITAL ARES", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ORBITAL ARES", WINDOW_W, WINDOW_H)

    frame_no = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_no += 1

        detections = detector.detect(frame)

        for obj in detections:

            x1, y1, x2, y2 = obj["bbox"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            cv2.putText(
                frame,
                f'{obj["label"]} {obj["confidence"]}',
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        cv2.putText(
            frame,
            f"Frame: {frame_no}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        display = resize_keep_ratio(frame, WINDOW_W, WINDOW_H)

        cv2.imshow("ORBITAL ARES", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()