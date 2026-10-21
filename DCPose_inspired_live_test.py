from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "Models/yolo26n-pose.pt"
CAMERA_INDEX = 1
SMOOTHING_WINDOW = 5
CONFIDENCE_THRESHOLD = 0.3

# COCO keypoint connections used by the YOLO pose model.
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def smooth_keypoints(history, current):
    """Average recent keypoints for the same detected person index."""
    history.append(current.copy())
    return np.mean(np.stack(history), axis=0)


def draw_pose(frame, keypoints, confidences, point_color, line_color):
    """Draw confident keypoints and COCO skeleton connections."""
    for start, end in SKELETON:
        if (
            confidences[start] >= CONFIDENCE_THRESHOLD
            and confidences[end] >= CONFIDENCE_THRESHOLD
        ):
            start_point = tuple(np.int32(keypoints[start]))
            end_point = tuple(np.int32(keypoints[end]))
            cv2.line(frame, start_point, end_point, line_color, 5, cv2.LINE_AA)

    for (x, y), confidence in zip(keypoints, confidences):
        if confidence >= CONFIDENCE_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), 4, point_color, -1, cv2.LINE_AA)


def center_portrait_crop(frame, aspect_ratio=8 / 9):
    """Crop the center into a portrait-oriented panel that fills half a 16:9 screen."""
    height, width = frame.shape[:2]
    target_width = int(height * aspect_ratio)
    if target_width >= width:
        return frame

    left = (width - target_width) // 2
    return frame[:, left:left + target_width]


model = YOLO(MODEL_PATH)
source = cv2.VideoCapture(CAMERA_INDEX)

if not source.isOpened():
    raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

window_name = "Temporal Comparison"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN,
)

keypoint_history = {}
frame_history = deque(maxlen=SMOOTHING_WINDOW)

while True:
    read_success, frame = source.read()
    if not read_success:
        break

    processed_frame = center_portrait_crop(frame)
    processed_frame = cv2.flip(processed_frame, 1)
    results = model(processed_frame, verbose=False)[0]

    raw_frame = processed_frame.copy()
    smoothed_frame = processed_frame.copy()

    if results.keypoints is not None and results.keypoints.xy is not None:
        keypoints = results.keypoints.xy.cpu().numpy()

        if results.keypoints.conf is not None:
            confidences = results.keypoints.conf.cpu().numpy()
        else:
            confidences = np.ones(keypoints.shape[:2])

        for person_index, person_keypoints in enumerate(keypoints):
            person_confidence = confidences[person_index]

            valid = person_confidence >= CONFIDENCE_THRESHOLD
            if not np.any(valid):
                continue

            if person_index not in keypoint_history:
                keypoint_history[person_index] = deque(
                    maxlen=SMOOTHING_WINDOW
                )

            smoothed = smooth_keypoints(
                keypoint_history[person_index],
                person_keypoints,
            )

            draw_pose(
                raw_frame,
                person_keypoints,
                person_confidence,
                point_color=(255, 80, 80),
                line_color=(255, 160, 160),
            )
            draw_pose(
                smoothed_frame,
                smoothed,
                person_confidence,
                point_color=(80, 80, 255),
                line_color=(160, 160, 255),
            )

    cv2.putText(
        raw_frame,
        "CONTROL: frame-by-frame YOLO-Pose",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 80, 80),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        smoothed_frame,
        "TEMPORAL: YOLO-Pose + keypoint smoothing",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 255),
        2,
        cv2.LINE_AA,
    )

    comparison = np.hstack((raw_frame, smoothed_frame))
    cv2.imshow(window_name, comparison)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

source.release()
cv2.destroyAllWindows()