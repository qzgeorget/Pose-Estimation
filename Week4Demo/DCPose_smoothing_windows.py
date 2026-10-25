import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "Models/yolo26n-pose.pt"
VIDEO_PATH = "Input_Videos/Dribbling_drill_short.mp4"
OUTPUT_PATH = "Output_Videos/Dribbling_drill_short_smoothed.mp4"
SMOOTHING_WINDOWS = (1, 3, 5, 7)
CONFIDENCE_THRESHOLD = 0.3

if len(SMOOTHING_WINDOWS) != 4 or len(set(SMOOTHING_WINDOWS)) != 4:
    raise ValueError("SMOOTHING_WINDOWS must contain four unique values")
if any(window < 1 for window in SMOOTHING_WINDOWS):
    raise ValueError("Smoothing windows must be positive integers")

SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def smooth_keypoints(history, current):
    return np.mean(np.stack(history), axis=0)


def draw_pose(frame, keypoints, confidences, point_color, line_color):
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


def add_label(frame, window):
    label = f"SMOOTHING WINDOW: {window}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


model = YOLO(MODEL_PATH)
source = cv2.VideoCapture(VIDEO_PATH)

if not source.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

source_fps = source.get(cv2.CAP_PROP_FPS)
if not source_fps or source_fps <= 0:
    source_fps = 30.0

frames = []
predictions = []

while True:
    read_success, frame = source.read()
    if not read_success:
        break

    processed_frame = frame
    result = model(processed_frame, verbose=False)[0]

    if result.keypoints is not None and result.keypoints.xy is not None:
        keypoints = result.keypoints.xy.cpu().numpy()
        if result.keypoints.conf is not None:
            confidences = result.keypoints.conf.cpu().numpy()
        else:
            confidences = np.ones(keypoints.shape[:2])
    else:
        keypoints = np.empty((0, 17, 2), dtype=np.float32)
        confidences = np.empty((0, 17), dtype=np.float32)

    frames.append(processed_frame)
    predictions.append((keypoints, confidences))

source.release()

if not frames:
    raise RuntimeError(f"Video contains no readable frames: {VIDEO_PATH}")

panel_height, panel_width = frames[0].shape[:2]
output_width = panel_width - (panel_width % 2)
output_height = panel_height - (panel_height % 2)
panel_output_size = (output_width // 2, output_height // 2)
writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    source_fps,
    (output_width, output_height),
)
if not writer.isOpened():
    raise RuntimeError(f"Could not create output video: {OUTPUT_PATH}")

for frame_index, processed_frame in enumerate(frames):
    panels = {
        window: processed_frame.copy()
        for window in SMOOTHING_WINDOWS
    }

    current_keypoints, current_confidences = predictions[frame_index]
    person_count = current_keypoints.shape[0]
    for person_index in range(person_count):
        person_confidence = current_confidences[person_index]
        if not np.any(person_confidence >= CONFIDENCE_THRESHOLD):
            continue

        for window in SMOOTHING_WINDOWS:
            # Center each window around the current frame. This uses past and
            # future frames, like the official DCPose demo's temporal inputs.
            start = max(0, frame_index - (window - 1) // 2)
            end = min(len(predictions), start + window)
            start = max(0, end - window)

            temporal_keypoints = []
            for prediction_index in range(start, end):
                keypoints, confidences = predictions[prediction_index]
                if person_index < keypoints.shape[0]:
                    temporal_keypoints.append(keypoints[person_index])

            if not temporal_keypoints:
                continue

            smoothed = smooth_keypoints(temporal_keypoints, current_keypoints[person_index])
            draw_pose(
                panels[window],
                smoothed,
                person_confidence,
                point_color=(80, 80, 255),
                line_color=(160, 160, 255),
            )

    for window in SMOOTHING_WINDOWS:
        add_label(panels[window], window)

    resized_panels = {
        window: cv2.resize(
            panels[window],
            panel_output_size,
            interpolation=cv2.INTER_AREA,
        )
        for window in SMOOTHING_WINDOWS
    }
    top_row = np.hstack((
        resized_panels[SMOOTHING_WINDOWS[0]],
        resized_panels[SMOOTHING_WINDOWS[1]],
    ))
    bottom_row = np.hstack((
        resized_panels[SMOOTHING_WINDOWS[2]],
        resized_panels[SMOOTHING_WINDOWS[3]],
    ))
    comparison = np.vstack((top_row, bottom_row))
    writer.write(comparison)

writer.release()
print(f"Saved comparison video to {OUTPUT_PATH}")
