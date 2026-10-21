"""Live YOLO26 person detection followed by ViTPose-S pose estimation."""

from pathlib import Path

import cv2
import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model
from ultralytics import YOLO


CAMERA_INDEX = 1
PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_MODEL = PROJECT_ROOT / "Models/yolo26n.pt"
PERSON_CLASS_ID = 0
DETECTION_CONFIDENCE = 0.4
KEYPOINT_CONFIDENCE = 0.3

COCO_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)


def draw_poses(
    frame: np.ndarray,
    boxes: np.ndarray,
    pose_results: list,
) -> np.ndarray:
    """Draw YOLO boxes and ViTPose keypoints on the webcam frame."""
    annotated = frame.copy()

    for box, pose_result in zip(boxes, pose_results):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        instances = pose_result.pred_instances
        keypoints = np.asarray(instances.keypoints[0])
        scores = np.asarray(instances.keypoint_scores[0])

        for start, end in COCO_SKELETON:
            if (
                scores[start] >= KEYPOINT_CONFIDENCE
                and scores[end] >= KEYPOINT_CONFIDENCE
            ):
                cv2.line(
                    annotated,
                    tuple(keypoints[start].astype(int)),
                    tuple(keypoints[end].astype(int)),
                    (255, 128, 0),
                    2,
                )

        for point, score in zip(keypoints, scores):
            if score >= KEYPOINT_CONFIDENCE:
                cv2.circle(
                    annotated,
                    tuple(point.astype(int)),
                    3,
                    (0, 0, 255),
                    -1,
                )

    return annotated


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    yolo = YOLO(YOLO_MODEL)

    mmpose_root = Path(__import__("mmpose").__file__).resolve().parent
    config = (
        mmpose_root
        / ".mim/configs/body_2d_keypoint/topdown_heatmap/coco/"
        "td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192.py"
    )
    checkpoint = (
        "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/"
        "topdown_heatmap/coco/"
        "td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192-"
        "4c101a76_20230314.pth"
    )
    pose_model = init_model(str(config), checkpoint, device=device)

    source = cv2.VideoCapture(CAMERA_INDEX)
    if not source.isOpened():
        raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}.")

    window_name = "YOLO26 + ViTPose-S Live Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            read_success, frame = source.read()
            if not read_success:
                raise RuntimeError("Could not read a frame from the webcam.")

            detection = yolo(
                frame,
                classes=[PERSON_CLASS_ID],
                conf=DETECTION_CONFIDENCE,
                verbose=False,
            )[0]
            boxes = detection.boxes.xyxy.cpu().numpy()

            if len(boxes) > 0:
                pose_results = inference_topdown(pose_model, frame, boxes)
                frame = draw_poses(frame, boxes, pose_results)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
