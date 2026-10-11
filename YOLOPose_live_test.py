from ultralytics import YOLO
import cv2

model = YOLO("../Models/yolo26n-pose.pt")

source = cv2.VideoCapture(1)
cv2.namedWindow("YOLOPose Live Test", cv2.WINDOW_NORMAL)

while True:
    read_success, frame = source.read()
    if not read_success:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOPose Live Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

source.release()
cv2.destroyAllWindows()
