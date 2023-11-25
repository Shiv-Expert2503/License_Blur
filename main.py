import warnings
warnings.filterwarnings('ignore')
import os
import cv2
from ultralytics import YOLO
from tracker import Tracker
import random

model_path = (os.path.join(os.getcwd(), 'Custom_Training', 'best.pt'))

video_path = os.path.join(os.getcwd(), 'sample.mp4')
model = YOLO(model_path)
tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]


def apply_mosaic(image, factor):
    h, w = image.shape[:2]

    # Resize the image to a smaller size
    small_img = cv2.resize(image, (int(w / factor), int(h / factor)), interpolation=cv2.INTER_NEAREST)

    # Resize the small image back to the original size
    result = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_NEAREST)

    return result


# -------------------------------------This is for video as an input

# video_out_path = os.path.join(os.getcwd(), 'out.mp4')
# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()
# cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
#                           (frame.shape[1], frame.shape[0]))
# while ret:
#
#     results = model(frame)
#
#     for result in results:
#         detections = []
#         for r in result.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = r
#             x1 = int(x1)
#             x2 = int(x2)
#             y1 = int(y1)
#             y2 = int(y2)
#             class_id = int(class_id)
#             detections.append([x1, y1, x2, y2, score])
#
#         tracker.update(frame, detections)
#
#         for track in tracker.tracks:
#             bbox = track.bbox
#             x1, y1, x2, y2 = map(int, bbox)
#             track_id = track.track_id
#
#             # cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track_id % len(colors)], 3)
#
#             # Extract the region of interest (ROI) inside the bounding box
#             roi = frame[y1:y2, x1:x2]
#
#             # Apply mosaic effect to the ROI
#             mosaic_roi = apply_mosaic(roi, 15)  # Adjust the factor as needed
#
#             # Replace the original ROI with the mosaic one
#             frame[y1:y2, x1:x2] = mosaic_roi
#
#     # resized_frame = cv2.resize(frame, (1280, 720))
#     # cv2.imshow('frame', resized_frame)
#     cap_out.write(frame)
#     ret, frame = cap.read()
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
# cap.release()
# cap_out.release()
# cv2.destroyAllWindows()


# -------------------------------This is for live detections

cap = cv2.VideoCapture(0)  # 0 represents the default camera (webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = map(int, bbox)

            # Extract the region of interest (ROI) inside the bounding box
            roi = frame[y1:y2, x1:x2]

            mosaic_roi = apply_mosaic(roi, 15)

            # Replace the original ROI with the mosaic one
            frame[y1:y2, x1:x2] = mosaic_roi
    # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('Number Plate Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
