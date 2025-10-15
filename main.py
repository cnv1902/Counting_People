import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

line_points = []
drawing = False

def draw_line_callback(event, x, y, flags, param):
    global line_points, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        line_points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = param.copy()
            cv2.line(frame_copy, line_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Counting Line", frame_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        line_points.append((x, y))
        cv2.line(param, line_points[0], line_points[1], (0, 0, 255), 2)
        cv2.imshow("Select Counting Line", param)

model = YOLO('yolov8n.pt')
video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)

ret, first_frame = cap.read()
if not ret:
    print("Không thể đọc video.")
    exit()

cv2.namedWindow("Select Counting Line")
cv2.setMouseCallback("Select Counting Line", draw_line_callback, first_frame)

while True:
    cv2.putText(first_frame, "Keo chuot de ve duong dem. Nhan 'c' de xac nhan.", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Select Counting Line", first_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and len(line_points) == 2:
        break
    elif key == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Select Counting Line")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

people_count = 0
object_states = defaultdict(int) 

def get_side_of_line(line_p1, line_p2, point):
    return (point[0] - line_p1[0]) * (line_p2[1] - line_p1[1]) - (point[1] - line_p1[1]) * (line_p2[0] - line_p1[0])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 3)

    results = model.track(frame, persist=True, verbose=False, classes=0)

    if results and results[0].boxes.id is not None:
        try:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                current_center = (float(x), float(y))

                prev_side = object_states[track_id]

                current_side_val = get_side_of_line(line_points[0], line_points[1], current_center)
                current_side = np.sign(current_side_val)

                if prev_side != 0 and current_side != 0 and prev_side != current_side:
                    people_count += 1

                if current_side != 0:
                    object_states[track_id] = current_side

                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        except (AttributeError, IndexError):
            pass

    cv2.putText(frame, f"Count: {people_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("YOLOv8 Tracking and Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()