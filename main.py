import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- CẤU HÌNH ---
FRAME_SKIP = 3  # Bỏ qua 2 frame, xử lý frame thứ 3. Tăng số này để video nhanh hơn.

# --- BIẾN TOÀN CỤC CHO VIỆC VẼ ĐƯỜNG ---
line_points = []
drawing = False

def draw_line_callback(event, x, y, flags, param):
    """
    Hàm callback xử lý sự kiện chuột để vẽ đường.
    """
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

# --- KHỞI TẠO ---
model = YOLO('yolov8n.pt')
video_path = 'input.mp4'
cap = cv2.VideoCapture(video_path)

# --- GIAI ĐOẠN CHỌN ĐƯỜNG ĐẾM ---
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

# --- GIAI ĐOẠN XỬ LÝ CHÍNH ---
people_count = 0
frame_counter = 0
last_results = None
# Dictionary để lưu trạng thái "bên" của từng đối tượng
object_states = defaultdict(int) 

# Hàm kiểm tra một điểm nằm ở phía nào của đường thẳng
def get_side_of_line(line_p1, line_p2, point):
    # Trả về > 0 nếu ở một bên, < 0 nếu ở bên kia, và 0 nếu trên đường
    return (point[0] - line_p1[0]) * (line_p2[1] - line_p1[1]) - (point[1] - line_p1[1]) * (line_p2[0] - line_p1[0])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_counter += 1
    cv2.line(frame, line_points[0], line_points[1], (0, 0, 255), 3)

    if frame_counter % FRAME_SKIP == 0:
        results = model.track(frame, persist=True, verbose=False, classes=0)
        last_results = results
    
    if last_results:
        try:
            boxes = last_results[0].boxes.xywh.cpu()
            track_ids = last_results[0].boxes.id.int().cpu().tolist()
        except AttributeError:
            track_ids = []

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            current_center = (float(x), float(y))

            # --- LOGIC ĐẾM ĐÃ THAY ĐỔI ---
            # Lấy trạng thái (bên) trước đó của đối tượng
            prev_side = object_states[track_id]
            
            # Lấy trạng thái (bên) hiện tại và chuẩn hóa về -1, 0, hoặc 1
            current_side_val = get_side_of_line(line_points[0], line_points[1], current_center)
            current_side = np.sign(current_side_val)

            # Nếu đối tượng đã đổi bên (và không phải lần đầu xuất hiện) -> đếm
            if prev_side != 0 and current_side != 0 and prev_side != current_side:
                people_count += 1
            
            # Cập nhật trạng thái bên mới cho đối tượng
            if current_side != 0:
                object_states[track_id] = current_side

            # --- HẾT LOGIC ĐẾM MỚI ---

            # Vẽ hộp bao (không vẽ ID)
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Hiển thị bộ đếm
    cv2.putText(frame, f"Count: {people_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Hiển thị khung hình kết quả
    cv2.imshow("YOLOv8 Tracking and Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()