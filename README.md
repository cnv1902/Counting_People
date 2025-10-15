# Counting People with YOLOv8

Đây là dự án đếm số người đi qua một đường trong video sử dụng YOLOv8 và OpenCV.

## Yêu cầu

- Python 3.8+
- OpenCV (`pip install opencv-python`)
- Ultralytics YOLO (`pip install ultralytics`)
- NumPy (`pip install numpy`)

## Cài đặt môi trường và thư viện

1. **Tạo môi trường ảo (khuyến nghị):**
   ```bash
   python -m venv venv
   ```
2. **Kích hoạt môi trường ảo:**
   - Trên Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Trên macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. **Cài đặt các thư viện cần thiết:**
   - Sử dụng file `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```
   - Hoặc cài từng thư viện:
     ```bash
     pip install ultralytics==8.3.45 opencv-python==4.10.0.84 torch>=2.1.0 torchvision>=0.16.0 torchaudio>=2.1.0 numpy>=1.26.0
     ```

4. **(Tuỳ chọn) Thoát môi trường ảo:**
   ```bash
   deactivate
   ```

## Hướng dẫn sử dụng

1. Đặt video cần xử lý vào cùng thư mục với mã nguồn và đổi tên thành `input.mp4` (hoặc sửa lại đường dẫn trong `main.py`).
2. Chạy chương trình:
   ```bash
   python main.py
   ```
3. Khi cửa sổ hiện lên, dùng chuột kéo để vẽ đường đếm. Nhấn `c` để xác nhận đường, hoặc `Esc` để thoát.
4. Chương trình sẽ bắt đầu đếm số người đi qua đường đã chọn. Nhấn `q` để kết thúc.

## Cấu hình

- Có thể chỉnh tốc độ xử lý bằng cách thay đổi biến `FRAME_SKIP` trong `main.py`.

## Ghi chú

- Mô hình sử dụng là `yolov8n.pt` (YOLOv8 Nano). Có thể thay đổi sang mô hình khác nếu cần.
- Chỉ đếm người (class 0 của YOLO).

## Liên hệ

Mọi thắc mắc vui lòng liên hệ qua email hoặc issue trên GitHub.