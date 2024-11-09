import cv2
from facenet_pytorch import MTCNN
import torch

# Bước 1: Khởi tạo MTCNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

# Bước 2: Thiết lập video capture
video_capture = cv2.VideoCapture(0)  # Sử dụng camera mặc định

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Chuyển đổi từ BGR sang RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện khuôn mặt
    boxes, _ = mtcnn.detect(img_rgb)

    # Vẽ khung cho các khuôn mặt được phát hiện
    if boxes is not None:
        for box in boxes:
            x_min, y_min, x_max, y_max = [int(b) for b in box]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Phóng to khung hình (800x600)
    frame_resized = cv2.resize(frame, (800, 600))

    # Hiển thị khung hình đã phóng to
    cv2.imshow('Face Detection', frame_resized)

    # Nhấn phím 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng tất cả cửa sổ
video_capture.release()
cv2.destroyAllWindows()
