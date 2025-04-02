from ultralytics import YOLO
import cv2

# Load mô hình đã huấn luyện
model = YOLO(r"C:\Users\Admin\runs\detect\face_detection5\weights\best.pt")  # Đường dẫn đến file mô hình

# Khởi động webcam
cap = cv2.VideoCapture(0)  # Sử dụng webcam (thay 0 bằng đường dẫn video nếu muốn test video)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán với YOLOv8
    results = model(frame, conf=0.5)  # conf=0.5 là ngưỡng tin cậy

    # Vẽ kết quả lên frame
    for result in results:
        frame = result.plot()

    # Hiển thị video với bounding box
    cv2.imshow("Face Detection", frame)

    # Nhấn 'Q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
