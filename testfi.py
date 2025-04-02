from ultralytics import YOLO
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 📌 Load mô hình YOLO để phát hiện khuôn mặt
yolo_model = YOLO(r"C:\Users\Admin\runs\detect\face_detection5\weights\best.pt")

# 📌 Load mô hình MobileNetV3 để nhận diện điểm rời rạc
keypoint_model = load_model("mobilenetv3_keypoints.h5")

# 🟢 Khởi động webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 🔍 Bước 1: Phát hiện khuôn mặt bằng YOLO
    results = yolo_model(frame, conf=0.5)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]

            # Chỉ tiếp tục nếu phát hiện khuôn mặt
            if face is not None and face.shape[0] > 0 and face.shape[1] > 0:
                # Resize về 224x224
                face_resized = cv2.resize(face, (224, 224))
                face_resized = np.expand_dims(face_resized, axis=0) / 255.0  # Chuẩn hóa

                # 🔍 Bước 2: Dự đoán điểm rời rạc với MobileNetV3
                keypoints = keypoint_model.predict(face_resized)[0]
                keypoints = keypoints.reshape(-1, 2)  # Đưa về dạng (68,2)

                # Hiển thị điểm trên khuôn mặt
                for (x, y) in keypoints:
                    x = int(x * (x2 - x1)) + x1
                    y = int(y * (y2 - y1)) + y1
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Vẽ điểm xanh

            # Vẽ khung mặt lên ảnh
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 🖥 Hiển thị video
    cv2.imshow("Face Detection & Keypoints", frame)

    # Nhấn 'Q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
