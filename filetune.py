import os
from ultralytics import YOLO

if __name__ == '__main__':
    # Đường dẫn đến mô hình đã huấn luyện trước đó
    model_path = r"C:\Users\Admin\runs\detect\train17\weights\best.pt"

    # Load mô hình YOLO đã huấn luyện trước
    model = YOLO(model_path)

    # Đường dẫn đến file cấu hình dataset (data.yaml)
    data_yaml = "dataset/data.yaml"

    # Fine-tune mô hình với dữ liệu mới
    results = model.train(
        data=data_yaml,          # File cấu hình dataset
        epochs=50,               # Số epoch huấn luyện
        imgsz=640,               # Kích thước ảnh đầu vào
        batch=16,                # Kích thước batch
        name="finetune_face_detection",  # Tên thư mục lưu kết quả
        patience=10,             # Số epoch chờ nếu không cải thiện
        device=0,                # Sử dụng GPU
        pretrained=True          # Sử dụng trọng số đã huấn luyện trước đó
    )

    print("🎯 Hoàn thành fine-tuning mô hình nhận diện khuôn mặt!")