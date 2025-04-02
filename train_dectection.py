from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    model.train(data="./dataset/data.yaml", 
                epochs=50, 
                batch=8, 
                workers=6,
                name="face_detection",
                device="cuda"
                )  # workers=6 để sử dụng multiprocessing
    print("🎯 Huấn Luyện hoàn tất!")