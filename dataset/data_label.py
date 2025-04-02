import cv2
import os
import pandas as pd
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Khởi tạo MTCNN
detector = MTCNN()

# Thư mục ảnh
IMAGE_DIR = "train/class1"
OUTPUT_CSV = "labels.csv"

# Kiểm tra thư mục tồn tại
if not os.path.exists(IMAGE_DIR):
    print(f"❌ Thư mục {IMAGE_DIR} không tồn tại!")
    exit()

# Lấy danh sách ảnh
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
data = []

# Hàm xử lý từng ảnh
def process_image(image_name):
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = cv2.imread(image_path)

    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        print(f"⚠️ Không thể đọc ảnh: {image_name}")
        return None

    h, w, _ = image.shape  # Kích thước ảnh gốc
    faces = detector.detect_faces(image)  # Phát hiện khuôn mặt

    results = []
    for idx, face in enumerate(faces):
        x, y, width, height = face["box"]

        # Giới hạn giá trị không bị âm
        x, y = max(0, x), max(0, y)

        # Chuẩn hóa tọa độ (x, y, w, h) về 0 -> 1
        x_norm = x / w
        y_norm = y / h
        w_norm = width / w
        h_norm = height / h

        # Bỏ qua bounding box quá nhỏ hoặc quá lớn
        if w_norm < 0.05 or h_norm < 0.05 or w_norm > 0.9 or h_norm > 0.9:
            print(f"⚠️ Bỏ qua bbox bất thường: {image_name} -> {x_norm, y_norm, w_norm, h_norm}")
            continue

        # Đánh số bbox theo faceID để tránh trùng lặp
        results.append([f"{image_name}_{idx}", image_name, x_norm, y_norm, w_norm, h_norm])
    
    return results if results else None

# Dùng ThreadPoolExecutor để chạy đa luồng
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(executor.map(process_image, image_files), total=len(image_files), desc="Processing images"))

# Gộp kết quả
data = [item for sublist in results if sublist for item in sublist]

# Lưu vào CSV nếu có dữ liệu
if data:
    df = pd.DataFrame(data, columns=["face_id", "image", "x", "y", "w", "h"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Đã lưu {len(data)} bounding boxes vào {OUTPUT_CSV}!")
else:
    print("⚠️ Không có dữ liệu để lưu!")
