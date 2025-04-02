import os
import cv2
import numpy as np

# Đường dẫn thư mục
IMAGE_FOLDER = "dataset/images/train"
LABEL_FOLDER = "dataset/labels/train"

# Kích thước đầu vào yêu cầu của MobileNetV3 Small
IMAGE_SIZE = (224, 224)
NUM_KEYPOINTS = 136  # 68 điểm x 2 tọa độ

def fix_labels():
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png'))]
    total_fixed = 0

    for img_file in image_files:
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        label_path = os.path.join(LABEL_FOLDER, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Kiểm tra ảnh có tồn tại không
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ Không thể đọc ảnh: {img_file}")
            continue

        # Kiểm tra kích thước ảnh
        h, w, c = image.shape
        if h < IMAGE_SIZE[0] or w < IMAGE_SIZE[1]:
            print(f"⚠️ Ảnh {img_file} quá nhỏ ({w}x{h}). Cần >= 224x224.")
            continue

        # Kiểm tra file nhãn có tồn tại không
        if not os.path.exists(label_path):
            print(f"❌ Không tìm thấy file nhãn cho: {img_file}")
            continue

        try:
            keypoints = np.loadtxt(label_path).flatten()
            num_points = keypoints.shape[0]

            if num_points < NUM_KEYPOINTS:
                print(f"🔧 Nhãn {label_path} thiếu {NUM_KEYPOINTS - num_points} giá trị. Đang bổ sung...")
                keypoints = np.pad(keypoints, (0, NUM_KEYPOINTS - num_points), mode='constant', constant_values=0)
                total_fixed += 1
            elif num_points > NUM_KEYPOINTS:
                print(f"🔧 Nhãn {label_path} có dư {num_points - NUM_KEYPOINTS} giá trị. Cắt bớt...")
                keypoints = keypoints[:NUM_KEYPOINTS]
                total_fixed += 1

            # Ghi đè lại file nhãn đã sửa
            np.savetxt(label_path, keypoints.reshape(-1, 2), fmt="%.6f")
        
        except Exception as e:
            print(f"❌ Lỗi đọc file nhãn {label_path}: {e}")

    print(f"\n✅ Đã sửa {total_fixed} file nhãn lỗi!")

# Chạy kiểm tra và sửa lỗi
fix_labels()
