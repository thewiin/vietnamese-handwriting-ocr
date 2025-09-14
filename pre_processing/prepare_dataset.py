import os
import json
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# --- 1. CẤU HÌNH ---
DATASET_TYPE = 'word'
TARGET_IMAGE_HEIGHT = 48
AUGMENTATION_COUNT = 2


# --- 2. CÁC HÀM XỬ LÝ ẢNH (Không thay đổi) ---
def preprocess_image(image_path):
    """Đọc, chuyển ảnh xám và resize ảnh."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    h, w = img.shape
    scale = TARGET_IMAGE_HEIGHT / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, TARGET_IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)


def augment_image(image):
    """Tăng cường dữ liệu cho ảnh."""
    augmented_images = []
    h, w = image.shape
    # Xoay nhẹ
    angle = random.uniform(-2, 2)
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    augmented_images.append(rotated)
    # Làm nghiêng
    shear_factor = random.uniform(-0.1, 0.1)
    M_shear = np.array([[1, shear_factor, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(image, M_shear, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    augmented_images.append(sheared)
    return augmented_images


# --- 3. QUY TRÌNH XỬ LÝ CHÍNH ---
def run_full_preprocessing():
    # --- PHẦN 1: THU THẬP VÀ CHIA ĐƯỜNG DẪN ---
    print("▶️  Bước 1: Bắt đầu quét, thu thập và chia đường dẫn...")
    raw_root_dir = os.path.join('data', 'raw', f'UIT_HWDB_{DATASET_TYPE}')
    train_source_dir = os.path.join(raw_root_dir, 'train_data')
    test_source_dir = os.path.join(raw_root_dir, 'test_data')

    def collect_paths_from_source(source_dir):
        path_label_pairs = []
        for dirpath, _, filenames in os.walk(source_dir):
            if 'label.json' in filenames:
                json_path = os.path.join(dirpath, 'label.json')
                with open(json_path, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                for img_name, text_label in labels.items():
                    original_img_path = os.path.join(dirpath, img_name)
                    if os.path.exists(original_img_path):
                        path_label_pairs.append((original_img_path, text_label))
        return path_label_pairs

    train_val_paths = collect_paths_from_source(train_source_dir)
    test_paths = collect_paths_from_source(test_source_dir)

    random.shuffle(train_val_paths)
    val_split_index = int(len(train_val_paths) * 0.9)
    train_paths = train_val_paths[:val_split_index]
    val_paths = train_val_paths[val_split_index:]

    print(f"✅ Chia dữ liệu hoàn tất: {len(train_paths)} Train, {len(val_paths)} Val, {len(test_paths)} Test.")

    # --- PHẦN 2: XỬ LÝ ẢNH VÀ LƯU KẾT QUẢ ---
    print("\n▶️  Bước 2: Bắt đầu xử lý ảnh, tăng cường và lưu kết quả...")
    processed_dir = os.path.join('data', 'processed', DATASET_TYPE)
    output_images_dir = os.path.join(processed_dir, 'images')

    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(output_images_dir)

    # Hàm trợ giúp để xử lý từng tập dữ liệu
    def process_split(path_list, split_name):
        label_file_path = os.path.join(processed_dir, f'{split_name}_label.txt')
        with open(label_file_path, 'w', encoding='utf-8') as f_out:
            for original_path, label in tqdm(path_list, desc=f"Processing {split_name} set"):

                # Tạo tên file mới unique
                path_parts = original_path.split(os.sep)
                unique_prefix = '_'.join(path_parts[-3:-1])
                base_img_name = os.path.basename(original_path)
                new_name = f"{unique_prefix}_{base_img_name}"

                processed_img = preprocess_image(original_path)
                if processed_img is not None:
                    # Lưu ảnh gốc đã xử lý
                    cv2.imwrite(os.path.join(output_images_dir, new_name), processed_img)
                    f_out.write(f"images/{new_name}\t{label}\n")

                    # Áp dụng augmentation nếu là tập train
                    if split_name == 'train':
                        augmented_images = augment_image(processed_img)
                        for i, aug_img in enumerate(augmented_images[:AUGMENTATION_COUNT]):
                            base, ext = os.path.splitext(new_name)
                            aug_name = f"{base}_aug{i}{ext}"
                            cv2.imwrite(os.path.join(output_images_dir, aug_name), aug_img)
                            f_out.write(f"images/{aug_name}\t{label}\n")

    # Chạy xử lý cho từng tập
    process_split(train_paths, 'train')
    process_split(val_paths, 'val')
    process_split(test_paths, 'test')

    print("\n🎉 Hoàn tất! Dữ liệu đã được xử lý và sẵn sàng để huấn luyện.")


if __name__ == '__main__':
    run_full_preprocessing()