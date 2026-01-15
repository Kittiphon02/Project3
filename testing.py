import cv2
import os
import random  # สำหรับการสุ่มเลือกภาพ
from ultralytics import YOLO

# ==========================
# 1) สร้างข้อมูล Ground Truth ของรูปทั้งหมด
#    กรณีนี้กำหนดสมมติว่ามีรูปภาพ 15 รูป
#    แต่เราจะสุ่มมาใช้ทดสอบแค่ 10 รูป
# ==========================
all_ground_truth = {
    "01.jpg": 1,
    "02.jpg": 1,
    "03.jpg": 3,
    "04.jpg": 5,
    "05.jpg": 1,
    "06.jpg": 1,
    "07.jpg": 1,
    "08.jpg": 2,
    "09.jpg": 1,
    "10.jpg": 1,
    "11.jpg": 3,
    "12.jpg": 1,
    "13.jpg": 2,
    "14.jpg": 2,
    "15.jpg": 2
}

# ==========================
# 2) เลือกมา 10 รูป (จะสุ่มหรือจะกำหนดเองก็ได้)
#    ตัวอย่างใช้ random.sample เลือก 10 รูปจากรายการใหญ่
# ==========================
num_to_test = 10
random_keys = random.sample(list(all_ground_truth.keys()), num_to_test)  # สุ่ม 10 รูป
# ถ้าคุณต้องการเลือกเองก็กำหนดเป็น list: 
# random_keys = ["01.jpg", "02.jpg", ..., "10.jpg"]

# สร้าง ground_truth เฉพาะ 10 รูปที่เลือก
test_ground_truth = {k: all_ground_truth[k] for k in random_keys}

# ==========================
# 3) โหลดโมเดล YOLO ที่ฝึกแล้ว
# ==========================
model_path = r"C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/runs/detect/train4/weights/best.pt"
model = YOLO(model_path)

# ==========================
# 4) พาธไปยังโฟลเดอร์เก็บรูปภาพ
#    (ปรับ path ให้ตรงกับโปรเจกต์ของคุณ)
# ==========================
image_folder = r"C:/Users/naren/Documents/Project3/Chicken_Project/Picture&Video"

# ==========================
# 5) ตัวแปรเก็บจำนวนภาพที่ทำนายถูกต้อง
# ==========================
correct_count = 0

# ==========================
# 6) วนลูปประมวลผลภาพ 10 รูป
# ==========================
for img_filename, expected_chicken in test_ground_truth.items():
    img_path = os.path.join(image_folder, img_filename)
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"ไม่พบรูป {img_path}, ข้ามภาพนี้ไป")
        continue

    # ใช้โมเดล YOLO ตรวจจับ
    results = model.predict(source=image, conf=0.5, save=False)

    # นับจำนวนวัตถุ (สมมติว่าทุกวัตถุ = ไก่)
    num_detected = len(results[0].boxes)

    # เปรียบเทียบผลลัพธ์กับ Ground Truth
    if num_detected == expected_chicken:
        correct_count += 1
        print(f"[{img_filename}] ถูกต้อง (ตรวจได้ {num_detected} คาดไว้ {expected_chicken})")
    else:
        print(f"[{img_filename}] ผิด (ตรวจได้ {num_detected} คาดไว้ {expected_chicken})")

# ==========================
# 7) สรุป Accuracy
# ==========================
total_images = len(test_ground_truth)
accuracy = (correct_count / total_images) * 100

print("\n===== สรุปผลการตรวจจับเฉพาะ 10 รูปที่เลือก =====")
print(f"รูปทั้งหมด: {total_images}")
print(f"ตรวจจับตรงตาม Ground Truth: {correct_count}")
print(f"Accuracy = {accuracy:.2f}%")
