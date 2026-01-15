import cv2
import matplotlib.pyplot as plt
import os

# โหลดภาพ
image_path = r'C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/test/images/-_mp4-0005_jpg.rf.8555f8080f293e0c3c30e69572d2abc8.jpg'
image = cv2.imread(image_path)

# ตรวจสอบว่าโหลดภาพได้หรือไม่
if image is None:
    print(f"ไม่สามารถโหลดไฟล์ภาพจาก: {image_path}")
else:
     # กำหนดพิกัดของกรอบหลายกรอบ (มุมซ้ายบนและมุมขวาล่าง)
    boxes = [
        ((5, 4), (482, 346)),       # กรอบที่ 1
        ((100, 100), (300, 250)),   # กรอบที่ 2
        # ((200, 50), (400, 200))     # กรอบที่ 3
    ]

    # วาดกรอบสี่เหลี่ยมหลายกรอบบนภาพ
    for idx, (top_left, bottom_right) in enumerate(boxes):
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # (ภาพ, มุมซ้ายบน, มุมขวาล่าง, สี, ความหนา)

    # สร้างชื่อไฟล์ตามชื่อภาพเดิม
    image_filename = os.path.basename(image_path).split('.')[0]  # แยกชื่อไฟล์โดยไม่เอานามสกุล

    # บันทึกภาพที่ตีกรอบโดยใช้ชื่อภาพเดิม
    save_image_path = fr'C:/Users/naren/Documents/Project3/Chicken_Project/Detect_image/normal/{image_filename}.jpg'
    os.makedirs(os.path.dirname(save_image_path), exist_ok=True)
    cv2.imwrite(save_image_path, image)

    # บันทึกค่าระยะของกรอบโดยใช้ชื่อไฟล์ภาพ
    label_path = fr'C:/Users/naren/Documents/Project3/Chicken_Project/Detect_Label/{image_filename}.txt'
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w') as file:
        file.write(f"Top Left: {top_left}\n")
        file.write(f"Bottom Right: {bottom_right}\n")

    # แสดงภาพที่มีกรอบ
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
