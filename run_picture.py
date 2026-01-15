import cv2
import os
import time
import requests
from collections import Counter
from ultralytics import YOLO
import mysql.connector

# ===========================
# ฟังก์ชันบันทึกภาพด้วยการเพิ่ม Timestamp ต่อท้ายชื่อไฟล์
# ===========================
def save_image_with_timestamp(image, output_folder, base_filename="detected_ou"):
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp_str}.jpg"

    output_image_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_image_path, image)
    print(f"Saved: {output_image_path}")
    return output_image_path

# ===========================
# ฟังก์ชันเชื่อมต่อและบันทึกข้อมูลลงในตาราง detection_events
# (แก้ไขให้ส่ง detection_accuracy แทน confidence_scores)
# ===========================
def log_detection_event(frame_number, chicken_count, abnormal_details, detection_accuracy, output_frame_path):
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='chicken_project'
    )
    cursor = connection.cursor()

    # ต้องมีคอลัมน์ detection_accuracy อยู่ในตาราง detection_events
    query = """
    INSERT INTO detection_events
    (frame_number, chicken_count, abnormal_details, detection_accuracy, output_frame_path)
    VALUES (%s, %s, %s, %s, %s)
    """
    values = (frame_number, chicken_count, abnormal_details, detection_accuracy, output_frame_path)

    cursor.execute(query, values)
    connection.commit()

    detection_id = cursor.lastrowid
    cursor.close()
    connection.close()
    return detection_id

# ===========================
# ตั้งค่า Telegram Bot
# ===========================
TELEGRAM_API_TOKEN = "7772933089:AAFtIeUQNg-aisV6inVaWe-Z9IFUYODuTJQ"
CHAT_ID = "8195254982"

def send_telegram_alert_with_image(message, image_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        data = {"chat_id": CHAT_ID, "caption": message}
        files = {"photo": photo}
        response = requests.post(url, data=data, files=files)
        if response.status_code == 200:
            print("✅ ส่งแจ้งเตือนไปยัง Telegram สำเร็จ")
        else:
            print(f"❌ เกิดข้อผิดพลาดในการส่ง Telegram: {response.status_code}")

# ===========================
# ตรวจสอบและสร้างโฟลเดอร์บันทึกผลลัพธ์
# ===========================
output_folder = 'OutputPicture'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ===========================
# โหลดโมเดล YOLO ที่ฝึกไว้
# ===========================
model_path = r"C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/runs/detect/train4/weights/best.pt"
model = YOLO(model_path)

# ===========================
# ชื่อคลาสที่เทรน
# ===========================
names = [
    'Scrape', 'Shocked', 'Sleep', 'Walk', 'to roost'
]

color_mapping = {
    "Walk": (0, 255, 0),
    "Scrape": (255, 0, 0),
    "Shocked": (0, 0, 255),
    "Sleep": (0, 165, 255),
    "to roost": (0, 255, 255)
}

abnormal_classes = ['Shocked']

# ===========================
# ภาพที่ต้องการตรวจจับ
# ===========================
image_path = r"C:/Users/naren/Documents/Project3/Chicken_Project/Picture&Video/28.png"

# ตรวจจับด้วย YOLO
results = model.predict(source=image_path, conf=0.5, save=False)

# จำนวนวัตถุทั้งหมด
frame_objects = len(results[0].boxes)
chicken_count = frame_objects

abnormal_counter = Counter()
image = cv2.imread(image_path)

# ลิสต์เก็บค่าความเชื่อมั่น
confidences = []

if image is None:
    print(f"ไม่พบรูป {image_path}")
else:
    # วนลูปแต่ละ bounding box
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        confidences.append(conf)

        class_id = int(box.cls[0])
        class_name = names[class_id] if class_id < len(names) else f"cls{class_id}"

        color = color_mapping.get(class_name, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if class_name in abnormal_classes:
            abnormal_counter[class_name] += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, "ABNORMAL", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # แสดง count
    cv2.putText(image, f"Chicken Count: {chicken_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # บันทึกภาพผลลัพธ์
    output_image_path = save_image_with_timestamp(image, output_folder, "detected_ou")

    # แจ้งเตือนถ้าพบ Shocked
    if abnormal_counter:
        summary_message = "📊 สรุปจำนวนวัตถุผิดปกติที่ตรวจพบ:\n"
        for cname, count in abnormal_counter.items():
            summary_message += f"- {cname}: {count}\n"
        send_telegram_alert_with_image(summary_message, output_image_path)

    print(f"จำนวนวัตถุที่ตรวจจับได้ในภาพ: {frame_objects}")
    print("จำนวนวัตถุผิดปกติที่ตรวจจับได้:")
    for cname, count in abnormal_counter.items():
        print(f"- {cname}: {count}")

    # คำนวณค่าเฉลี่ย Confidence → แปลงเป็นเปอร์เซ็นต์
    if len(confidences) == 0:
        detection_accuracy = 0.0
    else:
        avg_conf = sum(confidences) / len(confidences)
        detection_accuracy = avg_conf * 100.0

    # แปลงค่าให้เป็นข้อความ (หรือจะเก็บเป็น float ก็ได้)
    detection_accuracy_str = f"{detection_accuracy:.2f}"

    # บันทึกลงฐานข้อมูล
    abnormal_details = str(dict(abnormal_counter))
    detection_id = log_detection_event(
        frame_number=1,
        chicken_count=chicken_count,
        abnormal_details=abnormal_details,
        detection_accuracy=detection_accuracy_str,   # <-- ส่งค่า Accuracy แทน confidence_scores
        output_frame_path=output_image_path
    )
    print("ข้อมูลการตรวจจับถูกบันทึกด้วย detection_id =", detection_id)

    print("\n==== การคำนวณ Accuracy จาก Average Confidence ====")
    print(f"Detection Accuracy (Confidence %) = {detection_accuracy_str}%")
