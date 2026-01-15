import cv2
import os
import subprocess
import requests
from collections import Counter
from ultralytics import YOLO
import mysql.connector  # สำหรับเชื่อมต่อ MySQL
import time  # สำหรับ Timestamp

# ===========================
# ฟังก์ชันเชื่อมต่อและบันทึกข้อมูลลงในตาราง detection_events
# ===========================
def log_detection_event(frame_number, chicken_count, abnormal_details, detection_accuracy, output_frame_path):
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='chicken_project'
    )
    cursor = connection.cursor()
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
# ฟังก์ชันเชื่อมต่อและบันทึกข้อมูลลงในตาราง alerts
# ===========================
def log_alert(detection_id, alert_message, alert_image_path):
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='chicken_project'
    )
    cursor = connection.cursor()
    query = """
    INSERT INTO alerts
    (detection_id, alert_message, alert_image_path)
    VALUES (%s, %s, %s)
    """
    values = (detection_id, alert_message, alert_image_path)
    cursor.execute(query, values)
    connection.commit()
    alert_id = cursor.lastrowid
    cursor.close()
    connection.close()
    return alert_id

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
# ตรวจสอบและสร้างโฟลเดอร์สำหรับบันทึกไฟล์ถ้ายังไม่มี
# ===========================
output_folder = 'OutputPicture'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ===========================
# โหลดโมเดล YOLO ที่ฝึกมาแล้ว
# ===========================
model = YOLO('C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/runs/detect/train4/weights/best.pt')

# ===========================
# กำหนดชื่อคลาส
# ===========================
names = [
    'Scrape', 'Shocked', 'Sleep', 'Walk', 'to roost'
]

# ===========================
# สำหรับการตีกรอบให้แต่ละคลาสมีสีที่แตกต่างกัน (BGR)
# ===========================
color_mapping = {
    "Walk": (0, 255, 0),
    "Scrape": (255, 0, 0),
    "Shocked": (0, 0, 255),
    "Sleep": (0, 165, 255),
    "to roost": (0, 255, 255)
}

# ===========================
# เส้นทางไฟล์วิดีโอที่ต้องการตรวจจับ
# ===========================
video_path = 'C:/Users/naren/Documents/Project3/Chicken_Project/Picture&Video/Chicken6.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ ไม่สามารถเปิดไฟล์วิดีโอได้")
    exit()

# ===========================
# สร้างชื่อไฟล์วิดีโอผลลัพธ์ ไม่ให้ซ้ำ
# ===========================
timestamp_str = time.strftime("%Y%m%d_%H%M%S")
output_video_filename = f"detected_output_{timestamp_str}.mp4"
output_video_path = os.path.join(output_folder, output_video_filename)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

cv2.namedWindow('Video Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video Detection', 640, 480)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ตรวจจับวัตถุ
    abnormal_detected = False
    detected_classes = []
    confidence_list = []

    results = model.predict(
        source=frame,
        conf=0.6,
        iou=0.4,
        augment=True,
        save=False,
        verbose=False
    )

    # นับ bounding boxes
    frame_objects = len(results[0].boxes)

    # วาดกรอบ + label
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = names[class_id]

        detected_classes.append(class_name)
        confidence_list.append(conf)

        color = color_mapping.get(class_name, (255, 255, 255))
        if class_name == "Shocked":
            abnormal_detected = True

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # สมมติว่าทุกวัตถุคือไก่
    chicken_count = frame_objects
    cv2.putText(frame, f"Chicken Count: {chicken_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # ===============================================
    # สะสมข้อมูลลงฐานข้อมูลทุก 5 วินาที (fps * 5)
    # ===============================================
    if frame_count % (fps * 5) == 0:
        # 1) เก็บข้อมูล class ที่พบ + confidence
        c = Counter(detected_classes)
        abnormal_details = str(c)

        # 2) คำนวณ Average Confidence เป็นเปอร์เซ็นต์
        if len(confidence_list) > 0:
            avg_conf = sum(confidence_list) / len(confidence_list)
            detection_accuracy = avg_conf * 100
        else:
            detection_accuracy = 0.0

        # 3) แปลง detection_accuracy เป็น string (ถ้าคอลัมน์เป็น VARCHAR)
        detection_accuracy_str = f"{detection_accuracy:.2f}"

        # 4) บันทึกในตาราง detection_events
        detection_id = log_detection_event(
            frame_number=frame_count,
            chicken_count=chicken_count,
            abnormal_details=abnormal_details,
            detection_accuracy=detection_accuracy_str,  # <--- ส่งค่า accuracy แทน total/confidence
            output_frame_path=None  # ไม่มีการบันทึก path หรือจะใส่ก็ได้
        )

        # ถ้าตรวจพบ Shocked => แจ้งเตือน
        if abnormal_detected:
            alert_message = "⚠️ ตรวจพบวัตถุผิดปกติในวิดีโอ:\n- Shocked"
            alert_image_path = os.path.join(output_folder, 'alert_frame.jpg')
            cv2.imwrite(alert_image_path, frame)

            send_telegram_alert_with_image(alert_message, alert_image_path)
            alert_id = log_alert(detection_id, alert_message, alert_image_path)
            print(f"Alert logged with alert_id: {alert_id}")

        # ถ้าพบไก่น้อยกว่า 3 ตัว => แจ้งเตือน
        if chicken_count < 3:
            alert_message = f"⚠️ ตรวจจับไก่น้อยกว่า 3 ตัว (พบ {chicken_count} ตัว) ในวิดีโอ"
            alert_image_path = os.path.join(output_folder, 'alert_frame_low_chicken.jpg')
            cv2.imwrite(alert_image_path, frame)

            send_telegram_alert_with_image(alert_message, alert_image_path)
            alert_id = log_alert(detection_id, alert_message, alert_image_path)
            print(f"Alert logged with alert_id: {alert_id}")

    out.write(frame)
    cv2.imshow('Video Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ วิดีโอที่ตรวจจับเสร็จแล้วถูกบันทึกที่: {output_video_path}")
