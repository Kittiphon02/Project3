from ultralytics import YOLO
import cv2
import requests
from collections import Counter

# ตั้งค่า Telegram Bot
TELEGRAM_API_TOKEN = "7772933089:AAFe3e_KyH6zIniox4S1RyvgedYFIYuuwBY"  # แทนที่ด้วย API Token ของคุณ
CHAT_ID = "8195254982"  # แทนที่ด้วย Chat ID ของคุณ

# ฟังก์ชันส่งข้อความพร้อมรูปภาพไปยัง Telegram
def send_telegram_alert_with_image(message, image_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        data = {"chat_id": CHAT_ID, "caption": message}
        files = {"photo": photo}
        response = requests.post(url, data=data, files=files)
        if response.status_code == 200:
            print("ส่งข้อความและรูปภาพไปยัง Telegram สำเร็จ")
        else:
            print(f"เกิดข้อผิดพลาดในการส่งข้อความและรูปภาพ: {response.status_code}")

# โหลดโมเดลที่ฝึกมาแล้ว
model = YOLO('C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/runs/detect/train2/weights/best.pt')

# กำหนดชื่อคลาส
names = [
    'Breathing', 'abnormal chicken', 'allo preening', 'bird flu', 'crouch chicken',
    'dead', 'drinking', 'feeding', 'fighting', 'leg stretching', 'menacing',
    'newcastle', 'pecking', 'running', 'self preening', 'sleeping chickens',
    'stand bath', 'walking', 'wing and leg stretching', 'wing flap', 'wingstretching'
]

# รายการวัตถุผิดปกติที่ต้องการแจ้งเตือน
abnormal_classes = ['abnormal chicken', 'bird flu', 'dead', 'leg stretching', 'newcastle']

# เส้นทางภาพที่ต้องการตรวจจับ
image_path = 'C:/Users/naren/Documents/Project3/Chicken_Project/Picture&Video/siksik.png'

# ใช้โมเดลเพื่อตรวจจับวัตถุในภาพ
results = model.predict(source=image_path, save=True, conf=0.5)

# นับจำนวนวัตถุที่ตรวจจับได้ทั้งหมด
detected_objects = len(results[0].boxes)

# ตัวนับสำหรับวัตถุผิดปกติ
abnormal_counter = Counter()

# อ่านภาพต้นฉบับ
image = cv2.imread(image_path)

# ตรวจสอบวัตถุผิดปกติและใส่กรอบในภาพ
for box in results[0].boxes:
    class_id = int(box.cls[0])  # ID ของคลาส
    conf = box.conf[0]  # ค่าความมั่นใจ
    class_name = names[class_id]  # ชื่อคลาส

    # วาดกรอบและแสดงข้อความในภาพ
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # สีเขียวสำหรับกรอบทั่วไป
    label = f"{class_name} {conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ถ้าคลาสที่ตรวจจับอยู่ในรายการผิดปกติ
    if class_name in abnormal_classes:
        abnormal_counter[class_name] += 1
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # สีแดงสำหรับวัตถุผิดปกติ
        cv2.putText(image, "ABNORMAL", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# บันทึกภาพที่มีการไฮไลต์วัตถุผิดปกติ
output_image_path = 'OutputPicture/detected_out.jpg'
cv2.imwrite(output_image_path, image)

# ส่งภาพที่มีกรอบวัตถุพร้อมข้อความสรุปไปยัง Telegram
if abnormal_counter:
    summary_message = "📊 สรุปจำนวนวัตถุผิดปกติที่ตรวจพบ:\n"
    for class_name, count in abnormal_counter.items():
        summary_message += f"- {class_name}: {count}\n"
    send_telegram_alert_with_image(summary_message, output_image_path)

# แสดงจำนวนวัตถุที่ตรวจจับได้
print(f"จำนวนวัตถุที่ตรวจจับได้ในภาพ: {detected_objects}")

# แสดงจำนวนวัตถุผิดปกติ
print("จำนวนวัตถุผิดปกติที่ตรวจจับได้:")
for class_name, count in abnormal_counter.items():
    print(f"{class_name}: {count}")
