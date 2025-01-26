from ultralytics import YOLO
import cv2

# โหลดโมเดลที่ฝึกมาแล้ว
model = YOLO('C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/runs/detect/train2/weights/best.pt')

# เส้นทางไฟล์วิดีโอที่ต้องการตรวจจับ
video_path = 'C:/Users/naren/Documents/Project3/Chicken_Project/Picture&Video/farmchick.mp4'

# เปิดวิดีโอสำหรับการอ่าน
cap = cv2.VideoCapture(video_path)

# ตรวจสอบว่าเปิดวิดีโอสำเร็จหรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิดไฟล์วิดีโอได้")
    exit()

# กำหนดค่าการบันทึกวิดีโอผลลัพธ์
output_path = 'OutputPicture/detected_farmchick.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ตัวเข้ารหัสวิดีโอ
fps = int(cap.get(cv2.CAP_PROP_FPS))  # เฟรมเรตของวิดีโอ
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # ความกว้างของเฟรม
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # ความสูงของเฟรม
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# ตัวแปรสำหรับนับจำนวนวัตถุทั้งหมดในวิดีโอ
total_objects = 0

# อ่านเฟรมจากวิดีโอแบบวนลูป
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ใช้โมเดล YOLO ตรวจจับวัตถุในเฟรม
    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)

    # นับจำนวนวัตถุในเฟรมปัจจุบัน
    frame_objects = len(results[0].boxes)
    total_objects += frame_objects

    # วาดกรอบและใส่ป้ายกำกับในเฟรม
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # พิกัดกรอบ (x1, y1, x2, y2)
        conf = box.conf[0]  # ค่าความมั่นใจ
        class_id = int(box.cls[0])  # ID ของคลาส
        label = f"{model.names[class_id]} {conf:.2f}"  # ชื่อคลาสและความมั่นใจ

        # วาดกรอบสี่เหลี่ยมรอบวัตถุ
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # กรอบสีเขียว

        # ใส่ข้อความชื่อคลาส
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ใส่ข้อความจำนวนวัตถุในเฟรม
    text = f"Objects in frame: {frame_objects}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # บันทึกเฟรมลงในไฟล์วิดีโอผลลัพธ์
    out.write(frame)

    # แสดงผลเฟรมในหน้าต่าง (ไม่บังคับ)
    cv2.imshow('Video Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # กด 'q' เพื่อออกจากการแสดงผล
        break

# ปิดการทำงานของวิดีโอและหน้าต่าง
cap.release()
out.release()
cv2.destroyAllWindows()

# แสดงจำนวนวัตถุทั้งหมดในวิดีโอ
print(f"จำนวนวัตถุที่ตรวจจับได้ทั้งหมดในวิดีโอ: {total_objects}")
