import cv2
from ultralytics import YOLO
# 1) โหลดโมเดล (ไฟล์ best.pt ที่เทรนเสร็จแล้ว)
model_path = r"C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/runs/detect/train4/weights/best.pt"
model = YOLO(model_path)
# 2) กำหนดเส้นทางวิดีโอ (หรือใช้ 0 สำหรับ webcam)
video_path = r"C:/Users/naren/Documents/Project3/Chicken_Project/Picture&Video/Chicken6.mp4"
cap = cv2.VideoCapture(video_path)
# 3) รายชื่อคลาส (แก้ไขตามที่เทรน)
names = ['Scrape', 'Shocked', 'Sleep', 'Walk', 'to roost']
while True:
    ret, frame = cap.read()
    if not ret:
        break  # จบหากไม่มีเฟรมอ่านได้ (วิดีโอจบแล้ว)
    # 4) ใช้โมเดลเพื่อตรวจจับวัตถุในแต่ละเฟรม
    # โดยเรียกใช้งานโมเดลแบบ callable ส่ง image frame ตรงเข้าไป
    results = model(frame, conf=0.5)
    # 5) วนลูปผ่าน bounding boxes ที่ตรวจจับได้ในเฟรม
    for box in results[0].boxes:
        # ดึงค่าพิกัดกรอบ (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # ดึงค่าความมั่นใจ
        conf = box.conf[0]
        # ดึงคลาส ID
        class_id = int(box.cls[0])
        # ดึงชื่อคลาสจาก list names (หากมี)
        class_name = names[class_id] if class_id < len(names) else f"ID:{class_id}"
        
        # กำหนดสี (BGR) สำหรับกรอบ
        color = (0, 255, 0)  # สีเขียว
        # วาดกรอบบนเฟรม
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # เขียนข้อความ (ชื่อคลาส + ความมั่นใจ)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # 6) แสดงเฟรมที่มีการตรวจจับ
    cv2.imshow("Detection Result", frame)
    # กด 'q' เพื่อออกจาก loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
