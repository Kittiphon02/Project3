import cv2
from ultralytics import YOLO

# 1) โหลดโมเดล (ไฟล์ best.pt ที่เทรนเสร็จแล้ว)
model_path = r"C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/runs/detect/train4/weights/best.pt"
model = YOLO(model_path)
# 2) ทำนายผล (predict) จากภาพ
image_path = r"C:/Users/naren/Documents/Project3/Chicken_Project/Picture&Video/fgdrgr.png"
results = model.predict(source=image_path, conf=0.5, save=False)
# 3) โหลดภาพด้วย OpenCV
image = cv2.imread(image_path)
# (ถ้ามีรายชื่อคลาส สามารถกำหนดใน list ได้ เช่น)
names = ['Scrape', 'Shocked', 'Sleep', 'Walk', 'to roost']  # แก้ตามคลาสที่เทรน
# 4) วนลูปดู bounding boxes และคลาสที่ตรวจจับได้
for box in results[0].boxes:
    # ดึงค่าพิกัด
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    # ดึงค่าความมั่นใจ
    conf = box.conf[0]
    # ดึงคลาส ID
    class_id = int(box.cls[0])
    # หากมีรายชื่อคลาส ก็ดึงชื่อคลาสมาแสดง (ป้องกัน index error ด้วย)
    class_name = names[class_id] if class_id < len(names) else f"ID:{class_id}"
    # เลือกสีและวาดกรอบ (rectangle)
    color = (0, 255, 0)  # BGR (เขียว)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # เขียนข้อความชื่อคลาส + ความมั่นใจ
    label = f"{class_name} {conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, color, 2)
# 5) แสดงภาพที่มีกรอบบนหน้าต่าง
cv2.imshow("Detection Result", image)
cv2.waitKey(0)  # รอให้กดปุ่มใด ๆ
cv2.destroyAllWindows()
