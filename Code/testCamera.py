import cv2

# เปิดการเชื่อมต่อกับกล้อง Dell 2K QHD
camera_index = 2  # ลองเปลี่ยนหมายเลขนี้หากกล้องที่ต้องการไม่ทำงาน
cap = cv2.VideoCapture(camera_index)

# ตรวจสอบว่าเปิดกล้องสำเร็จหรือไม่
if not cap.isOpened():
    print(f"ไม่สามารถเปิดกล้องหมายเลข {camera_index} ได้")
    exit()

print("Dell 2K QHD Webcam เปิดใช้งานสำเร็จ! กด 'q' เพื่อออก")

# แสดงผลกล้องแบบเรียลไทม์
while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านข้อมูลจากกล้องได้")
        break

    # แสดงภาพที่ได้จากกล้อง
    cv2.imshow('Dell 2K QHD Webcam Test', frame)

    # กด 'q' เพื่อหยุดการทำงาน
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการเชื่อมต่อกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
