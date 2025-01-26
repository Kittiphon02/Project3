import cv2

# เปิดการเชื่อมต่อกับ Iriun Webcam
cap = cv2.VideoCapture(0)  # เปลี่ยน 0 เป็น 1 หรือหมายเลขอื่นหากมีหลายกล้อง

# ตรวจสอบว่าเปิดกล้องสำเร็จหรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิด Iriun Webcam ได้")
    exit()

print("Iriun Webcam เปิดใช้งานสำเร็จ! กด 'q' เพื่อออก")

# แสดงผลกล้องแบบเรียลไทม์
while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถอ่านข้อมูลจากกล้องได้")
        break

    # แสดงภาพที่ได้จากกล้อง
    cv2.imshow('Iriun Webcam Test', frame)

    # กด 'q' เพื่อหยุดการทำงาน
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการเชื่อมต่อกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
