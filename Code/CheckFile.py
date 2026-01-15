##ตรวจสอบว่ามีไฟล์หรือโฟลเดอร์ที่ระบุไว้ในเส้นทาง (path) หรือไม่

import os

train_images = "C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/train/images"
val_images = "C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/valid/images"

print("Train images exist:", os.path.exists(train_images))
print("Val images exist:", os.path.exists(val_images))
