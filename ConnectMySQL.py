import cv2
import os
import requests
from collections import Counter
from ultralytics import YOLO
import mysql.connector

# ===========================
# ฟังก์ชันเชื่อมต่อและบันทึกข้อมูลลงในตาราง detection_events
# ===========================
def log_detection_event(frame_number, chicken_count, abnormal_details, confidence_scores, output_frame_path):
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',  # ถ้าไม่มีรหัสผ่านให้ใส่เป็นค่าว่าง
        database='chicken_project'
    )
    cursor = connection.cursor()
    query = """
    INSERT INTO detection_events 
    (frame_number, chicken_count, abnormal_details, confidence_scores, output_frame_path)
    VALUES (%s, %s, %s, %s, %s)
    """
    values = (frame_number, chicken_count, abnormal_details, confidence_scores, output_frame_path)
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

# ... โค้ดส่วนอื่น ๆ สำหรับการตรวจจับวัตถุ ...
