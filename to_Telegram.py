import requests

# ===========================
# ตั้งค่า Telegram Bot
# ===========================
TELEGRAM_API_TOKEN = "7772933089:AAFtIeUQNg-aisV6inVaWe-Z9IFUYODuTJQ"  # ใส่ API Token ของคุณ
CHAT_ID = "8195254982"  # ใส่ Chat ID ของคุณ

# ===========================
# ฟังก์ชันส่งข้อความพร้อมรูปภาพไปยัง Telegram
# ===========================
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
# ตัวอย่างการเรียกใช้ฟังก์ชัน
# ===========================
if __name__ == '__main__':
    message = "นี่คือข้อความแจ้งเตือนพร้อมรูปภาพ"
    image_path = "C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/runs/detect/predict/fgdrgr.jpg"  
    send_telegram_alert_with_image(message, image_path)
