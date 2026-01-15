import requests

TELEGRAM_API_TOKEN = "7772933089:AAFtIeUQNg-aisV6inVaWe-Z9IFUYODuTJQ"
CHAT_ID = "8195254982"
MESSAGE = "Test message"

url = f"https://api.telegram.org/bot{TELEGRAM_API_TOKEN}/sendMessage"
params = {
    "chat_id": CHAT_ID,
    "text": MESSAGE
}
r = requests.get(url, params=params)
print(r.status_code, r.text)
