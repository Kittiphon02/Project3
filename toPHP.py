# ===========================
# ฟังก์ชันเชื่อมต่อและบันทึกข้อมูลลงในตาราง detection_events
# ===========================
def log_detection_event(frame_number, total_objects, abnormal_details, confidence_scores, output_frame_path):
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',  # ถ้าตั้งรหัสผ่านไว้ ให้ใส่ให้ถูกต้อง
        database='chicken_project'
    )
    cursor = connection.cursor()
    query = """
    INSERT INTO detection_events
    (frame_number, total_objects, abnormal_details, confidence_scores, output_frame_path)
    VALUES (%s, %s, %s, %s, %s)
    """
    values = (frame_number, total_objects, abnormal_details, confidence_scores, output_frame_path)
    cursor.execute(query, values)
    connection.commit()
    detection_id = cursor.lastrowid
    cursor.close()
    connection.close()
    return detection_id

# ===========================
# ฟังก์ชันเชื่อมต่อและบันทึกข้อมูลลงในตาราง alerts
# ===========================
def log_alert(detection_id, alert_message, alert_image_path):
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',  # ถ้าตั้งรหัสผ่านไว้ ให้ใส่ให้ถูกต้อง
        database='chicken_project'
    )
    cursor = connection.cursor()
    query = """
    INSERT INTO alerts
    (detection_id, alert_message, alert_image_path)
    VALUES (%s, %s, %s)
    """
    values = (detection_id, alert_message, alert_image_path)
    cursor.execute(query, values)
    connection.commit()
    alert_id = cursor.lastrowid
    cursor.close()
    connection.close()
    return alert_id
