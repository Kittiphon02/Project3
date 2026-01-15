import os

file_path = "C:/Users/naren/Documents/Project3/Chicken_Project/ultralytics/roboflow/data.yaml"

if os.path.exists(file_path):
    print(f"File found: {file_path}")
else:
    print("File not found. Check the path or move the file to the correct location.")
