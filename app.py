from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

# OpenCV dùng file .yml để lưu "Cơ sở dữ liệu" khuôn mặt
# Render là hệ thống file "chỉ đọc", ta phải lưu vào thư mục /tmp
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
FACE_DB_PATH = "/tmp/face_data.yml"

# Ánh xạ UID (string) sang ID (số nguyên) vì OpenCV chỉ nhận số
uid_to_id_map = {}
id_to_uid_map = {}
next_id = 1

# Hàm tải CSDL (nếu có) khi server khởi động
def load_database():
    global uid_to_id_map, id_to_uid_map, next_id
    if os.path.exists(FACE_DB_PATH):
        try:
            recognizer.read(FACE_DB_PATH)
            # (Với Render free tier, CSDL này sẽ mất khi server restart/ngủ)
            print("Database loaded from previous session (if any)")
        except Exception as e:
            print(f"Could not read DB: {e}")

load_database()

# Hàm trợ giúp: chuyển ảnh base64 sang ảnh OpenCV (ảnh xám)
def b64_to_cv_image(b64_string):
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]

    image_data = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_data))
    image_np = np.array(image)
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    return gray

# API 1: Thêm/Enroll Giảng viên
@app.route('/enroll', methods=['POST'])
def enroll_face():
    global next_id
    data = request.json
    if 'uid' not in data or 'image_base64' not in data:
        return jsonify({"status": "error", "message": "Missing uid or image"}), 400

    try:
        target_uid = data['uid']
        gray_image = b64_to_cv_image(data['image_base64'])

        # Tìm khuôn mặt
        faces = detector.detectMultiScale(gray_image, 1.3, 5)
        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face found"}), 400

        (x, y, w, h) = faces[0] # Lấy mặt đầu tiên
        face_roi = gray_image[y:y+h, x:x+w]

        # Gán ID (số) cho UID (chữ)
        if target_uid not in uid_to_id_map:
            uid_to_id_map[target_uid] = next_id
            id_to_uid_map[next_id] = target_uid
            current_id = next_id
            next_id += 1
        else:
            current_id = uid_to_id_map[target_uid]

        # "Train" (cập nhật mô hình)
        recognizer.update([face_roi], np.array([current_id]))

        # Lưu CSDL vào file tạm
        recognizer.write(FACE_DB_PATH)

        print(f"Enrolled/Updated face for UID: {target_uid} (mapped to ID: {current_id})")
        return jsonify({"status": "success", "message": "Face enrolled/updated"}), 201

    except Exception as e:
        print(f"Enroll error: {e}")
        return jsonify({"status": "error", "message": "Could not process image"}), 500

# API 2: Xác minh (Verify) - 1 đối 1
@app.route('/verify', methods=['POST'])
def verify_face():
    data = request.json
    if 'uid' not in data or 'image_base64' not in data:
        return jsonify({"status": "error", "message": "Missing uid or image"}), 400

    try:
        target_uid = data['uid']

        # 1. Kiểm tra UID có trong CSDL không
        if target_uid not in uid_to_id_map:
             return jsonify({"status": "error", "match": False, "message": "UID not enrolled"}), 404

        target_id = uid_to_id_map[target_uid]
        gray_image = b64_to_cv_image(data['image_base64'])

        # 2. Tìm mặt
        faces = detector.detectMultiScale(gray_image, 1.3, 5)
        if len(faces) == 0:
            return jsonify({"status": "error", "match": False, "message": "No face found in image"}), 400

        (x, y, w, h) = faces[0]
        face_roi = gray_image[y:y+h, x:x+w]

        # 3. Nhận diện
        # recognizer.predict trả về (id, confidence)
        # Confidence (độ tin cậy) của OpenCV càng THẤP thì càng GIỐNG
        id_predicted, confidence = recognizer.predict(face_roi)

        print(f"Verify request for UID: {target_uid} (ID: {target_id}).")
        print(f"OpenCV predicted ID: {id_predicted} with confidence: {confidence}")

        # 4. So sánh
        # Đặt ngưỡng tin cậy (ví dụ: dưới 70 là khớp)
        if id_predicted == target_id and confidence < 70:
            return jsonify({"status": "success", "match": True, "confidence": confidence})
        else:
            return jsonify({"status": "error", "match": False, "message": "Face does not match UID", "confidence": confidence})

    except Exception as e:
        print(f"Verify error: {e}")
        return jsonify({"status": "error", "match": False, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
