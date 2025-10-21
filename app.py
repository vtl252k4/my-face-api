from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
import urllib.request

app = Flask(__name__)

# === 1. CẤU HÌNH CLOUDINARY ===
# (Chúng ta sẽ KHÔNG hardcode Key ở đây, mà dùng Environment Variables)
cloudinary.config(
    cloud_name = os.environ.get('ddjx4cvgr'),
    api_key = os.environ.get('972296858383671'),
    api_secret = os.environ.get('BhrgCrWWrQfSXOtMg5xv0YH0nZM')
)
print("Cloudinary Configured.")

# === 2. CẤU HÌNH OPENCV (Như cũ) ===
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Đường dẫn file model trong bộ nhớ tạm của Render
FACE_DB_PATH = "/tmp/face_data.yml"
# Tên file model khi lưu trên Cloudinary
CLOUDINARY_FILE_NAME = "face_model/face_data" 

uid_to_id_map = {}
id_to_uid_map = {}
next_id = 1

# === 3. HÀM TẢI DATABASE TỪ CLOUDINARY (NÂNG CẤP) ===
def load_database():
    global uid_to_id_map, id_to_uid_map, next_id

    # Tạo thư mục /tmp nếu chưa có
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")

    try:
        print("Dang tim database tren Cloudinary...")
        # Tìm file trên Cloudinary bằng public_id
        file_info = cloudinary.api.resource(CLOUDINARY_FILE_NAME, resource_type="raw")

        # Lấy URL an toàn của file
        file_url = file_info['secure_url']
        print(f"Tim thay database, dang tai tu: {file_url}")

        # Tải file từ URL về đường dẫn /tmp/face_data.yml
        with urllib.request.urlopen(file_url) as response, open(FACE_DB_PATH, 'wb') as out_file:
            data = response.read()
            out_file.write(data)

        print("Tai database thanh cong.")

        # Đọc file model đã tải về
        recognizer.read(FACE_DB_PATH)

        # (Phần này cần cải tiến: bạn nên lưu map này cùng file YML,
        # nhưng hiện tại chúng ta sẽ tạm chấp nhận nó reset mỗi khi ngủ)
        # Giả sử chúng ta cần load lại map
        # Tạm thời bỏ qua việc load map, nó sẽ tự tạo map mới khi enroll
        print("Database da nap vao recognizer.")

    except Exception as e:
        # Lỗi này (404) sẽ xảy ra ở lần đầu tiên chạy (chưa có file)
        if "Not found" in str(e):
            print("Khong tim thay database (co the la lan dau chay). Khoi tao model moi.")
        else:
            print(f"Loi khi tai database: {e}")

# === 4. HÀM ENROLL (NÂNG CẤP) ===
@app.route('/enroll', methods=['POST'])
def enroll_face():
    global next_id
    data = request.json
    if 'uid' not in data or 'image_base64' not in data:
        return jsonify({"status": "error", "message": "Missing uid or image"}), 400

    try:
        target_uid = data['uid']
        gray_image = b64_to_cv_image(data['image_base64'])

        faces = detector.detectMultiScale(gray_image, 1.3, 5)
        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face found"}), 400

        (x, y, w, h) = faces[0]
        face_roi = gray_image[y:y+h, x:x+w]

        if target_uid not in uid_to_id_map:
            uid_to_id_map[target_uid] = next_id
            id_to_uid_map[next_id] = target_uid
            current_id = next_id
            next_id += 1
        else:
            current_id = uid_to_id_map[target_uid]

        # "Train" (cập nhật) model trong bộ nhớ
        recognizer.update([face_roi], np.array([current_id]))

        # Lưu CSDL ra file /tmp
        recognizer.write(FACE_DB_PATH)

        print(f"Da train cho UID: {target_uid}. Dang upload len Cloudinary...")

        # (MỚI) Upload file model lên Cloudinary
        cloudinary.uploader.upload(
            FACE_DB_PATH,
            public_id = CLOUDINARY_FILE_NAME,
            resource_type = "raw", # Quan trọng: báo đây là file thô, không phải ảnh
            overwrite=True # Ghi đè file cũ
        )

        print("Upload len Cloudinary thanh cong.")
        return jsonify({"status": "success", "message": "Face enrolled and saved"}), 201

    except Exception as e:
        print(f"Enroll error: {e}")
        return jsonify({"status": "error", "message": "Could not process image"}), 500

# === 5. HÀM VERIFY (Giữ nguyên) ===
@app.route('/verify', methods=['POST'])
def verify_face():
    # (Code hàm verify giữ nguyên 100% như cũ)
    data = request.json
    if 'uid' not in data or 'image_base64' not in data:
        return jsonify({"status": "error", "message": "Missing uid or image"}), 400
    try:
        target_uid = data['uid']
        if target_uid not in uid_to_id_map:
             # (Tạm thời) Nếu server vừa ngủ dậy và map bị reset
             # Ta yêu cầu enroll lại
             if not id_to_uid_map:
                 print("Server vua khoi dong lai, yeu cau enroll.")
                 # Tải lại CSDL
                 load_database()
                 # Nếu vẫn ko có, yêu cầu enroll
                 if target_uid not in uid_to_id_map:
                     return jsonify({"status": "error", "match": False, "message": "Server restart, please re-enroll"})
             else:
                 return jsonify({"status": "error", "match": False, "message": "UID not enrolled"}), 404

        target_id = uid_to_id_map[target_uid]
        gray_image = b64_to_cv_image(data['image_base64'])
        faces = detector.detectMultiScale(gray_image, 1.3, 5)
        if len(faces) == 0:
            return jsonify({"status": "error", "match": False, "message": "No face found in image"}), 400
        (x, y, w, h) = faces[0]
        face_roi = gray_image[y:y+h, x:x+w]
        id_predicted, confidence = recognizer.predict(face_roi)
        print(f"Verify request for UID: {target_uid} (ID: {target_id}).")
        print(f"OpenCV predicted ID: {id_predicted} with confidence: {confidence}")
        if id_predicted == target_id and confidence < 70:
            return jsonify({"status": "success", "match": True, "confidence": confidence})
        else:
            return jsonify({"status": "error", "match": False, "message": "Face does not match UID", "confidence": confidence})
    except Exception as e:
        print(f"Verify error: {e}")
        return jsonify({"status": "error", "match": False, "message": str(e)}), 500

# === 6. HÀM TRỢ GIÚP (Giữ nguyên) ===
def b64_to_cv_image(b64_string):
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    image_data = base64.b64decode(b64_string)
    image = Image.open(io.BytesIO(image_data))
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    return gray

# === 7. KHỞI ĐỘNG SERVER ===
if __name__ == '__main__':
    print("--- KHOI DONG SERVER ---")
    load_database() # (MỚI) Tải database ngay khi khởi động
    app.run(host='0.0.0.0', port=5000)
