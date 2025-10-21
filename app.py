from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

# "Cơ sở dữ liệu" tạm thời để lưu khuôn mặt (sẽ mất khi server restart)
# Trong thực tế nên dùng database, nhưng với free tier của Render, 
# server sẽ restart và mất dữ liệu. Đây là nhược điểm của gói miễn phí.
known_face_encodings = []
known_face_uids = []

@app.route('/')
def home():
    return "Face Recognition API is running."

# API 1: Thêm/Enroll Giảng viên
@app.route('/enroll', methods=['POST'])
def enroll_face():
    data = request.json
    if 'uid' not in data or 'image_base64' not in data:
        return jsonify({"status": "error", "message": "Missing uid or image"}), 400

    try:
        image_data = base64.b64decode(data['image_base64'])
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)

        face_encodings = face_recognition.face_encodings(image_np)
        if not face_encodings:
            return jsonify({"status": "error", "message": "No face found"}), 400

        face_encoding = face_encodings[0]

        # Kiểm tra xem UID đã tồn tại chưa
        if data['uid'] in known_face_uids:
            # Nếu đã tồn tại, cập nhật encoding
            idx = known_face_uids.index(data['uid'])
            known_face_encodings[idx] = face_encoding
            message = "Face updated"
        else:
            # Nếu chưa, thêm mới
            known_face_encodings.append(face_encoding)
            known_face_uids.append(data['uid'])
            message = "Face enrolled"

        print(f"Enrolled/Updated face with UID: {data['uid']}")
        return jsonify({"status": "success", "message": message}), 201

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

        # 1. Tìm encoding đã lưu của UID này
        known_encoding = None
        try:
            target_index = known_face_uids.index(target_uid)
            known_encoding = known_face_encodings[target_index]
        except ValueError:
            return jsonify({"status": "error", "match": False, "message": "UID not enrolled"}), 404

        # 2. Giải mã ảnh mới
        image_data = base64.b64decode(data['image_base64'])
        image = Image.open(io.BytesIO(image_data))
        unknown_image_np = np.array(image)

        unknown_encodings = face_recognition.face_encodings(unknown_image_np)
        if not unknown_encodings:
            return jsonify({"status": "error", "match": False, "message": "No face found in image"}), 400

        unknown_encoding = unknown_encodings[0] 

        # 3. So sánh 1-với-1
        is_match = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.5)[0]

        if is_match:
            return jsonify({"status": "success", "match": True})
        else:
            return jsonify({"status": "error", "match": False, "message": "Face does not match UID"})

    except Exception as e:
        print(f"Verify error: {e}")
        return jsonify({"status": "error", "match": False, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
