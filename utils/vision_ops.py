import cv2
import numpy as np
import os


def extract_face(image_path, target_size=(112, 112)):
    """
    读取图像，使用 Haar 级联检测人脸并裁剪，缩放至 target_size
    返回 shape: (3, 112, 112)
    """
    img = cv2.imread(image_path)
    if img is None: return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用 OpenCV 默认的人脸检测模型
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 如果检测到人脸，截取最大的一个；如果没有，则中心裁剪
    if len(faces) > 0:
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_img = img[y:y + h, x:x + w]
    else:
        h, w = img.shape[:2]
        min_dim = min(h, w)
        face_img = img[(h - min_dim) // 2:(h + min_dim) // 2, (w - min_dim) // 2:(w + min_dim) // 2]

    # 缩放并转为模型需要的 CHW 格式 (Channel, Height, Width)
    face_img = cv2.resize(face_img, target_size)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = np.transpose(face_img, (2, 0, 1))

    # 归一化到 [-1, 1]
    face_img = (face_img.astype(np.float32) - 127.5) / 128.0
    return face_img