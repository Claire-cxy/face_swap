import numpy as np
import os
import cv2
import insightface
from insightface.app import FaceAnalysis

# 当前工程目录
base_path = os.getcwd()
app = FaceAnalysis(name='buffalo_l', root=base_path)
app.prepare(ctx_id=0, det_size=(640, 640))

# 人脸转换模型加载
modelPath = os.path.join(base_path, 'models', "inswapper_128.onnx")
swapper = insightface.model_zoo.get_model(modelPath, root=base_path)


def get_max_face(app, img):
    faces = app.get(img)
    if faces is False or len(faces) < 1:
        return False

    # 遍历图像中所有人脸,获取人脸的字典信息,找到图像中面积最大的人脸
    areas = []
    for face in faces:
        bbox = face['bbox']
        area = abs((bbox[0] - bbox[2]) * (bbox[1] - bbox[3]))
        areas.append(area)
    index = np.argmax(areas)
    return faces[index]


def face_swap(img_src, img_tgt):
    # 获取源人脸信息,从图像中读取人脸相关信息
    face_src = get_max_face(app, img_src)
    # 获取目标人脸信息
    face_tgt = get_max_face(app, img_tgt)

    if face_tgt is None or face_tgt is False or face_src is None or face_src is False:
        print("-----no face--------------")
        return False

    # 人脸转换
    return swapper.get(img=img_tgt.copy(), target_face=face_tgt, source_face=face_src, paste_back=True)


if __name__ == '__main__':
    img_src = cv2.imread("./images/target_img.jpeg")
    img_tgt = cv2.imread("./images/sourceImg.jpg")
    res = face_swap(img_src, img_tgt)
    cv2.imwrite("images/out.jpg", res)
    cv2.imshow("images/out", res)
    cv2.waitKey(0)
