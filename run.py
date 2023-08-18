import streamlit as st
import numpy as np
import os
from insightface.app import FaceAnalysis
import insightface
from PIL import Image
import cv2
from datetime import datetime
import dlib
from face_swap_dlib import dlib_face_swap

# 全局变量存储:人脸变换的源,目标图像,人脸修复的源图像
default_img = os.path.join('images', "default.jpg")
base_path = os.getcwd()
file_gfpgan = os.path.join("gfpgan", "weights", "GFPGANv1.4.pth")
print(default_img)
print(file_gfpgan)

if "src_img" not in st.session_state:
    st.session_state.src_img = default_img
if "tgt_img" not in st.session_state:
    st.session_state.tgt_img = default_img
if "org_face_img" not in st.session_state:
    st.session_state.org_face_img = default_img

    # 创建人脸检测器
    det_face = dlib.get_frontal_face_detector()

    # 加载标志点检测器
    det_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")  # 68点

    # 打开图片
    img_dst = cv2.imread('./images/sourceImg.jpg')
    img_src = cv2.imread('./images/targetImg.jpg')


# 装饰,只在页面刷新时启动一次
@st.cache_resource
def model_load():
    # 人脸转换模型
    app = FaceAnalysis(name='buffalo_l', root=base_path)
    app.prepare(ctx_id=0, det_size=(640, 640))
    name = os.path.join(base_path, 'models', "inswapper_128.onnx")
    swapper = insightface.model_zoo.get_model(name, root=base_path)

    # 人脸修补模型
    from gfpgan import GFPGANer
    gfpgan_constructor = GFPGANer
    model_gfpgan = gfpgan_constructor(upscale=1,
                                      arch='clean',
                                      channel_multiplier=2,
                                      model_path=file_gfpgan,
                                      bg_upsampler=None)
    return app, swapper, model_gfpgan


# 读取文件,返回opencv格式图像
def pillow2cv(file):
    img = Image.open(file)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_cv


def get_max_face(app, img):
    faces = app.get(img)
    if len(faces) < 1:
        return False

    areas = []
    for face in faces:
        bbox = face['bbox']
        area = abs((bbox[0] - bbox[2]) * (bbox[1] - bbox[3]))
        areas.append(area)
    index = np.argmax(areas)
    return faces[index]


def face_swapper(res_info):
    if st.session_state.src_img == default_img or st.session_state.tgt_img == default_img:
        res_info.write("加载正确的图像")
        return False

    # img_src = Image.open(st.session_state.src_img)

    img_src = pillow2cv(st.session_state.src_img)

    # cv2.imread(st.session_state.src_img)
    face_src = get_max_face(m_app, img_src)
    if len(face_src) < 0:
        res_info.write("源图像没有检测到人脸")
        return False

    # img_tgt = cv2.imread(st.session_state.tgt_img)
    img_tgt = pillow2cv(st.session_state.tgt_img)
    print(img_tgt)
    face_tgt = get_max_face(m_app, img_tgt)

    print(face_tgt)
    if face_tgt is False:
        return False
    if len(face_tgt) < 0:
        res_info.write("目标图像没有检测到人脸")
        return False

    try:
        res = img_tgt.copy()
        res = m_swapper.get(res, face_tgt, face_src, paste_back=True)
        return res
    except Exception as e:
        res_info.write(e)
        return False


# 自定义函数修复人脸
def gfpgan_fix_faces_model(pil_img):
    np_image = np.array(pil_img)
    np_image_bgr = np_image[:, :, ::-1]
    cropped_faces, restored_faces, gfpgan_output_bgr = model_gfpgan.enhance(np_image_bgr, has_aligned=False,
                                                                            only_center_face=False, paste_back=True)
    np_image = gfpgan_output_bgr[:, :, ::-1]
    return Image.fromarray(np_image, 'RGB')


def get_max_face(app, img):
    faces = app.get(img)
    if len(faces) < 1:
        return False

    areas = []
    for face in faces:
        bbox = face['bbox']
        area = abs((bbox[0] - bbox[2]) * (bbox[1] - bbox[3]))
        areas.append(area)
    index = np.argmax(areas)
    return faces[index]


m_app, m_swapper, model_gfpgan = model_load()

# 主界面title
st.title("人脸转换演示")

tab1, tab2, tab3 = st.tabs(['insightface人脸转换', "gfpgan人脸修复", "dlib人脸转换"])
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        src_uploaded_file = st.file_uploader("源", type=["jpg", "jpeg", "png"])
        if src_uploaded_file is not None:
            st.session_state.src_img = src_uploaded_file
            image = Image.open(st.session_state.src_img)
            # 显示源图像
            st.image(image, caption='源图像', use_column_width=True)

    with col2:
        tgt_uploaded_file = st.file_uploader("目标", type=["jpg", "jpeg", "png"])
        if tgt_uploaded_file is not None:
            st.session_state.tgt_img = tgt_uploaded_file
            image = Image.open(st.session_state.tgt_img)
            st.image(image, caption='目标', use_column_width=True)

    flag = st.button("转换")
    res_img = st.empty()
    res_info = st.empty()
    if flag:
        res = face_swapper(res_info)
        if res is not False and not res.all == False:
            img_show = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            pillow_img_show = Image.fromarray(img_show, 'RGB')
            res_img.image(pillow_img_show)
            # 图像保存
            dt = datetime.now()
            str_time = dt.strftime("%Y-%m-%d-%H-%M-%S")
            save_name = os.path.join(base_path, 'images', str_time + "-swapper.png")
            pillow_img_show.save(save_name)
with tab2:
    col1, col2 = st.columns(2)
    with col1:

        uploaded_file = st.file_uploader("待修复人脸", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state.org_face_img = uploaded_file
            image = Image.open(st.session_state.org_face_img)
            st.image(image, caption='待修复人脸', use_column_width=True)
        else:
            st.session_state.org_face_img = default_img

    with col2:
        flag = st.button("人脸修复")
        # 按钮对齐
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        # 空白控件显示图像
        fixed_img_show = st.empty()
        if flag:
            if not st.session_state.org_face_img == default_img:
                pil_img = Image.open(st.session_state.org_face_img)
                fixed_img = gfpgan_fix_faces_model(pil_img)
                fixed_img_show.image(fixed_img, caption='人脸修复后', use_column_width=True)
                # 图像保存
                dt = datetime.now()
                str_time = dt.strftime("%Y-%m-%d-%H-%M-%S")
                save_name = os.path.join(base_path, 'images', str_time + "-face_fix.png")
                fixed_img.save(save_name)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        src_uploaded_file = st.file_uploader(label="源", type=["jpg", "jpeg", "png"], key="3-1")
        if src_uploaded_file is not None:
            st.session_state.src_img = src_uploaded_file
            image = Image.open(st.session_state.src_img)
            # 显示源图像
            st.image(image, caption='源图像', use_column_width=True)

    with col2:
        tgt_uploaded_file = st.file_uploader(label="目标", type=["jpg", "jpeg", "png"], key="3-2")
        if tgt_uploaded_file is not None:
            st.session_state.tgt_img = tgt_uploaded_file
            image = Image.open(st.session_state.tgt_img)
            st.image(image, caption='目标', use_column_width=True)

    flag = st.button("转换", key="3-3")
    res_img = st.empty()
    res_info = st.empty()
    if flag:
        img_src = cv2.imread("./images/JayChou.png")
        img_tgt = cv2.imread("./images/targetImg.jpg")
        res = dlib_face_swap(img_src, img_tgt)
        if not res.all == False:
            img_show = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            pillow_img_show = Image.fromarray(img_show, 'RGB')
            res_img.image(pillow_img_show)
            # 图像保存
            dt = datetime.now()
            str_time = dt.strftime("%Y-%m-%d-%H-%M-%S")
            save_name = os.path.join(base_path, 'images', str_time + "--swapper.png")
            pillow_img_show.save(save_name)
