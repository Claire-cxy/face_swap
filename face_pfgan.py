import os
from PIL import Image
import numpy as np


def gfpgan_load(file_path):
    from gfpgan import GFPGANer
    gfpgan_constructor = GFPGANer
    return gfpgan_constructor(
        upscale=1,
        arch='clean',
        bg_upsampler=None,
        channel_multiplier=2,
        model_path=file_path)


# 自定义函数修复人脸
def gfpgan_fix_faces_model(fix_model, pil_img):
    # pillow格式转换为opencv格式
    np_image = np.array(pil_img)
    np_image_bgr = np_image[:, :, ::-1]  # 通道反转
    _, _, gfpgan_output_bgr = fix_model.enhance(np_image_bgr,
                                                has_aligned=False,
                                                only_center_face=False,
                                                paste_back=True)
    # opencv格式重新转回pillow格式
    np_image = gfpgan_output_bgr[:, :, ::-1]
    return Image.fromarray(np_image, 'RGB')


gfpgan_file_path = os.path.join('gfpgan', 'weights', 'GFPGANv1.4.pth')
if __name__ == "__main__":
    model = gfpgan_load(gfpgan_file_path)
    img = Image.open("images/out.jpg")
    if img is None:
        print("failed to get picture")
        exit(1)

    fix_img = gfpgan_fix_faces_model(model, img)
    fix_img.save("./images/fix.png")
