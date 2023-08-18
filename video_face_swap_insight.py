import os

from face_swap_insight import face_swap
import cv2

# 打开视频文件
video_path = './video/target_video.MP4'
cap = cv2.VideoCapture(video_path)
img_src = cv2.imread("./images/target_img.jpeg")

# 检查是否成功打开视频文件
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# 设置视频的帧率和分辨率
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # width
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter("video/output_video.mp4", fourcc, frame_rate, (frame_width, frame_height), True)


# 循环读取帧并显示
i = 0
isOpened = cap.isOpened()
while isOpened:
    ok, frame = cap.read()

    # 检查是否成功读取帧
    if not ok:
        print("End of video")
        break
    res = face_swap(img_src, frame)  # 逐帧图片换脸
    if res is False:
        res = frame  # 无人脸,则按照原图写入

    # cv2.imwrite("video/pic/out_"+str(i)+".jpg", res)
    # cv2.waitKey(1)
    # cv2.imshow("video/pic/out_"+str(i), res)

    # 调整图片尺寸以适应视频分辨率
    # _img = cv2.imread("video/pic/out_"+str(i)+".jpg")
    output_video.write(res)
    i += 1
    print(str(i) + " processed!")


# filelist = os.listdir(path="video/pic")

# # 创建视频写入器
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output_video = cv2.VideoWriter("video/output_video.mp4", fourcc, frame_rate, (frame_width, frame_height), True)
#
# for item in filelist:
#     if item.endswith('.jpg'):
#         img = cv2.imread("./video/pic/"+item)
#         # cv2.imshow("./video/pic/"+item, img)
#         output_video.write(img)
#         # if cv2.waitKey(1) == ord('q'):
#         #     break

# 释放视频写入器
# 释放资源
output_video.release()
cap.release()
cv2.destroyAllWindows()
