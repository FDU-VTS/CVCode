import cv2
import numpy as np
import PIL.Image as Image
import threading
import os, glob
import matplotlib.pyplot as plt
from matplotlib import cm


def image_joint(image_list, opt):  # opt= vertical ,horizon
    image_num = len(image_list)
    image_size = image_list[0].size
    height = image_size[1]
    width = image_size[0]

    if opt == 'vertical':
        new_img = Image.new('RGB', (width, image_num * height), 255)
    else:
        new_img = Image.new('RGB', (image_num * width, height), 255)
    x = y = 0
    count = 0
    for img in image_list:

        new_img.paste(img, (x, y))
        count += 1
        if opt == 'horizontal':
            x += width
        else:
            y += height
    return new_img


def viz_flow(flow):
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
    # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
    hsv[...,2] = 255
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    return bgr


def get_flow(prvs, next):
    #get GRAY  return BGR
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    result = viz_flow(flow)

    return result


def extract_flow_thread(img_path, video_list, start, end):
    for idx in range(start, end):
        video = cv2.VideoCapture(video_list[idx])
        if video.isOpened():
            for _ in range(49):
                _, _ = video.read()

            rval, pre = video.read()
            # print(type(pre))
            for _ in range(49):
                _, _ = video.read()
            rval, next = video.read()

            pre = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
            next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
            flow = get_flow(pre, next)
            np.save(img_path+'flow/'+video_list[idx].split('/')[-1].replace('.avi', ''), flow)

        else:
            # rval = False
            print("wrong!")
            return 1

        video.release()


def extract_flow(img_path, video_path):
    #use multi-thread
    video_list = glob.glob(video_path + '*.avi')
    if not os.path.exists(img_path+'flow/'):
        os.mkdir(img_path+'flow/')
    threads = []
    t1 = threading.Thread(target=extract_flow_thread, args=(img_path, video_list, 0, 500))
    threads.append(t1)
    t2 = threading.Thread(target=extract_flow_thread, args=(img_path, video_list, 500, 1000))
    threads.append(t2)
    t3 = threading.Thread(target=extract_flow_thread, args=(img_path, video_list, 1000, 1500))
    threads.append(t3)
    t4 = threading.Thread(target=extract_flow_thread, args=(img_path, video_list, 1500, 2000))
    threads.append(t4)
    t5 = threading.Thread(target=extract_flow_thread, args=(img_path, video_list, 2000, 2500))
    threads.append(t5)
    t6 = threading.Thread(target=extract_flow_thread, args=(img_path, video_list, 2500, 3000))
    threads.append(t6)
    t7 = threading.Thread(target=extract_flow_thread, args=(img_path, video_list, 3000, len(video_list)))
    threads.append(t7)

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("finish!")


def extract_test_flow_thread(img_path, video_path, stride):
    video = cv2.VideoCapture(video_path)
    flow_stride = 0
    if video.isOpened():
        pre = None
        name = None

        for idx in range(50+stride*119+49):
            _, frame = video.read()
            if (idx-49)%stride==0:
                name = img_path + 'flow/' + video_path.split('/')[-1].replace('delogo1.avi', '') + str(idx + 1).zfill(6)
                pre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow_stride = 1
            if flow_stride == 50:
                flow_stride = 0
                next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                np.save(name, get_flow(pre, next))
            elif flow_stride != 0:
                flow_stride += 1

    else:
        # rval = False
        print("wrong!")
        return 1

    video.release()


def extract_test_flow(img_path, video_path):
    #use multi-thread

    if not os.path.exists(img_path+'flow/'):
        os.mkdir(img_path+'flow/')
    threads = []
    t1 = threading.Thread(target=extract_test_flow_thread, args=(img_path, video_path+'104207_1-04-S20100821071000000E20100821120000000_delogo1.avi', 1500))
    threads.append(t1)
    t2 = threading.Thread(target=extract_test_flow_thread, args=(img_path, video_path+'200608_C08-02-S20100626083000000E20100626233000000_clip1_delogo1.avi', 1500))
    threads.append(t2)
    t3 = threading.Thread(target=extract_test_flow_thread, args=(img_path, video_path+'200702_C09-01-S20100717083000000E20100717233000000_delogo1.avi', 1500))
    threads.append(t3)
    t4 = threading.Thread(target=extract_test_flow_thread, args=(img_path, video_path+'202201_1-01-S20100922060000000E20100922235959000_clip1_delogo1.avi', 900))
    threads.append(t4)
    t5 = threading.Thread(target=extract_test_flow_thread, args=(img_path, video_path+'500717_D11-03-S20100717083000000E20100717233000000_delogo1.avi', 1500))
    threads.append(t5)

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("finish!")


if __name__ == '__main__':
    img_path = '../../data/world_expo/test_frame/'
    video_path = '../../data/world_expo/test_video/'
    extract_test_flow(img_path, video_path)
