#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend

backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to input video", default="./vedio/final-0113-v2.mp4")
ap.add_argument("-c", "--class", help="name of class", default="beer_package")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

dict_class = {"beer_package": "箱装啤酒",
              "beef": "肥牛片",
              "sauce": "酱油",
              "paper_toilet_bag": "厕纸",
              "paper_extract_box": "抽纸",
              "yoghurt": "酸奶",
              "detergent": "洗衣液(瓶)",
              "water_package": "箱装矿泉水"
              }


def paint_chinese_opencv(im, chinese, pos, color):
    """
    显示中文字体
    :param im:
    :param chinese:
    :param pos:
    :param color:
    :return:
    """
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('font/simhei.ttf', size=20, encoding="utf-8")
    fillColor = color  # (255,0,0)
    position = pos  # (100,100)
    # chinese = chinese.encode('utf-8').decode('utf-8')
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=fillColor)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


def main(yolo):

    start = time.time()
    # Definition of the parameters
    max_cosine_distance = 0.5 #0.9 余弦距离的控制阈值
    nn_budget = None
    nms_max_overlap = 0.3 #非极大抑制的阈值

    # 定义类别个数的列表，用于保存各个类别的跟踪id
    counter1, counter2, counter3, counter4, counter5, counter6, counter7, counter8 = [], [], [], [], [], [], [], []

    # 初始化各类别的全局统计量
    count1, count2, count3, count4, count5, count6, count7, count8 = 0, 0, 0, 0, 0, 0, 0, 0

    # deep_sort 复用之前的模型，也可以自己训练(工作量大)
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    video_capture = cv2.VideoCapture(args["input"])

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./output/output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        # image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1]) # bgr to rgb
        boxs,class_names = yolo.detect_image(image)
        features = encoder(frame, boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        # 初始化每帧各类别统计变量
        i1, i2, i3, i4, i5, i6, i7, i8 = 0, 0, 0, 0, 0, 0, 0, 0
        indexIDs = []
        c = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

        for track, class_name in zip(tracker.tracks, class_names):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            print("relal class:" + dict_class[str(class_name[0])])
            # 分别保存每个类别的track_id, 并统计当前画面中的每个类别单独计数
            if class_name == ['paper_toilet_bag']:
                counter1.append(int(track.track_id))
                i1 = i1 + 1
            if class_name == ['beef']:
                counter2.append(int(track.track_id))
                i2 = i2 + 1
            if class_name == ['yoghurt']:
                counter3.append(int(track.track_id))
                i3 = i3 + 1
            if class_name == ['paper_extract_box']:
                counter4.append(int(track.track_id))
                i4 = i4 + 1
            if class_name == ['beer_package']:
                counter5.append(int(track.track_id))
                i5 = i5 + 1
            if class_name == ['detergent']:
                counter6.append(int(track.track_id))
                i6 = i6 + 1
            if class_name == ['sauce']:
                counter7.append(int(track.track_id))
                i7 = i7 + 1
            if class_name == ['water_package']:
                counter8.append(int(track.track_id))
                i8 = i8 + 1
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 3)
            # cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1] - 50)), 0, 5e-3 * 150, color, 2)
            # if len(class_names) > 0:
            #    class_name = class_names[0]
            #    cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)
            # 显示中文类别
            frame = paint_chinese_opencv(frame, dict_class[str(class_name[0])], (int(bbox[0]), int(bbox[1] - 20)), tuple(color))
            # cv2.putText(frame, str(class_name), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, color, 2)

            # bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
            # track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            # center point
            # cv2.circle(frame,  (center), 1, color, thickness)

            # draw motion path 画出移动路径
            # for j in range(1, len(pts[track.track_id])):
            #     if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
            #        continue
            #     thickness = int(np.sqrt(64 / float(j + 1)) * 2)
            #     cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                # cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        # 统计每类物品的总数
        count1, count2, count3 = len(set(counter1)), len(set(counter2)), len(set(counter3))
        count4, count5, count6 = len(set(counter4)), len(set(counter5)), len(set(counter6))
        count7, count8 = len(set(counter7)), len(set(counter8))
        if count1 != 0:
            frame = paint_chinese_opencv(frame, "厕纸  (当前/总计): "+str(i1) + str('/') + str(count1), (int(20), int(40)), (0, 255, 0))
        if count2 != 0:
            frame = paint_chinese_opencv(frame, "肥牛片  (当前/总计): "+str(i2) + str('/') + str(count2), (int(20), int(60)), (0, 255, 0))
        if count3 != 0:
            frame = paint_chinese_opencv(frame, "酸奶  (当前/总计): "+str(i3) + str('/') + str(count3), (int(20), int(80)), (0, 255, 0))
        if count4 != 0:
            frame = paint_chinese_opencv(frame, "抽纸   (当前/总计): "+str(i4) + str('/') + str(count4), (int(20), int(100)), (0, 255, 0))
        if count5 != 0:
            frame = paint_chinese_opencv(frame, "箱装啤酒   (当前/总计): "+str(i5) + str('/') + str(count5), (int(20), int(120)), (0, 255, 0))
        if count6 != 0:
            frame = paint_chinese_opencv(frame, "洗衣液(瓶) (当前/总计): "+str(i6) + str('/') + str(count6), (int(20), int(140)), (0, 255, 0))
        if count7 != 0:
            frame = paint_chinese_opencv(frame, "酱油      (当前/总计): "+str(i7) + str('/') + str(count7), (int(20), int(160)), (0, 255, 0))
        if count8 != 0:
            frame = paint_chinese_opencv(frame, "箱装矿泉水 (当前/总计): "+str(i8) + str('/') + str(count8), (int(20), int(180)), (0, 255, 0))

        cv2.putText(frame, "FPS: %f"%(fps), (int(20), int(20)), 0, 5e-3 * 100, (0, 255, 0), 3)
        # cv2.namedWindow("YOLO3_Deep_SORT", 0);
        # cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768);
        # cv2.imshow('YOLO3_Deep_SORT', frame)

        if writeVideo_flag:
            #save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps = (fps + (1./(time.time()-t1))) / 2
        # print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    # 打印各类最终的统计数量
    if len(tracker.tracks) != None:
        desc_str = "input vedio detect: "
        print(desc_str + str(count1) + " " + '厕纸')
        print(desc_str + str(count2) + " " + '肥牛片')
        print(desc_str + str(count3) + " " + '酸奶')
        print(desc_str + str(count4) + " " + '抽纸')
        print(desc_str + str(count5) + " " + '箱装啤酒')
        print(desc_str + str(count6) + " " + '洗衣液')
        print(desc_str + str(count7) + " " + '酱油')
        print(desc_str + str(count8) + " " + '箱装矿泉水')

    else:
        print("[No Found]")

    video_capture.release()

    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
