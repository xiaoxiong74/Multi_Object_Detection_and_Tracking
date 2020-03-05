#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from keras.layers import Input
from yolo3.utils import letterbox_image
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "./vedio/final-0113-v2.mp4")
ap.add_argument("-c", "--class",help="name of class", default = "person")
args = vars(ap.parse_args())

class YOLO(object):
    def __init__(self):
        self.model_path = '../keras-yolo3/logs/000/trained_weights.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/our_classes.txt'
        self.score = 0.25  # 0.8
        self.iou = 0.5
        self.model_image_size = (416, 416)
        # #具体参数可实验后进行调整
        # if args["class"] == 'person':
        #    self.score = 0.6 #0.8
        #    self.iou = 0.6
        #    self.model_image_size = (416,416)
        # if args["class"] == 'car':
        #    self.score = 0.6
        #    self.iou = 0.6
        #    self.model_image_size = (416, 416)
        # if args["class"] == 'bicycle' or args["class"] == 'motorcycle':
        #    self.score = 0.6
        #    self.iou = 0.6
        #    self.model_image_size = (416, 416)
        # if args["class"] == 'fire_extinguisher' or args["class"] == 'fireplug':
        #    self.score = 0.4#0.4
        #    self.iou = 0.6
        #    self.model_image_size = (416, 416)
        # if args["class"] == 'cup' or args["class"] == 'mouse':
        #    self.score = 0.6
        #    self.iou = 0.6

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        #self.model_image_size = (416, 416) # fixed size or (None, None) small targets:(320,320) mid targets:(960,960)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        #print(class_names)
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        # assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        # is_tiny_version = num_anchors==6 # default setting
        is_tiny_version = False
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))


        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return_boxs = []
        return_class_name = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            # 对某些类别可以做一些阈值的返回限制
            if predicted_class == 'beer_package' and out_scores[i] < 0.95:
                continue
            print(predicted_class, out_scores[i])
            box = out_boxes[i]
            #score = out_scores[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3]-box[1])
            h = int(box[2]-box[0])
            if x < 0 :
                w = w + x
                x = 0
            if y < 0 :
                h = h + y
                y = 0
            return_boxs.append([x,y,w,h])
            #print(return_boxs)
            return_class_name.append([predicted_class])
        #cv2.putText(image, str(self.class_names[c]),(int(box[0]), int(box[1] -50)),0, 5e-3 * 150, (0,255,0),2)
        #print("Found person: ",person_counter)
        return return_boxs, return_class_name

    def close_session(self):
        self.sess.close()
