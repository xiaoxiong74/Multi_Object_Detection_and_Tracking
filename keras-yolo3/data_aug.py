# -*- coding: utf-8 -*-
"""
# @Time    : 2019/12/27 15:48
# @Author  : xiaoxiong
# @Email   : xyf_0704@sina.com
# @File    : data_aug.py
# @Software: PyCharm
# DESC :
"""
from keras.preprocessing.image import ImageDataGenerator

path = 'VOCdevkit' # 类别子文件夹的上一级

dst_path = 'E:/C3D_Data/train_result'

# 图片生成器

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.02,
    height_shift_range=0.,
    shear_range=0.,
    horizontal_flip=True,
    vertical_flip=True
)

gen = datagen.flow_from_directory(
path,
target_size=(224, 224),
batch_size=2,
save_to_dir=dst_path,#生成后的图像保存路径
save_prefix='aug',
save_format='jpg')


for i in range(3):
    gen.next()
