import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import wandb

from glob import glob
import json
import os
from tqdm.notebook import tqdm
import time
import base64
import typing as t
import zlib
import gc

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics import AveragePrecision
import torchvision
from torchvision.transforms import transforms
from utils import fn_time, find_files, load_img, get_json_dataframe, poly_mask
from pycococreatortools import pycococreatortools
import pathlib
from tqdm import tqdm
import datetime

import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings('ignore')
from os.path import join as opj
from pycocotools.coco import COCO

def convert_tif2jpg():
    """
    将 tif 文件转换为 jpg
    tile 中数据集1为训练集, 2 为测试； tile 7033 数据集1:422 数据集2:1211 数据集3:5400
    """
    tile = pd.read_csv('/root/autodl-tmp/kaggle/datasets/original/tile_meta.csv')
    basenames = tile.query("dataset == 1").id.values

    n = len(basenames)
    train_n = n*0.8
    test_n = n*0.2
    test_basenames = basenames[np.random.choice(range(422), int(test_n))]
    train_basenames = list(set(basenames) - set(test_basenames))

    # files = find_files(file_dir, ext='tif')
    for basename in tqdm(test_basenames):
        # basename = pathlib.Path(file).stem
        file = "/root/autodl-tmp/kaggle/datasets/original/train/{}.tif".format(basename)
        if os.path.exists(file):
            img = cv2.imread(file, -1)  
            cv2.imwrite("/root/autodl-tmp/kaggle/datasets/coco_format/images/test/{}.jpg".format(basename),img)
        else:
            raise "file doesn't exist! file".format(file)
        
    for basename in tqdm(train_basenames):
        # basename = pathlib.Path(file).stem
        file = "/root/autodl-tmp/kaggle/datasets/original/train/{}.tif".format(basename)
        if os.path.exists(file):
            img = cv2.imread(file, -1)  
            cv2.imwrite("/root/autodl-tmp/kaggle/datasets/coco_format/images/train/{}.jpg".format(basename),img)
        else:
            raise "file doesn't exist! file".format(file)

def convert_2_inst(flag='train'):
    """
    tile 中数据集1为训练集, 2 为测试； tile 7033 数据集1:422 数据集2:1211 数据集3:5400
    转换为  inst_mask 格式.
    """
    file_dir = "/root/autodl-tmp/kaggle/datasets/coco_format/images/{}/".format(flag)
    files = find_files(file_dir, ext='jpg')
    polygons_df = get_json_dataframe("/root/autodl-tmp/kaggle/datasets/original/polygons.jsonl").set_index('id')

    for file in tqdm(files):
        basename = pathlib.Path(file).stem
        if os.path.exists(file):
            masks = []
            ann = polygons_df.loc[basename, :].annotations  # ann.id, ann['annotations']
            for i in range(len(ann)):
                inst = ann[i]
                if inst['type'] == 'blood_vessel':
                    polygon = [tuple(x) for x in ann[i]['coordinates'][0]]
                    binary_mask = poly_mask((512, 512), polygon, value=0)
                    masks.append(binary_mask)
        
            save_path = "/root/autodl-tmp/kaggle/datasets/coco_format/inst_mask/{}/{}.npy".format(flag, basename)
            np.save(save_path, np.array(masks))
        else:
            raise "file doesn't exist! file".format(file)
    # return masks
    
def convert_to_coco(flag = 'train'):
    """将 HuBMAP 数据转换为 coco format
        目前只处理 一种类别, unsure 和 glomerulus 的标记直接忽略
        flag: 训练集或测试集

        返回是直接将 生成的 instances 结果写入 json 文件中
    """
    process_dir = "/root/autodl-tmp/kaggle/datasets/coco_format/images/{}/".format(flag)
    polygons_df = get_json_dataframe("/root/autodl-tmp/kaggle/datasets/original/polygons.jsonl").set_index('id')
        
    # 实验没有 background 的编码
    CATEGORIES = [{
        'id': 1,
        'name': 'blood_vessel',
        'supercategory': 'hubmap',
    }
    ]

    INFO = {
    "description": "Dataset in coco format",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2023,
    "contributor": "weifeiouyang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]
    coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }

    
    image_id = 1
    segmentation_id = 1

    images_files = find_files(process_dir, ext='jpg')
    class_id = 1
    category_info = {'id': int(class_id), 'is_crowd': 0}

    for image_filename in images_files:
        image = load_img(image_filename)
        image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.shape[:2])
        coco_output["images"].append(image_info)

        # 对每个image提取对应的inst
        basename = pathlib.Path(image_filename).stem

        ann = polygons_df.loc[basename, :].annotations  # ann.id, ann['annotations']

        for i in range(len(ann)):
            inst = ann[i]
            if inst['type'] == 'blood_vessel':
                polygon = [tuple(x) for x in ann[i]['coordinates'][0]]
        #             vertices.append(polygon)
                binary_mask = poly_mask((512, 512), polygon, value=0)
                annotation_info = pycococreatortools.create_annotation_info(
                          segmentation_id, image_id, category_info, binary_mask,
                          binary_mask.shape, tolerance=2)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
          
                segmentation_id += segmentation_id 
        image_id += 1

    save_path = "/root/autodl-tmp/kaggle/datasets/coco_format/annotations/instances_{}.json".format(flag)
    with open(save_path , 'w') as output_json_file:
        json.dump(coco_output, output_json_file) 

        
def validate_coco():
    """暂时这里不是说验证 coco, 而是对比 coco_format 之后的数据, 与直接的给的注释是否吻合。
    """
    pass


if __name__ == "__main__":
    # convert_tif2jpg("/root/autodl-tmp/kaggle/datasets/original/train", flag='train')
    # convert_tif2jpg("/root/autodl-tmp/kaggle/datasets/original/test", flag='test')
    # convert_tif2jpg()
    # convert_to_coco('test')
    convert_2_inst(flag='test')