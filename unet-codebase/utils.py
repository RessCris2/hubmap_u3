import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
from torch.utils.data import WeightedRandomSampler
import json
import cv2
import os
from sklearn.model_selection import StratifiedKFold

#get sampler
def get_sampler(cfg):
    df = pd.read_csv(cfg.val_df)
    df = add_weights_on_df(df)
    weights = torch.tensor(df['cancer_weight'].to_numpy(), dtype = torch.float)
    num_samples = df.shape[0]#可自定
    wrs = WeightedRandomSampler(df.weight.tolist(), num_samples=10000)
    ewrs = ExhaustiveWeightedRandomSampler(df.weight.tolist(), num_samples=10000)
    return wrs

def add_weights_on_df(df):
    pos_weight = 1158/53548
    neg_weight = 1 - pos_weight
    df['cancer_weight'] = 1#不管阴性阳性都先是1
    df.loc[df.cancer == 1, "cancer_weight"] = len(df.loc[df.cancer == 0]) / len(df.loc[df.cancer == 1])   
    return df

def get_train_dataloader(train_dataset, cfg):
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=None,
        shuffle=True,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
        drop_last=True,
    )

    return train_dataloader

def get_val_dataloader(val_dataset, cfg):

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
    )
    
    return val_dataloader

def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def generateparam(cfg):
    pass

def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out

def json2mask():
    jsonpath = "/workspace/HubMAP_2023/kaggle_origin/polygons.jsonl"
    root = "/workspace/HubMAP_2023/data/512x512/mask"
    with open(jsonpath, 'r') as json_file:
        json_labels = [json.loads(line) for line in json_file]
    
    for row in json_labels:
        mask = np.zeros((512, 512), dtype=np.int32)
        for annot in row['annotations']:
            cords = annot['coordinates']
            if annot['type'] == "blood_vessel":
                for cord in cords:
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    mask[rr, cc] = 1
        name = row['id']
        mask = mask*255
        contours,_ = cv2.findContours(mask.astype(np.uint8), 1, 2)
        zero_img = np.zeros([mask.shape[0], mask.shape[1], 3], dtype="uint8")
        for p in contours:
            cv2.fillPoly(zero_img, [p], (255, 255, 255))
        cv2.imwrite(root + f"/{name}.png", zero_img)
        print(f'save {name}')


def json2glomerulusmask():
    jsonpath = "/workspace/HubMAP_2023/kaggle_origin/polygons.jsonl"
    root = "/workspace/HubMAP_2023/data/512x512/glomerulus_mask"
    with open(jsonpath, 'r') as json_file:
        json_labels = [json.loads(line) for line in json_file]
    for row in json_labels:
        mask = np.zeros((512, 512), dtype=np.int32)
        for annot in row['annotations']:
            cords = annot['coordinates']
            if annot['type'] == "glomerulus":
                for cord in cords:
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    mask[rr, cc] = 1
        name = row['id']
        mask = mask*255
        contours,_ = cv2.findContours(mask.astype(np.uint8), 1, 2)
        zero_img = np.zeros([mask.shape[0], mask.shape[1], 3], dtype="uint8")
        for p in contours:
            cv2.fillPoly(zero_img, [p], (255, 255, 255))
        cv2.imwrite(root + f"/{name}.png", zero_img)
        print(f'save {name}')


def only_bloodvessel():
    with open('/workspace/HubMAP_2023/kaggle_origin/polygons.jsonl', 'r') as json_file:
        json_labels = [json.loads(line) for line in json_file]
    print(len(json_labels))
    i = 0
    print(json_labels[2]['annotations'])
    return
    while i < len(json_labels):
        if json_labels[i]['annotations']['type'] == 'glomerulus':
            print(f'deleting {i}')
            json_labels.remove(json_labels[i])
        else:
            i += 1
    print(len(json_labels))

def generatesplit():
    with open('/workspace/HubMAP_2023/kaggle_origin/polygons.jsonl', 'r') as json_file:
        json_labels = [json.loads(line) for line in json_file]
    tile_meta = pd.read_csv('/workspace/HubMAP_2023/kaggle_origin/tile_meta.csv')
    submission = pd.DataFrame()
    ids = []
    source_wsi = []
    dataset = []
    for row in json_labels:
        id = row['id']
        sample = tile_meta[tile_meta['id'] == id]
        ids.append(id)
        source_wsi.append(int(sample.source_wsi.values))
        dataset.append(int(sample.dataset.values))
    submission['id'] = ids
    submission['source_wsi'] = source_wsi
    submission['dataset'] = dataset
    print(submission.head())
    #return

    cv = StratifiedKFold(n_splits=5, shuffle=False)
    splits = cv.split(submission.id, submission['source_wsi'])
    for fold, (_, valid_idx) in enumerate(splits):
        submission.loc[valid_idx, "fold"] = fold
    submission.to_csv('/workspace/HubMAP_2023/splits/splits_5fold_new.csv')

def getrestlist():
    existimage_path = '/workspace/HubMAP_2023/kaggle_origin/train'
    tiff_files = []
    for filename in os.listdir(existimage_path):
        if filename.endswith('.tif'):
            name = os.path.splitext(filename)[0]
            tiff_files.append(name)
    print(len(tiff_files))

    mask_path = '/workspace/HubMAP_2023/data/512x512/mask'
    mask_files = []
    for filename in os.listdir(mask_path):
        if filename.endswith('.png'):
            name = os.path.splitext(filename)[0]
            mask_files.append(name)
    #print(mask_files)

    for name in mask_files:
        if name in tiff_files:
            tiff_files.remove(name)
    #print(len(tiff_files))
    return tiff_files

def checksplits():
    df = pd.read_csv('/workspace/HubMAP_2023/splits/splits_5fold_new.csv')
    df_old = pd.read_csv('/workspace/HubMAP_2023/splits/splits_5fold.csv')
    wsi = [1,2]
    fold = [0,1,2,3,4]
    for i in wsi:
        for j in fold:
            _df = df_old[df['dataset'] == i]
            _df = _df[_df['fold'] == j]
            print(f'wsi {i}, fold {j} :{len(_df)}')    

def generate3classmask():
    jsonpath = "/workspace/HubMAP_2023/kaggle_origin/polygons.jsonl"
    root = "/workspace/HubMAP_2023/data/512x512/mask_3class"
    with open(jsonpath, 'r') as json_file:
        json_labels = [json.loads(line) for line in json_file]
    
    for row in json_labels:
        mask_blood = np.zeros((512, 512), dtype=np.int32)
        mask_glo = np.zeros((512, 512), dtype=np.int32)
        mask_unsure = np.zeros((512, 512), dtype=np.int32)
        for annot in row['annotations']:
            cords = annot['coordinates']
            if annot['type'] == "blood_vessel":
                for cord in cords:
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    mask_blood[rr, cc] = 1
            if annot['type'] == "glomerulus":
                for cord in cords:
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    mask_glo[rr, cc] = 1
            if annot['type'] == "unsure":
                for cord in cords:
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    mask_unsure[rr, cc] = 1
        name = row['id']
        mask_blood = mask_blood*255
        mask_glo = mask_glo*255
        mask_unsure = mask_unsure*255
        contours_blood,_ = cv2.findContours(mask_blood.astype(np.uint8), 1, 2)
        contours_glo,_ = cv2.findContours(mask_glo.astype(np.uint8), 1, 2)
        contours_unsure,_ = cv2.findContours(mask_unsure.astype(np.uint8), 1, 2)
        zero_img = np.zeros([mask_blood.shape[0], mask_blood.shape[1], 3], dtype="uint8")
        for p in contours_blood:
            cv2.fillPoly(zero_img, [p], (255, 255, 255))
        for p in contours_glo:
            cv2.fillPoly(zero_img, [p], (64, 64, 64))
        for p in contours_unsure:
            cv2.fillPoly(zero_img, [p], (180, 180, 180))
        cv2.imwrite(root + f"/{name}.png", zero_img)
        print(f'save {name}')

#json2mask()
#json2glomerulusmask()
#only_bloodvessel()
#generatesplit()
#getrestlist()
#checksplits()
#generate3classmask()

def generateD1split():
    with open('/workspace/HubMAP_2023/kaggle_origin/polygons.jsonl', 'r') as json_file:
        json_labels = [json.loads(line) for line in json_file]
    tile_meta = pd.read_csv('/workspace/HubMAP_2023/kaggle_origin/tile_meta.csv')
    d1_data = tile_meta[tile_meta['dataset'] == 1]
    d1_lst = np.array(d1_data.id)
    submission = pd.DataFrame()
    ids = []
    source_wsi = []
    dataset = []
    for row in json_labels:
        id = row['id']
        if id in d1_lst:
            sample = d1_data[d1_data['id'] == id]
            ids.append(id)
            source_wsi.append(int(sample.source_wsi.values))
            dataset.append(1)
    submission['id'] = ids
    submission['source_wsi'] = source_wsi
    submission['dataset'] = dataset
    print(submission.head())
    #return

    cv = StratifiedKFold(n_splits=5, shuffle=False)
    splits = cv.split(submission.id, submission['source_wsi'])
    for fold, (_, valid_idx) in enumerate(splits):
        submission.loc[valid_idx, "fold"] = fold
    submission.to_csv('/workspace/HubMAP_2023/splits/splits_5fold_dataset1.csv')

#generateD1split()

def json2unsuremask():
    jsonpath = "/workspace/HubMAP_2023/kaggle_origin/polygons.jsonl"
    root = "/workspace/HubMAP_2023/data/512x512/unsure_mask"
    with open(jsonpath, 'r') as json_file:
        json_labels = [json.loads(line) for line in json_file]
    
    for row in json_labels:
        mask = np.zeros((512, 512), dtype=np.int32)
        for annot in row['annotations']:
            cords = annot['coordinates']
            if annot['type'] == "unsure":
                for cord in cords:
                    rr, cc = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                    mask[rr, cc] = 1
        name = row['id']
        mask = mask*255
        contours,_ = cv2.findContours(mask.astype(np.uint8), 1, 2)
        zero_img = np.zeros([mask.shape[0], mask.shape[1], 3], dtype="uint8")
        for p in contours:
            cv2.fillPoly(zero_img, [p], (255, 255, 255))
        cv2.imwrite(root + f"/{name}.png", zero_img)
        print(f'save {name}')

#json2unsuremask()

def getd2idlist():
    tile_meta = pd.read_csv('/workspace/HubMAP_2023/kaggle_origin/tile_meta.csv')
    d2_data = tile_meta[tile_meta['dataset'] == 2]
    d2_lst = np.array(d2_data.id)
    return d2_lst
