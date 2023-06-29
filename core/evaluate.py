"""evaluate the model effects
"""
import numpy as np
import pandas as pd
import sys
sys.path.append("/root/autodl-tmp/archive/core/metrics")
from infer_base import predict_oneimg, predict_dir, loads_model_dynamic
from hubmap_config import  model_path_dict
from utils import load_img, get_bounding_box, fn_time, find_files
import torch
torch.cuda.empty_cache()
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import pathlib

def load_gt(img_path):
    true = {}
    true['masks'] = load_img(img_path.replace("images", 'inst_mask').replace("jpg", 'npy'))
    true['labels'] = true['masks'].shape[0] * [0]
    true['bboxes'] = np.stack([get_bounding_box(mask.astype(int)) for mask  in true['masks'] ])
    true['scores'] = true['masks'].shape[0] * [1]
    return true


@fn_time
def mask_rcnn_predict_oneimg(img_path, dataset_name, model_name, model_path_dict, model=None, load_model=True, score_thr=0.3, ):
    """
        load_model: 如果要直接传入数据 [config_file, checkpoint_file], 修改 config_file 才行？
    """
    if load_model:
        model = loads_model_dynamic(model_name, dataset_name, model_path_dict)
    result = inference_detector(model, img_path) ## 注意这里，可以使用 img_path, 不需要自己加载
    res = result.pred_instances.to_dict()
    pred = dict()
    pred['labels'] = res['labels'][res['scores']>score_thr].cpu().numpy()
    pred['masks'] = res['masks'][res['scores']>score_thr].cpu().numpy().astype(int)
    pred['bboxes'] = res['bboxes'][res['scores']>score_thr].cpu().numpy().astype(int)
    pred['scores'] = res['scores'][res['scores']>score_thr].cpu().numpy()
    # pred['centroids']= fetch_inst_centroid_maskrcnn(pred['masks'])

    if len(pred['labels']) == 0:
        raise ValueError

    true = load_gt(img_path)
    return pred, true


def mask_rcnn_predict_dir(pred_dir, dataset_name, model_name, model_path_dict,load_model=True):
    """
        is_exs: 是否差处理好数据
    """
    if load_model:
        model = loads_model_dynamic(model_name, dataset_name, model_path_dict)
    torch.cuda.empty_cache()

    files = find_files(pred_dir,'jpg')
    preds_result = []
    trues_result = []
    basenames =[]
    for file in files:
        try:
            pred, true = mask_rcnn_predict_oneimg(file, dataset_name, model_name, model_path_dict,  model, load_model=False)
            preds_result.append(pred)
            trues_result.append(true)
            basename = pathlib.Path(file).stem
            basenames.append(basename)
        except:
            print("{} is wrong!".format(file))
            continue
    return basenames, preds_result, trues_result


if __name__ == "__main__":
  pred_dir = "/root/autodl-tmp/kaggle/datasets/coco_format/images/test"
  dataset_name = 'hubmap'
  model_name = 'mask_rcnn'
  basenames, preds_result, trues_result = mask_rcnn_predict_dir(pred_dir, dataset_name, model_name, model_path_dict)
  # files = find_files(pred_dir,'jpg')
  # img_path = files[0]
  # pred, true = mask_rcnn_predict_oneimg(img_path, dataset_name, model_name, model_path_dict)

  def concat_res(result):
        processed = []
        for res in result:
            ress = dict(
            boxes=torch.tensor(res['bboxes']),
            scores=torch.tensor(res['scores']),
            labels=torch.tensor(res['labels']),
            masks =torch.tensor(res['masks'], dtype=torch.uint8)
             )
            processed.append(ress)
        return processed
    
  preds = concat_res(preds_result)
  trues = concat_res(trues_result)

  metric = MeanAveragePrecision(iou_type='segm', iou_thresholds=[0.6])
  metric.update(preds, trues)
  metrics0 = metric.compute()
  from pprint import pprint
  print(metrics0)
  # metrics0 = pd.DataFrame(metrics0, index=[0], columns=['map_50', 'map_75']).values[0].tolist()