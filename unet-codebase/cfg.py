from types import SimpleNamespace
cfg = SimpleNamespace(**{})

cfg.gpu = 0
cfg.device = "cuda:%d" % cfg.gpu
cfg.df = "/workspace/HubMAP_2023/splits/splits_5fold_new.csv"#dataframe的splits文件
cfg.train_dir = "/workspace/HubMAP_2023/kaggle_origin/train"#原图.tif文件夹
cfg.mask_dir = "/workspace/HubMAP_2023/data/512x512/mask"#mask文件夹
#cfg.mask_dir = "/workspace/HubMAP_2023/data/512x512/mask_d2modified"
#cfg.mask_dir = "/workspace/HubMAP_2023/data/512x512/glomerulus_mask"
cfg.polyjson = "/workspace/HubMAP_2023/kaggle_origin/polygons.jsonl"
cfg.img_size = (512, 512)
#cfg.output_dir = "/workspace/HubMAP_2023/models/effb7"
cfg.output_dir = "/workspace/HubMAP_2023/models/eff7u++_newbase"#权重保存的目录
cfg.encoder_name = 'timm-efficientnet-b7'
cfg.pos_weight = 1.0
cfg.in_channels = 3
cfg.smoothing = 0.1
cfg.classes = 1

cfg.epochs = 15
cfg.fold = 0
cfg.seed = 42
cfg.train_batch_size = 4
cfg.val_batch_size = 4
cfg.lr = 3e-4
cfg.lr_div = 1.0
cfg.lr_final_div = 10000.0
cfg.weight_decay = 1e-2
cfg.run_tta_val = False
cfg.num_workers = 32

#数据增强
import albumentations as A
from albumentations.pytorch import ToTensorV2
cfg.train_transforms = A.Compose([
                A.Rotate(limit=5, p=0.5),
                A.Affine(rotate=5, translate_percent=0.1, scale=[0.9,1.5], shear=0, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(cfg.img_size[0], cfg.img_size[1]),
                A.ShiftScaleRotate(always_apply=False, p=.2,
                                    shift_limit_x=(-1.0, 1.0),
                                    shift_limit_y=(-1.0, 1.0),
                                    scale_limit=(-0.1, 0.1),
                                    rotate_limit=(-5, 5),
                                    interpolation=0,
                                    border_mode=3,
                                    value=(0, 0, 0),
                                    mask_value=None,
                                    rotate_method='largest_box'),
])
cfg.val_transforms = A.Compose([
    A.Resize(cfg.img_size[0], cfg.img_size[1]),
])