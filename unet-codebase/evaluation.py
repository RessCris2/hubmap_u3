import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from metrics import DiceCoef
import torch.nn as nn

def run_eval(model, val_dataloader, cfg, epoch):
    model.eval()
    torch.set_grad_enabled(False)
    progress_bar = tqdm(range(len(val_dataloader)))
    tr_it = iter(val_dataloader)
    dicelst = []
    metric = DiceCoef()
    for itr in progress_bar:
        batch = next(tr_it)
        inputs = batch["image"].float().to(cfg.device)
        masks = batch["mask"].float().to(cfg.device)
        outputs = model(inputs)
        dice = metric(outputs,masks)
        dicelst.append(float(dice))
    print(np.mean(dicelst))
    return np.mean(dicelst)