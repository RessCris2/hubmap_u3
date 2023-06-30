import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torchvision import transforms as t
from sklearn.metrics import roc_auc_score
import utils

def run_train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    cfg,
    scaler,
    epoch,
    iteration,
    step,
    loss_function,
):
    model.train()
    losses = []
    progress_bar = tqdm(range(len(train_dataloader)))
    tr_it = iter(train_dataloader)
    for itr in progress_bar:
        batch = next(tr_it)
        inputs = batch["image"].float().to(cfg.device)
        masks = batch["mask"].float().to(cfg.device)
        iteration += 1
        step += cfg.train_batch_size
        torch.set_grad_enabled(True)
        with autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, masks)
        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        progress_bar.set_description(f"current loss: {loss:.4f} mean loss: {np.mean(losses):.4f} lr: {scheduler.get_last_lr()[0]:.6f}")
