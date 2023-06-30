import argparse
import gc
import importlib
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
import time
import utils
from dataset import MyDataset
from training import run_train
from evaluation import run_eval
from losses import Unet_CustomLoss
import segmentation_models_pytorch as smp

def main(cfg):

    os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)
    utils.set_seed(cfg.seed)
    df = pd.read_csv(cfg.df)
    train_df = df[df["fold"] != cfg.fold]
    val_df = df[df["fold"] == cfg.fold]
    if cfg.fold == 5:
        train_df = df
        val_df = df
    train_dataset = MyDataset(df=train_df,image_dir=cfg.train_dir,mask_dir=cfg.mask_dir,cfg=cfg,aug=cfg.train_transforms)
    val_dataset = MyDataset(df=val_df,image_dir=cfg.train_dir,mask_dir=cfg.mask_dir,cfg=cfg,aug=cfg.val_transforms)
    print("train: ", len(train_dataset), " val: ", len(val_dataset))
    train_dataloader = utils.get_train_dataloader(train_dataset, cfg)
    val_dataloader = utils.get_val_dataloader(val_dataset, cfg)

    model = smp.UnetPlusPlus(encoder_name=cfg.encoder_name,encoder_weights='imagenet',activation=None)
    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        epochs=cfg.epochs,
        steps_per_epoch=int(len(train_dataset) / cfg.train_batch_size),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=cfg.lr_div,
        final_div_factor=cfg.lr_final_div,
    )
    loss_function = Unet_CustomLoss()
    scaler = GradScaler()
    step = 0
    i = 0
    best_metric = 0.0
    optimizer.zero_grad()
    print("start from: ", best_metric)
    for epoch in range(cfg.epochs):
        print('Size: ', cfg.img_size)
        print('Model save to: ', cfg.output_dir)
        print('Batch size: ', cfg.train_batch_size)
        print('Fold: ', cfg.fold)
        print('Total epoch: ', cfg.epochs)
        print("EPOCH: ", epoch)
        gc.collect()
        
        run_train(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            scaler=scaler,
            epoch=epoch,
            iteration=i,
            step=step,
            loss_function=loss_function,
        )
        
        t1 = time.time()
        dice_mean = run_eval(
            model=model,
            val_dataloader=val_dataloader,
            cfg=cfg,
            epoch=epoch,
        )
        t2 = time.time()
        print('Val time: ', str(t2 - t1))
        if dice_mean > best_metric:
            print(f"SAVING CHECKPOINT: val_metric {best_metric:.5} -> {dice_mean:.5}")
            best_metric = dice_mean
            checkpoint = utils.create_checkpoint(
                model,
                optimizer,
                epoch,
                scheduler=scheduler,
                scaler=scaler,
            )
            torch.save(
                checkpoint,
                f"{cfg.output_dir}/fold{cfg.fold}/fold{cfg.fold}.pth",
            )
        print('\n')

if __name__ == "__main__":
    sys.path.append("configs")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-c", "--config", default="cfg", help="config filename")
    parser.add_argument("-f", "--fold", type=int, default=0, help="fold")
    parser_args, _ = parser.parse_known_args(sys.argv)
    cfg = importlib.import_module(parser_args.config).cfg
    cfg.fold = parser_args.fold
    main(cfg)