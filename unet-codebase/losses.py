import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

#Dice+Bce loss
class Unet_CustomLoss(nn.Module):
    
    def __init__(self):
        
        super(Unet_CustomLoss,self).__init__()
        
        self.diceloss = smp.losses.DiceLoss(mode='binary')
        self.bceloss = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.1)
        self.tverloss = smp.losses.TverskyLoss(mode='binary', alpha = 0.3, beta = 0.7)#panelize false positive
        
    def forward(self, output, mask):
        
        output = torch.squeeze(output)
        mask = torch.squeeze(mask)

        dice = self.diceloss(output , mask)
        bce = self.bceloss(output , mask)
        tver = self.tverloss(output, mask)

        loss = 0.4 * bce + 0.4 * dice + 0.2 * tver

        return loss

if __name__ == "__main__":

    # 与GT一样的sigmoid结果
    # pred = torch.Tensor([[
    #         [[0.95, 0.35],
    #         [0.05, 0.05]],
    #         [[0.05, 0.05],
    #         [0.05, 0.95]],
    #         [[0.95, 0.95],
    #         [0.05, 0.95]],
    # ]])

    #与GT一样的logits
    #
    pred = torch.Tensor([[
            [[10, -10],
            [-8, -9]],
            [[-7, -5],
            [-2.8, 4.2]],
            [[2.9, 7.6],
            [-4, 2]],
    ]])

    gt = torch.Tensor([[
            [[1, 0],
            [0, 0]],
            [[0, 0],
            [0, 1]],
            [[1, 1],
            [0, 1]],
    ]])
    loss = Unet_CustomLoss()
    print(loss(pred,gt))
