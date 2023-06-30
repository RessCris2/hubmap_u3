import torch
import torch.nn as nn
class DiceCoef(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
    
        super().__init__()

    def forward(self, y_pred, y_true, smooth=1.):

        m = nn.Sigmoid()
        y_pred = m(y_pred)
        
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        
        y_pred = torch.round((y_pred - y_pred.min()) / (y_pred.max() - y_pred.min()))
        
        intersection = (y_true * y_pred).sum()
        
        dice = (2.0 * intersection + smooth)/(y_true.sum() + y_pred.sum() + smooth)
        
        return dice

if __name__ == "__main__":
    # pred = torch.Tensor([[
    #         [[0, 0],
    #         [0, 1]],
    #         [[0, 0],
    #         [0, 0]],
    #         [[1, 0],
    #         [0, 1]],
    # ]])

    pred = torch.Tensor([[
            [[-2, -3],
            [-2, 1.5]],
            [[2, -2],
            [-2, -0.8]],
            [[1.3, -2.1],
            [-2, 1.1]],
    ]])
    gt = torch.Tensor([[
            [[0, 0],
            [0, 1]],
            [[1, 0],
            [0, 0]],
            [[1, 0],
            [0, 1]],
    ]])
    
    metric = DiceCoef()
    print(pred)
    num = metric(pred,gt)
    print(float(num))