import torch
import torch.nn as nn


class ZeroMaskedEntries(nn.Module):
    def __init__(self, input_shape):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__()
        self.output_dim = input_shape[1] # 4850
        self.repeat_dim = input_shape[2] # 50

    def forward(self, x, mask=None): # x.shape : (None, 4850, 50)
        if mask == None:
            return x
        mask = mask.type(torch.float32) # (None, 4850)
        mask = mask.unsqueeze(-1).expand(-1, -1, self.repeat_dim) # (None, 4850, 50)
        mask = mask.permute(0, 2, 1) # mask.shape: (None, 50, 4850)
        return x * mask
