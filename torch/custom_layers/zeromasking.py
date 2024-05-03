import torch
import torch.nn as nn



class ZeroMaskedEntries(nn.Module):
    def __init__(self, input_shape):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__()
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]


    def call(self, x, mask=None):
        if mask == None:
            return x
        mask = mask.type(torch.float32)
        mask = mask.unsqueeze(-1).expand(-1, -1, self.repeat_dim)
        mask = mask.permute(0, 2, 1)
        return x * mask
