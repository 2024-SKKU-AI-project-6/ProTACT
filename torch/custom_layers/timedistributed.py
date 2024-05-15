import torch
import torch.nn as nn

# class TimeDistributed(nn.Module):
#     def __init__(self, layer, time_steps, *args):        
#         super(TimeDistributed, self).__init__()
        
#         self.layers = nn.ModuleList([layer(*args) for i in range(time_steps)])

#     def forward(self, x):

#         batch_size, time_steps, C, H, W = x.size()
#         output = torch.tensor([])
#         for i in range(time_steps):
#           output_t = self.layers[i](x[:, i, :, :, :])
#           output_t  = y.unsqueeze(1)
#           output = torch.cat((output, output_t ), 1)
#         return output

import torch
import torch.nn as nn

class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # x: (None, max_num, max_len, embedding_dim)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        x_reshape.permute(1,0)
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y