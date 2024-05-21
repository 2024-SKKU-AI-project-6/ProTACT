import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_shape, op='attsum', activation='tanh', init_stdev=0.01):
        super(Attention, self).__init__()
        assert op in {"attsum", "attmean"}
        assert activation in {None, "tanh"}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        self.supports_masking = True
        self.input_shape = input_shape
        init_val_v = (torch.randn(
            input_shape[2]) * init_stdev).type(torch.float32)
        self.att_V = nn.Parameter(init_val_v)
        init_val_w = (torch.randn(
            input_shape[2], input_shape[2]) * init_stdev).type(torch.float32)
        self.att_W = nn.Parameter(init_val_w)

    def forward(self, x, mask=None):
        w = torch.matmul(x, self.att_W)
        if not self.activation:
            w = torch.tensordot(self.att_V, w, dims=[[0], [2]])
        elif self.activation == 'tanh':
            w = torch.tensordot(self.att_V, torch.tanh(w), dims=[[0], [2]])

        w = F.softmax(w, dim=-1)
        y = x * w.unsqueeze(2).expand_as(x)
        if self.op == 'attsum':
            y = torch.sum(y, dim=1)
        elif self.op == 'attmean':
            if mask is not None:
                y = torch.sum(y, dim=1) / torch.sum(mask, dim=1, keepdim=True)
            else:
                y = torch.mean(y, dim=1)
        return y.type(torch.float32)

    def get_config(self):
        return {"op": self.op, "activation": self.activation, "init_stdev": self.init_stdev}
