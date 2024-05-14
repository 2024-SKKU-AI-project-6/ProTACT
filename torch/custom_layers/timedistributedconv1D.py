import torch
import torch.nn as nn


class TimeDistributedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='valid', dilation=1, groups=1, bias=True):
        super(TimeDistributedConv1D, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        # Reshape input to (batch_size * sequence_length, in_channels, input_size)
        batch_size, seq_length, in_channels, input_size = x.size()
        x = x.view(batch_size * seq_length, in_channels, input_size)

        # Apply 1D convolution
        x = self.conv1d(x)

        # Reshape output to (batch_size, sequence_length, output_size, out_channels)
        output_size = x.size(-1)
        x = x.view(batch_size, seq_length, output_size, -1)

        x = x.permute(0, 1, 2, 3)

        return x
