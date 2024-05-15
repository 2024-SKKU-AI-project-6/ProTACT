import torch
import torch.nn as nn


class TimeDistributedAttention(nn.Module):
    def __init__(self, module):
        super(TimeDistributedAttention, self).__init__()
        self.module = module

    def forward(self, x):
        # Reshape input to (batch_size * sequence_length, input_size)
        batch_size, seq_length, in_channels, input_size = x.size()
        x = x.reshape(batch_size * seq_length, in_channels, input_size)

        # Apply the custom layer to each time step
        x = self.module(x)

        # Reshape output to (batch_size, sequence_length, output_size)
        x = x.reshape(batch_size, seq_length, -1)

        return x
