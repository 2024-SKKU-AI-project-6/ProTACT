import torch
import torch.nn as nn


class TimeDistributedConv1D(nn.Module):
    def __init__(self, maxlen, maxnum, out_channels, kernel_size, padding='valid'):
        super(TimeDistributedConv1D, self).__init__()
        self.conv1d_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(maxlen, out_channels,
                          kernel_size, padding=padding),
                # nn.ReLU()
            ) for _ in range(maxnum)  # there are 97 sentences
        ])

    def forward(self, x):
        sentence_embeddings = []
        for i in range(x.size(1)):  # Iterate over the sentence dimension
            sentence_embedding = self.conv1d_layers[i](x[:, i, :, :])
            sentence_embedding = sentence_embedding.permute(
                0, 2, 1)  # Swap dimensions
            sentence_embeddings.append(sentence_embedding)
        sentence_embeddings = torch.stack(sentence_embeddings, dim=1)
        return sentence_embeddings
