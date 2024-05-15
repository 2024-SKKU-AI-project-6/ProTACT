import torch
import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, maxnum, module):
        super(TimeDistributed, self).__init__()
        self.module = nn.ModuleList([
            nn.Sequential(
                module,
                # nn.ReLU()
            ) for _ in range(maxnum)  # there are 97 sentences
        ])

    def forward(self, x):
        sentence_embeddings = []
        for i in range(x.size(1)):  # Iterate over the sentence dimension
            sentence_embedding = self.module[i](x[:, i, :, :])
            sentence_embeddings.append(sentence_embedding)
        sentence_embeddings = torch.stack(sentence_embeddings, dim=1)

        return sentence_embeddings
