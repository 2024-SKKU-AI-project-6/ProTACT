import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim # 100
        self.num_heads = num_heads # 2

        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads # 50
        self.query_dense = nn.Linear(embedding_dim, embedding_dim)
        self.key_dense = nn.Linear(embedding_dim, embedding_dim)
        self.value_dense = nn.Linear(embedding_dim, embedding_dim)
        self.dense = nn.Linear(embedding_dim, embedding_dim)

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1)) # (1680, 2, 97, 97)
        depth = torch.FloatTensor([key.shape[-1]])[0]
        logits = matmul_qk / torch.sqrt(depth)
        attention_weights = F.softmax(logits, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        batch_size = x.size(0)

        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(
            query, key, value)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        # (batch_size, seq_len, embedding_dim)
        concat_attention = scaled_attention.reshape(
            batch_size, -1, self.embedding_dim)
        y = self.dense(concat_attention)
        return y
