import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention_PE(nn.Module):
    def __init__(self, embedding_dim, num_heads=8):
        super(MultiHeadAttention_PE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads
        self.query_dense = nn.Linear(embedding_dim, embedding_dim)
        self.key_dense = nn.Linear(embedding_dim, embedding_dim)
        self.value_dense = nn.Linear(embedding_dim, embedding_dim)
        self.dense = nn.Linear(embedding_dim, embedding_dim)

    # i don't know how to implement this in pytorch but i think this is the equivalent(and chat gpt gave me this code)
    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        depth = torch.FloatTensor([key.shape[-1]])[0]
        logits = matmul_qk / torch.sqrt(depth)
        attention_weights = F.softmax(logits, dim=-1)
        output = torch.matmul(attention_weights,  value)

        # print("output", output.shape)

        return output, attention_weights

    # def scaled_dot_product_attention(self, query, key, value):
    #     matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    #     depth = torch.FloatTensor(key.shape[-1])
    #     logits = matmul_qk / torch.sqrt(depth)
    #     attention_weights = F.softmax(logits, dim=-1)
    #     print("attetion weight", attention_weights.shape)
    #     print("value", value.shape)
    #     output = torch.matmul(attention_weights, value)
    #     return output, attention_weights

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, q):
        batch_size = x.size(0)

        query = self.query_dense(q)
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

        # print("y", y.shape)
        return y
