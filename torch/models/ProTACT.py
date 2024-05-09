import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention
from custom_layers.multiheadattention_pe import MultiHeadAttention_PE
from custom_layers.multiheadattention import MultiHeadAttention


class ProTACT(nn.Module):
    def __init__(self, pos_vocab_size, vocab_size, maxnum, maxlen, readability_feature_count,
                 linguistic_feature_count, configs, output_dim, num_heads, embedding_weights):
        super(ProTACT, self).__init__()
        self.embedding_dim = configs.EMBEDDING_DIM  # 50
        self.dropout_prob = configs.DROPOUT  # 0.5
        self.filters = configs.CNN_FILTERS  # 100
        self.kernel_size = configs.CNN_KERNEL_SIZE  # 5
        self.lstm_units = configs.LSTM_UNITS  # 100
        self.num_heads = num_heads  # 2

        # Essay Representation
        self.essay_pos_embedding = nn.Embedding(
            pos_vocab_size, self.embedding_dim, padding_idx=0)
        self.essay_pos_dropout = nn.Dropout(self.dropout_prob)
        # essay_pos_dropout.shape: (input.shape, embedding_dim) => (none, 4850, 50)
        self.essay_pos_conv = nn.Conv1d(
            self.embedding_dim, self.filters, self.kernel_size, padding='valid')
        # essay_pos_conv.
        self.essay_pos_attention = Attention(
            input_shape=(None, None, self.filters))

        self.essay_linquistic = nn.Linear(
            linguistic_feature_count,  self.filters)
        self.essay_readability = nn.Linear(
            readability_feature_count,  self.filters)

        self.essay_pos_MA = nn.ModuleList(
            [MultiHeadAttention(100, num_heads) for _ in range(output_dim)])
        self.essay_pos_MA_LSTM = nn.ModuleList(  # batch_first=True for (batch_size, sequence_length, input_size) same as keras
            [nn.LSTM(input_size=100,  hidden_size=self.lstm_units, batch_first=True) for _ in range(output_dim)])
        self.easay_pos_avg_MA_LSTM = nn.ModuleList(
            [Attention(input_shape=(None, None, self.filters)) for _ in range(output_dim)])

        # Prompt Representation
        self.prompt_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_weights), freeze=True, padding_idx=0)
        self.prompt_pos_embedding = nn.Embedding(
            pos_vocab_size, self.embedding_dim, padding_idx=0)
        self.prompt_dropout = nn.Dropout(self.dropout_prob)
        self.prompt_cnn = nn.Conv1d(
            self.embedding_dim, self.filters, self.kernel_size, padding='valid')
        self.prompt_attention = Attention(
            input_shape=(None, None, self.filters))

        self.prompt_MA = MultiHeadAttention(100, num_heads)
        self.prompt_MA_lstm = nn.LSTM(
            input_size=100,  hidden_size=self.lstm_units, batch_first=True)
        self.prompt_avg_MA_lstm = Attention(
            input_shape=(None, None, self.filters))

        self.es_pr_MA_list = nn.ModuleList(
            [MultiHeadAttention_PE(self.filters, num_heads) for _ in range(output_dim)])
        self.es_pr_MA_lstm_list = nn.ModuleList(
            [nn.LSTM(input_size=100,  hidden_size=self.lstm_units, batch_first=True) for _ in range(output_dim)])
        self.es_pr_avg_lstm_list = nn.ModuleList(
            [Attention(input_shape=(None, None, self.filters)) for _ in range(output_dim)])

        self.final_dense_list = nn.ModuleList([nn.Linear(
            2 * self.lstm_units + linguistic_feature_count + readability_feature_count, 1) for _ in range(output_dim)])

    def forward(self, pos_input, prompt_word_input, prompt_pos_input, linguistic_input, readability_input):
        # Essay Representation

        #     train_dataset = TensorDataset(
        #     torch.from_numpy(X_train_pos),
        #     torch.from_numpy(X_train_prompt),
        #     torch.from_numpy(X_train_prompt_pos),
        #     torch.from_numpy(X_train_linguistic_features),
        #     torch.from_numpy(X_train_readability),
        #     torch.from_numpy(Y_train)
        # )

        # pos_input = [(None, 4850)]
        pos_x = self.pos_embedding(pos_input)
        pos_x_maskedout = ZeroMaskedEntries()(pos_x)
        pos_drop_x = self.pos_dropout(pos_x_maskedout)
        pos_resh_W = pos_drop_x.view(-1, self.maxnum,
                                     self.maxlen, self.embedding_dim).transpose(2, 3)
        pos_zcnn = self.pos_cnn(pos_resh_W)
        pos_avg_zcnn = self.pos_attention(pos_zcnn)

        pos_MA_list = [self.pos_MA_list[i](
            pos_avg_zcnn) for i in range(self.output_dim)]
        pos_MA_lstm_list = [self.pos_MA_lstm_list[i](
            pos_MA_list[i]) for i in range(self.output_dim)]
        pos_avg_MA_lstm_list = [self.pos_avg_MA_lstm_list[i](
            pos_MA_lstm_list[i]) for i in range(self.output_dim)]

        # Prompt Representation
        prompt = self.prompt_embedding(prompt_word_input)
        prompt_maskedout = ZeroMaskedEntries()(prompt)
        prompt_pos = self.prompt_pos_embedding(prompt_pos_input)
        prompt_pos_maskedout = ZeroMaskedEntries()(prompt_pos)

        prompt_emb = prompt_maskedout + prompt_pos_maskedout
        prompt_drop_x = self.prompt_dropout(prompt_emb)
        prompt_resh_W = prompt_drop_x.view(-1, self.maxnum,
                                           self.maxlen, self.embedding_dim).transpose(2, 3)
        prompt_zcnn = self.prompt_cnn(prompt_resh_W)
        prompt_avg_zcnn = self.prompt_attention(prompt_zcnn)

        prompt_MA = self.prompt_MA(prompt_avg_zcnn)
        prompt_MA_lstm = self.prompt_MA_lstm(prompt_MA)
        prompt_avg_MA_lstm = self.prompt_avg_MA_lstm(prompt_MA_lstm)

        query = prompt_avg_MA_lstm

        es_pr_MA_list = [self.es_pr_MA_list[i](
            pos_avg_MA_lstm_list[i], query) for i in range(self.output_dim)]
        es_pr_MA_lstm_list = [self.es_pr_MA_lstm_list[i](
            es_pr_MA_list[i]) for i in range(self.output_dim)]
        es_pr_avg_lstm_list = [self.es_pr_avg_lstm_list[i](
            es_pr_MA_lstm_list[i]) for i in range(self.output_dim)]
        es_pr_feat_concat = [torch.cat(
            [rep, linguistic_input, readability_input], dim=-1) for rep in es_pr_avg_lstm_list]
        pos_avg_hz_lstm = torch.stack(
            [rep.unsqueeze(1) for rep in es_pr_feat_concat], dim=1)

        final_preds = []
        for index in range(self.output_dim):
            mask = [True] * self.output_dim
            mask[index] = False
            non_target_rep = pos_avg_hz_lstm[:, mask]
            target_rep = pos_avg_hz_lstm[:, index:index+1]
            att_attention = nn.Attention()([target_rep, non_target_rep])
            attention_concat = torch.cat([target_rep, att_attention], dim=-1)
            attention_concat = attention_concat.view(
                attention_concat.size(0), -1)
            final_pred = self.final_dense_list[index](attention_concat)
            final_preds.append(final_pred)

        y = torch.cat(final_preds, dim=-1)

        return y
