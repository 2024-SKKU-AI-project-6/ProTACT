import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention
from custom_layers.multiheadattention_pe import MultiHeadAttention_PE
from custom_layers.multiheadattention import MultiHeadAttention
from custom_layers.timedistributedconv1D import TimeDistributedConv1D
from custom_layers.timedistributedattention import TimeDistributed


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
        self.max_num = maxnum
        self.max_len = maxlen
        self.output_dim = output_dim
        self.linguistic_feature_count = linguistic_feature_count
        self.readability_feature_count = readability_feature_count

        # Essay Representation
        self.essay_pos_embedding = nn.Embedding(
            pos_vocab_size, self.embedding_dim, padding_idx=0)
        self.essay_pos_x_maskedout = ZeroMaskedEntries(
            input_shape=(None, self.max_num * self.max_len, self.embedding_dim))
        self.essay_pos_dropout = nn.Dropout(self.dropout_prob)

        # for sentence level representation(what about conv2d?)

        # self.essay_pos_conv = nn.Conv2d(self.max_num, self.max_len,
        #                                 self.embedding_dim, self.filters, self.kernel_size, padding='valid')
        # self.essay_pos_conv = nn.Conv1d(
        #     self.embedding_dim, self.filters, self.kernel_size, padding='valid')
        self.essay_pos_conv = TimeDistributedConv1D(
            maxlen=self.max_len, maxnum=self.max_num, out_channels=self.filters, kernel_size=self.kernel_size, padding='valid')

        self.essay_pos_attention = TimeDistributed(
            self.max_num, Attention(input_shape=(None, None, self.filters)))

        self.essay_linquistic = nn.Linear(
            linguistic_feature_count,  self.filters)
        self.essay_readability = nn.Linear(
            readability_feature_count,  self.filters)

        # essay level representation
        self.essay_pos_MA = nn.ModuleList(
            [MultiHeadAttention(100, num_heads) for _ in range(self.output_dim)])
        self.essay_pos_MA_LSTM = nn.ModuleList(  # batch_first=True for (batch_size, sequence_length, input_size) same as keras
            [nn.LSTM(input_size=100,  hidden_size=self.lstm_units, batch_first=True) for _ in range(self.output_dim)])

        self.easay_pos_avg_MA_LSTM = nn.ModuleList(
            [Attention(input_shape=(None, None, self.filters)) for _ in range(self.output_dim)])

        # Prompt Representation
        # word embedding
        self.prompt_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_weights), freeze=True, padding_idx=0)
        # pos embedding
        self.prompt_pos_embedding = nn.Embedding(
            pos_vocab_size, self.embedding_dim, padding_idx=0)
        self.prompt_maskedout = ZeroMaskedEntries(input_shape=(
            None, self.max_num * self.max_len, self.embedding_dim))
        self.prompt_pos_maskedout = ZeroMaskedEntries(input_shape=(
            None, self.max_num * self.max_len, self.embedding_dim))

        # prompt + essay
        self.prompt_dropout = nn.Dropout(self.dropout_prob)
        self.prompt_cnn = TimeDistributedConv1D(
            maxlen=self.max_len, maxnum=self.max_num, out_channels=self.filters, kernel_size=self.kernel_size, padding="valid")
        self.prompt_attention = TimeDistributed(
            self.max_num, Attention(input_shape=(None, None, self.filters)))

        self.prompt_MA = MultiHeadAttention(100, self.num_heads)
        self.prompt_MA_lstm = nn.LSTM(
            input_size=100,  hidden_size=self.lstm_units, batch_first=True)
        self.prompt_avg_MA_lstm = Attention(
            input_shape=(None, None, self.filters))

        self.es_pr_MA_list = nn.ModuleList(
            [MultiHeadAttention_PE(100, self.num_heads) for _ in range(self.output_dim)])
        self.es_pr_MA_lstm_list = nn.ModuleList(
            [nn.LSTM(input_size=100,  hidden_size=self.lstm_units, batch_first=True) for _ in range(self.output_dim)])
        self.es_pr_avg_lstm_list = nn.ModuleList(
            [Attention(input_shape=(None, None, self.filters)) for _ in range(self.output_dim)])

        # 이것도 trait 별로 레이어를 다르게 해야하는지....고민
        self.att_attention = nn.ModuleList([nn.MultiheadAttention(
            num_heads=1, embed_dim=self.filters+linguistic_feature_count + readability_feature_count, batch_first=True) for _ in range(self.output_dim)])
        self.final_dense_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    2 * (self.lstm_units + linguistic_feature_count + readability_feature_count), 1)
            ).to(torch.float32) for _ in range(self.output_dim)])
        # self.final_dense = nn.Sequential(
        #     nn.Linear(374, 1),  # hardcoded
        #     nn.Sigmoid()
        # )

    def forward(self, pos_input, prompt_word_input, prompt_pos_input, linguistic_input, readability_input):
        # Essay Representation

        # pos_input = [(None, 4850)]
        pos_x = self.essay_pos_embedding(pos_input)
        pos_x_maskedout = self.essay_pos_x_maskedout(pos_x)
        pos_drop_x = self.essay_pos_dropout(pos_x_maskedout).transpose(1, 2)
        # print("pos_drop_x", pos_drop_x.shape)
        # reshape the tensor to (none, maxnum, maxlen, embedding_dim)
        pos_resh_W = pos_drop_x.reshape(-1, self.max_num,
                                        self.max_len, self.embedding_dim)
        # (none, maxnum, embedding_dim, maxlen)
        pos_resh_W = pos_resh_W.permute(0, 1, 3, 2)
        pos_zcnn = self.essay_pos_conv(pos_resh_W)
        # (none, maxnum, embedding_dim, maxlen)
        # for fitting the attention layer
        pos_avg_zcnn = self.essay_pos_attention(pos_zcnn)
        # print("pos_avg_zcnn", pos_avg_zcnn.shape)

        pos_MA_list = [self.essay_pos_MA[i](
            pos_avg_zcnn) for i in range(self.output_dim)]
        pos_MA_lstm_list = [self.essay_pos_MA_LSTM[i](
            pos_MA_list[i]) for i in range(self.output_dim)]
        # print shape of pos_MA_lstm_list
        # print("pos_MA_lstm_list length: ", len(pos_MA_lstm_list))
        # print("pos_MA_lstm_list[0][0] shape: ", pos_MA_lstm_list[0][0].shape)
        pos_avg_MA_lstm_list = [self.easay_pos_avg_MA_LSTM[i](
            pos_MA_lstm_list[i][0]) for i in range(self.output_dim)]
        # print("pos_avg_MA_lstm len", len(pos_avg_MA_lstm_list))
        # print("pos_avg_MA_lstm[0] shape", pos_avg_MA_lstm_list[0].shape)

        # Prompt Representation
        prompt = self.prompt_embedding(prompt_word_input)
        prompt_maskedout = self.prompt_maskedout(prompt)
        prompt_pos = self.prompt_pos_embedding(prompt_pos_input)
        prompt_pos_maskedout = self.prompt_pos_maskedout(prompt_pos)

        # add word + pos embedding
        prompt_emb = prompt_maskedout + prompt_pos_maskedout
        prompt_drop_x = self.prompt_dropout(prompt_emb).transpose(1, 2)
        prompt_resh_W = prompt_drop_x.reshape(-1, self.max_num,
                                              self.max_len, self.embedding_dim)
        prompt_resh_W.permute(0, 1, 3, 2)
        prompt_zcnn = self.prompt_cnn(prompt_resh_W)

        prompt_avg_zcnn = self.prompt_attention(prompt_zcnn)
        prompt_MA = self.prompt_MA(prompt_avg_zcnn)
        prompt_MA_lstm = self.prompt_MA_lstm(prompt_MA)

        prompt_avg_MA_lstm = self.prompt_avg_MA_lstm(prompt_MA_lstm[0])

        query = prompt_avg_MA_lstm

        # essay-prompt attention
        # print(pos_avg_MA_lstm_list[0])
        es_pr_MA_list = [self.es_pr_MA_list[i](
            pos_avg_MA_lstm_list[i], query) for i in range(self.output_dim)]
        # print("es_pr_MA_list[0]", es_pr_MA_list[0])
        es_pr_MA_lstm_list = [self.es_pr_MA_lstm_list[i](
            es_pr_MA_list[i]) for i in range(self.output_dim)]
        # print("es_pr_MA_lstm_list", es_pr_MA_lstm_list)
        es_pr_avg_lstm_list = [self.es_pr_avg_lstm_list[i](
            es_pr_MA_lstm_list[i][0]) for i in range(self.output_dim)]
        # print("es_pr_avg_lstm_list", es_pr_avg_lstm_list)
        es_pr_feat_concat = [torch.cat(
            [rep, linguistic_input, readability_input], dim=-1) for rep in es_pr_avg_lstm_list]
        pos_avg_hz_lstm = torch.cat([rep.view(-1, 1, self.lstm_units + self.linguistic_feature_count +
                                    self.readability_feature_count) for rep in es_pr_feat_concat], dim=-2)

        # print("pos_avg_hz_lstm", pos_avg_hz_lstm.shape)
        # print("es_pr_feat_concat", es_pr_feat_concat)
        # pos_avg_hz_lstm = torch.stack(
        #     [rep.unsqueeze(1) for rep in es_pr_feat_concat], dim=1)
        # # print("pos_avg_hz_lstm", pos_avg_hz_lstm.shape)
        # pos_avg_hz_lstm = pos_avg_hz_lstm.squeeze(2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        final_preds = []
        for index in range(self.output_dim):
            mask = [True] * self.output_dim
            mask[index] = False
            non_target_rep = pos_avg_hz_lstm[:, mask]
            target_rep = pos_avg_hz_lstm[:, index:index+1]
            att_output, _ = self.att_attention[index](query=target_rep.to(torch.float32),
                                                      key=non_target_rep.to(
                torch.float32),
                value=non_target_rep.to(torch.float32))
            attention_concat = torch.cat([target_rep, att_output], dim=-1)
            # print("attention_concat", attention_concat.shape)
            # flatten?
            attention_concat = attention_concat.view(
                attention_concat.size(0), -1)
            # print("attention_concat: flatten", attention_concat.shape)

            # final_pred = self.final_dense(attention_concat.to(torch.float32))

            final_pred = torch.sigmoid(self.final_dense_list[index](
                attention_concat.to(torch.float32)))
            # final_pred = torch.sigmoid(
            #     nn.Linear(
            #         attention_concat.shape[-1], 1)(attention_concat.to(torch.float32)))

            final_preds.append(final_pred)

        y = torch.cat(final_preds, dim=1)

        return y
