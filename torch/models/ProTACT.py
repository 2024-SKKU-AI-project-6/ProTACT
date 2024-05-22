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
        self.max_num = maxnum
        self.max_len = maxlen
        self.output_dim = output_dim # 9

        ## Essay Representation
        self.essay_pos_embedding = nn.Embedding(
            pos_vocab_size, self.embedding_dim, padding_idx=0)
        self.essay_pos_x_maskedout = ZeroMaskedEntries(
            input_shape=(None, self.max_num * self.max_len, self.embedding_dim))
        self.essay_pos_dropout = nn.Dropout(self.dropout_prob)
        # self.essseay_pos_resh_W # in forward
        # essay_pos_dropout.shape: (input.shape, embedding_dim) => (none, 4850, 50)
        # reshape to (none, maxnum, maxlen, embedding_dim)

        # for sentence level representation(what about conv2d?)
        # self.essay_pos_conv = TimeDistributedConv1D(
        #     self.embedding_dim, self.filters, self.kernel_size, padding='valid')
        self.essay_pos_conv = nn.Conv1d(self.embedding_dim, self.filters, self.kernel_size, padding='valid')
        self.essay_pos_conv_list = nn.ModuleList(self.essay_pos_conv for i in range(self.max_num))
        

        #self.essay_pos_attention = TimeDistributedAttention(self.attention_module)
        self.essay_pos_attention_list = nn.ModuleList(Attention(
            input_shape=(None, None, self.filters)) for i in range(self.max_num))
        print("self.essay_pos_attention_list", self.essay_pos_attention_list)

        self.essay_pos_MA = nn.ModuleList(
            MultiHeadAttention(100, num_heads) for _ in range(self.output_dim))
        # self.lstm_unts: 100
        self.essay_pos_MA_LSTM = nn.ModuleList(  # batch_first=True for (batch_size, sequence_length, input_size) same as keras
            nn.LSTM(input_size=100,  hidden_size=self.lstm_units, batch_first=True) for _ in range(self.output_dim))
        self.essay_pos_avg_MA_LSTM = nn.ModuleList(
            Attention(input_shape=(None, None, self.filters)) for _ in range(self.output_dim))

        ## Prompt Representation
        self.prompt_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_weights), freeze=True, padding_idx=0)
        self.prompt_pos_embedding = nn.Embedding(
            pos_vocab_size, self.embedding_dim, padding_idx=0)
        # mask
        self.prompt_maskedout = ZeroMaskedEntries(input_shape=(
            None, self.max_num * self.max_len, self.embedding_dim))
        self.prompt_pos_maskedout = ZeroMaskedEntries(input_shape=(
            None, self.max_num * self.max_len, self.embedding_dim))

        # add word + pos embedding
        self.prompt_dropout = nn.Dropout(self.dropout_prob)
        #self.prompt_cnn = TimeDistributedConv1D(self.embedding_dim, self.filters, self.kernel_size, padding='valid')
        self.prompt_conv = nn.Conv1d(self.embedding_dim,self.filters,self.kernel_size, padding='valid')
        self.prompt_conv_list = nn.ModuleList(self.prompt_conv for i in range(self.max_num))
        #self.prompt_attention = TimeDistributedAttention
        self.prompt_attention = Attention(
            input_shape=(None, None, self.filters))
        self.prompt_attention_list = nn.ModuleList( self.prompt_attention for i in range(self.max_num))

        self.prompt_MA = MultiHeadAttention(100, num_heads)
        self.prompt_MA_lstm = nn.LSTM(
            input_size=100,  hidden_size=self.lstm_units, batch_first=True)
        self.prompt_avg_MA_lstm = Attention(
            input_shape=(None,None, self.filters))
        
        # 여기에서 합처진다.
        self.es_pr_MA_list = nn.ModuleList(
            [MultiHeadAttention_PE(self.filters, num_heads) for _ in range(self.output_dim)])
        self.es_pr_MA_lstm_list = nn.ModuleList(
            [nn.LSTM(input_size=100,  hidden_size=self.lstm_units, batch_first=True) for _ in range(self.output_dim)])
        self.es_pr_avg_lstm_list = nn.ModuleList(
            [Attention(input_shape=(None, None, self.filters)) for _ in range(self.output_dim)])
    
        
        # why 2 * self.lstm_units?
        self.att_attention_list = nn.ModuleList()
        self.flatten_list = nn.ModuleList()
        self.final_dense_list = nn.ModuleList()
        for _ in range(self.output_dim):
            self.att_attention_list.append(nn.MultiheadAttention(num_heads=1, embed_dim=self.filters+linguistic_feature_count + readability_feature_count))
            self.flatten_list.append(nn.Flatten())
            self.final_dense_list.append(nn.Linear(2*(self.filters + linguistic_feature_count + readability_feature_count), 1))
    def forward(self, pos_input, prompt_word_input, prompt_pos_input, linguistic_input, readability_input):
        ### 1. Essay Representation

        #     train_dataset = TensorDataset(
        #     torch.from_numpy(X_train_pos),
        #     torch.from_numpy(X_train_prompt),
        #     torch.from_numpy(X_train_prompt_pos),
        #     torch.from_numpy(X_train_linguistic_features),
        #     torch.from_numpy(X_train_readability),
        #     torch.from_numpy(Y_train)
        # )

        # pos_input = [(None, 4850)]
        pos_x = self.essay_pos_embedding(pos_input) # (pos_vocab_size, embedding_dim): (None, 4850, 50)
        pos_x_maskedout = self.essay_pos_x_maskedout(pos_x) # (None, pos_vocab_size, embedding_dim)
        pos_drop_x = self.essay_pos_dropout(pos_x_maskedout)  # (None, pos_vocab_size, embedding_dim)
        print("pos_drop_x: ", pos_drop_x.shape)
        
        # reshape the tensor to (none, maxnum, maxlen, embedding_dim)
        pos_resh_W = pos_drop_x.reshape(-1, self.max_num,
                                        self.max_len, self.embedding_dim) # (None, 97, 50, 50)
        print("pos_resh_W", pos_resh_W.shape) 
        
        # TimeDistributed 실행
        pos_resh_W = pos_resh_W.permute(0,1,3,2) # (none, maxnum, embedding_dim, maxlen)
        #pos_zcnn = self.essay_pos_conv(pos_resh_W) # (none, maxnum, maxlen, kernel)
        pos_zcnn = torch.tensor([])
        for i in range(self.max_num): # 97
            output_t = self.essay_pos_conv_list[i](pos_resh_W[:,i,:,:])  # (none, 46, 100)
            output_t  = output_t.unsqueeze(1) # (None, 1, 46,100)
            pos_zcnn = torch.cat((pos_zcnn, output_t ), 1) #(None, 2, 100, 46)
        pos_zcnn = pos_zcnn.permute(0,1,3,2)
        print("pos_zcnn", pos_zcnn.shape) # (none, 97, 46, 100)
        
        # for fitting the attention layer
        # from here...
        # pos_zcnn = pos_zcnn.view(-1, , , self.filters)
        # pos_avg_zcnn = self.essay_pos_attention(pos_zcnn)
        
        # TimeDistributed Attention
        pos_avg_zcnn = torch.tensor([])
        for i in range(self.max_num): # (None,46,100)
            output_t = self.essay_pos_attention_list[i](pos_zcnn[:,i,:]) # (None,100)
            output_t = output_t.unsqueeze(1)
            pos_avg_zcnn = torch.cat((pos_avg_zcnn, output_t),1)
        print("pos_avg_zcnn", pos_avg_zcnn.shape)
        
        
        pos_MA_list = [self.essay_pos_MA[i](pos_avg_zcnn) for i in range(self.output_dim)]
        pos_MA_lstm_list = [self.essay_pos_MA_LSTM[i](pos_MA_list[i])[0] for i in range(self.output_dim)]
        pos_avg_MA_lstm_list = [self.essay_pos_avg_MA_LSTM[i](pos_MA_lstm_list[i]) for i in range(self.output_dim)]

        ### 2. Prompt Representation 
        # word embedding
        prompt = self.prompt_embedding(prompt_word_input)
        prompt_maskedout = self.prompt_maskedout(prompt)
        
        # pos embedding
        prompt_pos = self.prompt_pos_embedding(prompt_pos_input)
        prompt_pos_maskedout = self.prompt_pos_maskedout(prompt_pos)
        
        # word + pos embedding
        prompt_emb = prompt_maskedout + prompt_pos_maskedout
        
        prompt_drop_x = self.prompt_dropout(prompt_emb).transpose(1, 2)
        prompt_resh_W = prompt_drop_x.reshape(-1, self.max_num, self.max_len, self.embedding_dim)
        prompt_resh_W = prompt_resh_W.permute(0,1,3,2) # (none, maxnum, embedding_dim, maxlen)
        prompt_zcnn = torch.tensor([])
        
        # TimeDistributed Conv1
        for i in range(self.max_num): # 97
            output_t = self.prompt_conv_list[i](prompt_resh_W[:,i,:,:])  # (none, 46, 100)
            output_t  = output_t.unsqueeze(1) # (None, 1, 46,100)
            prompt_zcnn = torch.cat((prompt_zcnn, output_t ), 1) #(None, 2, 100, 46)
        prompt_zcnn = prompt_zcnn.permute(0,1,3,2)
        print("prompt_zcnn", prompt_zcnn.shape) # (none, 97, 46, 100)
        
        #TimeDistributed Attention
        prompt_avg_zcnn = torch.tensor([])
        for i in range(self.max_num): # (None,46,100)
            output_t = self.prompt_attention_list[i](prompt_zcnn[:,i,:]) # (None,100)
            output_t = output_t.unsqueeze(1)
            prompt_avg_zcnn = torch.cat((prompt_avg_zcnn, output_t),1)
        
        print("prompt_avg_zcnn", prompt_avg_zcnn.shape)
        
        # for fitting the attention layer
        # prompt_zcnn = prompt_zcnn.view(-1,
        #                                self.max_num, self.max_len, self.filters)
        

        prompt_MA = self.prompt_MA(prompt_avg_zcnn)
        
        print("prompt_MA_lstm shape: ", prompt_MA.shape)
        prompt_MA_lstm= self.prompt_MA_lstm(prompt_MA)
        prompt_avg_MA_lstm = self.prompt_avg_MA_lstm(prompt_MA_lstm[0])
        

        query = prompt_avg_MA_lstm
        es_pr_MA_list = [self.es_pr_MA_list[i]( pos_avg_MA_lstm_list[i], query) for i in range(self.output_dim)]
        es_pr_MA_lstm_list = [self.es_pr_MA_lstm_list[i](es_pr_MA_list[i])[0] for i in range(self.output_dim)]
        es_pr_avg_lstm_list = [self.es_pr_avg_lstm_list[i](es_pr_MA_lstm_list[i]) for i in range(self.output_dim)]
        es_pr_feat_concat = [torch.cat([rep, linguistic_input, readability_input], dim=-1) for rep in es_pr_avg_lstm_list]
        print("linguistic_input: ",linguistic_input.shape)
        print("readability_input: ",readability_input.shape)
        print("es_pr_feat_concat: ",len(es_pr_feat_concat),", ",es_pr_feat_concat[0].shape)
        
        pos_avg_hz_lstm = torch.tensor([])
        for es_pr_feat in es_pr_feat_concat:
            es_pr_feat_unsqueeze = es_pr_feat.unsqueeze(1)
            pos_avg_hz_lstm = torch.cat((pos_avg_hz_lstm,es_pr_feat_unsqueeze),1)
        print(pos_avg_hz_lstm[0][0])
        
        final_preds = []
        for index in range(self.output_dim):
            mask = [True] * self.output_dim
            mask[index] = False
            non_target_rep = pos_avg_hz_lstm[:, mask]
            print("non_target_rep:",non_target_rep.shape)
            target_rep = pos_avg_hz_lstm[:, index:index+1]
            print("target_rep:",target_rep.shape)
                
            target_rep_t = target_rep.transpose(0,1).to(torch.float32)
            non_target_rep_t = non_target_rep.transpose(0,1).to(torch.float32)
            
            att_output, _ = self.att_attention_list[index](query=target_rep_t,key=non_target_rep_t,value=non_target_rep_t)
            att_output = att_output.transpose(0,1)
            print("att_output:",att_output.shape)
            attention_concat = torch.cat([target_rep.to(torch.float32), att_output], dim=-1)
            print("attention_concat:",attention_concat.shape)
            attention_concat = self.flatten_list[index](attention_concat)
            final_pred = torch.sigmoid(
                self.final_dense_list[index](attention_concat))
            
            final_preds.append(final_pred)
            

        y = torch.cat(final_preds, dim=-1)
        print("y:",y.shape)
        return y
