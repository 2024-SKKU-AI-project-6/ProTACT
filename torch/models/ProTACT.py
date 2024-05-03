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
        embedding_dim = configs.EMBEDDING_DIM
        dropout_prob = configs.DROPOUT
        filters= configs.CNN_FILTERS
        kernel_size = configs.CNN_KERNEL_SIZE
        lstm_units = configs.LSTM_UNITS
        num_heads = num_heads
        
        # Essay Representation
        self.essay_pos_embedding = nn.Embedding(pos_vocab_size, embedding_dim, padding_idx=0)
        self.essay_pos_dropout = nn.Dropout(dropout_prob)
        self.essay_pos_conv = nn.Conv1d(embedding_dim, filters, kernel_size)
        self.essay_pos_attention = Attention(filters)
        
        self.essay_linquistic = nn.Linear(linguistic_feature_count, filters)
        self.essay_readability = nn.Linear(readability_feature_count, filters)
        
        self.essay_pos_MA = nn.ModuleList([MultiHeadAttention(filters, num_heads) for _ in range(output_dim)])
        self.essay_pos_MA_LSTM = nn.ModuleList([nn.LSTM(filters, lstm_units) for _ in range(output_dim)])
        self.easay_pos_avg_MA_LSTM = nn.ModuleList([Attention(lstm_units) for _ in range(output_dim)])
        
        pass
    
    
    def correlation_coefficient(self, x, y):
        mask_value = -0.
        mask_x = (x != mask_value).type(torch.float32)
        mask_y = (y != mask_value).type(torch.float32)
        mask = mask_x*mask_y
        x_masked = x * mask
        y_masked = y * mask
        
        x_masked = (x_masked - (torch.sum(x_masked)/torch.sum(mask)))*mask
        y_masked = (y_masked - (torch.sum(y_masked)/torch.sum(mask)))*mask
        
        r_num = torch.sum(x_masked*y_masked)
        r_den = torch.sqrt(torch.sum(torch.square(x_masked)) * torch.sum(torch.square(y_masked)))
        
        r = 0.
        r = torch.where(r_den > 0, r_num / r_den, r+0)
        return r
    
    
    def cosine_sim(self, x, y):
        mask_value = 0.
        mask_x = (x != mask_value).type(torch.float32)
        mask_y = (y != mask_value).type(torch.float32)
        mask = mask_x*mask_y
        x_masked = x * mask
        y_masked = y * mask
        
        x_norm = F.normalize(x_masked, p=2, dim=0) * mask
        y_norm = F.normalize(y_masked, p=2, dim=0) * mask
        
        cos_sim = torch.sum(torch.mul(x_norm, y_norm))
        return cos_sim
    
    
    def trait_sim_loss_function(self, real, pred):
        mask_value = -1
        mask = (real != mask_value).type(torch.float32)
        
        real_trans = (real*mask).t()
        pred_trans = (pred*mask).t()
        
        sim_loss = 0.0
        cnt = 0.0
        ts_loss = 0.
        trait_num = 9
        print("trait num: ",trait_num)
        
        for i in range(1, trait_num):
            for j in range(i+1, trait_num):
                corr = self.correlation_coefficient(real_trans[i], real_trans[j])
                sim_loss = torch.where(corr>=0.7,  sim_loss + (1 - self.cosine_sim(pred_trans[i], pred_trans[j])), sim_loss)
                cnt = torch.where(corr >= 0.7, cnt + 1, cnt)
        
        loss = torch.where(cnt>0, sim_loss/cnt, loss)
        return loss
    
    
    def mse_loss_function(self, real, pred):
        mask_value = -1
        mask = (real != mask_value).type(torch.float32)
        mse = nn.MSELoss(reduction="none")
        return mse(real*mask, real*pred).mean()
    
    
    def loss_function(self, real, pred):
        alpha = 0.7
        mse_loss = self.mse_loss_function(real, pred)
        ts_loss = self.trait_sim_loss_function(real, pred)
        loss = alpha*mse_loss + (1-alpha)*ts_loss
        return loss
    
    
    def forward(self):
        pass
        
        