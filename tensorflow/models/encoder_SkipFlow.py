import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import keras
import tensorflow.keras.backend as K
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention
from custom_layers.skipflow import SkipFlow
from custom_layers.multiheadattention_pe import MultiHeadAttention_PE
from custom_layers.multiheadattention import MultiHeadAttention
from custom_layers.pos_encoding import PositionalEncoding


class MaskedSelection(layers.Layer):
    def __init__(self, **kwargs):
        super(MaskedSelection, self).__init__(**kwargs)

    def call(self, inputs):
        data, mask = inputs
        masked_data = tf.boolean_mask(data, mask, axis=-2)
        return masked_data

    def compute_output_shape(self, input_shape):
        # Assuming input_shape is a tuple (data_shape, mask_shape)
        return input_shape[0]
    
    

def correlation_coefficient(trait1, trait2):
    x = trait1
    y = trait2
    
    # maksing if either x or y is a masked value
    mask_value = -0.
    mask_x = K.cast(K.not_equal(x, mask_value), K.floatx())
    mask_y = K.cast(K.not_equal(y, mask_value), K.floatx())
    
    mask = mask_x * mask_y
    x_masked, y_masked = x * mask, y * mask
    
    mx = K.sum(x_masked) / K.sum(mask) # ignore the masked values when obtaining the mean
    my = K.sum(y_masked) / K.sum(mask) # ignore the masked values when obtaining the mean
    
    xm, ym = (x_masked-mx) * mask, (y_masked-my) * mask # maksing the masked values
    
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(K.square(xm)) * K.sum(K.square(ym)))
    r = 0.
    r = tf.cond(r_den > 0, lambda: r_num / (r_den), lambda: r+0)
    return r

def cosine_sim(trait1, trait2):
    x = trait1
    y = trait2
    
    mask_value = 0.
    mask_x = K.cast(K.not_equal(x, mask_value), K.floatx())
    mask_y = K.cast(K.not_equal(y, mask_value), K.floatx())
    
    mask = mask_x * mask_y
    x_masked, y_masked = x*mask, y*mask
    
    normalize_x = tf.nn.l2_normalize(x_masked,0) * mask # mask 값 반영     
    normalize_y = tf.nn.l2_normalize(y_masked,0) * mask # mask 값 반영
        
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_x, normalize_y))
    return cos_similarity
    
def trait_sim_loss(y_true, y_pred):
    mask_value = -1
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    
    # masking
    y_trans = tf.transpose(y_true * mask)
    y_pred_trans = tf.transpose(y_pred * mask)
    
    sim_loss = 0.0
    cnt = 0.0
    ts_loss = 0.
    #trait_num = y_true.shape[1]
    trait_num = 9
    print('trait num: ', trait_num)
    
    # start from idx 1, since we ignore the overall score 
    for i in range(1, trait_num):
        for j in range(i+1, trait_num):
            corr = correlation_coefficient(y_trans[i], y_trans[j])
            sim_loss = tf.cond(corr>=0.7, lambda: tf.add(sim_loss, 1-cosine_sim(y_pred_trans[i], y_pred_trans[j])), 
                            lambda: tf.add(sim_loss, 0))
            cnt = tf.cond(corr>=0.7, lambda: tf.add(cnt, 1), 
                            lambda: tf.add(cnt, 0))
    ts_loss = tf.cond(cnt > 0, lambda: sim_loss/cnt, lambda: ts_loss+0)
    return ts_loss
    
def masked_loss_function(y_true, y_pred):
    mask_value = -1
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    mse = keras.losses.MeanSquaredError()
    return mse(y_true * mask, y_pred * mask)

def total_loss(y_true, y_pred):
    alpha = 0.7
    mse_loss = masked_loss_function(y_true, y_pred)
    ts_loss = trait_sim_loss(y_true, y_pred)
    return alpha * mse_loss + (1-alpha) * ts_loss

def build_ProTACT(pos_vocab_size, vocab_size, maxnum, maxlen, readability_feature_count,
              linguistic_feature_count, configs, output_dim, num_heads, embedding_weights):
    embedding_dim = configs.EMBEDDING_DIM
    dropout_prob = configs.DROPOUT
    cnn_filters = configs.CNN_FILTERS
    cnn_kernel_size = configs.CNN_KERNEL_SIZE
    lstm_units = configs.LSTM_UNITS
    num_heads = num_heads
    
    ### 1. Essay Representation
    pos_input = layers.Input(shape=(maxnum*maxlen,), dtype='int32', name='pos_input')
    pos_x = layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size, input_length=maxnum*maxlen,
                             weights=None, mask_zero=True, name='pos_x')(pos_input)
    pos_x_maskedout = ZeroMaskedEntries(name='pos_x_maskedout')(pos_x)
    # positional encoding shape = (maxnum * maxlen) (embedding_dim)
    pos_positional_encoding_layer = PositionalEncoding(maxlen, embedding_dim,True)
    pos_positional_encoding = pos_positional_encoding_layer(pos_x_maskedout)
    pos_x_embedding = tf.keras.layers.Add()([pos_x_maskedout,pos_positional_encoding ])
    
    pos_drop_x = layers.Dropout(dropout_prob, name='pos_drop_x')(pos_x_embedding)
    pos_resh_W = layers.Reshape((maxnum, maxlen, embedding_dim), name='pos_resh_W')(pos_drop_x) # (97, 50, 50)
    # pos_zcnn = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='pos_zcnn')(pos_resh_W)
    # pos_avg_zcnn = layers.TimeDistributed(Attention(), name='pos_avg_zcnn')(pos_zcnn)
    
    ################# transforemr mean pooling/ #####################
    
    # Multi-head attention
    pos_multi_head_attention_layer = MultiHeadAttention(
        num_heads=2,
        embedding_dim= embedding_dim,
        dropout_rate=dropout_prob
    )
    pos_attention_output = layers.TimeDistributed(pos_multi_head_attention_layer,name="pos_attention_output")(pos_resh_W)
    pos_attention_output_norm = layers.TimeDistributed(layers.LayerNormalization(epsilon=1e-6),name="pos_attention_output_norm")(pos_resh_W + pos_attention_output)
    
    # Feed-forward network
    pos_ffn_output = layers.TimeDistributed(layers.Dense(cnn_filters, activation='relu'),name='pos_dense_1')(pos_attention_output_norm)
    pos_ffn_output = layers.TimeDistributed(layers.Dense(embedding_dim),name='pos_dense_2')(pos_ffn_output)
    pos_ffn_output = layers.TimeDistributed(layers.Dropout(dropout_prob))(pos_ffn_output)
    pos_ffn_output = layers.TimeDistributed(layers.LayerNormalization(epsilon=1e-6))(pos_attention_output + pos_ffn_output)
    pos_transformer_embedding = layers.TimeDistributed(layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),name='pos_transformer_embedding')(pos_ffn_output)
    ################# /transforemr mean pooling #####################

    linguistic_input = layers.Input((linguistic_feature_count,), name='linguistic_input')
    readability_input = layers.Input((readability_feature_count,), name='readability_input')

    pos_MA_list = [MultiHeadAttention(100,num_heads)(pos_transformer_embedding) for _ in range(output_dim)]
    pos_avg_MA_lstm_list = [SkipFlow(lstm_dim=lstm_units, model_type = "lstm", k = 4, maxlen=maxlen, eta=13, delta=50)(pos_MA) for pos_MA in pos_MA_list] 

    ### 2. Prompt Representation
    # word embedding
    prompt_word_input = layers.Input(shape=(maxnum*maxlen,), dtype='int32', name='prompt_word_input')
    prompt = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=maxnum*maxlen,
                             weights=embedding_weights, mask_zero=True, name='prompt')(prompt_word_input)
    prompt_maskedout = ZeroMaskedEntries(name='prompt_maskedout')(prompt)

    # pos embedding
    prompt_pos_input = layers.Input(shape=(maxnum*maxlen,), dtype='int32', name='prompt_pos_input')
    prompt_pos = layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size, input_length=maxnum*maxlen,
                             weights=None, mask_zero=True, name='pos_prompt')(prompt_pos_input)
    prompt_pos_maskedout = ZeroMaskedEntries(name='prompt_pos_maskedout')(prompt_pos) 
    
    prompt_positional_encoding_layer = PositionalEncoding(maxlen, embedding_dim,True)
    prompt_positional_encoding = prompt_positional_encoding_layer(prompt_pos_maskedout)
    
    # add word + pos embedding
    prompt_emb = tf.keras.layers.Add()([prompt_maskedout, prompt_pos_maskedout, prompt_positional_encoding ])
    
    prompt_drop_x = layers.Dropout(dropout_prob, name='prompt_drop_x')(prompt_emb)
    prompt_resh_W = layers.Reshape((maxnum, maxlen, embedding_dim), name='prompt_resh_W')(prompt_drop_x)
    prompt_multi_head_attention_layer = MultiHeadAttention(
        num_heads=2,
        embedding_dim= embedding_dim,
        dropout_rate=dropout_prob
    )
    prompt_attention_output = layers.TimeDistributed(prompt_multi_head_attention_layer,name="prompt_attention_output")(prompt_resh_W)
    prompt_attention_output_norm = layers.TimeDistributed(layers.LayerNormalization(epsilon=1e-6),name="prompt_attention_output_norm")(prompt_resh_W + prompt_attention_output)
    
    # Feed-forward network
    prompt_ffn_output = layers.TimeDistributed(layers.Dense(cnn_filters, activation='relu'),name='prompt_dense_1')(prompt_attention_output_norm)
    prompt_ffn_output = layers.TimeDistributed(layers.Dense(embedding_dim),name='prompt_dense_2')(prompt_ffn_output)
    prompt_ffn_output = layers.TimeDistributed(layers.Dropout(dropout_prob))(prompt_ffn_output)
    prompt_ffn_output = layers.TimeDistributed(layers.LayerNormalization(epsilon=1e-6))(prompt_attention_output + prompt_ffn_output)
    prompt_transformer_embedding = layers.TimeDistributed(layers.Lambda(lambda x: tf.reduce_mean(x, axis=1)),name='prompt_transformer_embedding')(prompt_ffn_output)
    ################# /transforemr mean pooling #####################
    
    prompt_MA_list = MultiHeadAttention(100, num_heads)(prompt_transformer_embedding)
    prompt_MA_lstm_list = layers.LSTM(lstm_units, return_sequences=True)(prompt_MA_list) 
    prompt_avg_MA_lstm_list = Attention()(prompt_MA_lstm_list)
    
    query = prompt_avg_MA_lstm_list

    es_pr_MA_list = [MultiHeadAttention_PE(100,num_heads)(pos_avg_MA_lstm_list[i], query) for i in range(output_dim)]
    es_pr_MA_lstm_list = [layers.LSTM(lstm_units, return_sequences=True)(pos_hz_MA) for pos_hz_MA in es_pr_MA_list]
    es_pr_avg_lstm_list = [Attention()(pos_hz_lstm) for pos_hz_lstm in es_pr_MA_lstm_list]
    es_pr_feat_concat = [layers.Concatenate()([rep, linguistic_input, readability_input]) # concatenate with non-prompt-specific features
                                 for rep in es_pr_avg_lstm_list]
    reshaped_layers = [
        layers.Reshape((1, lstm_units + linguistic_feature_count + readability_feature_count))(rep)
        for rep in es_pr_feat_concat
    ]
    pos_avg_hz_lstm = layers.Concatenate(axis=-2)(reshaped_layers)

    final_preds = []
    for index, rep in enumerate(range(output_dim)):
        mask = np.array([True for _ in range(output_dim)])
        mask[index] = False
        non_target_rep = MaskedSelection()([pos_avg_hz_lstm, mask])
        target_rep = pos_avg_hz_lstm[:, index:index+1]
        att_attention = layers.Attention()([target_rep, non_target_rep])
        attention_concat = layers.Concatenate(axis=-1)([target_rep, att_attention])
        attention_concat = layers.Flatten()(attention_concat)
        final_pred = layers.Dense(units=1, activation='sigmoid')(attention_concat)
        final_preds.append(final_pred)

    y = layers.Concatenate()([pred for pred in final_preds])

    model = keras.Model(inputs=[pos_input, prompt_word_input, prompt_pos_input, linguistic_input, readability_input], outputs=y)
    model.summary()
    model.compile(loss=total_loss, optimizer='rmsprop')

    return model