import tensorflow as tf
from transformers import BertConfig, TFBertModel
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, TimeDistributed, Flatten, Concatenate
from tensorflow.keras.optimizers import RMSprop
import numpy as np

def build_ProTACT(pos_vocab_size, vocab_size, maxnum, maxlen, readability_feature_count,
                  linguistic_feature_count, configs, output_dim, num_heads, embedding_weights):
    embedding_dim = configs.EMBEDDING_DIM
    dropout_prob = configs.DROPOUT
    cnn_filters = configs.CNN_FILTERS
    cnn_kernel_size = configs.CNN_KERNEL_SIZE
    lstm_units = configs.LSTM_UNITS

    # BERT Config 설정
    bert_config = BertConfig(
        hidden_size=embedding_dim,
        num_hidden_layers=12,
        num_attention_heads=num_heads,
        intermediate_size=3072,
        max_position_embeddings=512,
        vocab_size=vocab_size,
        hidden_dropout_prob=dropout_prob,
        attention_probs_dropout_prob=dropout_prob,
    )

    ### 1. Essay Representation
    pos_input = Input(shape=(maxnum, maxlen), dtype='int32', name='pos_input')
    pos_embedding_layer = layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size,
                                           mask_zero=True, name='pos_x')
    pos_x = pos_embedding_layer(pos_input)
    pos_resh_W = Reshape((maxnum, maxlen, embedding_dim), name='pos_resh_W')(pos_x)
    pos_zcnn = TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='pos_zcnn')(pos_resh_W)
    pos_avg_zcnn = TimeDistributed(Attention(), name='pos_avg_zcnn')(pos_zcnn)

    linguistic_input = Input((linguistic_feature_count,), name='linguistic_input')
    readability_input = Input((readability_feature_count,), name='readability_input')

    # BERT 모델 초기화
    pos_MA_bert_model = TFBertModel(bert_config)
    pos_MA_list = [pos_MA_bert_model(pos_avg_zcnn).last_hidden_state for _ in range(output_dim)]
    pos_avg_MA_lstm_list = [Attention()(pos_hz_lstm) for pos_hz_lstm in pos_MA_list]

    ### 2. Prompt Representation
    prompt_word_input = Input(shape=(maxnum, maxlen), dtype='int32', name='prompt_word_input')
    prompt_embedding_layer = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size,
                                              weights=embedding_weights, mask_zero=True, name='prompt')
    prompt = prompt_embedding_layer(prompt_word_input)
    prompt_pos_input = Input(shape=(maxnum, maxlen), dtype='int32', name='prompt_pos_input')
    prompt_pos_embedding_layer = layers.Embedding(output_dim=embedding_dim, input_dim=pos_vocab_size,
                                                  mask_zero=True, name='pos_prompt')
    prompt_pos = prompt_pos_embedding_layer(prompt_pos_input)
    
    prompt_emb = layers.Add()([prompt, prompt_pos])
    prompt_resh_W = Reshape((maxnum, maxlen, embedding_dim), name='prompt_resh_W')(prompt_emb)
    prompt_zcnn = TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'), name='prompt_zcnn')(prompt_resh_W)
    prompt_avg_zcnn = TimeDistributed(Attention(), name='prompt_avg_zcnn')(prompt_zcnn)

    # BERT 모델 초기화
    prompt_MA_bert_model = TFBertModel(bert_config)
    prompt_MA_list = prompt_MA_bert_model(prompt_avg_zcnn).last_hidden_state
    prompt_avg_MA_lstm_list = Attention()(prompt_MA_list)
    
    query = prompt_avg_MA_lstm_list

    es_pr_MA_list = [pos_MA_bert_model(pos_avg_MA_lstm_list[i], attention_mask=None).last_hidden_state for i in range(output_dim)]
    es_pr_avg_lstm_list = [Attention()(pos_hz_lstm) for pos_hz_lstm in es_pr_MA_list]
    es_pr_feat_concat = [Concatenate()([rep, linguistic_input, readability_input]) for rep in es_pr_avg_lstm_list]
    pos_avg_hz_lstm = tf.concat([Reshape((1, lstm_units + linguistic_feature_count + readability_feature_count))(rep)
                                 for rep in es_pr_feat_concat], axis=-2)

    final_preds = []
    for index, rep in enumerate(range(output_dim)):
        mask = np.array([True for _ in range(output_dim)])
        mask[index] = False
        non_target_rep = tf.boolean_mask(pos_avg_hz_lstm, mask, axis=-2)
        target_rep = pos_avg_hz_lstm[:, index:index+1]
        att_attention = layers.Attention()([target_rep, non_target_rep])
        attention_concat = tf.concat([target_rep, att_attention], axis=-1)
        attention_concat = Flatten()(attention_concat)
        final_pred = Dense(units=1, activation='sigmoid')(attention_concat)
        final_preds.append(final_pred)

    y = Concatenate()([pred for pred in final_preds])

    model = Model(inputs=[pos_input, prompt_word_input, prompt_pos_input, linguistic_input, readability_input], outputs=y)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=RMSprop())

    return model

# 예제 사용
class Config:
    EMBEDDING_DIM = 768
    DROPOUT = 0.1
    CNN_FILTERS = 128
    CNN_KERNEL_SIZE = 3
    LSTM_UNITS = 128

configs = Config()
pos_vocab = ['example1', 'example2']  # 예시 pos_vocab
word_vocab = ['word1', 'word2', 'word3']  # 예시 word_vocab
max_sentnum = 10  # 예시 값
max_sentlen = 50  # 예시 값
readability_feature_count = 5  # 예시 값
linguistic_feature_count = 5  # 예시 값
output_dim = 3  # 예시 값
num_heads = 12  # 예시 값
embedding_weights = None  # 예시 값

model = build_ProTACT(len(pos_vocab), len(word_vocab), max_sentnum, max_sentlen,
                      readability_feature_count, linguistic_feature_count, configs,
                      output_dim, num_heads, embedding_weights)
model.summary()