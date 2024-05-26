import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_models as tfm
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
tf.io.gfile.listdir(gs_folder_bert)

tokenizer = tfm.nlp.layers.FastWordpieceBertTokenizer(
    vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
    lower_case=True)

max_seq_length = 50

packer = tfm.nlp.layers.BertPackInputs(
    seq_length=max_seq_length,
    special_tokens_dict = tokenizer.get_special_tokens_dict())

class BertInputProcessor(tf.keras.layers.Layer):
  def __init__(self, tokenizer, packer):
    super().__init__()
    self.tokenizer = tokenizer
    self.packer = packer

  def call(self, inputs):
    token = self.tokenizer(inputs)
    packed = self.packer([token])

    return packed
    

import json

bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
encoder_config = tfm.nlp.encoders.EncoderConfig({
    'type':'bert',
    'bert': config_dict
})

bert_encoder = tfm.nlp.encoders.build_encoder(encoder_config)



# layer names
'''
input_word_ids
word_embeddings
input_type_ids
position_embedding
type_embeddings
add
embeddings/layer_norm
dropout
input_mask
self_attention_mask
transformer/layer_0
transformer/layer_1
transformer/layer_2
transformer/layer_3
transformer/layer_4
transformer/layer_5
transformer/layer_6
transformer/layer_7
transformer/layer_8
transformer/layer_9
transformer/layer_10
transformer/layer_11
tf.__operators__.getitem
pooler_transform
'''