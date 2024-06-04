import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

# Lambda layer to tile the positional encoding
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, embedding_dim, isWord=False):
        super(PositionalEncoding, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.isWord = isWord

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        pos_encoding = positional_encoding(self.sequence_length, self.embedding_dim)
        pos_encoding = tf.squeeze(pos_encoding, axis=0)  # Shape (sequence_length, embedding_dim)
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # Shape (1, sequence_length, embedding_dim)
        pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1])  # Shape (batch_size, sequence_length, embedding_dim)
        if self.isWord:
          maxnum = tf.shape(inputs)[1]
          pos_encoding = tf.tile(pos_encoding,[1,maxnum//self.sequence_length , 1]) #  
        
          
        return pos_encoding




