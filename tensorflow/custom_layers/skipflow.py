import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from scipy import stats



class NeuralTensorlayer(Layer):
	def __init__(self,output_dim,input_dim=None, **kwargs):
		self.output_dim=output_dim
		self.input_dim=input_dim
		if self.input_dim:
			kwargs['input_shape']=(self.input_dim,)
		super(NeuralTensorlayer,self).s__init__(**kwargs)

	def call(self,inputs,mask=None):
		e1=inputs[0]
		e2=inputs[1]
		batch_size=K.shape(e1)[0]
		k=self.output_dim

		feed_forward=K.dot(K.concatenate([e1,e2]),self.V)

		bilinear_tensor_products = [ K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1) ]

		for i in range(k)[1:]:
			btp=K.sum((e2*K.dot(e1,self.W[i]))+self.b,axis=1)
			bilinear_tensor_products.append(btp)

		result=K.tanh(K.reshape(K.concatenate(bilinear_tensor_products,axis=0),(batch_size,k))+feed_forward)

		return result

	def build(self,input_shape):
		mean=0.0
		std=1.0
		k=self.output_dim
		d=self.input_dim
		W_val=stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k,d,d))
		V_val=stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2*d,k))
		self.W=K.variable(W_val)
		self.V=K.variable(V_val)
		self.b=K.zeros((self.input_dim,))
		self._trainable_weights=[self.W,self.V,self.b]

	def compute_output_shape(self, input_shape):
		batch_size=input_shape[0][0]
		return(batch_size,self.output_dim)



class TemporalMeanPooling(Layer):
	def __init__(self, **kwargs):
		super(TemporalMeanPooling,self).__init__(**kwargs)
		self.supports_masking=True
		self.input_spec=InputSpec(ndim=3)

	def call(self,x,mask=None):
		if mask is None:
			mask=K.mean(K.ones_like(x),axis=-1)

		mask=K.cast(mask,K.floatx())
		return K.sum(x,axis=-2)/K.sum(mask,axis=-1,keepdims=True)

	def compute_mask(self,input,mask):
		return None

	def compute_output_shape(self,input_shape):
		return (input_shape[0],input_shape[2])



class SkipFlow(Layer):
    def __init__(self, **kwargs):
        super(SkipFlow, self).__init__(**kwargs)
        self.temporal_mean_pooling = TemporalMeanPooling()
        self.neural_tensor_layer = NeuralTensorlayer()
        
    def call(self, x):
        
        