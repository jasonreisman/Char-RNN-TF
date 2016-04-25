import numpy as np
import os
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

class RNNConfig:
	name = None
	cell_type = 'LSTM'
	num_hidden = -1
	num_layers = -1
	num_classes = -1

	def __init__(self, name, num_hidden, num_layers, num_classes):
		self.name = name
		self.num_hidden = num_hidden
		self.num_layers = num_layers
		self.num_classes = num_classes

	def get_checkpoint_path(self, dir):
		chkpt_path = os.path.join(dir, self._get_checkpoint_name())
		return chkpt_path

	def _get_checkpoint_name(self):
		chkpt_name = '%s-%s-%i-%i-%i.chkpt' % (self.name, self.cell_type, self.num_hidden, self.num_layers, self.num_classes)
		return chkpt_name

class RNN(object):
	# members
	_config = None
	_batch_size = None
	_seq_len = None
	_input_ids = None
	_target_ids = None
	_stack = None
	_state0 = None
	_state1 = None
	_probs = None
	_cost = None

	def __init__(self, config, batch_size, seq_len, **kwargs):
		assert config.name is not None
		assert config.num_hidden > 0
		assert config.num_layers > 0
		assert config.num_classes > 0
		self._config = config
		self._batch_size = batch_size
		self._seq_len = seq_len
		dropout = kwargs.get('dropout', 0)
		assert 0 <= dropout < 1
		self._build_network(dropout)

	def _build_network(self, dropout):
		# Legend for tensor shapes below:
		# 	B := batch size
		# 	C := number of classes
		# 	H := number of hidden units (aka layer size)
		# 	S := sequence length

		# keep a reference to _config to make code below simpler
		config = self._config

		# Create size BxS input and target placeholder tensors
		# These will be filled in with actual values at session runtime
		data_dims = [self._batch_size, self._seq_len]
		self._input_ids = tf.placeholder(tf.int32, data_dims)
		self._target_ids = tf.placeholder(tf.int64, data_dims)

		# Create an embedding tensor to represent integer inputs into H dimensions
		# This must be done on the CPU, according to:
		# https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/examples/tutorials/word2vec/word2vec_basic.py#L143
		# (Ops and variables pinned to the CPU because of missing GPU implementation)
		with tf.device("/cpu:0"):
			# embeddings is a CxH tensor
			embeddings = tf.get_variable('embeddings', [config.num_classes, config.num_hidden])
			# embedded is a BxSxH tensor
			embedded = tf.nn.embedding_lookup(embeddings, self._input_ids)
			# sequences is a list of length S containing Bx1xH tensors
			sequences = tf.split(1, self._seq_len, embedded)
			# perform a "squeeze" on each item in the sequence list 
			# inputs is a list of length S containing BxH tensors
			inputs = [tf.squeeze(seq, [1]) for seq in sequences]
		
		# create LSTM cell and stack
		cell = rnn_cell.BasicLSTMCell(config.num_hidden)
		if dropout > 0:
			keep_prob = 1 - dropout
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
		self._stack = rnn_cell.MultiRNNCell([cell]*config.num_layers)
		self._state0 = self._stack.zero_state(self._batch_size, tf.float32)

		# Pump the inputs through the RNN layers
		# outputs is a list of length S containing BxH tensors
		outputs, self._state1 = rnn.rnn(self._stack, inputs, initial_state=self._state0)
		assert len(outputs) == self._seq_len
		#assert outputs[0].get_shape() == (self._batch_size, config.num_hidden), outputs[0].get_shape()

		# Softmax weight tensor is HxC
		W_soft = tf.get_variable('W_soft', [config.num_hidden, config.num_classes])
		# Softmax bias tensor is Cx1
		b_soft = tf.get_variable('b_soft', [config.num_classes])

		# Reshape the output so that we can use it with the softmax weights and bias:
		# 	- concat makes list into a BxSH tensor,
		# 	- reshape converts the BxSH tensor into a BSxH tensor
		output = tf.reshape(tf.concat(1, outputs), [-1, config.num_hidden])
		#assert output.get_shape() == (self._batch_size*self._seq_len, config.num_hidden), output.get_shape()

		# logits is a (BSxH).(HxC) + 1xC = BSxC + 1xC = BSxC tensor
		logits = tf.nn.xw_plus_b(output, W_soft, b_soft)
		#assert logits.get_shape() == (self._batch_size*self._seq_len, config.num_classes), logits.get_shape()

		# probs is a BSxC tensor, with entry (i,j) containing the probability that batch i is class j
		self._probs = tf.nn.softmax(logits)
		#assert self._probs.get_shape() == (self._batch_size*self._seq_len, config.num_classes), self._probs.get_shape()

		# targets is a BSx1 tensor
		targets = tf.reshape(self._target_ids, [self._batch_size*self._seq_len])
		# cross_entropy is a BSx1 tensor
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)
		#assert cross_entropy.get_shape() == (self._batch_size*self._seq_len)
		
		# cost is a scalar containing the mean of cross_entropy losses
		self._cost = tf.reduce_mean(cross_entropy)

	def reset_initial_state(self):
		self._state0 = self._stack.zero_state(self._batch_size, tf.float32)

	@property
	def config(self):
		return _config

	@property
	def inputs(self):
		return self._input_ids

	@inputs.setter
	def inputs(self, input_ids):
		self._input_ids = input_ids

	@property
	def targets(self):
		return self._target_ids

	@targets.setter
	def targets(self, target_ids):
		self._target_ids = target_ids

	@property
	def initial_state(self):
		return self._state0

	@initial_state.setter
	def initial_state(self, state0):
		self._state0 = state0

	@property
	def final_state(self):
		return self._state1

	@property
	def probs(self):
		return self._probs

	@property
	def cost(self):
		return self._cost

def test_instantiation():
	config = RNNConfig('Test', 512, 2, 64)
	rnn = RNN(config, 50, 50)

if __name__=='__main__':
	test_instantiation()