import collections
import numpy as np
import os

class Dataset(object):
	_input_path = None
	_batch_size = None
	_seq_len = None
	_encoding = {}
	_decoding = {}
	_encoded_data = None
	_num_batches = -1

	def __init__(self, input_path, batch_size, seq_len):
		self._input_path = input_path
		self._batch_size = batch_size
		self._seq_len = seq_len
		self._load_and_create_data()
		self._create_batches()

	# JR: I don't think that this sorting by frequency stuff makes any difference
	# since the goal here seems to be that we map chars to some int identifier, 
	# not that there is any particular structure in the characters themselves.
	# If there is some advantage to assigning identifiers by frequencies, I don't see it yet
	def _load_and_create_data(self):
		input_path = self._input_path
		assert os.path.exists(input_path), "Input file %s doesn't exist :-(" % (input_path)
		with open(input_path, 'r') as f:
			raw_data = f.read()
		# get the frequency of each item in raw_data
		counter = collections.Counter(raw_data)
		# sort the frequencies in descending order
		sorted_descending = sorted(counter.items(), key=lambda x : -x[1])
		# convert pairs in sorted_descending into two lists (chars, freqs)
		chars, _ = zip(*sorted_descending)
		num_chars = len(chars)
		self._encoding = dict(zip(chars, range(num_chars)))
		# create a dict of int -> char
		self._decoding = dict(zip(range(num_chars), chars))
		# convert raw data from chars to symbols
		self._encoded_data = np.array(list(map(self.encode, raw_data)))

	def _create_batches(self):
		assert self._encoded_data is not None
		# figure out how many batches we can actually create
		items_per_batch = self._batch_size*self._seq_len + 1 # plus 1 to have final symbol in batch
		self._num_batches = int(len(self._encoded_data) / items_per_batch)
		assert self._num_batches > 0
		# resize data so its length is a perfect multiple of items_per_batch
		self._encoded_data = self._encoded_data[:self._num_batches * items_per_batch]

	def get_batches(self):
		items_per_batch = self._batch_size*self._seq_len
		for i in xrange(self._num_batches):
			x0 = self._encoded_data[i*items_per_batch:(i+1)*items_per_batch]
			y0 = self._encoded_data[i*items_per_batch+1:(i+1)*items_per_batch+1]
			x = x0.reshape([self._batch_size, -1])
			y = y0.reshape([self._batch_size, -1])
			assert x.shape == (self._batch_size, self._seq_len)
			assert y.shape == (self._batch_size, self._seq_len)			
			yield (x, y)

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def seq_len(self):
		return self._seq_len

	@property
	def num_classes(self):
		return len(self._encoding)

	def encode(self, char):
		assert char in self._encoding
		return self._encoding[char]

	def decode(self, symbol):
		assert symbol in self._decoding
		return self._decoding[symbol]

# class TestConfig(object):
# 	input_path = 'input.txt'
# 	batch_size = 4
# 	seq_len = 32

# if __name__=='__main__':
# 	config = TestConfig()
# 	reader = CharReader(config)
# 	for i, (x, y) in enumerate(reader.get_batches()):
# 		print '(%i, x) ------------------------' % (i)
# 		print ''.join(map(reader.decode, list(x.reshape(-1))))
# 		print '(%i, y) ------------------------' % (i)		
# 		print ''.join(map(reader.decode, list(y.reshape(-1))))


