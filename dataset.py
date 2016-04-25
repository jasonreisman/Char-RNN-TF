import collections
import hashlib
import json
import numpy as np
import os

class Dataset(object):
	_input_path = None
	_batch_size = None
	_seq_len = None
	_meta = None
	_encoding = {}
	_decoding = {}
	_num_batches = -1

	def __init__(self, input_path, batch_size, seq_len):
		self._input_path = input_path
		self._batch_size = batch_size
		self._seq_len = seq_len
		# get metadata (length, hash, char set) for the input file
		# load a cached copy if possible, otherwise create one
		self._load_or_create_metadata()
		assert Dataset._metadata_intact(self._meta)
		# create a dict of char -> int, and int -> char
		chars = self._meta['chars']
		num_chars = len(chars)
		self._encoding = dict(zip(chars, range(num_chars)))
		self._decoding = dict(zip(range(num_chars), chars))
		# figure out how many batches we can actually create
		data_length = self._meta['length']
		items_per_batch = self._batch_size*self._seq_len
		self._num_batches = int(data_length / items_per_batch) - 1
			# minus 1 to ensure that we have a final symbol in each batch
			# that matches the first symbol in the next batch
		assert self._num_batches > 0

	def _load_or_create_metadata(self):
		meta_path = self._input_path + ".meta"
		# if metadata file exists, load it
		if os.path.exists(meta_path):
			try:
				with open(meta_path, 'r') as f:
					meta = json.load(f)
				if Dataset._metadata_intact(meta):
					self._meta = meta
					return
			except:
				pass
		# no metadata file exists, create one
		self._create_metadata()

	@staticmethod
	def _metadata_intact(meta):
		if meta is None:
			return False
		if 'length' not in meta:
			return False
		if 'hash' not in meta:
			return False
		if 'chars' not in meta:
			return False
		return True		

	def _create_metadata(self):
		sha1 = hashlib.sha1()
		counter = None
		with open(self._input_path, 'r') as f:
			# read file in buffer size increments
			# and compute hash and char frequencies
			buf = None
			buf_size = 4096
			num_chars = 0
			while buf != "":
				buf = f.read(buf_size)
				num_chars += len(buf)
				sha1.update(buf)
				c = collections.Counter(buf)
				counter = counter + c if counter is not None else c
		# save number of chars in meta
		self._meta = {}
		self._meta['length'] = num_chars
		# save hash in meta
		self._meta['hash'] = sha1.hexdigest()
		# Sort the frequencies in descending order
		# (Strictly speaking this sort isn't necessary, 
		# since we're only trying to write out the list of characters, but hey)
		sorted_descending = sorted(counter.items(), key=lambda x : -x[1])
		# convert pairs in sorted_descending into two lists (chars, freqs)
		chars, _ = zip(*sorted_descending)
		self._meta['chars'] = chars
		# output the meta file
		with open(meta_path, 'w') as f:
			json.dump(self._meta, f)

	def get_batches(self):
		items_per_batch = self._batch_size*self._seq_len
		for i in xrange(self._num_batches):
			# read the next batch from the input file
			with open(self._input_path, 'r') as f:
				f.seek(i*items_per_batch)
				raw_data = f.read(items_per_batch + 1)
				assert len(raw_data) == items_per_batch + 1
			encoded_data = np.array(list(map(self.encode, raw_data)))
			x0 = encoded_data[:-1]
			y0 = encoded_data[1:]
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
	def hash(self):
		return self._meta['hash']

	@property
	def num_classes(self):
		return len(self._encoding)

	def encode(self, char):
		assert char in self._encoding
		return self._encoding[char]

	def decode(self, symbol):
		assert symbol in self._decoding
		return self._decoding[symbol]
