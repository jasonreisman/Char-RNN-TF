import argparse
import numpy as np
import os
import tensorflow as tf

import dataset
import network

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input file')
    parser.add_argument('--nhidden', type=int, default=128, help='Number of hidden RNN neurons')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--savedir', type=str, default='checkpoints', help='Directory where checkpoints are saved')
    parser.add_argument('--length', type=int, default=512, help='The number of characters that will be generated')
    parser.add_argument('--prime', type=str, default='The meaning of life is ', help='Text to prime the generator with')    
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator to pick characters with')        
    args = parser.parse_args()
    return args

def main():
	args = parse_args()

	# random seed affects np.random.choice below
	np.random.seed(args.seed)

	print 'Reading input'
	print '\t- Input file: %s' % (args.input)
	ds = dataset.Dataset(args.input, 50, 50)
	print 'Done reading input'

	print 'Building network'
	config = network.RNNConfig(ds.hash, args.nhidden, args.nlayers, ds.num_classes)
	rnn = network.RNN(config, 1, 1)
	chkpt_path = config.get_checkpoint_path(args.savedir)
	print '\t- Checkpoint path: %s' % (chkpt_path)
	print 'Done building network'

	print 'Initializing session'
	# Initializing the tensor flow variables
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		saver = tf.train.Saver(tf.global_variables())
		if tf.train.checkpoint_exists(chkpt_path):
			print '\t- Restoring graph from checkpoint'
			saver.restore(sess, chkpt_path)
		print 'Done initializing session'

		x = np.zeros((1,1))
		for cur in args.prime[:-1]:
			x[0,0] = ds.encode(cur)
			feed = {rnn.inputs:x}
			probs = sess.run([rnn.probs], feed_dict=feed)
			symbol = np.random.choice(ds.num_classes, p=np.reshape(probs[0], ds.num_classes))

		generated = args.prime
		cur = args.prime[-1]
		for i in range(args.length):
			x[0,0] = ds.encode(cur)
			feed = {rnn.inputs:x}
			probs = sess.run([rnn.probs], feed_dict=feed)
			symbol = np.random.choice(np.arange(ds.num_classes), p=np.reshape(probs[0], ds.num_classes))
			next = ds.decode(symbol)
			generated += next
			cur = next

		print generated

if __name__ == '__main__':
	main()