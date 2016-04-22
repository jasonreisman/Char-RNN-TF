import argparse
import numpy as np
import os
import tensorflow as tf

import dataset
import network

# random seed affects np.random.choice below
np.random.seed(0)

def ensure_checkpoint_dir(args):
	if not os.path.exists(args.savedir):
		os.makedirs(args.savedir)

def get_checkpoint_path(args, config):
	ensure_checkpoint_dir(args)
	chkpt_path = os.path.join(args.savedir, config.checkpoint_name)
	return chkpt_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input file')
    parser.add_argument('--nhidden', type=int, default=128, help='Number of hidden RNN neurons')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--savedir', type=str, default='checkpoints', help='Directory where checkpoints are saved')
    parser.add_argument('--length', type=int, default=512, help='The number of characters that will be generated')
    parser.add_argument('--prime', type=str, default='The meaning of life is ', help='Text to prime the generator with')    
    args = parser.parse_args()
    return args

def main():
	args = parse_args()
	print 'Reading input'
	print '\t- Input file: %s' % (args.input)
	ds = dataset.Dataset(args.input, 50, 50)
	print 'Done reading input'

	print 'Building network'
	name = os.path.basename(args.input)
	config = network.RNNConfig(name, args.nhidden, args.nlayers, ds.num_classes)
	rnn = network.RNN(config, 1, 1)
	chkpt_path = get_checkpoint_path(args, config)
	print '\t- Checkpoint path: %s' % (chkpt_path)
	print 'Done building network'

	print 'Initializing session'
	# Initializing the tensor flow variables
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		saver = tf.train.Saver(tf.all_variables())
		if os.path.exists(chkpt_path):
			print '\t- Restoring graph from previous checkpoint: %s' % (chkpt_path)
			saver.restore(sess, chkpt_path)
		print 'Done initializing session'

		x = np.zeros((1,1))
		state = rnn.initial_state.eval()
		for cur in args.prime[:-1]:
			x[0,0] = ds.encode(cur)
			feed = {rnn.inputs:x, rnn.initial_state:state}
			probs, state = sess.run([rnn.probs, rnn.final_state], feed_dict=feed)
			symbol = np.random.choice(ds.num_classes, p=probs[0])

		generated = args.prime
		cur = args.prime[-1]
		for i in range(args.length):
			x[0,0] = ds.encode(cur)
			feed = {rnn.inputs:x, rnn.initial_state:state}
			probs, state = sess.run([rnn.probs, rnn.final_state], feed_dict=feed)
			symbol = np.random.choice(ds.num_classes, p=probs[0])
			next = ds.decode(symbol)
			generated += next
			cur = next

		print generated

if __name__ == '__main__':
	main()