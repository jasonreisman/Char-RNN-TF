import argparse
import os
import tensorflow as tf
import time

import dataset
import network

def ensure_checkpoint_dir(savedir):
	if not os.path.exists(savedir):
		os.makedirs(savedir)

def get_checkpoint_path(savedir, config):
	ensure_checkpoint_dir(savedir)
	chkpt_path = os.path.join(savedir, config.checkpoint_name)
	return chkpt_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input file')
    parser.add_argument('--nhidden', type=int, default=128, help='Number of hidden RNN neurons')
    parser.add_argument('--nlayers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--batchsize', type=int, default=50, help='Batch size')
    parser.add_argument('--seqlen', type=int, default=50, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--savedir', type=str, default='checkpoints', help='Directory where checkpoints are saved')
    parser.add_argument('--savefreq', type=int, default=100, help='Number of batches between checkpoint creation')
    parser.add_argument('--gradclip', type=float, default=5., help='Max gradient norm before clipping')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    return args

def main():
	args = parse_args()
	print 'Reading input'
	print '\t- Input file: %s' % (args.input)
	ds = dataset.Dataset(args.input, args.batchsize, args.seqlen)
	print 'Done reading input'

	print 'Building network'
	config = network.RNNConfig(ds.hash, args.nhidden, args.nlayers, ds.num_classes)
	rnn = network.RNN(config, args.batchsize, args.seqlen)
	chkpt_path = get_checkpoint_path(args.savedir, config)
	print '\t- Checkpoint path: %s' % (chkpt_path)
	print 'Done building network'

	print 'Building optimizer'
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(rnn.cost, tvars), args.gradclip)
	lr = tf.Variable(0.0, trainable=False)
	optimizer = tf.train.AdamOptimizer(lr)
	train = optimizer.apply_gradients(zip(grads, tvars))
	print 'Done building optimzer'

	print 'Initializing session'
	# Initializing the tensor flow variables
	init = tf.initialize_all_variables()
	with tf.Session() as sess:
		sess.run(init)
		sess.run(tf.assign(lr, args.lr))
		# restore previous state if possible
		saver = tf.train.Saver(tf.all_variables())
		if os.path.exists(chkpt_path):
			print '\t- Restoring graph from previous checkpoint: %s' % (chkpt_path)
			saver.restore(sess, chkpt_path)
		print 'Done initializing session'
		num_iters = 0
		for i in range(args.epochs):
			state = rnn.initial_state.eval()
			for j, (x, y) in enumerate(ds.get_batches()):
				t0 = time.time()
				feed = {rnn.inputs:x, rnn.targets:y, rnn.initial_state:state}
				cost, state, _ = sess.run((rnn.cost, rnn.final_state, train), feed_dict=feed)
				t1 = time.time()
				print '\t- Epoch %i, Iter %i, loss: %.2f, time: %.2f' % (i, j, cost, t1-t0)
				num_iters += 1
				if num_iters % args.savefreq == 0:
					print '\t- Saving graph to checkpoint: %s' % (chkpt_path)
					saver.save(sess, chkpt_path)
		print 'Exiting after %i epochs' % (args.epochs)

if __name__ == '__main__':
	main()
