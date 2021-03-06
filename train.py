import argparse
import os
import tensorflow as tf
import time

import dataset
import network

def ensure_dir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)

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
    parser.add_argument('--dropout', type=float, default=0, help='Probability of a neuron dropping out during training')    
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
	rnn = network.RNN(config, args.batchsize, args.seqlen, dropout=args.dropout)
	chkpt_path = config.get_checkpoint_path(args.savedir)
	print '\t- Checkpoint path: %s' % (chkpt_path)
	print 'Done building network'

	print 'Building optimizer'
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(rnn.loss, tvars), args.gradclip)
	lr = tf.Variable(0.0, trainable=False)
	optimizer = tf.train.AdamOptimizer(lr)
	train = optimizer.apply_gradients(zip(grads, tvars))
	print 'Done building optimzer'

	print 'Initializing session'
	# Initializing the tensor flow variables
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		sess.run(tf.assign(lr, args.lr))
		# restore previous state if possible
		saver = tf.train.Saver(tf.global_variables())
		if tf.train.checkpoint_exists(chkpt_path):
			print '\t- Restoring graph from previous checkpoint'
			saver.restore(sess, chkpt_path)
		print 'Done initializing session'
		num_iters = 0
		for i in range(args.epochs):
			rnn.reset_initial_state()
			state = rnn.initial_state
			for j, (x, y) in enumerate(ds.get_batches()):
				t0 = time.time()
				feed = {rnn.inputs:x, rnn.targets:y}
				loss, _ = sess.run((rnn.loss, train), feed_dict=feed)
				t1 = time.time()
				print '\t- Epoch %i, Iter %i, loss: %.2f, time: %.2f' % (i, j, loss, t1-t0)
				num_iters += 1
				if num_iters % args.savefreq == 0:
					print '\t- Saving graph to checkpoint: %s' % (chkpt_path)
					ensure_dir(args.savedir)
					saver.save(sess, chkpt_path)
			print '\t- Saving graph to checkpoint: %s' % (chkpt_path)
			ensure_dir(args.savedir)
			saver.save(sess, chkpt_path)
		print 'Exiting after %i epochs' % (args.epochs)

if __name__ == '__main__':
	main()
