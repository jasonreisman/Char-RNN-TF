# Char-RNN-TF

A miniature clone of char-rnn written in TensorFlow (TF), based on Andrej Karpathy's wonderful blog post, [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), and [associated code](https://github.com/karpathy/char-rnn).  This was my first exposure to TF, so I tried to document the code (particularly `network.py`) in an effort to clarify aspects of TF's API that weren't clear to me when I first started reading TF code.

## Training

char-rnn-tff trains on input that lives in a single text file.  The simplest possible is invocation for training a network on a particular text file is
```
./train.py --input </path/to/your/input.txt>
```
which will train a two-layer network of 128 nuerons in each layer on the specified input file.

#### Network Width/Depth

The number of neurons in the hidden RNN layers can be controlled by supplying a `--nhidden` argument to `train.py`, and the number of layers can be controlled with the `--nlayers` argument.  For example, to train a 3 layer network with 512 neurons per layer:
```
./train.py --input </path/to/your/input.txt> --nhidden 512 --nlayers 3
```

#### Checkpoints

Checkpoint files, which output the state of the computation graph to disk (so it can be rehydrated later for additional training or string generation), are saved every `savefreq` batches.  The default frequency is to output a checkpoint after every 100 batches, but this can be changed by providing a `--savefreq` argument on the command line.  

The directory where checkpoints are written to can be specified with the `--savedir`.  The default directory is `./checkpoints`.

A unique checkpoint will be created for each combination of input file and network params.  For example, given an input file, the checkpoint for the 2 layer, 128 nueron network can be uniquely identified separately from the network trained on the same input file but with 3 layers of 512 neurons.

## Generation

After training for a while it's time to start generating some text.  The simplest possible invocation for generating text based upon a training file is
```
./generate.py --input </path/to/your/input.txt>
```
which will look up the checkpoint for the default network configuration (2 layers of 128 neurons each) and use that to generate a string of characters.  Just like with training, you can specify different network widths and depths with the `--nhidden` and `--nlayers` arguments, respectively.

The length of the generated string is controlled by the `--length` argument.  The default is 512.

You can also supply a string of text to "prime" the generator with using the `--prime` parameter.`  The default is, "The meaning of life is ".

