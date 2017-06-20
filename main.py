import math
import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
# from tensorflow.examples.tutorials.mnist import input_data

# from progressbar import ETA, Bar, Percentage, ProgressBar

# from vae import VAE
# from gan import GAN
# from discriminator_generator import dcgan
from dcgan import GAN

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("sample_dir", "samples_my", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("checkpoint_dir", "checkpoint_my", "checkpoint directory")
# flags.DEFINE_string("model", "gan", "gan or vae")
FLAGS = flags.FLAGS

if __name__ == "__main__":
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)

	model = GAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)
	model.train(FLAGS)
