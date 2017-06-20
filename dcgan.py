from discriminator_generator import discriminator, decoder

import math
import numpy as np 
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf 
from scipy.misc import imsave
import sys
import os

def concat_elu(inputs):
	return tf.nn.elu(tf.concat([-inputs, inputs],3))

class GAN():
	def __init__(self, hidden_size, batch_size, learning_rate):
		self.batch_size = batch_size
		self.hidden_size = hidden_size

		self.input_tensor = tf.placeholder(tf.float32, [None,28*28])

		with arg_scope([layers.conv2d, layers.conv2d_transpose],
						activation_fn = concat_elu,
						normalizer_fn = layers.batch_norm,
						normalizer_params={'scale':True}):
			with tf.variable_scope("model"):
				D1 = discriminator(self.input_tensor)
				D_params_num = len(tf.trainable_variables())
				G = decoder(tf.random_normal([batch_size, hidden_size]))
				self.sampled_tensor = G
			with tf.variable_scope('model', reuse=True):
				D2 = discriminator(G)

			self.D_loss = self.__get_discriminator_loss(D1,D2)
			self.G_loss = self.__get_generator_loss(D2)
			params = tf.trainable_variables()
			D_params = params[:D_params_num]
			G_params = params[D_params_num:]

			global_step = tf.contrib.framework.get_or_create_global_step()



			self.train_discriminator = layers.optimize_loss(
				self.D_loss, global_step, learning_rate/10., 'Adam', variables=D_params, update_ops=[])
			self.train_generator = layers.optimize_loss(
				self.G_loss, global_step, learning_rate, 'Adam', variables=G_params, update_ops=[])


			self.saver = tf.train.Saver()
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())

	def __get_discriminator_loss(self, D1, D2):

		# return losses.sigmoid_cross_entropy(D1, tf.ones(tf.shape(D1))) + \
		# 					losses.sigmoid_cross_entropy(D2, tf.zeros(tf.shape(D2)))
		d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits = D1, labels = tf.ones(tf.shape(D1)))
		d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits = D2, labels = tf.zeros(tf.shape(D2)))
		return tf.reduce_mean(d_loss_fake+d_loss_real)

	def __get_generator_loss(self, D2):
		# return losses.sigmoid_cross_entropy(D2, tf.ones(tf.shape(D2)))
		return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2, labels=tf.ones(tf.shape(D2))))

	def update_params(self, inputs):
		# d_loss_value = self.sess.run(self.train_discriminator, {self.input_tensor: inputs})

		# g_loss_value = self.sess.run(self.train_generator)

		# return g_loss_value

		_,d_loss_value = self.sess.run([self.train_discriminator, self.D_loss], {
			self.input_tensor:inputs})
		_,g_loss_value = self.sess.run([self.train_generator, self.G_loss])
		_,g_loss_value = self.sess.run([self.train_generator, self.G_loss])
		return g_loss_value, d_loss_value

	def train(self, config):
		self.data_X  = self.load_mnist()

		self.load(config.checkpoint_dir)

		for ep in range(config.epoch):
			total_batch = len(self.data_X)//self.batch_size
			train_loss = 0.0
			for b in range(total_batch):
				input_x = self.data_X[b*self.batch_size: (b+1)*self.batch_size]
				# g_loss = self.update_params(input_x)
				# train_loss += g_loss
				# g_loss = train_loss/(b*self.batch_size)
				# sys.stdout.write("\r {}/{} epoch {}/{} batch, g_loss:{:04f}"
				# 	.format(ep,config.epoch, b, total_batch, g_loss))
				g_loss,d_loss = self.update_params(input_x)
				sys.stdout.write("\r {}/{} epoch {}/{} batch, g_loss:{:.4f}, d_loss:{:.4f}".format(ep,config.epoch, b, total_batch, g_loss, d_loss))

				sys.stdout.flush()

				if(ep*total_batch+b)%100==0:
					samples = self.sess.run(self.sampled_tensor)
					# print(sampled_images.shape,"\n")
					samples = np.reshape(samples,[samples.shape[0],28,28,1])

					self.save_image(samples,'./{}/train_{}_{}.png'.format(config.sample_dir, ep, b))
					global_step = tf.contrib.framework.get_or_create_global_step()
					self.save(config.checkpoint_dir, global_step)





	def load_mnist(self):
		f = np.load('../mnist.npz')
		x_train,y_train,x_test,y_test = f['x_train'],f['y_train'],f['x_test'],f['y_test']

		X = np.concatenate((x_train, x_test),axis=0)
		X = np.reshape(X,[X.shape[0],-1])
		return X/255.

	def save_image(self, images, path):
		# img_height, img_width, channel = images.shape[1:]
		# print (images.shape)
		manifold_h = int(np.ceil(np.sqrt(images.shape[0])))
		manifold_w = int(np.floor(np.sqrt(images.shape[0])))
		shape = [manifold_h, manifold_w]

		images = np.squeeze(images)
		height,width = images.shape[1:]

		ret = np.zeros((shape[0]*height, shape[1]*width))

		for i,img in enumerate(images):
			h_idx = int(i/shape[0])
			w_idx = int(i%shape[1])
			# print (h_idx*height, w_idx*width)
			ret[h_idx*height:(h_idx+1)*height, w_idx*width:(w_idx+1)*width] = img

		imsave(path, ret)

	def save(self, checkpoint_dir, step):
		model_name = "DCGAN.model"
		# checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)


		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		# checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0
