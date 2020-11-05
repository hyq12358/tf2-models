import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers


class DenseNet121(object):
	def __init__(self, input_shape, output_dim, k=32, theta=.5):
		"""
		Arguments:
			input_shape: [width, height, channels]
			output_dim: num_classes
			k: growth rate
			theta: compression 			
		"""
		super(DenseNet121, self).__init__()
		self.k = k
		self.theta = theta
		x = layers.Input(shape=input_shape)
		h = layers.BatchNormalization()(x)
		h = layers.Activation('relu')(h)
		h = layers.Conv2D(filters=k, kernel_size=(7,7), strides=2, padding='SAME')(h)
		h = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='SAME')(h)
		h, n_channels = self._dense_block(h, 6)
		h, n_channels = self._transition(h, n_channels)
		h, n_channels = self._dense_block(h, 12)
		h, n_channels = self._transition(h, n_channels)
		h, n_channels = self._dense_block(h, 24)
		h, n_channels = self._transition(h, n_channels)
		h, n_channels = self._dense_block(h, 16)
		h = layers.GlobalAveragePooling2D()(h)
		h = layers.Dense(output_dim, activation='softmax')(h)
		self.model = Model(inputs=[x], outputs=[h])

	def _dense_block(self, x, n_blocks):
		h = x
		for i in range(n_blocks):
			prev = h
			h = layers.BatchNormalization()(h)
			h = layers.Activation('relu')(h)
			h = layers.Conv2D(128, kernel_size=(1,1))(h)
			h = layers.BatchNormalization()(h)
			h = layers.Activation('relu')(h)
			h = layers.Conv2D(filters=self.k, kernel_size=(3,3), padding='SAME')(h)
			h = layers.Concatenate(axis=-1)([prev,h])
		return 	h, h.shape[-1]

	def _transition(self, x, n_channels):
		h = layers.BatchNormalization()(x)
		h = layers.Activation('relu')(h)
		h = layers.Conv2D(int(n_channels*self.theta), kernel_size=(1,1))(h)
		h = layers.AveragePooling2D(pool_size=(2,2), strides=2)(h)
		return h, h.shape[-1]

	def __call__(self):
		return self.model


if __name__ == "__main__":
	np.random.seed(1234)
	tf.random.set_seed(1234)

	"""
	Load data
	"""
	cifar10 = tf.keras.datasets.cifar10
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	x_train, x_test = (x_train/255.0), (x_test/255.0)
	def preprocess(x, y):
		x = tf.cast(x/255.0, tf.float32)
		y = tf.cast(y, tf.int64)
		return x, y
	train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess).shuffle(50000).batch(64).prefetch(1)
	test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(64).prefetch(1)
	
	"""
	Compile model
	"""
	model = DenseNet121(input_shape=(32,32,3), output_dim=10)()
	model.compile(loss='sparse_categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])

	"""
	Training & Testing
	"""
	model.fit(train_db, epochs=10)
	model.evaluate(test_db)