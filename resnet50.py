import numpy as np 
import tensorflow as tf 
from tensorflow.keras import Model, layers
from sklearn.utils import shuffle

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

class ResNet50(Model):
	def __init__(self, input_shape, output_dim):
		super(ResNet50, self).__init__()
		self.conv1 = layers.Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='SAME')
		self.bn1 = layers.BatchNormalization()
		self.relu1 = layers.Activation('relu')
		self.pool1 = layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='SAME')
		self.res_block1 = [ResBlock(channels_in=channels_in, channels_out=256) for channels_in in [64,256,256]]
		self.conv2 = layers.Conv2D(512, kernel_size=(1,1), strides=2, padding='SAME')
		self.res_block2 = [ResBlock(channels_in=512, channels_out=512) for _ in range(4)]
		self.conv3 = layers.Conv2D(1024, kernel_size=(1,1), strides=2, padding='SAME')
		self.res_block3 = [ResBlock(channels_in=1024, channels_out=1024) for _ in range(6)]
		self.conv4 = layers.Conv2D(2048, kernel_size=(1,1), strides=2, padding='SAME')
		self.res_block4 = [ResBlock(channels_in=2048, channels_out=2048) for _ in range(3)]
		self.avg_pool = layers.GlobalAveragePooling2D()
		self.fc = layers.Dense(1000, activation='relu')
		self.out = layers.Dense(output_dim, activation='softmax')

	def call(self, x, training=None, mask=None):
		x = self.conv1(x)
		x = self.bn1(x, training=training)
		x = self.relu1(x)
		x = self.pool1(x)
		for res_block in self.res_block1:
			x = res_block(x, training=training)
		x = self.conv2(x)
		for res_block in self.res_block2:
			x = res_block(x, training=training)
		x = self.conv3(x)
		for res_block in self.res_block3:
			x = res_block(x, training=training)
		x = self.conv4(x)
		for res_block in self.res_block4:
			x = res_block(x, training=training)
		x = self.avg_pool(x)
		x = self.fc(x)
		x = self.out(x)
		return x

class ResBlock(Model):
	def __init__(self, channels_in, channels_out):
		super(ResBlock, self).__init__()
		self.conv1 = layers.Conv2D(channels_out//4, kernel_size=(1,1), padding='SAME')
		self.bn1 = layers.BatchNormalization()
		self.relu1 = layers.Activation('relu')
		self.conv2 = layers.Conv2D(channels_out//4, kernel_size=(3,3), padding='SAME')
		self.bn2 = layers.BatchNormalization()
		self.relu2 = layers.Activation('relu')
		self.conv3 = layers.Conv2D(channels_out, kernel_size=(1,1), padding='SAME')
		self.bn3 = layers.BatchNormalization()
		self.short_cut = self._short_cut(channels_in, channels_out)
		self.add = layers.Add()
		self.relu3 = layers.Activation('relu')

	def call(self, x, training=None, mask=None):
		z = x
		x = self.conv1(x)
		x = self.bn1(x, training=training)
		x = self.relu1(x)
		x = self.conv2(x)
		x = self.bn2(x, training=training)
		x = self.relu2(x)
		x = self.conv3(x)
		x = self.bn3(x, training=training)
		z = self.short_cut(z)
		x = self.add([z,x])
		x = self.relu3(x)
		return x

	def _short_cut(self, channels_in, channels_out):
		if channels_in == channels_out:
			return lambda x:x
		else:
			return layers.Conv2D(channels_out, kernel_size=(1,1), padding='SAME')


if __name__ == '__main__':
	np.random.seed(1234)
	tf.random.set_seed(1234)

	@tf.function
	def compute_loss(true, pred):
		return criterion(true, pred)

	@tf.function
	def train_step(x, y):
		with tf.GradientTape() as tape:
			pred = model(x, training=True)
			loss = compute_loss(y, pred)
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
		train_loss(loss)
		train_acc(y, pred)
		return pred

	@tf.function
	def test_step(x, y):
		pred = model(x, training=False)
		loss = compute_loss(y, pred)
		test_loss(loss)
		test_acc(y, pred)
		return pred

	"""
	Load data
	"""
	mnist = tf.keras.datasets.fashion_mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
	x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype(np.float32)

	"""
	Build model
	"""
	model = ResNet50(input_shape=(28, 28, 1), output_dim=10)
	model.build(input_shape=(None, 28, 28, 1))

	criterion = tf.losses.SparseCategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam()

	epochs = 5
	batch_size = 64
	n_batches = x_train.shape[0]//batch_size

	train_loss = tf.keras.metrics.Mean()
	test_loss = tf.keras.metrics.Mean()
	train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
	test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

	for epoch in range(epochs):
		x_train, y_train = shuffle(x_train, y_train, random_state=42)
		for batch in range(n_batches):
			start = batch*batch_size
			end = start+batch_size
			train_step(x_train[start:end], y_train[start:end])
			if batch %10==0:
				print('Train Cost: {:.3f}, Train Acc: {:.3f}'.format(train_loss.result(), train_acc.result()))
				train_loss.reset_states()
				train_acc.reset_states()
	        	
		pred = test_step(x_test, y_test)
		print('Epoch: {}, Valid Cost: {:.3f}, Valid Acc: {:.3f}'.format(
		epoch+1,
		test_loss.result(),
		test_acc.result()
		))
		test_loss.reset_states()
		test_acc.reset_states()