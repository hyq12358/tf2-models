import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.utils import shuffle


class LeNet(Model):
	def __init__(self):
		super(LeNet, self).__init__()
		self.conv1 = Conv2D(filters=6, kernel_size=(5,5), activation='relu')
		self.pool1 = MaxPooling2D(pool_size=(2,2), strides=2, padding='SAME')
		self.conv2 = Conv2D(filters=16, kernel_size=(5,5), activation='relu')
		self.pool2 = MaxPooling2D(pool_size=(2,2), strides=2, padding='SAME')
		self.flat = Flatten()
		self.fc1 = Dense(120, activation='relu')
		self.fc2 = Dense(84, activation='relu')
		self.out = Dense(10, activation='softmax')

	def call(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.flat(x)
		x = self.fc1(x)
		x = self.fc2(x)
		return self.out(x)


loss = tf.losses.SparseCategoricalCrossentropy()
optimizer = tf.optimizers.Adam(learning_rate=0.001)
model = LeNet()

train_loss = tf.metrics.Mean()
train_acc = tf.metrics.SparseCategoricalAccuracy()
test_loss = tf.metrics.Mean()
test_acc = tf.metrics.SparseCategoricalAccuracy()


if __name__ == "__main__":
	np.random.seed(1234)
	tf.random.set_seed(1234)


	@tf.function
	def compute_loss(label, pred):
	    return loss(label, pred)

	@tf.function
	def train_step(x, y):
	    with tf.GradientTape() as tape:
	        pred = model(x)
	        loss = compute_loss(y, pred)
	    grads = tape.gradient(loss, model.trainable_variables)
	    optimizer.apply_gradients(zip(grads, model.trainable_variables))
	    train_loss(loss)
	    train_acc(y, pred)
	    return pred

	@tf.function
	def test_step(x, y):
	    pred = model(x)
	    loss = compute_loss(y, pred)
	    test_loss(loss)
	    test_acc(y, pred)
	    return pred

	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = (x_train.reshape(-1, 28,28,1) / 255).astype(np.float32)
	x_test = (x_test.reshape(-1, 28,28,1) / 255).astype(np.float32)

	epochs = 50
	batch_size = 64
	batches = x_train.shape[0]//batch_size

	for epoch in range(epochs):
	    x_train, y_train = shuffle(x_train, y_train, random_state=42)
	    for batch in range(batches):
	        start = batch*batch_size
	        end = start+batch_size
	        train_step(x_train[start:end], y_train[start:end])

	    if epoch%5 == 0:
	        pred = test_step(x_test, y_test)
	        print("Epoch: {}, Valid Loss: {:.3f}, Valid Acc:{:.3f}".format(
	            epoch, test_loss.result(), test_acc.result()
	        ))