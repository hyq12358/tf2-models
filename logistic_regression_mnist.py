import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.utils import shuffle


class LogisticRegression(Model):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.out = Dense(10)

    def call(self, x, training=None, mask=None):
        return self.out(x)


loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=0.001)
model = LogisticRegression()

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
	x_train = (x_train.reshape(-1, 784) / 255).astype(np.float32)
	x_test = (x_test.reshape(-1, 784) / 255).astype(np.float32)

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