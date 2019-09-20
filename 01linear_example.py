import tensorflow as tf
from matplotlib import pyplot as plt

print(tf.__version__)

"""
1. Define the model
"""


class LinearModel:

    def __init__(self):
        self.n = tf.Variable(15.0)
        self.k = tf.Variable(0.0)

    def __call__(self, x):
        return self.n * x + self.k


# just a model test
model = LinearModel()
y = model(1)
print(y)


"""
2. Define a loss function
"""


@tf.function
def loss(predicted_y, target_y):
    return tf.reduce_mean(tf.square(predicted_y - target_y))


"""
3. Generate training data.
"""

num_examples = 1000
training_x = tf.random.normal(shape=[num_examples])
noise = tf.random.normal(shape=[num_examples])
training_y = training_x * 3 + 10 + noise

# lets plot the data and current predictions
plt.scatter(training_x, training_y, c='b')
plt.scatter(training_x, model(training_x), c='r')
plt.show()


"""
4. Fit the model to training data.
"""


def training_step(model, x, y, learning_rate):
    """
    This function do one step of training
    """
    with tf.GradientTape() as t:
        current_loss = loss(model(x), y)
    dn, dk = t.gradient(current_loss, [model.n, model.k])
    model.n.assign_sub(learning_rate * dn)
    model.k.assign_sub(learning_rate * dk)


# train the model in 10 steps
model = LinearModel()
for _ in range(40):
    training_step(model, training_x, training_y, learning_rate=0.1)
    print(model.n.numpy(), model.k.numpy(), loss(model(training_x), training_y).numpy())


# plot predictions now
plt.scatter(training_x, training_y, c='b')
plt.scatter(training_x, model(training_x), c='r')
plt.show()
