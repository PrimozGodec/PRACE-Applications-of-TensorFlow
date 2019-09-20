import tensorflow as tf
import numpy as np

# Import matrix with shape (2000, 2000).
R = np.dot(np.random.rand(2000, 10), np.random.rand(10, 2000))
R /= np.linalg.norm(R)
#R = np.loadtxt("matrix.csv", delimiter=',')

# Initial guess for the two factors.
scale = 0.01
G = tf.Variable(scale*np.random.rand(2000, 10))
H = tf.Variable(scale*np.random.rand(10, 2000))

# Choose a gradient based optimizer.
optimizer = tf.keras.optimizers.Adam()

# Perform gradient descent.
for i in range(200):
    with tf.GradientTape() as tape:
        tape.watch([G, H])
        absG = tf.abs(G)
        absH = tf.abs(H)
        dR = R - tf.matmul(absG, absH)
        loss = tf.reduce_sum(tf.square(dR))
    print(i, loss.numpy())
    dG, dH = tape.gradient(loss, [G, H])
    optimizer.apply_gradients([[dG, G], [dH, H]])
