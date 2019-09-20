import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

num_of_ast = 30
mean_ast_speed = 0.4
universe_rad = 10
ast_rad = 0.2
angular_velocity = 1
reaction_time = 0.1

batch_size = 1024
gd_steps = 1000


# Environment evolution - in numpy.

def replace_ast_with_new_one(x, y, vx, vy, i, j):
    # Random position.
    theta = np.random.rand()*2*np.pi
    x_new = universe_rad*np.cos(theta)
    y_new = universe_rad*np.sin(theta)
    # Random velocity.
    zeta = np.random.rand()*2*np.pi
    speed = (np.random.rand()+0.5)*mean_ast_speed
    vx_new = speed*np.cos(zeta)
    vy_new = speed*np.sin(zeta)
    # Replace position i, j.
    x[i, j] = x_new
    y[i, j] = y_new
    vx[i, j] = vx_new
    vy[i, j] = vy_new
    return x, y, vx, vy

def initial_state():
    x = np.zeros([batch_size, num_of_ast])
    y = np.zeros([batch_size, num_of_ast])
    vx = np.zeros([batch_size, num_of_ast])
    vy = np.zeros([batch_size, num_of_ast])
    for i in range(batch_size):
        for j in range(num_of_ast):
            x, y, vx, vy = replace_ast_with_new_one(x, y, vx, vy, i, j)
            x *= np.random.rand()
            y *= np.random.rand()
    return x, y, vx, vy

def update_ast(x, y, vx, vy, dt, d):
    # Move all asteroids dt forward in time.
    x += vx*dt.numpy()
    y += vy*dt.numpy()
    # Delete asteroids that were hit or flew out of our universe.
    dist = d.numpy()
    for i in range(batch_size):
        for j in range(num_of_ast):
            if dist[i, j] < ast_rad:
                replace_ast_with_new_one(x, y, vx, vy, i, j)
            elif np.sqrt(np.square(x[i, j]) + np.square(y[i, j])) > universe_rad:
                replace_ast_with_new_one(x, y, vx, vy, i, j)
    return x, y, vx, vy


# Model action of the model and its effectiveness - in tensorflow.

network = Sequential([
            Dense(input_dim=4*num_of_ast+1, units=2*num_of_ast, activation='relu'),
            Dense(input_dim=4*num_of_ast+1, units=2*num_of_ast, activation='relu'),
            Dense(1, activation='tanh')
            ])

def model(x, y, vx, vy, current_phi):
    state = np.concatenate([x, y, vx, vy, current_phi], axis=1)
    return network(state)

def dist(phi, x, y, vx, vy, dt): # phi and dt are tf.Variable.
    px = tf.cos(phi)
    py = tf.sin(phi)
    sx = vy - py
    sy = px - vx
    st = vx*py - vy*px
    s = tf.sqrt(tf.square(sx)+tf.square(sy)+tf.square(st))
    d = tf.abs((sx*x + sy*y + st*dt)/s)
    return d

def new_shot(x, y, vx, vy, current_phi):
    new_phi = model(x, y, vx, vy, current_phi)
    dt = angular_velocity * tf.abs(new_phi - current_phi) + reaction_time
    d = dist(new_phi, x, y, vx, vy, dt)
    punishment = tf.reduce_mean(tf.reduce_min(d, axis=1))
    update_ast(x, y, vx, vy, dt, d)
    return punishment, new_phi.numpy(), dt.numpy()


# Gradient descent.

optimizer = tf.keras.optimizers.Adam()

x, y, vx, vy = initial_state()
#plt.ion()
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.xlim([-10, 10])
#plt.ylim([-10, 10])
#line, = ax.plot(x[0], y[0], 'o')
current_phi = np.zeros([batch_size, 1])
for i in range(gd_steps):
    with tf.GradientTape() as tape:
        tape.watch(network.trainable_variables)
        punishment, current_phi, dt = new_shot(x, y, vx, vx, current_phi)
    grad = tape.gradient(punishment, network.trainable_variables)
    optimizer.apply_gradients(zip(grad, network.trainable_variables))
    print(punishment.numpy())
    #line.set_xdata(x[0])
    #line.set_ydata(y[0])
    #fig.canvas.draw()
    #fig.canvas.flush_events()
