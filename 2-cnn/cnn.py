from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Import MNIST data
mnist = input_data.read_data_sets("../1-dnn/", one_hot=True)

# Training Parameters
num_steps = 200
batch_size = 128
display_step = 10
strides = 1
k = 2

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

#
# Store layers weight & bias
# The first three convolutional layer
w_c_1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))
w_c_2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
w_c_3 = tf.Variable(tf.random_normal([3, 3, 64, 128]))
b_c_1 = tf.Variable(tf.zeros([32]))
b_c_2 = tf.Variable(tf.zeros([64]))
b_c_3 = tf.Variable(tf.zeros([128]))

# The second three convolutional layer weights
w_c_4 = tf.Variable(tf.random_normal([3, 3, 128, 256]))
w_c_5 = tf.Variable(tf.random_normal([3, 3, 256, 512]))
w_c_6 = tf.Variable(tf.random_normal([3, 3, 512, 1024]))
b_c_4 = tf.Variable(tf.zeros([256]))
b_c_5 = tf.Variable(tf.zeros([512]))
b_c_6 = tf.Variable(tf.zeros([1024]))

# Fully connected weight
w_f_1 = tf.Variable(tf.random_normal([7 * 7 * 1024, 2048]))
w_f_2 = tf.Variable(tf.random_normal([2048, 1024]))
w_f_3 = tf.Variable(tf.random_normal([1024, 512]))
b_f_1 = tf.Variable(tf.zeros([2048]))
b_f_2 = tf.Variable(tf.zeros([1024]))
b_f_3 = tf.Variable(tf.zeros([512]))

# output layer weight
w_out = tf.Variable(tf.random_normal([512, num_classes]))
b_out = tf.Variable(tf.zeros([num_classes]))

#
# Define model
x = tf.reshape(X, shape=[-1, 28, 28, 1])
# first layer convolution
conv1 = tf.nn.conv2d(x, w_c_1, strides=[1, strides, strides, 1], padding='SAME') + b_c_1
conv1 = tf.nn.relu(conv1)

# second layer convolution
conv2 = tf.nn.conv2d(conv1, w_c_2, strides=[1, strides, strides, 1], padding='SAME') + b_c_2
conv2 = tf.nn.relu(conv2)

# third layer convolution
conv3 = tf.nn.conv2d(conv2, w_c_3, strides=[1, strides, strides, 1], padding='SAME') + b_c_3
conv3 = tf.nn.relu(conv3)

# first Max Pooling (down-sampling)
pool_1 = tf.nn.max_pool(conv3, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# fourth layer convolution
conv4 = tf.nn.conv2d(pool_1, w_c_4, strides=[1, strides, strides, 1], padding='SAME') + b_c_4
conv4 = tf.nn.relu(conv4)

# fifth layer convolution
conv5 = tf.nn.conv2d(conv4, w_c_5, strides=[1, strides, strides, 1], padding='SAME') + b_c_5
conv5 = tf.nn.relu(conv5)

# sixth layer convolution
conv6 = tf.nn.conv2d(conv5, w_c_6, strides=[1, strides, strides, 1], padding='SAME') + b_c_6
conv6 = tf.nn.relu(conv6)

# second Max Pooling (down-sampling)
pool_2 = tf.nn.max_pool(conv6, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# first Fully connected layer
# Reshape conv6 output to fit fully connected layer input
fc1 = tf.reshape(pool_2, [-1, w_f_1.get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, w_f_1), b_f_1)
fc1 = tf.nn.relu(fc1)
# Apply Dropout
fc1 = tf.nn.dropout(fc1, dropout)

# second Fully connected layer
fc2 = tf.add(tf.matmul(fc1, w_f_2), b_f_2)
fc2 = tf.nn.relu(fc2)
fc2 = tf.nn.dropout(fc2, dropout)

# Third Fully connected layer
fc3 = tf.add(tf.matmul(fc2, w_f_3), b_f_3)
fc3 = tf.nn.relu(fc3)
fc3 = tf.nn.dropout(fc3, dropout)

# Output, class prediction
logits = tf.add(tf.matmul(fc3, w_out), b_out)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Please don't change these.
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2

# Start training
with tf.Session(config=config) as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                        Y: mnist.test.labels[:256],
                                        keep_prob: 1.0}))
