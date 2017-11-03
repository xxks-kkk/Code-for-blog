# feedforward_example.py

import tensorflow as tf
import numpy as np

# Example of training a feedforward network with one hidden layer to solve XOR.
if __name__=="__main__":
    # MAKE THE DATA
    # Synthetic data for XOR: y = x0 XOR x1
    train_xs = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]])
    train_ys = np.array([0, 1, 1, 1, 1, 0])
    # Define some constants
    # Inputs are of size 2
    feat_vec_size = 2
    # Let's use 10 hidden units
    embedding_size = 10
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    fx = tf.placeholder(tf.float32, feat_vec_size)
    # Other initializers like tf.random_normal_initializer are possible too
    V = tf.get_variable("V", [embedding_size, feat_vec_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    # Can use other nonlinearities: tf.nn.relu, tf.tanh, etc.
    z = tf.sigmoid(tf.tensordot(V, fx, 1))
    W = tf.get_variable("W", [num_classes, embedding_size])
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    # This is the actual prediction -- not used for training but used for inference
    one_best = tf.argmax(probs)

    # Input for the gold label so we can compute the loss
    label = tf.placeholder(tf.int32, 1)
    # Convert a value-based representation (e.g., [2]) into the one-hot representation ([0, 0, 1])
    # Because label is a tensor of dimension one, the one-hot is actually [[0, 0, 1]], so
    # we need to flatten it.
    label_onehot = tf.reshape(tf.one_hot(label, num_classes), shape=[num_classes])
    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    # TRAINING ALGORITHM CUSTOMIZATION
    # Decay the learning rate by a factor of 0.99 every 10 gradient steps (for larger datasets you'll want a slower
    # weight decay schedule
    decay_steps = 10
    learning_rate_decay_factor = 0.99
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.1
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(lr)
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 100
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_xs)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: train_xs[ex_idx],
                                                                                  label: np.array([train_ys[ex_idx]])})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
            print ("Loss for iteration " + repr(i) + ": " + repr(loss_this_iter))
        # Evaluate on the train set
        train_correct = 0
        for ex_idx in xrange(0, len(train_xs)):
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                  feed_dict={fx: train_xs[ex_idx]})
            if (train_ys[ex_idx] == pred_this_instance):
                train_correct += 1
            print ("Example " + repr(train_xs[ex_idx]) + "; gold = " + repr(train_ys[ex_idx]) + "; pred = " +\
                  repr(pred_this_instance) + " with probs " + repr(probs_this_instance))
            print ("  Hidden layer activations for this example: " + repr(z_this_instance))
        print (repr(train_correct) + "/" + repr(len(train_ys)) + " correct after training")
