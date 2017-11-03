# models.py

import tensorflow as tf
import numpy as np
import random
import datetime
from sentiment_data import *
import os

# We suppress the warning messages raised by the tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
def train_ffnn(train_exs, dev_exs, test_exs, word_vectors):
    train_xs = np.array([np.mean(word_vectors.vectors[ex.indexed_words,:],axis=0) for ex in train_exs])
    train_ys = np.array([ex.label for ex in train_exs])

    X_dev = np.array([np.mean(word_vectors.vectors[ex.indexed_words,:],axis=0) for ex in dev_exs])
    y_dev = np.array([ex.label for ex in dev_exs])

    X_test = np.array([np.mean(word_vectors.vectors[ex.indexed_words,:],axis=0) for ex in test_exs])
    y_test = np.array([ex.label for ex in test_exs])


    batch_size = 10
    feat_vec_size = train_xs.shape[1]
    embedding_size = 150
    num_classes = 2
    num_epochs = 50
    initial_learning_rate = 0.1
    decay_steps = 10
    learning_rate_decay_factor = 0.99

    graph = tf.Graph()
    with graph.as_default():
        tf_y_train = tf.placeholder(tf.int32, batch_size)  # Input for the gold label so we can compute the loss
        label_onehot = tf.one_hot(tf_y_train, num_classes)

        # We have no hidden layer at all
        if os.environ.get('FC_LAYER', 1) == '0':
            print "NUM HIDDEN LAYER = 0"
            with tf.name_scope('softmax'):
                tf_X_train = tf.placeholder(tf.float32, [batch_size, feat_vec_size])
                V_h1 = tf.get_variable("V_h1", [feat_vec_size, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
                probs = tf.nn.softmax(tf.tensordot(tf_X_train, V_h1, 1))
                one_best = tf.argmax(probs, axis=1)
        # We have 2 hidden layers
        elif os.environ.get('FC_LAYER', 1) == '2':
            print "NUM HIDDEN LAYER = 2"
            with tf.name_scope('h1') as scope:
                tf_X_train = tf.placeholder(tf.float32, [None, feat_vec_size])
                V_h1 = tf.get_variable("V_h1", [feat_vec_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
                z_h1 = tf.sigmoid(tf.tensordot(tf_X_train, V_h1, 1))# Can use other nonlinearities: tf.nn.relu, tf.tanh, etc.
                W_h1 = tf.get_variable("W_h1", [embedding_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
                h1 = tf.tensordot(z_h1, W_h1, 1)

            with tf.name_scope('h2') as scope:
                V_h2 = tf.get_variable("V_h2", [embedding_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
                W_h2 = tf.get_variable("W_h2", [embedding_size, num_classes])
                z_h2 = tf.sigmoid(tf.tensordot(h1, V_h2, 1))

            with tf.name_scope('softmax'):
                probs = tf.nn.softmax(tf.tensordot(z_h2, W_h2, 1))
                one_best = tf.argmax(probs, axis=1)
        # We have 1 hidden layer
        else:
            print "NUM HIDDEN LAYER = 1"
            with tf.name_scope('h1') as scope:
                tf_X_train = tf.placeholder(tf.float32, [None, feat_vec_size])
                V_h1 = tf.get_variable("V_h1", [feat_vec_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
                z_h1 = tf.sigmoid(tf.tensordot(tf_X_train, V_h1, 1))# Can use other nonlinearities: tf.nn.relu, tf.tanh, etc.
                W_h1 = tf.get_variable("W_h1", [embedding_size, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))

            with tf.name_scope('softmax'):
                probs = tf.nn.softmax(tf.tensordot(z_h1, W_h1, 1))
                one_best = tf.argmax(probs, axis=1)

        loss = tf.negative(tf.reduce_sum(tf.multiply(tf.log(probs), label_onehot)))

        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(initial_learning_rate,global_step,decay_steps,learning_rate_decay_factor, staircase=True)

        # Plug in any first-order method here! We'll use SGD with momentum
        if os.environ.get('OPT', 'SGD') == 'SGD':
            print "OPT = SGD"
            opt = tf.train.GradientDescentOptimizer(learning_rate)
            grads = opt.compute_gradients(loss)
            train_op = opt.apply_gradients(grads)
            # The above three lines can be replaced by the following line. Use the above three lines for experiment purpose.
            # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        elif os.environ.get('OPT', 'SGD') == 'ADAM':
            print "OPT = ADAM"
            opt = tf.train.AdamOptimizer(learning_rate)
            grads = opt.compute_gradients(loss)
            train_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()  # run this first to initialize variables
        arr_train = np.arange(len(train_xs))
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(num_epochs):
            np.random.shuffle(arr_train)
            loss_this_iter = 0
            for batch_idx in xrange(0, len(train_xs)/batch_size):
                ex_idx = arr_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
                feed_dict = {
                    tf_X_train: train_xs[ex_idx],
                    tf_y_train: train_ys[ex_idx]
                }
                [_, loss_this_batch] = sess.run([train_op, loss], feed_dict = feed_dict)
                step_idx += 1
                loss_this_iter += loss_this_batch
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)

        # Evaluate on the train set
        train_correct = 0
        for batch_idx in xrange(0, len(train_xs)/batch_size):
            ex_idx = arr_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            feed_dict = {
                tf_X_train: train_xs[ex_idx]
            }
            [pred_this_batch] = sess.run([one_best],feed_dict=feed_dict)
            train_correct += np.sum(np.equal(train_ys[ex_idx],pred_this_batch))
        print str(float(train_correct)/len(train_ys))[:7] + " correct on the training set"

        # Evaluate on the dev set
        train_correct = 0
        arr_dev = np.arange(len(X_dev))
        for batch_idx in xrange(0, len(X_dev)/batch_size):
            ex_idx = arr_dev[batch_idx*batch_size:(batch_idx+1)*batch_size]
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            feed_dict = {
                tf_X_train: X_dev[ex_idx]
            }
            [pred_this_batch] = sess.run([one_best],feed_dict=feed_dict)
            train_correct += np.sum(np.equal(y_dev[ex_idx],pred_this_batch))
        print str(float(train_correct)/len(y_dev))[:7] + " correct on the dev set"

        # Evaluate on the test set
        sentimentExamples = []
        train_correct = 0
        arr_test = np.arange(len(X_test))
        for batch_idx in xrange(0, len(X_test) / batch_size):
            ex_idx = arr_test[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
            # quantities from the running of the computation graph, namely the probabilities, prediction, and z
            feed_dict = {
                tf_X_train: X_test[ex_idx]
            }
            [pred_this_batch] = sess.run([one_best], feed_dict=feed_dict)
            for i in y_test[ex_idx]:
                sentimentExamples.append(SentimentExample(test_exs[i].indexed_words, pred_this_batch[i]))
    return sentimentExamples


# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
# NOTE: Two hidden layer FFNN that achieves good result
def train_ffnn2(train_exs, dev_exs, test_exs, word_vectors):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    dev_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    # Also store the sequence lengths -- this could be useful for training LSTMs
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])
    # Labels
    dev_labels = np.array([ex.label for ex in dev_exs])
    dev_xs = dev_mat
    dev_ys = dev_labels.reshape(-1, 1)

    train_xs = train_mat
    train_ys = train_labels_arr.reshape(-1, 1)
    # Define some constants
    embedding_size = 10
    num_classes = 2
    batch_size = 100
    vec_size = 300

    # DEFINING THE COMPUTATION GRAPH
    # Define the core neural network
    fx = tf.placeholder(tf.float32, [None, vec_size])
    # Other initializers like tf.random_normal_initializer are possible too
    W1 = tf.get_variable("W1", [vec_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b = tf.get_variable("b1", [embedding_size])
    # Can use other nonlinearities: tf.nn.relu, tf.tanh, etc.
    z = tf.sigmoid(tf.matmul(fx, W1) + b)
    W2 = tf.get_variable("W2", [embedding_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b2 = tf.get_variable("b2", [embedding_size])
    z = tf.nn.softmax(tf.matmul(z, W2) + b2)
    W3 = tf.get_variable("W3", [embedding_size, num_classes], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b3 = tf.get_variable("b3", [num_classes])
    probs = tf.nn.softmax(tf.matmul(z, W3) + b3)
    # This is the actual prediction -- not used for training but used for inference
    one_best = tf.reshape(tf.argmax(probs, axis=1), shape=[-1, 1])

    # Input for the gold label so we can compute the loss
    label = tf.placeholder(tf.int32, [None, 1])
    y_ = tf.reshape(tf.one_hot(label, num_classes), shape=[-1, num_classes])
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(probs), reduction_indices=[1]))

    decay_steps = 10
    learning_rate_decay_factor = 1.0
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks

    initial_learning_rate = 1.0
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)

    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    num_epochs = 1000
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
        iters = int(len(train_xs) / batch_size)

        embedded_train = []
        for i in range(len(train_xs)):
            tmp = []
            for word in train_xs[i]:
                tmp.append(word_vectors.get_train_embedding(int(word), add=False))
            embedded_train.append(tmp)
        train_xs = []
        print('data shape ', np.shape(embedded_train))
        print('reading word embeddings ...')
        for i in range(len(embedded_train)):
            train_xs.append(np.mean(embedded_train[i][:len(train_exs[i].indexed_words)], 0))
        print('input shape: ', np.shape(train_xs))

        embedded_dev = []
        for i in range(len(dev_xs)):
            tmp = []
            for word in dev_xs[i]:
                tmp.append(word_vectors.get_train_embedding(int(word), add=False))
            embedded_dev.append(tmp)
        dev_xs = []
        print('data shape ', np.shape(embedded_dev))
        print('reading word embeddings ...')
        for i in range(len(embedded_dev)):
            dev_xs.append(np.mean(embedded_dev[i][:len(dev_exs[i].indexed_words)], 0))
        print('input shape: ', np.shape(dev_xs))

        for i in range(num_epochs):
            loss_this_epoch = 0
            print('number of epoch: ', i)
            for iter in range(iters):
                [_, loss_this_batch, summary] = sess.run([train_step, loss, merged],
                                                            feed_dict={fx: train_xs[batch_size*iter:batch_size*(iter+1)],
                                                                       label: train_ys[batch_size*iter:batch_size*(iter+1)]})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_epoch += loss_this_batch
            print("Loss for epoch " + repr(i) + ": " + "{0:.2f}".format(loss_this_epoch))
            preds, _ = sess.run([one_best, probs], feed_dict={fx: train_xs})
            print('training accuracy: ', "{0:.2f}".format(np.mean(np.equal(preds, train_ys))))
            preds, _ = sess.run([one_best, probs], feed_dict={fx: dev_xs})
            print('dev accuracy: ', "{0:.2f}".format(np.mean(np.equal(preds, dev_ys))))
            print()


# Analogous to train_ffnn, but trains CNN.
# Here, I implement CNN for sentiment analysis
# Reference:
# 1. [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
def train_cnn(train_exs, dev_exs, test_exs, word_vectors):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    # Labels
    train_labels_arr = np.array([ex.label for ex in train_exs])

    train_xs = train_mat
    train_ys_2cols = np.array(list(zip(1-train_labels_arr, train_labels_arr)))

    X_dev = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    y_dev = np.array([ex.label for ex in dev_exs])
    y_dev_2cols = np.array(list(zip(1-y_dev, y_dev)))

    X_test = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in test_exs])
    y_test = np.array([ex.label for ex in test_exs])
    y_test_2cols = np.array(list(zip(1-y_test, y_test)))

    vocab_size = word_vectors.vectors.shape[0]
    embedding_size = word_vectors.vectors.shape[1]
    filter_sizes = [3,4,5]
    num_filters = 128
    l2_reg_lambda = 0.0
    dropout_keep_prob_train = 0.5
    dropout_keep_prob_dev = 1.0
    # number steps = number of sentences / batch_size
    batch_size = 64
    num_epochs = 20
    learning_rate = 1e-3

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=train_xs.shape[1],
                num_classes=train_ys_2cols.shape[1],
                vocab_size=vocab_size,
                embedding_size=embedding_size,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                activation_func='relu',
                l2_reg_lambda=l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0.0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cnn.loss, global_step=global_step)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # Use the pre-trained word embeddings
            sess.run(cnn.word_embeddings.assign(word_vectors.vectors))

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob_train
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def dev_step(x_batch, y_batch):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob_dev,
                }
                step, loss, accuracy, predictions = sess.run(
                    [global_step, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def test_step(x_batch, y_batch):
                """
                 Evaluates model on a test set and save the predictions
                 as a list of sentimentExamples as a return
                 """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob_dev,
                }
                predictions = sess.run(cnn.predictions, feed_dict)

                # Save the prediction result as a list of SentimentExample
                sentimentExamples = []
                for i in range(len(x_batch)):
                    sentimentExamples.append(SentimentExample(x_batch[i], predictions[i]))
                return sentimentExamples

            # Generate batches
            for epoch in range(num_epochs):
                batches = batch_iter(
                    list(zip(train_xs, train_ys_2cols)), batch_size=batch_size, num_epochs=num_epochs)
                # Training loop. For each batch...
                # num batches = (num training examples / batch_size)
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                print("\nEvaluation:")
                dev_step(X_dev, y_dev_2cols)
                print("")
            return test_step(X_test, y_test_2cols)



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 activation_func='relu',
                 l2_reg_lambda=0.0):
        # sequence_length  :  The length of sentence.
        # num_classes      :  Number of classes in the output layer, two in our case (positive and negative).
        # vocab_size       :  The size of our vocabulary. This is needed to define the size of our embedding layer,
        #                     which will have shape [vocabulary_size, embedding_size].
        # embedding_size   :  The dimensionality of our embeddings.
        # filter_sizes     :  The number of words we want our convolutional filters to cover. We will have num_filters for each size specified here.
        #                     For example, [3, 4, 5] means that we will have filters that slide over 3, 4 and 5 words respectively,
        #                     for a total of 3 * num_filters filters.
        # num_filters      :  The number of filters per filter size
        # activation_func  :  The activation function we use: "tanh", "relu"

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.word_embeddings = tf.get_variable(name="word_embeddings", shape=[vocab_size, embedding_size])
            self.embedded_chars = tf.nn.embedding_lookup(self.word_embeddings, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Implement convolution layer and 1-max pooling strategy
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv")
                # Apply nonlinearity
                if activation_func == 'tanh':
                    h = tf.nn.tanh(tf.nn.bias_add(conv, b), name = "tanh")
                elif activation_func == 'relu':
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# Same task but using LSTM instead
def train_lstm(train_exs, dev_exs, test_exs, word_vectors):
    seq_max_len = 60
    train_mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    train_labels = np.array([ex.label for ex in train_exs])

    train_xs = train_mat
    train_ys = train_labels.reshape(-1, 1)
    b = np.zeros((len(train_ys), 2), dtype=np.float32)
    train_ys = np.concatenate(train_ys, axis=0)
    b[np.arange(len(train_ys)), train_ys] = 1.0
    train_ys = b

    embedded_train = []
    for i in range(len(train_xs)):
        tmp = []
        for word in train_xs[i]:
            tmp.append(word_vectors.get_train_embedding(int(word), add=False))
        embedded_train.append(tmp)
    train_xs = embedded_train

    # Parameters
    learning_rate = 0.1
    batch_size = 128
    epochs = 100
    iters = int(len(train_xs)/batch_size)

    # Network Parameters
    seq_max_len = 60  # Sequence max length
    n_hidden = 128  # hidden layer num of features
    n_classes = 2  # linear sequence or not

    # tf Graph input
    x = tf.placeholder("float", [None, seq_max_len, 300])
    y = tf.placeholder("float", [None, n_classes])
    # A placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }


    def dynamicRNN(x, seqlen, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, axis=1)
        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                                    sequence_length=seqlen)

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        print(np.shape(outputs))
        outputs = tf.stack(outputs)
        print(np.shape(outputs))
        outputs = tf.transpose(outputs, [1, 0, 2])
        print(np.shape(outputs))

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

        # Linear activation, using outputs computed above
        return tf.matmul(outputs, weights['out']) + biases['out']


    pred = dynamicRNN(x, seqlen, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        for epoch in range(epochs):
            for iter in range(iters-1):
                batch_x, batch_y, batch_seqlen = [train_xs[iter*batch_size:min((iter+1)*batch_size, len(train_xs))],
                                                  train_ys[iter * batch_size:min((iter + 1) * batch_size, len(train_xs))],
                                                  train_seq_lens[iter * batch_size:min((iter + 1) * batch_size, len(train_xs))]]
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            # Calculate batch accuracy & loss
            acc, loss = sess.run([accuracy, cost], feed_dict={x: train_xs, y: train_ys, seqlen: train_seq_lens})
            print("Epoch " + str(epoch) + ", Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

        print("Optimization Finished!")