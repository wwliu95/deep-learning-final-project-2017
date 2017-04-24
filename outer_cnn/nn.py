import os
import random
import itertools
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn

from model import *
#from recommendation_worker_nn import *
from parser_file import *
from utils import *

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["GLOG_minloglevel"] ="3"

print (" --- Recsys Challenge 2017  --- ")

USERS_FILE        = "../users.csv"
ITEMS_FILE        = "../items.csv"
INTERACTIONS_FILE = "../interactions.csv"
TARGET_USERS      = "../targetUsers.csv"
TARGET_ITEMS      = "../targetItems.csv"
OUT_FILE          = "out_cnn.csv"
FEATURE_DIM       = 6
TH                = -10
N_CLASS           = 5

### set all variables
seed = 128
rng = np.random.RandomState(seed)
input_num_units = 60
output_num_units = 5
dropout = 1  # Dropout, probability to keep units
epochs = 50
batch_size = 128
learning_rate = 0.01


'''
1) Parse the challenge data, exclude all impressions
   Exclude all impressions
'''
with tf.device('/gpu:0'):
    (header_users, users) = select(USERS_FILE, lambda x: True, build_user, lambda x: int(x[0]))
    (header_items, items) = select(ITEMS_FILE, lambda x: True, build_item, lambda x: int(x[0]))

    #transfer data into ndarray
    X_val = []
    #Y_prob_val = np.zeros([1, 5])
    Y_val = np.zeros(N_CLASS)
    ui_score = dict()
    cnt = 0
    for line in itertools.islice(open(INTERACTIONS_FILE), 100000):
        cnt += 1
        if is_header(line):
            header = process_header(line.strip().split('\t'))
        else:
            cmp = line.strip().split('\t')
            inter_type = int(cmp[header['interaction_type']])
            user_id = int(cmp[header['user_id']])
            item_id = int(cmp[header['item_id']])
            if inter_type != 0:
                ui_score[user_id] = ui_score.get(user_id, dict())
                ui_score[user_id][item_id] = ui_score[user_id].get(item_id, set())
                ui_score[user_id][item_id].add(inter_type)
        if cnt % 1000000 == 0:
            print("... reading line %d from file %s" % (cnt, INTERACTIONS_FILE))
    for u_id, jobs in ui_score.items():
        for j_id, inter_types in jobs.items():
            x = getFeat(users[u_id], items[j_id])
            X_val.append(x)
            y = getMultiIntScore(inter_types)
            Y_val = np.concatenate((Y_val, y))


    train_x = np.stack(X_val).reshape(-1, input_num_units)
    train_x = preproc(train_x)
    print(train_x.shape)
    Y_val = Y_val.reshape(-1, N_CLASS)
    #print(Y_val[1:10,:])
    train_y = Y_val[1:]
    print(train_y.shape)
    #train_y = Y_val[1:]

    split_size = int(train_x.shape[0] * 0.7)

    train_x, val_x = train_x[:split_size], train_x[split_size:]
    train_y, val_y = train_y[:split_size], train_y[split_size:]

    # define placeholders
    x = tf.placeholder(tf.float32, [None, input_num_units])
    y = tf.placeholder(tf.float32, [None, output_num_units])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


    # Create some wrappers for simplicity
    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')


    # Create model
    def conv_net(x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 10, 6, 1])

        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        print(weights['wd1'].get_shape().as_list()[0])
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out


    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])),#4*3*32
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([3*2*64, 84])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([84, output_num_units]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([84])),
        'out': tf.Variable(tf.random_normal([output_num_units]))
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)



    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    print('mark-1')
    with tf.Session() as sess:
        print('mark0')
        # create initialized variables
        sess.run(init)

        ### for each epoch, do:
        ###   for each batch, do:
        ###     create pre-processed batch
        ###     run optimizer by feeding batch
        ###     find cost and reiterate to minimize

        for epoch in range(epochs):
            avg_cost = 0
            total_batch = int(train_x.shape[0] / batch_size)
            for i in range(total_batch):
                batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], train_x, train_y, input_num_units, rng)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

                avg_cost += c / total_batch


            print("Epoch:", (epoch + 1), "loss =", "{0:.5f}".format(float(avg_cost)))

        print("\nTraining complete!")
        # find predictions on val set
        # one correct
        # pred_temp = tf.losses.mean_squared_error(pred, y)
        #val_loss = tf.reduce_mean(tf.cast(pred_temp, "float"))
        #v=val_loss.eval({x: val_x.reshape(-1, input_num_units), y: val_y, keep_prob: dropout})
        #print("Validation Loss:", float(v/1e10))

        print('mark2')

        '''
        4) Create target sets for items and users
        '''
        target_users = []
        for line in open(TARGET_USERS):
            target_users += [int(line.strip())]
        target_users = set(target_users)

        target_items = []
        for line in open(TARGET_ITEMS):
            target_items += [int(line.strip())]

        with open(OUT_FILE, 'w') as fp:
            pos = 0
            average_score = 0.0
            num_evaluated = 0.0
            for i in target_items:
                data = []
                ids  = []

                # build all (user, item) pair features based for this item
                for u in target_users:
                    xx = getFeat(users[u], items[i])
                    #f = x.features()
                    #data += [f]
                    #ids += [u]
                    if float(len(set(users[u].title).intersection(set(items[i].title)))) > 0:
                        f = preproc(xx)
                        data += [f]
                        ids  += [u]

                if len(data) > 0:
                    data = np.array(data).reshape(-1, input_num_units)
                    test_x = data.astype('float')
                    test_y = np.zeros((data.shape[0], output_num_units))
                    '''
                    test_prediction = tf.nn.softmax(tf.nn.relu(tf.matmul(
                            tf.nn.relu(tf.add(tf.matmul(data, weights['hidden']), biases['hidden']))) + bout)))
                    '''

                    #print(type(data))
                    #print(type(test_y))
                    #print(type(output_layer))
                    prediction = tf.sigmoid(pred)
                    #prediction = tf.cast(pred, "float")
                    #ypred = prediction.eval({x: data})
                    classification = sess.run(prediction, feed_dict={x: test_x, keep_prob: dropout})
                    ypred = classification[:,0] + 5*classification[:, 1] + 5*classification[:, 2] - 10*classification[:, 3] +20*classification[:, 4]
                    #print(classification.shape)
                    #print(ypred)
                    #ypred = classification[0]
                    #score need updated
                    user_ids = sorted(
                        [
                            (ids_j, ypred_j) for ypred_j, ids_j in zip(ypred, ids) if ypred_j > TH
                            ],
                        key=lambda x: -x[1]
                    )[0:99]

                    # write the results to file
                    if len(user_ids) > 0:
                        item_id = str(i) + "\t"
                        fp.write(item_id)
                        for j in range(0, len(user_ids)):
                            user_id = str(user_ids[j][0]) + ","
                            fp.write(user_id)
                        user_id = str(user_ids[-1][0]) + "\n"
                        fp.write(user_id)
                        fp.flush()

                        # Every 100 iterations print some stats
                if pos % 100 == 0:
                    percentageDown = str(pos / float(len(target_items)))
                    print(OUT_FILE + " " + percentageDown)
                pos += 1


