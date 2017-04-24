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

USERS_FILE        = "users.csv"
ITEMS_FILE        = "items.csv"
INTERACTIONS_FILE = "interactions.csv"
TARGET_USERS      = "targetUsers.csv"
TARGET_ITEMS      = "targetItems.csv"
OUT_FILE          = "out_nn_no_filter.csv"
FEATURE_DIM       = 6
TH                = -10
N_CLASS           = 5

# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)


'''
1) Parse the challenge data, exclude all impressions
   Exclude all impressions
'''
with tf.device('/gpu:0'):
    (header_users, users) = select(USERS_FILE, lambda x: True, build_user, lambda x: int(x[0]))
    (header_items, items) = select(ITEMS_FILE, lambda x: True, build_item, lambda x: int(x[0]))

    #transfer data into ndarray
    X_val = np.zeros([1, FEATURE_DIM])
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
            X_val = np.concatenate((X_val, np.array([x])), axis=0)
            #print(X_val.shape)
            #Y_prob_val = np.append(Y_prob_val, getScore(inter_types, u_id, users))
            #Y_val = np.append(Y_val, getIntScore(inter_types))
            y = getMultiIntScore(inter_types)
            Y_val = np.concatenate((Y_val, y))


    train_x = X_val[1:]
    Y_val = Y_val.reshape(-1, N_CLASS)
    #print(Y_val[1:10,:])
    train_y = Y_val[1:]
    #train_y = Y_val[1:]

    split_size = int(train_x.shape[0] * 0.7)

    train_x, val_x = train_x[:split_size], train_x[split_size:]
    train_y, val_y = train_y[:split_size], train_y[split_size:]

    ### set all variables

    # number of neurons in each layer
    input_num_units = 6
    hidden_num_units = 50
    output_num_units = 5


    # define placeholders
    x = tf.placeholder(tf.float32, [None, input_num_units])
    y = tf.placeholder(tf.float32, [None, output_num_units])

    # set remaining variables
    epochs = 5
    batch_size = 128
    learning_rate = 0.01

    ### define weights and biases of the neural network (refer this article if you don't understand the terminologies)

    weights = {
        'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
        'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
    }

    biases = {
        'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
        'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
    }

    hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)

    output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
    #output_layer = tf.sigmoid(output_layer)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_layer, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.initialize_all_variables()
    '''
    2) Build recsys
    '''


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
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

                avg_cost += c / total_batch


            print("Epoch:", (epoch + 1), "loss =", "{0:.5f}".format(float(avg_cost)))

        print("\nTraining complete!")
        # find predictions on val set
        # one correct
        pred_temp = tf.losses.mean_squared_error(output_layer, y)
        val_loss = tf.reduce_mean(tf.cast(pred_temp, "float"))
        print("Validation Loss:", val_loss.eval({x: val_x.reshape(-1, input_num_units), y: val_y})/100)

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
                    #if xx[0] > 0:
                    f = xx
                    data += [f]
                    ids  += [u]

                if len(data) > 0:
                    data = np.array(data)
                    test_y = np.zeros((data.shape[0], output_num_units))
                    '''
                    test_prediction = tf.nn.softmax(tf.nn.relu(tf.matmul(
                            tf.nn.relu(tf.add(tf.matmul(data, weights['hidden']), biases['hidden']))) + bout)))
                    '''

                    #print(type(data))
                    #print(type(test_y))
                    #print(type(output_layer))
                    prediction = tf.sigmoid(output_layer)
                    #prediction = tf.cast(pred, "float")
                    #ypred = prediction.eval({x: data})
                    classification = sess.run(prediction, feed_dict={x: data})
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


