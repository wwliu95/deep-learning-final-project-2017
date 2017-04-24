import os
import random
import itertools
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn

from model import *
from recommendation_worker_nn import *
from parser_file import *

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["GLOG_minloglevel"] ="3"

print (" --- Recsys Challenge 2017 Baseline --- ")

N_WORKERS         = 1
USERS_FILE        = "../users.csv"
ITEMS_FILE        = "../items.csv"
INTERACTIONS_FILE = "../interactions.csv"
TARGET_USERS      = "../targetUsers.csv"
TARGET_ITEMS      = "../targetItems.csv"

#initialization

'''
1) Parse the challenge data, exclude all impressions
   Exclude all impressions
'''
with tf.device('/cpu:0'):
    (header_users, users) = select(USERS_FILE, lambda x: True, build_user, lambda x: int(x[0]))
    (header_items, items) = select(ITEMS_FILE, lambda x: True, build_item, lambda x: int(x[0]))

    builder = InteractionBuilder(users, items)
    (header_interactions, interactions) = select(
        INTERACTIONS_FILE,
        lambda x: x[2] != '0',
        builder.build_interaction,
        lambda x: (int(x[0]), int(x[1]))
    )


    '''
    2) Build recsys
    cpus = multiprocessing.cpu_count()ing data
    '''
    data    = np.array([interactions[key].features() for key in interactions.keys()],dtype='float32')
    labels  = np.array([interactions[key].label() for key in interactions.keys()])


    def get_train_inputs():
        x = tf.constant(data)
        y = tf.constant(labels)
        return x, y


    print('mark-1')
    with tf.Session() as session:
        print('mark0')
        clf = tf.contrib.learn.DNNLinearCombinedClassifier(
            n_classes=2, linear_feature_columns=learn.infer_real_valued_columns_from_input(data),
            linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.05),
            dnn_feature_columns=learn.infer_real_valued_columns_from_input(data),dnn_hidden_units=[8, 8, 5],
            dnn_optimizer=tf.train.AdagradOptimizer(learning_rate=0.05)
            )
        clf.fit(input_fn=get_train_inputs, steps=500)

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


    filename = 'solution_Flying_Machine_t1.csv'
    classify_worker(target_items, target_users, items, users, filename, clf)

    #with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:
        #classify_worker(target_items, target_users, items, users, filename, clf1, clf2, clf3,clf)
    '''
    for i, u in itertools.product(target_items, target_users):
        data = []
        ids  = []
        x = Interaction(users[u], items[i], -1)
        f = x.features()
        if x.title_match() > 0:
            f = x.features()
            data += [f]
            ids  += [u]
        if data:
            dtest = np.array(data)
            print('mark3')
            print(time.time())
            ypred = clf.predict_classes(dtest)

 '''
