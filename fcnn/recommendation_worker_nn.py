


from model import *
import numpy as np
import tensorflow as tf


#TH = 0.8
#TH = 0.0
#TH = 0.6
TH = -10
def classify_worker(item_ids, target_users, items, users, output_file, clf):
#def classify_worker(item_ids, target_users, items, users, output_file, sub_model1, sub_model2, sub_model3, model):
    with open(output_file, 'w') as fp:
        pos = 0
        average_score = 0.0
        num_evaluated = 0.0
        for i in item_ids:
            data = []
            ids  = []

            # build all (user, item) pair features based for this item
            for u in target_users:
                x = Interaction(users[u], items[i], -1)
                f = x.features()
                #data += [f]
                #ids += [u]
                if x.title_match() > 0:
                    f = x.features()
                    data += [f]
                    ids  += [u]

            if len(data) > 0:
                predict = tf.argmax(output_layer, 1)
                pred = predict.eval({x: test_x.reshape(-1, input_num_units)})

                # use all items with a score above the given threshold and sort the result
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
                try:
                    score = str(average_score / num_evaluated)
                except ZeroDivisionError:
                    score = 0
                percentageDown = str(pos / float(len(item_ids)))
                print(output_file + " " + percentageDown + " " + str(score))
            pos += 1
