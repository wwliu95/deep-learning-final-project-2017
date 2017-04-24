import numpy as np

def getScore(interaction_types, user_id, users):
    '''
    Parameters:
    interaction_types: a set of interaction types
                       for a user-item pair.
    user_id: user id
    users: {uid: userObj,...} dict that contains all users
    Return:
    score of a user-item pair
    '''
    score = 0
    for interType in interaction_types:
        if interType == 1:
            temp = 1
        elif interType == 2 or interType == 3:
            temp = 5
        elif interType == 4:
            temp = -10
        elif interType == 5:
            temp = 20
        weight = 2 if users[user_id].is_premium else 1
        score += (weight * temp)
    return score


def getIntScore(interaction_types):
    '''
    Parameters:
    interaction_types: a set of interaction types
                       for a user-item pair.
    user_id: user id
    users: {uid: userObj,...} dict that contains all users
    Return:
    score of a user-item pair
    '''
    if 4 in interaction_types:
        return 0
    else:
        return 1

def getMultiIntScore(interaction_types):
    label = np.zeros(6)
    #print(type(interaction_types))
    label[list(interaction_types)] = 1
    return label[1:]



def getFeat(user, item):
    def title_match():
        if float(len(set(user.title).intersection(set(item.title)))) > 0:
            return 1.0
        else:
            return 0.0
        #return float(len(set(user.title).intersection(set(item.title))))


    def clevel_match():
        if user.clevel == item.clevel:
            return 1.0
        else:
            return 0.0

    def indus_match():
        if user.indus == item.indus:
            return 1.0
        else:
            return 0.0

    def discipline_match():
        if user.disc == item.disc:
            return 1.0#2
        else:
            return 0.0

    def country_match():
        if user.country == item.country:
            return 1.0
        else:
            return 0.0

    def region_match():
        if user.region == item.region:
            return 1.0
        else:
            return 0.0


    return [
        title_match(), clevel_match(), indus_match(),
        discipline_match(), country_match(), region_match()
    ]


def batch_creator(batch_size, dataset_length, train_x, train_y, input_num_units, rng):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)

    batch_x = train_x[[batch_mask]].reshape(-1, input_num_units)
    batch_y = train_y[[batch_mask]]

    return batch_x, batch_y