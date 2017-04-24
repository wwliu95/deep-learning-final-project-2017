from model import *
from itertools import islice


def is_header(line):
    return "recsyschallenge" in line


def process_header(header):
    x = {}
    pos = 0
    for name in header:
        x[name.split(".")[1]] = pos
        pos += 1
    return x


def select(from_file, where, toObject, index):
    header = None
    data = {}
    i = 0
    if from_file == 'interactions.csv':
        g = islice(open(from_file), 10000)
    else:
        g = open(from_file)
    for line in g:
        if is_header(line):
            header = process_header(line.strip().split("\t"))
        else:
            cmp = line.strip().split("\t")
            if where(cmp):
                obj = toObject(cmp, header)
                if obj != None:
                    data[index(cmp)] = obj
        i += 1
        if i % 100000 == 0:
        #if i % 100 == 0:
            print("... reading line " + str(i) + " from file " + from_file)
    return (header, data)


def build_user(str_user, names):
    return User(
        [int(x) for x in str_user[names["jobroles"]].split(",") if len(x) > 0],
        int(str_user[names["career_level"]]),
        int(str_user[names["industry_id"]]),
        int(str_user[names["discipline_id"]]),
        str_user[names["country"]],
        str_user[names["region"]],
        int(str_user[names["experience_n_entries_class"]]),
        int(str_user[names["experience_years_experience"]]),
        int(str_user[names["experience_years_in_current"]]),
        int(str_user[names["edu_degree"]]),
        [int(x) for x in str_user[names["edu_fieldofstudies"]].split(",") if len(x) > 0],
        int(str_user[names["wtcj"]]),
        int(str_user[names["premium"]])
    )


def build_item(str_item, names):
    return Item(
        [int(x) for x in str_item[names["title"]].split(",") if len(x) > 0],
        int(str_item[names["career_level"]]),
        int(str_item[names["industry_id"]]),
        int(str_item[names["discipline_id"]]),
        str_item[names["country"]],
        str_item[names["region"]],
        int(str_item[names["employment"]]),
        int(str_item[names["is_payed"]])
    )

'''
class InteractionBuilder:
    def __init__(self, user_dict, item_dict):
        self.user_dict = user_dict
        self.item_dict = item_dict

    def build_interaction(self, str_inter, names):
        if int(str_inter[names['item_id']]) in self.item_dict and int(str_inter[names['user_id']]) in self.user_dict:
            return Interaction(
                self.user_dict[int(str_inter[names['user_id']])],
                self.item_dict[int(str_inter[names['item_id']])],
                int(str_inter[names["interaction_type"]])
            )
        else:
            return None
'''
