import numpy as np

class User:
    def __init__(self, title, clevel, indus, disc, country, region, n_entries, yr, es_yr, edu, edu_field, wtcj, is_premium):
        self.title = title
        self.clevel = clevel
        self.indus = indus
        self.disc = disc
        self.country = country
        self.region = region

        self.n_entries = n_entries
        self.yr = yr
        self.es_yr = es_yr
        self.edu = edu
        self.edu_field = edu_field
        self.wtcj = wtcj
        self.is_premium = is_premium


    def user_country(self):
        if self.country == 'de':
            return 1
        elif self.country == 'at':
            return 2
        elif self.country == 'ch':
            return 3
        else:
            return 0

    def edu_fieldofstudies(self):
        return int(len(self.edu_field))

    def user_title(self):
        return self.title

    def user_is_premium(self):
        if self.is_premium == 0:
            return False
        else:
            return True

    # design 10 features
    def features(self):
        return [
            self.clevel, self.indus, self.disc, self.user_country(),
            self.n_entries, self.yr, self.edu, self.edu_fieldofstudies(),
            self.wtcj, self.is_premium
        ]


class Item:
    def __init__(self, title, clevel, indus, disc, country, region, employ, is_paid):
        self.title = title
        self.clevel = clevel
        self.indus = indus
        self.disc = disc
        self.country = country
        self.region = region

        self.employ = employ
        self.is_paid = is_paid


    def item_country(self):
        if self.country == 'de':
            return 1
        elif self.country == 'at':
            return 2
        elif self.country == 'ch':
            return 3
        else:
            return 0


    def item_title(self):
        return self.title


    def item_is_paid(self):
        if self.is_paid == 0:
            return False
        else:
            return True

    #design 6 featuers
    def features(self):
        return [
            self.clevel, self.indus, self.disc, self.item_country(),
            self.employ, self.is_paid
        ]
