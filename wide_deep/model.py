'''
Modeling users, interactions and items from
the recsys challenge 2017.

by Daniel Kohlsdorf
'''

class User:

    def __init__(self, title, clevel, indus, disc, country, region, exp_years, 
                 exp_yesars_cur, edu_degree, wtcj, premium):
        self.title   = title
        self.clevel  = clevel # [0~6] 0 for unknown 
        self.indus   = indus
        self.disc    = disc
        self.country = country
        self.region  = region
        self.exp_years = exp_years # [0~6] 0 for unknown
        self.exp_yesars_cur = exp_yesars_cur # [0~6] 0 for unknown
        self.edu_degree = edu_degree # [0~3] 0 for unknown
        #self.edu_spec = edu_spec # (unused) list of int, 0 for unknown
        self.wtcj = wtcj # [0,1]
        self.premium = premium # (unused) [0,1]

    def getFeatures(self):
        return [self.title, self.clevel, self.indus, 
                self.disc, self.country, self.region, self.premium]

class Item:

    def __init__(self, title, clevel, indus, disc, country, region, paid, employment):
        self.title   = title
        self.clevel  = clevel # [0~6] 0 for unknown 
        self.indus   = indus
        self.disc    = disc
        self.country = country
        self.region  = region
        self.paid    = paid
        self.employment = employment # [0~5] 0 for unknown

    def getFeatures(self):
        return [self.title, self.clevel, self.indus, 
                self.disc, self.employment, self.paid,
                self.country, self.region]

class Interaction:
    
    def __init__(self, user, item, interaction_type):
        self.user = user
        self.item = item
        self.interaction_type = interaction_type

    def title_match(self):
        return float(len(set(self.user.title).intersection(set(self.item.title))))

    def clevel_match(self): # career level
        if self.user.clevel == 0 or self.item.clevel == 0:
            return 0
        else:
            diff = abs(self.user.clevel - self.item.clevel)
            return 1.0 - float(diff)/6

    def indus_match(self): # industry
        if self.user.indus == self.item.indus:
            return 1.0
        else:
            return 0.0

    def discipline_match(self):
        if self.user.disc == self.item.disc:
            return 2.0
        else:
            return 0.0

    def country_match(self):
        if self.user.country == self.item.country:
            return 1.0
        else:
            return 0.0

    def region_match(self):
        if self.user.region == self.item.region:
            return 1.0
        else:
            return 0.0
    
    def exp_match(self):
        if self.user.exp_years == 0 or self.item.clevel == 0:
            return 0
        else:
            diff = abs(self.user.exp_years - self.item.clevel)
            return 1.0 - float(diff)/6.0

    def employ_match(self):
        if self.item.employment in set([2,4,5]) and self.user.exp_years <= 2:
            return 1
        elif self.item.employment in set([1,3]) and self.user.exp_years >= 2:
            return 1
        return 0

    def features(self):
        return [
            self.title_match(), self.clevel_match(), self.indus_match(), 
            self.discipline_match(), self.country_match(), self.region_match()
        ]

    def label(self):
        if self.interaction_type == 4: # deleted
             return 0
        else:
             return 1


