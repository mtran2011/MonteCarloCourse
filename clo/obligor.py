import math

class Obligor(object):
    '''
    Class for an obligor
    Attributes:
        name (str): obligor name
        inducode (int): industry code
        ctrycode (int): country code
        rating (str): obligor credit rating
        pdcurve (list): curve for rate of defaults
    '''
    
    def __init__(self, name, inducode, ctrycode, rating, pdcurve):
        self.name = name
        self.inducode = inducode
        self.ctrycode = ctrycode
        self.rating = rating
        self.pdcurve = pdcurve