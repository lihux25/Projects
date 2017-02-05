#import ROOT
import numpy as np
import pandas as pd
import math
import pickle

class DataGetter:

    def __init__(self, listToGet):
        self.listToGet = listToGet
        self.list2 = ["event." + v + "[i]" for v in self.listToGet]

        self.theStrCommand = "[" + ", ".join(self.list2) + "]"

    def getData(self, event, i):
        return eval(self.theStrCommand)

    def getList(self):
        return self.listToGet
