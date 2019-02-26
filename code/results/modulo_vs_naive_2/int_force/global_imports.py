import numpy as np
import pandas as pd
import itertools


class timer:
    def __init__(self):
        import time
        self.time=time
        self.start = time.time()

    def current(self):
        return "{:} sec".format(self.time.time() - self.start)

    def __del__(self):
        print("closing timer with value {:}".format(self.current()))
        return self.current()