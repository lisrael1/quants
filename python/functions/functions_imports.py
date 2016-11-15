from time import time
start_time = time()
from numpy import matrix as m
from numpy import *
import pandas as ps
from collections import OrderedDict
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brent,minimize,fmin
from pylab import plot,show,grid,xlabel,ylabel,title,legend,close,savefig,ion,draw,figure,annotate
from time import sleep
from multiprocessing import Pool
from operator import methodcaller
set_printoptions(precision=6, threshold=None, edgeitems=None, linewidth=100, suppress=1, nanstr=None, infstr=None, formatter=None)
import warnings
warnings.filterwarnings("ignore")
