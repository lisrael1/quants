from time import time
from datetime import datetime
from os import makedirs,path,getpid,getenv
start_time = time()
from sys import platform,getsizeof,argv
from  optparse import OptionParser
if "win" in platform:
   dlmtr="\\"
   img_type=".png"
else:
     dlmtr="/"
     img_type=".jpg"
from numpy import matrix as m
from numpy import *
import pandas as pd
from collections import OrderedDict
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brent,minimize,fmin
from pylab import plot,show,grid,xlabel,ylabel,title,legend,close,savefig,ion,draw,figure,annotate,text,subplot
from time import sleep
from multiprocessing import Pool
from operator import methodcaller
from struct import unpack
set_printoptions(precision=6, threshold=None, edgeitems=None, linewidth=10000, suppress=1, nanstr=None, infstr=None, formatter=None)
import warnings,psutil,configparser
warnings.filterwarnings("ignore")

#print "simulation time done importing libraries: ",time() - start_time,"sec"
