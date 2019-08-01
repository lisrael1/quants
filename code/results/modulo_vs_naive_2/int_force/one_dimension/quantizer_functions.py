'''
	originally taken from quants\code\arch\2_inputs\functions\functions_quantizer.py
'''

from os.path import realpath
from sys import path
root_folder = realpath(__file__).replace('\\', '/').rsplit('int_force', 1)[0]
path.append(root_folder)

from sys import platform
if "win" in platform:
    dlmtr="\\"
    img_type = ".png"
else:
   dlmtr = "/"
   img_type = ".jpg"
# from numpy import *
# from pylab import plot,show,grid,xlabel,ylabel,title,legend,close,savefig,ion,draw,figure,annotate,text,subplot
import numpy as np

np.set_printoptions(precision=6, threshold=None, edgeitems=None, linewidth=10000, suppress=1, nanstr=None, infstr=None, formatter=None)
import warnings

warnings.filterwarnings("ignore")


# from time import time
# from datetime import datetime
# from os import makedirs,path,getpid,getenv
# start_time = time()
# from sys import platform,getsizeof,argv
# from  optparse import OptionParser
# if "win" in platform:
#    dlmtr="\\"
#    img_type=".png"
# else:
#      dlmtr="/"
#      img_type=".jpg"
# from numpy import matrix as m
# from numpy import *
# import pandas as pd
# from collections import OrderedDict
# from scipy.integrate import quad
# from scipy.stats import norm
# from scipy.optimize import brent,minimize,fmin
# from pylab import plot,show,grid,xlabel,ylabel,title,legend,close,savefig,ion,draw,figure,annotate,text,subplot
# from time import sleep
# from multiprocessing import Pool
# from operator import methodcaller
# from struct import unpack
# set_printoptions(precision=6, threshold=None, edgeitems=None, linewidth=10000, suppress=1, nanstr=None, infstr=None, formatter=None)
# import warnings,psutil,configparser
# warnings.filterwarnings("ignore")


class simple_quantizer():
    """
    when first_quantizer_bar_at_zero is True, you have quantizer bar at 0, when not, if bin_size is 0.1, you will have bar quantizer at ...,-0.15,-0.05,0.05,0.15,...
    for example:
                print simple_quantizer(0.1,11).quantizise([-0.91,0.11,-0.49])#if you put (0.1,5), you will get the same results...
                      [[-0.9  0.1 -0.5]]
                print simple_quantizer(0.1,20).quantizise([-0.91,0.11,-0.49])
                      [[-0.95  0.15 -0.45]]
                print simple_quantizer(0.1,3).modulo_edge_to_edge
                      0.3
    """

    def __init__(self, bin_size, number_of_quants):
        if np.isnan(bin_size) or np.isinf(bin_size):
            bin_size = 0;
        self.bin_size = bin_size
        self.number_of_quants = number_of_quants
        if self.bin_size == 0:
            self.modulo_edge_to_edge = "disable"
        else:
            self.modulo_edge_to_edge = self.bin_size * self.number_of_quants  # you have half bin at the left side and half at the right

    def quantizise(self, numbers):
        if (self.bin_size == 0):
            return np.mat(numbers)
        # rounding to the quantizer (rint is rounding to the closes integer):
        numbers = np.mat(numbers)
        if self.number_of_quants % 2:
            q = self.bin_size * np.rint(numbers / self.bin_size)
        else:
            q = self.bin_size * np.rint((numbers + self.bin_size * 0.5) / self.bin_size) - self.bin_size * 0.5
        return q

    def __iter__(self):
        return self.__dict__.iteritems()


class quantizer():
    """
        for now, the quantizer is around 0
        example:
            quantizer(bin_size=1.3, number_of_quants=5, max_val=None, all_quants=None, mu=0, sigma=1, modulo_edge_to_edge=None).plot_pdf_quants()
        more examples:
            q=quantizer(0.1,4)
              bin size = 0.1, number of quants = 4
            q=quantizer(0.1,4,10)
            q=quantizer(all_quants=[-0.15,-0.05,0.05,0.15])
              entering the quantizers
            print q
    """

    def __init__(self, bin_size=None, number_of_quants=None, max_val=None, all_quants=None, mu=0, sigma=1, modulo_edge_to_edge=None):
        self.mu = mu
        self.sigma = float(sigma)
        self.var = self.sigma ** 2
        self.expected_mse = -1

        if bin_size is not None and number_of_quants is not None and max_val is None:
            self.bin_size = float(bin_size)
            self.number_of_quants = number_of_quants
            self.max_val = self.bin_size * (self.number_of_quants - 1) / 2.0
            self.all_quants = [i * self.bin_size - self.max_val for i in range(number_of_quants)]
            self.modulo_edge_to_edge = self.max_val * 2 + self.bin_size
        elif number_of_quants is not None and modulo_edge_to_edge is not None:
            self.number_of_quants = int(number_of_quants)
            self.modulo_edge_to_edge = float(modulo_edge_to_edge)
            self.bin_size = self.modulo_edge_to_edge / float(self.number_of_quants + 1)
            self.max_val = self.bin_size * (self.number_of_quants - 1) / 2.0
            self.all_quants = [i * self.bin_size - self.max_val for i in range(number_of_quants)]
        elif all_quants is not None:
            self.all_quants = all_quants
            self.bin_size = all_quants[1] - all_quants[0]
            self.number_of_quants = len(all_quants)
            self.max_val = self.all_quants[-1]
            self.modulo_edge_to_edge = self.max_val * 2 + self.bin_size
        self.all_quants = np.array(self.all_quants)

    def quantizise(self, numbers):
        """
            this function quantize a number by a given quantizer with bins
        """
        if (self.bin_size == 0):
            return np.mat(numbers)
        # rounding to the quantizer (rint is rounding to the closes integer):
        numbers = np.mat(numbers)
        q = np.rint((numbers + self.max_val) / (1.0 * self.bin_size)) * self.bin_size - self.max_val
        # taking the edges:
        # the modulo should do this so i dropped it here
        ##		q=mat([self.max_val if i>self.max_val else -self.max_val if i<-self.max_val else i for i in q.A1]).reshape(q.shape)
        ##		q=mat(q).reshape(q.shape)
        return q

    def plot_pdf_quants(self, plot_mod_edge=False):
        from scipy.stats import norm
        import pylab as plt

        x = np.arange(-self.max_val - 2 * self.bin_size, self.max_val + 2 * self.bin_size, self.sigma / 1000.0)
        plt.plot(x, norm.pdf(x, self.mu, self.sigma), label="pdf")
        plt.plot(self.all_quants, np.zeros(self.number_of_quants), "D", label="bin middle")
        # plt.plot(np.append(self.all_quants-self.bin_size/2, self.all_quants[-1]+self.bin_size/2), np.zeros(self.number_of_quants+1), "D", label="bin edges")
        # for edge in np.append(self.all_quants-self.bin_size/2, self.all_quants[-1]+self.bin_size/2):
        for edge in self.all_quants[1:]-self.bin_size/2:
            plt.axvline(edge, color='g', alpha=0.7, linestyle='--')
        if plot_mod_edge:
            plt.plot([-self.modulo_edge_to_edge / 2, self.modulo_edge_to_edge / 2], np.zeros(2), "D", label="modulo edge")
        plt.legend(loc="best")
        # error=analytical_error(self.bin_size,self.number_of_quants,self.mu,self.sigma)
        mse = analytical_error(quantizer_i=self)
        plt.title("#bins=" + str(self.number_of_quants) + ", bin size=" + str(self.bin_size) + "\nmu=" + str(self.mu) + ", sigma=" + str(self.sigma) + ", MSE=" + str(mse))
        plt.grid()
        if 1:
            plt.show()
        else:
            plt.savefig("temp" + dlmtr + "quantizer on bell" + dlmtr + "quantizer on bell " + str(self.number_of_quants) + " quants" + img_type)
            plt.close()

    if 0:
        def __repr__(self):
            print("-----")
            print("bin_size:", self.bin_size)
            print("number_of_quants:", self.number_of_quants)
            print("max_val:", self.max_val)
            print("modulo_edge_to_edge:", self.modulo_edge_to_edge)
            print("all_quants:", self.all_quants)
            return ""

    def __iter__(self):
        return self.__dict__.iteritems()


def analytical_error(bin_size=None, quantizer_i=None):
    '''
    get quants, mu and sigma and return the analytic error for those values
    note that mu should be around 0 because of the mod in the mse_for_single_dot
    examples:
        print analytical_error(2.5,5,0,1)
        plot([analytical_error(i/10.0,5,0,1) for i in range(1,100)])
        show()
    '''
    from scipy.stats import norm
    from scipy.integrate import quad
    from int_force.methods.methods import sign_mod

    if bin_size is not None:
        q = quantizer(bin_size=bin_size, number_of_quants=quantizer_i.number_of_quants)
    else:
        q = quantizer_i

    def quantizise_single(x, quantizer_i):
        # taking max and min values to be at the last bins
        # rounding to the quantizer_i:
        q = np.rint((x + quantizer_i.max_val) / (1.0 * quantizer_i.bin_size)) * quantizer_i.bin_size - quantizer_i.max_val
        # cutting the edges to the mas quantizer_i value:
        if q > quantizer_i.max_val:
            q = quantizer_i.max_val
        if q < -quantizer_i.max_val:
            q = -quantizer_i.max_val
        return q

    def mse_for_single_dot(x, quantizer_i):
        simple_one = 0
        if simple_one:
            return norm(quantizer_i.mu, quantizer_i.sigma).pdf(x) * ((x - quantizise_single(x, quantizer_i)) ** 2)
        else:
            # TODO: add dither?
            # dither=random.uniform(0,quantizer_i.bin_size)
            return norm(quantizer_i.mu, quantizer_i.sigma).pdf(x) * ((x - quantizise_single(sign_mod(np.array(x), quantizer_i.modulo_edge_to_edge), quantizer_i)) ** 2)

    mse = quad(mse_for_single_dot, -10 * q.sigma, 10 * q.sigma, args=(q))[0]
    see_itterations = 0
    if see_itterations:
        print(q.bin_size, mse)
    return mse


def find_best_quantizer(number_of_quants, sigma, mu=0):
    '''
        looks for best quantizer by number of quants and normal dist args
        this one is for quantizer, not for simple_quantizer
        example:
            number_of_quants=5
            sigma=1.0
            q=find_best_quantizer(number_of_quants,sigma)
            print q
            q.plot_pdf_quants()
    :param number_of_quants:
    :param sigma:
    :param mu:
    :return:
    '''
    from scipy.optimize import brent, minimize, fmin

    # we will start looking from sigma
    stop_looking_at_bin_size_error = sigma / (1000.0 * number_of_quants)  # at brent /100 took 100 sec, /10 took 80 sec and /1 took 60 sec
    # bin_size=fmin(analytical_error,start_looking_from,xtol=stop_looking_at_bin_size_error,ftol=sigma*100,args=(number_of_quants,mu,sigma),disp=False).tolist()[0]
    q = quantizer(bin_size=1, number_of_quants=number_of_quants, sigma=sigma, mu=mu)
    print('running estimation')
    if 0:  # seems like the fmin is not working here
        debug_searching_func = False
        start_looking_from = 4 * sigma / number_of_quants
        bin_size = fmin(analytical_error, start_looking_from, xtol=stop_looking_at_bin_size_error, ftol=sigma * 100, args=(q,), disp=debug_searching_func).tolist()[0]
    else:
        bin_size = brent(analytical_error, args=(q,), tol=stop_looking_at_bin_size_error)
    print('done estimation')
    # bin_size=brent(analytical_error,args=(q,))
    return quantizer(bin_size=bin_size, number_of_quants=number_of_quants, mu=mu, sigma=sigma)


# print brent(analytical_error,args=(1000,0,1))
# print minimize(analytical_error,(1,0.1),method='TNC',args=(1000,0,1))


def find_best_quantizer_parallel_for_1_sigma(number_of_quants):
    return find_best_quantizer(number_of_quants=number_of_quants, sigma=1)


def plot_quantizer_on_normal_bell_example():
    quantizer(bin_size=2, number_of_quants=5, max_val=None, all_quants=None, mu=0, sigma=1, modulo_edge_to_edge=None).plot_pdf_quants()


if __name__ == "__main__":
    # find_best_quantizer_parallel_for_1_sigma(number_of_quants=5).plot_pdf_quants()
    plot_quantizer_on_normal_bell_example()
