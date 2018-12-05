import theano
from theano import function
import theano.tensor as T
from termcolor import colored
from scipy.optimize import minimize, fmin_bfgs, fmin_cg
import numpy as np
import pickle
import sys as sys

from lib import *

args = sys.argv
if len(args) < 2:
    print("No test specified!")
    exit(1)

log_dir = str(args[1])
log_dir += '/'


# compute bounding box for modulus
x_min  = -1.
y_min  = -1.
x_max = 1.
y_max = 1.

# load results
po = open(log_dir + "fnls.pickle", "rb")
fnls = pickle.load( po)
po.close()

po = open(log_dir + "c_samples.pickle", "rb")
c_samples = pickle.load(po)
po.close()

# plotting
centroid_heatmap(c_samples, log_dir, x_min, x_max, y_min, y_max,bins=20)
centroid_plot(c_samples, log_dir, x_min, x_max, y_min, y_max)
plot_autocorr(c_samples, log_dir)
fnl_histogram(fnls, log_dir)
