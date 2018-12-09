import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_density(pickle_dat, log_dir, fname):
    x, y, h = pickle_dat

    n = int(len(h) ** .5)
    H = np.array(h).reshape(n, n)

    plt.figure()
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel('$x$-coordinate')
    plt.ylabel('$y$-coordinate')
    plt.imshow(H, extent=(np.amin(x), np.amax(x),
                          np.amin(y), np.amax(y)), cmap='jet')
    plt.colorbar()
    plt.savefig(log_dir + fname + ".pdf")

if __name__ == "__main__":
    log_dir = '../tex/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fname = "density_criss_cross"
    po = open(fname + ".pickle", "rb")
    dat = pickle.load(po)
    plot_density(dat, log_dir, fname)
    po.close()

    fname = "density_squeeze"
    po = open(fname + ".pickle", "rb")
    dat = pickle.load(po)
    plot_density(dat, log_dir, fname)
    po.close()
