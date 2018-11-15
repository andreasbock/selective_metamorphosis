import numpy as np
import pylab as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def sample_circle(num_landmarks, scale=1, shift=0):
    thetas = np.linspace(0, 2*np.pi, num=num_landmarks+1)[:-1]
    positions = scale * np.array([[np.cos(x), np.sin(x)] for x in thetas]) + shift
    return positions

def criss_cross(num_landmarks):
    test_name = 'criss_cross'
    q0x = np.linspace(-1,1,num_landmarks)
    q0 = -np.ones([num_landmarks,2])
    q0[:,1] = q0x
    q1x = np.linspace(-1,1,num_landmarks)
    q1 = np.ones([num_landmarks,2])
    q1[:,1] = q1x[::-1]
    return q0, q1, test_name

def pringle(num_landmarks):
    test_name = 'pringle'
    scale = 1
    num_top = num_landmarks // 2
    if num_landmarks % 2 == 0:
        num_bot = num_top
    else:
        num_bot = num_top + 1

    thetas = np.linspace(0, np.pi, num=num_top+1)[:-1]
    positions_top = scale * np.array([[np.cos(x), np.sin(x)] for x in thetas])

    thetas = np.linspace(np.pi, 2*np.pi, num=num_bot+1)[:-1]
    positions_bot = scale * np.array([[np.cos(x), np.sin(x)] for x in thetas])[::-1]

    q0 = sample_circle(num_landmarks)
    q1 = np.append(positions_top, positions_bot, axis=0)
    return q0, q1, test_name

def squeeze(num_landmarks):
    test_name = 'squeeze'

    if num_landmarks != 8:
        print("only works for 8 landmarks at the moment!")
        exit()

    scale = 1

    thetas = np.linspace(0, 2*np.pi, num=num_landmarks+1)[:-1]
    q1 = scale * np.array([[np.cos(x), np.sin(x)] for x in thetas])

    k = 0
    for p in q1:
        x, y = p
        if abs(y) < 1e-04:
            q1[k] = (0., y)
        k += 1

    k = 0
    pert = 2.
    for p in q1:
        x, y = p
        if abs(x) > 2.5:
            q1[k] = (x + - pert*np.sign(x), y)
        k += 1

    q0 = sample_circle(num_landmarks)

    return q0, q1, test_name

def triangle_flip(num_landmarks):
    test_name = 'triangle_flip'

    if num_landmarks % 3 != 0:
        print("Want a nice image, so satisfy 'num_landmarks % 3 == 0' !")
        exit()
    scale = 1

    a = np.array([1e-06, 0])  # lazy with sign(0) in reflection
    b = np.array([-2.5, 5])
    c = np.array([2.5, 5])

    # interpolate between them to generate points
    ss = num_landmarks // 3

    q0_a = [(1-s/ss) * a + s/ss * b for s in range(ss)] # [`a` -> `b`)
    q0_b = [(1-s/ss) * b + s/ss * c for s in range(ss)] # [`b` -> `c`)
    q0_c = [(1-s/ss) * c + s/ss * a for s in range(ss)] # [`c` -> `a`)

    q0 = np.array(q0_a + q0_b + q0_c)

    # flip to generate q1 (reflect about x=2.5)
    q1 = np.copy(q0)
    k = 0
    for k in range(num_landmarks):
        x, y = q1[k]
        dist = 2 * np.sqrt((y - 2.5)**2)
        q1[k] = x, y - np.sign(y - 2.5) * dist
        k += 1

    return q0, q1, test_name

class map_estimator():
    def __init__(self, xs, center):
        self.xs = np.copy(xs)
        self.center = center

def trace_plot(fnls, log_dir):
    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Functional')
    plt.plot(range(len(fnls)), fnls, 'r*-')
    plt.savefig(log_dir + 'functional_traceplot.pdf')

def centroid_plot(c_samples, log_dir):
    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Centroid position')
    cx, cy = zip(*c_samples)
    plt.plot(cx, cy, 'go-', alpha=0.3)
    plt.savefig(log_dir + 'centroid_evolution.pdf')

def sample_autocov(k, m):
    # number of samples
    n = np.shape(m)[0]
    mu = np.mean(m)

    # compute autocov
    est = 0
    for i in range(n - k):
        est += np.dot(m[i + k] - mu, m[i] - mu)

    return est / n

def plot_autocorr(c_samples, fname):
    plt.figure()
    num = 100
    lags = np.linspace(0, len(c_samples), num, dtype=int)

    ac = lambda k: sample_autocov(k, c_samples)
    acf = list(map(ac, lags)) / sample_autocov(0, c_samples)

    plt.plot(lags, acf, 'r.-')
    plt.xlabel('Lag')
    plt.ylabel('Sample autocorrelation')
    plt.grid(linestyle='dotted')
    plt.savefig(fname + 'autocorrelation.pdf')

def fnl_histogram(fnls, fname):
    plt.figure()
    bins = len(fnls) // 1
    plt.hist(fnls, bins=bins, density=1, facecolor='green', alpha=0.75)
    plt.xlabel('Metamorphosis functional')
    plt.ylabel('Number of observed values')
    plt.grid(linestyle='dotted')
    plt.savefig(fname + 'functional_histogram.pdf')

def plot_q(x0, xs, N, fname, title=None):
    plt.figure()
    plot_landmarks(x0, color='r', start_style='x--')
    plot_landmarks_traj(xs, N, lw=1)
    plot_landmarks(xs[-1], start_style='o--')
    if title:
        plt.title(title)
    plt.grid(linestyle='dotted')
    plt.savefig(fname + '.pdf')
    plt.close()

def plot_landmarks_traj(x, N, lw=.1):
    if len(x.shape) == 2:
        x = x.reshape((1,1,) + x.shape)
    if len(x.shape) == 3:
        x = x.reshape((1,) + x.shape)
    if len(x.shape) == 5:
        for i in range(x.shape[0]):
            plot_landmarks_traj(x[i], N, lw)
        return

    for i in range(N):
        plt.plot(x[:,0,i,0], x[:,0,i,1], 'k-', lw=lw)

def plot_landmarks(x, x0=None, lw=1., line_style='g--', markersize=5, color='b',
    start_style='x--', end_style='o-'):
    if len(x.shape) == 2:
        x = x.reshape((1, 1,) + x.shape)
    if len(x.shape) == 3:
        x = x.reshape((1,) + x.shape)
    if len(x.shape) == 5:
        for i in range(x.shape[0]):
            plot_landmarks(x[i], x0=x0, lw=lw, line_style=line_style,
                start_style=start_style, end_style=end_style,
                markersize=markersize, color=color)
    if not x0 is None:
        x = np.concatenate((x0.reshape((1,)+x0.shape),x),axis=0)

    plt.plot(np.concatenate((x[0,0,:,0],[x[0,0,0,0],])),np.concatenate((x[0,0,:,1],[x[0,0,0,1],])),start_style,color=color,markersize=markersize)
    if x.shape[0] > 1:
        plt.plot(np.concatenate((x[-1,0,:,0],[x[-1,0,0,0],])),np.concatenate((x[-1,0,:,1],[x[-1,0,0,1],])),end_style,color=color,markersize=markersize)

    for i in range(x.shape[2]):
        plt.plot(x[:,0,i,0],x[:,0,i,1],line_style,lw=lw)


