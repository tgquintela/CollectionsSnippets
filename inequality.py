
"""

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2652960/

TODO
----
Kakwani index: measure of progressivity of intervention
Atkinson index: https://en.wikipedia.org/wiki/Atkinson_index
Suits index: https://en.wikipedia.org/wiki/Suits_index

Properties measures: https://en.wikipedia.org/wiki/Income_inequality_metrics

"""
import scipy
import numpy as np
import matplotlib.pyplot as plt


##################### Plot tools #############
##############################################
def plot_distribution(dist):
    fig = plt.figure()
    plt.hist(dist, int(np.sqrt(len(dist))/2))
    return fig


def plot_cumdistribution(dist):
    cum_y = np.sort(dist).cumsum()/np.sum(dist)
    x = np.linspace(0, 1, len(cum_y))
    fig = plt.figure()
    plt.ylim(0, 1)
    plt.plot(x, cum_y)
    plt.plot(x, x)
    return fig


##################### Artificial samples #####
##############################################
def sample_gaussian_bounded(n, m, std):
    dist = np.zeros(n)
    for i in range(n):
        while True:
            r = np.random.randn()*std+m
            if r > 0:
                dist[i] = r
                break
    return dist


def sample_square_gaussian(n, std):
    dist = np.random.randn(n)*std
    dist = dist**2
    return dist


def sample_abs_gaussian(n, std):
    dist = np.abs(np.random.randn(n)*std)
    return dist


def sample_poisson(n, lam=1.0):
    dist = np.random.poisson(lam=lam, size=n)
    return dist


def sample_deltadirac(n, magnitude=1.0):
    return np.ones(n)*magnitude


def sample_pareto(n, xm=1, alpha=1.161):
    unisample = np.random.random(n)
    dist = np.power(1-unisample, -1./alpha) / xm
    return dist


def sample_lognormal(n, m, std):
    dist = np.log(sample_gaussian_bounded(n, m, std) + 1)
    return dist


##################### Unequality measures ####
##############################################
def GiniIndex(dist):
    cum_y = np.sort(dist).cumsum()/np.sum(dist)
    x = np.linspace(0, 1, len(cum_y))
    return 2*(0.5-scipy.integrate.simps(cum_y, x))


def HooverIndex(dist, quantity=None):
    if quantity is None:
        quantity = np.ones(len(dist))
    A_i_rate = quantity/len(dist)
    E_i_rate = dist/np.sum(dist)
    HI = 1/2.*np.sum(np.abs(A_i_rate - E_i_rate))
    return HI


def TheilIndex(dist):
    TI = np.mean((dist / dist.sum()) * np.log(dist / dist.sum()))
    return TI


def Ratio2020Index(dist):
    idx = len(dist)/5
    dist_s = np.sort(dist)
    R2020I = dist_s[(len(dist)-idx):].sum()/dist_s[:idx].sum()
    return R2020I


def Ratio1010Index(dist):
    idx = len(dist)/10
    dist_s = np.sort(dist)
    R1010I = dist_s[(len(dist)-idx):].sum()/dist_s[:idx].sum()
    return R1010I


def RatioHalfIndex(dist):
    idx = len(dist)/2
    dist_s = np.sort(dist)
    RHI = dist_s[(len(dist)-idx):].sum()/dist_s.sum()
    return RHI


def Ratio1pIndex(dist):
    assert(len(dist) > 100)
    idx = len(dist)/100
    dist_s = np.sort(dist)
    R1pI = dist_s[(len(dist)-idx):].sum()/dist_s.sum()
    return R1pI


def Ratio001pIndex(dist):
    assert(len(dist) > 10000)
    idx = len(dist)/100
    dist_s = np.sort(dist)
    R001pI = dist_s[(len(dist)-idx):].sum()/dist_s.sum()
    return R001pI


def PalmaRatioIndex(dist):
    idx = len(dist)/10
    dist_s = np.sort(dist)
    PRI = dist_s[(len(dist)-idx):].sum()/dist_s[:4*idx].sum()
    return PRI


def CoefficientVariationIndex(dist):
    return dist.std()/dist.mean()
