

import numpy as np
import matplotlib.pyplot as plt


def explore_size_dependent(sizes):
    normal_mean, normal_std = [], []
    powerlaw_mean, powerlaw_std = [], []
    for s in sizes:
        samples_mean = np.array([np.random.randn(s).mean()
                                 for i in range(100)])
        samples_std = samples_mean.std()
        normal_mean.append(samples_mean.mean())
        normal_std.append(samples_std)
        samples_mean = np.array([np.random.pareto(2, s).mean()
                                 for i in range(100)])
        samples_std = samples_mean.std()
        powerlaw_mean.append(samples_mean.mean())
        powerlaw_std.append(samples_std)
    normal_mean = np.array(normal_mean)
    normal_std = np.array(normal_std)
    powerlaw_mean = np.array(powerlaw_mean)
    powerlaw_std = np.array(powerlaw_std)
    return normal_mean, normal_std, powerlaw_mean, powerlaw_std


def discriminant(normal_mean, normal_std, powerlaw_mean, powerlaw_std):
    normal_mean -= normal_mean[-1]
    powerlaw_mean -= powerlaw_mean[-1]
    return normal_mean, normal_std, powerlaw_mean, powerlaw_std


def plot_size_dependent(sizes, normal_mean, normal_std, powerlaw_mean,
                        powerlaw_std):
    fig = plt.figure()
    ax = plt.gca()
    plt.plot(sizes, normal_mean)
    plt.plot(sizes, powerlaw_mean)

    ax.plot(sizes, normal_mean)
    ymax, ymin = normal_mean+normal_std, normal_mean-normal_std
    ax.fill_between(sizes, ymax, ymin, alpha=.2)
    ax.plot(sizes, powerlaw_mean)
    ymax, ymin = powerlaw_mean+powerlaw_std, powerlaw_mean-powerlaw_std
    ax.fill_between(sizes, ymax, ymin, alpha=.2)

    return fig

fig = plot_size_dependent(sizes, *discriminant(*explore_size_dependent(sizes)))
