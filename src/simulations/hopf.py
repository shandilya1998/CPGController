import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

colors = ['r', 'b', 'g', 'c', 'm', 'y']

dt = 0.001
N = 10000

def hopf(x, y, mu, omega):
    x += (-y*omega + x*(mu - x**2 -y**2))*dt
    y += (x*omega + y*(mu - x**2 -y**2))*dt
    return x, y

t = np.arange(N) * dt

def create():
    fig, ax = plt.subplots(1,1, figsize = (4,4))
    mu = 1
    omega = 30
    for color in colors:
        x = random.random()*5
        y = random.random()*5
        X = [x]
        Y = [y]
        for i in range(N-1):
            x, y = hopf(x, y, mu, omega)
            X.append(x)
            Y.append(y)
        ax.plot(np.array(X), np.array(Y), color)
    fig.savefig('TrajectoryHopfOscillatormu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')

    fig, ax = plt.subplots(1,1, figsize = (4,4))
    x = random.random()*10
    y = random.random()*10
    X = [x]
    Y = [y]
    for i in range(N-1):
        x, y = hopf(x, y, mu, omega)
        X.append(x)
        Y.append(y)
    ax.plot(t[:1000], X[:1000])
    fig.savefig('Oscillationsxmu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')

    fig, ax = plt.subplots(1,1, figsize = (4,4))
    ax.plot(t[:1000], Y[:1000])
    fig.savefig('Oscillationsymu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')

    X = np.array(X)
    Y = np.array(Y)
    r = np.sqrt(X**2 + Y**2)
    fig, ax = plt.subplots(1,1, figsize = (4,4))
    ax.plot(t[:1000], r[:1000])
    fig.savefig('TrendRmu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')

    mu = 10
    omega = 20
    fig, ax = plt.subplots(1,1, figsize = (4,4))
    for color in colors:
        x = random.random()*5
        y = random.random()*5
        X = [x]
        Y = [y]
        for i in range(N-1):
            x, y = hopf(x, y, mu, omega)
            X.append(x)
            Y.append(y)
        ax.plot(np.array(X), np.array(Y), color)
    fig.savefig('TrajectoryHopfOscillatormu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')

    fig, ax = plt.subplots(1,1, figsize = (4,4))
    x = random.random()*10
    y = random.random()*10
    X = [x]
    Y = [y]
    for i in range(N-1):
        x, y = hopf(x, y, mu, omega)
        X.append(x)
        Y.append(y)
    ax.plot(t[:1000], X[:1000])
    fig.savefig('Oscillationsxmu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')

    fig, ax = plt.subplots(1,1, figsize = (4,4))
    ax.plot(t[:1000], Y[:1000])
    fig.savefig('Oscillationsymu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')

    X = np.array(X)
    Y = np.array(Y)
    r = np.sqrt(X**2 + Y**2)
    fig, ax = plt.subplots(1,1, figsize = (4,4))
    ax.plot(t[:1000], r[:1000])
    fig.savefig('TrendRmu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')

    mu = 4
    omega = 16
    fig, ax = plt.subplots(1,1, figsize = (4,4))
    for color in colors:
        x = random.random()*5
        y = random.random()*5
        X = [x]
        Y = [y]
        for i in range(N-1):
            x, y = hopf(x, y, mu, omega)
            X.append(x)
            Y.append(y)
        ax.plot(np.array(X), np.array(Y), color)
    fig.savefig('TrajectoryHopfOscillatormu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')

    fig, ax = plt.subplots(1,1, figsize = (4,4))
    x = random.random()*10
    y = random.random()*10
    X = [x]
    Y = [y]
    for i in range(N-1):
        x, y = hopf(x, y, mu, omega)
        X.append(x)
        Y.append(y)
    ax.plot(t[:1000], X[:1000])
    fig.savefig('Oscillationsxmu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')

    fig, ax = plt.subplots(1,1, figsize = (4,4))
    ax.plot(t[:1000], Y[:1000])
    fig.savefig('Oscillationsymu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')

    X = np.array(X)
    Y = np.array(Y)
    r = np.sqrt(X**2 + Y**2)
    fig, ax = plt.subplots(1,1, figsize = (4,4))
    ax.plot(t[:1000], r[:1000])
    fig.savefig('TrendRmu{mu}omega{omega}.png'.format(
        mu = mu,
        omega = omega
    ))
    plt.close('all')


def mu_trend():
    dm = 0.1
    mu = 0.1
    omega = 30
    fig, ax = plt.subplots(1,1,figsize = (4,4))
    r = []
    MU = []
    for i in tqdm(range(10000)):
        x = random.random()*5
        y = random.random()*5
        X = [x]
        Y = [y]
        for i in range(N-1):
            x, y = hopf(x, y, mu, omega)
        r.append(np.sqrt(x*x + y*y))
        MU.append(mu)
        mu += dm
    ax.plot(np.arange(10000), r)
    fig.savefig('mu_hopf.png')
mu_trend()
