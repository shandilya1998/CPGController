import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pickle
import matplotlib
matplotlib.use('Agg')
import argparse

def objective_polynomial_5(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

def remove_outliers(x, max_deviations = 3):
    if isinstance(x, list):
        x = np.array(x)
    mean = np.mean(x)
    std = np.std(x)
    distance_from_mean = abs(x - mean)
    not_outlier = distance_from_mean < max_deviations * std
    x = x[not_outlier]
    return x


def plot_fit_curve_polymonial_5(x, y, path, figsize = (6,6)):
    popt, _ = curve_fit(objective_polynomial_5, x, y)
    a, b, c, d, e, f = popt
    fig, ax = plt.subplots(1,1, figsize = figsize)
    ax.scatter(x, y)
    x_line = np.arange(min(x), max(x), 1)
    y_line = objective_polynomial_5(x_line, a, b, c, d, e, f)
    ax.plot(x_line, y_line, '--', color='red')
    fig.savefig(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_path',
        type = str,
        help = 'path of quantity to be plotted'
    )
    parser.add_argument(
        '--out_path',
        type = str,
        help = 'output file path'
    )
    args = parser.parse_args()

    pkl = open(args.in_path, 'rb')
    y = pickle.load(pkl)
    pkl.close()
    y = remove_outliers(y)
    x = np.arange(0, len(y), 1.0)
    plot_fit_curve_polymonial_5(
        x,
        y,
        args.out_path
    )
