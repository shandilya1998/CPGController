import numpy as np
from scipy.optimize import curve_fit
import pickle
import matplotlib
matplotlib.use('Agg')
import argparse
import matplotlib.pyplot as plt

def objective_polynomial_5(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

def remove_outliers(x, max_deviations = 1):
    if isinstance(x, list):
        x = np.array(x)
    mean = np.mean(x)
    std = np.std(x)
    distance_from_mean = abs(x - mean)
    not_outlier = distance_from_mean < max_deviations * std
    x = x[not_outlier]
    return x


def plot_fit_curve_polymonial_5(x, y, path = None, figsize = (6,6), plot = True,
        x_label = None, y_label = None, title = None):
    popt, _ = curve_fit(objective_polynomial_5, x, y)
    a, b, c, d, e, f = popt
    fig = None
    ax = None
    if plot:
        fig, ax = plt.subplots(1,1, figsize = figsize)
        ax.scatter(x, y)
    x_line = np.arange(min(x), max(x), 1)
    y_line = objective_polynomial_5(x_line, a, b, c, d, e, f)
    if plot:
        ax.plot(x_line, y_line, '--', color='red')
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if title is not None:
            ax.set_title(title)
        fig.savefig(path)
    if not plot:
        return x_line, y_line

def plot_reward_split(rewards, path, episodes, max_steps, rnn_steps):
    pos_rewards = []
    penalty = []
    err_penalty = []
    sum_rewards = []
    sum_penalty = []
    total_rewards = []
    p_r = 0.0
    p = 0.0
    err_p = 0.0
    for i in range(episodes * max_steps * rnn_steps):
        if i % (max_steps * rnn_steps) == 0:
            if p_r != 0:
                pos_rewards.append(p_r)
            if p != 0:
                penalty.append(p)
            if err_p != 0:
                err_penalty.append(err_p)
            if (p_r + p) != 0:
                sum_rewards.append(p_r + p)
            if (p + err_p):
                sum_penalty.append(p + err_p)
            total_rewards.append(p_r + p + err_p)
            p_r = 0.0
            p = 0.0
            err_p = 0.0
        reward = rewards[i]
        if reward >= 0.0:
            p_r += reward
        elif 0.0 > reward > -1.0:
            p += reward
        elif reward <= -1.0:
            err_p += reward
    if len(pos_rewards) == 0:
        pos_rewards = [0.0]* episodes
    if len(err_penalty) == 0:
        err_penalty = [0.0]* episodes
    if len(penalty) == 0:
        penalty = [0.0]* episodes
    if len(sum_rewards) == 0:
        sum_rewards = [0.0]* episodes
    if len(sum_penalty) == 0:
        sum_penalty = [0.0]* episodes
    if len(total_rewards) == 0:
        total_rewards = [0.0]* episodes
    pos_rewards = remove_outliers(pos_rewards)
    penalty = remove_outliers(penalty)
    err_penalty = remove_outliers(err_penalty)
    sum_rewards = remove_outliers(sum_rewards)
    sum_penalty = remove_outliers(sum_penalty)
    total_rewards = remove_outliers(total_rewards)
    """
    x1 = np.arange(0, len(pos_rewards), 1.0)
    x_line_1, _pos_rewards = plot_fit_curve_polymonial_5(
        x1, pos_rewards, plot = False
    )
    """
    x2 = np.arange(0, len(penalty), 1.0)
    x_line_2, _penalty = plot_fit_curve_polymonial_5(
        x2, penalty, plot = False
    )
    x3 = np.arange(0, len(err_penalty), 1.0)
    x_line_3, _err_penalty = plot_fit_curve_polymonial_5(
        x3, err_penalty, plot = False
    )
    x4 = np.arange(0, len(sum_rewards), 1.0)
    x_line_4, _sum_rewards = plot_fit_curve_polymonial_5(
        x4, sum_rewards, plot = False
    )
    x5 = np.arange(0, len(sum_penalty), 1.0)
    x_line_5, _sum_penalty = plot_fit_curve_polymonial_5(
        x5, sum_penalty, plot = False
    )
    x6 = np.arange(0, len(total_rewards), 1.0)
    x_line_6, _total_rewards = plot_fit_curve_polymonial_5(
        x6, total_rewards, plot = False
    )
    fig, ax = plt.subplots(3,2, figsize = (12, 18))
    """
    ax[0][0].scatter(x1, pos_rewards)
    ax[0][0].plot(x_line_1, _pos_rewards, '--', color = 'red')
    ax[0][0].set_title('Reinforcing Reward')
    ax[0][0].set_xlabel('episodes')
    ax[0][0].set_ylabel('reward')
    """
    ax[0][1].scatter(x2, penalty)
    ax[0][1].plot(x_line_2, _penalty, '--', color = 'red')
    ax[0][1].set_title('Penalty')
    ax[0][1].set_xlabel('episodes')
    ax[0][1].set_ylabel('reward')
    ax[1][0].scatter(x3, err_penalty)
    ax[1][0].plot(x_line_3, _err_penalty, '--', color = 'red')
    ax[1][0].set_title('Error Penalty')
    ax[1][0].set_xlabel('episodes')
    ax[1][0].set_ylabel('reward')
    ax[1][1].scatter(x4, sum_rewards)
    ax[1][1].plot(x_line_4, _sum_rewards, '--', color = 'red')
    ax[1][1].set_title('Rewards')
    ax[1][1].set_xlabel('episodes')
    ax[1][1].set_ylabel('reward')
    ax[2][0].scatter(x5, sum_penalty)
    ax[2][0].plot(x_line_5, _sum_penalty, '--', color = 'red')
    ax[2][0].set_title('Total Penalty')
    ax[2][0].set_xlabel('episodes')
    ax[2][0].set_ylabel('reward')
    ax[2][1].scatter(x6, total_rewards)
    ax[2][1].plot(x_line_6, _total_rewards, '--', color = 'red')
    ax[2][1].set_title('Total Rewards')
    ax[2][1].set_xlabel('episodes')
    ax[2][1].set_ylabel('reward')
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
