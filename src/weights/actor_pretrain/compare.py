import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiments',
        help = 'A comma separated list of experiment ids in filesystem',
        type =  lambda x: x.split(','),
    )
    parser.add_argument(
        '--ids',
        help = 'A comma separated list of experiment ids to use as legends',
        type =  lambda x: x.split(','),
    )
    parser.add_argument(
        '--item',
        help = 'Choice of item to plot and compare',
        choices = ['total', 'action', 'mu', 'omega'],
    )
    parser.add_argument(
        '--model',
        help = 'Choice of model to plot losses for',
        choices = ['actor', 'enc'],
    )
    args = parser.parse_args()
    return args

def find_max_epoch(experiment, model):
    path = os.path.join(
        'exp{exp}'.format(exp = experiment),
        'pretrain_{model}'.format(model = model),
    )
    files = os.listdir(path)
    numbers = []
    for f in files:
        if 'action' in f or 'mu' in f or 'omega' in f or 'ckpt' in f or 'png' in f or 'checkpoint' in f:
            continue
        else:
            try:
                numbers.append(int(f[:-7].split('_')[-1]))
            except ValueError:
                print(f)
                pass
    if len(numbers) == 0:
        for f in files:
            if 'ckpt' in f or 'png' in f or 'checkpoint' in f:
                continue
            else:
                try:
                    numbers.append(int(f[:-7].split('_')[-1]))
                except ValueError:
                    print(f)
                    pass
    return max(numbers)

def get_filename(experiment, model, item, epoch):
    if item == 'total':
        item = ''
    else:
        item += '_'
    path = os.path.join(
        'exp{exp}'.format(exp = experiment),
        'pretrain_{model}'.format(model = model),
        'loss_{item}{exp}_pretrain_{model}_{epoch}.pickle'.format(
            item = item,
            exp = experiment,
            model = model,
            epoch = epoch
        )
    )
    return path

def main():
    args = parse_arguments()
    files = []
    epochs = []
    losses = []
    fig, ax = plt.subplots(1,1, figsize = (7,7))
    for i, exp in enumerate(args.experiments):
        epoch = find_max_epoch(exp, args.model)
        epochs.append(epoch)
        f = get_filename(exp, args.model, args.item, epoch)
        pkl = open(f, 'rb')
        loss = pickle.load(pkl)
        pkl.close()
        model = args.model
        if model == 'enc':
            model = 'encoder'
        ax.plot(loss, label = 'Experiment {exp} {model} {item} mse loss'.format(
            exp = args.ids[i],
            model = model,
            item = args.item
        ))
    ax.legend(loc = 'upper right')
    path = 'loss_pretrain_{model}_{item}_experiments'.format(
        model = args.model,
        item = args.item
    )
    for exp in args.experiments:
        path += '_{exp}'.format(exp = exp)
    path += '.png'
    fig.savefig(path)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()

