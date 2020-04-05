import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """
    retrieving arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="Run SO-GAAL.")
    parser.add_argument('--path', nargs='?', default='data/onecluster',
                        help='Input data path.')
    parser.add_argument('--stop_epochs', type=int, default=20,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--lr_d', type=float, default=0.01,
                        help='Learning rate of discriminator.')
    parser.add_argument('--lr_g', type=float, default=0.0001,
                        help='Learning rate of generator.')
    parser.add_argument('--decay', type=float, default=1e-6,
                        help='Decay.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    return parser.parse_args()


def load_data(path):
    """
    loading data from the given path.
    """
    data = pd.read_table('{path}'.format(path=path), sep=',', header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    id = data.pop(0)
    y = data.pop(1)
    data_x = data.values
    data_id = id.values
    data_y = y.values
    data_y[data_y == 'nor'] = 1
    data_y[data_y == 'out'] = 0
    data_y = data_y.astype(np.int32)
    return data_x, data_y, data_id


def plot(train_history, run, save_dir='plots/'):
    """
    plot training history given the history dictionary.
    """
    dy = train_history['discriminator_loss']
    gy = train_history['generator_loss']
    aucy = train_history['auc']
    x = np.linspace(1, len(dy), len(dy))

    plt.plot(x, dy, color='blue', label='Discriminator Loss')
    plt.plot(x, gy, color='red', label='Generator Loss')
    plt.plot(x, aucy, color='yellow', linewidth='3', label='ROC AUC')
    # plot the final value of AUC
    plt.axhline(y=aucy[-1], c='k', ls='--', label='AUC={}'.format(round(aucy[-1], 4)))

    plt.legend(loc='best')
    plt.savefig(save_dir + str(run) + '.png')
    plt.show()
