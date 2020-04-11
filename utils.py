import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import copy
from collections import namedtuple


def parse_args():
    """
    retrieving arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="Run SO-GAAL.")
    parser.add_argument('--path', nargs='?', default='data/LHC_data.csv',
                        help='Input data path.')
    parser.add_argument('--stop_epochs', type=int, default=20,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--lr_d', type=float, default=0.01,
                        help='Learning rate of discriminator.')
    parser.add_argument('--lr_g', type=float, default=0.0001,
                        help='Learning rate of generator.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Mini-batch size.')
    parser.add_argument('--decay', type=float, default=1e-6,
                        help='Decay.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum.')
    parser.add_argument('--plot_every', type=int, default=10,
                        help='Learning curves plotting frequency.')
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
    plt.plot(x, aucy, color='yellow', linewidth='3', label='ROC, AUC={}'.format(round(aucy[-1], 4)))

    plt.legend(loc='best')
    plt.savefig(save_dir + str(run) + '.png')
    plt.show()


class RunBuilder:
    """
    Class that builds runs from parameters' OrderedDict
    """
    @staticmethod
    def generate_run(args):
        # construct a string from arguments to identitfy the current run
        # construct the run ID from time
        year = str(datetime.now().year)
        month = '{:02d}'.format(datetime.now().month)
        day = '{:02d}'.format(datetime.now().day)
        hour = '{:02d}'.format(datetime.now().hour)
        minute = '{:02d}'.format(datetime.now().minute)
        second = '{:02d}'.format(datetime.now().second)
        current_run_id = year + month + day + hour + minute + second

        # transform args to dict
        args_dict = vars(copy.deepcopy(args))

        # remove data path from args: the / raise errors
        data_path = args_dict.pop('path')

        # we are going to keep just the dataset name
        dataset_name = data_path[data_path.rfind('/') + 1:]

        # remove extensions if any
        dataset_name = dataset_name.split('.')[0] if '.' in dataset_name else dataset_name

        # construct a namedtuple
        Run = namedtuple('Run', ['run_id', 'dataset', *args_dict.keys()])

        # construct the id of current run
        current_run = Run(current_run_id, dataset_name, *args_dict.values())

        return current_run
