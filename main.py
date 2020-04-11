import subprocess
from collections import OrderedDict


if __name__ == '__main__':
    # creating the args
    args = OrderedDict(path='./data/onecluster',
                       stop_epochs=100,
                       lr_d=0.0001,
                       lr_g=0.0004,
                       batch_size=64,
                       decay=1e-6,
                       momentum=0.9,
                       plot_every=1)

    # creating the command
    command = 'python train.py'
    for arg in args:
        command += ' --{} {}'.format(arg, args[arg])

    # running the command
    print(command)
    subprocess.call(command.split(' '), shell=True)
