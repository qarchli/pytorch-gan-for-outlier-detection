import numpy as np
import pandas as pd
from collections import defaultdict
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn import metrics

from utils import parse_args, load_data, plot, RunBuilder
from model import Discriminator, Generator


train = True  # to add an early stopping condition if necessary
args = parse_args()
data_x, data_y, data_id = load_data(args.path)

# specify the device on which the model will be trained
TRAINING_ON_GPU = torch.cuda.is_available()
device = torch.device("cuda:0" if TRAINING_ON_GPU else "cpu")

print('The dimension of the training data: {}*{}'.format(data_x.shape[0], data_x.shape[1]))
print('GPU available: ', TRAINING_ON_GPU)

if train:
    latent_size = data_x.shape[1]
    data_size = data_x.shape[0]
    batch_size = min(args.batch_size, data_size)
    stop = 0
    epochs = args.stop_epochs * 3
    train_history = defaultdict(list)

    # create tensors from np array
    data_x_tensor = torch.Tensor(data_x).to(device)
    data_y_tensor = torch.Tensor(data_y).to(device)

    # Create a tensor dataset
    train_set = torch.utils.data.TensorDataset(data_x_tensor,
                                               data_y_tensor)
    # Create a train loader
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)

    print('data_size: {}, batch_size: {}, latent_size: {}'.format(data_size, batch_size, latent_size))

    # create discriminator
    discriminator = Discriminator(latent_size, data_size)
    discriminator_optim = optim.SGD(discriminator.parameters(), lr=args.lr_d, dampening=args.decay, momentum=args.momentum)
    discriminator_criterion = F.binary_cross_entropy
    discriminator.to(device)  # send the network to the specified device
    print(discriminator)

    # Create generator
    generator = Generator(latent_size)
    generator_optim = optim.SGD(generator.parameters(), lr=args.lr_g, dampening=args.decay, momentum=args.momentum)
    generator_criterion = F.binary_cross_entropy
    generator.to(device)
    print(generator)

    # Start training epochs
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))

        # go over batches of data in the train_loader
        for i, data in enumerate(train_loader):
            print('\tTesting for epoch {} batch_index {}/{}'.format(epoch + 1, i + 1, len(train_loader)))

            # Generate noise
            noise_size = batch_size
            noise = np.random.uniform(0, 1, (int(noise_size), latent_size))
            noise = torch.tensor(noise, dtype=torch.float32).to(device)

            # Get training data
            data_batch, _ = data

            # Generate potential outliers
            generated_data = generator(noise)

            # Concatenate real data to generated data
            # X = torch.tensor(np.concatenate([data_batch, generated_data]), dtype=torch.float32)
            X = torch.cat((data_batch, generated_data))
            Y = torch.tensor(np.array([1] * batch_size + [0] * int(noise_size)), dtype=torch.float32).unsqueeze(dim=1).to(device)

            # Train discriminator
            # enable training mode
            discriminator.train()
            # getting the prediction
            discriminator_pred = discriminator(X)
            # compute the loss
            discriminator_loss = discriminator_criterion(discriminator_pred, Y)
            # reset the gradients to avoid gradients accumulation
            discriminator_optim.zero_grad()
            # compute the gradients of loss w.r.t weights
            discriminator_loss.backward(retain_graph=True)
            # update the weights
            discriminator_optim.step()
            # Store the loss for later use
            train_history['discriminator_loss'].append(discriminator_loss.item())

            # Train generator
            # create fake labels
            trick = torch.tensor(np.array([1] * noise_size), dtype=torch.float32).unsqueeze(dim=1).to(device)
            discriminator.eval()  # freeze the discriminator
            if stop == 0:
                generator.train()  # enable training mode for the generator
                generator_loss = generator_criterion(discriminator(generated_data), trick)
                generator_optim.zero_grad()
                generator_loss.backward(retain_graph=True)
                generator_optim.step()
                train_history['generator_loss'].append(generator_loss.item())
            else:
                generator.eval()  # enable evaluation mode
                generator_loss = generator_criterion(discriminator(generated_data), trick)
                train_history['generator_loss'].append(generator_loss.item())

            # unfreeze the discriminator's layers
            for param in discriminator.parameters():
                param.requires_grad = True

        # Stop training generator
        if epoch + 1 > args.stop_epochs:
            stop = 1

        # Detection result
        discriminator.eval()
        p_value = discriminator(data_x_tensor)
        if TRAINING_ON_GPU:
            p_value = p_value.cpu()
        p_value = pd.DataFrame(p_value.detach().numpy())
        data_y = pd.DataFrame(data_y)
        result = np.concatenate([p_value, data_y], axis=1)
        result = pd.DataFrame(result, columns=['p', 'y'])
        result = result.sort_values('p', ascending=True)

        # Calculate the AUC
        fpr, tpr, _ = metrics.roc_curve(result['y'].values, result['p'].values)
        AUC = metrics.auc(fpr, tpr)

        for _ in train_loader:
            train_history['auc'].append(AUC)

        if (epoch + 1) % args.plot_every == 0:

            # generate a run id
            current_run = RunBuilder.generate_run_id(args)

            # plot and save
            plot(train_history, current_run)

            # save generator and discriminator weights
            discriminator.save(current_run)
            generator.save(current_run)

            # remove previous plots and weights
            try:
                previous_run
            except NameError:
                pass
            else:
                try:
                    os.remove('./plots/' + str(previous_run) + '.png')
                    os.remove(generator.checkpoint_dir + str(previous_run))
                    os.remove(discriminator.checkpoint_dir + str(previous_run))
                except OSError:
                    pass
            # update the previous run
            previous_run = current_run

    # generate a run id
    current_run = RunBuilder.generate_run_id(args)

    # plot and save summary plot
    plot(train_history, current_run)

    # save models
    discriminator.save(current_run)
    generator.save(current_run)
