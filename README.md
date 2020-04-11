# pytorch-gan-for-outlier-detection
This is the PyTorch Implementation of this paper:

[Generative Adversarial Active Learning for Unsupervised Outlier Detection](https://arxiv.org/abs/1809.10816) by Yezheng Liu, Zhe Li, Chong Zhou, Yuanchun Jiang, Jianshan Sun, Meng Wang and Xiangnan He.

This repository also corresponds to the code for the post I have written on Generative Adversarial Networks for Unsupervised Outlier Detection (link to the post will be added later).

## Dependecies
Install the requirements using this command:

```bash
sudo pip install -r requirements.txt
```

## Usage

Once dependencies are installed, you can open `main.py` and specify the following arguments :

```bash
--path [PATH]         		Input data path.
--stop_epochs STOP_EPOCHS   Stop training generator after stop_epochs.
--lr_d LR_D           		Learning rate of discriminator.
--lr_g LR_G           		Learning rate of generator.
--batch_size BATCH_SIZE     The training batch size.
--decay DECAY         		Decay.
--momentum MOMENTUM   		Momentum.
--plot_every PLOT_EVERY     Learning curves plotting frequency.
```
then run this command to train the model:

```bash
python main.py
```

## ...

Learning curves as well as the model's weights are saved respectively in `./plots` and `./chkpt`, every `plot_every` epochs, to track the training process. The `RunBuilder` class takes care of generating a unique identifier for each run based on the time it was executed on. As a TODO, I have add the possibility of hyper-parameters tuning.

## References

#### Generative Adversarial Active Learning for Unsupervised Outlier Detection

Yezheng Liu, Zhe Li, Chong Zhou, Yuanchun Jiang, Jianshan Sun, Meng Wang and Xiangnan He

https://arxiv.org/abs/1809.10816

#### GAAL-based Outlier Detection

Original implementation in Keras by *leibinghe*

https://github.com/leibinghe/GAAL-based-outlier-detection