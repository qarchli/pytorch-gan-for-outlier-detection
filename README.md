# pytorch-gan-for-outlier-detection
This is the PyTorch Implementation of this paper:

[Generative Adversarial Active Learning for Unsupervised Outlier Detection](https://arxiv.org/abs/1809.10816) by Yezheng Liu, Zhe Li, Chong Zhou, Yuanchun Jiang, Jianshan Sun, Meng Wang and Xiangnan He.

This repository also corresponds to the code for the post I have written on Generative Adversarial Networks for Unsupervised Outlier Detection (link to the post will be added later).

## Usage

Once dependencies are installed, you can run this command to train the model.

```bash
python main.py --path data/onecluster --stop_epochs 1000 --lr_d 0.01 --lr_g 0.0001 --decay 1e-6 --momentum 0.9
```

Running `python main.py -h` get you more details about the arguments:

```bash
-h, --help            		show this help message and exit
--path [PATH]         		Input data path.
--k K                 		Number of sub_generator.
--stop_epochs STOP_EPOCHS   Stop training generator after stop_epochs.
--lr_d LR_D           		Learning rate of discriminator.
--lr_g LR_G           		Learning rate of generator.
--decay DECAY         		Decay.
--momentum MOMENTUM   		Momentum.
```

## References

#### Generative Adversarial Active Learning for Unsupervised Outlier Detection

Yezheng Liu, Zhe Li, Chong Zhou, Yuanchun Jiang, Jianshan Sun, Meng Wang and Xiangnan He

https://arxiv.org/abs/1809.10816

#### GAAL-based Outlier Detection

Original implementation in Keras by *leibinghe*

https://github.com/leibinghe/GAAL-based-outlier-detection