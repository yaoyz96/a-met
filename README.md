# Detach and Unite: A Simple Meta-Transfer for Few-Shot Learning

This repo is for the paper 'Detach and Unite: A Simple Meta-Transfer for Few-Shot Learning'

## Prerequisites

The following packages are required to run the scripts:

- PyTorch 1.8
- Python 3.7

## Datasets

For Omniglot, the data split settings is proposed by [1,2]. For mini-ImageNet, the data split is proposed by [3].

[1] Vinyals O, Blundell C, Lillicrap T, et al. Matching networks for one shot learning[J]. Advances in neural information processing systems, 2016, 29.

[2] Lake B M, Salakhutdinov R, Tenenbaum J B. Human-level concept learning through probabilistic program induction[J]. Science, 2015, 350(6266): 1332-1338.

[3] Ravi S, Larochelle H. Optimization as a model for few-shot learning[C]//International conference on learning representations. 2017.

## Code Structures

There are five parts in the code:

- backbones: It contains the backbone network (Conv4 and ResNet-12) for the experiments.
- configs: Training settings.
- datasets: Dataloader and Samplers of different datasets.
- models: Baseline network and model register.
- utils: Help tools.


## Training scripts

- Pre-training

> python met.py --config train.yaml --gpu 0 --seed 1 --mode train --mannr whole 

- Train MET

> python met.py --config train.yaml --gpu 0 --seed 1 --mode train --mannr episode

- Train A-MET

> python amet.py --config train_amet.yaml --gpu 0 --seed 1 --mode train --mannr episode


## Acknowledgment

Our project references the codes in the following repos.

- [meta-baseline](https://github.com/yinboc/few-shot-meta-baseline)

- [matching networks](https://github.com/activatedgeek/Matching-Networks)
