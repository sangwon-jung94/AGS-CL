# [Continual Learning with Node Importance based Adaptive Group Sparse Regularization (AGS-CL)]() 

------

## **Execution Details**

### 1. Supervised Learning

### Requirements

- Python 3
- Cifar100/Cifar10/100 : GPU 1080Ti / Pytorch 1.3.1+cu9.2 / CUDA 9.2
- Omniglot : GPU Titan RTX / Pytorch 1.3.1 / CUDA 10.0
- CUB200 : GPU 1080Ti / Pytorch 1.0.0+cu9.2 / CUDA 9.2

#### 1) Download dataset

- Omniglot : https://drive.google.com/file/d/1WxFZQyt3v7QRHwxFbdb1KO02XWLT0R9z/view?usp=sharing
- CUB200 : https://github.com/visipedia/tf_classification/wiki/CUB-200-Image-Classification

#### 2) Execution command

```
# Cifar100
$ python3 ./main.py --experiment split_cifar100 --approach gs --lamb 400 --mu 10 --rho 0.3 --eta 0.9 

# Cifar10/100
$ python3 ./main.py --experiment split_cifar10_100 --approach gs --lamb 7000 --mu 20 --rho 0.2 --eta 0.9 

# Omniglot
$ python3 ./main.py --experiment omniglot --approach gs --lamb 1000 --mu 7 --rho 0.5 --eta 0.9 

# CUB200
$ cd LargeScale_AGS
$ python3 ./main.py --dataset CUB200 --trainer gs --lamb 1.5 --mu 0.5 --rho 0.1 --eta 0.9 
```

#### 3) Result(Average accuracy)

|        | CIFAR100 | CIFAR-10/100 | Omniglot | CUB-200 |
| :----: | :------: | :----------: | :------: | :-----: |
| AGS-CL |   64.2   |   76.0       |  82.2    |  82.9   |

------

### 2. Reinforcement Learning (Atari)

### Requirements

- Python 3.6
- Pytorch 1.2.0+cu9.2 / CUDA 9.2
- OpenAI Gym, Baselines

#### Notes

The experimental environment for reinforcement learning is built based on [pytorch-a2c-ppo-acktr-gaail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) 

#### 1) Install OpenAI Gym, Baselines

​	Follow below links for installation

​	[OpenAI Gym](https://github.com/openai/gym#installation), [Baselines](https://github.com/openai/baselinesn)

#### 2) Execution command

```
# Fine-tuning
$ CUDA_VISIBLE_DEVICES=0 python3 main_rl.py --approach 'fine-tuning' --seed 0 --date 200605  

# EWC
$ CUDA_VISIBLE_DEVICES=0 python3 main_rl.py --approach 'ewc' --seed 0 --date 200605 

# AGS-CL
$ CUDA_VISIBLE_DEVICES=0 python3 main_rl.py --approach 'gs' --seed 0 --date 200605 --gs-mu 0.1 --gs-lamb 1000
```
