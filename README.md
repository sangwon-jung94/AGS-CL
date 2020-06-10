# [Continual Learning with Node Importance based Adaptive Group Sparse Regularization (AGS-CL)](https://arxiv.org/abs/2003.13726) 

**Sangwon Jung, Hongjoon Ahn, Sungmin Cha and Taesup Moon**

**[M.IN.D Lab](https://mindlab-skku.github.io), Sungkyunkwan University**

------

## **Implementation Details**

### 1. Supervised Learning

### Requirements

- Python 3
- Pytorch 1.2.0+cu9.2 / CUDA 9.2

#### 1) Clone git

```
$ git clone https://github.com/sangwon79/Continual-Learning-with-Node-Importance-based-Adaptive-Group-Sparse-Regularization.git
```



#### 2) Download dataset

​	dataset link

#### 3) Execution command

```
$ python3 ./main.py ~~
```



------

### 2. Reinforcement Learning (Atari)

### Requirements

- Python 3.6
- Pytorch 1.2.0+cu9.2 / CUDA 9.2
- OpenAI Gym, Baselines

#### Notes

This code is implemented by reference to [pytorch-a2c-ppo-acktr-gaail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) 

#### 1) Clone git

```
$ git clone https://github.com/sangwon79/Continual-Learning-with-Node-Importance-based-Adaptive-Group-Sparse-Regularization.git
```



#### 2) Install OpenAI Gym, Baselines

​	Follow below links for installation

​	[OpenAI Gym](https://github.com/openai/gym#installation), [Baselines](https://github.com/openai/baselinesn)

#### 3) Implemetation commend

```
# Fine-tuning
$ CUDA_VISIBLE_DEVICES=0 python3 main_rl.py --approach 'fine-tuning' --seed 0 --date 200605  

# EWC
$ CUDA_VISIBLE_DEVICES=3 python3 main_rl.py --approach 'ewc' --seed 0 --date 200605 

# AGS-CL
$ CUDA_VISIBLE_DEVICES=3 python3 main_rl.py --approach 'gs' --seed 0 --date 200605 --gs-mu 0.1 --gs-lamb 1000
```

## **Citation**

```
@article{jung2020adaptive,
  title={Adaptive Group Sparse Regularization for Continual Learning},
  author={Jung, Sangwon and Ahn, Hongjoon and Cha, Sungmin and Moon, Taesup},
  journal={arXiv preprint arXiv:2003.13726},
  year={2020}
}
```

