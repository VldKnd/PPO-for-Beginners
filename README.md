<p align="center">
    <img src="img/ipp.png"
         alt="IPP"/>
</p>

The work have been done by Vladimir Kondratyev, Tom Reppelin and Paul Fayard.

## Introduction
The last years have brought a lot of attention to gradient policy based methods, due to modern popularity of neural network and development of auto-grad software. It can be seen by variety of new algorithms with neural network function approximation that are being introduced.

* Deep Q-Learning
* Vanilla Policy Gradient Methods
* Trust Region Policy Gradient Methods

These algorithms still have some challenges. Q-learning have been shown to fail on many tasks and is badly understood, Vanilla Policy Gradient method have poor data efficiency and not robust and TRPO is too complex.
The goal of our research article is to improve the state of affairs by giving an algorithm that has the data efficiency and reliable performance of TRPO  (Trust Region Policy Optimisation), but largely reduces the complexity of the algorithm. The authors propose two new surrogate objectives by introducing new types of regularisation and clipping . By bounding the difference between distributions after parameters update, the PPO allows to do the optimization of the policies by sampling first the trajectories of the given policy and then doing several steps of optimisation over them, which increases the data efficiency of the algorithm. 
Authors also discuss the empirical results and provide experiments.
The experiments are a comparison of the performances of various versions of previous surrogate objectives and show, that in many scenarios the clipped probability ratios performs the best. They also compare proximal policy optimization to several previous algorithms from the literature, suggesting that PPO performs significantly better on games like Atari in terms of sample efficiency.

# Report

This is just a brief introduction to our research work, you can find the main report either in /latex_source file, which is a folder with latex code or by looking at Report.pdf file, which has an already compiled document.

# Implementations

The goal of this experimental part is to understand, step by step, how to code our PPO from scratch in python. 
Not really knowing where to start, we looked at the existing codes already on GitHub. We then came across the PPO-for-Beginners repo, created by Eric Yu. He has developed a repo consisting of 5 python files, which code PPO from Scratch with PyTorch. Here is the link of the [repo](https://github.com/ericyangyu/PPO-for-Beginners).
He adds to his code three Medium articles, where he explains step by step and in a very intuitive way the different steps of its implementation. 
Rather than starting from scratch, we decided to focus on this repo, and on these 3 articles, in order to understand in detail the way Eric Yu implemented PPO. 

# Conclusion
The PPO article proposes the novel way to optimise the value function of agent with policy parameters by differential function. It improves on previously proposed method Trust Region Policy Optimisation, by utilising clipping of the gradient as a way to enforce closeness between distributions. This provides the possibility to re-utilise the batched sampled data multiple times, which strongly improves the data efficiency of TRPO, but keeps its reliable performance. Also the article proposes the answer to one of the problems with choice of hyper parameter, representing the impact of Kullback Leiber divergence between two policies in the loss function. Its does in automatic manner, improving on the TRPO method.
We saw that in terms of performance, PPO performs significantly better in the continuous domain, and in games like Atari in terms of sample efficiency (especially in terms of fast training).

## Usage

To train from scratch:
```
python main.py
```

To test model:
```
python main.py --mode test --actor_model ppo_actor.pth
```

To train with existing actor/critic models:
```
python main.py --actor_model ppo_actor.pth --critic_model ppo_critic.pth
```
This is the fork of the reposetory made by Eric Yu. 
For original code implementation please refer to  [Eric Yu repo](https://github.com/ericyangyu/PPO-for-Beginners) or [Medium article](https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8).
