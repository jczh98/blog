---
title: 《LEARNING TO OPTIMIZE》
updated: 2019-07-06 19:49:49
categories:
- paper
---
这篇paper提出了一种结合reinforcement learning与optimize的机器学习objective function的优化算法寻找方法。文章首先提出了一种general的optimize框架，即优化为设计一个$\pi$函数，使得每次迭代的$\Delta x$由$\pi$函数决定。常见的optimize函数的$\pi$: Gradient Descent: $-\gamma \nabla f(x^{i-1})$, Momentum: $-\gamma \sum_{j=0}^{i-1}\alpha^{i-1-j}\nabla f(x^{i-1})$ 。让机器学习优化算法便规约到了让寻找特定objective function下的$\pi$函数下，这个函数本身可以被看做为目标函数的优化策略，所以可以结合reinforcement learning里的policy method去求解这样的一个问题。

## Guide policy search

前面介绍了一些related work诸如meta learning, program induction, hyperparameter optimization之类的，大意是：你们的方法很好，但在我们的task上无用或只在special task上起作用（希望我没读错）。然后再介绍了一堆reinforcement learning的背景知识，比如markov decision process之类的前置技能。然后花了一页讲了这篇paper的核心，guide policy search。

这是一种policy method，主要是为了解决model-free方法再大型网络里效率不高的问题而提出的的一种结合model-free search和model-based search的方法。通过不断交替计算trajectories和训练策略网络逐步迭代（怎么这么像GAN呢）。GPS的数学描述如下
$$
\min_{\theta,\eta} \mathbb{E} [\sum_{t=0}^{T}c(s_t)] \quad \text{s.t.} \quad \psi(a_t|s_t,t;\eta)=\pi(a_t|s_t;\theta) \forall a_t,s_t,t
$$
即在限制stationary policy与time-varying target policy相等下最小化一定trajectories上的cost的均值，假定$\psi$在每个时刻状态上服从conditionally Gaussian(没看懂…)，并把约束放宽到$\psi$与$\pi$两者的KL-divergence足够小。

对于更新$\psi$（轨迹相）,用二次函数近似cost,即$c(s_t) \approx {1\over 2}s_t^TC_ts_t+d_t^Ts_t+h_t$，之后展开式子之后可用dp方法解决

对于更新$\pi$（监督相）,GPS最小化KL-divergence.假定固定协方差忽略多重变量。
$$
\mathbb{E}_{\psi}\left[\sum_{t=0}^{T}\left(\mathbb{E}_{\pi}\left[a_{t} | s_{t}\right]-\mathbb{E}_{\psi}\left[a_{t} | s_{t}, t\right]\right)^{T} G_{t}^{-1}\left(\mathbb{E}_{\pi}\left[a_{t} | s_{t}\right]-\mathbb{E}_{\psi}\left[a_{t} | s_{t}, t\right]\right)\right]
$$
Note:没有经过仔细推导，感觉数学还是太弱了，有时间一定好好打数理基础。

## Experiments

大概就用GPS去建模目标函数的$\pi$ ，主要用一个50个hidden layer的小型网络拟合$\pi$，实现细节似乎与GPS没什么差别，在logistic regression, robust linear regression, neural net classifer上进行了不同的实验，与Gradient Descent， Momentum， Conjugate Gradient， L-BFGS进行对比实验。大概可以看出收敛得比较快，收敛值比较高。

## 主要贡献

- 提出了解决optimize的一种新思路，即使在AutoML盛行的2019年里，我浅薄的知识也很少看到相关的work
- 将强化学习中policy search用在了自动化搜索方面，或许可以扩展出强化学习应用在神经网络搜索里，不过这似乎属于遗传算法/进化算法的内容？

(才疏学浅，看不出来了)

## 思考

- 优化方法了解得比较少，优化方法真的如paper开头里提出的一样能用统一框架描述吗？
- ICLR近几年的主题都有optimize相关的paper，hand-enginered algorithm真的比不过搜索出来的方法吗？
