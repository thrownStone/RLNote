# 20.12.7
moop：my own opinion

# 20.12.12
reinforcement learning chapter2 主要内容：搜索与贪心的平衡（balancing exploration & exploit）
相关方法：

- 概率贪心；
- softmax-->基于每个action的value改变action被选中的概率、pursuit-->缓慢贪心；
- 使当前最优的action被选中的概率逐渐逼近1、其他的逼近0；
- reinforcement comparison 也是改变概率，但是改变概率的方式与softmax略有差别，引入了preference值

# 20.12.13
any problem of learning goal-directed behavior can be reduced to three
signals passing back and forth between an agent and its environment: 

- one signal to represent the choices made by the agent (the actions), 
- one signal to represent the basis on which the choices are made (the states), 
- one signal to define the agent’s goal (the rewards)

where the return, Rt , is defined as some specific function of the reward sequence。In the simplest case
the return is the sum of the rewards: Rt = rt+1 + rt+2 + rt+3 + · · · + rT

The value functions V π and Qπ can be estimated from experience.

π policy? probability?

chapter3的主题是描述适用强化学习的问题特征

# 20.12.14
## chapter3的部分总结性摘抄
Reinforcement learning is about learning from interaction how to behave in order to achieve a goal.

- the actions are the choices made by the agent; 

- the states are the basis for making the choices; 

- the rewards are the basis for evaluating the choices.

A policy is a stochastic rule by which the agent selects actions as a function of states

The undiscounted formulation is appropriate for episodic tasks

the discounted formulation is appropriate for continuing tasks

Most of the current theory of reinforcement learning is restricted to finite MDPs

value function->用于确定选择哪个action？
“A policy’s value functions assign to each state, or state–action pair, the expected 
return from that state, or state–action pair, given that the agent uses the policy. The
optimal value functions assign to each state, or state–action pair, the largest expected
return achievable by any policy. A policy whose value functions are optimal is an
optimal policy.”完全没理解

# 20.12.21
## 《reinforcement learning》1.1

rl2大特点：trial-and-error search & delayed reward

rl不去描述学习方法，而是去刻画问题。去刻画agent在与环境交互以实现目标过程中所面对的问题的最重要方面。这就要求agent在一定程度上能够感知environment的state，并采取可以改变当前状态的action。由此，公式表达中应包含：state、action和goal

# 20.12.22
## 《reinforcement learning》
### 1.1
监督学习需要外部提供已经被判断好的样本。但是在交互场景中，很难提供既正确又有代表性的行为样本。（代表性指的是能代表所有agent需要交互的场景下正确的行为）

rl的难点之一就是平衡explore和exploitation。单独使用二者其一都无法成功完成任务。

rl的另一个特点是它明确考虑了agent和一个未知环境交互的整个问题。其他的方法只考虑子问题而忽略如何使子问题有效的应用到更完整的场景中（fit into larger picture）

agent需要一个清晰的目标，即：agent可以通过当前直接的感受来判断自己实现这个目标的进度

### 1.2
rl中，agent可以通过经验来不断完善（refine）自己的技巧

### 1.3
rl系统的四个主要子元素：policy、reward function、value function、model of environment

policy决定了在给定的时间内agent的行为方式。policy是这样一种映射，能够实现：感知到的state of environment --> action in that state。policy是rl中agent的核心，只靠policy agent就足够决定行为。一般情况，policy可能是随机的。

reward func定义了rl问题中的目标（moop：优化问题中的目标函数）。reward func是一个状态立即、固有的价值（原文：desirability）。
agent无法改变reward func，reward func是修改policy的依据。如果在某个policy下，做出的action收获了较低的reward，那么可以修改policy。一般情况，reward func是随机的。

value func是一个状态长远情况下的价值（desirability）。某一个状态本身的价值可能不高，但他后续状态的价值可能很高，这就会使这个状态具有一个high value和一个low reward。

reward和value的关系。reward是主要的，value是次要的。但是，我们以value作为选择动作（action choice）的依据，因为value包含着未来更多的reward。

因此，value的目的是获取更多的reward，如果没有reward，也就没有value。value是某个状态预期的reward。

事实上，rl算法中最重要的组成部分就是能够有效估计value的方法。

model用来预测{state，action}所导致的下一个状态和reward。model：{state action} -predict-> {state' reward}

BIngo游戏的例子
state	棋盘中可能出现的任意一种情况
policy	在状态下采取行动的规则；	每个policy有对应的取胜概率，这个概率从大量的experience中得来

### 2.1 老虎机游戏
evaluative feedback能够指出action有多好或多坏，但是无法告知是否是最好的或最坏的

老虎机游戏

机器有n个杆（lever），每轮选一个杆，每个杆都有对应的期望/平均收益。

一次选择叫做一次play

lever对应的expected reward当作action的value

每次做选择时，如果选择estimated value最大的action（即greedy action）认为是exploit

如果不做greedy action则认作是在explore

explore和exploitation的平衡在于：估计的value、不确定度、剩余的机会数

### 2.2  action-value method
列举了2种评估value的简单方法，指出了纯粹的explore不如explore和exploitation的结合
Q*(a)：true value
Q(a)：estimated value
sample-average method：根据大数定律，当次数足够多的时候，Q收敛于Q*
epsilon-greedy：以epsilon的概率进行explore，等概地从action中选择一个action。该方法的好处是，当尝试次数很多时，所有的action的Q都会收敛于Q*

### 2.3 softmax selection
epsilon-greedy存在的不足是等概地从action中进行选择，更合理的方式应该是Q值更高具有更多被选中的机会。引入了Boltzmann distribution，确定action的选中概率
softmax中存在的问题是温度系数tao难以确定。tao越高，各个action的选中概率越接近，tao越小高Q值action被选中的概率越高
softmax和epsilon哪个更好还不清楚

### 2.4 evaluation vs instruction
由于每个action做完之后只能得到一个reward，所以无法确定该action是否正确，即该action是否是最优解。又因为正确性是可以通过尝试所用的action并比较他们的reward就能够得到的，因此，这类问题天然地要求对可选择的action进行搜索。
instruction learning中的agent会被明确告知哪个action是正确的选择，因此agent不需要在action中进行搜索，但需要在参数空间（parameter space）中进行搜索。
因此，在action选择的规则修正方面，一条简单的instruction就可以直接用来指导规则的修改，但是在evaluation中，必须在比较完其他action之后，才能做出推断（inference）

# 20.12.23
## 《reinforcement learning》
### 2.4 evaluation vs instruction
binary bandit
该问题中，只有两个动作以及对应的2个可能的结果：成功 or 失败

如果每个action对应的reward是固定的，那么supervised method可以发挥得很出色；但如果对应的reward是以概率出现的，那么supervised method将有可能在两个action中震荡（eg2个action的成功概率都小于0.5或大于0.5）。

Lrp是一种类似于supervised的方法，他的原理是：如果某个action的reward是success，那么选择这个action的概率会以一定的速度（alpha）向1增加，同时另一个action的选择概率减小相同的大小；如果该action的reward是failure，则概率以同样的速度向0减小。

Lri与Lrp原理基本相同，唯一的区别是只在取得success时进行概率的更新操作，若是failure则直接忽略。

binary bandit task是一个非常有指导意义的特殊的混合了supervised和reinforcement2方面的例子。在某些问题中，supervised发挥的不错，但在有些问题中表现很差。这说明对于有的问题，我们需要更合适的方法。

### 2.5 Incremental Implementation

Qk+1 = Qk + 1/(k+1)[rk+1 - Qk]

NewEstimate←OldEstimate + StepSize [Target − OldEstimate].

Target只是假设的，实际上target有可能是noise

### 2.6 Tracking a Nonstationary Problem

之前考虑的问题中，环境是静态的；但在实际问题中，环境多是动态的。为了更有效的评估Q值（value of action），应当赋予不同时刻的采样值不同大小权重。

最新获得的采样值拥有更大的权重，之前获得的采样值则应被赋予较小的权重。

一种方式就是指数渐进权重平均：(1-alpha)^k*Q0 + sum(alpha(1-alpha)^k-i*ri)。其中，Q0是初始估计值，alph则是step size

### 2.7 Optimistic Initial Values

Q0赋值出现偏差，会在后续引起bias。在sample average中，只要每个action都被选中一次，这个bias就可以被解决；但是在恒定alpha中，bias会永远存在，但是会逐渐减小。
将Q0设置的比较高（optimistic initialization），也能起到鼓励explore的效果。因为，每个action被选中后，更新后的Q(a)都会小于其他的Q0。
乐观初始化并不适用于非静态的环境，因为初始情况只会出现一次，所以我们不能过于重视它。

### 2.8 Reinforcement Comparison

如何判断一个action的reward是大还是小，这就需要一个参考依据，即reference reward。一种直接的想法是用之前收获的reward的平均值作为reference。

大于reference reward即判定为高reward，小于reference reward即判定为低reward。基于这种方式的学习方法就被称为reinforcement comparison。

comparison不对action的value进行估计，而是维持对reward的总体估计。为了挑选action，该方法会维护对各个动作的preference，即p(a)。该p(a)会与softmax结合从而确定每个action被选中的概率。

# 20.12.24
## 《reinforcement learning》

























