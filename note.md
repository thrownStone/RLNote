# 20.12.7
moop：my own opinion

公式格式展示：

When $( a \ne 0 )$, there are two solutions to $(ax^2 + bx + c = 0)$ and they are:
$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$
	
$$
\begin{aligned}
\dot{x} & = \sigma(y-x) \\
\dot{y} & = \rho x - y - xz \\
\dot{z} & = -\beta z + xy
\end{aligned}
$$

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

reward func定义了rl问题中的目标（moop：优化问题中的目标函数）。reward func是一个状态**立即、固有**的价值（原文：desirability）。
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
介绍了1种评估value的简单方法，指出了纯粹的explore不如explore和exploitation的结合

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

一种方式就是指数渐进权重平均：(1-alpha)^k*Q0 + sum(alpha(1-alpha)^k-i*ri)。其中，Q0是初始估计值，alpha则是step size

### 2.7 Optimistic Initial Values

Q0赋值出现偏差，会在后续引起bias。在sample average中，只要每个action都被选中一次，这个bias就可以被解决；但是在恒定alpha中，bias会永远存在，但是会逐渐减小。

将Q0设置的比较高（optimistic initialization），也能起到鼓励explore的效果。因为，每个action被选中后，更新后的Q(a)都会小于其他的Q0。

乐观初始化并不适用于非静态的环境，因为初始情况只会出现一次，所以我们不能过于重视它。

### 2.8 Reinforcement Comparison

如何判断一个action的reward是大还是小，这就需要一个参考依据，即reference reward。一种直接的想法是用之前收获的reward的平均值作为reference。

大于reference reward即判定为高reward，小于reference reward即判定为低reward。基于这种方式的学习方法就被称为reinforcement comparison。

comparison不对action的value进行估计，而是维持对reward的总体估计。为了挑选action，该方法会维护对各个动作的preference，即p(a)。该p(a)会与softmax结合从而确定每个action被选中的概率。

# 20.12.27
## 《reinforcement learning》
### 2.8 Reinforcement Comparison
每次play结束之后，会更新action被选中的概率，更新的公式为：

p_t+1(a_t) = p_t(a_t) + beta * (r_t - reference_t)

每次play之后，reference_t的更新服从：

reference_t+1 = reference_t + alpha * (r_t - reference_t)

两个公式中的alpha和beta均为正数step-size parameter，其中，alpha属于(0,1]。

### 2.9 Pursuit Methods
pursuit同时结合了action-value和comparison2种方法的特点，即：既维护每个action的Q(a)，同时也维护对每个action的preference。

value和preference的关系是：preference对value是贪心的。

pursuit最简单的一种情况是：preference即动作被选择的概率。pi_t(a)表示action被选中的概率，那么action被更新的公式为：

pi_t+1(a) = pi_t(a) + beta * (1 - pi_t(a)) if a = a*

pi_t+1(a) = pi_t(a) + beta * (0 - pi_t(a)) else

其中，a*表示具有最大value的action

### 2.10 Associative Search
associative search是寻找一种映射关系：给定一个situation，能够得到在当前situation时应该选择哪个action。

一个例子，改进的n-arm bandit game。每轮每个action的value分布都会发生改变，但是当value发生改变时，machine的color也会变（相当于提供了一个具有标识作用的clue）。这样，可以统计得到每个color下，action的value分布。

moop：这里的situation与前文中提到的state（state of environment）类似。但是state会受到action的影响，本例中situation的变化与action无关。这也是书中提到的“If actions are allowed to affect the next situation as well as the reward, then we have the full reinforcement learning problem.”


### 2.11 Conclusion
chapter2主要介绍了几种通过改变action被选中的概率来平衡explore和exploitation的方法。

一个promising的idea是用观测值Q的不确定度为依据，来鼓励explore。不确定度越高的Q所对应的action，越有价值被explore。

## 3 The Reinforcement Learning Problem
Our objective in this chapter is to describe the reinforcement learning problem in a broad sense

We introduce key elements of the problem’s mathematical structure, such as value functions and Bellman equations.
### 3.1 The Agent–Environment Interface
elements:

- agent: learner & decision maker
- environment: everything outside learner
- state: the representation of environment. s_t belongs to S
- action: a_t belongs to A(s_t). state & action 都是在离散时间上发生的
- policy: a mapping from state to PDF of actions. 特定状态下，每个动作被选中的条件概率分布(p(a_t=a|s_t=s))
- interact: agent对environment做出的一系列action以及environment对action的回应和状态的改变

The agent-environment boundary represents the limit of the agent's absolute control

判断属于agent还是environment的标准是agent是否绝对控制该thing。例如，reward computation虽然是已知的，但是agent无法改变它，所以属于environment。

# 20.12.28
## 《reinforcement learning》
### 3.1 The Agent–Environment Interface
所有学习目标导向行为的问题都可以被简化成3个signals：

- action: signal to represent the choice made by the agent.
- state: sgnal to represent做出决定所基于的basis.
- reward: signal to define the agent's goal.

如何represent action & state 会强烈地影响到agent的performance。目前，如何合理的represent更多的是一门艺术而非科学。

#### example 3.1：Bopreactor 生物反应器

agent需要确定reactor每分钟的温度以及搅拌速率，从而使得反应产物的生成速率最快。在这个问题中：

- action: [temperature; rate]
- state: [传感器读数; 加入的原料数量]
- reward: 反应产物生成速率

#### example 3.2: Pick-and-Place Robot 捡东西的机器人

agent控制机器重复pick和place物品的动作，目标是让动作fast and smooth。在这个问题中：

- action: 在每个关节发动机上加载的电压
- state: 关节处的角度和速度
- reward: 快速平滑的拾放物品，如果出现急停等现象，可以赋予负reward

#### example 3.3: Recycling Robot 回收机器人

agent负责做出high level decision，让机器人去寻找空易拉罐 or 静止 or 去充电。在这个问题中：

- action: [search; stay; recharge]
- state: [当前的电量; 当前有无易拉罐]
- reward: 拿到can赋予正分，耗尽电量赋予大额负分

### 3.2 Goals and Rewards
agent追求的不是局部最优（max immediate reward），而是全局最优（max cumulative reward）

用reward signal来表示目标是rl区别于其他learning的最大特点

reward signal的作用是使用者告诉agent你想要它实现什么，而不是告诉它如何去实现。

“In particular, the reward signal is not the place to impart to the agent prior knowledge about how to achieve what we want it to do. For example, a chess-playing agent should be rewarded only for actually winning, not for achieving subgoals such as taking its opponent抯 pieces or gaining control of the center of the board.”

关于boundary between agent and environment：“the boundary of the learning agent not at the limit of its physical body, but at the limit of its control”

### 3.3 Returns

我们希望max expected return，此时，Rt是reward序列的函数，最简单的情况是：

	Rt = rt+1 + rt+2 + ... + rT

- 对于有明确的terminal state的问题，T是有限的，这种问题成为episodic task；
- 对于没有terminal state的问题，T是无穷的，这种问题成为continual task；

为了让continual task的Rt收敛，Rt的计算形式改写为下示：

	Rt = rt+1 + gamma * rt+2 + gamma^2 * rt+3 + ... + gamma^(T-t-1) * rT

gamma称为衰减率(discount rate)，gamma属于[0, 1]。gamma越接近1，说明agent看得越远。

#### example 3.4: Pole-Balancing
小车上用铰链固定一根杆，如果杆倾斜超过一定角度或者小车撞到墙壁即认为failure。

该问题既可看作episodic task也可以看作是continual task：

- episodic: 失败即terminal state
- continual: 每次失败，gamma的k重新开始计数

# 20.12.30
## 《reinforcement learning》
### 3.4 Unified Notation for Episodic and Continuing Tasks
这一节的内容是统一表示episodic task 和 continual task

episodic task中存在很多episodes，所以用符号s_t,i来表示：t时刻第i个episode的状态。

因为在实际讨论中，经常只讨论1个episode或者讨论所有episodes中都共通的东西，因此将s_t,i简写为s_t。

为了能统一表示，将episode的termination变为进入一个reward为0的吸收态（absorbing state）。

根据以上规定，可以将return改写为：

	Rt = sum(k=0, T)gamma^k * r_t+k+1

其中，T=∞与gamma=1不可以同时成立。

### 3.5 The Markov Property
这一节讨论的是要求state signal提供什么，以及什么是state signal不能提供的

environment和state signal有一个很有趣的性质——Markov property

state signal应该包含瞬时的感知信息，但不应该只包含这些。state representation可以是根据之前一系列sensation而构建的更复杂的形式。

In a word，当前的state应当包含之前state的信息（Markov property）

state signal不必告诉agent所有的environment information，即使一些information对解决问题很有帮助。我们更在意state signal的memory。“In short, we don't fault an agent for not knowing something that matters, but only for having
known something and then forgotten it”

“Much of the information about the sequence is lost, but all that really matters for the future
of the game is retained.”不在乎过去所有的information，只在乎会影响未来状态的information。
后续状态的发展与之前的“path”无关，只与当前的state有关。（Markov property）

	{s_t, a_t} --> r_t+1

	P{s_t+1=s', r_t+1 = r' | s_t, a_t, r_t, s_t-1, a_t-1, ..., r1, s0, a0} = P{s_t+1=s', r_t+1 = r' | s_t, a_t}

### 3.6 Markov Decision Processes
	transition probabilities: P^a_ss' = P{s_t+1 = s' | s_t = s, a_t = a}

	expected value of the next reward: R^a_ss' = E{r_t+1 | s_t = s, a_t = a, s_t+1 = s'}

#### Example 3.7: Recycling Robot MDP

robot的state set 和 action set 如下所示：

- state set: {low, high}
- action set: A(high) = {search, wait}; A(low) = {search, wait, recharge}

原文中列出了转移概率和期望回报值。在本例中，期望回报值可以理解为每种动作平均可以得到的can。对于low-->high的情况，回报值为-3，
作为对耗尽电池的惩罚。

MDP transition graph包含2种节点：

- state node
- action node

### 3.7 Value Functions

value function是根据state计算这个satae的预期收益，即expected return。expected return可以评估一个state的好坏。

value function和具体的policy有关 （policy --> action's PDF

state-value function的计算公式如下：

	V^pi(s) = E_pi{Rt | s_t = s} = E_pi{sum(k=0, T)gamma^k * r_t+k+1 | s_t = s}

需要注意的是，terminal state的value永远是0.

action-value function: {s_t=s, a_t = a} --> Rt

	Q^pi(s, a) = E_pi{Rt | s_t = s, a_t = a}

获得V^pi 和 Q^pi 有两种方法：Monte Carlo 以及 parameterized function

rl中的value function的基本特性是满足某种迭代关系(与DP相同)

Bellmen Equation基于个人理解的推导:

	V^pi(s)

	= E_pi{Rt | s_t = s}

	= E<Rt>{Rt | policy = pi, s_t = s}						<>表示求均值的变量

	= E<r_t+1>{r_t+1 + gamma*Rt | policy = pi, s_t = s}		(注意，这里的Rt时刻从t+2开始

	= E<a>E<s'>E<r_t+1>{r_t+1 + gamma*Rt | policy = pi, s_t = s, a_t = a, s_t+1 = s'}

	= E<a>E<s'>E<r_t+1>{r_t+1| policy = pi, s_t = s, a_t = a, s_t+1 = s'} + E<a>E<s>E<r_t+1>{gamma*Rt | policy = pi, s_t = s, a_t = a, s_t+1 = s'}

	= E<a>E<s'>{R^a_ss'| policy = pi} + E<a>E<s>{gamma*Rt | policy = pi, s_t = s, a_t = a, s_t+1 = s'}

	= E<a>E<s'>{R^a_ss'| policy = pi} + E<a>E<s>E_pi{gamma*Rt | s_t+1 = s'}

	= E<a>E<s'>[R^a_ss' + gamma*V^pi(s')]

# 20.12.31
## 《reinforcement learning》
### 3.7 Value Functions
	backup diagram
	s_t   		s
	    	   / \
	a_t 	  a' a''
	r_t+1     |   |
	s_t+1     s'  s''

### 3.8 Optimal Value Functions
optimal pi*: 

	pi* = max pi. (pi >= pi', only if for all s V^pi(s) >= V^pi'(s))

optimal state-value function V*:

	V*(s) = max<pi> V^pi(s). (for all s)

optimal action-value function Q*:

	Q*(s, a) = max<pi> Q^pi(s, a). (for all s and a)
	Q*(s, a) = E{r_t+1 + gamma * V*(s_t+1) | s_t = s, a_t = a}
			 = E<s'>{R^a_ss' + gamma * V*(s') | s_t = s, a_t = a}

relationship between V* and Q*:

	V*(s) = max<a> Q^pi*(s, a)		(PS. 个人认为这里的Q^pi*可以直接用Q*表示，因为根据上文提到的Q*的公式可以确定，使Q取到最大值的pi就是pi*，因此Q* = Q^pi*)
	这个等式之所以成立，是因为V*是V的上界。只有当policy只为能取最大值的action分配概率，其他action为概率0时，才能达到V的上界即V*

Bellman optimality equation for V*：

	V*(s) = max<a> E<s'>{R^a_ss' + gamma * V*(s') | s_t = s, a_t = a}

bellman equation for Q*:

	Q* = max<pi> Q^pi(s, a)
	Q*(s, a) = E<s'>{R^a_ss' + max<pi> [ gamma * V*(s') ] | s_t = s, a_t = a}
			 = E<s'>{R^a_ss' + max<pi> [ gamma * max<a'> Q^pi*(s', a')]}
			 = E<s'>{R^a_ss' + max<a'> [ gamma * max<pi> Q^pi*(s', a')]}
			 = E<s'>{R^a_ss' + max<a'> [ gamma * Q*(s', a')]}
			 （= max<a'> E<s'>{R^a_ss' + [ gamma * Q*(s', a')]}）


# 21.1.13
## 《reinforcement learning》
### 3.8 Optimal Value Functions

figure 3.7 直观解释了V* 和 Q*的表达式含义

Bellman optimality equation包含两部分：V* 和 Q*

在实际问题中，Bellman equation实际是一个系统方程，每一个state都有一个对应的方程

确定了R^a_ss'和P^a_ss'就可以确定一个environment的dynamics

一旦确定了V* 就可以很容易的制定optimal policy。If you have the optimal value function, V∗, then the actions that appear best after a one-step search will be optimal actions.

确定了V* 后，任何一种追求短期reward的贪心算法都将会是optimal policy，因为**V\* 将长远optimal expected return转变成了一个数，一个在每个state下都可以立即确定的数**

一旦确定了Q* 就可以很容易的选出最优action。因为Q* function有效的将所有一步搜索的结果缓存了下来（cache）。**Q\*把optimal expected long-term return转变成了一个数，一个在每个(state, action)下都可以立即得到的数**


# 21.1.14
## 《reinforcement learning》
### 3.8 Optimal Value Functions

#### example 3.11: Bellman Optimality Equations for the Recycling Robot
V* Bellman optimal formulation示例，推导过程中中间过程有标记错误

直接求解Bellman optimality equation类似于枚举搜索，需要考虑所有的可能、计算这些可能出现的概率以及期望reward。

这种解法需要至少满足3个条件/假设才可以：

	1. 准确地知道environment的dynamics（P^a_ss' & R^a_ss'）
	2. 有足够的计算资源来完成解的计算
	3. 是马尔可夫过程

但是在应用中，这3种假设彼此间常常是冲突的。例如，在都满足1和3时，2可能无法满足。

许多不同的决策方法可以视作是Bellman optimality equation的近似解，动态规划（DP）与bellman optimality equation关系更加密切。

RL中的一些方法也可以看作是Bellman optimality equation的近似解，在这些方法中，使用实际经验转移（actual experienced transition）代替了期望转移（expected transition）

### 3.9 Optimality and Approximation

实际问题中，agent很少能够学到optimal policy。

一个好的关于optimality的定义把书中所描述的学习方法组织了起来（organize），并且也提供了一种理解不同learning algorithms理论性质的方法。但是optimality是ideal，agent无法达到，只能不同程度上去逼近。

agent在学习过程中面临的一个非常critical的问题是：**算力限制（computation power）**。尤其是在**一步决策中（a single time step）**所拥有的computation power。

**存储能力**也是一个重要的限制（**memory available**）。当state数量比较少时，可以使用表格或数组来记录数据，这种情况称为tabular case。但在实际问题中，state的数量往往比较多，因此需要使用包含更多参数的方程来表示Bellman optimality equation。

我们对RL问题的框架（framing）设计强迫我们必须解决近似问题。

由于RL的on-line nature，RL方法往往会花费更多的精力去学习为经常遇到的state做出good decision，代价是更少的关注不经常出现的state。
**这是区分RL和其他逼近MDP问题的关键性质（key property）**

### 3.10 Summary

- The reinforcement learning agent and its environment interact over **a sequence of discrete time steps**
- the actions are the choices made by the agent; 
- the states are the basis for making the choices; 
- the rewards are the basis for evaluating the choices.
- A policy is a **stochastic rule** by which the agent **selects actions** as a **function of states**

policy是选择action的rule，policy的具体formulation由Q*(s, a)决定。

RL问题在agent获取knowledge程度不同时是不同的：

- complete knowledge：environment完全可知
- MDP：model包含一步转移概率和期望reward
- incomplete knowledge：无法获取关于environment的完美model

# 21.1.18
DRL参考网址：https://simoninithomas.github.io/deep-rl-course/#syllabus

补充：适合DRL的问题

	1. 非凸优化
	2. 缺少求解所需的基础信息
	3. 传统模型假设or限制条件过于严格，将限制条件放宽时可以使用DRL
	注：DRL的效果和参数设置、state、action、reward的构建都有关系，不同的配置会使最终的结果不同

## 《reinforcement learning》
### 4 Dynamic Programming

key idea of DP, the use of value functions to organize and structure the search for good policies

### 4.1 Policy Evaluation
如何得到approximation V(s)

Bellman Equation for Vk(s)在k趋近无穷时能够收敛到V^pi. 这种算法叫做iterative policy evaluation（为什么随意赋初始值，一定会在无穷处收敛？

V^pi本质是期望，根据backup图，可以直观地看出，只要最后存在终止状态 or 新增的量为无穷小，均值一定会收敛，这保证了V^pi的**存在**和**唯一**，即书中所说 “either γ <1 or eventual termination is guaranteed from all states under the policy π.”）

Bellman Equation for Vk(s)的更新操作是一种fullbackup（全备份）。每次迭代都会备份一次state的value用来计算新的估计值、

在实际应用中使用不可能无穷迭代下去，因此，要设置可以接受的差值，使用差值比较的方法来判断何时停止。

# 21.1.19
## 《reinforcement learning》
### 4.2 Policy Improvement
如何提升V(s)

Q^π(s, π(s)) ≥ V^π(s).

π‘(s) = arg max< a > Q^pi(s, a)

Policy improvement thus must give us a strictly better policy except when the original policy is already optimal

the policy improvement theorem carries:

	Q^pi(s, pi'(s)) = sum(a) pi'(s, a) * Q^pi(s, a)

### 4.3 Policy Iteration
实际应用中，不断迭代：逼近V(s) -> 提升V(s)

### 4.4 Value Iteration
value iteration：截断policy evaluation，减少搜索时间

formulation:

	Vk+1(s) = max<a> E{r_t+1 = gamma * Vk(s') | st = s, at = a}

### 4.5 Asynchronous Dynamic Programming
解决搜索状态空间耗时较长的问题，单位给出具体方法

### 4.6 Generalized Policy Iteration
再次阐述了iteration。当达到（接近）bellman optimal equation时稳定。

### 4.7 Efficiency of Dynamic Programming

### 5.1 Monte Carlo Policy Evaluation
Whereas the DP diagram (Figure 3.4a) shows all possible transitions, the Monte Carlo diagram shows only those sampled on the one episode.

An important fact about Monte Carlo methods is that the estimates for each state are independent

### 5.2 Monte Carlo Estimation of Action Values
If a model is not available, then it is particularly useful to estimate action values

The only complication is that many relevant state?action pairs may never be visited.

the first step of each episode starts at a state?action pair, and that every such
pair has a nonzero probability of being selected as the start. This guarantees that all
state?action pairs will be visited an infinite number of times in the limit of an infinit
number of episodes. We call this the assumption of exploring starts.

### 5.3 Monte Carlo Control
to approximate optimal policies

### 5.4 On-Policy Monte Carlo Control
How can we avoid the unlikely assumption of exploring starts？

On-policy methods attempt to evaluate
or improve the policy that is used to make decisions

### 5.5 Evaluating One Policy While Following Another

### 5.8 Summary

Monte Carlo的三种优势:

- First, they can be used to learn op
timal
behavior directly from interaction with the environment, with no model of the
environment抯 dynamics
- Second, they can be used with simulation or sample mod
els.
- Third, it is easy and efficient to focus Monte Carlo methods
on a small subset of the states.

In designing Monte Carlo control methods we have followed the overall schema of
generalized policy iteration (GPI)

Rather than use a model to compute the
value of each state, they simply average many returns that start in the state

In on-policy methods, the agent commits to always
exploring and tries to find the best policy that still explores. In off-policy methods, the
agent also explores, but learns a deterministic optimal policy that may be unrelated
to the policy followed.

### 6.1 TD Prediction
Monte Carlo methods must **wait until the end of the episode** to determine the increment to V (st) (only then is Rt known), TD methods need wait only until the next time step using the observed reward r_t+1 and the estimate V(st +1). **TD can work online**

# 21.1.21
## 《reinforcement learning》
### lecture 4 
TD can learn before knowing the final outcome

- TD can learn online after every step
- MC must wait until end of episode before return is known

TD can learn without the final outcome

- TD can learn from incomplete sequences
- MC can only learn from complete sequences
- TD works in continuing (non-terminating) environments
- MC only works for episodic (terminating) environments

MC: 

	V(St ) = V(St) + alpha * (Gt - V(St))
	high var, low bias

TD:

	V(St) =  V(St) + alpha * (Rt+1 + V(St+1) - V(St ))
	low var, high bias

关于MC TD 和DP的图示

eligibility trace用来解决可信度的问题。频繁发生的事件更可信还是最近发生过的事件更可信？ET将两种观点都考虑进去。原有的ET随时间衰减，若事件出现，则加上一个冲激。

### lecture 6
Estimate value function with function approximation how 提速？

为什么Δw = -1/2...？ A:应该是为了后面微分时好计算

on-policy: that is, they don’t use old data, which makes them weaker on sample efficiency.

off-policy, so they are able to reuse old data very efficiently

### lecture 7
learn policy directly from experience

### lecture 8
learn model directly from experience

### An Intro to DRL
State s: is a complete description of the state of the world (there is no hidden information). In a fully observed environment.

Observation o: is a partial description of the state. In a partially observed environment.

find this optimal policy (hence solving the RL problem) there are two main types of RL methods:

- Policy-based-methods: Train our policy directly to learn which action to take, given a state.
- Value-based methods: Train a value function to learn which state is more valuable and using this value function to take the action that leads to it.

The Q-Learning is the RL algorithm that

- Trains Q-Function, an action-value function that contains, as internal memory, a Q-table that contains all the state-action pair values.
- When the training is done, we have an optimal Q-Function, so an optimal Q-Table.

Off-policy: using a different policy for acting and updating.

On-policy: using the same policy for acting and updating.

DQN:

- Preprocessing is an important step. We want to reduce the complexity of our states to reduce the computation time needed for training.
- why we stack frames together?We stack frames together because it helps us to handle the problem of temporal limitation.
- Avoid forgetting previous experiences. Our solution: create a “replay buffer.” This stores experience tuples while interacting with the environment, and then we sample a small batch of tuple to feed our neural network. And Reducing correlation between experiences

CNN介绍：https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/

卷积是一种filter，可以提取image的不同特征，如：边缘、锐利等

https://zhuanlan.zhihu.com/p/21421729 DQN初步介绍及ReplayMemory出现的原因（从回忆中学习，避免相邻state间的相关性）

https://cs231n.github.io/convolutional-networks/（CNN）

https://www.zhihu.com/question/52668301/answer/194998098（YJango，CNN）

https://blog.csdn.net/zuolixiangfisher/article/details/89500624?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control（placeholder用法）

https://zhuanlan.zhihu.com/p/27609238（SGD讲解）

# 21.1.22

parameter & hyper parameter：

模型参数是根据数据自动估算的。但模型超参数是手动设置的，并且在过程中用于帮助估计模型参数。（https://zhuanlan.zhihu.com/p/37476536）


最初DQN的不足：

- However, the problem is that we using the same parameters (weights) for estimating the target and the Q value. As a consequence, there is a big correlation between the TD target and the parameters (w) we are changing.
- Therefore, it means that at every step of training, our Q values shift but also the target value shifts. So, we’re getting closer to our target but the target is also moving. It’s like chasing a moving target! This lead to a big oscillation in training.

Fixed DQN：
- 2 network （DQN & TargetNetwork）
- update Targetwork w- with DQN's parameter

DDQN：

- how are we sure that the best action for the next state is the action with the highest Q-value?
If non-optimal actions are regularly given a higher Q value than the optimal best action, the learning will be complicated.
- use our DQN network to select what is the best action to take for the next state
- use our target network to calculate the target Q value of the next state taking the mentioned action.

Dueling DQN：

- decompose Q(s,a) as the sum of:

	V(s): the value of being at that state  
	A(s,a): the advantage of taking that action at that state (how much better is to take this action versus all other possible actions at that state).


- This is particularly useful for states where their actions do not affect the environment in a relevant way. In this case, it’s unnecessary to calculate the value of each action.

Prioritized Exoerience Replay：

- take in priority experience where there is a big difference between our prediction and the TD target, since it means that we have a lot to learn about it.

**python规范：https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/#id16**

DRL architure

- construct environment(state space, action space)
- set up hyperparameters(learning_rate...)
- construct DQN
- other function or class. i.e, memory
- get enough sample into memory
- train agent/DQN
- test

### code problem

#### generate_channel_gain_centralized
- rho = 1 #scipy.special.jn(0, 2*math.pi*fd*T_time_slot)
- Fading_output = complex(rho * Fading_input.real + random.gauss(0, ((1 - rho**2) / 2)**(1/2))
- Channel_gain_output = Average_channel_gain * Fading_gain. Average_channel-gain?

#### Main.py
empty

#### Main_SR.py
- n_element_state = 2 * No_user + 4. what is n_element_state? ==> t-1时刻与t+1时刻的快衰落状态，以及另外4种状态
- n_network_input = (n_element_state, )?
- model_eval
- boundary = 20?
- inner_region = 5?

# 21.1.23
### code problem
#### Main_SR.py
- channel_gain_bd_user? SNR * | fast fading channel gain | ^ 2 ==> total channel gain
- channel_gain_bd_user_old, channel_gain_bd_user_new?
- fading_bd_user?
- channel_gain_bd_user_last1?
- state_old[4:No_user+4, u] = Channel_gain_BD_user_last2[u*No_user:(u+1)*No_user] **/ SNR_BD_Avg[u]**?
- Channel from BS to BD without transmission power. state_old[3, u] = Channel_gain_BS_BD[u]

# 21.1.26
### code problem
#### Main_SR.py
