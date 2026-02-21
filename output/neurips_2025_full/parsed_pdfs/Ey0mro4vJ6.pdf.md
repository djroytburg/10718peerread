## Continuous Soft Actor-Critic: An Off-Policy Learning Method Robust to Time Discretization

## Huimin Han

Zhongtai Securities Institute for Financial Studies Shandong University Jinan, 250100 P. R. China hanhuiminhhm@mail.sdu.edu.cn

## Shaolin Ji ∗

Zhongtai Securities Institute for Financial Studies Shandong University

Jinan, 250100 P. R. China jsl@sdu.edu.cn

## Abstract

Many Deep Reinforcement Learning (DRL) algorithms are sensitive to time discretization, which reduces their performance in real-world scenarios. We propose Continuous Soft Actor-Critic, an off-policy actor-critic DRL algorithm in continuous time and space. It is robust to environment time discretization. We also extend the framework to multi-agent scenarios. This Multi-Agent Reinforcement Learning (MARL) algorithm is suitable for both competitive and cooperative settings. Policy evaluation employs stochastic control theory, with loss functions derived from martingale orthogonality conditions. We establish scaling principles for hyperparameters of the algorithm as the environment time discretization δt changes ( δt → 0 ). We provide theoretical proofs for the relevant theorems. To validate the algorithm's effectiveness, we conduct comparative experiments between the proposed algorithm and other mainstream methods across multiple tasks in Virtual Multi-Agent System (VMAS). Experimental results demonstrate that the proposed algorithm achieves robust performance across various environments with different time discretization parameter settings, outperforming other methods.

## 1 Introduction

Recently, reinforcement learning algorithms such as Proximal Policy Optimization (PPO, Schulman et al. [2017]), Soft Actor-Critic (SAC, Haarnoja et al. [2018a]), and Deep Deterministic Policy Gradient (DDPG, Lillicrap et al. [2016]) have demonstrated remarkable success in domains such as large language models, robotics, autonomous driving, and so on. However, there remains little research on the continuous-time learning algorithms for stochastic environments and multi-agent reinforcement learning. And the achievements of deep reinforcement learning studies under the discrete-time frameworks may lack robustness to time discretization. Experimental studies by Henderson et al. [2018] and Tallec et al. [2019] verify that many DRL algorithms developed under discrete-time frameworks lack robustness to the hyperparameters, particularly the time step. Furthermore, Tallec et al. [2019] formally prove that Q -learning does not exist in continuous time, though their analysis is conducted under deterministic environment. These findings raise the following question:

∗ Corresponding author

- Do algorithms such as DQN (Mnih et al. [2015]) and SAC collapse when the time discretization step δt → 0 in stochastic environments?
- Can we propose a robust algorithm for time discretization under stochastic environments?
- How should we adjust the hyperparameters of the algorithm when the time step δt changes?

## 1.1 Original contributions

Aiming to answer the aforementioned questions, we establish a theoretical framework for stochastic continuous reinforcement learning utilizing stochastic control theory. The contributions of this research are listed as follows:

(i) This paper presents a comprehensive analysis of continuous-time reinforcement learning algorithms in stochastic environments. We investigate the optimality of value function approximation, the impact of time discretization, hyperparameter settings, and algorithm implementation. To the best of our knowledge, this work also pioneers the first finite multi-agent actor-critic algorithm designed for continuous-time settings, offering the analysis of finite multi-agent systems in stochastic continuoustime environments.

(ii) In deterministic environments such as Tallec et al. [2019] and Doya [2000], alterations to the time discretization parameter ( δt ) introduce fundamental inconsistencies. In stochastic environments, both the configuration of value function approximation and time discretization changes introduce significant challenges. The mean squared temporal difference error (MSTDE) used in policy evaluation (PE) cannot be applied to stochastic continuous setting. This invalidates popular RL algorithms such as DQN and SAC fail to approximate true value functions in such environments. To address this, we use a novel PE method grounded in martingale theory.

(iii) We derive the hyperparameter scaling laws for our proposed algorithm under stochastic continuous settings as the time step δt changes. This is different from those proposed by Tallec et al. [2019] under the ordinary differential equation framework.

(iv) The MARL algorithm developed in this research is designed for finite-agent systems and adapts to both competitive and cooperative scenarios. We provide experimental implementation of the algorithm in stochastic continuous-time settings. The implementation leverages BenchMARL (Bettini et al. [2024]) to compare with MARL baselines, including MASAC, MAPPO (Yu et al. [2022]), MADDPG (Lowe et al. [2017]), IQL (Tan [1993]), and QMIX (Rashid et al. [2020]) across various environments. Empirical evidence demonstrates that performance of algorithms such as IQL significantly degrades as the time step δt decreases, aligning with theoretical predictions. Experimental results preliminarily indicate that our proposed Continuous (Multi-Agent) Soft ActorCritic (abbreviated as CSAC and CMASAC) outperforms other methods when δt is small, confirming its robustness to time discretization.

## 1.2 Related work

Continuous-time reinforcement learning Baird [1994] and Doya [2000] studied algorithms in the limit of discrete time step δt approaching zero, from discrete-time and continuous-time perspectives, respectively. Tallec et al. [2019] formally proved that Q -learning cannot exist in continuous-time deterministic environments. Consequently, methods based on Q -learning such as DQN and DDPG also fail in continuous-time settings. Tallec et al. [2019] further provided experimental evidence that DQN and DDPG lack robustness to variations in time discretization. Doya [2000] and Tallec et al. [2019] provide studies of algorithms with continuous-time limits under deterministic environments. From a stochastic perspective, Jia and Zhou [2022a] and Jia and Zhou [2022b] investigate the Policy Evaluation problem under the framework of stochastic optimal control theory. However, their work focuses on the conceptual aspects of on-policy frameworks. There has been no substantial exploration of practical off-policy algorithms, impacts of time discretization scales, hyperparameter tuning, or implementation considerations. Through our design, we ensure the feasibility of gradient backpropagation during policy updates and propose a practical off-policy algorithm robust to time discretization.

Soft Actor-Critic Haarnoja et al. [2018a] and Haarnoja et al. [2018b] developed an off-policy actor-critic DRL algorithm based on the maximum entropy framework. This approach has become

a well-performing RL algorithm on a range of continuous control tasks. In this research, we also consider continuous states and actions, with the actor aiming to maximize the Shannon entropy.

Multi-agent reinforcement learning Multi-agent reinforcement learning has achieved practical successes in autonomous driving (Mirowski et al. [2017]), AlphaGo (Silver et al. [2016]), and StarCraft (Vinyals et al. [2019]), but with few results in continuous-time settings. Wang and Zhou [2020] establish a continuous framework for entropy-regularized RL. For large-scale agent systems, mean-field game approaches such as those proposed by Guo et al. [2022], Guo et al. [2024] have been explored. For finite agent systems, algorithms such as Yu et al. [2022], Lowe et al. [2017] are compared in this work.

## 2 Preliminaries

We first briefly introduce frameworks for continuous-time reinforcement learning problems and analyze their properties within these frameworks. Time discretization occurs when implementing these algorithms.

## 2.1 Framework

Let (Ω , F , P ; {F t } t ≥ 0 ) be a filtered probability space, in which denote a standard n-dimensional Brownian motion W = { W t , t ≥ 0 } . {F t W } t ≥ 0 denote the natural filtration generated by W , and P ( U ) denote the set of probability measures taking values on the action space U . 2 We denote Z t by random variable that is uniformly distributed on [0 , 1] , independent of W , and Z t , 0 ≤ t ≤ s are mutually independent, then F s = F s W ∨ σ ( Z t , 0 ≤ t ≤ s ) . The admissible control u π = { u π t , 0 ≤ t ≤ T } is {F t } t ≥ 0 -progressively measurable process representing the actions generated by agent's policy π ( ·| t, x ) ∈ P ( U ) . The agent aims to control the stochastic dynamical system:

<!-- formula-not-decoded -->

to maximize the entropy-regularized expected cumulative reward:

<!-- formula-not-decoded -->

The optimal value function V ( t, x ) for state x at time t is defined by

<!-- formula-not-decoded -->

## 2.2 Multi-agent framework

We denote U and V by the action spaces of two agents respectively, u π ( t ) , v π ( t ) representing actions generated by policy π u , π v respectively. Analogous to the single-agent setting, the corresponding system dynamics and value functions for agents with their policies π = ( π u , π v ) , can be formulated as:

<!-- formula-not-decoded -->

2 In this paper, policies are modeled as parametric distributions (Gaussian policies). To align with conventional notation (e.g., Sutton and Barto [2018]), we do not strictly differentiate between density functions, probability measures, and policies, instead uniformly representing them by the notation π ∈ P ( U ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We assume that problems (1)-(2) and (3)-(4) remain well-posed throughout the analysis. 3 For simplicity, this paper focuses on a two-agent scenario. We emphasize that all results can be naturally extended to finite multi-agent systems.

## 3 Main results

## 3.1 Policy evaluation

Mean squared temporal difference error In discrete-time settings, algorithms like DQN, DDPG, and SAC apply Bellman's principle of optimality to estimate state-action value function Q ( x, u ) , defining their loss functions with mean-square TD error . As for continuous settings, Doya [2000] also adopts this method for state value function J ( x ) . However, Jia and Zhou [2022a] argue that in stochastic environments, this method cannot guarantee convergence to the value function. We outline a brief description about this issue below.

Let us recall Doya's TD Algorithm. The deterministic system satisfies:

<!-- formula-not-decoded -->

where r can encapsulates terms such as reward, discount, and regularization. Using parameterized functions J θ , Doya aims at minimizing the mean-square TD error :

<!-- formula-not-decoded -->

For stochastic system (1), (3), for any fixed ( t, x ) ∈ [0 , T ] × R n , define process:

<!-- formula-not-decoded -->

{ M s , t ≤ s ≤ T } is a square-integrable martingale. Regarding uniform discrete time intervals δt , when δt → 0 , the discrete-time MSTDE becomes

̸

<!-- formula-not-decoded -->

where ⟨ M ⟩ T denotes the quadratic variation of the martingale M . From equation (9), it can be observed that the MSTDE corresponding to the true value function J is not zero. Therefore, minimizing MSTDE to learn parameterized function J θ cannot guarantee convergence to value function J in continuous-time and space environments. The martingale property of the value function motivates a novel approach to policy evaluation.

3 This point is guaranteed by Assumption 2 and Definition 1 provided in the Appendix B.

Martingale orthogonality condition To ensure that the learning process eventually converges to the value function, we impose constraints on the parameterized function J θ to preserve the martingale property. Jia and Zhou [2022a] propose the following proposition.

Proposition 1. A process M ∈ L 2 F ([0 , T ]) is a martingale if and only if

<!-- formula-not-decoded -->

We assume all value functions J and their approximators J θ discussed in this work satisfy the following assumptions.

Assumption 1. For all θ ∈ Θ , J, J θ ∈ C 1 , 2 ([0 , T ) × R n ) ∩ C ([0 , T ] × R n ) and satisfies the polynomial growth condition in x. Moreover, J θ ( t, x ) is a smooth function in θ . ∂J θ ∂θ , ∂ 2 J θ ∂θ 2 ∈ C 1 , 2 ([0 , T ) × R n ) ∩ C ([0 , T ] × R n ) satisfying the polynomial growth condition in x.

Theorem 1. A function is the value function associated with the policy π if it satisfies terminal condition J ( T, x ; π ) = h ( x ) , and for any given ( t, x ) ∈ [0 , T ] × R n and admissible policy ˜ π ∈ P ( U ) , define

<!-- formula-not-decoded -->

is an ( F s , P ) -martingale on [ t, T ] .

According to Proposition 1 and Theorem 1, the approximating function J θ is the value function associated with policy π when the martingale orthogonality condition:

<!-- formula-not-decoded -->

for any ξ t ∈ L 2 F [0 , T ] holds.

In this work, we propose an off-policy algorithm to learn the value function by imposing the constraint:

<!-- formula-not-decoded -->

Here, ˜ π denotes the behavior policy, π ϕ denotes the approximated policy parameterized by neural network. This implies that we can learn the value function J of a given target policy π based on data generated by a different admissible policy ˜ π . We emphasize that the state transitions (1) and rewards r are inherent properties of the environment, determined only by the current state and action, and independent of the policy π . Similar to the framework in Haarnoja et al. [2018a], where policies are modeled as Gaussian distributions with entropy regularization, all policies cover the same action space, and reparameterized sampling is employed in the policy gradient step, making importance sampling weights unnecessary. 4

Martingale orthogonality condition for multi-agent systems For multi-agent systems, the martingale orthogonality conditions also hold. By generalizing Theorem 1, where we denote the policy of agent 2 as ¯ π u , the following theorem guarantees the policy evaluation for agent 1.

Theorem 2. A function J 1 ( · , · ; ¯ π u , π v ) is the value function associated with the policy π = (¯ π u , π v ) if it satisfies the terminal condition J 1 ( T, x ; ¯ π u , π v ) = h 1 ( x ) , and for any fixed ( t, x ) ∈ [0 , T ] × R n and admissible policies (¯ π u , ˜ π v ) , define

<!-- formula-not-decoded -->

is an ( F s , P ) -martingale on [ t, T ] .

4 Proof of Theorem 1 and a more detailed discussion can be found in the Appendix B.1 and Appendix C.

By integrating Theorem 2 with the single-agent analysis presented earlier, we derive a critic network update rule for multi-agent reinforcement learning analogous to (13):

<!-- formula-not-decoded -->

The evaluation of J 2 also obeys the above theorem and condition.

## 3.2 Policy gradient

The following theorems are proposed to characterize policy gradients in both single-agent and multiagent reinforcement learning. The related proofs are provided in the Appendix B. These theorems extend the policy gradient theorem in Jia and Zhou [2022b] to broader settings.

Theorem 3. Consider an admissible parameterized policy π ϕ within the dynamical system (1), for any ( t, x ) ∈ [0 , T ] × R n , its policy gradient g ( t, x ; ϕ ) = ∂ ∂ϕ J ( t, x ; π ϕ ) admits the following representation:

<!-- formula-not-decoded -->

Theorem 4. Consider an admissible parameterized policy π ϕ = (¯ π u ϕ 2 , π v ϕ 1 ) within the dynamical system (3), for any ( t, x ) ∈ [0 , T ] × R n , its policy gradient g 1 ( t, x ; ¯ ϕ 2 , ϕ 1 ) := ∂J 1 ( t,x ;¯ π u ϕ 2 ,π v ϕ 1 ) ∂ϕ 1 admits the following representation:

<!-- formula-not-decoded -->

The policy gradient of another agent g 2 ( t, x ; ϕ 2 , ¯ ϕ 1 ) := ∂J 1 ( t,x ; π u ϕ 2 , ¯ π v ϕ 1 ) ∂ϕ 2 has same representation.

## 3.3 Continuous Soft Actor-Critic

Continuous Soft Actor-Critic The entropy term in (2) enhances the exploration capability of policy π . Hyperparameters (e.g., temperature λ ) and soft update techniques follow the implementation of Haarnoja et al. [2018b]. To stabilize training, we employ a separate function approximator for the soft value J ( t, x t ) , which minimizes the martingale orthogonality conditions through gradient descent. We integrate key techniques for automatic adaptation of temperature parameters λ via dual gradient descent, as proposed by Haarnoja et al. [2018b]. We choose ξ t = ∂J θ ( t,X t ) ∂θ and we optimize the temperature parameter λ via:

<!-- formula-not-decoded -->

where the target entropy ¯ H is typically task-specific. In practice, we solve (18) using stochastic gradient descent. Algorithm 1 outlines the entire procedure.

Continuous Multi-Agent Soft Actor-Critic The algorithmic workflow of the multi-agent system aligns with the single-agent case, where the function approximator J , reward r , and entropy term are replaced by their multi-agent counterparts. We adopt the Centralized Training with Decentralized Execution (CTDE) framework. Coordination between the two agents is achieved through alternating updates. The complete procedure is detailed in Algorithm 2 in Appendix A.

## Algorithm 1 Continuous Soft Actor-Critic Algorithm

Inputs: time step δt , number of epochs N , number of mesh grids K , number of gradient step L , batch size I , initial learning rates α θ , α ϕ , initial θ, ϕ, θ target, discount factor β , and temperature parameter λ . J θ ( · , · ) defining functional form of the value function, π ϕ ( · | · ) defining functional form of the policy, D defining buffer of transitions, opt J , opt π defining optimizer.

Interactive program: an environment simulator ( x ′ , r ) = Environment δt ( t, x, u ) that takes current time-state pair ( t, x ) and action u as inputs and generates state x ′ at time t + δt and the instantaneous reward r at time t .

## Learning procedure:

for j = 1 to N do

Initialize k = 0 . Observe the initial state x 0 and store x t k ← x 0 .

while k &lt; K do

Generate action u t k ∼ π ϕ ( · | t k , x t k ) .

Apply u t k to the environment simulator ( x, r ) = Environment δt ( t k , x t k , u t k ) , and observe the output new state x and reward r .

Store x t k +1 ← x and r t k ← r in D , D = D ∪ ( x t k , u t k , r t k , x t k +1 , d t k +1 ) , d is the episode termination signal. Update k ← k +1 .

## end while

for l = 1 to L do

Sampled a batch of I transitions ( x t i , x t i +1 , u t i , r t i ) from D and a batch of u ϕ t i from π ϕ . Compute

<!-- formula-not-decoded -->

Update θ (policy evaluation) with opt , learning rate α δt , and ∆ θ .

J θ Update ϕ (policy gradient) with opt π , learning rate α ϕ δt and ∆ ϕ . Adjust temperature λ . Update θ target ← τθ +(1 -τ ) θ target . end for end for

## 3.4 Hyperparameter scaling

The optimality of value functions in our continuous-time algorithm is well established. Here, we analyze the impact of time step δt on parameter updates. Time-dependent scaling laws for returns, discount factors, learning rates, and temperatures are derived from discrete-time formulations ( δt = 1 ).

Learning rate scaling As proven in Tallec et al. [2019], the state-action value function Q ( x, u ) and state value function V ( x ) after time discrete satisfy: Q π δt ( x, u ) = V π δt ( x ) + O ( δt ) . In learning framework, the advantage function satisfy: A ψ δt ( x, u ) = Q δt ( x, u ) -V θ δt ( x ) = O ( δt ) . Consequently, Q -learning or A -learning gradients scale as O ( δt ) , causing vanishing gradients during backpropagation. Empirical validation is provided in Table 6, which displays MADDPG gradients across four update steps. To address this, Tallec et al. [2019] redefine the advantage function as A ψ ( x, u ) = Q δt ( x,u ) -V θ δt ( x ) = O (1) and scaling the learning rate.

<!-- formula-not-decoded -->

Extending this insight to stochastic continuous environments, the martingale difference term also satisfies δM θ t = O ( δt ) . To stabilize gradient magnitudes, we normalize it as δM θ t /δt, while preserving the martingale structure of J θ . Table 7 validates this normalization strategy by comparing gradients under scaled versus unscaled configurations across three update steps in the proposed algorithm.

<!-- image -->

(a) Navigation.

<!-- image -->

(b) Sampling.

Figure 1: Two VMAS multi-robot control tasks used in the experiments.

We analyze the algorithm in a continuous-time framework with temporal discretization during implementation. To ensure that discrete-time parameter trajectories converge to well-defined continuoustime limits, learning rate scaling is essential. This is formalized in the following theorem.

Theorem 5. Let ( x t , u t ) be some exploration trajectory under time discretization. Set the learning rates to η θ δt = α θ δt β and η ϕ δt = α ϕ δt β for some β &gt; 0 , and learn the parameters θ and ϕ by iterating (19) along the trajectory ( x t , u t ) . Then, when t &gt; 0 :

(i) If β = 1 the discrete parameter trajectories converge to continuous parameter trajectories;

(ii) If β &gt; 1 the parameters stay at their initial values;

(iii) If β &lt; 1 , the parameters can reach infinity.

Other hyperparameter scalings (i) Discount factor: By comparing our framework with the continuous Markov Decision Process (MDP) formulation in Tallec et al. [2019], we derive the relationship γ = e -β . Then from the parameter update rule in Equation (19), the discrete-time discount factor e -β δt under variable time step scaling becomes: e -β δt = e -βδt = γ δt . (ii) Temperature: Equation (19) demonstrates linear scaling of the temperature parameter with the discretization interval: λ δt = λ · δt. (iii) Reward: The reward term scales proportionally to the time step via formula (19): r δt = r · δt.

## 4 Experiments

Tasks We conducted experiments using multiple tasks in the VMAS simulator (Bettini et al. [2022]). The visual representations of the Navigation and Sampling tasks are illustrated in Figure 1.

Implementations Following Bettini et al. [2024], we employ their network architectures, default hyperparameters, and other configurations in our experiments. Descriptions of tasks, random seeds, network architectures, optimizer settings and other implementation details are all documented in the Appendix E. 5 For fair comparison, the proposed algorithm in this work utilizes the common hyperparameters tuned in Bettini et al. [2024], which may not reflect its optimal performance, yet remains valid for robustness verification.

Results Here, we employ the simulation environments navigation and sampling (validated in Bou et al. [2024]), along with multi-agent reinforcement learning algorithms including MAPPO, MASAC, and MADDPG (which demonstrated superior performance in Bettini et al. [2024]), as well as Q -learning-based approaches QMIX and IQL, to conduct a comparative analysis of these algorithms.

(i) As shown in Table 4, when δt decreases from 0 . 1 to 0 . 01 , the performance of popular algorithms gradually declines. We selected two representative algorithms-MADDPG, which demonstrated the best performance in experiments from Bettini et al. [2024], and MASAC, a discrete-time method of the actor-critic class-for further investigation. The results in Table 1 reconfirm our viewpoint in the sampling environment.

5 The repository includes code: https://github.com/hh11813/continuous-soft-actor-critic

Figure 2: Performance profile for two VMAS tasks.

<!-- image -->

(ii) In contrast, the performance of CMASAC remains stable. Results are shown in Table 2 and 3. Figures 2 present the performances of algorithms MAPPO, MASAC, MADDPG and TEST (a time scaling-free variant of CMASAC, ensuring parameter consistency for cross-algorithm fairness) at δt = 0 . 01 . The results demonstrate that the martingale approach enhances the algorithm's robustness against small δt . 6

(iii) To ensure the comprehensiveness of the experiments and eliminate the impact of extraneous factors on the conclusions, we increased the number of random seeds to five, normalized the rewards, and adjusted the hyperparameters accordingly. The results are summarized in Table 5. The experimental results in Table 5 verify that the temporal robustness of the method proposed in this paper outperforms other methods.

We also conducted a set of experiments in a single-agent environment, which demonstrate the advantages of the off-policy approach. For detailed results, please refer to Appendix C.

## 5 Limitations and future work

Given that the problem formulation is based on stochastic differential equations and constrained by computational resources, we limit our experimental validation to selected benchmarks. The proposed method targets continuous space-time problems and thus may not be suitable for discretevalued spaces. The theory of multi-agent reinforcement learning requires further exploration. This work serves as an exploratory investigation into continuous-time multi-agent reinforcement learning problems.

## 6 Conclusion

We propose Continuous Soft Actor-Critic, a novel off-policy reinforcement learning algorithm designed for stochastic continuous-time environments, and further generalize it to multi-agent settings. By bridging principles between stochastic optimal control theory and reinforcement learning, we address critical limitations of existing algorithms in continuous-time environments and preliminarily validate our claims. Central to our approach is the enforcement of the martingale property for value functions, coupled with gradient and hyperparameter scaling laws. This results in a δt -robust parameter update rule. The off-policy nature of the algorithm ensures high sample efficiency, as it enables reuse of historical trajectories. The experimental results on benchmark tasks demonstrate the proposed algorithm's robustness to time discretization.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Key R&amp;D Program of China (NO. 2023YFA1008701) and the Key Project of the National Natural Science Foundation of China (No. 12431017).

6 Further experimental details, scenario-specific analyses, and extended results in different tasks are provided in the Appendix D.

δt

Median

IQM

Mean

Optimality Gap

Median

IQM

Mean

Optimality Gap

Median

IQM

Mean

Optimality Gap

Table 1: Aggregate scores under sampling with different δt

MASAC(0.1)

MADDPG(0.1)

0.66 [0.62, 0.7]

0.66 [0.62, 0.7]

0.66 [0.62, 0.7]

0.34 [0.3, 0.38]

0.94 [0.86, 1.0]

0.94 [0.86, 1.0]

0.94 [0.86, 1.0]

MASAC(0.01)

0.43 [0.37, 0.46]

0.43 [0.37, 0.46]

0.43 [0.37, 0.46]

0.06 [0.0, 0.14]

0.57 [0.54, 0.63]

Table 2: Aggregate scores of CMASAC under navigation

δt

IQM

Mean

Median

Optimality Gap

0.1

0.96 [0.92, 1.0]

0.96 [0.92, 1.0]

0.96 [0.92, 1.0]

0.01

0.93 [0.86, 1.0]

0.93 [0.86, 1.0]

0.93 [0.86, 1.0]

0.04 [0.0, 0.08]

0.07 [0.0, 0.14]

Table 3: Aggregate scores of CMASAC under sampling

δt

IQM

Mean

Median

Optimality Gap

0.1

0.84 [0.64, 1.0]

0.84 [0.64, 1.0]

0.84 [0.64, 1.0]

0.01

0.92 [0.83, 1.0]

0.92 [0.83, 1.0]

0.92 [0.83, 1.0]

0.16 [0.0, 0.36]

0.08 [0.0, 0.17]

Table 4: Aggregate scores under navigation with different δt

| δt =0.1        | QMIX             | IQL               | MAPPO             | MASAC             |
|----------------|------------------|-------------------|-------------------|-------------------|
| Median         | 0.92 [0.9, 0.93] | 0.99 [0.98, 1.0]  | 0.97 [0.96, 0.98] | 0.89 [0.89, 0.89] |
| IQM            | 0.92 [0.9, 0.93] | 0.99 [0.98, 1.0]  | 0.97 [0.96, 0.98] | 0.89 [0.89, 0.89] |
| Mean           | 0.92 [0.9, 0.93] | 0.99 [0.98, 1.0]  | 0.97 [0.96, 0.98] | 0.89 [0.89, 0.89] |
| Optimality Gap | 0.08 [0.07, 0.1] | 0.01 [0.0, 0.02]  | 0.03 [0.02, 0.04] | 0.11 [0.11, 0.11] |
| δt =0.01       | QMIX             | IQL               | MAPPO             | MASAC             |
| Median         | 0.86 [0.63, 1.0] | 0.88 [0.83, 0.95] | 0.8 [0.73, 0.84]  | 0.51 [0.5, 0.52]  |
| IQM            | 0.86 [0.63, 1.0] | 0.88 [0.83, 0.95] | 0.8 [0.73, 0.84]  | 0.51 [0.5, 0.52]  |
| Mean           | 0.86 [0.63, 1.0] | 0.88 [0.83, 0.95] | 0.8 [0.73, 0.84]  | 0.51 [0.5, 0.52]  |
| Optimality Gap | 0.14 [0.0, 0.37] | 0.12 [0.05, 0.17] | 0.2 [0.16, 0.27]  | 0.49 [0.48, 0.5]  |
| δt             | MADDPG(0.1)      | MADDPG(0.01)      |                   |                   |
| Median         | 0.8 [0.76, 0.84] | 0.78 [0.62, 0.91] |                   |                   |
| IQM            | 0.8 [0.76, 0.84] | 0.78 [0.62, 0.91] |                   |                   |
| Mean           | 0.8 [0.76, 0.84] | 0.78 [0.62, 0.91] |                   |                   |
| Optimality Gap | 0.2 [0.16, 0.24] | 0.22 [0.09, 0.38] |                   |                   |

Table 5: Aggregate scores under navigation with δt = 0 . 01

QMIX

0.75 [0.69, 0.81]

0.74 [0.67, 0.83]

0.75 [0.69, 0.81]

0.25 [0.19, 0.31]

MADDPG

0.88 [0.79, 0.96]

0.9 [0.76, 0.98]

0.88 [0.79, 0.96]

0.12 [0.04, 0.21]

IQL

0.92 [0.89, 0.94]

0.92 [0.88, 0.95]

0.92 [0.89, 0.94]

0.08 [0.06, 0.11]

MAPPO

0.87 [0.87, 0.89]

0.87 [0.86, 0.89]

0.87 [0.87, 0.89]

0.13 [0.11, 0.13]

MASAC

0.49 [0.48, 0.5]

0.49 [0.47, 0.5]

0.49 [0.48, 0.5]

0.51 [0.5, 0.52]

MADDPG(0.01)

0.91 [0.79, 1.0]

0.91 [0.79, 1.0]

0.91 [0.79, 1.0]

0.09 [0.0, 0.21]

## References

- Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G. Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In Advances in Neural Information Processing Systems , volume 34, pages 29304-29320, 2021.
- Leemon C. Baird. Reinforcement learning in continuous time: Advantage updating. In Proceedings of 1994 IEEE International Conference on Neural Networks (ICNN'94) , volume 4, pages 2448-2453. IEEE, 1994.
- Matteo Bettini, Ryan Kortvelesy, Jan Blumenkamp, and Amanda Prorok. VMAS: A vectorized multi-agent simulator for collective robot learning. In International Symposium on Distributed Autonomous Robotic Systems , pages 42-56. Springer, 2022.
- Matteo Bettini, Amanda Prorok, and Vincent Moens. BenchMARL: Benchmarking multi-agent reinforcement learning. Journal of Machine Learning Research , 25(217):1-10, 2024.
- Albert Bou, Matteo Bettini, Sebastian Dittert, Vikash Kumar, Shagun Sodhani, Xiaomeng Yang, Gianni De Fabritiis, and Vincent Moens. TorchRL: A data-driven decision-making library for pytorch. In International Conference on Learning Representations , 2024.
- Kenji Doya. Reinforcement learning in continuous time and space. Neural Computation , 12(1): 219-245, 2000.
- Rihab Gorsane, Omayma Mahjoub, Ruan John de Kock, Roland Dubb, Siddarth Singh, and Arnu Pretorius. Towards a standardised performance evaluation protocol for cooperative marl. In Advances in Neural Information Processing Systems , volume 35, pages 5510-5521, 2022.
- Xin Guo, Renyuan Xu, and Thaleia Zariphopoulou. Entropy regularization for mean field games with learning. Mathematics of Operations Research , 47(4):3239-3260, 2022.
- Xin Guo, Anran Hu, and Junzi Zhang. Mf-omo: An optimization formulation of mean-field games. SIAM Journal on Control and Optimization , 62(1):243-270, 2024.
- Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning , pages 1861-1870. PMLR, 2018a.
- Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, et al. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905 , 2018b.
- Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep reinforcement learning that matters. In Proceedings of the AAAI conference on Artificial Intelligence , volume 32. Association for the Advancement of Artificial Intelligence (AAAI), 2018.
- Yanwei Jia and Xun Yu Zhou. Policy evaluation and temporal-difference learning in continuous time and space: a martingale approach. Journal of Machine Learning Research , 23(154):1-55, 2022a.
- Yanwei Jia and Xun Yu Zhou. Policy gradient and actor-critic learning in continuous time and space: theory and algorithms. Journal of Machine Learning Research , 23(275):1-50, 2022b.
- Yanwei Jia and Xun Yu Zhou. q-learning in continuous time. Journal of Machine Learning Research , 24(161):1-61, 2023.
- Yanwei Jia and Xun Yu Zhou. Erratum to 'q-learning in continuous time'. 2025.
- Ioannis Karatzas and Steven Shreve. Brownian motion and stochastic calculus , volume 113. Springer Science &amp; Business Media, 2014.
- Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. In International Conference on Learning Representations , 2016.

- Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in Neural Information Processing Systems , volume 30, 2017.
- Piotr Mirowski, Razvan Pascanu, Fabio Viola, Hubert Soyer, Andy Ballard, Andrea Banino, Misha Denil, Ross Goroshin, Laurent Sifre, Koray Kavukcuoglu, Dharshan Kumaran, and Raia Hadsell. Learning to navigate in complex environments. In International Conference on Learning Representations , 2017.
- Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature , 518(7540):529-533, 2015.
- Tabish Rashid, Mikayel Samvelyan, Christian Schroeder De Witt, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. Monotonic value function factorisation for deep multi-agent reinforcement learning. Journal of Machine Learning Research , 21(178):1-51, 2020.
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint http://arxiv.org/abs/1707.06347 , 2017.
- David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. Nature , 529(7587):484-489, 2016.
- Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction . The MIT Press, second edition, 2018.
- Corentin Tallec, Léonard Blier, and Yann Ollivier. Making deep q-learning methods robust to time discretization. In International Conference on Machine Learning , volume 97, pages 6096-6104. PMLR, 2019.
- Ming Tan. Multi-agent reinforcement learning: Independent vs. cooperative agents. In International Conference on Machine Learning , pages 330-337, 1993.
- Oriol Vinyals, Igor Babuschkin, Wojciech M Czarnecki, Michaël Mathieu, Andrew Dudzik, Junyoung Chung, David H Choi, Richard Powell, Timo Ewalds, Petko Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. Nature , 575(7782):350-354, 2019.
- Haoran Wang and Xun Yu Zhou. Continuous-time mean-variance portfolio selection: A reinforcement learning framework. Mathematical Finance , 30(4):1273-1308, 2020.
- Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. The surprising effectiveness of ppo in cooperative multi-agent games. In Advances in Neural Information Processing Systems , volume 35, pages 24611-24624, 2022.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Yes. Our main contributions are also detailed in Sec. 1. Also see Sec. 4 and Appendix for more theoretical and experimental evidence.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Yes, please see Sec. 5 for limitations.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We detail the assumptions and proofs of theoretical results in Sec. 3 and Appendix B.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We release the codes at https://github.com/hh11813/ continuous-soft-actor-critic to reproduce the results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We release the codes at https://github.com/hh11813/ continuous-soft-actor-critic

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: Please see Sec. 4 and Appendix E.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Our experimental results incorporate confidence intervals and other statistical measures. Please see Sec. 4 and Appendix D.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We report in Appendix E.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We followed the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: This work focuses on a academic, publicly-available continuous algorithm. This work is not related to any private or personal data, and there's no explicit negative social impacts.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Yes, we credited them in appropriate ways.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: The paper introduces assets with comprehensive documentation.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## A Algorithm

Algorithm 2 illustrates the workflow of the Continuous Multi-Agent Soft Actor-Critic.

## B Proofs

Throughout the proofs, by convention we denote by A ◦ B the inner product between A and B , by ∥ x ∥ 2 the Euclidean norm of x , ∥ A ∥ F the Frobenius norm of A , and A ′ is transpose of A . We denote by a ∨ b the larger of a and b, and by a ∧ b the smaller of the two numbers. We denote by I the indicator function, I A ( x ) = 1 when x ∈ A and I A ( x ) = 0 when x / ∈ A . For a measurable set U , we denote by P ( U ) the set of probability distributions over U . We assume the following conditions hold to maintain the well-posedness of our problem.

Assumption 2. (i) b, ¯ σ, r i , h i , i = 1 , 2 are all continuous functions in their respective arguments;

(ii) b, ¯ σ are uniformly Lipschitz continuous in x , i.e., for φ ∈ { b, ¯ σ } , there exists a constant C &gt; 0 such that

<!-- formula-not-decoded -->

(iii) b, ¯ σ have linear growth in x , i.e., for φ ∈ { b, ¯ σ } , there exists a constant C &gt; 0 such that

<!-- formula-not-decoded -->

(iv) r i and h i have polynomial growth in ( x, u, v ) and x respectively, i.e., there exists a constant C &gt; 0 and µ ≥ 1 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following gives the precise definition of admissible policies.

Definition 1. A policy π = π ( · | · , · ) is called admissible if

(i) π ( · | t, x ) ∈ P ( U ) , supp π ( · | t, x ) = U for every ( t, x ) ∈ [0 , T ] × R n , and π ( u | t, x ) : ( t, x, u ) ∈ [0 , T ] × R n × U → R is measurable;

(ii) π ( u | t, x ) is continuous in ( t, x ) and uniformly Lipschitz continuous in x in the total variation distance, i.e., ∫ U | π ( u | t, x ) -π ( u | r, x ′ ) | du → 0 as ( r, x ′ ) → ( t, x ) , and there is a constant C &gt; 0 independent of ( t, u ) such that

<!-- formula-not-decoded -->

(iii) For any given α &gt; 0 , the entropy of π and its α -moment have polynomial growth in x , i.e., there are constants C = C ( α ) &gt; 0 and µ ′ = µ ′ ( α ) ≥ 1 such that ∣ ∣ ∫ U -log π ( u | t, x ) π ( u | t, x ) du ∣ ∣ ≤ C ( 1 + | x | µ ′ ) , and ∫ U | u | α π ( u | t, x ) du ≤ C ( 1 + | x | µ ′ ) , ∀ ( t, x ) ∈ [0 , T ] × R n .

Under Assumption 2 along with Definition 1, the well-poseness of problems (1)-(2) and (3)-(4) can be guaranteed.

Obviously problem (1)-(2) and (3)-(4) related to stochastic optimal control problem and stochastic game respectively, but the probability space is no longer (Ω , F W , P W ) but (Ω , F , P ) . 7 We recall the existing results for stochastic control in (Ω , F W , P W ) . Value function J can be characterized by a PDE based on the celebrated Feynman-Kac formula:

<!-- formula-not-decoded -->

where ∂J ∂x ∈ R n is the gradient, and ∂ 2 J ∂x 2 ∈ R n × n is the Hessian.

7 Readers may refer to Jia and Zhou [2025] for further discussion on extended probability spaces, which does not affect the methodology of this paper and thus is not elaborated here.

<!-- formula-not-decoded -->

## Algorithm 2 Continuous Multi-Agent Soft Actor-Critic Algorithm

Inputs: time step δt , number of epochs N , number of mesh grids K , number of gradient step L , batch size I , initial learning rates α θ , α ϕ , initial θ 1 , θ 2 , ϕ 1 , ϕ 2 , θ target 1 , θ target 2 , discount factor β , and temperature parameter λ 1 , λ 2 . J θ 1 1 ( · , · ) , J θ 2 2 ( · , · ) defining functional form of the value function, π u ϕ 1 ( · | · ) , π v ϕ 2 ( · | · ) defining functional form of the policy, D defining buffer of transitions, opt J , opt π defining optimizer.

Interactive program: an environment simulator ( x ′ , r 1 , r 2 ) = Environment δt ( t, x, u, v ) that takes current time-state pair ( t, x ) and action u, v as inputs and generates state x ′ at time t + δt and the instantaneous reward r 1 , r 2 at time t .

## Learning procedure:

for j = 1 to N do

Initialize k = 0 . Observe the initial state x 0 and store x t k ← x 0 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Apply v t k , u t k to the environment simulator ( x, r 1 , r 2 ) = Environment δt ( t k , x t k , u t k , v t k ) , and observe the output new state x and reward r 1 , r 2 .

Store x t k +1 ← x and r t k 1 ← r 1 , r t k 2 ← r 2 in D , D = D ∪ ( x t k , u t k , v t k , r t k , x t k +1 , d t k +1 ) . Update k ← k +1 .

## end while

for l = 1 to L do

Sampled a batch of I transitions ( x t i , u t i , v t i , x t i +1 , r t i 1 ) from D and a batch of v ϕ 1 t i from π v ϕ 1 . Compute

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Update θ 1 (policy evaluation) with opt J , learning rate α θ δt , and ∆ θ 1 .

Update ϕ 1 (policy gradient) with opt π , learning rate α ϕ δt and ∆ ϕ 1 .

Sampled a batch of I transitions ( x t i , u t i , v t i , x t i +1 , r t i 2 ) from D and a batch of u ϕ 2 t i from π u ϕ 2 . Compute

Adjust temperature λ 1 . Update θ target 1 ← τθ 1 +(1 -τ ) θ target 1 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Update θ 2 with opt J , learning rate α θ δt , and ∆ θ 2 .

Update ϕ 2 with opt π , learning rate α ϕ δt and ∆ ϕ 2 .

Adjust temperature λ 2 . Update θ target 2 ← τθ 2 +(1 -τ ) θ target 2 .

end for end for

## B.1 Proof of Theorem 1

Proof. Let J be the value function with policy π ∈ P ( U ) , applying Itô's lemma to the process e -βs J ( s, X ˜ π s ) , we obtain for 0 ≤ t &lt; s ≤ T ,

<!-- formula-not-decoded -->

where for any ( t, x, u ) ∈ [0 , T ] × R n × U,

<!-- formula-not-decoded -->

Since process (11) with J keeps martingalety, and the term ∫ · · · dW s ′ of the right hand side of (20) is local martingale, we derive that the first term of the right hand side of (20) is also a local martingale. For continuous local martingale with finite variation, by Chapter 1, Exercise 5.21, on Karatzas and Shreve [2014],

<!-- formula-not-decoded -->

P -almost surely holds.

Denote

<!-- formula-not-decoded -->

We shall demonstrate that

<!-- formula-not-decoded -->

Since f is a continuous function, if (22) is not true, there exists ( t ∗ , x ∗ , u ∗ ) and ϵ &gt; 0 such that | f ( t ∗ , x ∗ , u ∗ ) | &gt; ϵ . Without loss of generality, we assume f ( t ∗ , x ∗ , u ∗ ) &gt; ϵ . Then exists δ &gt; 0 such that f ( r, x ′ , u ′ ) &gt; ϵ/ 2 for all ( r, x ′ , u ′ ) with | r -t ∗ | ∨ | x ′ -x ∗ | ∨ | u ′ -u ∗ | &lt; δ . Consider the state process X ˜ π , starting from ( t ∗ , x ∗ , u ∗ ) , namely, { X ˜ π s , t ∗ ≤ s ≤ T } follows (1) with X ˜ π t ∗ = x ∗ and u ˜ π t ∗ = u ∗ . Define

<!-- formula-not-decoded -->

The continuity of X ˜ π implies that τ &gt; t ∗ , P -almost surely.

(21) means that there exists Ω 0 ∈ F with P (Ω 0 ) = 0 such that for all ω ∈ Ω \ Ω 0 and s ∈ [ t ∗ , T ] , ∫ s t ∗ e -βs ′ f ( s ′ , X ˜ π s ′ ( ω ) , u ˜ π s ′ ( ω ) ) ds ′ = 0 . It follows from Lebesgue's differentiation theorem that for any ω ∈ Ω \ Ω 0 ,

<!-- formula-not-decoded -->

Consider the set Z ( ω ) = { s ∈ [ t ∗ , τ ( ω )] : u ˜ π s ( ω ) ∈ B δ ( u ∗ ) } ⊂ [ t ∗ , τ ( ω )] , where B δ ( u ∗ ) = { u ′ ∈ U : | u ′ -u ∗ | &lt; δ } is the neighborhood of u ∗ . Because f ( s, X ˜ π s ( ω ) , u ˜ π s ( ω ) ) &gt; ϵ 2 when s ∈ Z ( ω ) , we conclude that Z ( ω ) has Lebesgue measure zero for any ω ∈ Ω \ Ω 0 . That is,

<!-- formula-not-decoded -->

Integrating ω with respect to P and applying Fubini's theorem, we obtain

<!-- formula-not-decoded -->

Since τ &gt; t ∗ , P -almost surely, the above implies

<!-- formula-not-decoded -->

However, this contradicts Definition 1 about an admissible policy ˜ π . Indeed, Definition 1-(i) stipulates supp ˜ π ( · | t, x ) = U for any ( t, x ) , hence ∫ B δ ( u ∗ ) ˜ π ( u | t, x ) du &gt; 0 . Then the continuity in Definition 1-(ii) yields

<!-- formula-not-decoded -->

it is a contradiction. Hence we conclude f ( t, x, u ) = 0 for every ( t, x, u ) ∈ [0 , T ] × R n × U. Then we have

<!-- formula-not-decoded -->

Assume that the function J θ satisfies the assumptions of Theorem 1. Combining the above results with the hypotheses of Theorem 1, we conclude that J θ satisfies:

<!-- formula-not-decoded -->

Then J θ is the unique viscosity solution among polynomially growing functions of (23). J θ is the value function associated with policy π .

## B.2 Proof of Theorem 2

Consider system:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Value function with policies (¯ π u , π v ) admits the following representation:

<!-- formula-not-decoded -->

The problem (24)-(25) is equavilent to (3)-(4) and the solution of the SDE (3) X π with policies (¯ π u , π v ) shares the same distribution as ˜ X π of (24).

Proof. Considering the equivalent formulation (24)-(25), by applying the same method for process e -βs J 1 ( s, ˜ X ˜ π s ) with policy π = (¯ π u , π v ) as used in the proof of Theorem 1, we can get for 0 ≤ t &lt; s ≤ T :

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Following the methodology of proving Theorem 1, we extend the argument to the multi-agent case. Since process (14) with J 1 keeps martingalety, and the term ∫ · · · dW s ′ of the right hand side of (26) is local martingale, we derive that the first term of the right hand side of (26) is also a local martingale. Then

<!-- formula-not-decoded -->

P -almost surely holds.

Denote

<!-- formula-not-decoded -->

We shall demonstrate that

<!-- formula-not-decoded -->

Since f is a continuous function, if (28) is not true, there exists ( t ∗ , x ∗ , v ∗ ) and ϵ &gt; 0 such that | f ( t ∗ , x ∗ , v ∗ ) | &gt; ϵ . Without loss of generality, we assume f ( t ∗ , x ∗ , v ∗ ) &gt; ϵ . Then exists δ &gt; 0 such that f ( r, x ′ , v ′ ) &gt; ϵ/ 2 for all ( r, x ′ , v ′ ) with | r -t ∗ | ∨ | x ′ -x ∗ | ∨ | v ′ -v ∗ | &lt; δ . Consider the state process X ˜ π , starting from ( t ∗ , x ∗ , v ∗ ) , namely, { X ˜ π s , t ∗ ≤ s ≤ T } follows (24) with X ˜ π t ∗ = x ∗ and v ˜ π t ∗ = v ∗ . Define

<!-- formula-not-decoded -->

The continuity of X ˜ π implies that τ &gt; t ∗ , P -almost surely.

(27) means that there exists Ω 0 ∈ F with P (Ω 0 ) = 0 such that for all ω ∈ Ω \ Ω 0 , for all s ∈ [ t ∗ , T ] , ∫ s t ∗ e -βs ′ f ( s ′ , X ˜ π s ′ ( ω ) , v ˜ π s ′ ( ω ) ) ds ′ = 0 . It follows from Lebesgue's differentiation theorem that for any ω ∈ Ω \ Ω 0 ,

<!-- formula-not-decoded -->

Consider the set Z ( ω ) = { s ∈ [ t ∗ , τ ( ω )] : v ˜ π s ( ω ) ∈ B δ ( v ∗ ) } ⊂ [ t ∗ , τ ( ω )] , where B δ ( v ∗ ) = { v ′ ∈ V : | v ′ -v ∗ | &lt; δ } is the neighborhood of v ∗ . Because f ( s, X ˜ π s ( ω ) , v ˜ π s ( ω ) ) &gt; ϵ 2 when s ∈ Z ( ω ) , we conclude that Z ( ω ) has Lebesgue measure zero for any ω ∈ Ω \ Ω 0 . That is,

<!-- formula-not-decoded -->

Integrating ω with respect to P and applying Fubini's theorem, we obtain

<!-- formula-not-decoded -->

Since τ &gt; t ∗ , P -almost surely, the above implies

<!-- formula-not-decoded -->

However, this contradicts Definition 1 about an admissible policy ˜ π . Indeed, Definition 1-(i) stipulates supp ˜ π ( · | t, x ) = V for any ( t, x ) , hence ∫ B δ ( v ∗ ) ˜ π ( v | t, x ) dv &gt; 0 . Then the continuity in Definition 1-(ii) yields

<!-- formula-not-decoded -->

it is a contradiction. Hence we conclude f ( t, x, v ) = 0 for every ( t, x, v ) ∈ [0 , T ] × R n × V. Then we have

<!-- formula-not-decoded -->

Assume that the function J θ 1 satisfies the assumptions of Theorem 2. Combining the above results with the hypotheses of Theorem 2, we conclude that J θ 1 satisfies:

<!-- formula-not-decoded -->

Then J θ 1 is the unique viscosity solution among polynomially growing functions of (29). J θ 1 is the value function associated with policy π . The proof for J 2 follows a procedure analogous to that of J 1 .

## B.3 Proof of Theorem 3

Proof. For value function J with policy π , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Let π ϕ be the parametric family of policies with the parameter ϕ ∈ Φ . We aim to compute the policy gradient g ( t, x ; ϕ ) := ∂J ( t,x ; π ϕ ) ∂ϕ ∈ R L ϕ at the current time-state pair ( t, x ) .

We take the derivative in ϕ on both sides of (30) and we have

<!-- formula-not-decoded -->

Define

<!-- formula-not-decoded -->

Then (31) can be written as

<!-- formula-not-decoded -->

The Feynman-Kac formula represents g ( t, x ; ϕ ) as

<!-- formula-not-decoded -->

Apply Itô lemma to J ( s, X π s ; ϕ ) , we obtain

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

By substituting equation (34) into equation (33), the theorem is proven.

## B.4 Proof of Theorem 4

Proof. From the Feymann-Kac formula, we have J 1 ( t, x ; ¯ π u , π v ) with policy π = (¯ π u , π v ) satisfies

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Let ( π u ϕ 2 , π v ϕ 1 ) be the parametric family of policies with the parameter ϕ 1 ∈ Φ 1 ⊂ R L ϕ 1 , ϕ 2 ∈ Φ 2 ⊂ R L ϕ 2 . We aim to compute the policy gradient g 1 ( t, x ; ¯ ϕ 2 , ϕ 1 ) := ∂J 1 ( t,x ;¯ π u ϕ 2 ,π v ϕ 1 ) ∂ϕ 1 , g 2 ( t, x ; ϕ 2 , ¯ ϕ 1 ) := ∂J 2 ( t,x ; π u ϕ 2 , ¯ π v ϕ 1 ) ∂ϕ 2 .

Taking the derivative in ϕ 1 on both sides of the HJB equation (35), we obtain:

<!-- formula-not-decoded -->

Define

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

Then g 1 is represented by

<!-- formula-not-decoded -->

Apply Itô lemma to J 1 ( t, X π s ; ¯ ϕ 2 , ϕ 1 ) on [ t, t + δt ] ,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

By substituting equation (39) into equation (37), the theorem is proven. The proof for g 2 ( t, x ; ϕ 2 , ¯ ϕ 1 ) follows a procedure analogous to that of g 1 ( t, x ; ¯ ϕ 2 , ϕ 1 ) .

## B.5 Proof of Theorem 5

Proof. By algorithm 1, for time discretization δt , we have

<!-- formula-not-decoded -->

According to Assumption 1, using Itô's fomula for J θ , we have

<!-- formula-not-decoded -->

where W δt denotes the Brownian motion over transitions Hereafter, we denote by ˜ C a constant which is independent of δt . Then (40) becomes

<!-- formula-not-decoded -->

(i) When β = 1 , let δt -→ 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Parameters in discrete-time convergence to continuous trajectory charactirized by well-posed ordinary differential equation.

- (ii) When β &gt; 1 , let δt -→ 0 , we have

<!-- formula-not-decoded -->

Parameters in discrete-time convergence to initial state θ 0 .

(iii) When β &lt; 1 , let δt -→ 0 , we have

<!-- formula-not-decoded -->

Parameters in discrete-time can reach infinity.

## C Off-policy approach

Compared to the q -framework proposed in Jia and Zhou [2023], two limitations arise. First, constraints (27) or (21) are not easily verifiable in general environments. In particular, when the normalizing constant in the Gibbs measure is unavailable, it renders methods relying solely on a q -network without a policy network infeasible (Algorithms 1-3). Second, the policy-network-maintaining method introduced in Jia and Zhou [2023] remains an on-policy approach. And the proposed learning method in Jia and Zhou [2023]:

<!-- formula-not-decoded -->

introduces bias in off-policy settings. This leads to the accumulation of biases at each update step throughout the training.

The method proposed in this paper is a practical continuous-time off-policy algorithm. The approach ensures unbiased gradient estimation by policy network and reparameterized sampling during the policy gradient steps. From the update formula (19), it can be observed that in policy gradient update step, the only component related to the interaction trajectory is the martingale difference term.

As stated in Theorem 1, when the martingale property is ensured, the policy gradient update step becomes consistent with the on-policy estimation in Theorem 3. We emphasize that the martingale merely serves as a bridge in our approach to reinforcement learning and we approximately ensure the martingale property of the process M s via (19) instead of enforcing it. As mentioned in Section 3, we can choose whether to adopt importance sampling based on the specific problem, as it is not mandatory. The update method (19) effectively balances computational efficiency with mitigating the effects of distribution shift. And

<!-- formula-not-decoded -->

we can taking the martingale difference term δM as the advantage function, the update rule of policy π ϕ aligns with discrete-time formulations.

## C.1 Comparative experiment

To facilitate a comparative analysis of the methods, we adopt the linear quadratic problem presented in Jia and Zhou [2023] as a benchmark. 8 The system dynamics are governed by matrix

<!-- formula-not-decoded -->

and the objective is to maximize the payoff

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In our simulation, the system parameters are set as A = -1 , B = C = 0 , D = 1 , and the initial state x 0 = 0 . The cost function parameters are M = N = Q = 2 , R = P = 1 , with temperature λ = 0 . 1 . The target policy, associated with parameterized q-function, is defined as π ( · | x ) = N ( ψ 1 x + ψ 2 , λe ψ 3 ) in Jia and Zhou [2023] and we approximate this target using a parameterized policy π ( · | x ) = N ( ϕ 1 x + ϕ 2 , e ϕ 3 ) . The experiments use a time discretization step of δt = 0 . 1 . Each of the learning algorithms is run for a sufficiently long duration of T = 10 6 steps.

The behavior policy is set to a normal distribution N ( -x -1 , 1) :

Figure 3: Comparison of learning curves Using behavior policy N ( -x -1 , 1) .

<!-- image -->

The behavior policy is set to a normal distribution N ( x +1 , 1) :

8 The repository includes code accompanied by implementation notes: https://github.com/hh11813/ continuous-soft-actor-critic

Figure 4: Comparison of learning curves using behavior policy N ( x +1 , 1) .

<!-- image -->

We can observe that the other method collapse, while our approach remains stable. This reflects the bias in learning results caused by on-policy method when there is a difference between the behavior policy and the target policy. The presence of the upper and lower bounds on the parameter curve in Figure 3 and Figure 4 is a result of the parameter clipping step incorporated into learning process in Jia and Zhou [2023].

## D Plots and tables

## D.1 Gradients

Gradient explosion can be effectively addressed through gradient clipping. We focus on the vanishing gradient problem in this discussion. Tables 6 and 7 illustrate the gradient changes during the update process with seed { 0 } .

## D.2 Results

Navigation (i) We compare the performance of multiple algorithms under varying time discretization parameters δt , with the baseline hyperparameter discount factor set to γ = 0 . 9 . Results presented in Tables 4 and 8 consistently highlight the sensitivity to δt , indicating a lack of robustness. The experiments reported in Tables 2, 3 and 1 employ 3 × 10 5 frames, while Table 4 use 1 . 2 × 10 5 frames, all with random seeds { 0 , 1 , 2 } .

Table 8: Aggregate scores under navigation

| δt =0.1        | QMIX                                | IQL                           | MAPPO             | MASAC             |
|----------------|-------------------------------------|-------------------------------|-------------------|-------------------|
| Median IQM     | 0.97 [0.97, 0.97] 0.97 [0.97, 0.97] | 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] | 0.98 [0.98, 0.98] | 0.97 [0.97, 0.97] |
|                |                                     |                               | 0.98 [0.98, 0.98] | 0.97 [0.97, 0.97] |
| Mean           | 0.97 [0.97, 0.97]                   | 1.0 [1.0, 1.0]                | 0.98 [0.98, 0.98] | 0.97 [0.97, 0.97] |
| Optimality Gap | 0.03 [0.03, 0.03]                   | 0.0 [0.0, 0.0]                | 0.02 [0.02, 0.02] | 0.03 [0.03, 0.03] |
| δt =0.01       | QMIX                                | IQL                           | MAPPO             | MASAC             |
| Median         | 0.63 [0.57, 0.67]                   | 0.61 [0.6, 0.62]              | 0.55 [0.54, 0.57] | 0.55 [0.54, 0.56] |
| IQM            | 0.63 [0.57, 0.67]                   | 0.61 [0.6, 0.62]              | 0.55 [0.54, 0.57] | 0.55 [0.54, 0.56] |
| Mean           | 0.63 [0.57, 0.67]                   | 0.61 [0.6, 0.62]              | 0.55 [0.54, 0.57] | 0.55 [0.54, 0.56] |
| Optimality Gap | 0.37 [0.33, 0.43]                   | 0.39 [0.38, 0.4]              | 0.45 [0.43, 0.46] | 0.45 [0.44, 0.46] |
| δt             | MADDPG (0.1)                        | MADDPG (0.01)                 |                   |                   |
| Median         | 0.91 [0.88, 0.93]                   | 0.7 [0.52, 0.89]              |                   |                   |
| IQM            | 0.91 [0.88, 0.93]                   | 0.7 [0.52, 0.89]              |                   |                   |
| Mean           | 0.91 [0.88, 0.93]                   | 0.7 [0.52, 0.89]              |                   |                   |
| Optimality Gap | 0.09 [0.07, 0.12]                   | 0.3 [0.11, 0.48]              |                   |                   |

Table 6: Gradients of MADDPG (critic network) under sampling

| δt =0.1                                                                                                                                                           | δt =0.01                                                                                                                                                          |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlp.params.4.bias: 0.0382 mlp.params.4.weight: 0.1038 mlp.params.2.bias: 0.0237 mlp.params.2.weight: 0.0594 mlp.params.0.bias: 0.0353 mlp.params.0.weight: 0.0793 | mlp.params.4.bias: 0.0007 mlp.params.4.weight: 0.0047 mlp.params.2.bias: 0.0004 mlp.params.2.weight: 0.0043 mlp.params.0.bias: 0.0003 mlp.params.0.weight: 0.0032 |
| mlp.params.4.bias: 0.0260 mlp.params.4.weight: 0.0797 mlp.params.2.bias: 0.0161 mlp.params.2.weight: 0.0444 mlp.params.0.bias: 0.0241 mlp.params.0.weight: 0.0571 | mlp.params.4.bias: 0.0108 mlp.params.4.weight: 0.0172 mlp.params.2.bias: 0.0058 mlp.params.2.weight: 0.0156 mlp.params.0.bias: 0.0042 mlp.params.0.weight: 0.0128 |
| mlp.params.4.bias: 0.0190 mlp.params.4.weight: 0.0502 mlp.params.2.bias: 0.0123 mlp.params.2.weight: 0.0522 mlp.params.0.bias: 0.0191 mlp.params.0.weight: 0.0910 | mlp.params.4.bias: 0.0133 mlp.params.4.weight: 0.0193 mlp.params.2.bias: 0.0071 mlp.params.2.weight: 0.0170 mlp.params.0.bias: 0.0051 mlp.params.0.weight: 0.0141 |
| mlp.params.4.bias: 0.0188 mlp.params.4.weight: 0.0383 mlp.params.2.bias: 0.0120 mlp.params.2.weight: 0.0453 mlp.params.0.bias: 0.0184 mlp.params.0.weight: 0.0828 | mlp.params.4.bias: 0.0076 mlp.params.4.weight: 0.0165 mlp.params.2.bias: 0.0041 mlp.params.2.weight: 0.0153 mlp.params.0.bias: 0.0029 mlp.params.0.weight: 0.0110 |

Table 7: Gradients of CMASAC (critic network) under navigation ( δt =0.01)

| CMASAC                                                                                                                                                            | TEST                                                                                                                                                              |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlp.params.4.bias: 0.1544 mlp.params.4.weight: 2.7737 mlp.params.2.bias: 0.0648 mlp.params.2.weight: 0.1422 mlp.params.0.bias: 0.7261 mlp.params.0.weight: 0.9115 | mlp.params.4.bias: 0.2554 mlp.params.4.weight: 4.0866 mlp.params.2.bias: 0.0002 mlp.params.2.weight: 0.0033 mlp.params.0.bias: 0.0040 mlp.params.0.weight: 0.0447 |
| mlp.params.4.bias: 0.5774 mlp.params.4.weight: 9.1671 mlp.params.2.bias: 0.0168 mlp.params.2.weight: 0.0744 mlp.params.0.bias: 0.1725 mlp.params.0.weight: 0.4351 | mlp.params.4.bias: 0.5774 mlp.params.4.weight: 9.2355 mlp.params.2.bias: 0.0012 mlp.params.2.weight: 0.0059 mlp.params.0.bias: 0.0181 mlp.params.0.weight: 0.0448 |
| mlp.params.4.bias: 0.2314 mlp.params.4.weight: 3.9341 mlp.params.2.bias: 0.0493 mlp.params.2.weight: 0.1036 mlp.params.0.bias: 0.5558 mlp.params.0.weight: 0.6696 | mlp.params.4.bias: 0.2569 mlp.params.4.weight: 4.1117 mlp.params.2.bias: 0.0006 mlp.params.2.weight: 0.0043 mlp.params.0.bias: 0.0106 mlp.params.0.weight: 0.0666 |

Table 9: Aggregate scores under navigation

|                | CMASAC            |
|----------------|-------------------|
| Median         | 0.96 [0.93, 0.99] |
| IQM            | 0.96 [0.92, 1.0]  |
| Mean           | 0.96 [0.93, 0.99] |
| Optimality Gap | 0.04 [0.01, 0.07] |

(ii) By increasing the number of random seeds to four and using more frames ( 6 × 10 5 ), we explore the performance of the proposed algorithm under δt = 0 . 01 . Comparison with the baseline results in Table 2 demonstrates that variations in extraneous factors such as random seeds did not have a decisive impact on the method's performance.

(iii) To separately examine the effects of the martingale orthogonality condition and parameter scaling, we compare the performance of the non-scaling CMASAC algorithm (TEST) and the MASAC algorithm in a navigation task using 6 × 10 5 frames, with random seeds { 0 , 1 , 2 } and δt = 0 . 01 . Table 10 and Figure 5, 6 present the experimental results. Table 10 indicates that the martingale orthogonality condition enhances the performance of the algorithm when δt is small.

Table 10: Aggregate scores under navigation

|                | TEST             |
|----------------|------------------|
| Median         | 0.96 [0.93, 1.0] |
| IQM            | 0.96 [0.93, 1.0] |
| Mean           | 0.96 [0.93, 1.0] |
| Optimality Gap | 0.04 [0.0, 0.07] |

<!-- image -->

Figure 5: Performance profile for navigation task

Figure 6: Aggregate score performance for navigation task

<!-- image -->

The TEST algorithm updates parameters according to modified version of Equation (19) in Algorithm 1:

<!-- formula-not-decoded -->

with learning rates α θ and α ϕ .

Sampling (i) In Section 4 of the paper, we evaluate the performance of the MASAC and MADDPG algorithms in the sampling task. Here, we present results form additional algorithms.

Table 11: Aggregate scores under sampling

| δt =0.1        | QMIX              | IQL               | MAPPO             |
|----------------|-------------------|-------------------|-------------------|
| Median         | 0.87 [0.75, 1.0]  | 0.93 [0.89, 0.96] | 0.53 [0.45, 0.62] |
| IQM            | 0.87 [0.75, 1.0]  | 0.93 [0.89, 0.96] | 0.53 [0.45, 0.62] |
| Mean           | 0.87 [0.75, 1.0]  | 0.93 [0.89, 0.96] | 0.53 [0.45, 0.62] |
| Optimality Gap | 0.13 [0.0, 0.25]  | 0.07 [0.04, 0.11] | 0.47 [0.38, 0.55] |
| δt =0.01       | QMIX              | IQL               | MAPPO             |
| Median         | 0.53 [0.46, 0.57] | 0.67 [0.46, 1.0]  | 0.45 [0.29, 0.76] |
| IQM            | 0.53 [0.46, 0.57] | 0.67 [0.46, 1.0]  | 0.45 [0.29, 0.76] |
| Mean           | 0.53 [0.46, 0.57] | 0.67 [0.46, 1.0]  | 0.45 [0.29, 0.76] |
| Optimality Gap | 0.47 [0.43, 0.54] | 0.33 [0.0, 0.54]  | 0.55 [0.24, 0.71] |

Similar to Table 8, when the discount factor is set to γ = 0 . 9 , the results in Table 12 demonstrate that the algorithms exhibit unstable performance with respect to time discretization δt .

Table 12: Aggregate scores under sampling

| δt =0.1        | QMIX              | IQL               | MAPPO             |
|----------------|-------------------|-------------------|-------------------|
| Median         | 0.93 [0.89, 0.98] | 0.93 [0.88, 1.0]  | 0.54 [0.44, 0.59] |
| IQM            | 0.93 [0.89, 0.98] | 0.93 [0.88, 1.0]  | 0.54 [0.44, 0.59] |
| Mean           | 0.93 [0.89, 0.98] | 0.93 [0.88, 1.0]  | 0.54 [0.44, 0.59] |
| Optimality Gap | 0.07 [0.02, 0.11] | 0.07 [0.0, 0.12]  | 0.46 [0.41, 0.56] |
| δt =0.01       | QMIX              | IQL               | MAPPO             |
| Median         | 0.62 [0.52, 0.72] | 0.62 [0.52, 0.71] | 0.51 [0.33, 0.86] |
| IQM            | 0.62 [0.52, 0.72] | 0.62 [0.52, 0.71] | 0.51 [0.33, 0.86] |
| Mean           | 0.62 [0.52, 0.72] | 0.62 [0.52, 0.71] | 0.51 [0.33, 0.86] |
| Optimality Gap | 0.38 [0.28, 0.48] | 0.38 [0.29, 0.48] | 0.49 [0.14, 0.67] |

For the sampling task with the discount factor γ = 0 . 99 , CMASAC maintains superior performance under the time discretization δt = 0 . 01 . As demonstrated by the consistent results in Table 13, adjustments to the discount factor do not compromise the robustness of CMASAC, highlighting its stability across varying hyperparameter configurations. Experiments in Tables 11, 12 and 13 were conducted using different numbers of frames ( 6 × 10 4 , 2 . 4 × 10 5 ).

(ii) For the extended evaluation, we test CMASAC with 3 × 10 6 frames and δt = 0 . 01 on sampling , the corresponding results are shown in Table 14 and 15. The unscaled continuous algorithm (TEST) demonstrated superior results compared to other methods (Table 15, Figures 7 and 8). Cross-referencing Tables 14 and 15 further validates that martingale orthogonality condition and hyperparameter scaling jointly contribute to performance improvements.

Table 13: Aggregate scores of CMASAC under sampling

|                | CMASAC           |
|----------------|------------------|
| IQM            | 0.95 [0.92, 1.0] |
| Mean           | 0.95 [0.92, 1.0] |
| Median         | 0.95 [0.92, 1.0] |
| Optimality Gap | 0.05 [0.0, 0.08] |

Table 14: Aggregate scores under sampling

|                | CMASAC           |
|----------------|------------------|
| Median         | 0.97 [0.94, 1.0] |
| IQM            | 0.97 [0.94, 1.0] |
| Mean           | 0.97 [0.94, 1.0] |
| Optimality Gap | 0.03 [0.0, 0.06] |

Table 15: Aggregate scores under sampling

Figure 7: Performance profile for sampling task

|                | TEST             | MASAC            | MAPPO             | MADDPG           |
|----------------|------------------|------------------|-------------------|------------------|
| Median         | 0.92 [0.84, 1.0] | 0.81 [0.8, 0.81] | 0.76 [0.72, 0.81] | 0.8 [0.77, 0.84] |
| IQM            | 0.92 [0.84, 1.0] | 0.81 [0.8, 0.81] | 0.76 [0.72, 0.81] | 0.8 [0.77, 0.84] |
| Mean           | 0.92 [0.84, 1.0] | 0.81 [0.8, 0.81] | 0.76 [0.72, 0.81] | 0.8 [0.77, 0.84] |
| Optimality Gap | 0.08 [0.0, 0.16] | 0.19 [0.19, 0.2] | 0.24 [0.19, 0.28] | 0.2 [0.16, 0.23] |

<!-- image -->

Figure 8: Aggregate score performance for sampling task

<!-- image -->

Balance We evaluate the proposed algorithm in balance environment. The result in Table 16 demonstrates the potential of CMASAC in near-continuous-time settings. A comparison of the results in Tables 16 and 17 under δt = 0 . 01 reveals that both the martingale orthogonality condition and hyperparameter scaling contribute to improved performance and increased robustness of the

algorithms in near-continuous environments. Test results of the method in more scenarios remain to be explored.

Table 16: Aggregate scores of CMASAC under balance

| δt             | 0.01             |
|----------------|------------------|
| IQM            | 0.94 [0.88, 1.0] |
| Mean           | 0.94 [0.88, 1.0] |
| Median         | 0.94 [0.88, 1.0] |
| Optimality Gap | 0.06 [0.0, 0.12] |

Table 17: Aggregate scores of TEST under balance

| δt             | 0.01             |
|----------------|------------------|
| IQM            | 0.93 [0.89, 1.0] |
| Mean           | 0.93 [0.89, 1.0] |
| Median         | 0.93 [0.89, 1.0] |
| Optimality Gap | 0.07 [0.0, 0.11] |

## D.3 Significance

The experimental evaluation in this study was conducted using tool (based on Agarwal et al. [2021] and Gorsane et al. [2022]): https://github.com/instadeepai/marl-eval .

Considering the setting in which a reinforcement learning algorithm is evaluated on M tasks and N independent runs executed for each task. Then we derive normalized score x m,n , m = 1 , · · · , M and n = 1 , · · · , N . Here, we briefly describe the meaning of the experimental results tables and figures.

- IQM: Inter-quantile Mean
- Optimality Gap: Optimality Gap can be thought of as the how far an algorithm is from optimal performance at a given task.
- Confidence interval (CI): These provides an estimated possible range for an unknown value. We choose a 95 confidence interval in our experiments.
- Aggregate score performance: The confidence intervals shown alongside the point estimates (black bars) are the 95 stratified bootstrap confidence intervals.
- Performance profiles: The algorithm's normalized score on the m th task as a real-valued random variable X m . Then, the score x m,n is a realization of the random variable X m,n , which is identically distributed as X m . For τ ∈ R , we define the tail distribution function of X m as F m ( τ ) = P ( X m &gt; τ ) . For any collection of scores y 1: K , the empirical tail distribution function is given by ˆ F ( τ ; y 1: K ) = 1 K ∑ K k =1 1 [ y k &gt; τ ] . In particular, we write ˆ F m ( τ ) = ˆ F ( τ ; x m, 1: N ) . This explains the meaning of Figure 7 and the significance of the axes.

## E Implementation details

## E.1 Environments

Figures 1 and 9 illustrate several VMAS task scenarios.

- Navigation : Randomly spawned agents (circles with surrounding dots) need to navigate to randomly spawned goals (smaller circles). Agents need to use LIDARs (dots around them) to avoid running into each other. For each agent, we compute the difference in the relative distance to its goal over two consecutive timesteps. The mean of these values over all agents composes the shared reward, incentivizing agents to move towards their goals. Each agent observes its position, velocity, lidar readings, and relative position to its goal.

<!-- image -->

(a) Balance.

Figure 9: VMAS multi-robot control task-Balance-used in the experiments.

- Sampling : Agents are spawned randomly in a workspace with an underlying Gaussian density function composed of three Gaussian modes. Agents need to collect samples by moving in this field. The field is discretized to a grid (with agent-sized cells) and once an agent visits a cell its sample is collected without replacement and given as a reward to the whole team. Agents can use a lidar to sense each other in order to coordinate exploration. Apart from lidar, position, and velocity observations, each agent observes the values of samples in the 3x3 grid around it.
- Balance : Agents (blue circles) are spawned uniformly spaced out under a line upon which lies a spherical package (red circle). The team and the line are spawned at a random x position at the bottom of the environment. The environment has vertical gravity. The relative x position of the package on the line is random. In the top half of the environment, a goal (green circle) is spawned. The agents have to carry the package to the goal. Each agent receives the same reward which is proportional to the distance variation between the package and the goal over two consecutive timesteps. The team receives a negative reward of -10 for making the package or the line fall to the floor. The observations for each agent are: its position, velocity, relative position to the package, relative position to the line, relative position between package and goal, package velocity, line velocity, line angular velocity, and line rotation mod π . The environment is done either when the package or the line falls or when the package touches the goal.

## E.2 Hyperparameters

Random seeds We select random seeds { 0 , 1 , 2 } .

Network architecture The policy and critic models are constructed with MLP layers, and the MLP architecture is defined as follows:

- num\_cells: [256, 256]
- layer\_class: torch.nn.Linear
- activation\_class: torch.nn.Tanh

Hyperparameters details Tables 18, 19 and 20 show configurations of different algorithms. These algorithm-specific hyperparameters take precedence over the common hyperparameters.

Table 20: Config details of selected algorithms

| QMIX                                                       |
|------------------------------------------------------------|
| delay_value: True loss_function: "l2" mixing_embed_dim: 32 |

And the shared parameters across all experimental algorithms are listed below:

Table 18: Config details of selected algorithms

| IQL                                   | MASAC                                                                                                                                                                                                                                                                       | MADDPG                                                                                |
|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| loss_function: "l2" delay_value: True | share_param_critic: True num_qvalue_nets: 2 loss_function: "l2" delay_qvalue: True target_entropy: "auto" discrete_target_entropy_weight: 0.2 alpha_init: 1.0 min_alpha: null max_alpha: null fixed_alpha: False scale_mapping: "biased_softplus_1.0" use_tanh_normal: True | share_param_critic: True loss_function: "l2" delay_value: True use_tanh_mapping: True |

Table 19: Config details of selected algorithms

| MAPPO                                                                                                                                                                                      | CMASAC                                                                                                                                                                                                              |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| share_param_critic: True clip_epsilon: 0.2 entropy_coef: 0.0 critic_coef: 1.0 loss_critic_type: "l2" minibatch_advantage: False scale_mapping: "biased_softplus_1.0" use_tanh_normal: True | share_param_critic: True num_value_nets: 1 loss_function: "l2" target_entropy: "auto" alpha_init: 1.0 min_alpha: null max_alpha: null fixed_alpha: False scale_mapping: "biased_softplus_1.0" use_tanh_normal: True |

- share\_policy\_params: True
- (discount factor) gamma: 0.99
- (learning rate) lr: 0.00005
- (adam optimizer) adam\_eps: 0.000001
- (soft target update) polyak\_tau: 0.005
- (initial epsilon for annealing) exploration\_eps\_init: 0.8
- (final epsilon after annealing) exploration\_eps\_end: 0.01
- max\_n\_frames: 3\_000\_000
- on\_policy\_collected\_frames\_per\_batch: 6000
- on\_policy\_n\_envs\_per\_worker: 10
- on\_policy\_n\_minibatch\_iters: 45
- on\_policy\_minibatch\_size: 400
- off\_policy\_collected\_frames\_per\_batch: 6000
- off\_policy\_n\_envs\_per\_worker: 10
- off\_policy\_n\_optimizer\_steps: 1000
- off\_policy\_train\_batch\_size: 128
- off\_policy\_memory\_size: 1\_000\_000
- off\_policy\_init\_random\_frames: 0
- off\_policy\_use\_prioritized\_replay\_buffer: False

- evaluation\_interval: 120\_000
- evaluation\_episodes: 10
- evaluation\_deterministic\_actions: False

## Evaluation details

- Evaluation intervals: It refers to the fixed number of time steps, after which training is suspended, to be able to evaluate an algorithm for a fixed number of runs/epsiodes. The evaluation frequency must ideally be associated with a duration which we record as the evaluation duration.
- Number of independent evaluations per interval: This is the amount of evaluations that are performed at each evaluation interval.

Computer resources The experiments are conducted on a system equipped with Intel Xeon Silver 4314 CPU (2.40GHz, 16 physical cores) and an NVIDIA RTX 4090 GPU (24GB VRAM). Each independent algorithm experiment run consumes about 1.5GB of CPU RAM and 2.2GB of GPU VRAM on average.