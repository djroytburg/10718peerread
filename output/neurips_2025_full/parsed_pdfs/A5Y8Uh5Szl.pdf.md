## Exploration from a Primal-Dual Lens: Value-Incentivized Actor-Critic Methods for Sample-Efficient Online RL

Tong Yang ∗ CMU

Bo Dai † Georgia Tech

## Abstract

Online reinforcement learning (RL) with complex function approximations such as transformers and deep neural networks plays a significant role in the modern practice of artificial intelligence. Despite its popularity and importance, balancing the fundamental trade-off between exploration and exploitation remains a longstanding challenge; in particular, we are still in lack of efficient and practical schemes that are backed by theoretical performance guarantees. Motivated by recent developments in exploration via optimistic regularization, this paper provides an interpretation of the principle of optimism through the lens of primal-dual optimization. From this fresh perspective, we set forth a new value-incentivized actor-critic (V AC) method, which optimizes a single easy-to-optimize objective integrating exploration and exploitation - it promotes state-action and policy estimates that are both consistent with collected data transitions and result in higher value functions. Theoretically, the proposed V AC method has near-optimal regret guarantees under linear Markov decision processes (MDPs) in both finite-horizon and infinite-horizon settings, which can be extended to the general function approximation setting under appropriate assumptions.

## 1 Introduction

In online reinforcement learning (RL) [Sutton et al., 1998], an agent learns to update their policy in an adaptive manner while interacting with an unknown environment to maximize long-term cumulative rewards. In conjunction with complex function approximation such as large neural networks and foundation models to reduce dimensionality, online RL has achieved remarkable performance in a wide variety of applications such as game playing [Silver et al., 2017], control [Mnih et al., 2015], language model post-training [OpenAI, 2023, Team et al., 2023] and reasoning [Guo et al., 2025], and many others.

Despite its popularity, advancing beyond current successes is severely bottlenecked by the cost and constraints associated with data collection. While simulators can subsidize data acquisition in certain domains, many real-world applications-such as clinical trials, recommendation systems and autonomous driving-operate under conditions where gathering interaction data is expensive, timeconsuming or potentially risky. In these high-stake scenarios, managing the fundamental yet delicate trade-off between exploration (gathering new information about the environment) and exploitation (leveraging existing knowledge to maximize rewards) requires paramount care. Naive exploration schemes, such as the ϵ -greedy method, are known to be sample-inefficient as they explore randomly

∗ Carnegie Mellon University; Emails: tongyang@andrew.cmu.edu .

† Georgia Institute of Technology; Email: bodai@cc.gatech.edu .

‡ Fundamental AI Research, Meta; Email: linx@meta.com .

§ Yale University; Emails: yuejie.chi@yale.edu .

Lin Xiao ‡ Meta

Yuejie Chi § Meta &amp; Yale

without strategic information gathering [Dann et al., 2022]. Arguably, it is still an open challenge to develop practical online RL algorithms that come with provable sample-efficiency guarantees, especially in the presence of function approximation.

Addressing this limitation, significant research attempts have been made to develop statistically efficient approaches, often guided by the principle of optimism in the face of uncertainty [Lattimore and Szepesvári, 2020]. Prominent approaches include constructing optimistic estimates with data-driven confidence sets [Auer et al., 2008, Agarwal et al., 2023, Chen et al., 2025, Foster et al., 2021], as well as employing Bayesian methods like Thompson sampling [Russo et al., 2018] and its optimistic variants [Agrawal and Jia, 2017, Zhang, 2022]. While appealing theoretically, translating them into practical algorithms compatible with general function approximators often proves difficult. Many such theoretically-grounded approaches either suffer from prohibitive computational complexity or exhibit underwhelming empirical performance when scaled to complex problems.

Recently, Liu et al. [2024] introduced an intriguing framework termed Maximize to Explore (MEX) for online RL, which optimizes a single objective function over the state-action value function (i.e., Q -function), elegantly unifying estimation, planning and exploration in one framework. In addition, MEX comes with appealing sub-linear regret guarantees under function approximation. However, the practical optimization of the MEX objective presents significant challenges due to its inherent bi-level structure. Specifically, it incorporates the optimal value function derived from the target Q -function as a regularizer [Kumar and Becker, 1982], which is not directly amenable to first-order optimization toolkits. As a result, nontrivial modifications are introduced in the said implementation of MEX, making it challenging to ablate the benefit of the MEX framework. This practical hurdle raises a crucial question:

Can we design a sample-efficient model-free online RL algorithm that optimizes a unifying objective function, but without resorting to complex bilevel optimization?

## 1.1 Our contribution

In this paper, we answer this question in the affirmative, introducing a novel actor-critic method that achieves near-optimal regret guarantees by optimizing a single non-bilevel objective. Our contributions are summarized as follows.

- Incentivizing exploration from the primal-dual perspective. We start by offering a new interpretation of MEX, where optimistic regularization-central to MEX-arises naturally from a Lagrangian formulation within a primal-dual optimization perspective [Dai et al., 2018, Nachum and Dai, 2020]. Specifically, we demonstrate that the seemingly complex MEX objective function can be derived as the regularized Lagrangian of a canonical value maximization problem, subject to the constraint that the Q -function satisfies the Bellman optimality equation . This viewpoint allows deeper understanding of the structure of the MEX objective and its exploration mechanism.
- VAC: Value-incentivized actor-critic method. Motivated by this Lagrangian interpretation, we develop the value-incentivized actor-critic (V AC) method for online RL, which jointly optimizes the Q -function and the policy under function approximation over a single objective function. Different from MEX, VAC optimizes a regularized Lagrangian constructed with respect to the Bellman consistency equation as the constraint, naturally accommodating the interplay between the Q -function and the policy. This formulation preserves the crux of optimistic regularization, while allowing differentiable optimization of the Q -function and the policy simultaneously under general function approximation.
- Theoretical guarantees of VAC. Wesubstantiate the efficacy of VAC with rigorous theoretical analysis, by proving it achieves a rate of ˜ O ( dH 2 √ T ) regret under the setting of episodic linear Markov decision processes (MDPs) [Jin et al., 2020], where d is the feature dimension, H is the horizon length, and T is the number of episodes. We further extend the analysis to the infinite-horizon discounted setting and the general function approximation setting under similar assumptions of prior art [Liu et al., 2024].

In summary, our work bridges the gap between theoretically efficient exploration principles and practical applicability in challenging online RL settings with function approximation.

## 1.2 Related work

We discuss a few lines of research that are closely related to our setting, focusing on those with theoretical guarantees under function approximation.

Regret bounds for online RL under function approximation. Balancing the explorationexploitation trade-off is of fundamental importance in the design of online RL algorithms. Most existing methods with provable guarantees rely on the construction of confidence sets and perform constrained optimization within the confident sets, including model-based [Wang et al., 2025, Foster et al., 2023b, Chen et al., 2025], value-based [Agarwal et al., 2023, Jin et al., 2021, Xie et al., 2023], policy optimization [Liu et al., 2023], and actor-critic [Tan et al., 2025] approaches, to name a few. Regret guarantees for approaches based on posterior sampling [Osband and Van Roy, 2017] are provided in [Zhong et al., 2022, Li and Luo, 2024, Agarwal and Zhang, 2022] under function approximation. Regret analysis under the linear MDP model [Jin et al., 2020] has also been actively established for various methods, e.g., for the episodic setting [Zanette et al., 2020, Jin et al., 2020, Papini et al., 2021] and for the infinite-horizon setting [Zhou et al., 2021, Moulin et al., 2025]. However, the confident sets computation and posterior estimation are usually intractable with general function approximator, making the algorithm difficult to be applied.

Exploration via optimistic estimation. Exploration via optimistic estimation has been actively studied recently due to its promise in practice, which has been examined over a wide range of settings such as bandits [Kumar and Becker, 1982, Liu et al., 2020, Hung et al., 2021], RL with human feedback [Cen et al., 2024, Xie et al., 2024, Zhang et al., 2024], single-agent RL [Mete et al., 2021, Liu et al., 2024, Chen et al., 2025], and Markov games [Foster et al., 2023a, Xiong et al., 2024, Yang et al., 2025]. Tailored to online RL, most of the optimistic estimation algorithms are model-based, with a few exceptions such as the model-free variant of MEX in [Liu et al., 2020], but still with computationally challenges.

Primal-dual optimization in RL. Primal-dual formulation has been exploited in RL for handling the 'double-sampling" issue [Dai et al., 2017, 2018] from an optimization perspective. By connecting through the linear programming view of MDP [De Farias and Van Roy, 2004, Puterman, 2014, Wang, 2017, Neu et al., 2017, Lakshminarayanan et al., 2017, Bas-Serrano et al., 2021], a systematic framework [Nachum et al., 2019b] has been developed for offline RL, which induces concrete algorithms for off-policy evaluation [Nachum et al., 2019a, Uehara et al., 2020, Yang et al., 2020], confidence interval evaluation [Dai et al., 2020], imitation learning [Kostrikov et al., 2019, Zhu et al., 2020, Ma et al., 2022, Sikchi et al., 2023], and policy optimization [Nachum et al., 2019b, Lee et al., 2021]. However, how to exploit the primal-dual formulation in online RL setting has not been investigated formally to the best of our knowledge.

Paper organization and notation. The rest of this paper is organized as follows. We describe the background, and illuminate the connection between exploration and primal-dual optimization in Section 2. We present the proposed VAC method, and state its regret guarantee in Section 3. Section 4 provide numerical experiments to corroborate the effectiveness of the proposed method. Finally, we conclude in Section 5. The proofs and generalizations to the infinite-horizon and general function approximation settings are deferred to the appendix.

Notation. Let ∆( A ) be the probability simplex over the set A , and [ n ] denote the set { 1 , . . . , n } . For any x ∈ R n , we let ‖ x ‖ p denote the ℓ p norm of x , where p ∈ [1 , ∞ ] . The d -dimensional ℓ 2 ball of radius R is denoted by B d 2 ( R ) , and the d × d identity matrix is denoted by I d .

## 2 Background and Motivation

## 2.1 Background

Episodic Markov decision processes. Let M = ( S , A , P, r, H ) be a finite-horizon episodic MDP, where S and A denote the state space and the action space, respectively, H ∈ N + is the horizon length, and P = { P h } h ∈ [ H ] and r = { r h } h ∈ [ H ] are the inhomogeneous transition kernel and the reward function: for each time step h ∈ [ H ] , P h : S × A ↦→ ∆( S ) specifies the probability

distribution over the next state given the current state and action at step h , and r h : S × A ↦→ [0 , 1] is the reward function at step h . We let π = { π h } h ∈ [ H ] : S × [ H ] ↦→ ∆( A ) denote the policy of the agent, where π h ( ·| s ) ∈ ∆( A ) specifies an action selection rule at time step h .

<!-- formula-not-decoded -->

For any given policy π , the value function at step h , denoted by V π h : S ↦→ R , is given as which measures the expected cumulative reward starting from state s at time step h until the end of the episode. The expectation is taken over the randomness of the trajectory generated following a i ∼ π i ( ·| s i ) and the MDP dynamics s i +1 ∼ P i ( ·| s i , a i ) for i = h, . . . , H . We define V π H ( s ) := 0 for all s ∈ S . The value function at the beginning of the episode, when h = 1 , is often denoted simply as V π ( s ) := V π 1 ( s ) . Given an initial state distribution s 1 ∼ ρ over S , we also define V π ( ρ ) := E s 1 ∼ ρ [ V π 1 ( s 1 )] .

Similarly, the Q -function of policy π at step h , denoted by Q π h : S × A ↦→ R , is defined as which measures the expected discounted cumulative reward starting from state s and taking action a at time step h , and following policy π thereafter, according to the time-dependent transitions. We define Q π H +1 ( s, a ) := 0 and Q π ( s, a ) := Q π 1 ( s, a ) for all ( s, a ) ∈ S × A . They satisfy the Bellman consistency equation, given by, for all ( s, a ) ∈ S × A , h ∈ [ H ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is known that there exists at least one optimal policy π ⋆ = ( π ⋆ 1 , . . . , π ⋆ H ) that maximizes the value function V π ( s ) for all initial states s ∈ S [Puterman, 2014]. The corresponding optimal value function and Q-function are denoted as V ⋆ and Q ⋆ , respectively. In particular, they satisfy the Bellman optimality equation, given by, for all ( s, a ) ∈ S × A , h ∈ [ H ] :

<!-- formula-not-decoded -->

Goal: regret minimization in online RL. In this paper, we are interested in the online RL setting, where the agent interacts with the episodic MDP sequentially for T episodes, where in the t -th episode ( t ⩾ 1 ), the agent executes a policy π t = { π t,h } H h =1 learned based on the data collected up to the ( t -1) -th episode. To evaluate the performance of the learned policy, our goal is to minimize the cumulative regret, defined as

<!-- formula-not-decoded -->

which measures the sub-optimality gap between the values of the optimal policy and the learned policies over T episodes. In particular, we would like the regret to scale sub-linearly in T , so the sub-optimality gap is amortized over time.

## 2.2 Motivation: revisiting MEX from primal-dual lens

Recently, MEX [Liu et al., 2024] emerges as a promising framework for online RL, which balances exploration and exploitation in a single objective while naturally enabling function approximation. Consider a function class Q = ∏ H h =1 Q h of the Q -function. For any f = { f h } h ∈ [ H ] ∈ Q , we denote the corresponding Q -function Q f = { Q f,h } h ∈ [ H ] with Q f,h = f h . At the beginning of the t -th episode, given the collection D t -1 ,h of transition tuples ( s h , a h , s h +1 ) at step h up to the ( t -1) -th episode, MEX [Liu et al., 2024] (more precisely, its model-free variant) updates the Q-function estimate as where α ⩾ 0 is some regularization parameter, and L t ( f ) is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which satisfies

<!-- formula-not-decoded -->

Consequently, by setting δ h ( s, a ) := r h ( s, a ) + max a ∈A Q f,h +1 ( s ′ , a ) in (11), the Lagrangian objective (9) becomes

<!-- formula-not-decoded -->

By replacing the population distribution D h with its samples in D t -1 ,h at each round, then we recover the model-free MEX algorithm in (7).

However, (6) is a bilevel optimization problem where in the lower level, another optimization problem max a ∈A Q f,h ( s, a ) needs to be computed in (7). This can be can be computationally intensive if not intractable. In this paper, inspired from this primal-dual view, we derive a more implementable algorithm.

5 It is possible to use an ( s, a, h ) -dependent regularization too.

<!-- formula-not-decoded -->

where ξ h = ( s h , a h , s h +1 ) is the transition tuple. The first term in (6) promotes exploration by searching for Q -functions with higher values, while the second term ensures the Bellman consistency of the Q -function with the collected data transitions. The policy is then updated greedily from Q f t to collect the next batch of data. While Liu et al. [2024] offered strong regret guarantees of MEX, there is little insight provided into the design of (6), which is deeply connected to the reward-biased framework in Kumar and Becker [1982].

Interpretation from primal-dual lens. We offer a new interpretation of MEX, where optimistic regularization arises naturally from a regularized Lagrangian formulation of certain constrained value maximization problem within a primal-dual optimization perspective. As a brief detour to build intuition, we consider a value maximization problem over the Q-function with the exact (i.e., population) Bellman optimality equation as the constraints:

with the boundary condition Q f,H +1 = 0 . When the optimal Q -function is realizable, i.e., Q ⋆ ∈ Q , the unique solution of (8) recovers the true optimal Q -function Q ⋆ .

<!-- formula-not-decoded -->

How is this connected to the MEX objective? Introducing the dual variables { λ h } h ∈ [ H ] , the regularized Lagrangian of (8) can be written as where β &gt; 0 is the regularization parameter of the dual variable, 5 and D h denotes a properly defined joint distribution over the transition tuples that covers the state-action space over ( s, a ) . We invoke the trick in Dai et al. [2018], Baird [1995], which deals with the double-sampling issue , and reparameterize the dual variable

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 3 Value-incentivized Actor-Critic Method

## 3.1 Algorithm development

We now develop the proposed value-incentivized actor-critic method. In contrast to the model-free MEX for (12), we consider a value maximization problem over both the Q -function and the policy with the exact (i.e., population) Bellman consistency equation as the constraints:

where P = ∏ H h =1 P h is the policy class. This formulation explicits the joint optimization over the Q -function (critic) and the policy (actor), and uses the Bellman's consistency equation as the constraint, rather than the Bellman's optimality equation, which is key to obtain a more tractable optimization problem.

<!-- formula-not-decoded -->

Similar as (9), we can write the regularized Lagrangian of (13) as

<!-- formula-not-decoded -->

Similar to earlier discussion, we also consider the reparameterization (10) which gives

<!-- formula-not-decoded -->

V π f ( s ) := E a ∼ π 1 ( ·| s ) [ Q f, 1 ( s, a )] , and V π f ( ρ ) := E s ∼ ρ [ V π f ( s ) ] . (16) Note that, the objective function (15) is easier to optimize over both Q f and π . Replacing the population distribution D h of ξ = ( s, a, s ′ ) by its empirical samples leads to the proposed algorithm, which is termed value-incentivized actor-critic (V AC) method; see Algorithm 1 for a summary.

## Algorithm 1 Value-incentivized Actor-Critic (VAC) for finite-horizon MDPs

- 1: Input: regularization coefficient α &gt; 0 .
- 2: Initialization: dataset D 0 ,h := ∅ for all h ∈ [ H ] .
- 4: Update Q-function estimation and policy:
- 3: for t = 1 , · · · , T do

<!-- formula-not-decoded -->

- 5: Data collection: run π t to obtain a trajectory { s t, 1 , a t, 1 , s t, 2 , . . . , s t,H +1 } , and update the dataset D t,h ← D t -1 ,h ∪ { ( s t,h , a t,h , s t,h +1 ) } for all h ∈ [ H ] .
- 6: end for

In Algorithm 1, at t -th iteration, given dataset D t -1 ,h of transitions ( s h , a h , s h +1 ) collected from the previous iterations for all h ∈ [ H ] , and use the current policy π t to collect new action a ′ for each tuples, we define the loss function as follows:

<!-- formula-not-decoded -->

where ξ h = ( s h , a h , s h +1 ) is the transition tuple. To approximately solve the optimization problem (17), which is the sample version of (15), we can, in practice, employ first-order method, i.e. ,

- Critic evaluation: Given the policy π t -1 fixed, we solve the saddle-point problem for f t as biased policy evaluation for π t -1 , i.e. ,

<!-- formula-not-decoded -->

- Policy update: Given the critic f is fixed, we can update the policy π through policy gradient following the gradient calculation in Nachum et al. [2019b].

Clearly, the proposed VAC recovers an actor-critic style algorithm, therefore, demonstrating the practical potential of the proposed algorithm. However, we emphasize the critic evaluation step is different from the vanilla policy evaluation, where we have V π f ( ρ ) to bias the policy value. In contrast, MEX only admits an actor-critic implementation for α = 0 (corresponding to vanilla actorcritic when there is no exploration) since their data loss term requires the optimal value function, while the data loss term L t ( f, π ) is policy-dependent in VAC.

## 3.2 Theoretical guarantees

The design of VAC is versatile and can be implemented with arbitrary function approximation. To corroborate its efficacy, however, we focus on understanding its theoretical performance in the linear MDP model, which is popular in the literature [Jin et al., 2020, Lu et al., 2021].

Assumption 1 (linear MDP, Jin et al. [2020]) . There exist unknown vectors ζ h ∈ R d and unknown (signed) measures µ h = ( µ (1) h , · · · , µ ( d ) h ) over S such that where ϕ h : S × A ↦→ R d is a known feature map satisfying ‖ ϕ h ( s, a ) ‖ 2 ⩽ 1 , and max {‖ ζ h ‖ 2 , ‖ µ h ( S ) ‖ 2 } ⩽ √ d , for all ( s, a, s ′ ) ∈ S × A × S and all h ∈ [ H ] .

<!-- formula-not-decoded -->

We also need to specify the function class Q for the Q -function and the policy class P for the policy. Under the linear MDP, it suffices to represent Q -function linearly w.r.t. ϕ h ( s, a ) , i.e. , Q h ( s, a ) = ϕ h ( s, a ) ⊤ θ h , and the log-linear function approximation for the policy derived from the max-entropy policy [Ren et al., 2022], with the following two regularization assumptions on the weights.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under these assumptions, we first state the regret bound of Algorithm 1 in Theorem 1.

Theorem 1. Suppose Assumptions 1-3 hold. We let B = T log |A| dH in Assumption 3, and set

Then for any δ ∈ (0 , 1) , with probability at least 1 -δ , the regret of VAC (cf. Algorithm 1) satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 1 shows that by choosing B = ˜ O ( T/dH ) and α = ˜ O ( 1 H √ T ) , the regret of VAC is no larger than the order of ˜ O ( dH 2 √ T ) up to log -factors. Compared to the minimax lower bound ˜ Ω( d √ H 3 T ) [He et al., 2023], this suggests that our bound is near-optimal up to a factor of √ H , but with practical implementation generalizable to arbitrary function approximator.

Extension to the infinite-horizon setting. Our algorithm and theory can be extended to the infinite-horizon discounted setting leveraging the sampling procedure in Yuan et al. [2023, Algorithm 3]. We demonstrate that the sample complexity of VAC is no larger than ˜ O ( d 2 (1 -γ ) 5 ε 2 ) to return an ε -optimal policy, where γ is the discount factor. This rate is near-optimal up to polynomial factors of 1 1 -γ and logarithmic factors. We leave the details to the appendix.

Extension to the general function approximation. Our theoretical analysis can also be extended to general function approximation, under standard assumptions for general function approximation such as low generalized Eluder coefficient (GEC) [Zhong et al., 2022, Liu et al., 2024]. The corresponding tight regret bound is provided in Appendix B.3, which matches the bound given in Liu et al. [2024, Corollary 5.2] under similar assumptions.

Extension to KL-regularized MDPs. Recently, MDPs regularized by the Kullback-Leibler (KL) divergence KL ( π ‖ π ref ) , with respect to a reference policy π ref = { π ref ,h } h ∈ [ H ] : S × [ H ] ↦→ ∆( A ) , has attracted much attention for preventing over-optimization and increasing stability of the learning process [Ouyang et al., 2022, Yang et al., 2025]. Our framework of VAC can be extended straightforwardly, by invoking the soft Bellman consistency equation in the derivation:

<!-- formula-not-decoded -->

where τ &gt; 0 is the regularization parameter. We omit the details for conciseness.

## 4 Experiments

We provide numerical experiments to demonstrate the efficacy of the value-incentivized regularization in the actor-critic framework.

Setup. Weevaluate on two challenging continuous-control benchmarks in MuJoCo [Todorov et al., 2012]: Ant-v4 and Walker2d-v4. For the base learner, we adopt Soft Actor-Critic (SAC) implemented in Stable-Baselines3 [Raffin et al., 2021] and add a simple sample-based value-incentivized term to its critic objective.

Critic update. With two critics { Q θ j } 2 j =1 and target networks { Q θ -j } 2 j =1 , the SAC target is

Here, r ( s, a ) denotes the one-step reward, and π denotes the current stochastic policy used by SAC for target evaluation (i.e., a ′ ∼ π ( · | s ′ ) ). Our modified critic objective uses minibatch sample averages (replacing population expectations) and reads

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here we use a single Monte Carlo sample 1 N ∑ N i =1 Q θ j ( s, a i ) to approximate V π f ( s ) = E a ∼ π ( ·| s ) [ Q f ( s, a )] . We found that setting N = 1 , i.e., using a single policy sample is good enough. We use a minibatch B of size 256 sampled uniformly from a replay buffer of size 10 6 . The buffer stores the historical data: during the first 100 steps we act uniformly at random (warm-up). After warm-up, the current policy selects one action at each step, and the resulting ( s, a, r ( s, a ) , s ′ ) is appended to the replay buffer. We optimize the critic with Adam (learning rate 3 × 10 -4 ), perform one gradient step, and update target networks every step via Polyak averaging with τ polyak = 0 . 005 . Training starts after collecting 100 steps. The entropy coefficient is tuned automatically by optimizing a learnable log-temperature to match a target entropy.

Policy update. The actor is updated with the standard SAC loss

<!-- formula-not-decoded -->

Figure 1: Ant-v4 with 1 /α ∈ { 0 , 2000 } . Shaded area indicates standard deviation across 3 seeds.

<!-- image -->

Figure 2: Walker2d-v4 with 1 /α ∈ { 0 , 1000 } . Shaded area indicates standard deviation across 3 seeds.

<!-- image -->

estimated with one reparameterized sample per state using the Tanh-squashed Gaussian policy; we optimize the actor with Adam (learning rate 3 × 10 -4 ) in lockstep with the critic. VAC modifies only the critic objective above, leaving the actor update identical to SAC.

Network architecture. Both critics are separate MLPs with two hidden layers of 256 ReLU units each ('twin Q'), and the actor is an MLP with the same hidden sizes producing a Gaussian policy with Tanh-squashed actions.

Results. We run both experiments for 10 6 iterations over 3 seeds. Figures 1 and 2 summarize performance. For each task, we plot the best return across the three seeds and the average return over seeds; shaded regions denote standard deviation. The VAC regularization improves sample efficiency compared to SAC.

## 5 Conclusion

In this paper, we develop a provably sample-efficient actor-critic method, called value-incentivized actor-critic (V AC), for online RL with a single easy-to-optimize objective function that avoids complex bilevel optimization in the presence of complex function approximation. We theoretically establish V AC's efficacy by proving it achieves ˜ O ( √ T ) -regret in both episodic and discounted settings. Our work suggests that a unified Lagrangian-based objective offers a promising direction for principled and practical online RL, allowing many venues for future developments. Further, we empirically validate V AC's performance on MuJoCo tasks. Follow-up efforts will focus on more empirical validation, and extending the algorithm design to multi-agent settings.

## Acknowledgments and Disclosure of Funding

This work of T. Yang and Y. Chi is supported in part by the grants NSF DMS-2134080, CCF2106778, and ONR N00014-19-1-2404. T. Yang is also graciously supported by the CMU Wei

Shen and Xuehong Zhang Presidential Fellowship. The work of B. Dai is supported in part by the grants NSF ECCS-2401391, IIS-2403240, and ONR N00014-25-1-2173.

## References

- Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems , 24, 2011.
- A. Agarwal and T. Zhang. Non-linear reinforcement learning in large action spaces: Structural conditions and sample-efficiency of posterior sampling. In Conference on Learning Theory , pages 2776-2814. PMLR, 2022.
- A. Agarwal, Y. Jin, and T. Zhang. Vo q l: Towards optimal regret in model-free RL with nonlinear function approximation. In The Thirty Sixth Annual Conference on Learning Theory , pages 9871063. PMLR, 2023.
- S. Agrawal and R. Jia. Posterior sampling for reinforcement learning: worst-case regret bounds. arXiv preprint arXiv:1705.07041 , 2017.
- P. Auer, T. Jaksch, and R. Ortner. Near-optimal regret bounds for reinforcement learning. Advances in neural information processing systems , 21, 2008.
- A. Ayoub, Z. Jia, C. Szepesvari, M. Wang, and L. Yang. Model-based reinforcement learning with value-targeted regression. In International Conference on Machine Learning , pages 463-474. PMLR, 2020.
- L. Baird. Residual algorithms: Reinforcement learning with function approximation. In Proceedings of the twelfth international conference on machine learning , pages 30-37, 1995.
- J. Bas-Serrano, S. Curi, A. Krause, and G. Neu. Logistic q-learning. In International conference on artificial intelligence and statistics , pages 3610-3618. PMLR, 2021.
- A. Beck. First-order methods in optimization . SIAM, 2017.
- S. Cen, J. Mei, K. Goshvadi, H. Dai, T. Yang, S. Yang, D. Schuurmans, Y. Chi, and B. Dai. Valueincentivized preference optimization: A unified approach to online and offline RLHF. arXiv preprint arXiv:2405.19320 , 2024.
- F. Chen, S. Mei, and Y. Bai. Unified algorithms for RL with decision-estimation coefficients: PAC, reward-free, preference-based learning and beyond. The Annals of Statistics , 53(1):426-456, 2025.
- B. Dai, N. He, Y. Pan, B. Boots, and L. Song. Learning from conditional distributions via dual embeddings. In Artificial Intelligence and Statistics , pages 1458-1467. PMLR, 2017.
- B. Dai, A. Shaw, L. Li, L. Xiao, N. He, Z. Liu, J. Chen, and L. Song. SBEED: Convergent reinforcement learning with nonlinear function approximation. In International conference on machine learning , pages 1125-1134. PMLR, 2018.
- B. Dai, O. Nachum, Y. Chow, L. Li, C. Szepesvári, and D. Schuurmans. Coindice: Off-policy confidence interval estimation. Advances in neural information processing systems , 33:93989411, 2020.
- C. Dann, Y. Mansour, M. Mohri, A. Sekhari, and K. Sridharan. Guarantees for epsilon-greedy reinforcement learning with function approximation. In International conference on machine learning , pages 4666-4689. PMLR, 2022.
- D. P. De Farias and B. Van Roy. On constraint sampling in the linear programming approach to approximate dynamic programming. Mathematics of operations research , 29(3):462-478, 2004.
- S. Du, S. Kakade, J. Lee, S. Lovett, G. Mahajan, W. Sun, and R. Wang. Bilinear classes: A structural framework for provable generalization in rl. In International Conference on Machine Learning , pages 2826-2836. PMLR, 2021.

- B. L. Edelman, S. Goel, S. Kakade, and C. Zhang. Inductive biases and variable creation in selfattention mechanisms. In International Conference on Machine Learning , pages 5793-5831. PMLR, 2022.
- D. Foster, D. J. Foster, N. Golowich, and A. Rakhlin. On the complexity of multi-agent decision making: From learning in games to partial monitoring. In The Thirty Sixth Annual Conference on Learning Theory , pages 2678-2792. PMLR, 2023a.
- D. J. Foster, S. M. Kakade, J. Qian, and A. Rakhlin. The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487 , 2021.
- D. J. Foster, N. Golowich, and Y. Han. Tight guarantees for interactive decision making with the decision-estimation coefficient. In The Thirty Sixth Annual Conference on Learning Theory , pages 3969-4043. PMLR, 2023b.
- D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, et al. Deepseek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
- J. He, H. Zhao, D. Zhou, and Q. Gu. Nearly minimax optimal reinforcement learning for linear Markov decision processes. In International Conference on Machine Learning , pages 1279012822. PMLR, 2023.
7. Y.-H. Hung, P.-C. Hsieh, X. Liu, and P. Kumar. Reward-biased maximum likelihood estimation for linear stochastic bandits. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 7874-7882, 2021.
- C. Jin, Z. Yang, Z. Wang, and M. I. Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on learning theory , pages 2137-2143. PMLR, 2020.
- C. Jin, Q. Liu, and S. Miryoosefi. Bellman Eluder dimension: New rich classes of RL problems, and sample-efficient algorithms. Advances in neural information processing systems , 34:1340613418, 2021.
- I. Kostrikov, O. Nachum, and J. Tompson. Imitation learning via off-policy distribution matching. arXiv preprint arXiv:1912.05032 , 2019.
- P. Kumar and A. Becker. A new family of optimal adaptive controllers for markov chains. IEEE Transactions on Automatic Control , 27(1):137-146, 1982.
- C. Lakshminarayanan, S. Bhatnagar, and C. Szepesvári. A linearly relaxed approximate linear program for markov decision processes. IEEE Transactions on Automatic control , 63(4):1185-1191, 2017.
- T. Lattimore and C. Szepesvári. Bandit algorithms . Cambridge University Press, 2020.
- J. Lee, W. Jeon, B. Lee, J. Pineau, and K.-E. Kim. Optidice: Offline policy optimization via stationary distribution correction estimation. In International Conference on Machine Learning , pages 6120-6130. PMLR, 2021.
- Y. Li and Z. Luo. Prior-dependent analysis of posterior sampling reinforcement learning with function approximation. In International Conference on Artificial Intelligence and Statistics , pages 559-567. PMLR, 2024.
- Q. Liu, G. Weisz, A. György, C. Jin, and C. Szepesvári. Optimistic natural policy gradient: a simple efficient policy optimization framework for online RL. Advances in Neural Information Processing Systems , 36:3560-3577, 2023.
- X. Liu, P.-C. Hsieh, Y. H. Hung, A. Bhattacharya, and P. Kumar. Exploration through reward biasing: Reward-biased maximum likelihood estimation for stochastic multi-armed bandits. In International Conference on Machine Learning , pages 6248-6258. PMLR, 2020.
- Z. Liu, M. Lu, W. Xiong, H. Zhong, H. Hu, S. Zhang, S. Zheng, Z. Yang, and Z. Wang. Maximize to explore: One objective function fusing estimation, planning, and exploration. Advances in Neural Information Processing Systems , 36, 2024.

- R. Lu, G. Huang, and S. S. Du. On the power of multitask representation learning in linear mdp. arXiv preprint arXiv:2106.08053 , 2021.
- Y. Ma, A. Shen, D. Jayaraman, and O. Bastani. Versatile offline imitation from observations and examples via regularized state-occupancy matching. In International Conference on Machine Learning , pages 14639-14663. PMLR, 2022.
- A. Mete, R. Singh, X. Liu, and P. Kumar. Reward biased maximum likelihood estimation for reinforcement learning. In Learning for Dynamics and Control , pages 815-827. PMLR, 2021.
- V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, and G. Ostrovski. Human-level control through deep reinforcement learning. Nature , 518(7540):529-533, 2015.
- A. Moulin, G. Neu, and L. Viano. Optimistically optimistic exploration for provably efficient infinitehorizon reinforcement and imitation learning. arXiv preprint arXiv:2502.13900 , 2025.
- O. Nachum and B. Dai. Reinforcement learning via fenchel-rockafellar duality. arXiv preprint arXiv:2001.01866 , 2020.
- O. Nachum, Y. Chow, B. Dai, and L. Li. Dualdice: Behavior-agnostic estimation of discounted stationary distribution corrections. Advances in neural information processing systems , 32, 2019a.
- O. Nachum, B. Dai, I. Kostrikov, Y. Chow, L. Li, and D. Schuurmans. Algaedice: Policy gradient from arbitrary experience. arXiv preprint arXiv:1912.02074 , 2019b.
- G. Neu, A. Jonsson, and V. Gómez. A unified view of entropy-regularized markov decision processes. arXiv preprint arXiv:1705.07798 , 2017.

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 , 2023.

- I. Osband and B. Van Roy. Why is posterior sampling better than optimism for reinforcement learning? In International conference on machine learning , pages 2701-2710. PMLR, 2017.
- L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems , 35:27730-27744, 2022.
- M. Papini, A. Tirinzoni, A. Pacchiano, M. Restelli, A. Lazaric, and M. Pirotta. Reinforcement learning in linear MDPs: Constant regret and representation selection. Advances in Neural Information Processing Systems , 34:16371-16383, 2021.
- M. L. Puterman. Markov decision processes: discrete stochastic dynamic programming . John Wiley &amp;Sons, 2014.
- A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann. Stablebaselines3: Reliable reinforcement learning implementations. GitHub repository, 2021. URL https://github.com/DLR-RM/stable-baselines3 . DLR-RM/stable-baselines3.
- T. Ren, T. Zhang, L. Lee, J. E. Gonzalez, D. Schuurmans, and B. Dai. Spectral decomposition representation for reinforcement learning. arXiv preprint arXiv:2208.09515 , 2022.
- D. J. Russo, B. Van Roy, A. Kazerouni, I. Osband, Z. Wen, et al. A tutorial on thompson sampling. Foundations and Trends® in Machine Learning , 11(1):1-96, 2018.
- H. Sikchi, Q. Zheng, A. Zhang, and S. Niekum. Dual rl: Unification and new methods for reinforcement and imitation learning. arXiv preprint arXiv:2302.08560 , 2023.
- D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, et al. Mastering the game of Go without human knowledge. Nature , 550 (7676):354-359, 2017.
- W. Sun, N. Jiang, A. Krishnamurthy, A. Agarwal, and J. Langford. Model-based RL in contextual decision processes: PAC bounds and exponential improvements over model-free approaches. In Conference on learning theory , pages 2898-2933. PMLR, 2019.

- R. S. Sutton, A. G. Barto, et al. Reinforcement learning: An introduction , volume 1. MIT press Cambridge, 1998.
- K. Tan, W. Fan, and Y. Wei. Actor-critics can achieve optimal sample efficiency. arXiv preprint arXiv:2505.03710 , 2025.
- G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 , 2023.
- E. Todorov, T. Erez, and Y. Tassa. MuJoCo: A physics engine for model-based control. 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems , pages 5026-5033, 2012.
- M. Uehara, J. Huang, and N. Jiang. Minimax weight and q-function learning for off-policy evaluation. In International Conference on Machine Learning , pages 9659-9668. PMLR, 2020.
- M. Wang. Primal-dual π learning: Sample complexity and sublinear run time for ergodic markov decision problems. arXiv preprint arXiv:1710.06100 , 2017.
- Z. Wang, D. Zhou, J. C. Lui, and W. Sun. Model-based RL as a minimalist approach to horizon-free and second-order bounds. In The Thirteenth International Conference on Learning Representations , 2025.
- T. Xie, D. J. Foster, Y . Bai, N. Jiang, and S. M. Kakade. The role of coverage in online reinforcement learning. In The Eleventh International Conference on Learning Representations , 2023.
- T. Xie, D. J. Foster, A. Krishnamurthy, C. Rosset, A. Awadallah, and A. Rakhlin. Exploratory preference optimization: Harnessing implicit Q ⋆ -approximation for sample-efficient RLHF. arXiv preprint arXiv:2405.21046 , 2024.
- N. Xiong, Z. Liu, Z. Wang, and Z. Yang. Sample-efficient multi-agent RL: An optimization perspective. In The Twelfth International Conference on Learning Representations , 2024.
- L. Yang and M. Wang. Sample-optimal parametric Q-learning using linearly additive features. In International Conference on Machine Learning , pages 6995-7004, 2019.
- M. Yang, O. Nachum, B. Dai, L. Li, and D. Schuurmans. Off-policy evaluation via the regularized lagrangian. Advances in Neural Information Processing Systems , 33:6551-6561, 2020.
- T. Yang, S. Cen, Y. Wei, Y. Chen, and Y. Chi. Federated natural policy gradient and actor critic methods for multi-task reinforcement learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- T. Yang, B. Dai, L. Xiao, and Y. Chi. Incentivize without bonus: Provably efficient model-based online multi-agent RL for markov games. arXiv preprint arXiv:2502.09780 , 2025.
- R. Yuan, S. S. Du, R. M. Gower, A. Lazaric, and L. Xiao. Linear convergence of natural policy gradient methods with log-linear policies. In International Conference on Learning Representations , 2023.
- A. Zanette, D. Brandfonbrener, E. Brunskill, M. Pirotta, and A. Lazaric. Frequentist regret bounds for randomized least-squares value iteration. In International Conference on Artificial Intelligence and Statistics , pages 1954-1964. PMLR, 2020.
- S. Zhang, D. Yu, H. Sharma, H. Zhong, Z. Liu, Z. Yang, S. Wang, H. Hassan, and Z. Wang. Selfexploring language models: Active preference elicitation for online alignment. arXiv preprint arXiv:2405.19332 , 2024.
- T. Zhang. Feel-good Thompson sampling for contextual bandits and reinforcement learning. SIAM Journal on Mathematics of Data Science , 4(2):834-857, 2022.
- H. Zhong, W. Xiong, S. Zheng, L. Wang, Z. Wang, Z. Yang, and T. Zhang. GEC: A unified framework for interactive decision making in MDP, POMDP, and beyond. arXiv preprint arXiv:2211.01962 , 2022.

- D. Zhou, J. He, and Q. Gu. Provably efficient reinforcement learning for discounted MDPs with feature mapping. In International Conference on Machine Learning , pages 12793-12802. PMLR, 2021.
- Z. Zhu, K. Lin, B. Dai, and J. Zhou. Off-policy imitation learning from observations. Advances in neural information processing systems , 33:12402-12413, 2020.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims made in the abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations of the work.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best

judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We provide the full set of assumptions and a complete (and correct) proof.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We provide the full set of information needed to reproduce the main experimental results.

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

Answer: [No]

Justification: We do not provide open access to the data and code since the experiments are simple and are not central to the contribution.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We provide all the details necessary to understand the results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We run experiments over 3 seeds and report the standard deviation of the returns.

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

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: The experiments are simple and can be run on a single CPU.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We conform to the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This is a foundational research paper and does not have any societal impact.

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

Justification: This is a foundational research paper and does not have any societal impact.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We properly credit and mention the license and terms of use.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: We do not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: We have no crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: We have no crowdsourcing or research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We do not use LLMs in the core methods.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## A Technical Lemmas

We provide some technical lemmas that will be used in our proofs.

Lemma 2 (Freedman's inequality, Lemma D.2 in Liu et al. [2024]) . Let { X t } t ⩽ T be a real-valued martingale difference sequence adapted to filtration {F t } t ⩽ T . If | X t | ⩽ R almost surely, then for any η ∈ (0 , 1 /R ) it holds that with probability at least 1 -δ ,

Lemma 3 (Covering number of ℓ 2 ball, Lemma D.5 in Jin et al. [2020]) . For any ϵ &gt; 0 and d ∈ N + , the ϵ -covering number of the ℓ 2 ball of radius R in R d is bounded by (1 + 2 R/ϵ ) d .

<!-- formula-not-decoded -->

Lemma 4 (Lemma 11 in Abbasi-Yadkori et al. [2011]) . Let { x s } s ∈ [ T ] be a sequence of vectors with x s ∈ V for some Hilbert space V . Let Λ 0 be a positive definite matrix and define Λ t = Λ 0 + ∑ t s =1 x s x ⊤ s . Then it holds that

Lemma 5 (Lemma F.3 in Du et al. [2021]) . Let X ⊂ R d and sup x ∈X ‖ x ‖ 2 ⩽ B X . Then for any n ∈ N + , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 6 (Corollary A.7 in Edelman et al. [2022]) . Define the softmax function as softmax ( · ) : R d → ∆ d by softmax ( x ) i = exp( x i ) ∑ d j =1 exp( x j ) for all i ∈ [ d ] and x ∈ R d . Then for any x, y ∈ R d , we have

<!-- formula-not-decoded -->

## B Proofs for Episodic MDPs

## B.1 Proof of Theorem 1

Notation and preparation. For notation simplicity, we let f ⋆ := Q ⋆ be the optimal Q-function. We let Π := ∆( A ) S denote the whole policy space. We have P h ⊂ Π for all h ∈ [ H ] . We also define the transition tuples

<!-- formula-not-decoded -->

Given any policy profile π = { π h } h ∈ [ H ] and f = { f h : S × A ↦→ R } , we define P π h f as

<!-- formula-not-decoded -->

and let P π f := { P π h f } h ∈ [ H ] . Let be the parameter space of Q h and P h , respectively for all h ∈ [ H ] . We also define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We'll repeatedly use the following lemma, which guarantees that under Assumption 1, the optimal Q-function Q ⋆ is in Q , and P π f ∈ Q for any f ∈ Q and π ∈ Π H . Similar results can be found in the literature, e.g., Jin et al. [2020]. For completeness, we include the proof of Lemma 7 in Appendix B.2.1.

Lemma 7 (Linear MDP ⇒ Bellman completeness + realizability) . Under Assumption 1, we have

- (realizability) Q ⋆ ∈ Q ;
- (Bellman completeness) ∀ π ∈ Π and f ∈ Q , P π f ∈ Q .

We also use the following lemma, which bounds the difference between the optimal value function V ⋆ and max π ∈P V π -the optimal value over the policy class P , where we let

<!-- formula-not-decoded -->

and ˜ π ⋆ = { ˜ π ⋆ h } h ∈ [ H ] be the optimal policy within the policy class P . The proof of Lemma 8 is deferred to Appendix B.2.2. Lemma 8 (model error with log-linear policies) . Under Assumptions 1-3, we have

<!-- formula-not-decoded -->

where B is defined in Assumption 3.

Main proof. We first decompose the regret (cf. (5)) as follows:

<!-- formula-not-decoded -->

Step 1: bounding term (i). The linear MDP assumption guarantees that Q ⋆ ∈ Q by Lemma 7, and by definition (28), π ⋆ is in P . Thus by our update rule (17), we have which gives

<!-- formula-not-decoded -->

Invoking Lemma 8, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus to bound (i), it suffices to bound L t ( f ⋆ , ˜ π ⋆ ) - L t ( f t , π t ) for each t ∈ [ T ] . To introduce our lemmas, we define ℓ h : Q h ×S × A × Π ↦→ R for all h ∈ [ H ] as

We give the following lemma that bounds (i), whose proof is given in Appendix B.2.3.

Lemma 9. Suppose Assumptions 1-3 hold. For any δ ∈ (0 , 1) , with probability at least 1 -δ , for any t ∈ [ T ] , we have

<!-- formula-not-decoded -->

for some absolute constant C &gt; 0 . Here, d π i ρ,h is the state-action visitation distribution induced by policy π i at step h .

By (31) and Lemma 9, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 2: bounding term (ii). For any λ &gt; 0 , we define

<!-- formula-not-decoded -->

We use the following lemma to bound (ii), whose proof is in Appendix B.2.4.

Lemma 10. Under Assumption 1, for any η &gt; 0 , we have

<!-- formula-not-decoded -->

By Lemma 10, we have

<!-- formula-not-decoded -->

Step 3: combining (i) and (ii). Substituting (34) and (36) into (30), and letting η = α 2 , we have

<!-- formula-not-decoded -->

Setting λ = 1 √ T , α = ( 1 H 2 T log(log |A| T/δ ) log ( 1 + T 3 / 2 d )) 1 / 2 , and B = T log |A| dH in the above bound, we have with probability at least 1 -δ ,

<!-- formula-not-decoded -->

for some absolute constant C ′ &gt; 0 . This completes the proof of Theorem 1.

## B.2 Proof of key lemmas

## B.2.1 Proof of Lemma 7

Assumption 1 guarantees that

<!-- formula-not-decoded -->

where ν ⋆ h ∈ R d satisfies

Moreover, for any f ∈ Q , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ζ h ∈ R d satisfies

In addition, we have

<!-- formula-not-decoded -->

Thus P π f ∈ Q .

<!-- formula-not-decoded -->

## B.2.2 Proof of Lemma 8

From Lemma 7, it is known that for all h ∈ [ H ] , there exists ν ⋆ h ∈ Θ h such that

Let where B is defined in Assumption 3. It follows that π h ∈ P h , and for all s ∈ S , π h ( ·| s ) is the solution to the following optimization problem [Beck, 2017, Example 3.71]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, H ( · ) is the entropy function satisfying

The optimality of π h for (41), together with (42), implies

<!-- formula-not-decoded -->

which further indicates

<!-- formula-not-decoded -->

The desired bound (29) follows from the above inequality and the fact that V ⋆ h ( s ) = max a ∈A Q ⋆ ( s, a ) ⩾ V π ′ f ⋆ ,h ( s ) for any policy profile π ′ , s ∈ S and h ∈ [ H ] .

## B.2.3 Proof of Lemma 9

We bound the two terms L t ( f ⋆ , ˜ π ⋆ ) and -L t ( f t , π t ) on the left-hand side of (33) separately. Step 1: bounding -L t ( f t , π t ) . Given f, f ′ ∈ Q , data tuple ξ = ( s, a, s ′ ) and policy profile π = { π h } H h =1 ∈ Π H , we define the random variable

<!-- formula-not-decoded -->

where a ′ ∼ π h +1 ( ·| s ′ ) . Then we have (recall we define P π f in (25))

<!-- formula-not-decoded -->

which indicates that for any f, f ′ ∈ Q , ξ and π ,

<!-- formula-not-decoded -->

For any f ∈ Q , π ∈ Π H and t ∈ [ T ] , we define X t f,π,h as where ξ t,h := ( s t,h , a t,h , s t,h +1 ) is the transition tuple collected at time t and step h . Then we have for any f ∈ Q :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality uses the fact that P π f ∈ Q , which is guaranteed by Lemma 7. Here, we define

<!-- formula-not-decoded -->

Therefore, to upper bound -L t ( f t , π t ) = -∑ H h =1 L t,h ( f t , π t ) , it suffices to bound -∑ t -1 i =1 X i f t ,π t ,h for all h ∈ [ H ] . In what follows, we use Freedman's inequality (Lemma 2) and a covering number argument similar to that in Yang et al. [2025] to give the desired bound.

Step 1.1: building the covering argument. We start with some basic preparation on the covering argument. For any X ⊂ R d , let N ( X , ϵ, ‖·‖ ) be the ϵ -covering number of X with respect to the norm ‖·‖ . Assumption 2 and Assumption 3 guarantee that (cf. (26)) Θ h ⊂ B d 2 ( H √ d ) and Ω = B d 2 ( BH √ d ) for all h , where we use B d 2 ( R ) to denote the ℓ 2 ball of radius R in R d . Thus by Lemma 3 we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any ϵ &gt; 0 . This suggests that for any ϵ &gt; 0 , there exists an ϵ -net Θ h,ϵ ⊂ Θ h and an ϵ -net Ω ϵ ⊂ Ω such that

<!-- formula-not-decoded -->

For any f h = f θ,h ∈ Q h with θ h ∈ Θ h , there exists θ h,ϵ ∈ Θ h,ϵ such that ‖ θ h -θ h,ϵ ‖ 2 ⩽ ϵ , and we let f h,ϵ := f θ h,ϵ and define

<!-- formula-not-decoded -->

In addition, for any π h ∈ P h , there exists ω h ∈ Ω and ω h,ϵ ∈ Ω ϵ such that ‖ ω h -ω h,ϵ ‖ 2 ⩽ ϵ , such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We claim that for any f ∈ Q and π ∈ P , there exists f ϵ ∈ Q ϵ and π ϵ ∈ P ϵ such that

The proof of (55) is deferred to the end of this proof.

<!-- formula-not-decoded -->

Step 1.2: bounding the mean and variance. Assumption 1 ensures X t f,π h is bounded:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now bound E s t,h +1 ∼ P h ( ·| s t,h ,a t,h ) [ X t f,π,h ] . Notice that where the expectation of the last term satisfies

<!-- formula-not-decoded -->

Combining (48), (57) and (58), we have

<!-- formula-not-decoded -->

Now we consider the martingale variance term. Define the filtration F t := σ ( D t ) (the σ -algebra generated by the dataset D t := ∪ H h =1 D t,h ). We have

<!-- formula-not-decoded -->

where we define d π ρ,h to be the state-action visitation distribution at step h and time t under policy profile π and initial state distribution ρ , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, we have where the first equality follows from (45) and (46), and the second inequality follows from Jenson's inequality.

Step 1.3: applying Freedman's inequality and finishing up. By Lemma 2, (56), (60) and (62), and noticing that ℓ h ( f, s, a, π ) is only related to f h , f h +1 and π h +1 , we have with probability at least 1 -δ , for all t ∈ [ T ] , h ∈ [ H ] , f ϵ ∈ Q ϵ and π ϵ ∈ P ϵ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C 1 , C ′ 1 &gt; 0 are absolute constants. From (63) we deduce that for all t ∈ [ T ] , f ϵ ∈ Q ϵ , and π ϵ ∈ P ϵ , we have with probability at least 1 -δ ,

Note that for any t ∈ [ T ] and h ∈ [ H ] , there exist θ t,h ∈ Θ h and ω t,h ∈ Ω such that f t,h = f θ t,h ∈ Q h and π t,h = π ω t,h ∈ P h . We can choose θ t,h,ϵ ∈ Θ h,ϵ and ω t,h,ϵ ∈ Ω ϵ such that ‖ θ t,h -θ t,h,ϵ ‖ 2 ⩽ ϵ and ‖ ω t,h -ω t,h,ϵ ‖ 2 ⩽ ϵ . We let f t,ϵ := { f θ t,h,ϵ } h ∈ [ H ] ∈ Q ϵ and π t,ϵ := { π ω t,h,ϵ } h ∈ [ H ] ∈ P ϵ . Then by (64) we have for all t ∈ [ T ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last line follows from (55) and (59).

<!-- formula-not-decoded -->

Note that for any tuple ξ = ( s, a, s ′ ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last line follows from Bellman's optimality equation:

Note that by Lemma 8, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging the above inequality into (67) and (68) leads to

The above bounds (70) and (50) imply that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we also use the definitions of Y t f,h (c.f. (66)) and ˜ f ⋆ (c.f. (66)). Thus to bound L t ( f ⋆ , ˜ π ⋆ ) , below we bound the sum ∑ t -1 i =1 Y i f,h for any f ∈ Q , t ∈ [ T ] and h ∈ [ H ] by applying Freedman's inequality and the covering argument. By a similar argument as earlier, we have for any f ∈ Q , there exists f ϵ ∈ Q ϵ such that whose proof is deferred to the end. We next compute the key quantities required to apply Freedman's inequality.

- Repeating a similar derivation of (59), we have

which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- where a ′ ∼ ˜ π ⋆ h +1 ( ·| s t,h +1 ) and the second inequality uses Jenson's inequality. · Last but not least, it's easy to verify that

Invoking Lemma 2, and setting η as we have with probability at least 1 -δ , for all f ϵ ∈ Q ϵ , t ∈ [ T ] , h ∈ [ H ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Reorganizing the above inequality, we have for any f ϵ ∈ Q ϵ , t ∈ [ T ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last line makes use of the fact that -x 2 + bx ⩽ b 2 / 4 . Combining (79) and (72), we have with probability at least 1 -δ , for any t ∈ [ T ] and f ∈ Q ,

<!-- formula-not-decoded -->

where C 2 &gt; 0 is an absolute constant. Plugging this into (71), we have

Step 3: combining the two bounds. Combining (65) and (81), we have for any t ∈ [ T ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some absolute constant C &gt; 0 . Letting ϵ = 1 T , we obtain the desired result.

Proof of (55) and (72) . By Assumption 1, we have

∀ ( s, a ) ∈ S × A : | f h ( s, a ) -f h,ϵ ( s, a ) | ⩽ ‖ ϕ h ( s, a ) ‖ 2 ‖ θ h -θ h,ϵ ‖ 2 ⩽ ϵ, (83) and thus for any f ∈ Q and π ∈ P , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the last inequality we use (83).

Similarly, by Lemma 6, we have

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

where the first inequality follows from Hölder's inequality and the fact that

Combining (84) and (86), we have the desired bound in (55):

<!-- formula-not-decoded -->

∣ ∣ X t f ϵ ,π ϵ ,h -X t f,π,h ∣ ∣ ⩽ ∣ ∣ X t f ϵ ,π ϵ ,h -X t f ϵ ,π,h ∣ ∣ + ∣ ∣ X t f ϵ ,π,h -X t f,π,h ∣ ∣ ⩽ 16 Hϵ +8 H 2 ϵ = 24 H 2 ϵ. Similarly, we have (72) follows by

<!-- formula-not-decoded -->

where the last inequality uses (83).

## B.2.4 Proof of Lemma 10

First note that for any policy profile π ∈ Π H , any f ∈ Q and h ∈ [ H ] , we have (note that V f,H +1 = 0 )

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The above two expressions (87) and (88) together give that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Assumption 1, for any f ∈ Q , there exists θ f ∈ Θ such that f h ( s, a ) = 〈 θ f,h , ϕ h ( s, a ) 〉 . Thus we have where W h ( f, π ) satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

under Assumption 1. We define

Then we have

<!-- formula-not-decoded -->

For all t ∈ [ T ] and h ∈ [ H ] , we define

<!-- formula-not-decoded -->

where I d is the d × d identity matrix. Then by Lemma 4, we have

<!-- formula-not-decoded -->

Further, we could use Lemma 5 to bound the last term in (97), and obtain

<!-- formula-not-decoded -->

where in the last line, we use the definition of d ( λ ) (c.f. (35)) and the fact that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is ensured by Assumption 1.

Observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where 1 {·} is the indicator function.

To give the desired bound, we will bound (a) and (b) separately.

Bounding (a). We have for any λ &gt; 0 ,

<!-- formula-not-decoded -->

Note that ‖ W h ( f t , π t ) ‖ Λ t,h ( λ ) can be bounded as follows:

<!-- formula-not-decoded -->

where we use (93), (96) and the fact that √ a + b ⩽ √ a + √ b for any a, b ⩾ 0 .

The above two bounds (101) and (102) together give

<!-- formula-not-decoded -->

where in the second inequality we use Cauchy-Schwarz inequality and the fact that

<!-- formula-not-decoded -->

The first term (a-i) in (103) could be bounded as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To bound (a-ii), note that for any π, π ′ ∈ Π H , we have

where the inequality follows from Jenson's inequality, and recall ℓ h ( f, s, a, π ) is defined in (32). Combining (106) and (98), we could bound (a-ii) in (103) as follows:

<!-- formula-not-decoded -->

Plugging (105) and (107) into (103), we have

<!-- formula-not-decoded -->

Bounding (b). By Assumption 1 and (95), we have

<!-- formula-not-decoded -->

Combining the above inequality with (98), we have

<!-- formula-not-decoded -->

Combining (a) and (b). Plugging (108) and (110) into (100), we have

<!-- formula-not-decoded -->

The first term in the right hand side of (111) could be bounded as and the second term in the right hand side of (111) could be bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any η &gt; 0 , where in both (112) and (113), we use the fact that √ ab ⩽ a + b 2 for any a, b ⩾ 0 .

Substituting (112) and (113) into (111) and reorganizing the terms, we have

<!-- formula-not-decoded -->

This gives the desired result.

## B.3 Extension to general function approximation

We now extend the analysis to finite-horizon MDPs with general function approximation. We first state our assumptions in this section.

- (realizability) Q ⋆ ∈ Q .

Assumption 4 (Q-function class) . The Q-function class Q = ∏ H h =1 Q h satisfies

- (Bellman completeness) ∀ π ∈ P and f ∈ Q , P π f ∈ Q .
- (boundedness) ∀ f h ∈ Q h , ‖ f h ‖ ∞ ⩽ H +1 -h .

Assumption 4 is a standard condition in prior literature involving general function approximation [Liu et al., 2024, Assumption 3.1], [Jin et al., 2021, Assumption 2.1]. In particular, Assumption 4 holds under linear MDPs (c.f. Assumption 1), as established inLemma 7. Under Assumption 4, we set the policy class P as follows.

Assumption 5 (Policy class) . The policy class = H h

<!-- formula-not-decoded -->

with some constant B &gt; 0 .

<!-- formula-not-decoded -->

Moreover, drawing upon the work of Zhong et al. [2022], Liu et al. [2024], we require the MDP to feature a low generalized Eluder coefficient (GEC). This characteristic is essential for ensuring that the minimization of in-sample prediction error, based on historical data, also effectively limits out-of-sample prediction error.

Assumption 6 (Generalized Eluder coefficient, Assumption 4.2 in Liu et al. [2024]) . Given any λ &gt; 0 , there exists d ( λ ) ∈ R + such that for any sequence { f t } T t =1 ⊂ Q , { π t } T t =1 ⊂ P , we have

<!-- formula-not-decoded -->

For each ˜ λ &gt; 0 , we denote the smallest ˜ d ( ˜ λ ) ∈ R + that makes (116) hold as d GEC ( ˜ λ ) . From Lemma 10 we can see that under linear MDPs (c.f. Assumption 1), Assumption 6 holds with d GEC ( ˜ λ ) ≲ Hd ( ˜ λ dH ) , where d ( · ) is defined in (35). Moreover, as demonstrated by Zhong et al. [2022], RL problems characterized by a low Generalized Eluder Coefficient (GEC) constitute a significantly broad category, such as linear MDPs [Yang and Wang, 2019, Jin et al., 2020], linear mixture MDPs [Ayoub et al., 2020], MDPs of bilinear classes [Du et al., 2021], MDPs with low witness rank [Sun et al., 2019], and MDPs with low Bellman Eluder dimension [Jin et al., 2021], see Zhong et al. [2022] for a more detailed discussion.

We let N ( Q h , ϵ, ‖·‖ ∞ ) denote the ϵ -covering number of Q h w.r.t. the ℓ ∞ norm, and assume the ϵ -nets Q h,ϵ are finite.

Assumption 7 (Finite ϵ -nets) . N ( ϵ ) := max h ∈ [ H ] N ( Q h , ϵ, ‖·‖ ∞ ) &lt; + ∞ .

The following theorem gives the regret bound under the above more general assumptions.

Theorem 11 (Regret under general function approximation) . Suppose Assumptions 4, 5, 6, 7 hold. We let B = T log |A| H in Assumption 5, and set

Then for any δ ∈ (0 , 1) , with probability at least 1 -δ , the regret of Algorithm 1 satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under linear MDPs, (118) reduces to (22) given in Theorem 1. Besides, this bound also matches (is slightly tighter than) the bound given in Corollary 5.2 of Liu et al. [2024] under similar assumptions.

## B.4 Proof of Theorem 11

In this proof, we use the same notations as in the proof of Theorem 1 in Appendix B.1. First, we define

<!-- formula-not-decoded -->

and ˜ π ⋆ = { ˜ π ⋆ h } h ∈ [ H ] . Using the same argument as Lemma 8, we have the following lemma.

Lemma 12 (model error with log linear policies) . Under Assumption 4 and 5, we have

<!-- formula-not-decoded -->

where B is defined in Assumption 5.

We bound the two terms in the regret decomposition (30) separately.

Bounding term (i). Following the same analysis as (31), we have

<!-- formula-not-decoded -->

Lemma 13. Suppose Assumption 4, 5, 7 hold. For any δ ∈ (0 , 1) , with probability at least 1 -δ , for any t ∈ [ T ] , we have

It boils down to bound L t ( f ⋆ , ˜ π ⋆ ) -L t ( f t , π t ) for each t ∈ [ T ] . Recall the definition of ℓ h ( f, s, a, π ) in (32), we give the following lemma, whose proof is deferred to Appendix B.2.3.

<!-- formula-not-decoded -->

for some absolute constant C &gt; 0 .

By (121) and Lemma 13, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining (i) and (ii). Substituting (123) and (124) into (30), and letting η = α 2 , we have

Regret ( T ) ⩽ αCTH 3 log ( N ( ϵ/B ) HT δ ) + ( CH 2 αT +1 ) T log |A| B + 2 d GEC ( ˜ λ ) α + √ d GEC ( ˜ λ ) HT + ˜ λHT. Setting

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

in the above bound, we have with probability at least 1 -δ , for some absolute constant C ′ &gt; 0 . This completes the proof of Theorem 11.

## B.4.1 Proof of Lemma 13

The proof is similar to the proof of Lemma 9 given in Appendix B.2.3. We use the same notations as in Appendix B.2.3, and also bound the two terms L t ( f ⋆ , ˜ π ⋆ ) and -L t ( f t , π t ) in the left-hand side of (122) separately.

<!-- formula-not-decoded -->

Bounding -L t ( f t , π t ) . Same as in (48), here we also define then for any f ∈ Q :

<!-- formula-not-decoded -->

where we use the fact that P π f ∈ Q guaranteed by Assumption 4. Therefore, to upper bound -L t ( f t , π t ) = -∑ H h =1 L t,h ( f t , π t ) , it suffices to bound -∑ t -1 i =1 X i f t ,π t ,h for all h ∈ [ H ] .

<!-- formula-not-decoded -->

For all h ∈ [ H ] , there exists an ϵ -net Q h,ϵ of Q h w.r.t. the ℓ ∞ norm such that where the last relation is due to Assumption 4. Then for any f ∈ Q h , there exists f h,ϵ ∈ Q h,ϵ such that

<!-- formula-not-decoded -->

and thus for any f ∈ Q and π ∈ P , we have

<!-- formula-not-decoded -->

where in the last inequality we use (129) and the boundedness of f h and f h +1 assumed in Assumption 4.

In addition, there exists Q h,ϵ/B of Q h w.r.t. the ℓ ∞ norm such that

<!-- formula-not-decoded -->

We define then we have

<!-- formula-not-decoded -->

and by Assumption 5, for any π h ∈ P h , there exists Q h ∈ Q h,ϵ/B such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

There also exists Q h,ϵ/B ∈ Q h,ϵ/B such that

We let

Then by Lemma 6, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In other words, we have shown that P h,ϵ is an 2 ϵ -net of P h w.r.t. the ℓ 1 norm. Therefore, we have

<!-- formula-not-decoded -->

where the first inequality follows from Hölder's inequality and the fact that for all ( s, a ) ∈ S × A , f ∈ Q and π ∈ P , which is ensured by Assumption 4. Combining (130) and (138), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand, Assumption 4 ensures X t f,π h is bounded:

<!-- formula-not-decoded -->

Thus following the same argument as in Appendix B.2.3 that leads to (63), here we could obtain that for any δ ∈ (0 , 1) , with probability at least 1 -δ , for all t ∈ [ T ] , h ∈ [ H ] , f ϵ ∈ Q ϵ = ∏ H h =1 Q h,ϵ and π ϵ ∈ P ϵ = ∏ H h =1 P h,ϵ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C 1 , C ′ 1 &gt; 0 are absolute constants.

From (141) we deduce that for all t ∈ [ T ] , f ϵ ∈ Q ϵ , and π ϵ ∈ P ϵ , we have with probability at least 1 -δ ,

<!-- formula-not-decoded -->

By (137), for any t ∈ [ T ] and h ∈ [ H ] , we can choose f t,h,ϵ ∈ Q h,ϵ and π t,h,ϵ ∈ P h,ϵ such that

<!-- formula-not-decoded -->

Then by (142) we have for all t ∈ [ T ] ,

<!-- formula-not-decoded -->

where the last line follows from (139) and (59).

Bounding L t ( f ⋆ , π ⋆ ) . Same as in (66), for any f ∈ Q and t ∈ [ T ] , we define where we define

<!-- formula-not-decoded -->

Then following the same argument that leads to (79), setting η in Lemma 2 as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have with probability at least 1 -δ , for any f ϵ ∈ Q ϵ , t ∈ [ T ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last line makes use of the fact that -x 2 + bx ⩽ b 2 / 4 .

Moreoever, for any t ∈ [ T ] , h ∈ [ H ] , we have

<!-- formula-not-decoded -->

Combining (147) and (148), we have with probability at least 1 -δ , for any t ∈ [ T ] and f ∈ Q ,

<!-- formula-not-decoded -->

where C 2 &gt; 0 is an absolute constant.

By (71) we have

<!-- formula-not-decoded -->

Combining the two bounds. Combining (144) and (150), we have for any t ∈ [ T ] ,

<!-- formula-not-decoded -->

for some absolute constant C &gt; 0 . Letting ϵ = 1 T , we obtain the desired result.

## C Value-incentivized Actor-Critic Method for Discounted MDPs

Infinite-horizon MDPs. Let M = ( S , A , P, r, γ ) be an infinite-horizon discounted MDP, where S and A denote the state space and the action space, respectively, γ ∈ [0 , 1) denotes the discount factor, P : S × A ↦→ ∆( S ) is the transition kernel, and r : S × A ↦→ [0 , 1] is the reward function. A policy π : S ↦→ ∆( A ) specifies an action selection rule, where π ( a | s ) specifies the probability of taking action a in state s for each ( s, a ) ∈ S × A . For any given policy π , the value function, denoted by V π : S ↦→ R , is given as

<!-- formula-not-decoded -->

which measures the expected discounted cumulative reward starting from an initial state s 0 = s , where the randomness is over the trajectory generated following a t ∼ π ( ·| s t ) and the MDP dynamic s t +1 ∼ P ( ·| s t , a t ) . Given an initial state distribution s 0 ∼ ρ over S , we also define V π ( ρ ) :=

E s ∼ ρ [ V π ( s )] with slight abuse of notation. Similarly, the Q-function of policy π , denoted by Q π : S × A ↦→ R , is defined as

<!-- formula-not-decoded -->

which measures the expected discounted cumulative reward with an initial state s 0 = s and an initial action a 0 = a , with expectation taken over the randomness of the trajectory. It is known that there exists at least one optimal policy π ⋆ that maximizes the value function V π ( s ) for all states s ∈ S [Puterman, 2014], whose corresponding optimal value function and Q-function are denoted as V ⋆ and Q ⋆ , respectively. We also define the state-action visitation distribution d π ρ ∈ ∆( S ×A ) induced by policy π and initial state distribution ρ as

<!-- formula-not-decoded -->

## C.1 Algorithm development

Similar as (13), we start with an optimization problem:

<!-- formula-not-decoded -->

Writing the regularized Lagrangian system of (155) as

<!-- formula-not-decoded -->

Similar to the finite-horizon case, we use the reparameterization (10) which gives

<!-- formula-not-decoded -->

which is easier to optimize over both Q f and π . The population primal-dual optimization problem (157) prompts us to design the proposed algorithm, by computing the sample version of (157), see Algorithm 2, where we let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In Algorithm 2, at iteration t , given dataset D t -1 collected from the previous iterations, we define the loss function as follows:

We compute (160) in each iteration, which is the sample version of (157), and use the current policy π t to collect new data following the sampling procedure in Algorithm 3, which is also used in Yuan et al. [2023, Algorithm 3], Yang et al. [2024, Algorithm 5], and Yang et al. [2025, Algorithm 7]. Algorithm 3 has an expected iteration number E [ h +1] = 1 1 -γ , and it guarantees P ( s h = s, a h = a ) = d π ρ ( s, a ) [Yuan et al., 2023] for any ( s, a ) ∈ S × A and any policy π .

Algorithm 2 Value-incentivized Actor-Critic (VAC) for infinite-horizon discounted MDPs.

- 1: Input: regularization coefficient α &gt; 0 .
- 2: Initialization: dataset D 0 := ∅ .
- 4: Update Q-function estimation and policy:

<!-- formula-not-decoded -->

- 3: for t = 1 , · · · , T do
- 5: Data collection: sample ( s t , a t , s ′ t ) ← Sampler ( π t , ρ ) , and update the dataset D t = D t -1 ∪ { ( s t , a t , s ′ t ) } .
- 6: end for

Algorithm 3 Sampler for ( s, a ) ∼ d π ρ and s ′ ∼ P ( ·| s, a )

- 1: Input: policy π , initial state distribution ρ , player index n .
- 2: Initialization: s 0 ∼ ρ , a 0 ∼ π ( ·| s 0 ) , time step h = 0 , variable X ∼ Bernoulli( γ ).
- 3: while X = 1 do
- 4: Sample s h +1 ∼ P ( ·| s h , a h )
- 6: h ← h +1
- 5: Sample a h +1 ∼ π ( ·| s h +1 )
- 7: X ∼ Bernoulli( γ )
- 8: end while
- 9: Sample s h +1 ∼ P ( ·| s h , a h )
- 10: return ( s h , a h , s h +1 ) .

## C.2 Theoretical guarantees

Same as the finite-horizon setting, we assume the following d -dimensional linear MDP model.

Assumption 8 (infinite-horizon linear MDP) . There exists unknown vector ζ ∈ R d and unknown (signed) measures µ = ( µ (1) , · · · , µ ( d ) ) over S such that where ϕ : S × A → R d is a known feature map satisfying ‖ ϕ ( s, a ) ‖ 2 ⩽ 1 , and max {‖ ζ ‖ 2 , ‖ µ ( S ) ‖ 2 } ⩽ √ d , for all ( s, a, s ′ ) ∈ S × A × S .

<!-- formula-not-decoded -->

Similar as for the finite case, under Assumption 8, we only need to set the Q-function class to be linear and the policy class P to be the set of log-linear policies.

<!-- formula-not-decoded -->

Assumption 9 (linear Q -function class (infinite-horizon)) . The function class Q is defined as

Assumption 10 (log-linear policy class (infinite-horizon)) . The policy class P is defined as with some constant B &gt; 0 .

<!-- formula-not-decoded -->

We give the regret bound of Algorithm 2 in Theorem 14.

Theorem 14 (infinite-horizon) . Suppose Assumptions 8-10 hold. We let B = T log |A| (1 -γ ) d in Assumption 10 and set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then for any δ ∈ (0 , 1) , with probability at least 1 -δ , the regret of Algorithm 2 satisfies

Note that

<!-- formula-not-decoded -->

thus Theorem 14 guarantees that the iteration complexity to reach ϵ -accuracy w.r.t. value suboptimality for any ϵ &gt; 0 is ˜ O ( d 2 (1 -γ ) 4 ϵ 2 ) , and the total sample complexity is ˜ O ( d 2 (1 -γ ) 5 ϵ 2 ) .

## C.3 Proof of Theorem 14

Notation. For notation simplicity, we let f ⋆ := Q ⋆ be the optimal Q-function. We let Π := ∆( A ) S denote the set of all policies. We also define transition tuples

<!-- formula-not-decoded -->

Given any policy π and f : S × A → R , we define P π f as

<!-- formula-not-decoded -->

We let

<!-- formula-not-decoded -->

be the parameter space of Q and P , respectively.

We'll repeatedly use the following lemma, which is a standard consequence of linear MDP.

Lemma 15 (Linear MDP ⇒ Bellman completeness + realizability (infinite-horizon)) . Under Assumption 8, we have

- (realizability) Q ⋆ ∈ Q ;
- (Bellman completeness) ∀ π ∈ Π and f ∈ Q , P π f ∈ Q .

We'll also use the following lemma, which bounds the difference between the optimal value function V ⋆ ( ρ ) and max π ∈P V π ( ρ ) -the optimal value over the policy class P , where we let

<!-- formula-not-decoded -->

Lemma 16 (model error with log linear policies (infinite-horizon)) . Under Assumptions 8-10, we have

<!-- formula-not-decoded -->

where B is defined in Assumption 10.

We omit the proofs of the above two lemmas due to similarity to that of the finite-horizon setting.

Main proof of Theorem 14. Given the regret decomposition in (30), we will bound the two terms separately.

Step 1: bounding term (i). Similar to the argument in the finite-horizon setting, invoking Lemma 16, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus to bound (i), we only need to bound L t ( f ⋆ , ˜ π ⋆ ) - L t ( f t , π t ) for each t ∈ [ T ] . Define ℓ : Q×S ×A× Π as

Lemma 17. Suppose Assumption 8-10 hold. For any δ ∈ (0 , 1) , with probability at least 1 -δ , for any t ∈ [ T ] , we have

<!-- formula-not-decoded -->

for some absolute constant C &gt; 0 .

By (168) and Lemma 17, we have

<!-- formula-not-decoded -->

which gives

<!-- formula-not-decoded -->

Step 2: bounding term (ii). For any λ &gt; 0 , we define

<!-- formula-not-decoded -->

We use the following lemma to bound (ii), whose proof is deferred to Appendix C.4.2.

Lemma 18. Under Assumption 8, for any η &gt; 0 , we have

<!-- formula-not-decoded -->

By Lemma 18, we have

<!-- formula-not-decoded -->

Step 3: combining (i) and (ii). Substituting (171) and (174) into (30), and letting η = α 2 , we have

<!-- formula-not-decoded -->

Setting

<!-- formula-not-decoded -->

in the above bound, we have with probability at least 1 -δ ,

<!-- formula-not-decoded -->

for some absolute constant C ′ &gt; 0 . This completes the proof of Theorem 14.

## C.4 Proof of key lemmas

## C.4.1 Proof of Lemma 17

We bound the two terms L t ( f ⋆ , ˜ π ⋆ ) and -L t ( f t , π t ) in the left-hand side of (170) separately. Given f, f ′ : S × A → R , data tuple ξ = ( s, a, s ′ ) and policy π , we define the random variable

<!-- formula-not-decoded -->

where a ′ ∼ π ( ·| s ′ ) . Then we have (recall we define P π f in (164))

Combining (177) and (178), we deduce that for any f, f ′ : S × A → R , ξ and π ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Bounding -L t ( f t , π t ) . For any f ∈ Q , π and t ∈ [ T ] , we define X t f,π as

Then we have for any f ∈ Q :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality uses the fact that P π f ∈ Q , which is guaranteed by Lemma 15. Therefore, to upper bound -L t ( f t , π t ) , we only need to bound -∑ t -1 i =1 X i f t ,π t .

<!-- formula-not-decoded -->

Below we use Freedman's inequality (Lemma 2) and a covering number argument to give the desired bound. Repeating a similar argument as the finite-horizon setting, for any ϵ &gt; 0 , there exists an ϵ -net Θ ϵ ⊂ Θ and an ϵ -net Ω ϵ ⊂ Ω such that

Let Q ϵ := { f ϵ = f θ ϵ : θ ϵ ∈ Θ ϵ } , and P ϵ := { π ϵ ( a | s ) = exp( ϕ ( s,a ) ⊤ ω ϵ ) ∑ a ′ ∈A exp( ϕ ( s,a ′ ) ⊤ ω ϵ ) : ω ϵ ∈ Ω ϵ } . For any f ∈ Q and π ∈ P , there exists f ϵ ∈ Q ϵ and π ϵ ∈ P ϵ such that

To invoke Freedman's inequality, we calculate the following quantities.

<!-- formula-not-decoded -->

- Assumption 8 ensures that X t f,π is bounded:

<!-- formula-not-decoded -->

- Repeating the argument for (59), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- Furthermore, we have

<!-- formula-not-decoded -->

where the first equality follows from (177) and (178), and the second inequality follows from Jenson's inequality.

Therefore, by Lemma 2, we have with probability at least 1 -δ , for all t ∈ [ T ] , f ϵ ∈ Q ϵ , π ϵ ∈ P ϵ :

<!-- formula-not-decoded -->

where C 1 &gt; 0 is an absolute constant. From (188) we deduce that for all t ∈ [ T ] f ϵ ∈ Q ϵ , and π ϵ ∈ P ϵ ,

<!-- formula-not-decoded -->

Note that for any t ∈ [ T ] , there exist θ t ∈ Θ and ω t ∈ Ω such that f t = f θ t ∈ Q and π t = π ω t ∈ P . We can choose θ t,ϵ ∈ Θ ϵ and ω t,ϵ ∈ Ω ϵ such that ‖ θ t -θ t,ϵ ‖ 2 ⩽ ϵ and ‖ ω t -ω t,ϵ ‖ 2 ⩽ ϵ . We let f t,ϵ := f θ t,ϵ ∈ Q ϵ . Then by (189) we have for all t ∈ [ T ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last line follows from (183) and (185).

Bounding L t ( f ⋆ , ˜ π ⋆ ) . For any f ∈ Q and t ∈ [ T ] , we define Y t f := E a ′ ∼ ˜ π ⋆ ( ·| s ′ t ) [ l ( f ⋆ , f, ξ t , ˜ π ⋆ ) 2 -l ( f ⋆ , ˜ f ⋆ , ξ t , ˜ π ⋆ ) 2 ] , where ˜ f ⋆ := P ˜ π ⋆ f ⋆ . (191) Note that for any tuple ξ = ( s, a, s ′ ) , we have

<!-- formula-not-decoded -->

∣ ∣ ∣ l ( f ⋆ , f ⋆ , ξ, ˜ π ⋆ ) 2 -l ( f ⋆ , ˜ f ⋆ , ξ, ˜ π ⋆ ) 2 ∣ ∣ ∣ = ∣ ∣ ∣ l ( f ⋆ , f ⋆ , ξ, π ⋆ ) + l ( f ⋆ , ˜ f ⋆ , ξ, ˜ π ⋆ ) ∣ ∣ ∣ ∣ ∣ ∣ l ( f ⋆ , f ⋆ , ξ, ˜ π ⋆ ) -l ( f ⋆ , ˜ f ⋆ , ξ, ˜ π ⋆ ) ∣ ∣ ∣ ⩽ 4 1 -γ ∣ ∣ ∣ ∣ E s ′ ∼ P ( ·| s,a ) a ′ ∼ ˜ π ⋆ ( ·| s ′ ) [ l ( f ⋆ , f ⋆ , ξ, ˜ π ⋆ )] ∣ ∣ ∣ ∣ , (192) where the last line follows from (179). Furthermore, we have where the last line uses Bellman's optimality equation

By Lemma 16, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging the above inequality into (193) and (192), we have

The above bound (196) implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- Repeating the argument for (59), we have

where we also use the definitions of Y t f , ˜ f ⋆ (c.f. (191)), and L t (c.f. (159)). Thus to bound L t ( f ⋆ , ˜ π ⋆ ) , below we bound the sum ∑ t -1 i =1 Y i f for any f ∈ Q and t ∈ [ T ] . To invoke Freedman?s inequality, we calculate the following quantities.

which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- We have

where the first line uses (by (178))

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where a ′ ∼ ˜ π ⋆ ( ·| s ′ t ) , and the second inequality uses Jenson's inequality. · Last but not least, it's easy to verify that

Invoking Lemma 2, and setting η in Lemma 2 as for each f ϵ ∈ Q ϵ , we have with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Reorganizing the above inequality, we have for any f ϵ ∈ Q ϵ , t ∈ [ T ] :

<!-- formula-not-decoded -->

where the last line makes use of the fact that x 2 + bx b 2 / 4 .

Moreoever, for any t ∈ [ T ] , we have

-⩽

<!-- formula-not-decoded -->

where the last inequality uses | f ( s, a ) -f ϵ ( s, a ) | ⩽ ‖ ϕ ( s, a ) ‖ 2 ‖ θ -θ ϵ ‖ 2 ⩽ ϵ . Combining (204) and (205), we have with probability at least 1 -δ , for any t ∈ [ T ] and f ∈ Q ,

<!-- formula-not-decoded -->

where C 2 &gt; 0 is an absolute constant.

By (197) we have

Combining the two bounds. Combining (190) and (207), we have for any t ∈ [ T ] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some absolute constant C &gt; 0 . Letting ϵ = 1 T , we obtain the desired result.

## C.4.2 Proof of Lemma 18

First note that for any policy π and f : S × A → R , we have and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The above two expressions (209) and (210) together give that

where we define and

<!-- formula-not-decoded -->

By Assumption 8, for any f ∈ Q , there exists θ f ∈ Θ such that f ( s, a ) = 〈 θ f , ϕ ( s, a ) 〉 . Thus we have

<!-- formula-not-decoded -->

where W ( f, π ) satisfies

<!-- formula-not-decoded -->

under Assumption 8. We define

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For all t ∈ [ T ] , we define where I d is the d × d identity matrix. Then by Lemma 4, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Further, we could use Lemma 5 to bound the last term in (219), and obtain

<!-- formula-not-decoded -->

where in the last line, we use the definition of d γ ( λ ) (c.f. (172)) and the fact that which is ensured by Assumption 8.

Observe that

<!-- formula-not-decoded -->

where 1 {·} is the indicator function.

To give the desired bound, we will bound (a) and (b) separately.

<!-- formula-not-decoded -->

Bounding (a). We have for any λ &gt; 0 ,

<!-- formula-not-decoded -->

‖ W ( f t , π t ) ‖ Λ t ( λ ) can be bounded as follows:

<!-- formula-not-decoded -->

where we use (215), (218) and the fact that √ a + b ⩽ √ a + √ b for any a, b ⩾ 0 . (223) and (224) together give

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(a-i) in (225) could be bounded as follows:

<!-- formula-not-decoded -->

To bound (a-ii), note that for any π, π ′ ∈ Π , we have

<!-- formula-not-decoded -->

where the inequality follows from Jenson's inequality, and recall ℓ ( f, s, a, π ) is defined in (169). Combining (228) and (220), we could bound (a-ii) in (225) as follows:

<!-- formula-not-decoded -->

Plugging (227) and (229) into (225), we have

<!-- formula-not-decoded -->

Bounding (b). By Assumption 8 and (217), we have

<!-- formula-not-decoded -->

Combining the above inequality with (220), we have

<!-- formula-not-decoded -->

Combining (a) and (b). Plugging (230) and (232) into (222), we have

<!-- formula-not-decoded -->

The first term in the right hand side of (233) could be bounded as

<!-- formula-not-decoded -->

and the second term in the right hand side of (233) could be bounded as

<!-- formula-not-decoded -->

for any η &gt; 0 , where in both (234) and (235), we use the fact that √ ab ⩽ a + b 2 for any a, b ⩾ 0 . Substituting (234) and (235) into (233) and reorganizing the terms, we have

<!-- formula-not-decoded -->

This gives the desired result.