## Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning

## Zijun Chen

Department of Computer Science and Engineering Hong Kong University of Science and Technology zchendg@connect.ust.hk

## Shengbo Wang

Daniel J. Epstein Department of Industrial and Systems Engineering University of Southern California shengbow@usc.edu

## Nian Si ∗

Department of Industrial Engineering and Decision Analytics Hong Kong University of Science and Technology niansi@ust.hk

## Abstract

Motivated by practical applications where stable long-term performance is critical-such as robotics, operations research, and healthcare-we study the problem of distributionally robust (DR) average-reward reinforcement learning. We propose two algorithms that achieve near-optimal sample complexity. The first reduces the problem to a DR discounted Markov decision process (MDP), while the second, Anchored DR Average-Reward MDP, introduces an anchoring state to stabilize the controlled transition kernels within the uncertainty set. Assuming the nominal MDP is uniformly ergodic, we prove that both algorithms attain a sample complexity of ˜ O ( | S || A | t 2 mix ε -2 ) for estimating the optimal policy as well as the robust average reward under KL and f k -divergence-based uncertainty sets, provided the uncertainty radius is sufficiently small. Here, ε is the target accuracy, | S | and | A | denote the sizes of the state and action spaces, and t mix is the mixing time of the nominal MDP. This represents the first finite-sample convergence guarantee for DR average-reward reinforcement learning. We further validate the convergence rates of our algorithms through numerical experiments.

## 1 Introduction

Reinforcement learning (RL) [35] is a core machine learning framework in which agents learn to make decisions by interacting with their environments to maximize long-term rewards. RL has been successfully applied across a wide range of domains-from classic applications in robotics and control systems [19, 13] to more recent advances in game playing [21, 6, 8] and large language model (LLM)-driven reasoning tasks [47, 14].

A central assumption in RL is that the training environment (e.g. a simulator) faithfully represents the real-world deployment setting. In practice, however, this assumption rarely holds, leading to fragile policies underperform when exposed to mismatches between training and deployment environments. This remains a major obstacle to translating RL's successes in simulated settings to reliable performance in real-world applications.

To address this challenge, Zhou et al. [56] built upon the distributionally robust Markov decision process (DR-MDP) framework [16, 27, 48] to propose a distributionally robust reinforcement learning

∗ Corresponding author

(DR-RL) framework. Subsequent work advanced the field, including both model-free [22, 40, 43] and model-based settings [29, 50, 7, 33], as well as approaches for offline learning [32] and generative models [40, 43, 53, 7], along with functional approximations [2, 24].

However, the aforementioned developments predominantly focus on discounted-reward or finitehorizon settings, while the average-reward case remains largely overlooked. This gap is significant because average-reward reinforcement learning is crucial in many practical applications where long-term performance matters more than short-term gains. For example:

- Control systems (e.g., robotics, autonomous vehicles) often require optimizing steady-state performance rather than cumulative discounted rewards.
- Operations research problems (e.g., inventory management, queueing systems) rely on long-run average metrics for stability and efficiency.
- Healthcare or energy management applications may prioritize sustained optimal performance over finite-time rewards.

Average-reward RL is not only important but also more challenging in terms of algorithm design and theoretical analysis. In the standard (non-robust) RL setting, the minimax sample complexity for generative models in discounted-reward cases was resolved as early as 2013 [1]. In contrast, analogous results for average-reward settings were only developed much later, with recent advances in Wang et al. [37], Zurek and Chen [57, 58, 59, 60] under granular structural assumptions.

This paper marks the first systematic analysis of the statistical properties of distributional robust average-reward MDPs (DR-AMDPs) in the tabular setting, addressing a critical gap in the literature.

Specifically, we propose two algorithms that achieve near-optimal (in a minmax sense) sample complexity for learning DR-AMDPs. The first is based on a reduction to the DR discounted-reward MDP (DR-DMDP), where a discount factor γ must be carefully chosen to balance the trade-off between finite sample statistical error and the algorithmic bias introduced by the reduction. The second algorithm, anchored DR-AMDP, modifies the entire uncertainty set of transition kernels by introducing an anchoring state with a certain calibration probability.

To demonstrate their statistical efficiency, we consider a tabular setting where the nominal MDP is uniformly ergodic (Definition 3.3) with a uniform mixing time upper bound t mix for all stationary, Markovian, and deterministic policies. We show that, to learn the optimal robust average reward and policy within ε accuracy, both algorithms achieve a sample complexity of ˜ O ( | S || A | t 2 mix ε -2 ) under KL and f k -divergence uncertainty sets, assuming a sufficiently small uncertainty radius δ (to be defined). Here, | S | and | A | denote the cardinality of the state and action spaces, respectively. Compared to standard (non-robust) average-reward RL literature, this rate is optimal in its dependence on | S || A | and ε .

Our analysis establishes three key contributions to the theory of DR-RL under the average-reward criterion. First, we address a fundamental modeling challenge: conventional uncertainty sets can contain MDPs that are not unichain, thereby invalidating the standard Bellman equations. To resolve this issue, we derive structural conditions on the uncertainty set that ensure stability for all MDPs within it. Second, we develop and analyze the reduction yielding the first stability-sensitive sample complexity bound for DR-DMDPs of ˜ O ( | S || A | t 2 mix (1 -γ ) -2 ε -2 ) and hence the aforementioned upper bound for DR-AMDPs. Building on this framework, we introduce the anchored algorithm and show that its output coincides with that of the reduction approach under a suitable choice of the anchoring parameter. Third, both algorithms are designed to function without requiring prior knowledge of model-specific parameters, particularly the mixing time t mix . Collectively, our work offers a unified treatment that connects robustness, stability, algorithm design, and finite-sample guarantees for DR-RL under the average-reward criterion.

The remainder of this paper is organized as follows: Section 2 surveys existing results for both standard and robust RLs. Section 3 introduces the mathematical preliminaries, including key notations and problem formulation. Our main theoretical contributions, including algorithmic development and sample complexity analysis, are presented in Section 4. Finally, Section 5 provides empirical validation of our theoretical findings and Section 6 concludes the paper and discusses future work.

Table 1: Summary of S.O.T.A. sample complexity results in the literature, where t mix is defined in 3.2.

|          | Type                                        | Sample Complexity                                                                                                                                                                | Origin                                                                                 |
|----------|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Standard | Discounted Discounted Mixing Average Mixing | ˜ Θ( &#124; S &#124;&#124; A &#124; (1 - γ ) - 3 ϵ - 2 ) ˜ Θ ( &#124; S &#124;&#124; A &#124; t mix (1 - γ ) - 2 ε - 2 ) ˜ Θ ( &#124; S &#124;&#124; A &#124; t mix ε - 2 )      | Azar et al. [1], Li et al. [20] Wang et al. [37] Wang et al. [38], Zurek and Chen [57] |
| DR-RL    | Discounted Discounted Mixing Average Mixing | ˜ O ( &#124; S &#124;&#124; A &#124; (1 - γ ) - 4 ϵ - 2 ) ˜ O ( &#124; S &#124;&#124; A &#124; t 2 mix (1 - γ ) - 2 ε - 2 ) ˜ O ( &#124; S &#124;&#124; A &#124; t 2 mix ε - 2 ) | Shi and Chi [32], Wang et al. [43] Theorem 4.3 Theorem 4.4 &4.5                        |

## 2 Literature Review

Sample Complexity of Discounted-Reward Tabular RL: There is an extensive literature on the sample complexity of tabular reinforcement learning. In standard (non-robust) settings, the minimax sample complexity for discounted-reward problems has been well studied. Early works by Azar et al. [1], Li et al. [20] established the minimax rate of ˜ Θ( | S || A | (1 -γ ) -3 ε -2 ) . More recent research has shifted toward instance-dependent bounds that leverage structural properties of the MDP. For example, Wang et al. [38], Zurek and Chen [59] derive tighter instance-dependent bounds of ˜ O ( | S || A | t mix (1 -γ ) -2 ε -2 ) and ˜ O ( | S || A | H(1 -γ ) -2 ε -2 ) under the assumptions that P is uniformly ergodic or weakly communicating, where H denotes the span of the relative value function.

Sample Complexity of Average-Reward Tabular RL: Recently, there has been growing interest in the sample complexity of average-reward reinforcement learning in standard (non-robust) settings. Early work by Jin and Sidford [17] established a bound of ˜ O ( | S || A | t 2 mix ε -2 ) using primal-dual stochastic mirror descent. A rate of ˜ O ( | S || A | t mix ε -3 ) was established later via a reduction to the discounted MDP setting [18]. Subsequent analyses achieved tighter bounds: Wang et al. [38] obtained a rate of ˜ O ( | S || A | t mix ε -2 ) , while Zurek and Chen [57, 59] derived instance-dependent optimal rates of ˜ O ( | S || A | H ε -2 ) for weakly communicating MDPs. Further refinements [58, 60] achieved similar rates without requiring prior knowledge, using plug-in and span penalization approaches. Beyond the generative model setting, Zhang et al. [55] and Chen [5] provided finite-sample analyses for synchronous and asynchronous Q-learning, respectively. Asymptotic properties were also studied in Yu et al. [54], Wan et al. [36] for asynchronous Q -learning.

DR-DMDP and DR-RL: Our work builds on the theoretical foundations of robust MDPs [11, 16, 27, 48, 49, 31, 42], which primarily develop dynamic programming principles under the discountedreward setting. Recent advances in distributionally robust reinforcement learning (DR-RL) have investigated the sample complexity of DR-DMDPs under various divergence-based uncertainty sets. For example, Wang et al. [43], Shi and Chi [32] establish a model-free upper bound of ˜ O ( | S || A | (1 -γ ) -4 ε -2 ) under KL-divergence. Under χ 2 -divergence, Shi et al. [33] obtain a similar upper bound, while Clavier et al. [7] show that l p -norm constraints admit a tighter minimax rate of ˜ Θ( | S || A | (1 -γ ) -3 ε -2 ) . In this paper, by incorporating the mixing time parameter, we present a 'instance-dependence" sample complexity bound of ˜ O ( | S || A | t 2 mix (1 -γ ) -2 ε -2 ) , which improves the dependence on the effective horizon from (1 -γ ) -4 to (1 -γ ) -2 . Several other works contribute to the theoretical and algorithmic landscape of DR-RL in various settings, including Panaganti and Kalathil [28], Yang et al. [52], Xu et al. [50], Blanchet et al. [3], Liu et al. [23], Wang et al. [41], Yang et al. [51].

Distributionally Robust Average-Reward MDPs: While the sample complexity of learning DRDMDPs has been extensively studied, the average-reward setting remains relatively underexplored. Wang et al. [46, 44, 45] propose robust relative value iteration and TD/Q-learning algorithms and prove their convergence, but without providing sample complexity guarantees. Grand-Clément et al. [12] show that for ( s, a ) -rectangular uncertainty sets, the optimal policy can be stationary and deterministic; however, this result may not extend to s -rectangular uncertainty sets. More recently, Wang and Si [39] study average-reward robust MDPs under s -rectangular uncertainty and provide

one-sided weak communication conditions that ensure the existence of solutions to the Bellman optimality equation. Although existing work has investigated the existence and structure of optimal policies in average-reward DR-RL, non-asymptotic sample complexity bounds remain an open question.

We provide a summary of state-of-the-art sample complexity results in the literature in Table 1. In particular, we establish the first sample complexity guarantees for the average-reward DR-RL formulation, which achieves optimal dependence for ε and | S || A | .

## 3 Preliminaries

## 3.1 Markov Decision Processes

We briefly review and define some notations for classical tabular MDP models. Let ∆( S ) denotes the probability simplex over R S . Afinite discounted MDP (DMDP) is defined by the tuple ( S , A , r, P, γ ) . Here, S , A denote the finite state and action spaces respectively; r : S × A → [0 , 1] is the reward function; P = { p s,a ∈ ∆( S ) : ( s, a ) ∈ S × A } is the controlled transition kernel, and γ ∈ (0 , 1) is the discount factor. An average-reward MDP (AMDP) model, on the other hand, is specified by ( S , A , r, P ) without the discount factor.

Define the canonical space Ω = ( S × A ) N equipped with F the σ -field generated by cylinder sets. The state-action process { ( S t , A t ) , t ≥ 0 } is defined by the point evaluation X t ( ω ) = s t , A t ( ω ) = a t for all t ≥ 0 for any ω = ( s 0 , a 0 , s 1 , a 1 , . . . ) ∈ Ω . A general history dependent policy π = ( π t ) t ≥ 0 ∈ Π HD is a sequence of the agent's decision rule. Here, the decision rule π t at time t is a mapping π t : ( S × A ) t × S → ∆( A ) , signifying the conditional distribution of A t given the history. It is known in the literature [30, 12] that to achieve optimal decision making in the context of infinite horizon AMDPs, DMDPs, or their robust variants (to be introduced), it suffices to consider the policy class Π of stationary, Markov, and deterministic policies; i.e. π ∈ Π can be seen as a function π : S → A . Thus, in the subsequent development, we restrict our discussion to Π .

As in Wang et al. [38] a policy π ∈ Π and an initial distribution µ ∈ ∆( S ) uniquely defines a probability measure on (Ω , F ) . We will always assume that µ is the uniform distribution over S . The expectation under this measure is denoted by E π P . To simplify notation, we define P π ( s, s ′ ) := ∑ a ∈ A π ( a | s ) p s,a ( s ′ ) and r π ( s ) := ∑ a ∈ A π ( a | s ) r ( s, a ) .

Discounted-reward MDP (DMDP): Given a DMDP instance ( S , A , r, P, γ ) and π ∈ Π , the discounted value function V π P : S → R is defined as: V π P ( s ) = E π P [ ∑ ∞ t =0 γ t r ( S t , A t ) | S 0 = s ] . An optimal policy π ∗ ∈ Π achieves the optimal value V ∗ P ( s ) := max π ∈ Π V π P ( s ) .

Average-reward MDP (AMDP): For AMDP model ( S , A , r, P ) and π ∈ Π , the long-run averagereward function g π P : S → R is defined as g π P ( s ) := lim sup T →∞ T -1 E π P [ ∑ T -1 t =0 r ( S t , A t ) | S 0 = s ] .

When P is uniformly ergodic (a.k.a.unichain, to be defined later), the g π P is constant across states [30]. In this context, an optimal policy π ∗ ∈ Π achieves the long-run average-reward max π ∈ Π g π P .

## 3.2 Uniform Ergodicity

Motivated by engineering applications where policies induce systems that are stable in the long run, we consider a stability property of MDPs known as uniform ergodicity , a stronger version of the unichain property. In this setting, the controlled Markov chain induced by any reasonable policy converges in distribution to a unique steady state in total variation distance ∥·∥ TV defined by ∥ p -q ∥ TV := sup A ⊂ S | p ( A ) -q ( A ) | for probability vectors p, q ∈ ∆( S ) .

We start with reviewing concepts relevant to uniformly ergodic Markov chains.

Definition 3.1. (Uniform Ergodicity) A transition kernel K ∈ R S × S is uniformly ergodic if one of the following holds

- There exists a probability measure ρ for which ∥ K n ( s, · ) -ρ ∥ TV → 0 for all s ∈ S .
- K satisfies the ( m,p ) -Doeblin condition : For some m ∈ N and p ∈ (0 , 1] if there exists a probability measure ψ and a stochastic kernel R s.t. K m ( s, s ′ ) = pψ ( s ′ ) + (1 -p ) R ( s, s ′ ) .

It is well known [25] that ρ must be the unique stationary distribution of K and that two conditions are equivalent. The ψ and R in the Doeblin condition are known as the minorization measure and the residual kernel respectively.

Next, we introduce the mixing and minorization times associated with a uniformly ergodic kernel K .

Definition 3.2. (Mixing Time and Minorization Time) Define the mixing time of a uniformly ergodic transition kernel P as t mix ( K ) := inf { m ≥ 1 : max s ∈ S ∥ K m ( s, · ) -ρ ( · ) ∥ TV ≤ 1 4 } , and the minorization time as t minorize ( K ) := inf { m/p : min s ∈ S K m ( s, · ) ≥ pψ ( · ) for some ψ ∈ ∆( S ) } .

It is shown in Theorem 1 of Wang et al. [37] that for a uniformly ergodic transition kernel K , these metrics of stability are equivalent up to constants: t minorize ( K ) ≤ 22 t mix ( K ) ≤ 22 log(16) t minorize ( K ) .

While the MDP sample complexity literature typically uses the mixing time as a complexity parameter, for our purposes, the Doeblin condition and the associated minorization time offer sharper theoretical insights into how adversarial robustness affects the statistical complexity of RL. Given the equivalence between t mix ( K ) and t minorize ( K ) , and the latter's advantage in revealing these insights, we will use t minorize ( K ) time throughout this work.

Having reviewed the uniform ergodicity of a stochastic kernel K , we define uniformly ergodic MDPs.

Definition 3.3 (Uniformly Ergodic MDP) . An MDP (or its controlled transition kernel P ) is said to be uniformly ergodic if for all policies π ∈ Π , t minorize ( P π ) &lt; ∞ . Then, define t minorize := max π ∈ Π t minorize ( P π ) &lt; ∞ .

To provide sharper sample complexity results, it is useful to define the following upper bound parameter on m .

<!-- formula-not-decoded -->

This is well defined: In Appendix A, Lemma A.3, we prove that for any transition kernel P π , the equality m/p = t minorize ( P π ) is always attained by some m and p s.t. min s ∈ S P π m ( s, · ) ≥ pψ ( · ) .

It is easy to see that m ∨ ≤ t minorize , and we will demonstrate by the example in Section 5 that it is possible for m ∨ = 1 while t minorize can be arbitrarily large.

## 3.3 Distributionally Robust Discounted-Reward and Average-Reward MDPs

This paper focuses on a robust MDP setting where the stochastic dynamics of the system is influenced by adversarial perturbations on the transition structure. We assume the presence of an adversary that can transition probabilities within KL or f k -divergence uncertainty sets. Specifically, for probability measures q, p ∈ ∆( S ) where q is absolutely continuous w.r.t. p , denoted by q ≪ p , we define D KL ( q || p ) := ∑ s ∈ S log( q ( s ) /p ( s )) q ( s ) and D f k ( q || p ) := ∑ s ∈ S f k ( q ( s ) /p ( s )) p ( s ) . Here, the function f k is defined for k ∈ (1 , ∞ ) by f k ( t ) = ( t k -kt + k -1) / ( k ( k -1)) . When k = 2 , D f k is the χ 2 -divergence.

We assume that the underlying MDP has an unknown nominal controlled transition kernel

<!-- formula-not-decoded -->

For each ( s, a ) ∈ S × A we define the uncertainty set under divergence D = D KL , D f k and parameter δ &gt; 0 centered at p s,a by P s,a ( D,δ ) := { p : D ( p ∥ p s,a ) ≤ δ } . This set contains all possible adversarial perturbations of the transition out of ( s, a ) . Note that the parameter δ controls the size of the P s,a ( D,δ ) , quantifying the power of the adversary. The uncertainty set for the entire controlled transition kernel is P ( D,δ ) := × ( s,a ) ∈ S × A P s,a ( D,δ ) . An uncertainty set of this product from is called SA-rectangular [42].

We will suppress the dependence of D and δ when it is clear from the context. Also, for notation simplicity, define the mapping Γ P s,a : R S → R for P s,a ⊂ ∆( S ) by

<!-- formula-not-decoded -->

Optimal distributionally robust Bellman operators T ∗ γ and T ∗ are central to our algorithmic design.

Definition 3.4. The optimal DR Bellman operators T ∗ γ , T ∗ : R S → R S are defined by

<!-- formula-not-decoded -->

DR-DMDP: A DR-DMDP model is given by the tuple ( S , A , P , r, γ ) . For fixed π ∈ Π , define the DR value function

<!-- formula-not-decoded -->

See Iyengar [16] for a rigorous construction of the expectation E π P . Then, the optimal value function is V ∗ P ( s ) := max π ∈ Π V π P ( s ) . It is well known (c.f. Iyengar [16]) that V ∗ P is the unique solution of the DR Bellnbman equation: V ∗ P = T ∗ γ ( V ∗ P ) .

Note that the expectation E π P is under the adversarial perturbation from a Markovian policy class ( P ) N . It is possible to consider other information structures for the adversary while retaining the satisfaction of the Bellman equation [42].

DR-AMDP: A DR-AMDP model is given by the tuple ( S , A , P , r ) . To simplify our presentation, we restrict our consideration to uniformly ergodic DR-AMDPs.

For each π ∈ Π we define the DR long-run average-reward function by

<!-- formula-not-decoded -->

Natually, the optimal average reward is g ∗ P ( s ) := max π ∈ Π g π P ( s ) .

This paper focuses on a setting where the DR-AMDP is uniformly ergodic in the following sense.

Definition 3.5. A DR-AMDP (or P ) is said to be uniformly ergodic if for all controlled kernels Q ∈ P , Q is uniformly ergodic as in Definition 3.3.

We note that P = P ( D,δ ) is compact in the sense that P s,a ( D,δ ) is a compact subset of ∆( S ) for all s, a . With uniform ergodicity and compactness, Wang et al. [44] shows that g ∗ P ( s ) is constant for s ∈ S which uniquely solves the DR Bellman equations.

Proposition 3.6 (Theorems 7 an 8 of Wang et al. [44]) . If P is uniformly ergodic with a uniformly bounded minorization time, then g ∗ P ( s ) ≡ g ∗ P is constant in s ∈ S . Moreover, there exists a solution ( g, v ) of v ( s ) = T ∗ ( v )( s ) -g ∗ ( s ) for all s ∈ S and any such solution satisfies g ( s ) = g ∗ P for all s ∈ S . Moreover, the policy π ∗ ( s ) ∈ arg max a ∈ A { r ( s, a ) + Γ P s,a ( v ) } achieves the optimal average-reward g ∗ P .

## 4 DR-AMDP: Algorithms and Sample Complexity Upper Bound

In this section, we introduce two algorithms for DR-AMDPs and establish their sample complexity upper bounds. Before presenting the algorithms and results, we first specify the assumptions on the data-generating process and MDP models, along with insights into their rationale and relevance.

We assume the availability of a simulator, a.k.a. a generative model , which allows us to sample independently from the nominal controlled transition kernel p s,a , for any ( s, a ) ∈ S × A . Given sample size n , we sample i.i.d. { S (1) s,a , · · · , S ( n ) s,a } from p s,a and construct the empirical transition probability

<!-- formula-not-decoded -->

Unlike Wang et al. [44], which requires a unichain assumption on every element of P , we only assume that the nominal controlled transition kernel P is uniformly ergodic. We will establish that this weaker condition, coupled with a properly constrained adversarial uncertainty set in Assumption 2, will still guarantee the uniform ergodicity for all Q ∈ P .

Assumption 1. The nominal controlled transition kernel P in (3.2) is uniformly ergodic with minorization time t minorize as in Definition 3.3.

To introduce limits on the adversarial power and facilitate our sample complexity analysis, we introduce the following complexity metric parameter:

Definition 4.1. Define the minimum support as:

<!-- formula-not-decoded -->

Assumption 2. Suppose the parameter δ satisfies δ ≤ 1 8 m 2 ∨ p ∧ when P = P ( D KL , δ ) , and δ ≤ 1 max { 8 , 4 k } m 2 ∨ p ∧ when P = P ( D f k , δ ) .

Here, the constant 1 / 8 can potentially be relaxed. As mentioned earlier, this restriction on the adversarial power parameter δ ensures the minorization times remain uniformly bounded across the uncertainty set by a constant multiple of the nominal controlled kernel's minorization time.

Proposition 4.2. Suppose Assumptions 1 hold, and P = P ( D KL , δ ) or P ( D f k , δ ) satisfying Assumption 2. Then, for all Q ∈ P and π ∈ Π , t minorize ( Q π ) ≤ 2 t minorize , where t minorize is from Assumption 1.

The proof is deferred to Appendix B, E. We further note that without Assumption 2, the Hard MDP instance in Section 5 will have a non-mixing worst-case adversarial kernel and state-dependent optimal average reward even when δ = Θ( p ∧ /m 2 ∨ ) . This emphasizes the necessity of limiting the adversarial power to obtain a stable worst-case system and state-independent average reward.

We propose two algorithms: Reduction to DR-DMDP and Anchored DR-AMDP. Notably, these are the first to provide finite-sample guarantees for DR-AMDPs and achieve the canonical n -1 / 2 convergence rate in policy and estimation. Furthermore, both algorithms operate without requiring prior knowledge of t minorize . Together, these contributions represent foundational advances in the study of data-driven learning of DR-AMDPs.

## 4.1 Reduction to DR-DMDP

First, we present the algorithmic reduction from DR-AMDP to DR-DMDP. The algorithm design in this section is inspired by prior works [46, 18]. Specifically, we apply value iteration to an auxiliary empirical DR-DMDP model to obtain both the value function and optimal policy. Utilizing a calibrated discount γ = 1 -n -1 / 2 where n is the input sample size in Algorithm 1, we achieve an ε -approximation of the target DR-AMDP value and policy with the auxiliary DR-DMDP using ˜ O ( | S || A | t 2 minorize p -1 ∧ ε -2 ) samples.

## Algorithm 1 Distributional Robust DMDP: DR -DMDP( γ, n, D )

Input: Discount factor γ ∈ (0 , 1) , sample size n ≥ 1 , D = D KL or D f k . For all ( s, a ) ∈ S × A , compute the n -sample empirical transition probability ̂ p s,a as in (4.1) Construct the uncertainty set as ̂ P = × ( s,a ) ∈ S × A ̂ P s,a where ̂ P s,a = { p : D ( p || ̂ p s,a ) ≤ δ } . Compute the solution V ∗ P as the solution to the empirical DR Bellman equation; i.e. ∀ s ∈ S :

<!-- formula-not-decoded -->

Then, extract any optimal policy ̂ π ∗ ∈ Π from ̂ π ∗ ( s ) ∈ arg max a ∈ A { r ( s, a ) + γ Γ ̂ P s,a ( V ∗ ̂ P ) } . return ̂ π ∗ , V ∗ ̂ P

With the help of Proposition 4.2, the Algorithm 1 has the following optimal sample complexity guarantee.

Theorem 4.3. Suppose P = P ( D KL , δ ) or P ( D f k , δ ) and Assumptions 1, 2 are in force. Then, for any n ≥ 32 p -1 ∧ log(2 | S | 2 | A | /β ) , the policy ̂ π ∗ and value function V ∗ P returned by Algorithm 1

̂

̂

satisfy

<!-- formula-not-decoded -->

with probability at least 1 -β , where the constants c, c ′ ≤ 96 2 for both the KL and f k cases.

The proof of Theorem 4.3 is deferred to Appendix D, G. We note that Theorem 4.3 implies that to achieve an ε -optimal policy as well as producing a uniform ε -error estimate of V ∗ P with high probability using Algorithm 1, we need ˜ O ( | S || A | t 2 minorize (1 -γ ) -2 p -1 ∧ ε -2 ) samples. Compared to state-of-the-art sample complexity results for DR-DMDPs [32, 43, 33], Theorem 4.3 provides a significant refinement: when the nominal controlled kernel is uniformly ergodic, the effective horizon dependence improves to (1 -γ ) -2 . Notably, this (1 -γ ) -2 scaling is also known to be optimal in the non-robust setting [37], which corresponds to DR-DMDPs when δ = 0 . As we will show, this optimal dependence directly enables the canonical n -1 / 2 convergence rate for policy learning and value estimation in the DR-AMDP setting.

## Algorithm 2 Reduction to DMDP

Input: Samples size n . Assign γ = 1 -1 / √ n and run Algorithm 1 with input DR -DMDP( γ, n ) to obtain ̂ π ∗ , V ∗ ̂ P . return ˆ π ∗ , V ∗ / √ n

̂ P

Theorem 4.4. Suppose P = P ( D KL , δ ) or P ( D f k , δ ) and Assumptions 1 and 2 are in force. Then for any n ≥ 32 p -1 ∧ log(2 | S | 2 | A | /β ) , the policy ̂ π ∗ and value function V ∗ ̂ P / √ n returned by Algorithm 2 satisfies

<!-- formula-not-decoded -->

with probability 1 -β , where the constants c, c ′ ≤ 120 2 for both the KL and f k cases.

Again, we remark that Theorem 4.4 implies that to achieve an ε -optimal policy as well as producing a uniform ε -error estimate of the optimal robust long-run average reward with high probability using Algorithm 2, we need ˜ O ( | S || A | t 2 minorize p -1 ∧ ε -2 ) samples.

## 4.2 Anchored DR-AMDP

In this section, we develop anchored DR-AMDP Algorithm 3 that avoids solving a DR-DMDP subproblem. Inspired by Fruit et al. [10], Zurek and Chen [58]'s anchoring approach for classical MDPs, our anchored DR-AMDP approach modifies the entire uncertainty set of controlled transition kernels via a uniform anchoring state s 0 and a calibrated strength parameter ξ . We show that Algorithm 3 enjoys the same error and sample complexity upper bounds to Algorithm 2.

Theorem 4.5. Suppose Assumption 1 and 2 are in force. Then for any n ≥ 32 p -1 ∧ log(2 | S | 2 | A | /β ) , the policy ̂ π ∗ and value function g ∗ ̂ P returned by Algorithm 3 satisfies (4.4) with V ∗ ̂ P / √ n replaced by g ∗ ̂ P with probability at least 1 -β .

This theorem implies the same ˜ O ( | S || A | t 2 minorize p -1 ∧ ε -2 ) sample complexity to achieve an ε -optimal policy and value estimation.

Sketched Proof of Theorems 4.4 and 4.5. Our proof begin with establishing that, under Assumption 2, each adversarial transition kernel Q ∈ P consists of conditional distributions q s,a that are absolutely continuous with respect to the nominal distributions p s,a , with a uniform lower bound 1 -1 2 m ∨ on its Radon-Nikodym derivative. This guarantees that t minorize ( Q π ) ≤ O ( t minorize ) for all Q ∈ P and π ∈ Π .

## Algorithm 3 Anchored DR-AMDP

Input: Sample size n ≥ 1 and divergence D = D KL or D f k . For all ( s, a ) ∈ S × A , compute the n -sample empirical transition probability ̂ p s,a as in (4.1). Let ξ = 1 / √ n and fixed any anchoring point s 0 ∈ S . Construct the anchored empirical uncertainty set as ̂ P = × ( s,a ) × S × A ̂ P s,a , where ̂ P s,a = { (1 -ξ ) p + ξ 1 e ⊤ s 0 : D ( p ∥ ̂ p s,a ) ≤ δ } . Solve the empirical DR average reward Bellman equation

<!-- formula-not-decoded -->

for a solution pair ( g ∗ ̂ P , v ∗ ̂ P ) .

Extract an optimal policy ̂ π ∗ ∈ Π as ̂ π ∗ ( s ) ∈ arg max a ∈ A { r ( s, a ) + Γ ̂ P s,a ( v ∗ ̂ P ) } return ̂ π ∗ , g ∗ ̂ P

Figure 1: Transition diagram of the hard MDP instance in Wang et al. [37].

<!-- image -->

Next, combining Theorem D.1 with Lemma C.3, we establish that the policy error satisfies

<!-- formula-not-decoded -->

This reduces the analysis of the policy error to bounding the estimation error of the DR Bellman operator evaluated at V π P . As the rewards cancel out, it remains to show that the DR functional applied to V π P satisfy appropriate concentration bound.

To this end, we apply the strong duality for the DR functional, the bound in Lemma C.6, and a Bernstein-type inequality to show that for any function V , the deviation satisfies

<!-- formula-not-decoded -->

with high probability.

Finally, by selecting the parameters γ = 1 -1 / √ n and η = 1 / √ n , and noting that Span( V π P ) ≤ O ( t minorize ) , we complete the proof for the KL-divergence case of Theorems 4.4 and 4.5. The argument under the f k -divergence formulation proceeds in an analogous manner.

## 5 Numerical Experiments

In this section, we present numerical experiments to validate our theoretical results. We employ the Hard MDP family introduced in Wang et al. [37], which confirms a minimax sample complexity lower bound of Ω( t minorize ε -2 ) for estimating the average reward to within an ε absolute error in the non-robust setting, matching the known upper bound. Our experiments show an empirical convergence rate of n -1 / 2 for both algorithms, validating them as the first algorithms that achieve this rate in the DR-AMDP setting.

Definition 5.1 (Hard MDP Family in Wang et al. [37]) . This family of MDP instances has S = { 1 , 2 } , A = { 1 , 2 } , and reward function r (1 , · ) = 1 and r (2 , · ) = 0 . The controlled transition kernel P is parameterized by p with transition diagram given in Figure 1.

Observe that under this controlled transition kernel, all stationary policies induce the same transition matrix P π . Moreover, restricting p ∈ (0 , 1 2 ] we have P π m = (1 -(1 -2 p ) m ) 1 2 J +(1 -2 p ) m I , where

.

J is the matrix of all 1 and I is the identity matrix. Therefore, P π i is ( m, (1 -(1 -2 p ) m )) -Doeblin. Thus, the minorization time of P is inf m ≥ 1 m/ (1 -(1 -2 p ) m ) = 1 2 p .

This example clarifies our use of m ∨ in Definition 3.3: while m π ≡ 1 for all p ∈ (0 , 1 / 2] , the minorization time t minorize is unbounded, approaching infinity as p goes to 0 .

Next, we evaluate the performance of Algorithm 2 and 3 by analyzing their value approximation errors under both KL and χ 2 uncertainty sets. χ 2 is a special case of f k -divergence with k = 2 .

The sub-figures in Figure 2 presents the error achieved by the algorithms using a total of n transition samples for every state-action pair. Each data point in the plots corresponds to a single estimate generated by one independent run of the corresponding algorithm. Then, we compute the l ∞ -error between the estimator and the ground-truth average-reward, which is computed via value iteration.

We then perform regression on data points on each MDP instance with the same parameter p . The plots demonstrate the error converging with rate n -1 / 2 , evidenced by the slope of -1 / 2 in Figure 2 on a log-log scale. We observe a remarkably low variance around the regression line of both algorithms, given that each data point is a single independent run of the corresponding algorithm.

Figure 2: Comparative numerical experiments on (a-b) Algorithm 2 and (c-d) Algorithm 3 for the hard MDP instance, demonstrating ε -dependence under different divergence measures.

<!-- image -->

In addition to these experiment, we also perform a larger scale experiment to stress test our algorithm. Due to space limitations, the report is provided in Appendix H.

## 6 Conclusion and Future Work

In this work, we study distributionally robust average-reward reinforcement learning under a generative model. We first establish an instance-dependent bound of ˜ O ( | S || A | t 2 minorize (1 -γ ) -2 ε -2 ) for DR-DMDP. Building on this result, we propose two a priori knowledge-free algorithms with finitesample complexity ˜ O ( | S || A | t 2 minorize ε -2 ) . Our work provides novel insights into the relationship between uniform ergodicity and sample complexity under distributional robustness.

While our results rely on the assumptions of uniform ergodicity and constraints on the uncertainty size, we acknowledge these as potential limitations. For future work, we plan to generalize these results to weakly communicating settings and, potentially, multichain MDPs, and investigate broader uncertainty sets (e.g., l p -balls and Wasserstein metrics).

## Acknowledgement

N. Si gratefully acknowledges the support from the Hong Kong Research Grants Council [Themebased Research Scheme T32-615/24-R].

## References

- [1] Azar, M. G., Munos, R., and Kappen, H. J. (2013). Minimax pac bounds on the sample complexity of reinforcement learning with a generative model. Machine learning , 91(3):325-349.
- [2] Blanchet, J., Lu, M., Zhang, T., and Zhong, H. (2023). Double pessimism is provably efficient for distributionally robust offline reinforcement learning: Generic algorithm and robust partial coverage. Advances in Neural Information Processing Systems , 36:66845-66859.
- [3] Blanchet, J., Lu, M., Zhang, T., and Zhong, H. (2024). Double pessimism is provably efficient for distributionally robust offline reinforcement learning: Generic algorithm and robust partial coverage. Advances in Neural Information Processing Systems , 36.
- [4] Boucheron, S., Lugosi, G., and Bousquet, O. (2004). Concentration inequalities. In Bousquet, O., von Luxburg, U., and Rätsch, G., editors, Advanced Lectures on Machine Learning: ML Summer Schools 2003, Canberra, Australia, February 2 - 14, 2003, Tübingen, Germany, August 4 - 16, 2003, Revised Lectures , pages 208-240. Springer Berlin Heidelberg.
- [5] Chen, Z. (2025). Non-asymptotic guarantees for average-reward q-learning with adaptive stepsizes. arXiv preprint arXiv:2504.18743 .
- [6] Choi, J. J., Laibson, D., Madrian, B. C., and Metrick, A. (2009). Reinforcement learning and savings behavior. The Journal of finance , 64(6):2515-2534.
- [7] Clavier, P., Shi, L., Le Pennec, E., Mazumdar, E., Wierman, A., and Geist, M. (2024). Nearoptimal distributionally robust reinforcement learning with general l \_ p norms. Advances in Neural Information Processing Systems , 37:1750-1810.
- [8] Deng, Y., Bao, F., Kong, Y., Ren, Z., and Dai, Q. (2017). Deep direct reinforcement learning for financial signal representation and trading. IEEE Transactions on Neural Networks and Learning Systems , 28(3):653-664.
- [9] Duchi, J. and Namkoong, H. (2021). Learning models with uniform performance via distributionally robust optimization. The Annals of Statistics , 49.
- [10] Fruit, R., Pirotta, M., Lazaric, A., and Ortner, R. (2018). Efficient bias-span-constrained exploration-exploitation in reinforcement learning. In International Conference on Machine Learning , pages 1578-1586. PMLR.
- [11] González-Trejo, J., Hernández-Lerma, O., and Hoyos-Reyes, L. F. (2002). Minimax control of discrete-time stochastic systems. SIAM Journal on Control and Optimization , 41(5):1626-1659.
- [12] Grand-Clément, J., Petrik, M., and Vieille, N. (2025). Beyond discounted returns: Robust Markov decision processes with average and Blackwell optimality. arXiv:2312.03618 [math].
- [13] Gu, S., Holly, E., Lillicrap, T., and Levine, S. (2017). Deep reinforcement learning for robotic manipulation with asynchronous off-policy updates. In 2017 IEEE international conference on robotics and automation (ICRA) , pages 3389-3396. IEEE.
- [14] Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. (2025). Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 .
- [15] Hu, Z. and Hong, L. J. (2013). Kullback-leibler divergence constrained distributionally robust optimization. Available at Optimization Online , 1(2):9.
- [16] Iyengar, G. N. (2005). Robust dynamic programming. Mathematics of Operations Research , 30(2):257-280.

- [17] Jin, Y. and Sidford, A. (2020). Efficiently solving mdps with stochastic mirror descent.
- [18] Jin, Y . and Sidford, A. (2021). Towards tight bounds on the sample complexity of average-reward mdps. In International Conference on Machine Learning , pages 5055-5064. PMLR.
- [19] Kober, J., Bagnell, J. A., and Peters, J. (2013). Reinforcement learning in robotics: A survey. The International Journal of Robotics Research , 32(11):1238-1274.
- [20] Li, G., Wei, Y., Chi, Y., Gu, Y., and Chen, Y. (2020). Breaking the sample size barrier in model-based reinforcement learning with a generative model. Advances in neural information processing systems , 33:12861-12872.
- [21] Li, Y., Szepesvari, C., and Schuurmans, D. (2009). Learning exercise policies for american options. In Artificial Intelligence and Statistics , pages 352-359.
- [22] Liu, Z., Bai, Q., Blanchet, J., Dong, P., Xu, W., Zhou, Z., and Zhou, Z. (2022a). Distributionally Robust $Q$-Learning. In Proceedings of the 39th International Conference on Machine Learning , pages 13623-13643. PMLR. ISSN: 2640-3498.
- [23] Liu, Z., Bai, Q., Blanchet, J., Dong, P., Xu, W., Zhou, Z., and Zhou, Z. (2022b). Distributionally robust Q-learning. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S., editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 13623-13643. PMLR.
- [24] Ma, X., Liang, Z., Blanchet, J., Liu, M., Xia, L., Zhang, J., Zhao, Q., and Zhou, Z. (2022). Distributionally robust offline reinforcement learning with linear function approximation. arXiv preprint arXiv:2209.06620 .
- [25] Meyn, S. P. and Tweedie, R. L. (2012). Markov chains and stochastic stability . Springer Science &amp;Business Media.
- [26] Milgrom, P. and Segal, I. (2002). Envelope theorems for arbitrary choice sets. Econometrica , 70(2):583-601.
- [27] Nilim, A. and El Ghaoui, L. (2005). Robust control of markov decision processes with uncertain transition matrices. Operations Research , 53(5):780-798.
- [28] Panaganti, K. and Kalathil, D. (2021). Sample complexity of robust reinforcement learning with a generative model.
- [29] Panaganti, K. and Kalathil, D. (2022). Sample Complexity of Robust Reinforcement Learning with a Generative Model. In Proceedings of The 25th International Conference on Artificial Intelligence and Statistics , pages 9582-9602. PMLR. ISSN: 2640-3498.
- [30] Puterman, M. L. (2009). Markov Decision Processes: Discrete Stochastic Dynamic Programming . Number v.414 in Wiley Series in Probability and Statistics. John Wiley &amp; Sons, Inc, Hoboken.
- [31] Shapiro, A. (2022). Distributionally robust modeling of optimal control. Operations Research Letters , 50(5):561-567.
- [32] Shi, L. and Chi, Y. (2024). Distributionally robust model-based offline reinforcement learning with near-optimal sample complexity. Journal of Machine Learning Research , 25(200):1-91.
- [33] Shi, L., Li, G., Wei, Y., Chen, Y., Geist, M., and Chi, Y. (2024). The curious price of distributional robustness in reinforcement learning with a generative model.
- [34] Si, N., Zhang, F., Zhou, Z., and Blanchet, J. (2020). Distributionally Robust Policy Evaluation and Learning in Offline Contextual Bandits. In Proceedings of the 37th International Conference on Machine Learning , pages 8884-8894. PMLR. ISSN: 2640-3498.
- [35] Sutton, R. S. and Barto, A. G. (2018). Reinforcement learning: An introduction . MIT press.
- [36] Wan, Y., Yu, H., and Sutton, R. S. (2024). On convergence of average-reward q-learning in weakly communicating markov decision processes.

- [37] Wang, S., Blanchet, J., and Glynn, P. (2023a). Optimal Sample Complexity of Reinforcement Learning for Mixing Discounted Markov Decision Processes. arXiv:2302.07477 [cs].
- [38] Wang, S., Blanchet, J., and Glynn, P. (2024a). Optimal Sample Complexity for Average Reward Markov Decision Processes. arXiv:2310.08833 [cs].
- [39] Wang, S. and Si, N. (2025). Bellman optimality of average-reward robust markov decision processes with a constant gain.
- [40] Wang, S., Si, N., Blanchet, J., and Zhou, Z. (2023b). A Finite Sample Complexity Bound for Distributionally Robust Q-learning. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics , pages 3370-3398. PMLR. ISSN: 2640-3498.
- [41] Wang, S., Si, N., Blanchet, J., and Zhou, Z. (2023c). A finite sample complexity bound for distributionally robust Q-learning.
- [42] Wang, S., Si, N., Blanchet, J., and Zhou, Z. (2024b). On the Foundation of Distributionally Robust Reinforcement Learning. arXiv:2311.09018 [cs].
- [43] Wang, S., Si, N., Blanchet, J., and Zhou, Z. (2024c). Sample complexity of variance-reduced distributionally robust q-learning. Journal of Machine Learning Research , 25(341):1-77.
- [44] Wang, Y., Velasquez, A., Atia, G., Prater-Bennette, A., and Zou, S. (2023d). Robust averagereward markov decision processes. Proceedings of the AAAI Conference on Artificial Intelligence , 37(12):15215-15223.
- [45] Wang, Y., Velasquez, A., Atia, G., Prater-Bennette, A., and Zou, S. (2024d). Robust averagereward reinforcement learning. Journal of Artificial Intelligence Research , 80:719-803.
- [46] Wang, Y., Velasquez, A., Atia, G. K., Prater-Bennette, A., and Zou, S. (2023e). Model-free robust average-reward reinforcement learning. In International Conference on Machine Learning , pages 36431-36469. PMLR.
- [47] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems , 35:24824-24837.
- [48] Wiesemann, W., Kuhn, D., and Rustem, B. (2013). Robust markov decision processes. Mathematics of Operations Research , 38(1):153-183.
- [49] Xu, H. and Mannor, S. (2010). Distributionally robust markov decision processes. In NIPS , pages 2505-2513.
- [50] Xu, Z., Panaganti, K., and Kalathil, D. (2023). Improved sample complexity bounds for distributionally robust reinforcement learning.
- [51] Yang, W., Wang, H., Kozuno, T., Jordan, S. M., and Zhang, Z. (2023). Avoiding model estimation in robust markov decision processes with a generative model.
- [52] Yang, W., Zhang, L., and Zhang, Z. (2021). Towards theoretical understandings of robust markov decision processes: Sample complexity and asymptotics.
- [53] Yang, W., Zhang, L., and Zhang, Z. (2022). Toward theoretical understandings of robust markov decision processes: Sample complexity and asymptotics. The Annals of Statistics , 50(6):32233248.
- [54] Yu, H., Wan, Y., and Sutton, R. S. (2025). Asynchronous stochastic approximation and averagereward reinforcement learning.
- [55] Zhang, S., Zhang, Z., and Maguluri, S. T. (2021). Finite sample analysis of average-reward td learning and q-learning. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W., editors, Advances in Neural Information Processing Systems , volume 34, pages 1230-1242. Curran Associates, Inc.

- [56] Zhou, Z., Bai, Q., Zhou, Z., Qiu, L., Blanchet, J., and Glynn, P. (2021). Finite-sample regret bound for distributionally robust offline tabular reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pages 3331-3339. PMLR.
- [57] Zurek, M. and Chen, Y. (2023). Span-based optimal sample complexity for average reward mdps. arXiv preprint arXiv:2311.13469 .
- [58] Zurek, M. and Chen, Y. (2024a). The Plug-in Approach for Average-Reward and Discounted MDPs: Optimal Sample Complexity Analysis. arXiv:2410.07616 [cs].
- [59] Zurek, M. and Chen, Y. (2024b). Span-Based Optimal Sample Complexity for Weakly Communicating and General Average Reward MDPs. arXiv:2403.11477 [cs].
- [60] Zurek, M. and Chen, Y. (2025). Span-agnostic optimal sample complexity and oracle inequalities for average-reward rl. arXiv preprint arXiv:2502.11238 .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All the main claims accurately reflect the paper's contribution.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations due to assumptions made for the theoretical guarantee are discussed in Section 6.

## Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The full set of assumptions is formally stated in Section 4, with rigorous proofs provided in the appendix.

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

Justification: The complete algorithmic implementation under the generative model is provided in Section 4, including detailed pseudo-code. Section 5 specifies all experimental configurations and parameter settings, ensuring full reproducibility of our results.

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

Justification: We have included the complete experimental code in the supplemental materials.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Section 5 documents all experimental parameters and implementation details necessary for understanding the results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We present statistical significance by fitting the regression line, which is clearly outlined in the figures in Section 5.

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

Justification: Our sample complexity analysis provides statistical guarantees that are independent of computational power considerations, which represent a distinct aspect from our theoretical focus.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: NeurIPS Code of Ethics has been scrutinised and followed carefully.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: This paper establishes the first sample complexity upper bound for DR-AMDP, representing a significant advance in distributionally robust reinforcement learning theory.

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

Justification: There is no such risk.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

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

Justification: No such assets are included.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not include such kind of experiment.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: We don't have human participants in the study.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Our core methodology, including algorithm design and theoretical analysis, was developed independently without employing LLM.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendices

## A Notations and Basic Properties

In this section, we present the technical proof for DR-MDPs. Before introducing the theoretical foundations and analyzing related statistical properties, we first define key notations and auxiliary quantities to facilitate subsequent analysis.

For ARMDPs, it is useful to consider the span semi-norm Puterman [30]. For vector u ∈ R d , let e = (1 , · · · , 1) ⊤ , and define:

<!-- formula-not-decoded -->

Note that the span semi-norm satisfies the triangle inequality

<!-- formula-not-decoded -->

Our analysis relies extensively on two fundamental operators: the DR discounted policy Bellman operator T π γ and its optimal counterpart T ∗ γ . These operators are defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, we define the empirical DR discounted policy operator ̂ T π γ and its optimal counterpart ̂ T ∗ γ as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It has been shown that the DR value function V π P is the unique fixed-point of the DR discounted policy operator (A.2), a.k.a. V π P is the solution to the DR discounted Bellman equation: V π P = T π γ ( V π P ) Iyengar [16], Puterman [30], Nilim and El Ghaoui [27].

We introduce some technical notations. For function v : S → R , let

<!-- formula-not-decoded -->

Notice that with the above notation, we simplify the expectation as E p [ v ] = p [ v ] .

For probability measure p, q ∈ ∆( S ) , we say that p is absolutely continuous w.r.t. q , denoted by p ≪ q , if q ( s ) = 0 implies that p ( s ) = 0 . If p ≪ q , we define the likelihood ratio, a.k.a. Radon-Nikodym derivative,

<!-- formula-not-decoded -->

We say that p and q are mutually absolutely continuous, denoted by p ∼ q if p ≪ q and q ≪ p .

For p ∈ ∆( S ) , we also define the L ∞ ( p ) norm of a function v : S → R by

<!-- formula-not-decoded -->

In the DR setting, given uncertainty set P s,a and a function V : S → R , we say p ∗ is a worst-case measure if

<!-- formula-not-decoded -->

Also, recall Definition 4.1, we define the minimal support probability p ∧ in measuring the samples required.

<!-- formula-not-decoded -->

The sample complexity's dependence on p ∧ emerges from two theoretical requirements. First, accurate estimation of the worst-case transition kernel demands that samples capture the distribution's support, necessitating at least Ω(1 / p ∧ ) samples to ensure all non-zero probability transitions are observed. Second, the perturbed transition kernel needs to preserve certain mixing characteristics. This is crucial for us to establish a uniform high probability bound on the minorization times of the controlled kernels in the uncertainty set.

Specifically, we consider the "good events" set Ω n,d , as the collection of empirical measures that remain sufficiently close to the nominal transition kernel P . Recall from (4.1) that

For any d &gt; 0 we define,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as the relative difference between ̂ p s,a and p s,a is close up to d . Then, define:

<!-- formula-not-decoded -->

Theorem A.1 (Bernstein's inequality, Theorem 3 in Boucheron et al. [4]) . Let X 1 , X 2 , . . . , X n be independent random variables with E [ X i ] = µ and | X i -µ | ≤ M almost surely. Then we have:

with probability at least

<!-- formula-not-decoded -->

By Bernstein's inequality, we could bound the probability measure of Ω n,d :

Lemma A.2. When the relative difference d satisfies:

then

<!-- formula-not-decoded -->

Proof. Let supp( p s,a ) := { s ′ : p s,a ( s ′ ) &gt; 0 } . Given n i.i.d. samples { S (1) s,a , S (2) s,a , · · · , S ( n ) s,a } drawn from p s,a . We define the indicator variables:

Note that X i s,a ( s ′ ) ∼ Bernoulli( p s,a ( s ′ )) for all 1 ≤ i ≤ n . Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By Union bound, we have:

<!-- formula-not-decoded -->

Then, by Bernstein's inequality (A.8), for any action-state pairs ( s, a ) ∈ S × A , and next state s ′ ∈ supp( p s,a ) , since E p s,a [ X i s,a ( s ′ )] = p s,a ( s ′ ) , and | X i s,a ( s ′ ) -p s,a | ≤ 1 , we have:

<!-- formula-not-decoded -->

with probability at least 1 -β | S | 2 | A | . Thus, let

<!-- formula-not-decoded -->

then, for all ( s, a ) ∈ S × A , and s ′ ∈ supp( p s,a ) :

<!-- formula-not-decoded -->

Where ( i ) relies on Var (Bernoulli( p s,a ( s ′ ))) = p s,a ( s ′ )(1 -p s,a ( s ′ )) , then we conclude, for all ( s, a ) ∈ S × A and s ′ ∈ supp( p s,a ) , with probability 1 -β | S | 2 | A | , we have:

<!-- formula-not-decoded -->

Then we conclude that:

<!-- formula-not-decoded -->

And further:

<!-- formula-not-decoded -->

Proved.

Lemma A.3. Let the transition kernel K be uniformly ergodic. If t minorize ( K ) &lt; ∞ , then there exists an ( m,p ) pair, such that:

Proof. By definition:

<!-- formula-not-decoded -->

As t minorize ( K ) &lt; ∞ , then, there exists a constant C &gt; 0 , such that

<!-- formula-not-decoded -->

As the feasible ( m,p ) -pair such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

m p ≤ C , then m ≤ Cp ≤ C because p ∈ (0 , 1] , we conclude:

<!-- formula-not-decoded -->

Define C m and p max ( m ) as:

<!-- formula-not-decoded -->

With this definition, note that

<!-- formula-not-decoded -->

We will show that the set C m is closed, hence p max ( m ) ∈ C m is achieved.

Since S is finite, ∆( S ) ⊂ R | S | is compact. Consider any sequence { p n } ⊆ C m such that p n → p . Then, there exists ψ n ( S ) , such that

<!-- formula-not-decoded -->

As ∆( S ) is compact, sequence { ψ n } has subsequence { ψ n k } , such that:

<!-- formula-not-decoded -->

and a corresponding { p n k } such that p n k → p . Then for any ( s, s ′ ) ∈ S × S :

<!-- formula-not-decoded -->

We have p n k → p and ψ n k ( s ′ ) → ψ ( s ′ ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus:

i.e. p ∈ C m . Hence, C m is a closed and p max ( m ) ∈ C m .

Therefore,

<!-- formula-not-decoded -->

Since C is finite, there exists a m ∗ ∈ { 1 , 2 , · · · , ⌊ C ⌋} such that

<!-- formula-not-decoded -->

Lemma A.4. Suppose the controlled transition kernel P is uniformly ergodic. If for all Q ∈ P , p s,a ≪ q s,a holds for all ( s, a ) ∈ S × A , then P is uniformly ergodic.

Proof. Since P is a uniformly ergodic, then for all π ∈ Π , t minorize ( P π ) , by Lemma A.3 there exists ( m π , p π ) such that:

<!-- formula-not-decoded -->

For any Q ∈ P and state pair ( s 0 , s m π ) ∈ S × S , the m π -step transition probability of P π , and Q π can be expressed as:

<!-- formula-not-decoded -->

Define:

<!-- formula-not-decoded -->

Since p s,a ≪ q s,a holds for all ( s, a ) pairs, we have for any tuple ( s 1 , s 2 , · · · , s m π -1 ) ∈ U ( s 0 , s m π ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

by absolute continuity, it implies:

Then:

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

Denote c := min ( s,s ′ ) ∈ S × S c ( s, s ′ ) &gt; 0 , we conclude:

<!-- formula-not-decoded -->

Since P π satisfies ( m π , p π ) -Doeblin condition, for some ψ ∈ ∆( S ) , we have:

<!-- formula-not-decoded -->

Q π satisfies ( m π , q π ) -Doeblin condition where q π = cp π . Thus, we concludes for all Q ∈ P and π ∈ Π , Q π satisfies ( m π , q π ) -Doeblin condition, and:

<!-- formula-not-decoded -->

Finally, we concludes, for all Q ∈ P :

<!-- formula-not-decoded -->

P is uniformly ergodic.

Having established the uniformly ergodic property preservation under absolute continuity, we now introduce additional technical tools central to our analysis. Our proof strategy fundamentally relies on the span semi-norm framework for value functions in DMDPs, the following proposition from Wang et al. [37] formalizes this connection:

Proposition A.5 (Proposition 6.1 in Wang et al. [37]) . Suppose P π satisfies ( m,p ) -Doeblin condition, and V π P is the value function associate with kernel P under policy π , then Span( V π P ) ≤ 3 m/p

Our core approach involves approximating the DR-AMDP through its DR-DMDP counterpart. To establish this connection rigorously, we require the following fundamental lemma that bridges discounted and average-reward value functions:

Lemma A.6 (Lemma 1 in Wang et al. [38]) . Suppose P π satisfies ( m π , p π ) -Doeblin condition and t minorize ( P π ) &lt; ∞ , then:

<!-- formula-not-decoded -->

Proposition A.7 (Restatement of Proposition 3.6) . If P is uniformly ergodic with a uniformly bounded minorization time, then g ∗ P ( s ) ≡ g ∗ P is constant in s ∈ S . Moreover, there exists a solution ( g, v ) of v ( s ) = T ∗ ( v )( s ) -g ∗ ( s ) for all s ∈ S and any such solution satisfies g ( s ) = g ∗ P for all s ∈ S . Moreover, the policy π ∗ ( s ) ∈ arg max a ∈ A { r ( s, a ) + Γ P s,a ( v ) } achieves the optimal average-reward g ∗ P .

Proof. Since P is uniformly ergodic with uniformly bounded minorization time, for any stationary policy π the kernel Q π satisfies an ( m P π , p Q π ) -Doeblin condition. Then, by Theorem UE in Wang et al. [37],

<!-- formula-not-decoded -->

where ρ Q π denotes the unique stationary distribution of Q π . Let m := sup Q ∈P ,π ∈ Π m Q π &lt; ∞ and p := inf Q ∈P ,π ∈ Π p Q π &gt; 0 . Choose s ′ ∈ S such that ρ Q π ( s ′ ) ≥ 1 |S| . Let

<!-- formula-not-decoded -->

Then, for any s ∈ S ,

<!-- formula-not-decoded -->

Now, for a fixed s 0 ∈ S , consider the iteration

<!-- formula-not-decoded -->

where e is the all-ones vector. Combine the fact there exists a positive integer J such that for all Q ∈ P and any stationary deterministic policy π , there exists a state s ′ ∈ S , such that Q J π ( s, s ′ ) &gt; 0 , applying Theorem 8 in Wang et al. [45], ( ω t , v t ) converges to a solution ( g, v ) of

<!-- formula-not-decoded -->

which proves existence. Combining Theorem 7 in Wang et al. [45], we have that g ( s ) = g ∗ P for all s ∈ S and

<!-- formula-not-decoded -->

achieves the optimal average reward g ∗ P .

## B Uniform Ergodicity of the KL Uncertainty Set

In this section, we prove the uniform ergodic properties over P and ̂ P under Assumption 1 and 2. To achieve this, we establish the uniform Doeblin condition through a careful analysis of the Radon-Nikodym derivatives between perturbed and nominal transition kernels q s,a and p s,a . we propose some concepts in facilitating to bound the Radon-Nikodym derivative between derivative between perturbed kernel q s,a and nominal p s,a .

Proposition B.1. Suppose δ ≤ 1 8 m 2 ∨ p ∧ , then for all q s,a ∈ P s,a , the Radon-Nikodym derivative satisfies

<!-- formula-not-decoded -->

holds for all ( s, a ) ∈ S × A

Proof. Consider q s,a ∈ P s,a , then with KL-constraint, for any s ′ ∈ S , where p s,a ( s ′ ) &gt; 0 , we have:

<!-- formula-not-decoded -->

While the last inequality is derived by log-sum inequality. Let:

<!-- formula-not-decoded -->

where y ∈ [ p ∧ , 1 -p ∧ ] , since:

then

And (2) refers to the fact that

<!-- formula-not-decoded -->

The above functional dependence is optimal in polynomial. Hence, when t = 1 -1 2 m ∨ , we have:

<!-- formula-not-decoded -->

However, under the assumption that δ ≤ 1 8 m 2 ∨ p ∧ , the preceding inequality leads to a contradiction. As y ∈ [ p ∧ , 1 -p ∧ ] , let y = p s,a ( s ′ ) , we establish the uniform lower bound:

<!-- formula-not-decoded -->

This inequality holds uniformly across all:

- State-action pairs ( s, a ) ∈ S × A
- Next states s ′ ∈ supp( p s,a )

<!-- formula-not-decoded -->

And h ( y, y ) = 0 , we know for any fixed y , h ( x, y ) is convex with respect to x on x ∈ (0 , y ) . Since h (0 , y ) = log ( 1 1 -y ) ≥ log ( 1 1 -p ∧ ) ≥ p ∧ &gt; δ . By mean value theorem there exists a unique x ∗ ( y ) s.t. h ( x ∗ ( y ) , y ) = δ . Hence, define x ∗ ( y ) := min x ∈ (0 ,y ) { x : h ( x, y ) = δ } , and for any fixed t ∈ (0 , 1) , if x ∗ ( y ) &lt; ty , then:

<!-- formula-not-decoded -->

Here (1) refers to the h ( ty, y ) expansion at y = 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, the Radon-Nikodym derivative admits a uniform lower bound over the uncertainty set P s,a :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all ( s, a ) ∈ S × A .

Followed by the boundedness of the Radon-Nikodym derivative, we are able to show the uniform ergodic properties on the uncertainty P

Proposition B.2 (Restatement of the KL case in Proposition 4.2) . Suppose P is uniformly ergodic, and δ ≤ 1 8 m 2 ∨ p ∧ , then P = P ( D KL , δ ) is uniformly ergodic and for all Q ∈ P and π ∈ Π :

<!-- formula-not-decoded -->

where t minorize is from Assumption 1.

Proof. By Lemma A.3, since P is uniformly ergodic, then there exists an ( m π , p π ) pair, such that:

<!-- formula-not-decoded -->

For all Q ∈ P , by Proposition B.1, we have for all ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

Then, for all state pairs ( s 0 , s m π ) ∈ S × S , consider Q m π π , we have:

<!-- formula-not-decoded -->

The inequality (1) follows from:

<!-- formula-not-decoded -->

(1) holds. The result (2) is derived from the ( m π , p π ) -Doeblin condition satisfied by P π . This implies that for every policy π ∈ Π , the perturbed kernel Q π maintains a ( m π , p π 2 ) -Doeblin condition. Crucially, this conclusion holds uniformly across all policies in Π . Furthermore, the minorization time satisfies:

<!-- formula-not-decoded -->

where t minorize = max π ∈ Π t minorize ( P π ) by Assumption 1. Thus t minorize ( Q π ) is uniformly bounded over Q ∈ P and π ∈ Π :

<!-- formula-not-decoded -->

P is uniformly ergodic.

Building upon Proposition B.2, we establish that all perturbed transition kernels Q ∈ P , Q π satifies the ( m π , p π 2 ) -Doeblin condition, and P preserves uniformly ergodic given P is uniformly ergodic with appropriate adversarial power constraint. We further extends the uniform ergodicity to empirical kernels ̂ P π and empirical uncerainty sets ̂ P . Although these results are not essential for proving our main theorems, they provide valuable methodological insights for uniform ergodicity theory:

- (i) Offering a technical blueprint for extending classical ergodic theory to DR settings
- (ii) Laying the theoretical foundation for analyzing uncertainty sets in Markov models
- (iii) Opening new research directions for perturbation analysis of ergodic processes.

These findings may prove particularly useful for future studies in DR-MDP and related ares.

Lemma B.3. Suppose P is uniformly ergodic. When the sample size:

<!-- formula-not-decoded -->

the empirical nominal transition kernel ̂ P is also uniformly ergodic with:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. When the sample size satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then by Bernsteins' inequality and A.2:

<!-- formula-not-decoded -->

the probability of the event set Ω n, 1 2 m ∨ is bounded by:

<!-- formula-not-decoded -->

For any fixed ( s, a ) ∈ S × A , and d ∈ [0 , 1] on the event of Ω n,d ( p s,a ) , we have, for all s ′ ∈ supp( p s,a ) :

<!-- formula-not-decoded -->

on the event set Ω n, 1 2 m ∨ , and then:

Then, by Lemma A.3, for any π ∈ Π , there exists a ( m π , p π ) pair such that m π p π = t minorize ( P π ) . Consider the transition matrix ̂ P π , we have, for all ( s, s ′ ) ∈ S × S :

<!-- formula-not-decoded -->

which implies ̂ P π satisfies the ( m π , p π 2 ) -Doeblin Condition, and:

<!-- formula-not-decoded -->

̂ P is uniformly ergodic with probability 1 -β .

Combining Proposition B.2 with Lemma B.3, we establish that for all empirical transition kernels ̂ Q ∈ ̂ P , the minorization time satisfies

<!-- formula-not-decoded -->

This bound yields the following immediate Corollary.

Corollary B.4. Suppose the nominal transition kernel P is uniformly ergodic, and δ ≤ 1 16 m 2 ∨ p ∧ , then when the sample complexity

<!-- formula-not-decoded -->

the empirical uncertainty set ̂ P is uniformly ergodic and satisfies: following holds with probability 1 -β

<!-- formula-not-decoded -->

(B.6)

with probability 1 -β .

Proof. First, by Lemma A.3, for any π ∈ Π , there exists a ( m π , p π ) pair such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ p ∧ ≥ ( 1 -1 2 m ∨ ) p ∧ ≥ 1 2 p ∧ , and ̂ P π satisfies ( m π , p π 2 ) -Doeblin condition with probability 1 -β . Then, by Proposition B.2, as

<!-- formula-not-decoded -->

It implies for all ̂ Q ∈ ̂ P , ̂ Q π satisfies ( m π , p π 4 ) -Doeblin condition, and:

<!-- formula-not-decoded -->

̂ P is uniformly ergodic with probability 1 -β .

Then, consider by Lemma B.3, when

Theorem B.5. Under Assumptions 1 and when δ ≤ 1 16 m 2 ∨ p ∧ , when sample size satisfies:

<!-- formula-not-decoded -->

- P is uniformly ergodic, and for any Q ∈ P and π ∈ Π , the minorization time of Q π :

<!-- formula-not-decoded -->

then, with probability 1 .

- ̂ P is uniformly ergodic and for any π ∈ Π , the minorization time of ̂ P π :

<!-- formula-not-decoded -->

with probability 1 -β .

- ̂ P is uniformly ergodic, and for any ̂ Q ∈ ̂ P , the minorization time of ̂ Q π :

<!-- formula-not-decoded -->

with probability 1 -β .

Proof. The result follows by synthesizing three key components:

1. The uniform Doeblin condition for KL-constrained uncertainty set (Proposition B.2)
2. The Doeblin condition for empirical transition kernel (Lemma B.3)
3. The uniform Doeblin condition for empirical KL-constrained uncertainty set (Corollary B.4)

The combination of 1-3 yields the claimed uniform bounds through careful propagation of the minorization parameters across different uncertainty sets.

## C Properties of the Bellman Operator: KL-Case

In this section, we aim to bound the error between DR discounted Bellman operator (A.2) and the empirical DR discounted Bellman operator (A.4). In the DR setting, it is challenging to work with the primal formulation in the operators:

<!-- formula-not-decoded -->

To overcome this difficulty, we instead work with the dual formula by using the strong duality.

Lemma C.1 (Theorem 1 of Hu and Hong [15]) . For any ( s, a ) ∈ S × A , let P s,a be the uncertainty set centered at the nominal transition kernel p s,a . Then, for any δ &gt; 0 :

<!-- formula-not-decoded -->

for any V : S → R .

Since the reward and value function are bounded, directly apply Lemma C.1 to the r.h.s of Equation (3.3), V ∗ P , and ( g ∗ P , v ∗ P ) satisfied the following dual form of the optimal DR Bellman equation:

<!-- formula-not-decoded -->

Our analyses are inspired by the approach in Wang et al. [43]. To carry out our analysis, we first introduce some notation. As in Wang et al. [43], we denote the KL-dual functional under the nominal transition kernel p s,a :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At the same time, define the help measures as:

<!-- formula-not-decoded -->

By Definition (A.7), it is clear when d &lt; 1 , p s,a ( t ) ∼ p s,a holds for all ( s, a ) ∈ S × A and t ∈ [0 , 1] on Ω n,d .

We first introduce the auxiliary lemma that will be used useful in facilitating the later proof:

Lemma C.2. For any ( s, a ) ∈ S × A , and value function V 1 , V 2 ∈ R S , we have:

<!-- formula-not-decoded -->

Proof. For any q ∈ P s,a , we have:

then

Thus:

<!-- formula-not-decoded -->

Switch V 1 and V 2 , we have:

Then we conclude:

Proved

Then, we bound the error of empirical value function V π ̂ P and the true value function V π P with respect to the the Bellman operators (A.3) and (D.27) by the following lemma:

Lemma C.3. Let π be any policy, and V π P and V π ̂ P are the fixed points to the DR Bellman Operators (A.2) , and (A.4) , where V π P = T π γ ( V π P ) and V π ̂ P = ̂ T π γ ( V π ̂ P ) , then we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since q is a probability measure, by Hölder's inequality with | q | for all q ∈ P s,a

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. DR Bellman operators T π γ and ̂ T π γ are γ -contractions, i.e., for any two value functions V 1 , V 2 ∈ R S , we have:

<!-- formula-not-decoded -->

where the inequalities ( i ) , ( ii ) are concluded by Lemma C.2. Thus, we have:

<!-- formula-not-decoded -->

and

Proved.

Next, we aim to bound the approximation error ∥ ∥ ∥ ̂ T π γ ( V π P ) -T π γ ( V π P ) ∥ ∥ ∥ ∞ . Previous approaches relies on estimating via KL-dual functionals with optimal multipliers α ∗ ∈ [0 , δ -1 (1 -γ ) -1 ] . While this yields an bound of ˜ O ( δ -1 (1 -γ ) -1 ) , it ultimately leads to suboptimal O (1 /ε 4 ) sample complexity. Building on Wang et al. [43]'s breakthrough in achieving δ -independent bounds through KL-dual analysis, we make two key advances:

1. Targeted Value Function Analysis: Instead of considering the entire value function space [0 , (1 -γ ) -1 ] S , we restrict analysis to V π P specifically. This allows us to replace the (1 -γ ) -1 dependence with the span semi-norm of V π P .
2. Error Rate Improvement: Combining the Span( V π P ) dependent error bound with Proposition A.5, we improve the bound from ˜ O ( δ -1 (1 -γ ) -1 ) to:

<!-- formula-not-decoded -->

As shown in Section D, these refinements ultimately yield the improved smaple complexity of ˜ O ( | S || A | t 2 minorize p -1 ∧ ε -2 ) .

Lemma C.4. Let p 1 , p 2 , p ∈ ∆( S ) s.t. p 1 , p 2 ≪ p . Define ∆ := p 1 -p 2 . Then, for any V : S → R and j ∈ (0 , 1] ,

<!-- formula-not-decoded -->

Proof. First we note that for any k ∈ R , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, if we shift the value function V ′ = V -inf s ∈ S V ( s ) -Span( V ) 2 , where ∥ V ′ ∥ ∞ = 1 2 Span( V ) , then, it is equivalent to show:

<!-- formula-not-decoded -->

Thus, we only need to show

As ∆[1] = 0 :

<!-- formula-not-decoded -->

Then:

<!-- formula-not-decoded -->

Consider the first term ∥ ∥ ∥ ∥ α j ( e -( V + ∥ V ∥ ∞ ) /α -1 ) e -( V + ∥ V ∥ ∞ ) /α ∥ ∥ ∥ ∥ ∞ , denote

<!-- formula-not-decoded -->

Taking the derivative of f ( α ) , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

i.e., the bound with respect to l ∞ of V and replace ∥·∥ ∞ by 1 2 Span( · ) . WLOG, we assume inf s ∈ S V ( s ) = 0 . Then for a fixed c &gt; 0 , decompose the domain of α ∈ [0 , c ∥ V ∥ ∞ ] ∪ ( c ∥ V ∥ ∞ , ∞ ) = K 1 ∪ K 2 , we have:

<!-- formula-not-decoded -->

For K 1 ( c ) , we have

<!-- formula-not-decoded -->

For K 2 ( c ) , the condition is more complicated

<!-- formula-not-decoded -->

replace t = 2 ∥ V ∥ ∞ , and c = j (2 ∥ V ∥ ) j -

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

combine the fact: lim t → 0 ∂ α f ( α ) = 0 . It implies ∂ α f ( α ) ≤ 0 , and further f ( α ) is decreasing with respect to α . Therefore, over K 2 :

<!-- formula-not-decoded -->

Combine the previous result, we get the result where:

<!-- formula-not-decoded -->

and when j ∈ [0 , 1]

Then:

<!-- formula-not-decoded -->

In select c = 2 log 2 , the minimax optimality is achieved, we have:

<!-- formula-not-decoded -->

As the the above result is invariant under the constant shift of V , let V ′ = V -inf s ∈ S V ( s ) -Span( V ) 2 , we have:

<!-- formula-not-decoded -->

Proved

Lemma C.5. For any value function V with span semi-norm Span( V ) :

- If Span( V ) = 0 , the optimal Lagrange multiplier α ∗ = 0 , and for all q s,a ∈ P s,a , q s,a is a worst-case measure.

̸

- If Span( V ) = 0 , the optimal Lagrange multiplier α ∗ &gt; 0 , and:

is a worst-case measure

<!-- formula-not-decoded -->

Proof. From Si et al. [34], for optimal Lagrange multiplier α ∗ , it sufficient to consider α ∈ [0 , δ -1 ∥ V ∥ L ∞ ( p s,a ) ] .

When Span( V ) = 0 , it implies V is a constant over supp( p s,a ) , and:

<!-- formula-not-decoded -->

Thus, as f ( p s,a , V, α ∗ ) = sup α ≥ 0 f ( p s,a , V, α ) , α ∗ = 0 , and for all q s,a ∈ P s,a , q s,a is a worst-case measure since V is a constant function on supp( p s,a ) . When Span( V ) = 0 , α ∗ satisfies:

<!-- formula-not-decoded -->

̸

As α ∗ is the optimal Lagrange multiplier and f is differentiable, consider the first-order partial derivative with respect to α :

<!-- formula-not-decoded -->

and lim α → 0 ∂ α f &gt; 0 . As α ∗ is the optimal multiplier, ∂ α f ( p s,a , V, α ) | α = α ∗ = 0 , and

<!-- formula-not-decoded -->

Further:

<!-- formula-not-decoded -->

Define the measure:

Thus:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus f ( p s,a , V, α ) is concave for α &gt; 0 . α ∗ is the unique optimal multiplier where α ∗ &gt; 0 and p ∗ s,a satisfies:

<!-- formula-not-decoded -->

Therefore, we show p ∗ s,a is a worst-case measure.

Lemma C.6. Let p s,a be the nominal transition kernel, and ̂ p s,a be the empirical transition kernel, then the below inequality holds:

<!-- formula-not-decoded -->

on Ω n,d , when d ≤ 2

<!-- formula-not-decoded -->

Proof. Recall the general KL-dual functional under the probability measure p s,a , value function V , and parameter α is:

<!-- formula-not-decoded -->

Then:

<!-- formula-not-decoded -->

Further, consider the difference of | f ̂ p s,a ,V,α -f p s,a ,V,α | :

<!-- formula-not-decoded -->

On Ω n,d where d ≤ 1 2 , we have for all τ ∈ [0 , 1] , and s ′ ∈ supp( p s,a ) :

<!-- formula-not-decoded -->

Apply Lemma C.4, we have:

<!-- formula-not-decoded -->

Where inequality ( i ) is by Lemma C.4, and the inequality ( ii ) is by Equation (C.24). The difference of KL-dual functional is bounded by:

<!-- formula-not-decoded -->

on Ω n,d when d ≤ 1 2

<!-- formula-not-decoded -->

Lemma C.7. When n ≥ 32 p -1 ∧ log(2 | S | 2 | A | /β | ) , then for any π ∈ Π , the l ∞ -error of the emprical DR Bellman operator ̂ T π γ and the DR Bellman operator T π γ can be bounded by:

<!-- formula-not-decoded -->

with probability 1 -β , where P π is the transition kernel induced by controlled transition kernel P and policy π .

Proof. First, by Bernstein's inequality, when:

<!-- formula-not-decoded -->

Then:

<!-- formula-not-decoded -->

with probability at least 1 -β for all ( s, a ) ∈ S × A and s ′ ∈ supp( p s,a ) , thus, let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then d ≤ 1 2 , and:

Then we consider the difference of the Bellman operators for particular value function V π P , on Ω n,d

<!-- formula-not-decoded -->

on Ω n,d , where ( i ) is derived by C.6. Combine Proposition A.5, and Lemma A.3, when P π is ( m π , p π ) -Doeblin with m π /p π = t minorize ( P π ) , and Span( V π P ) ≤ 3 m π /p π , thus, we have:

<!-- formula-not-decoded -->

withe probability 1 -β . Let:

<!-- formula-not-decoded -->

Then with probability at least 1 -β , for any π ∈ Π , the following bound holds:

<!-- formula-not-decoded -->

Proved.

## D Sample Complexity Analysis: KL Uncertainty Set

In this section, we prove the sample complexity bound as shown in Theorem 4.4, and Theorem 4.5

## D.1 DR-DMDP under KL Uncertainty Set

Lemma D.1. Let ̂ π ∗ = arg max π ∈ Π V π ̂ P , then the following inequality holds:

<!-- formula-not-decoded -->

Proof. The left direction of the inequality is trivial. For the right one inequality, we have:

<!-- formula-not-decoded -->

Proved.

Theorem D.2 (Restatement of Theorem 4.3) . Suppose Assumptions 1, and 2 are in force. Then, for any n ≥ 32 p -1 ∧ log(2 | S | 2 | A | /β ) , the policy ̂ π ∗ and value function V ∗ ̂ P returned by Algorithm 1 satisfies:

<!-- formula-not-decoded -->

with probability 1 -β . Consequently, the sample complexity to achieve ε -optimal policy and value function with probability 1 -β is:

<!-- formula-not-decoded -->

where c = 2 · 72 2 , and c = 2 · 36 2 respectively.

Proof. For any π ∈ Π , as V π P is the fixed point to the DR Bellman operator (A.2), where V π P = T π γ ( V π P ) , by Lemma C.7, when:

<!-- formula-not-decoded -->

then, with probability 1 -β , for all π ∈ Π :

<!-- formula-not-decoded -->

where ( i ) is derived by Lemma C.3, and ( ii ) relies on Proposition B.2 where t minorize ( Q π ) is uniformly bounded for all Q ∈ P and π ∈ Π . We conclude, when

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have with probability 1 -β . Since for value function evaluation:

<!-- formula-not-decoded -->

holds for the same sample complexity condition and high probability guarantee, we prove the Theorem.

Remark D.3 . In the proof of Theorem D.2, we establish a high probability guarantee for the uniform value function approximation error: max π ∈ Π ∥ ∥ ∥ V π ̂ P -V π P ∥ ∥ ∥ ∞ ≤ ˜ ( Ot minorize (1 -γ ) -1 n -1 / 2 ) . Crucially, this uniform bound simultaneously controls both:

- The policy gap: V ∗ P -V ̂ π ∗ P
- The value function approximation error: ∥ ∥ ∥ V ∗ P -V ∗ P ∥ ∥ ∥

<!-- formula-not-decoded -->

via the relationship:

<!-- formula-not-decoded -->

This simultaneous control ensures the final error guaranee in (D.2) holds without requiring additional division of the confidence parameter β . The key observation is that our uniform concentration bound on value functions automatically propagates to both policy selection and value estimation errors.

## D.2 Reduction to DR-DMDP Approach under KL Uncertainty Set

Theorem D.4 (Restatement of Theorem 4.4) . Suppose Assumption 1, and 2 are in force, then for any

<!-- formula-not-decoded -->

the policy ̂ π ∗ and value function V ∗ ̂ P √ n returned by Algorithm 2 satisfies:

<!-- formula-not-decoded -->

with probability 1 -β . Hence, the sample complexity of achieving an ε -error in either optimal policy or value estimation is

<!-- formula-not-decoded -->

where c = 2 · 96 2 , and c = 2 · 48 2 repectively.

Proof. Initially, let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the ̂ π ∗ policy evaluation, we have:

0

≤

g

∗

π

∗

P

-

g

̂

P

<!-- formula-not-decoded -->

Then, by definition:

<!-- formula-not-decoded -->

Then, we have:

<!-- formula-not-decoded -->

For V ∗ ̂ P √ n value evaluation, we have:

<!-- formula-not-decoded -->

Combine (D.10), we have:

<!-- formula-not-decoded -->

Therefore, we have:

<!-- formula-not-decoded -->

We consider the error bound term by term, for simplicity, we denote γ = 1 -γ .

Step 1: bounding and hence:

π

π

P P ∞ When 0 ≤ g π P ( s ) -(1 -γ ) V π P ( s ) , then, for any ε &gt; 0 , there exist an P ε ∈ P such that

γ

)

V

∥

.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking limit as ε → 0 , we conclude when 0 ≤ g π P ( s ) -(1 -γ ) V π P ( s ) :

<!-- formula-not-decoded -->

Similarly, when 0 ≥ g π P ( s ) -(1 -γ ) V π P ( s ) , let consider P ε such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then we have:

Taking limit as ε → 0 , we conclude when 0 ≥ g π P ( s ) -(1 -γ ) V π P ( s ) :

<!-- formula-not-decoded -->

And thus:

And further:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, by Lemma A.6, over the nominal uncertainty set P :

<!-- formula-not-decoded -->

By Proposition B.2, t minorize ( Q π ) is uniformly bounded on P × Π by 2 t minorize , we have:

<!-- formula-not-decoded -->

As the above inequality holds for all π ∈ Π , further, plug back γ = 1 -1 √ n , we conclude that:

<!-- formula-not-decoded -->

∥

g

-

(1

-

with probability 1 .

Step 2: bounding ∥ ∥ ∥ V π ̂ P -V π P ∥ ∥ ∥ ∞ .

By the definition of V π ̂ P and V π P , we know V π P and V π ̂ P are the solutions to the DR Bellman equations V π P = T π γ ( V π P ) , V π ̂ P = T π γ ( V π ̂ P ) thus, by Lemma C.3:

<!-- formula-not-decoded -->

Combine Lemma C.7, and Proposition 4.2 we have when:

<!-- formula-not-decoded -->

then, with probability at least 1 -β , for all π ∈ Π .

<!-- formula-not-decoded -->

Further, with the choice of γ = 1 -1 √ n , we conclude when n ≥ 32 p ∧ log 2 | S | 2 | A | β , then:

<!-- formula-not-decoded -->

with probability 1 -β .

Step 3: combining the previous results. For π ∗ policy evaluation:

<!-- formula-not-decoded -->

wth probability 1 -β . Where the last inequality uses the trival bounds where p ∧ ≤ 1 2 , and | S | , | A | ≥ 1 . Thus, when n = 2 · 96 2 t 2 minorize p ∧ ε 2 , the policy ̂ π ∗ satisfies:

<!-- formula-not-decoded -->

simultaneously with probability 1 -β . The sample complexity of achieving an ε -error in either optimal policy or value value estimation is:

<!-- formula-not-decoded -->

where c = 2 · 96 2 and c = 2 · 48 2 repectively. Recall the minorization time is equivalent to mixing time, the total sample used is:

<!-- formula-not-decoded -->

Proved.

Remark D.5 . Notice the error guarantee relies on the relative error between ̂ p s,a and p s,a is less or equal thant 1 2 , hence P (Ω c n, 1 2 ) &lt; β , thus, the lowerbound of n is ˜ Ω(1 / p ∧ ) .

## D.3 Anchored DR-AMDP Approach under KL Uncertainty Set

This section analyzes the anchored algorithm and establishes a sample complexity upper bound for the anchored DR-AMDP Algorithm 3. Key to our analysis is the insight that the anchored DR-AMDP can be identified with a DR-DMDP with discount factor γ = 1 -ξ where ξ is the anchoring parameter.

Fix s 0 ∈ S , recall that

<!-- formula-not-decoded -->

We consider the anchored DR average Bellman equation

<!-- formula-not-decoded -->

Lemma D.6. Assume that P is uniformly ergodic. Let V ∗ P be the solution to the DR Bellman equation with discounted parameter γ = 1 -ξ :

<!-- formula-not-decoded -->

Then, ( g, v ) = ( ξV ∗ P ( s 0 ) , V ∗ P ) is a solution pair to the anchored DR average Bellman equation (D.23) . Moreover, for all solution pairs ( g ′ , v ′ ) to (D.23) , g ′ ≡ ξV ∗ P ( s 0 ) .

Proof. As P is uniformly ergodic, for all Q ∈ P , Q is uniformly ergodic. Thus, for Q ∈ P , by Lemma A.3, for any π ∈ Π there exists an ( m π , p π ) pair such that m π p π = t minorize ( Q π ) . As for all Q ∈ P , π ∈ Π , and ( s 0 , s m π ) ∈ S × S :

<!-- formula-not-decoded -->

Since Q π satisfies ( m π , p π ) -Doeblin condition, it follows that Q satifies ( m π , (1 -ξ ) m π p π ) -Doeblin condition. Consequently, Q is uniformly ergodic, which further implies that P is uniformly ergodic. Given the uniform ergodicity of P , by Proposition 3.6, if ( g, v ) is a pair of the solutions to the anchored DR average Bellman equation:

<!-- formula-not-decoded -->

Then g = g ∗ P is unique. Next, we show ( g, v ) = ( ξV ∗ P ( s 0 ) , V ∗ P ) is the solution to the anchored DR average Bellman equation:

<!-- formula-not-decoded -->

Thus, we show ( ξV ∗ P ( s 0 ) , V ∗ P ) is a pair of solution to Equation (D.23), combine Lemma 3.6, we know g ≡ ξV ∗ P ( s 0 ) is unique, where V ∗ P is the optimal value function of DR discounted Bellman operator (A.3) with parameter 1 -ξ .

The above Lemma D.6 holds for any uncertainty set P , hence ( g, v ) = ( ξV ∗ P ( s 0 ) , V ∗ P ) and ( g, v ) = ( ξV ∗ ̂ P ( s 0 ) , V ∗ ̂ P ) are the solutions to the anchored DR average Bellman equation (D.26) and empirical anchored DR average Bellman equation (D.27) respectively:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, the equivalent also holds for the DR policy equations:

Lemma D.7. If P is uniformly ergodic, let V π P be the solution to the DR discounted policy equation with γ = 1 -ξ :

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

is a solution pair of the anchored DR average policy Bellman equation:

<!-- formula-not-decoded -->

Moreover, for all solution pairs ( g ′ , v ′ ) to (D.29) , g ′ ≡ ξV π P ( s 0 ) .

Proof. The proof is similar with Lemma D.6. As g = g π P is unique, we only need to show (D.28) is a solution pair:

<!-- formula-not-decoded -->

Thus, by Theorem 6 in Wang et al. [44], we show ( ξV π P ( s 0 ) , V π P ) is a pair of solution to Equation (D.29), where g = ξV π P ( s 0 ) is unique.

With these auxiliary result, we present our proof to the following main result.

Theorem D.8 (Restatement of Theorem 4.5) . Suppose Algorithm 3 is in force. Then for any:

<!-- formula-not-decoded -->

The output policy ̂ π ∗ and approximate average value function g ∗ ̂ P satisfies:

<!-- formula-not-decoded -->

with probability at least 1 -β . Hence, the sample complexity of achieving an ε -error in both optimal policy and value estimation is

<!-- formula-not-decoded -->

where c = 2 · 96 2 and c = 2 · 48 2 repectively.

Proof. For policy evaluation, consider policy ̂ π ∗ returned by Algorithm 3, by Lemma 3.6, we know ̂ π ∗ is an optimal policy for the anchored empirical uncertainty set ̂ P :

<!-- formula-not-decoded -->

further

<!-- formula-not-decoded -->

For average value function evaluation g ∗ ̂ P :

<!-- formula-not-decoded -->

Then, we analysis ∥ ∥ ∥ g π ̂ P -g π P ∥ ∥ ∥ ∞ , by Lemma D.7, we know g π ̂ P = ξV π ̂ P ( s 0 ) , then:

<!-- formula-not-decoded -->

Since for all Q ∈ P , by Proposition B.1, we know p s,a ≪ q s,a for all ( s, a ) ∈ S × A . Then by Lemma A.4, for all Q ∈ P , Q is also uniformly ergodic, and Span( g π P ) = 0 . Thus:

<!-- formula-not-decoded -->

With the choice ξ = 1 √ n , by Lemma C.3 and Lemma C.7, when n ≥ 32 p ∧ log 2 | S | 2 | A | β , with probability 1 -β , for all π ∈ Π :

<!-- formula-not-decoded -->

Thus, combine the results:

<!-- formula-not-decoded -->

with probability 1 -β . And when:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability 1 -β , ̂ π ∗ is an ε -optimal policy. Simultaneously,

<!-- formula-not-decoded -->

we get:

with probability 1 -β . And when:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability 1 -β , g ∗ ̂ P is an ε -optimal value function.

## E Uniform Ergodicity of the f k Uncertainty Set

In this section, we prove the technique results similar in f k -divergence constraints given the Assumption 1 and 2. Like what we have in KL-diverence, we first bound the Radon-Nikodym derivative between perturbed kernel q s,a and nominal p s,a .

Lemma E.1. Suppose δ ≤ 1 max { 8 , 4 k } m 2 ∨ p ∧ , then for all q s,a ∈ P s,a , the Radon-Nikodym derivative of q s,a satisfies:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we get:

for all ( s, a ) ∈ S × A

Proof. We prove the Lemma by contradiction. For any ( s, a ) ∈ S × A , consider q s,a ∈ P s,a , suppose there exists s ′ ∈ S , such that

Then for any q s,a ∈ P s,a , we have:

<!-- formula-not-decoded -->

Define the helper function g k ( t ) := 1 k ( t -1) 2 , when k ≥ 2 , we have:

<!-- formula-not-decoded -->

The nominator is let

Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We conclude that p k ( t ) ≥ 0 on t ∈ [0 , 1] , and thus:

<!-- formula-not-decoded -->

Hence, when t = r &lt; 1 -1 2 m ∨ :

<!-- formula-not-decoded -->

However, the above inequality contradict to (E.1) where f k ( r ) p ∧ ≤ δ . For the case when k ∈ (1 , 2) , consider the function:

<!-- formula-not-decoded -->

It is easy to see that lim t → 1 -h k ( t ) = 1 2 , and h k (0) = 1 k . Hence h k (0) ≥ lim t → 1 -h k ( t ) . Then, the derivative:

<!-- formula-not-decoded -->

Denote the nominator as:

then we have:

<!-- formula-not-decoded -->

When 1 &lt; k &lt; 2 , the second order derivative d 2 t q k ( t ) &gt; 0 on t ∈ (0 , 1) . dq k ( t ) is monotone increasing, and as d t q k ( t ) | t =1 = 0 , we have d t q k ( t ) ≤ 0 on t ∈ (0 , 1] , which sequentially implies q k ( t ) is monotone decreasing,

<!-- formula-not-decoded -->

And further d t h k ( t ) ≤ 0 , h k ( t ) is monotone decreasing from 1 k to 1 2 . Thus, for 1 &lt; k &lt; 2 :

<!-- formula-not-decoded -->

Combine the previous results, we have, when t = r &lt; 1 -1 2 m ∨

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently, we prove, for all k ∈ (1 , ∞ ) when δ ≤ 1 · max { 8 , 4 k } m 2 ∨ p ∧ , the Radon-Nikodym derivative of between any q s,a ∈ P s,a and p s,a satisfies:

<!-- formula-not-decoded -->

which contradict to

Proved.

<!-- formula-not-decoded -->

Further, similiar with Proposition B.2 we have when δ ≤ 1 4 max { 2 ,k } m 2 ∨ p ∧ , P is uniformly ergodic with:

<!-- formula-not-decoded -->

The idea is as same as the KL-divergence case, as under Assumption 2, the Radon-Nikodym derivative is uniform bounded by 1 -1 2 m ∨ over P , thus, we have for any Q ∈ P , Q m π π ( s, s ′ ) ≥ p π 2 P m π π ( s, s ′ ) for all ( s, s ′ ) ∈ S × S .

Proposition E.2 (Restatement of Proposition 4.2) . Suppose P is uniformly ergodic, and δ ≤ 1 max { 8 , 4 k } m 2 ∨ p ∧ , then P = P ( D f k , δ ) is uniformly ergodic and for all Q ∈ P , and π ∈ Π :

<!-- formula-not-decoded -->

where t minorize is from Assumption 1.

Proof. By Lemma A.3, since P is uniformly ergodic, then there exists an ( m π , p π ) pair, such that:

<!-- formula-not-decoded -->

For all Q ∈ P , by Lemma E.1, we have for all ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

Then, for all state pairs ( s 0 , s m π ) ∈ S × S , consider Q m π π , we have:

<!-- formula-not-decoded -->

The proof of the above inequality is the same as in (B.4). This implies that for every policy π ∈ Π , the perturbed kernel Q π maintains a ( m π , p π 2 ) -Doeblin condition. Furthermore, the minorization time satisfies:

<!-- formula-not-decoded -->

where t minorize = max π ∈ Π t minorize ( P π ) by Assumption 1. Thus t minorize ( Q π ) is uniformly bounded over Q ∈ P and π ∈ Π :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

P is uniformly ergodic.

We furthr establish uniform ergodicity properties for both the empirical transition kernel ̂ P and its empirical uncertainty set ̂ P , serving as a probabilistic counterpart to Theorem B.5. Whle not directly impacting our sample complexity results, this analysis reveals that when the robustness parameter satisfies:

both ̂ P and ̂ P maintain uniform ergodicity with high probability.

Corollary E.3. Under Assumption 1 and δ ≤ 1 max { 16 , 8 k } m 2 ∨ p ∧ , when the smaple size satisfies:

<!-- formula-not-decoded -->

- P is uniformly ergodic, for any Q ∈ P and π ∈ Π , the minorization time of Q π :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then:

with probability 1 .

- ̂ P is uniformly ergodic and for any π ∈ Π , the minorization time of ̂ P π :

<!-- formula-not-decoded -->

with probability 1 -β .

- ̂ P is uniformly ergodic, for any ̂ Q ∈ ̂ P and π ∈ Π , the minorization time of ̂ Q π :

<!-- formula-not-decoded -->

with probability 1 -β .

Proof. Since P is uniformly ergodic, by Lemma A.3, for any π ∈ Π , there exists an ( m π , p π ) pair such that m π p π = t minorize ( P π ) . First, for P , by Lemma E.1, for all Q ∈ P , we have:

<!-- formula-not-decoded -->

and there exists an ψ ∈ ∆( S ) , for any ( s 0 , s m π ) ∈ S × A :

<!-- formula-not-decoded -->

which implies Q π satisifes ( m π , p π 2 ) -Doeblin condition with probability 1 . And further, we have

<!-- formula-not-decoded -->

And P is uniformly ergodic.

Second, for empirical kernel ̂ P , by Lemma B.3, we have when

<!-- formula-not-decoded -->

empiricla kernel ̂ P π satisifes ( m π , p π 2 ) -Doeblin condition, and ̂ P is uniformly ergodic with probability 1 -β .

Third, since when n ≥ 32 m 2 ∨ p ∧ log 2 | S | 2 | A | β ,

<!-- formula-not-decoded -->

On Ω n, 1 2 m ∨ , ̂ p ∧ ≥ ( 1 -1 2 m ∨ ) p ∧ ≥ 1 2 p ∧ . By Lemma E.1

<!-- formula-not-decoded -->

It implies fo all ̂ Q ∈ ̂ P , ̂ Q π satisifes ( m π , 1 2 ( p π 2 ) ) -Doeblin condition, ̂ P is uniformly ergodic and

<!-- formula-not-decoded -->

with probability 1 -β .

## F Properties of the Bellman Operator: f k -Case

In this section, we target to bound the error between DR discounted Bellman opertaor and empirical DR discounted Bellman operator under f k -divergnce. Similar to the KL-case. We override the notations for the f k -case, and introduce the f k -duality

Lemma F.1 (Lemma 1 of Duchi and Namkoong [9]) . For any ( s, a ) ∈ S × A , let P s,a be the f k -uncertainty set centered at the nominal transition kernel p s,a . Then, for any δ &gt; 0 , let k ∗ = k k -1 :

<!-- formula-not-decoded -->

where c k ( δ ) = (1 + k ( k -1) δ ) 1 k , ( · ) + = max {· , 0 } and V : S → R is the value function.

we denote the f k -dual functional under the nominal transition kernel p s,a :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At the same time, we follow the auxiliary measures and function used in KL-case:

<!-- formula-not-decoded -->

When d &lt; 1 , p s,a ( t ) ∼ p s,a holds for all ( s, a ) ∈ S × A and t ∈ [0 , 1] on Ω n,d .

Lemma F.2. For any δ , the supremum of f ( p s,a , V, α ) is achieved at α ∗ ≥ ess inf p s,a V . If α ∗ &gt; ess inf p s,a V , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where p ∗ s,a defined as below:

is a worst-case measure. When α ∗ = ess inf p s,a V , the worst-case measure is given as:

where U = { s ′ : V ( s ′ ) = ess inf p s,a V } .

<!-- formula-not-decoded -->

Proof. A directly consequence for f ( p s,a , V, α ) is

<!-- formula-not-decoded -->

Thus, f is monotone increasing as α &lt; ess inf p s,a V , thus, the supremum of f ( p s,a , V, α ) is achieved at α ∗ ≥ ess inf p s,a V . The first order derivative of f ( p s,a , V, α ) with respect to α is given as:

<!-- formula-not-decoded -->

By Proposition 1 of Duchi and Namkoong [9], the dual form of f ( p s,a , V, α ) is concave with respect to α , and the supremum is achieved at α ∗ where ∂ α f ( p s,a , V, α ) | α = α ∗ = 0 . Since:

<!-- formula-not-decoded -->

By the first order condition, we have:

<!-- formula-not-decoded -->

then

Whiche implies:

<!-- formula-not-decoded -->

Then, to show p ∗ s,a is a worst-case measure, it is sufficient to show p ∗ s,a [ V ] = f ( p s,a , V, α ∗ ) and D f k ( p ∗ s,a || p s,a ) = δ . We have:

<!-- formula-not-decoded -->

Where ( i ) is derived by Equation (F.7). Moreover, by the definition of f k -divergence we have:

<!-- formula-not-decoded -->

Here we proved that when α ∗ &gt; ess inf p s,a V :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

p ∗ s,a is a worst-case measure. Further we show when α ∗ = ess inf p s,a V , p ∗ s,a defined below is a worst-case measure:

̸

✶ where U = { s ′ : V ( s ′ ) = ess inf p s,a V } . As p ∗ s,a ( V ) = ess inf p s,a V , then we only need to show D f k ( p ∗ s,a ∥ p s,a ) ≤ δ . To show this, we divide V into two cases. First, if V is a constant function on supp( p s,a ) , then U = S , then for all q s,a ∈ P s,a , q s,a [ V ] = ess inf p s,a V , and p ∗ s,a = p s,a , p s,a ∈ P s,a is a worst-case measure. Second, if V is not a constnat, then S \ U = ∅ , observe that, by continuity, there exists an ϵ 0 &gt; 0 , such that for all 0 &lt; ϵ ≤ ϵ 0 ,

<!-- formula-not-decoded -->

At the same time, as f ( p s,a , V, α ∗ ) = sup α f ( p s,a , V, α ) , then

<!-- formula-not-decoded -->

which implies:

<!-- formula-not-decoded -->

Then, we can compute the f k -divergence as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Where ( i ) follows by (F.11). Thus, we proved the Lemma.

<!-- formula-not-decoded -->

Proof. First, if V is essentially the constant, then Span( V ) = 0 , and

<!-- formula-not-decoded -->

And hence α ∗ ≥ ∥ V ∥ L ∞ ( p s,a ) .

When V is not essentially the constant, let U = { s ′ : V ( s ′ ) = ess inf p s,a V } , and S \ U = ∅ , hence:

Recall Lemma F.2, the worst-case measure is given as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence:

Inequality ( i ) is derived by 1 (1 -x ) c -1 ≥ x c when c &gt; 0 . However, the above result contradict to the assumption where δ ≤ 1 2 k p ∧ . Thus, we conclude α ∗ &gt; ∥ V ∥ L ∞ ( p s,a ) .

<!-- formula-not-decoded -->

holds. Then, we show the following Lemma:

<!-- formula-not-decoded -->

̸

Lemma F.4. Let p 1 , p 2 , p ∈ ∆( S ) s.t. p 1 , p 2 ≪ p . Define ∆ := p 1 -p 2 . Then, for any V : S → R and k ∗ &gt; 1 :

<!-- formula-not-decoded -->

Proof. Notice that for all c ∈ R :

<!-- formula-not-decoded -->

let c = ess inf p V and V ′ = V -c , then ∥ V ′ ∥ L ∞ ( p ) = Span( V ) , and we have:

<!-- formula-not-decoded -->

First, consider the case where α &gt; 2 k ∗ Span( V ) , then, for any s ∈ S , and V ′ ( s ) &gt; 0 , we have:

<!-- formula-not-decoded -->

And when V ′ ( s ) = 0 , ( α -V ′ ( s )) k ∗ -1 ≥ α k ∗ -1 2 holds trivially. Further, with tayler expansion, there exists a ξ where ξ ( s ) ∈ ( 0 , Span( V ) )

α , such that:

<!-- formula-not-decoded -->

Then, we can derive that:

<!-- formula-not-decoded -->

The equality ( i ) follows from the property that ∆[ c ] = 0 for any constant function c , since the difference between two measurres ∆ annihilates constants. The inequality ( ii ) is obtained by applying the condition α ≥ 2 k ∗ Span( V ′ ) , which ensures sufficient regularization. Finally, ( iii ) emerges from the fundamental constraint k ∗ &gt; 1 in our parameter condition. ∗

Second, we consider the case where Span( V ) ≤ α ≤ 2 k Span( V ) , actually, the the above result holds trivally:

<!-- formula-not-decoded -->

Combine the previous two cases, we derived the result:

<!-- formula-not-decoded -->

Proved

To establish Lemma F.6 bounding the dual functional difference, we build on Lemma F.4. While this result is analogous to Lemma C.7 for the KL-divergence case, the analysis for f k -divergence requires first applying the Envelope Theorem to characterize the variational behavior of the dual optimization problem.

Lemma F.5 (Envelope Theorem, Corollary 3 of Milgrom and Segal [26]) . Denote V as:

<!-- formula-not-decoded -->

Where X is a convex set in a linear space and f : X × [0 , 1] → R is a concave function. Also suppose that t 0 , and that there is some x ∗ ∈ X ∗ ( t 0 ) such that d t f ( x ∗ , t 0 ) exists. Then V is differentiable at t 0 and

<!-- formula-not-decoded -->

Lemma F.6. Let p s,a be the nominal transition kernel, and ̂ p s,a be the empirical transition kernel, when δ ≤ 1 max { 8 , 4 k } m 2 ∨ p ∧ , then the below inequality holds

<!-- formula-not-decoded -->

on Ω n,d , when d ≤ 1 2

Proof. Since

<!-- formula-not-decoded -->

by Lemma F.3, we have α ∗ &gt; ∥ V ∥ L ∞ ( p s,a ) , thus, we only need to consider the case where α ≥ ∥ V ∥ L ∞ ( p s,a ) , then recall

<!-- formula-not-decoded -->

is concave with respect to α , then denote G ( t ) and α ∗ ( t ) we have:

<!-- formula-not-decoded -->

Combine previous results, we have:

<!-- formula-not-decoded -->

Then, by Lemma F.5, we have, as G ( t ) is differentiable on [0 , 1] , then:

<!-- formula-not-decoded -->

Then we have:

<!-- formula-not-decoded -->

Since α ∗ ( t ) is the optimal multiplier for sup α ≥∥ V ∥ L ∞ ( ps,a ) g s,a ( τ, α ) , Using Lemma F.2, we have, for all t ∈ [0 , 1] :

<!-- formula-not-decoded -->

Thus:

<!-- formula-not-decoded -->

Using the fact α ∗ ≥ ∥ V ∥ L ∞ ( p s,a ) by the formula where k ∗ = k k -1 , we know k &gt; 1 , then, apply Lemma F.4,

<!-- formula-not-decoded -->

Further, on the events set Ω n,d , we have, when d ≤ 1 2 :

<!-- formula-not-decoded -->

We derived the result:

Proved.

Lemma F.7. When n ≥ 32 p -1 ∧ log(2 | S | 2 | A | /β ) , then for any π ∈ Π , the l ∞ -error of the empirical DR Bellmam operator ̂ T π γ and the DR Bellman operator ̂ T π γ can be bounded as:

<!-- formula-not-decoded -->

with probability at least 1 -β , where P π is the transition kernel induced by controlled transition kernel P and policy π .

<!-- formula-not-decoded -->

Proof. By Union bound and Bernstein's inequality, we have, when:

<!-- formula-not-decoded -->

the relative error of ̂ p s,a ( s ′ ) could be bounded as:

<!-- formula-not-decoded -->

Thus, let d = √ 8 n p ∧ log 2 | S | 2 | A | β ≤ 1 2 , we have:

<!-- formula-not-decoded -->

Then, on Ω n,d , by Lemma F.6, we could bound the error as:

<!-- formula-not-decoded -->

The inequality ( i ) is derived by Lemma F.4. Since Lemma A.3, there exists an ( m π , p π ) pair such that m π /p π = t minorize ( P π ) , combine Proposition A.5 , when P π is ( m π , p π ) -Doeblin, Span( V π P ) ≤ 3 m π /p π , we have: when

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then with probability 1 -β .

## G Sample Complexity Analysis: f k Uncertainty Set

We proceed with the analysis of the Algorithm1, 2 and 3 in the f k -divergence case.

## G.1 DR-DMDP under f k Uncertainty Set

Building upon these analytical foundations, we derive the following sample complexity bound for DR-MDPs with f k -divergence uncertainty sets:

Theorem G.1 (Restatement of Theorem 4.4) . Suppose Assumptions 1, and 2 are in force. Then for any n ≥ 32 p -1 ∧ log(2 | S | 2 | A | /β ) , the policy ̂ π ∗ and value function V ∗ ̂ P returned by Algorithm 1

satisfies:

<!-- formula-not-decoded -->

with probability 1 -β . Consequently, the sample complexity to achieve ε -optimal policy and value function with probability 1 -β is:

<!-- formula-not-decoded -->

where c = 2 · 96 2 , and c = 2 · 48 2 repectively.

Proof. For any π , as V π P is the solution to V π P = T π γ ( V P ) , by Lemma F.7 when:

<!-- formula-not-decoded -->

with probability 1 -β , we have:

<!-- formula-not-decoded -->

Since the above result holds for all π ∈ Π , then:

<!-- formula-not-decoded -->

By Lemma D.1, as ̂ π ∗ = arg max π ∈ Π V π ̂ P , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, when:

we have:

with probability 1 -β

.

At the same time, when n ≥ 32 p -1 ∧ log(2 | S | 2 | A | /β ) , we have:

<!-- formula-not-decoded -->

with probability 1 -β concurrently.

## G.2 Reduction to DR-DMDP Approach under f k Uncertainty Set

Theorem G.2 (Restatement of Theorem 4.4) . Suppose Assumptions 1, 2 are in force. Then, for any

<!-- formula-not-decoded -->

The output policy ̂ π ∗ and V ∗ ̂ P √ n returned by Algorithm 2 satisifes:

<!-- formula-not-decoded -->

with probability 1 -β , and the sample complexity to achieve ε -optimal policy and value function with probability 1 -β is:

<!-- formula-not-decoded -->

Proof. Similar with The proof of Theorem D.4 we have:

<!-- formula-not-decoded -->

As, with Proposition B.2, we have P is uniformly ergodic, and for all Q ∈ P and π ∈ Π :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since by Lemma F.7, we have, when n ≥ 32 p ∧ log 2 | S | 2 | A | β :

<!-- formula-not-decoded -->

with probability 1 -β . Combine the intermediate results, we have:

<!-- formula-not-decoded -->

Thus:

with probability 1 -β , where ( i ) and ( ii ) are derived simply by the trival parameter bound p ∧ ≤ 1 2 , S , A ≥ 1 , and β &lt; 1 . The sample complexity required for policy evaluation on ̂ π ∗ and value evaluation on V ∗ ̂ P √ n in achieving ε -optimality are:

<!-- formula-not-decoded -->

where c = 2 · 120 2 and c = 2 · 60 2 respectively.

## G.3 Anchored DR-AMDP Approach under f k Uncertainty Set

Theorem G.3 (Restatement of Theorem 4.5) . Suppose Assumption 1, and 2 are in force. Then for any

<!-- formula-not-decoded -->

The policy ̂ π ∗ and g ∗ ̂ P returned by Algorithm 3 satisifes:

<!-- formula-not-decoded -->

with probability 1 -β , and the sample complexity to achieve ε -optimal policy and value function with probability 1 -β is:

<!-- formula-not-decoded -->

Proof. Similarily with what we have in Theorem D.8, ̂ π ∗ is an optimal policy for the anchored empirical uncertainty set ̂ P :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since g π ̂ P = ξV π ̂ P ( s 0 ) , for any π ∈ Π

<!-- formula-not-decoded -->

where ( i ) relies to P is uniformly ergodic, thus, g π P ( s ) = g π P for all s ∈ S , g π P ≡ c for some constant c . Then, we have:

<!-- formula-not-decoded -->

so:

In addition:

With the choice of ξ = 1 √ n , by Lemma C.3 and Lemma F.7, when n ≥ 32 p ∧ log 2 | S | 2 | A | β , then for all π ∈ Π :

<!-- formula-not-decoded -->

with probability 1 -β . We concludes, when:

<!-- formula-not-decoded -->

then with probability 1 -β

<!-- formula-not-decoded -->

holds. And the sample complexity required to achieve ε -optimality for both policy and average value function is:

<!-- formula-not-decoded -->

where t mix := max π ∈ Π t mix ( P π ) &lt; ∞ , since t mix is equivalent to t minorize up to a constant, proved.

## H Additional Experiment

To further expand on our results in Section 4, we provide additional experiments on baseline comparison and large-scale MDPs. First, we include comparisons with DR RVI Q-learning [46] on Hard MDP Instance 5.1 as baseline. Table 2 and Figure 3 shows the error performance for DR RVI Q-learning and the two algorithms for p ∧ = 0 . 9 and δ = 0 . 1 . They demonstrated that the error levels of our two algorithms are comparable and significantly outperform the previous baseline.

Table 2: Performance comparison with DR RVI Q-learning.

| Sample                 |    10 |     32 |    100 |    316 |   1000 |    3162 |   10000 |   31622 |   100000 |
|------------------------|-------|--------|--------|--------|--------|---------|---------|---------|----------|
| DR RVI Q-learning [46] | 0.184 | 0.0736 | 0.0747 | 0.062  | 0.0552 | 0.0446  | 0.036   | 0.0308  |  0.0261  |
| Reduction to DMDP      | 0.121 | 0.0595 | 0.0535 | 0.0274 | 0.0139 | 0.00665 | 0.00349 | 0.00321 |  0.00117 |
| Anchored DR-AMDP       | 0.167 | 0.0652 | 0.0626 | 0.029  | 0.0116 | 0.00751 | 0.00327 | 0.00237 |  0.00123 |

Further, to demonstrate the capability of our framework in solving large-scale MDPs, we consider a large-scale MDP with 20 states and 30 actions, as in Wang et al. [46]. The nominal transition distribution is specified by p s,a ∼ N (1 , σ s,a ) , where σ s,a ∼ Uniform [0 , 100] , followed by normalization. We then choose the uncertainty size δ = 0 . 4 to introduce stronger perturbations, following the setting in Wang et al. [46] under the KL-divergence. Note that although δ = 0 . 4 violates Assumption 2, the slope of the linear regression of our results on the logarithmic scale remains very close to -1 / 2 , further supporting our theoretical guarantees. This observation empirically validates the theoretical results established in our theorems.

Further, to demonstrate the capability of our framework in solving large-scale MDP instances, we consider a large-scale MDP with 20 states and 30 actions, as in Wang et al. [46]. Specifically, Algorithm 2 and Algorithm 3 are evaluated on two distinct large-scale instances, respectively, to verify their effectiveness. The nominal transition distribution is specified by p s,a ∼ N (1 , σ s,a ) ,

<!-- image -->

Figure 3: KL-divergence case for Algorithm 2 and 3

<!-- image -->

where σ s,a ∼ Uniform [0 , 100] , followed by normalization. We then choose the uncertainty size δ = 0 . 4 to introduce stronger perturbations, following the setting in Wang et al. [46] under the KL-divergence. Note that although δ = 0 . 4 violates Assumption 2, the slope of the linear regression of our results on the logarithmic scale remains very close to -1 / 2 , further supporting our theoretical guarantees. This observation empirically validates the theoretical results established in our theorems.

Table 3: Performance on large-scale MDP across different sample sizes.

| Sample            |     10 |     32 |    100 |     316 |    1000 |    3162 |   10000 |    31622 |   100000 |
|-------------------|--------|--------|--------|---------|---------|---------|---------|----------|----------|
| Reduction to DMDP | 0.104  | 0.0654 | 0.0307 | 0.0246  | 0.00828 | 0.00482 | 0.00466 | 0.00312  | 0.00112  |
| Anchored DR-AMDP  | 0.0655 | 0.0372 | 0.0269 | 0.00479 | 0.00373 | 0.00115 | 0.00133 | 0.000942 | 0.000561 |