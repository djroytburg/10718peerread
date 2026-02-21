## Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning

## Emile Anand ∗

Georgia Institute of Technology Atlanta, GA 30308 emile@gatech.edu

## Guannan Qu

Carnegie Mellon University Pittsburgh, PA 94035 gqu@andrew.cmu.edu

## Abstract

Designing efficient algorithms for multi-agent reinforcement learning (MARL) is fundamentally challenging because the size of the joint state and action spaces grows exponentially in the number of agents. These difficulties are exacerbated when balancing sequential global decision-making with local agent interactions. In this work, we propose a new algorithm SUBSAMPLE-MFQ ( Subsample -M eanF ieldQ -learning) and a decentralized randomized policy for a system with n agents. For any k ≤ n , our algorithm learns a policy for the system in time polynomial in k . We prove that this learned policy converges to the optimal policy on the order of ˜ O (1 / √ k ) as the number of subsampled agents k increases. In particular, this bound is independent of the number of agents n .

## 1 Introduction

Reinforcement Learning (RL) has become a popular framework to solve sequential decision making problems in unknown environments and has achieved tremendous success in a wide array of domains such as playing the game of Go [Silver et al., 2016], robotic control [Kober et al., 2013], and autonomous driving [Kiran et al., 2022, Lin et al., 2023a]. A key feature of most real-world systems is their uncertain nature, and thus, RL has emerged as a powerful tool for learning optimal policies for multi-agent systems to operate in unknown environments [Kim and Giannakis, 2017, Zhang et al., 2021, Lin et al., 2024, Anand and Qu, 2024]. While early RL works focused on the single-agent setting, multi-agent RL (MARL) has recently achieved impressive success in many applications, such as coordinating robotic swarms [Preiss et al., 2017, DeWeese and Qu, 2024], real-time bidding [Jin et al., 2018], ride-sharing [Li et al., 2019], and stochastic games [Jin et al., 2020].

Despite growing interest in MARL, extending RL to multi-agent settings poses significant computational challenges due to the curse of dimensionality . MARL is fundamentally difficult as agents in the real-world not only interact with the environment but also with each other [Shapley, 1953]: if each of the n agents has a state space S and action space A , the global state-action space has size ( | S || A | ) n , which is exponential in n . Thus, many RL algorithms (such as temporal difference learning and tabular Q -learning) require computing and storing an ( | S || A | ) n -sized Q-table [Sutton et al., 1999, Bertsekas and Tsitsiklis, 1996]. This scalability issue has been observed in a variety of MARL settings [Blondel and Tsitsiklis, 2000, Papadimitriou and Tsitsiklis, 1999, Littman, 1994].

∗ Work partially done while a visiting student at Carnegie Mellon University and intern at Cognition AI.

## Ishani Karmarkar

Stanford University Palo Alto, CA, 94305 ishanik@stanford.edu

An exciting line of work that addresses this intractability is mean-field MARL [Lasry and Lions, 2007, Yang et al., 2018, Gu et al., 2021, 2022, Hu et al., 2023]. The mean-field approach assumes agents are homogeneous in their state-action spaces, enabling their interactions to be approximated by a two-agent setting: here, each agent interacts with a representative 'mean agent' which evolves as the empirical distribution of states of all other agents. With these assumptions, mean-field MARL learns an optimal policy with sample complexity O ( n | S || A | | S || A | ) , which is polynomial in the number of agents. However, if n is large, this remains prohibitive even for moderate values of | S | and | A | .

Motivated by this problem, in this paper we study the cooperative setting-where agents work collaboratively to maximize a structured global reward-and ask: Can we design a scalable MARL algorithm for learning an approximately optimal policy in a cooperative multi-agent system?

Contributions. We answer this question affirmatively. Our key contributions are outlined below.

Subsampling algorithm. We model the problem as a Markov Decision Process (MDP) with a global agent and n local agents. We propose SUBSAMPLE-MFQ to address the challenge of MARL with a large number of local agents. SUBSAMPLE-MFQ selects k ≤ n local agents to learn a deterministic policy ˆ π est k by applying mean-field value iteration on the k -local-agent subsystem to learn ˆ Q est k , which can be viewed as a smaller Q function. SUBSAMPLE-MFQ then deploys a stochastic policy π est k , where the global agent samples k local agents uniformly at each step and uses ˆ π est k to determine its action, while each local agent samples k -1 other local agents and uses ˆ π est k to determine its action.

Sample complexity and theoretical guarantee. As the number of local agents increases, the size of ˆ Q k scales polynomially with k , rather than polynomially with n as in mean-field MARL. Analogously, when the size of the local agent's state space grows, the size of ˆ Q k scales exponentially with k , rather than exponentially with n , as in traditional Q -learning. The key analytic technique underlying our results is a novel MDP sampling result. This result shows that the performance gap between π est k and the optimal policy π ∗ is at most ˜ O (1 / √ k ) , with high probability. The choice of k reveals a fundamental trade-off between the size of the Q -table and the optimality of π est k . For example, if k is set to O (log n ) , SUBSAMPLE-MFQ is the first centralized MARL algorithm to achieve a polylogarithmic run-time in n , representing an exponential speedup over the previously best-known polytime mean-field MARL methods, while maintaining a decaying optimality gap as n gets large,

While our results are theoretical in nature, we hope SUBSAMPLE-MFQ will further exploration of sampling in Markov games, and potentially inspire new practical multi-agent algorithms.

Related work. MARL has a rich history, starting with early works on Markov games [Littman, 1994, Sutton et al., 1999], which are a multi-agent extension of MDPs. MARL has since been actively studied [Zhang et al., 2021] in a broad range of settings. MARL is most similar to the category of 'succinctly described' MDPs [Blondel and Tsitsiklis, 2000], where the state/action space is a product space formed by the individual state/action spaces of multiple agents, and where the agents interact to maximize an objective. A recent line of research constrains the problem to sparse networked instances to enforce local interactions between agents [Qu et al., 2020a, Lin et al., 2020, Mondal et al., 2022]. In this formulation, the agents correspond to vertices on a graph who only interact with nearby agents. By exploiting Gamarnik's correlation decay property from combinatorial optimization [Gamarnik et al., 2009], they overcome the curse of dimensionality by simplifying the problem to only search over the policy space derived from the truncated graph to learn approximately optimal solutions. However, as the underlying network structure becomes dense with many local interactions, the neighborhood of each agent gets large, and these algorithms become intractable.

Mean-Field RL . Under assumptions of homogeneity in the state/action spaces of the agents, the problem of densely networked multi-agent RL was studied by Yang et al. [2018], Gu et al. [2021], who approximated the solution in polynomial time with a mean-field approach where the approximation error scales in O (1 / √ n ) . In contrast, our work achieves subpolynomial runtimes by directly sampling from this mean-field distribution. Cui and Koeppl [2022] introduce heterogeneity to mean-field MARL by modeling non-uniform interactions through graphons; however, these methods crucially assume the existence of graphon sequences that converge in cut-norm to the finite graph. In the cooperative setting, Subramanian et al. [2022], Cui et al. [2023] studies a mean-field setting with q types of homogeneous agents; however, their learned policy does not provably converge to the optimum.

Other related works. Our work is related to factored MDPs, where there is a global action affecting every agent; however, in our case, each agent has its own action [Min et al., 2023, Lauer and

Riedmiller, 2000]. Jin et al. [2020] reduces the dependence of the product action space to an additive dependence with V-learning. Our work further reduces the complexity of the joint state space, which has not been previously accomplished. We add to the growing literature on the Centralized Training with Decentralized Execution regime [Zhou et al., 2023], as our algorithm learns a provably approximately optimal policy using centralized information, but makes decisions using only local information during execution. Finally, one can efficiently approximate the Q -table through function approximation [Jin et al., 2021]. However, achieving theoretical bounds on the performance loss due to function approximation is intractable without strong assumptions such as linear Bellman completeness or low Bellman-Eluder dimension [Golowich and Moitra, 2024]. While our work primarily studies the finite tabular setting, we extend it to non-tabular linear MDPs in Appendix J.

## 2 Preliminaries

Notation. For k, n ∈ N where k ≤ n , let ( [ n ] k ) denote the set of k -sized subsets of [ n ] = { 1 , . . . , n } . For any vector z ∈ R d , let ∥ z ∥ 1 and ∥ z ∥ ∞ denote the standard ℓ 1 and ℓ ∞ norms of z respectively. Let ∥ A ∥ 1 denote the matrix ℓ 1 -norm of A ∈ R n × m . Given variables s 1 , . . . , s n , s ∆ := { s i : i ∈ ∆ } for ∆ ⊆ [ n ] . We use ˜ O ( · ) to suppress polylogarithmic factors in all problem parameters except n . For a discrete measurable space ( X , F ) , the total variation distance between probability measures ρ 1 , ρ 2 is given by TV( ρ 1 , ρ 2 ) = 1 2 ∑ x ∈X | ρ 1 ( x ) -ρ 2 ( x ) | . Next, x ∼ D ( · ) denotes that x is a random element sampled from a distribution D , and we denote that x is a random sample from the uniform distribution over a finite set Ω by x ∼ U (Ω) . We include a detailed notation table in Table 1.

## 2.1 Problem formulation

We consider a system of n + 1 agents, where agent g is a 'global decision making agent' and the remaining n agents, denoted by [ n ] , are 'local agents.' At time t , the agents are in state s ( t ) = ( s g ( t ) , s 1 ( t ) , ..., s n ( t )) ∈ S := S g ×S n l , where s g ( t ) ∈ S g denotes the global agent's state, and for each i ∈ [ n ] , s i ( t ) ∈ S l denotes the state of the i 'th local agent. The agents cooperatively select actions a ( t ) = ( a g ( t ) , a 1 ( t ) , ..., a n ( t )) ∈ A := A g ×A n l , where a g ( t ) ∈ A g denotes the global agent's action and a i ( t ) ∈ A l denotes the i 'th local agent's action. At time t +1 , the next state for all the agents is independently generated by stochastic transition kernels P g : S g ×S g ×A g → ∆( S g ) and P l : S l ×S l ×S g ×A l → ∆( S l ) as follows:

<!-- formula-not-decoded -->

The system then collects a structured stage reward r ( s ( t ) , a ( t )) where the reward r : S × A → R depends on s ( t ) and a ( t ) through Equation (2), and where r g and r l is typically application specific.

<!-- formula-not-decoded -->

A policy π : S → P ( A ) maps from states to distributions of actions such that a ∼ π ( ·| s ) . Given γ ∈ (0 , 1) , we seek to learn a policy π that maximizes the ( γ -discounted) value for each s ∈ S :

<!-- formula-not-decoded -->

The cardinality of the search space simplex for the optimal policy is |S g ||S l | n |A g ||A l | n , which is exponential in the number of agents, underscoring the need for efficient approximation algorithms.

To efficiently learn policies that maximize the objective, we make the following standard assumptions:

Assumption 2.1 (Finite state/action spaces) . We assume that the state and action spaces of all the agents in the MARL game are finite: |S l | , |S g | , |A g | , |A l | &lt; ∞ . Appendix J of the supplementary material relaxes this assumption to the non-tabular setting with infinite continuous sets.

Assumption 2.2 (Bounded rewards) . The components of the reward function are bounded. Specifically, ∥ r g ( · , · ) ∥ ∞ ≤ ˜ r g , and ∥ r l ( · , · , · ) ∥ ∞ ≤ ˜ r l . This implies ∥ r ( · , · ) ∥ ∞ ≤ ˜ r g + ˜ r l := ˜ r .

Definition 2.3 ( ϵ -optimal policy) . Given a policy simplex Π , π ∈ Π is an ϵ -optimal policy if V π ( s ) ≥ sup π ∗ ∈ Π V π ∗ ( s ) -ϵ .

Motivating examples. Below we give examples of two cooperative MARL settings which are naturally modeled by our setting. Our experiments in Appendix B reveal a monotonic improvement in the learned policies as k → n , while providing a substantial speedup over mean-field Q -learning 2 .

Figure 1: Bounded exploration in warehouse accidents, and traffic congestions with Gaussian squeeze.

<!-- image -->

- Gaussian squeeze: In this task, n homogeneous agents determine individual actions a i to jointly maximize the objective r ( x ) = xe -( x -µ ) 2 /σ 2 , where x = ∑ n i =1 a i , and µ and σ are the predefined mean and variance of the system. In scenarios of traffic congestion, each agent i ∈ [ n ] is a controller trying to send a i vehicles into the main road, where controllers coordinate with each other to avoid congestion, hence avoiding either over-use or under-use, thereby contributing to the entire system. This GS problem is previously studied in Yang et al. [2018], and serves as an ablation study on the impact of subsampling for MARL.
- Constrained exploration: Consider an M × M grid. Each agent's state is a coordinate in [ M ] × [ M ] . The state represents the center of a d × d box where the global agent constrains the local agents' movements. Initially, all agents are in the same location. At each time-step, the local agents take actions a i ( t ) ∈ R 2 (e.g., up, down, left, right) to transition between states and collect rewards. The transition kernel ensures that local agents remain within the d × d box dictated by the global agent, using knowledge of a i ( t ) , s g ( t ) , and s i ( t ) . In warehouse settings where shelves have collapsed, creating hazardous or inaccessible areas, we want agents to clean these areas. However, exploration in these regions may be challenging due to physical constraints or safety concerns, causing exploration in these regions to be disincentivized from the local agents' perspectives. Through an appropriate design of the reward and transition functions, the global agent could guide the local agents to focus on specific d × d grids, allowing efficient cleanup while minimizing risk.

Capturing heterogeneity. Following Mondal et al. [2022], Xu and Klabjan [2023], our model can capture heterogeneity in the local agents by modeling agent types as part of the state: to do this, we assign a type ε i ∈ E to each local agent by letting S l = E × S ′ l , where E is a set that enumerates possible types that are treated as a fixed part of the agent's state, and S ′ l is the latent state space of any local agent. The transition and reward functions can vary depending on the agent's type. The global agent can provide unique signals to local agents of each type by letting s g ∈ S g and a g ∈ A g denote a state/action vector where each element corresponds to a type.

## 3 Algorithmic Approach: Subsampled Value Iteration

Q-learning. Our starting point is the classic Q-learning framework [Watkins and Dayan, 1992] for offline-RL, which seeks to learn the Q -function Q : S × A → R . For any policy π , Q π ( s, a ) = E π [ ∑ ∞ t =0 γ t r ( s ( t ) , a ( t )) | s (0)= s, a (0)= a ] . Initially, Q 0 ( s, a )=0 , for all ( s, a ) ∈ S×A . Then, for all t ∈ [ T ] , it is updated as Q t +1 ( s, a ) ←T Q t ( s, a ) , where the Bellman operator T is

<!-- formula-not-decoded -->

It is well known that T is γ -contractive, ensuring that the above updates converge to a unique Q ∗ such that T Q ∗ = Q ∗ . The optimal policy π ∗ : S → A can then be computed greedily as

2 We provide supporting code for the algorithm and experiments in https://github.com/emiletimothy/ Mean-Field-Subsample-Q-Learning

π ∗ ( s ) = arg max a ∈A Q ∗ ( s, a ) . However, the update complexity for the Q -function is O ( |S||A| ) = O ( |S g ||S l | n |A g ||A l | n ) , which is exponential in the number of local agents increases.

Mean-field transformation. To address this, mean-field MARL [Yang et al., 2018] (under homogeneity assumptions) studies the empirical distribution function F z [ n ] : Z l → R , for Z l := S l ×A l :

<!-- formula-not-decoded -->

Let µ n ( Z l ) = { b n | b ∈ { 0 , . . . , n }} |Z l | be the space of |Z l | -sized vectors (or |S l | × |A l | -sized tables), where each entry is in { 0 , 1 /n, 2 /n, . . . , 1 } . Intuitively, µ n ( Z l ) is a discrete distribution over ( S l , A l ) where each probability assigned is a multiple of 1 /n . Then, F z [ n ] ∈ µ n ( Z l ) indicates the proportion of agents in each state-action pair.

As in Q-learning, in mean-field Q-learning initially ˆ Q 0 ( s g , s 1 , a g , a 1 , F z [ n ] \ 1 ) = 0 . At each time t ∈ [ T ] , we update ˆ Q as ˆ Q t +1 = ˆ T ˆ Q t , where ˆ T is the Bellman operator in distribution space :

<!-- formula-not-decoded -->

Since the Q -function is permutation-invariant in the homogeneous local agents, one sees that for each t ≥ 0 , Q t ( s g , s [ n ] , a g , a [ n ] ) = ˆ Q t ( s g , s 1 , a g , a 1 , F z [ n ] \ 1 ) . In other words, mean-field Q-learning implements the same updates as standard Q-learning. However, the update complexity of ˆ Q is only O ( |S g ||A g ||Z l | n |Z l | ) , which scales polynomially in n but exponentially in |Z l | .

Remark 3.1. Existing methods use sample complexity ˜ O (min {|S g ||A g ||Z l | n , |S g ||A g ||Z l | n |Z l | } ) : one uses Q -learning if |Z l | n -1 &lt; n |Z l | , and mean-field value iteration otherwise. In each regime, as n scales, the update complexity becomes computationally infeasible.

To further reduce the update complexity, we propose SUBSAMPLE-MFQ to overcome the polynomial (in n ) sample complexity of mean-field Q -learning and the exponential (in n ) sample complexity of traditional Q -learning. We begin by motivating the intuition behind our algorithms.

## 3.1 Overview of approach.

Offline Planning: Algorithm 1. First, the global agent randomly samples a subset of local agents ∆ ⊆ [ n ] such that | ∆ | = k , for k ≤ n . It then ignores all other local agents [ n ] \ ∆ , and performs value iteration (using m samples to update the Q -function in each iterate) to approximately learn the Q -function ˆ Q est k,m and policy ˆ π est k,m for this surrogate subsystem of k local agents. We denote the surrogate reward gained by this subsystem at each time step by r ∆ : S × A → R , where

<!-- formula-not-decoded -->

In Theorem E.3, we show that ∥ ˆ Q est k,m -Q ∗ ∥ ∞ is Lipschitz continuous with respect to the TV-distance between the state/action pairs of the subsampled agents and the full set of n agents. Equipped with this approximation guarantee, we show how to construct an approximately optimal policy on the full system on n agents. In general, converting this policy on k local agents to a policy on the full n -agent system without sacrificing error guarantees can be intractable, and there is a line on works on centralized-training decentralized-execution (CTDE) [Xu and Klabjan, 2023, Zhou et al., 2023] which shows that designing performant decentralized policies can be highly non-trivial.

Online Execution: Algorithm 2. In order to circumvent this obstacle and convert the optimality of each agent's action in the k local-agent subsystem to an approximate optimality guarantee on the full n -agent system, we propose a randomized policy π est k,m , where the global agent samples ∆ ∈ ( [ n ] k ) at each time-step to derive an action a g ← ˆ π est k,m ( s g , s ∆ ) , and where each local i agent samples k -1 other local agents ∆ i to derive an action a i ← ˆ π est k,m ( s g , s i , s ∆ i ) . Finally, Theorem 4.4 shows that the policy π est k,m converges to the optimal policy π ∗ as k → n with rate ˜ O ( 1 √ k ) .

## 3.2 Algorithm description.

We now formally describe the algorithms. Before describing Algorithm 1 ( SUBSAMPLE-MFQ : Learning) and Algorithm 2 ( SUBSAMPLE-MFQ : Execution) in detail, it will be helpful to first define the empirical distribution function :

Definition 3.2 (Empirical Distribution Function) . For any population ( z 1 , . . . , z n ) ∈ Z n l , where Z l := S l ×A l , define the empirical distribution function F z ∆ : Z l → R + for all z := ( z s , z a ) ∈ S l ×A l and for all ∆ ∈ ( [ n ] k ) by F z ∆ ( x ) := 1 k ∑ i ∈ ∆ 1 { s i = z s , a i = z a } .

Let µ k ( Z l ) := { b k | b ∈ { 0 , . . . , k } } |Z l | be the space of |Z l | -sized vectors (or |S l | × |A l | -sized tables) where each entry is in { 0 , 1 /k, 2 /k, . . . , 1 } . Intuitively, µ k ( Z l ) is a discrete distribution over ( S l , A l ) where each probability assigned is a multiple of 1 /k . Here, F z ∆ ∈ µ k ( Z l ) indicates the proportion of agents (in the k -local-agent subsystem) at each state/action pair.

Algorithm 1 (Offline learning). Let m ∈ N denote the sample size for the learning algorithm with sampling parameter k ≤ n . As in Remark 3.1, when |Z l | k ≤ |Z l | k |Z l | , the algorithm uses traditional value-iteration, and when |Z l | k &gt; |Z l | k |Z l | , it uses mean-field value iteration. We formally describe the procedure for each regime below:

Regime with Large State/Action Space: When |Z l | k ≤ |Z l | k |Z l | , we iteratively learn the optimal Q -function for a subsystem with k -local agents denoted by ˆ Q t k,m : S g ×S k l ×A g ×A k l → R , which is initialized to 0. At time t , we update

<!-- formula-not-decoded -->

where ˜ T k,m is the empirically adapted Bellman operator in Equation (8). Since the system is unknown, ˜ T k,m cannot directly perform the Bellman update, so it instead uses m random samples s j g ∼ P g ( ·| s g , a g ) and s j i ∼ P l ( ·| s i , s g , a i ) for each j ∈ [ m ] , i ∈ ∆ to approximate the system:

<!-- formula-not-decoded -->

Since ˜ T k,m is γ -contractive, Algorithm 1 applies value iteration with ˜ T until ˆ Q k,m converges to a fixed point satisfying ˜ T k,m ˆ Q est k,m = ˆ Q est k,m , yielding a deterministic policy ˆ π est k,m ( s g , s ∆ ) where

<!-- formula-not-decoded -->

Regime with Large Number of Agents: When |Z l | k &gt; |Z l | k |Z l | , we learn the optimal mean-field Q -function for a k local agent system, denoted by ˆ Q t k,m : S g ×S l × µ k -1 ( Z l ) ×A g ×A l → R , which is initialized to 0. At time t , we update

<!-- formula-not-decoded -->

where ˆ T k,m is the empirically adapted mean-field Bellman operator in Equation (10). Similarly, as the system is unknown, ˆ T k,m cannot directly perform the Bellman update and instead uses m random samples s j g ∼ P g ( ·| s g , a g ) and s j i ∼ P l ( ·| s i , s g , a i ) , ∀ j ∈ [ m ] , i ∈ ∆ to approximate the system:

<!-- formula-not-decoded -->

Since ˆ Q t k,m depends on s ∆ and a ∆ through F z ∆ , ˆ T k,m is also γ -contractive. So, Algorithm 1 applies value iteration with ˆ T until ˆ Q k,m converges to a fixed point satisfying ˆ T k,m ˆ Q est k,m = ˆ Q est k,m , yielding a deterministic policy ˆ π est k,m ( s g , s 1 , F s ˜ ∆ ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Algorithm 1 SUB-SAMPLE-MFQ : Learning

Require: Amulti-agent system as in Section 2, number of iterations T , sampling parameters k ∈ [ n ] , m ∈ N , and discount factor γ ∈ (0 , 1) .

```
1: Let ∆ = { 1 , . . . , k } , ˜ ∆ = { 2 , . . . , k } , and µ k -1 ( Z l )= { b k -1 : b ∈ { 0 , 1 , . . . , k -1 }} |S l |×|A l | . 2: if |Z l | k ≤ |Z l | k |Z l | then 3: // Sub-sampled version of standard Q -learning 4: Initialize ˆ Q 0 k,m ( s g , s ∆ , a g , a ∆ ) = 0 , ∀ ( s g , s ∆ , a g , a ∆ ) . 5: for t = 1 to T do 6: for ( s g , s ∆ , a ∆ , a g ) ∈ S g ×S k l ×A g ×A k l do 7: ˆ Q t +1 k,m ( s g , s ∆ , a g , a ∆ )= ˜ T k,m ˆ Q t k,m ( s g , s ∆ , a g , a ∆ ) 8: Let the greedy policy be ˆ π est k,m ( s g , s ∆ ) := arg max a g ∈A g ,a ∆ ∈A k l ˆ Q T k,m ( s g , s ∆ , a g , a ∆ ) . 9: else 10: // Sub-sampled version of mean-field Q -learning 11: Initialize ˆ Q 0 k,m ( s g , s 1 , F z ˜ ∆ , a 1 , a g ) = 0 , ∀ ( s g , s ∆ , a ∆ , a g ) . 12: for t = 1 to T do 13: for ( s g , s 1 , F z ˜ ∆ , a 1 , a g ) ∈ S g ×S l × µ k -1 ( Z l ) ×A l ×A g do 14: ˆ Q t +1 k,m ( s g , s 1 , F z ˜ ∆ , a 1 , a g ) = ˆ T k,m ˆ Q t k,m ( s g , s 1 , F z ˜ ∆ , a 1 , a g ) 15: Let the greedy policy be ˆ π est k,m ( s g , s i , F s ˜ ∆ ):=arg max a g ∈A g ,a i ∈A l , F a ˜ ∆ ∈ µ k -1 ( A l ) ˆ Q T k,m ( s g , s 1 , F z ˜ ∆ , a 1 , a g )
```

<!-- formula-not-decoded -->

Remark 3.3. Since mean-field Q -learning maintains the same updates as standard Q -learning (from LemmaC.19 which follows by noting that the Q -function is permutation-invariant in the homogeneous local agents), the deterministic policies ˆ π est k,m ( s g , s ∆ ) and ˆ π est k,m ( s g , s 1 , F s ¯ ∆ ) are equivalent.

Algorithm 2 (Online implementation). In Algorithm 2, ( SUBSAMPLE-MFQ : Execution) the global agent samples local agents ∆( t ) ∼ U ( [ n ] k ) at each step to derive action a g ( t ) = [ˆ π est k,m ( s g , s ∆ ( t ))] g , and each local agent i samples other local agents ∆ i ( t ) ∼ U ( [ n ] \ i k -1 ) to derive action a i ( t ) = [ˆ π est k,m ( s g , s i , s ∆ ( t ))] l . The system then incurs a reward r ( s, a ) . This procedure of first sampling agents and then applying ˆ π est k,m is denoted by a stochastic policy π est k,m ( a | s ) , where π est k,m ( a g | s ) is the global agent's action distribution and π est k,m ( a l | s ) is the local agent's action distribution:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The agents then transition to their next states.

## 4 Theoretical Guarantees and Analysis Approach

We now show the value of the expected discounted cumulative reward produced by π est k,m is approximately optimal, where the optimality gap decays as k → n and m grows.

Bellman noise. We introduce the notion of Bellman noise, which is used in the main theorem. Note that ˆ T k,m is an unbiased estimator of the adapted Bellman operator ˆ T k ,

<!-- formula-not-decoded -->

Let ˆ Q 0 k ( s g , s ∆ , a g , a ∆ ) = 0 . For t &gt; 0 , let ˆ Q t +1 k = ˆ T k ˆ Q t k , where ˆ T k is defined for k ≤ n in Equation (13). Then, ˆ T k is also a γ -contraction with fixed-point ˆ Q ∗ k . By the law of large numbers, lim m →∞ ˆ T k,m = ˆ T k and ∥ ˆ Q est k,m -ˆ Q ∗ k ∥ ∞ → 0 as m →∞ . For finite m , ϵ k,m := ∥ ˆ Q est k,m -ˆ Q ∗ k ∥ ∞

## Algorithm 2 SUBSAMPLE-MFQ : Execution

Require: Parameter T ′ for the number of iterations for the decision-making sequence. Sampling parameter k ∈ [ n ] , m ∈ N . Discount factor γ . Policy ˆ π est k,m ( s g , F s ∆ ) .

- 1: Learn ˆ π est k,m from Algorithm 1.
- 2: Sample ( s g (0) , s [ n ] (0)) ∼ s 0 , where s 0 is a distribution on the initial global state ( s g , s [ n ] )
- 3: Initialize the total reward R 0 = 0 .
- 4: Policy π est k,m ( s ) is defined as follows:
- 5: for t = 0 to T ′ do
- 6: Choose ∆ uniformly at random from ( [ n ] k ) and let a g ( t ) = [ˆ π est k,m ( s g ( t ) , s ∆ ( t ))] g .
- 7: for i = 1 to n do
- 8: Choose ∆ i uniformly at random from ( [ n ] \ i k -1 ) and let a i ( t ) = [ˆ π est k,m ( s g ( t ) , s i ( t ) , s ∆ i ( t ))] l .
- 9: Let s g ( t +1) ∼ P g ( ·| s g ( t ) , a g ( t )) .
- 10: Let s i ( t +1) ∼ P l ( ·| s i ( t ) , s g ( t ) , a i ( t )) , ∀ i ∈ [ n ] .
- 11: R t +1 = R t + γ t · r ( s, a )

is the well-studied Bellman noise. To compare the performance between π ∗ and π est k , we define the value function of a policy π :

Definition 4.1. For a given policy π , the value function V π : S → R for S := S g ×S n l is given by:

<!-- formula-not-decoded -->

Intuitively, V π ( s ) is the expected discounted cumulative reward when starting from state s and applying actions from the policy π across an infinite horizon.

With the above preparations, we are primed to present our main result: a high-probability bound on the optimality gap for our learned policy π est k,m that decays with rate ˜ O (1 / √ k ) .

Theorem 4.2. Let π est k,m denote the learned policy deployed in SUBSAMPLE-MFQ: Execution . Then, for all s 0 ∈ S := S g ×S n l , we have

<!-- formula-not-decoded -->

We prove Theorem 4.4 in Appendix G, and provide a proof sketch in Appendix D. We also generalize the result to stochastic rewards in Appendix H.

To control the Bellman noise ϵ k,m , we show that for sufficiently many samples m ∗ , ϵ k,m ∗ also decays on the order of ˜ O (1 / √ k ) with high probability. For this, we introduce Lemma 4.3:

Lemma 4.3 (Controlling the Bellman Noise.) . For k ∈ [ n ] , let

<!-- formula-not-decoded -->

be the number of samples in Equation (10) . If the number of iterations T satisfies T ≥ 2 1 -γ log ˜ r √ k 1 -γ then with probability at least 1 -1 100 e k , the Bellman error satisfies ϵ k,m ∗ ≤ ˜ O ( 1 √ k ) .

We defer the proof of Lemma 4.3 to Appendix G.1. To simplify notation, let π est k := π est k,m ∗ . Then, by combining Lemma 4.3 and Theorem 4.2, we obtain our main result in Theorem 4.4:

Theorem 4.4. Let π est k denote the learned policy from SUBSAMPLE-MFQ: Execution where the number of samples m is determined in Lemma 4.3. Suppose T ≥ 2 1 -γ log ˜ r √ k 1 -γ . Then, ∀ s 0 ∈ S := S g ×S n l , with probability at least 1 -1 / 100 e k , we have 3 ,

<!-- formula-not-decoded -->

3 The 1 / 100 e k term can be replaced to any arbitrary δ &gt; 0 at the cost of attaching logarithmic dependencies of δ on the 4 / √ k term in the error bound.

Remark 4.5. One could also derive an alternate bound that retains the T factor via a γ T dependence in the final bound (which shows an exponentially decaying error with the time horizon T ). Moreover, in Lemma G.10, we show that the query complexity is on the order of O ( mT | S g || S l | k | A g || A l | k ) , and we bound T and set m = poly(1 / (1 -γ )) to attain the 1 / √ k rate.

Remark 4.6. Additionally, our poly(1 / (1 -γ )) -dependence may be loose since we do not use more complicated variance reduction techniques as in Sidford et al. [2018a,b], Wainwright [2019], Jin et al. [2024] to optimize the number of samples m which is used to bound the Bellman error ϵ k,m . Moreover, incorporating variance reduction would significantly complicate the algorithm and intuition.

Our analysis hinges on two non-trivial technical steps. Firstly, for an intermediary step in Theorem E.3 we establish that TV distance, rather than the stronger KL-divergence, is the correct metric to use (as the KL-divergence between F z ∆ and F z [ n ] decays too slowly as k → n , as we show in Lemma F.6). This requires exploiting a recent extension of the DKW inequality to sampling without replacement [Anand and Qu, 2024], and showing that the transition dynamics we study saturates the dataprocessing inequality for TV-distance. Secondly, we adapt the celebrated performance-difference lemma [Kakade and Langford, 2002] to our multi-agent setting, which entails a principled analysis and careful probabilistic argument (to which we refer the reader to Appendix G).

Remark 4.7. We also extend the formulation of SUBSAMPLE-MFQ to off-policy Q -learning Chen et al. [2021b], Chen and Maguluri [2022a], which replaces the generative oracle assumption with a stochastic approximation scheme that learns an approximately optimal policy using historical data. Appendix I provides theoretical guarantees with a similar decaying optimality gap as in Theorem 4.4.

Remark 4.8. The asymptotic sample complexity of Algorithm 1 for learning ˆ π est k for a fixed k is min { ˜ O ( |Z l | k |S g ||A g | ) , ˜ O ( k |Z l | |Z l ||S g ||A g | ) } , which is at least polynomially faster the standard Q -learning or mean-field value iteration as discussed in Remark 3.1. By Theorem 4.4, as k → n , the optimality gap decays, revealing a fundamental trade-off in the choice of k : increasing k improves the performance of the policy, but increases the size of the Q -function. We explore this trade-off further in our experiments. If we set k = O (log n ) , this leads to a sample complexity of min { ˜ O ( n log |Z l | |S g ||A g | ) , ˜ O ((log n ) |Z l | |S g ||A g | ) } . This is an exponential speedup on the complexity from mean-field value iteration (from poly( n ) to poly(log n ) ), as well as over traditional value-iteration (from exp( n ) to poly( n ) ), where the optimality gap decays to 0 with rate O ( 1 √ log n ) .

Remark 4.9. If k = O (log n ) , SUBSAMPLE-MFQ handles |E| ≤ O (log n/ log log n ) types of local agents, since the run-time of the learning algorithm becomes poly( n ) . This surpasses the previous heterogeneity capacity from Mondal et al. [2022], which only handles constant |E| ≤ O (1) . Increasing the agent heterogeneity using this type formulation does make the algorithm somewhat more expensive since it factors into the query complexity via the state space of the local agents; however, since the optimality gap of the learned policy is on the order of ˜ O (1 / √ k ) (modulo very small √ log |S l ||A l | factors), increasing the amount of agent heterogeneity does not degrade the quality of the learned policy as the theoretical bound does not get worse. Moreover, recent methods from the graphon mean-field MARL community might be able to enable stronger heterogeneity in the system [Cui and Koeppl, 2022, Anand and Liaw, 2025, Hu et al., 2023].

In the non-tabular setting with infinite state/action spaces, one could replace the Q -learning algorithm with any arbitrary value-based RL method that learns ˆ Q k with function approximation [Sutton et al., 1999] such as deep Q -networks [Silver et al., 2016]. Doing so raises an additional error that factors into Theorem 4.4. We formalize this below.

Assumption 4.10 (Linear MDP with infinite state spaces) . Suppose S g and S l are infinite compact sets. Furthermore, suppose there exists a feature map ϕ : S × A → R d and d unknown (signed) measures µ = ( µ 1 , . . . , µ d ) over S and a vector θ ∈ R d such that for any ( s, a ) ∈ S × A , we have P ( ·| s, a ) = ⟨ ϕ ( s, a ) , µ ( · ) ⟩ and r ( s, a ) = ⟨ ϕ ( s, a ) , θ ⟩ .

The existence of ϕ : S × A → R d implies one can estimate the Q -function of any policy as a linear function. This assumption is commonly used in policy iteration algorithms Lattimore et al. [2020], Wang et al. [2023], and allows one to obtain sample complexity bounds that are independent of |S l | and |A l | . Finally, as is standard in RL, we assume bounded feature-norms [Tkachuk et al., 2023]:

Assumption 4.11 (Bounded features) . We assume that ∥ ϕ ( s, a ) ∥ 2 ≤ 1 for all ( s, a ) ∈ S × A .

Then, through a reduction from Zhang et al. [2024], Ren et al. [2024] that uses function approximation to learn the spectral features ϕ k for ˆ Q k , we derive a performance guarantee for the learned policy π est k , where the optimality gap decays with k .

Theorem 4.12. When π est k is derived from the spectral features ϕ k learned in ˆ Q k , and M is the number of samples used in the function approximation, then

<!-- formula-not-decoded -->

We defer the proof of Theorem 4.12 to Appendix J.

## 5 Conclusion and Future Works

This work develops subsampling for mean field MARL in a cooperative system with a global decisionmaking agent and n homogeneous local agents. We propose SUBSAMPLE-MFQ which learns each agent's best response to the mean effect from a sample of its neighbors, allowing an exponential reduction on the sample complexity of approximating a solution to the MDP. We provide a theoretical analysis on the optimality gap of the learned policy, showing that (with high probability) the learned policy converges to the optimal policy with the number of agents k sampled at the rate ˜ O (1 / √ k ) , and validate our theoretical results through numerical experiments. We show that the decay rate is maintained, on expectation, when the reward functions are stochastic and when agents learn from a single trajectory on historical data through off-policy Q -learning. Finally, we extend this result to the non-tabular setting with infinite state and action spaces under assumptions of a linear MDP model.

Limitations and future work. Our current work assumes that the global and local agents cooperate to optimize a structured reward under a specific dynamic model. While this model is more general than the federated learning setting, one direction would be to extend our algorithms and analysis to weaker network assumptions. We believe our framework, which can handle dense subgraphs, as well as expander-graph decompositions [Reingold, 2008, Anand and Umans, 2023, Anand, 2025] may be amenable for this. Secondly, our current work incorporates mild heterogeneity among agents and assumes they are cooperative; thus, another avenue would be to consider settings with competitive agents or more complex agent heterogeneity. Finally, it would be exciting to generalize this work to the online no-regret setting.

Societal Impacts. This work is theoretical and foundational in nature. As such, while it enables more scalable multi-agent algorithms, it is not tied to any specific applications or deployments.

Acknowledgements. This work was supported by NSF Grants 2154171, CAREER Award 2339112, CMUCyLab Seed Funding, and CCF 2338816. We gratefully acknowledge insightful discussions with Siva Theja Maguluri, Yiheng Lin, Jan van den Brand, Adam Wierman, Yunbum Kook, Yi Wu, Sarah Liaw, and Rishi Veerapaneni. Finally, we thank the anonymous NeurIPS reviewers for their helpful feedback.

## References

- Emile Anand. Towards the pseudorandomness of expander random walks for read-once acc0 circuits, 2025. URL https://arxiv.org/abs/2501.07752 .
- Emile Anand and Sarah Liaw. Feel-good thompson sampling for contextual bandits: a markov chain monte carlo showdown, 2025. URL https://arxiv.org/abs/2507.15290 .
- Emile Anand and Guannan Qu. Efficient reinforcement learning for global decision making in the presence of local agents at scale. arXiV , 2024. URL https://arxiv.org/abs/2403.00222 .
- Emile Anand and Chris Umans. Pseudorandomness of the Sticky Random Walk. Caltech Undergraduate Thesis , 2023. URL https://thesis.library.caltech.edu/16116/7/AnandCaltech\_ Final\_Thesis\_Submission2.pdf .
- Emile Anand, Jan van den Brand, Mehrdad Ghadiri, and Daniel J. Zhang. The Bit Complexity of Dynamic Algebraic Formulas and Their Determinants. In Karl Bringmann, Martin Grohe, Gabriele Puppis, and Ola Svensson, editors, 51st International Colloquium on Automata, Languages, and Programming (ICALP 2024) , volume 297 of Leibniz International Proceedings in Informatics (LIPIcs) , pages 10:1-10:20, Dagstuhl, Germany, 2024. Schloss Dagstuhl Leibniz-Zentrum für Informatik. ISBN 978-3-95977-322-5. doi: 10.4230/LIPIcs.ICALP.2024. 10. URL https://drops.dagstuhl.de/storage/00lipics/lipics-vol297-icalp2024/ LIPIcs.ICALP.2024.10/LIPIcs.ICALP.2024.10.pdf .
- Emile Anand, Jan van den Brand, and Rose McCarty. The structural complexity of matrix-vector multiplication. arXiv preprint arXiv:2502.21240 , 2025. URL https://www.arxiv.org/pdf/ 2502.21240 .
- Stefan Banach. Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales. Fundamenta Mathematicae , 3(1):133-181, 1922. URL http://eudml.org/doc/ 213289 .
- Dimitri P. Bertsekas and John N. Tsitsiklis. Neuro-Dynamic Programming . Athena Scientific, 1st edition, 1996. ISBN 1886529108. URL https://web.stanford.edu/~bvr/pubs/NDPoverview. pdf .
- Vincent D. Blondel and John N. Tsitsiklis. A Survey of Computational Complexity Results in Systems and Control. Automatica , 36(9):1249-1274, 2000. ISSN 0005-1098. doi: https://doi.org/10.1016/ S0005-1098(00)00050-9. URL https://www.sciencedirect.com/science/article/abs/ pii/S0005109800000509 .
- Shreyas Chaudhari, Srinivasa Pranav, Emile Anand, and José M. F. Moura. Peer-to-peer learning dynamics of wide neural networks. IEEE International Conference on Acoustics, Speech, and Signal Processing 2025 , 2024. URL https://arxiv.org/abs/2409.15267 .
- Zaiwei Chen and Siva Theja Maguluri. Sample complexity of policy-based methods under off-policy sampling and linear function approximation. In International Conference on Artificial Intelligence and Statistics , pages 11195-11214. PMLR, 2022a. URL https://proceedings.mlr.press/ v151/chen22i/chen22i.pdf .
- Zaiwei Chen and Siva Theja Maguluri. Sample complexity of policy-based methods under offpolicy sampling and linear function approximation. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera, editors, Proceedings of The 25th International Conference on Artificial Intelligence and Statistics , volume 151 of Proceedings of Machine Learning Research , pages 11195-11214. PMLR, 28-30 Mar 2022b. URL https://proceedings.mlr.press/v151/ chen22i/chen22i.pdf .
- Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. Finite-sample analysis of off-policy td-learning via generalized bellman operators. Advances in Neural Information Processing Systems , 34:21440-21452, 2021a. URL https://proceedings.neurips.cc/ paper/2021/file/b2eeb7362ef83deff5c7813a67e14f0a-Paper.pdf .

- Zaiwei Chen, Siva Theja Maguluri, Sanjay Shakkottai, and Karthikeyan Shanmugam. A lyapunov theory for finite-sample guarantees of asynchronous q-learning and td-learning variants. arXiv preprint arXiv:2102.01567 , 2021b. URL https://pubsonline.informs.org/doi/10.1287/ opre.2022.0249 .
- Zaiwei Chen, Siva Theja Maguluri, and Martin Zubeldia. Concentration of contractive stochastic approximation: Additive and multiplicative noise. The Annals of Applied Probability , 35(2): 1298-1352, 2025.
- Kai Cui and Heinz Koeppl. Learning Graphon Mean Field Games and Approximate Nash Equilibria. In International Conference on Learning Representations , 2022. URL https://openreview. net/forum?id=0sgntlpKDOz .
- Kai Cui, Christian Fabian, and Heinz Koeppl. Multi-Agent Reinforcement Learning via Mean Field Control: Common Noise, Major Agents and Approximation Properties, 2023. URL https: //arxiv.org/abs/2303.10665 .
- Alex DeWeese and Guannan Qu. Locally interdependent multi-agent MDP: Theoretical framework for decentralized agents with dynamic dependencies. In Forty-first International Conference on Machine Learning , 2024. URL https://arxiv.org/pdf/2406.06823 .
- Thomas G. Dietterich, George Trimponias, and Zhitang Chen. Discovering and removing exogenous state variables and rewards for reinforcement learning, 2018. URL https://arxiv.org/abs/ 1806.01584 .
- A. Dvoretzky, J. Kiefer, and J. Wolfowitz. Asymptotic Minimax Character of the Sample Distribution Function and of the Classical Multinomial Estimator. The Annals of Mathematical Statistics , 27 (3):642 - 669, 1956. doi: 10.1214/aoms/1177728174. URL https://doi.org/10.1214/aoms/ 1177728174 .
- Dylan J Foster, Alexander Rakhlin, Ayush Sekhari, and Karthik Sridharan. On the Complexity of Adversarial Decision Making. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems , 2022. URL https://openreview.net/forum?id=pgBpQYss2ba .
- Scott Fujimoto, David Meger, and Doina Precup. Off-policy deep reinforcement learning without exploration. In International conference on machine learning , pages 2052-2062. PMLR, 2019. URL https://arxiv.org/pdf/1812.02900 .
- David Gamarnik, David Goldberg, and Theophane Weber. Correlation Decay in Random Decision Networks, 2009. URL https://www.jstor.org/stable/24540896 .
- Subhashis Ghosal and Aad van der Vaart. Fundamentals of Nonparametric Bayesian Inference: Space of Probability Densities , page 516-527. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2017. URL https://www.cambridge. org/core/books/fundamentals-of-nonparametric-bayesian-inference/ C96325101025D308C9F31F4470DEA2E8 .
- Noah Golowich and Ankur Moitra. The role of inherent bellman error in offline reinforcement learning with linear function approximation, 2024. URL https://arxiv.org/abs/2406.11686 .
- Haotian Gu, Xin Guo, Xiaoli Wei, and Renyuan Xu. Mean-Field Controls with Q-Learning for Cooperative MARL: Convergence and Complexity Analysis. SIAM Journal on Mathematics of Data Science , 3(4):1168-1196, 2021. doi: 10.1137/20M1360700. URL https://doi.org/10. 1137/20M1360700 .
- Haotian Gu, Xin Guo, Xiaoli Wei, and Renyuan Xu. Mean-Field Multi-Agent Reinforcement Learning: A Decentralized Network Approach, 2022. URL https://arxiv.org/pdf/2108. 02731.pdf .
- Yuanquan Hu, Xiaoli Wei, Junji Yan, and Hengxi Zhang. Graphon Mean-Field Control for Cooperative Multi-Agent Reinforcement Learning. Journal of the Franklin Institute , 360(18):14783-14805, 2023. ISSN 0016-0032. URL https://doi.org/10.1016/j.jfranklin.2023.09.002 .

- Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Jacob Abernethy and Shivani Agarwal, editors, Proceedings of Thirty Third Conference on Learning Theory , volume 125 of Proceedings of Machine Learning Research , pages 2137-2143. PMLR, 09-12 Jul 2020. URL https://proceedings.mlr.press/v125/jin20a.html .
- Chi Jin, Qinghua Liu, and Sobhan Miryoosefi. Bellman eluder dimension: New rich classes of rl problems, and sample-efficient algorithms, 2021. URL https://arxiv.org/abs/2102.00815 .
- Junqi Jin, Chengru Song, Han Li, Kun Gai, Jun Wang, and Weinan Zhang. Real-time bidding with multi-agent reinforcement learning in display advertising. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management , CIKM '18, page 2193-2201, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450360142. doi: 10.1145/3269206.3272021. URL https://dl.acm.org/doi/10.1145/3269206.3272021 .
- Yujia Jin, Ishani Karmarkar, Aaron Sidford, and Jiayi Wang. Truncated variance reduced value iteration. In Advances in Neural Information Processing Systems , 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf . Also available as arXiv preprint arXiv:2405.12952.
- Sham Kakade and John Langford. Approximately Optimal Approximate Reinforcement Learning. In Claude Sammut and Achim Hoffman, editors, Proceedings of the Nineteenth International Conference on Machine Learning (ICML 2002) , pages 267-274, San Francisco, CA, USA, 2002. Morgan Kauffman. ISBN 1-55860-873-7. URL http://ttic.uchicago.edu/~sham/papers/ rl/aoarl.pdf .
- Seung-Jun Kim and Geogios B. Giannakis. An Online Convex Optimization Approach to Realtime Energy Pricing for Demand Response. IEEE Transactions on Smart Grid , 8(6):2784-2793, 2017. doi: 10.1109/TSG.2016.2539948. URL https://ieeexplore.ieee.org/document/ 7438918 .
- B Ravi Kiran, Ibrahim Sobh, Victor Talpaert, Patrick Mannion, Ahmad A. Al Sallab, Senthil Yogamani, and Patrick Pérez. Deep Reinforcement Learning for Autonomous Driving: A Survey. IEEE Transactions on Intelligent Transportation Systems , 23(6):4909-4926, 2022. doi: 10.1109/ TITS.2021.3054625. URL https://ieeexplore.ieee.org/document/9351818 .
- Robert Kleinberg. A multiple-choice secretary algorithm with applications to online auctions. In Proceedings of the Sixteenth Annual ACM-SIAM Symposium on Discrete Algorithms , SODA '05, page 630-631, USA, 2005. Society for Industrial and Applied Mathematics. ISBN 0898715857. URL https://dl.acm.org/doi/10.5555/1070432.1070519 .
- Jens Kober, J. Andrew Bagnell, and Jan Peters. Reinforcement Learning in Robotics: A Survey. The International Journal of Robotics Research , 32(11):1238-1274, 2013. doi: 10.1177/0278364913495721. URL https://doi.org/10.1177/0278364913495721 .
- Kasper Green Larsen, Omar Montasser, and Nikita Zhivotovskiy. Derandomizing multi-distribution learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://openreview.net/forum?id=twYE75Mnkt .
- Jean-Michel Lasry and Pierre-Louis Lions. Mean Field Games. Japanese Journal of Mathematics , 2(1):229-260, March 2007. ISSN 1861-3624. doi: 10.1007/s11537-007-0657-8. URL https: //link.springer.com/article/10.1007/s11537-007-0657-8 .
- Tor Lattimore, Csaba Szepesvari, and Gellert Weisz. Learning with good feature representations in bandits and in RL with a generative model. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 5662-5670. PMLR, 13-18 Jul 2020. URL https://proceedings. mlr.press/v119/lattimore20a.html .
- Martin Lauer and Martin A. Riedmiller. An algorithm for distributed reinforcement learning in cooperative multi-agent systems. In Proceedings of the Seventeenth International Conference on Machine Learning , ICML '00, page 535-542, San Francisco, CA, USA, 2000. Morgan Kaufmann Publishers Inc. ISBN 1558607072. URL https://dl.acm.org/doi/10.5555/645529.658113 .

- Gen Li, Yuting Wei, Yuejie Chi, Yuantao Gu, and Yuxin Chen. Sample Complexity of Asynchronous Q-Learning: Sharper Analysis and Variance Reduction. IEEE Transactions on Information Theory , 68(1):448-473, 2022. doi: 10.1109/TIT.2021.3120096. URL https://arxiv.org/abs/2006. 03041 .
- Minne Li, Zhiwei Qin, Yan Jiao, Yaodong Yang, Jun Wang, Chenxi Wang, Guobin Wu, and Jieping Ye. Efficient ridesharing order dispatching with mean field multi-agent reinforcement learning. In The World Wide Web Conference , WWW '19, page 983-994, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450366748. doi: 10.1145/3308558.3313433. URL https://doi.org/10.1145/3308558.3313433 .
- Yiheng Lin, Guannan Qu, Longbo Huang, and Adam Wierman. Distributed Reinforcement Learning in Multi-Agent Networked Systems. CoRR , abs/2006.06555, 2020. URL https://arxiv.org/ abs/2006.06555 .
- Yiheng Lin, James A Preiss, Emile Timothy Anand, Yingying Li, Yisong Yue, and Adam Wierman. Online Adaptive Policy Selection in Time-Varying Systems: No-Regret via Contractive Perturbations. In Thirty-seventh Conference on Neural Information Processing Systems , 2023a. URL https://openreview.net/forum?id=hDajsofjRM .
- Yiheng Lin, James A Preiss, Emile Timothy Anand, Yingying Li, Yisong Yue, and Adam Wierman. Learning-augmented Control via Online Adaptive Policy Selection: No Regret via contractive Perturbations. In ACM SIGMETRICS, Workshop on Learning-augmented Algorithms: Theory and Applications 2023 , 2023b. URL https://learning-augmented-algorithms.github.io/ papers/sigmetrics23-lata-posters-paper5.pdf .
- Yiheng Lin, James A. Preiss, Fengze Xie, Emile Anand, Soon-Jo Chung, Yisong Yue, and Adam Wierman. Online policy optimization in unknown nonlinear systems. In Shipra Agrawal and Aaron Roth, editors, Proceedings of Thirty Seventh Conference on Learning Theory , volume 247 of Proceedings of Machine Learning Research , pages 3475-3522. PMLR, 30 Jun-03 Jul 2024. URL https://proceedings.mlr.press/v247/lin24a.html .
- Michael L. Littman. Markov Games as a Framework for Multi-Agent Reinforcement Learning. In Machine learning proceedings , Elsevier, page 157-163, 1994. URL https://www.sciencedirect. com/science/article/abs/pii/B9781558603356500271 .
- P. Massart. The Tight Constant in the Dvoretzky-Kiefer-Wolfowitz Inequality. The Annals of Probability , 18(3):1269 -1283, 1990. doi: 10.1214/aop/1176990746. URL https://projecteuclid.org/journals/annals-of-probability/volume-18/ issue-3/The-Tight-Constant-in-the-Dvoretzky-Kiefer-Wolfowitz-Inequality/ 10.1214/aop/1176990746.full .
- Yifei Min, Jiafan He, Tianhao Wang, and Quanquan Gu. Cooperative Multi-Agent Reinforcement Learning: Asynchronous Communication and Linear Function Approximation. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 24785-24811. PMLR, 23-29 Jul 2023. URL https://proceedings.mlr.press/v202/min23a.html .
- Washim Uddin Mondal, Mridul Agarwal, Vaneet Aggarwal, and Satish V. Ukkusuri. On the Approximation of Cooperative Heterogeneous Multi-Agent Reinforcement Learning (MARL) Using Mean Field Control (MFC). Journal of Machine Learning Research , 23(1), jan 2022. ISSN 1532-4435. URL https://jmlr.org/papers/volume23/21-1312/21-1312.pdf .
- Michael Naaman. On the Tight Constant in the Multivariate Dvoretzky-Kiefer-Wolfowitz Inequality. Statistics &amp; Probability Letters , 173:109088, 2021. ISSN 0167-7152. doi: https://doi.org/10. 1016/j.spl.2021.109088. URL https://www.sciencedirect.com/science/article/pii/ S016771522100050X .
- Christos H. Papadimitriou and John N. Tsitsiklis. The Complexity of Optimal Queuing Network Control. Mathematics of Operations Research , 24(2):293-305, 1999. ISSN 0364765X, 15265471. URL https://www.jstor.org/stable/3690486 .

- James A. Preiss, Wolfgang Honig, Gaurav S. Sukhatme, and Nora Ayanian. Crazyswarm: A large nano-quadcopter swarm. In 2017 IEEE International Conference on Robotics and Automation (ICRA) , pages 3299-3304, 2017. doi: 10.1109/ICRA.2017.7989376. URL https://ieeexplore. ieee.org/document/7989376/ .
- Guannan Qu, Yiheng Lin, Adam Wierman, and Na Li. Scalable Multi-Agent Reinforcement Learning for Networked Systems with Average Reward. In Proceedings of the 34th International Conference on Neural Information Processing Systems , NIPS'20, Red Hook, NY, USA, 2020a. Curran Associates Inc. ISBN 9781713829546. URL https://proceedings.neurips.cc/paper/2020/ file/168efc366c449fab9c2843e9b54e2a18-Paper.pdf .
- Guannan Qu, Adam Wierman, and Na Li. Scalable Reinforcement Learning of Localized Policies for Multi-Agent Networked Systems. In Alexandre M. Bayen, Ali Jadbabaie, George Pappas, Pablo A. Parrilo, Benjamin Recht, Claire Tomlin, and Melanie Zeilinger, editors, Proceedings of the 2nd Conference on Learning for Dynamics and Control , volume 120 of Proceedings of Machine Learning Research , pages 256-266. PMLR, 10-11 Jun 2020b. URL https://proceedings. mlr.press/v120/qu20a.html .
- Omer Reingold. Undirected Connectivity in Log-Space. J. ACM , 55(4), sep 2008. ISSN 0004-5411. doi: 10.1145/1391289.1391291. URL https://doi.org/10.1145/1391289.1391291 .
- Zhaolin Ren, Runyu, Zhang, Bo Dai, and Na Li. Scalable spectral representations for network multiagent control, 2024. URL https://arxiv.org/abs/2410.17221 .
- Ross D. Shachter. Bayes-ball: The rational pastime (for determining irrelevance and requisite information in belief networks and influence diagrams), 2013.
- Lloyd S. Shapley. A value for n-person games. In Contributions to the Theory of Games , 1953. URL https://www.rand.org/content/dam/rand/pubs/papers/2021/P295.pdf .
- Aaron Sidford, Mengdi Wang, Xian Wu, Lin F. Yang, and Yinyu Ye. Near-optimal time and sample complexities for solving discounted markov decision process with a generative model. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 31, pages 7739-7750, 2018a.
- Aaron Sidford, Mengdi Wang, Xian Wu, and Yinyu Ye. Variance reduced value iteration and faster algorithms for solving markov decision processes. In Proceedings of the Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , New Orleans, LA, USA, 2018b. Society for Industrial and Applied Mathematics. doi: 10.1137/1.9781611975031.116.
- David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the Game of Go with Deep Neural Networks and Tree Search. Nature , 529(7587):484-489, January 2016. ISSN 14764687. doi: 10.1038/nature16961. URL https://www.nature.com/articles/nature16961 .
- Sriram Ganapathi Subramanian, Matthew E. Taylor, Mark Crowley, and Pascal Poupart. Decentralized mean field games, 2022. URL https://arxiv.org/abs/2112.09099 .
- Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In S. Solla, T. Leen, and K. Müller, editors, Advances in Neural Information Processing Systems , volume 12. MIT Press, 1999. URL https://proceedings.neurips.cc/paper\_files/paper/1999/file/ 464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf .
- Volodymyr Tkachuk, Seyed Alireza Bakhtiari, Johannes Kirschner, Matej Jusup, Ilija Bogunovic, and Csaba Szepesvári. Efficient planning in combinatorial action spaces with applications to cooperative multi-agent reinforcement learning. arXiV , 2023. URL https://proceedings.mlr. press/v206/tkachuk23a/tkachuk23a.pdf .
- Alexandre B. Tsybakov. Introduction to Nonparametric Estimation . Springer Publishing Company, Incorporated, 1st edition, 2008. ISBN 0387790519. URL https://link.springer.com/book/ 10.1007/b13794 .

- Martin J Wainwright. Variance-reduced q-learning is minimax optimal. arXiv preprint arXiv:1906.04697 , 2019.
- Jiangxing Wang, Deheng Ye, and Zongqing Lu. More centralized training, still decentralized execution: Multi-agent conditional policy factorization, 2023. URL https://arxiv.org/abs/ 2209.12681 .
- Christopher J. C. H. Watkins and Peter Dayan. Q-learning. Machine Learning , 8(3):279-292, May 1992. ISSN 1573-0565. doi: 10.1007/BF00992698. URL https://link.springer.com/ article/10.1007/BF00992698 .
- Mengfan Xu and Diego Klabjan. Decentralized randomly distributed multi-agent multi-armed bandit with heterogeneous rewards. In Thirty-seventh Conference on Neural Information Processing Systems , 2023. URL https://openreview.net/forum?id=DqfdhM64LI .
- Yaodong Yang, Rui Luo, Minne Li, Ming Zhou, Weinan Zhang, and Jun Wang. Mean Field MultiAgent Reinforcement Learning. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pages 5571-5580. PMLR, 10-15 Jul 2018. URL https://proceedings. mlr.press/v80/yang18d.html .
- Hongming Zhang, Tongzheng Ren, Chenjun Xiao, Dale Schuurmans, and Bo Dai. Provable representation with efficient planning for partially observable reinforcement learning. In Proceedings of the 41st International Conference on Machine Learning , ICML'24. JMLR.org, 2024. URL https://dl.acm.org/doi/10.5555/3692070.3694540 .
- Kaiqing Zhang, Zhuoran Yang, and Tamer Ba¸ sar. Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms, 2021. URL https://arxiv.org/abs/1911.10635 .
- Yihe Zhou, Shunyu Liu, Yunpeng Qing, Kaixuan Chen, Tongya Zheng, Yanhao Huang, Jie Song, and Mingli Song. Is centralized training with decentralized execution framework centralized enough for marl?, 2023. URL https://arxiv.org/abs/2305.17352 .

## A Mathematical Background and Additional Remarks

## Outline of the Appendices .

- Appendix B presents numerical simulations on the performance of SUBSAMPLE-MFQ on the Gaussian squeeze and constrained exploration tasks.
- Appendix C presents notation and basic lemmas involving the learned ˆ Q k -function, as well as a stable and practical implementation of Algorithm 1.
- Appendix D presents a proof sketch of our main result in Theorem 4.4.
- Appendix E presents the proof of the Lipschitz continuity between ˆ Q k and Q ∗
- Appendix F presents our bound on the TV-distance in Theorem D.2.
- Appendix G proves the bound on the optimality gap between the learned policy ˜ π est k,m and the optimal policy π ∗ .
- Appendix H presents an extension of the result to stochastic rewards.
- Appendix I presents an extension of the result to off-policy learning that follows by using historical data.
- Appendix J presents an extension of the result to continuous state/action spaces under a linear MDP assumption.
- Appendix K presents some practical derandomized variants with randomness-sharing.

Table 1: Important notations in this paper.

| Notation                                                                                                                                                                                                                                                                                                | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ∥ · ∥ 1 ∥ · ∥ ∞ Z + R d [ m ] ( [ m ] k ) a g s g a 1 , . . . ,a n s 1 , . . . , s n a s z i µ k ( Z l ) µ ( Z l ) s ∆ σ ( z ∆ , z ′ ∆ ) π ∗ ˆ π ∗ k ˜ π est k P g ( ·&#124; s g ,a g ) P l ( ·&#124; a i , s i , s g ) r g ( s g ,a g ) r l ( s i , s g ,a i ) r ( s,a ) r ∆ ( s,a ) T ˆ T k Π Θ ( y ) | ℓ 1 (Manhattan) norm; ℓ ∞ norm; The set of strictly positive integers; The set of d -dimensional reals; The set { 1 ,...,m } , where m ∈ Z + ; The set of k -sized subsets of { 1 ,...,m } ; a g ∈ A g is the action of the global agent; s g ∈ S g is the state of the global agent; a 1 , . . . ,a n ∈ A n l are the actions of the local agents 1 , . . .,n ; s 1 , . . . , s n ∈ S n l are the states of the local agents 1 , . . .,n ; a = ( a g ,a 1 , . . . ,a n ) ∈ A g ×A n l is the tuple of actions of all agents; s = ( s g , s 1 , . . . , s n ) ∈ S g ×S n l is the tuple of states of all agents; z i = ( s i ,a i ) ∈ Z l , for i ∈ [ n ] ; µ k ( Z l ) = { 0 , 1 /k, 2 /k, . . . , 1 } &#124;Z l &#124; ; µ ( Z l ) := µ n ( Z l ) { 0 , 1 /n, 2 /n, . . . , 1 } &#124;Z l &#124; ; For ∆ ⊆ [ n ] , and a collection of variables { s 1 , . . . , s n } , s ∆ := { s i : i ∈ ∆ } ; Product sigma-algebra generated by sequences z ∆ and z ′ ∆ ; π ∗ is the optimal deterministic policy function such that a = π ∗ ( s ) ; ˆ π ∗ k is the optimal deterministic policy function on a constrained system of &#124; ∆ &#124; = k local agents; ˜ π est k is the stochastic policy map learned with parameter k such that a ∼ ˜ π est k ( s ) ; P g ( ·&#124; s g ,a g ) is the stochastic transition kernel for the state of the global agent; P l ( ·&#124; a i , s i , s g ) is the stochastic transition kernel for the state of local agent i ∈ [ n ]; r g is the global agent's component of the reward; r l is the component of the reward for local agent i ∈ [ n ] ; r ( s,a ) = r g ( s g ,a g )+ 1 n ∑ i ∈ [ n ] r l ( s i , s g ,a i ) is the reward of the system; r ∆ ( s,a ) = r g ( s g ,a g )+ 1 &#124; ∆ &#124; ∑ i ∈ ∆ r l ( s i , s g ,a i ) is the constrained system's reward with &#124; ∆ &#124; = k local agents; T is the centralized Bellman operator; ˆ T k is the Bellman operator on a constrained system of &#124; ∆ &#124; = k local agents; ℓ 1 projection of y onto set Θ ; |

Definition A.1 (Lipschitz continuity) . Given metric spaces ( X , d X ) and ( Y , d Y ) and a constant L &gt; 0 , a map f : X → Y is L -Lipschitz continuous if for all x, y ∈ X , d Y ( f ( x ) , f ( y )) ≤ L · d X ( x, y ) .

Theorem A.2 (Banach-Caccioppoli fixed point theorem [Banach, 1922]) . Consider the metric space ( X , d X ) , and T : X → X such that T is a γ -Lipschitz continuous mapping for γ ∈ (0 , 1) . Then, by the Banach-Cacciopoli fixed-point theorem, there exists a unique fixed point x ∗ ∈ X for which T ( x ∗ ) = x ∗ . Additionally, x ∗ = lim s →∞ T s ( x 0 ) for any x 0 ∈ X .

## B Numerical Experiments

This section provides numerical simulations for the examples outlined in Section 2. All experiments were run on a 2-core CPU server with 12GB RAM. We chose a parameter complexity for each simulation that was sufficient to emphasize characteristics of the theory, such as the complexity improvement and the decaying optimality gap. 4

Figure 2: a) Reward optimality gap (log scale) with π est k,m running 300 iterations. b) Computation time (in minutes) against sampling parameter k , for k ≤ n = 8 , to learn policy ˆ π est k,m . c) Discounted cumulative rewards for k ≤ n = 50 .

<!-- image -->

## B.1 Constrained Exploration

Let S g = S l = [6] 2 where for s a ∈ {S g , S l } , s ( i ) a denotes the projection of s a onto its i 'th coordinate. Further, let A l = A g = { 'up", 'down", 'right", 'left" } given formally by { ( 1 0 ) , ( -1 0 ) , ( 0 1 ) , ( 0 -1 ) } . Let Π D ( x ) denote the ℓ 1 -projection of x onto set D . Then, let s t +1 g ( s t g , a t g ) = Π S g ( s t g + a t g ) , s t +1 i ( s t i , s t g , a t i ) = Π S l ( s t i + | s t i -s t g | + a t i ) . We let the global agent's reward be r g ( s g , a g ) = 2 ∑ 3 i =1 ✶ { a g = ( 1 0 ) , s (1) g ≤ i } +2 ∑ 6 i =3 ✶ { a g = ( -1 0 ) , s (1) g &gt; i } , and the local agent's rewards be r l ( s i , s g , a i ) = { a t i = 0 } +6 · { s (1) i = s (1) g } +2 · ∥ s t i -s t g ∥ 1 .

✶ ✶ Intuitively, the reward function is designed to force the global agent to oscillate vertically in the grid, while forcing the local agents to oscillate horizontally around the global agent, thereby simulating a constrained exploration. This model has been studied previously in Lin et al. [2023b].

̸

For this task, we ran a simulation with n = 8 agents, with m = 20 samples in the empirically adapted Bellman operator. We provide simulation results in Figure 2a. We observe monotonic improvements in the cumulative discounted rewards as k → n . Since k = n recovers value-iteration and mean-field MARL algorithms, the reward at k = n is the baseline we compare our algorithm to. When k &lt; n , we observe that the reward accrued by SUBSAMPLE-MFQ is only marginally less than the reward gained by value-iteration.

## B.2 Gaussian Squeeze (GS)

In this task, n homogeneous agents determine their individual action a i to jointly maximize the objective r ( x ) = xe -( x -µ ) 2 /σ 2 , where x = ∑ n i =1 a i , a i ∈ { 0 , . . . , 9 } , and µ and σ are the predefined mean and variance of the system. We consider a homogeneous system devoid of a global

4 We provide supporting code for the algorithm and experiments in https://github.com/emiletimothy/ Mean-Field-Subsample-Q-Learning

agent to match the mean-field setting in Yang et al. [2018], where the richness of our model setting can still express GS. We use the global agent to model the state of the system, given by the number of vehicles in each controller's lane.

We set S l = [4] , A l = { 0 , . . . , 4 } , and S g = S l such that s g = ⌈ 1 n ∑ n i =1 s i ⌉ , and A g = { 0 } . The transition functions are given by s i ( t + 1) = s i ( t ) -1 · ✶ { s i ( t ) &gt; s g ( t ) } + Ber( p ) and s g ( t +1) = ⌈ 1 n ∑ n i =1 s i ( t +1) ⌉ , where Ber( p ) is a Bernoulli random variable with parameter p &gt; 0 . Finally, the global agent's reward function is given by r g ( s g , a g ) = -s g and the local agent's reward function is given by r l ( s i , s g , a i ) = s i · e -( s i -s g ) 2 / 4 -max { s i , a i } .

For this task, we ran a small-scale simulation with n = 8 agents, and a large-scale simulation with n = 50 agents, and used m = 20 samples in the empirical Bellman operator. We provide simulation results in Figure 2b and Figure 2c, where Figure 2b demonstrates the exponential improvement in computational complexity of SUBSAMPLE-MFQ , and Figure 2c demonstrates a monotonic improvement in the cumulative rewards, consistent with Theorem 4.4. Here, both metrics outperform the mean-field value iteration benchmark.

## C Notation and Basic Lemmas

For convenience, we restate below the various Bellman operators under consideration.

Definition C.1 (Bellman Operator T ) .

<!-- formula-not-decoded -->

Definition C.2 (Adapted Bellman Operator ˆ T k ) . The adapted Bellman operator updates a smaller Q function (which we denote by ˆ Q k ), for a surrogate system with the global agent and k ∈ [ n ] local agents denoted by ∆ , using mean-field value iteration and j ∈ ∆ such that:

<!-- formula-not-decoded -->

Definition C.3 (Empirical Adapted Bellman Operator ˆ T k,m ) . The empirical adapted Bellman operator ˆ T k,m empirically estimates the adapted Bellman operator update using mean-field value iteration by drawing m random samples of s g ∼ P g ( ·| s g , a g ) and s i ∼ P l ( ·| s i , s g , a i ) for i ∈ ∆ , where for ℓ ∈ [ m ] , the ℓ 'th random sample is given by s ℓ g and s ℓ ∆ , and j ∈ ∆ :

<!-- formula-not-decoded -->

Lemma C.4. For any ∆ ⊆ [ n ] such that | ∆ | = k , suppose 0 ≤ r ∆ ( s, a ) ≤ ˜ r . Then, for all t ∈ N , ˆ Q t k ≤ ˜ r 1 -γ .

Proof. The proof follows by induction on t . The base case follows from ˆ Q 0 k := 0 . For the induction, note that by the triangle inequality ∥ ˆ Q t +1 k ∥ ∞ ≤ ∥ r ∆ ∥ ∞ + γ ∥ ˆ Q t k ∥ ∞ ≤ ˜ r + γ ˜ r 1 -γ = ˜ r 1 -γ .

Remark C.5. By the law of large numbers, lim m →∞ ˆ T k,m = ˆ T k , where the error decays in O (1 / √ m ) by the Chernoff bound. Also, ˆ T n := T . Further, Lemma C.4 is independent of the choice of k . Therefore, for k = n , this implies an identical bound on Q t . An identical argument implies the same bound on ˆ Q t k,m .

T satisfies a γ -contractive property under the infinity norm [Watkins and Dayan, 1992]. We similarly show that ˆ T k and ˆ T k,m satisfy a γ -contractive property under infinity norm in Lemmas C.6 and C.7.

Lemma C.6. ˆ T k satisfies the γ -contractive property under infinity norm:

<!-- formula-not-decoded -->

Proof. Suppose we apply ˆ T k to ˆ Q k ( s g , F z ∆ , a g ) and ˆ Q ′ k ( s g , F z ∆ , a g ) for | ∆ | = k . Then:

∥ ˆ T ˆ Q ′ -ˆ T ˆ Q ∥

<!-- formula-not-decoded -->

The equality cancels common r ∆ ( s, a ) terms in each operator. The second line uses Jensen's inequality, maximizes over actions, and bounds expected values with the maximizers of the random variables.

Lemma C.7. ˆ T k,m satisfies the γ -contractive property under infinity norm.

Proof. Similarly to Lemma C.6, suppose we apply ˆ T k,m to ˆ Q k,m ( s g , F z ∆ , a g ) and ˆ Q ′ k,m ( s g , F z ∆ , a g ) . Then:

<!-- formula-not-decoded -->

The first inequality uses the triangle inequality and the general property | max a ∈ A f ( a ) -max b ∈ A f ( b ) | ≤ max c ∈ A | f ( a ) -f ( b ) | . The last line recovers the definition of infinity norm.

Remark C.8. The γ -contractivity of ˆ T k and ˆ T k,m attracts the trajectory between two ˆ Q k and ˆ Q k,m functions on the same state-action tuple by γ at each step. Repeatedly applying the Bellman operators produces a unique fixed-point from the Banach fixed-point theorem which we introduce in Definitions C.9 and C.10.

Definition C.9 ( ˆ Q ∗ k -function) . Suppose ˆ Q 0 k := 0 and let ˆ Q t +1 k ( s g , F z ∆ , a g ) = ˆ T k ˆ Q t k ( s g , F z ∆ , a g ) for t ∈ N . Denote the fixed-point of ˆ T k by ˆ Q ∗ k such that ˆ T k ˆ Q ∗ k ( s g , F z ∆ , a g ) = ˆ Q ∗ k ( s g , F z ∆ , a g ) .

Definition C.10 ( ˆ Q est k,m -function) . Let ˆ Q 0 k,m := 0 and ˆ Q t +1 k,m ( s g , F z ∆ , a g ) = ˆ T k,m ˆ Q t k,m ( s g , F z ∆ , a g ) for t ∈ N . Then, the Banach-Cacciopoli fixed-point of the adapted Bellman operator ˆ T k,m is ˆ Q est k,m such that ˆ T k,m ˆ Q est k,m ( s g , F z ∆ , a g ) = ˆ Q est k,m ( s g , F z ∆ , a g ) .

Corollary C.11. Observe that by recursively using the γ -contractive property for T time steps:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Further, noting that ˆ Q 0 k = ˆ Q 0 k,m := 0 , ∥ ˆ Q ∗ k ∥ ∞ ≤ ˜ r 1 -γ , and ∥ ˆ Q est k,m ∥ ∞ ≤ ˜ r 1 -γ from Lemma C.4:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark C.12. Corollary C.11 characterizes the error decay between ˆ Q T k and ˆ Q ∗ k and shows that it decays exponentially in the number of Bellman iterations by a γ T multiplicative factor.

Furthermore, we characterize the maximal policies greedy policies obtained from Q ∗ , ˆ Q ∗ k , and ˆ Q est k,m .

Definition C.13 (Optimal policy π ∗ ) . The greedy policy derived from Q ∗ is

<!-- formula-not-decoded -->

Definition C.14 (Optimal subsampled policy ˆ π ∗ k ) . The greedy policy from ˆ Q ∗ k is

<!-- formula-not-decoded -->

Definition C.15 (Optimal empirically subsampled policy ˆ π est k,m ) . The greedy policy from ˆ Q est k,m is given by

<!-- formula-not-decoded -->

Figure 4 details the analytic flow on how we use the empirical adapted Bellman operator to perform value iteration on ˆ Q k,m to get ˆ Q est k,m which approximates Q ∗ .

Algorithm 3 gives a stable implementation of Algorithm 1 with learning rates { η t } t ∈ [ T ] . Algorithm 3 is provably numerical stable under fixed-point arithmetic [Anand et al., 2024, 2025]. ˆ Q t k,m in Algorithm 3 is γ -contractive as in Lemma C.6, given an appropriately conditioned sequence of learning rates η t :

## Algorithm 3 Stable (Practical) Implementation of Algorithm 1: SUBSAMPLE-Q : Learning

Require: A multi-agent system as described in Section 2. Parameter T for the number of iterations in the initial value iteration step. Hyperparameter k ∈ [ n ] . Discount parameter γ ∈ (0 , 1) . Oracle O to sample s ′ g ∼ P g ( ·| s g , a g ) and s i ∼ P l ( ·| s i , s g , a i ) for all i ∈ [ n ] . Learning rate sequence { η t } t ∈ [ T ] where η t ∈ (0 , 1] .

- 1: Let ∆ = [ k ] .
- 2: for ( s g , s 1 , F z ∆ \ 1 ) ∈ S g ×S l × µ k -1 ( Z l ) do
- 3: for ( a g , a 1 ) ∈ A g ×A l do
- 4: Set ˆ Q 0 k,m ( s g , s 1 , F z ∆ \ 1 , a 1 , a g ) = 0
- 5: for t = 1 to T do
- 6: for ( s g , s 1 , F z ∆ \ 1 ) ∈ S g ×S l × µ k -1 ( Z l ) do
- 7: for ( a g , a 1 ) ∈ A g ×A l do

8:

<!-- formula-not-decoded -->

- 9: Let the approximate policy be

<!-- formula-not-decoded -->

Theorem C.16. As T →∞ , if ∑ T t =1 η t = ∞ , and ∑ T t =1 η 2 t &lt; ∞ , then Q -learning converges to the optimal Q function asymptotically with probability 1 .

Furthermore, finite-time guarantees with the learning rate and sample complexity have been shown in Chen and Maguluri [2022b], which when adapted to our ˆ Q k,m framework in Algorithm 3 yields:

Theorem C.17 (Chen and Maguluri [2022b]) . For all t = { 1 , . . . , T } and for any ϵ &gt; 0 , if the learning rate sequence η t satisfies η t = (1 -γ ) 4 ϵ 2 and T = |S l ||A l | k |S l ||A l | |S g ||A g | (1 -γ ) 5 ϵ 2 , then

<!-- formula-not-decoded -->

Definition C.18 (Star Graph S n ) . For n ∈ N , the star graph S n is the complete bipartite graph K 1 ,n .

S n captures the graph density notion by saturating the set of neighbors of the central node. Such settings find applications beyond RL as well [Chaudhari et al., 2024, Anand and Qu, 2024, Li et al., 2019]. The cardinality of the search space simplex for the optimal policy is exponential in n , so it cannot be naively modeled by an MDP: we need to exploit the symmetry of the local agents. This intuition allows our subsampling algorithm to run in polylogarithmic time (in n ). Some works leverage an exponential decaying property that truncates the search space for policies over immediate neighborhoods of agents; however, this still relies on the assumption that the graph neighborhood for the agent is sparse [Qu et al., 2020a,b]; however, S n is not locally sparse; hence, previous methods do not apply to this problem instance.

Figure 3: Star graph S n

<!-- image -->

Finally, we argue why mean-field value iteration faithfully performs the same update as value iteration.

Lemma C.19 (Equivalence of Mean-Field Value Iteration and Value Iteration) . Since the local agents 1 , . . . , n are all homogeneous in their state/action spaces, the ˆ Q k -function only depends on them through their empirical distribution F z ∆ , proving the lemma. Therefore, for the remainder of the paper, we will use ˆ Q k ( s g , F z ∆ , a g ) := ˆ Q k ( s g , s i , F z ∆ \ i , a g , a i ) := ˆ Q k ( s g , s ∆ , a g , a ∆ ) interchangeably, unless making a remark about the computational complexity of learning each function.

## D Proof Sketch

This section details an outline for the proof of Theorem 4.4, as well as some key ideas. At a high level, our SUBSAMPLE-MFQ framework recovers exact mean-field Q learning and traditional value iteration when k = n and as m →∞ . Further, as k → n , ˆ Q ∗ k should intuitively get closer to Q ∗ from which the optimal policy is derived. Thus, the proof is divided into three major steps: firstly, we prove a Lipschitz continuity bound between ˆ Q ∗ k and ˆ Q ∗ n in terms of the total variation (TV) distance between F z ∆ and F z [ n ] . Next, we bound the TV distance between F z ∆ and F z [ n ] . Finally, we bound the value differences between π est k and π ∗ by bounding Q ∗ ( s, π ∗ ( s )) -Q ∗ ( s, π est k ( s )) and then using the performance difference lemma from Kakade and Langford [2002].

Step 1: Lipschitz Continuity Bound. To compare ˆ Q ∗ k ( s g , F s ∆ , a g ) with Q ∗ ( s, a g ) , we prove a Lipschitz continuity bound between ˆ Q ∗ k ( s g , F s ∆ , a g ) and ˆ Q ∗ k ′ ( s g , F s ∆ ′ , a g ) with respect to the TV distance measure between s ∆ ∈ ( s [ n ] k ) and s ∆ ′ ∈ ( s [ n ] k ′ ) :

Theorem D.1 (Lipschitz continuity in ˆ Q ∗ k ) . For all ( s, a ) ∈ S × A , ∆ ∈ ( [ n ] k ) and ∆ ′ ∈ ( [ n ] k ′ ) ,

<!-- formula-not-decoded -->

We defer the proof of Theorem D.1 to Appendix E. See Figure 4 for a comparison between the ˆ Q ∗ k learning and estimation process, and the exact Q -learning framework.

Step 2: Bounding Total Variation (TV) Distance. We bound the TV distance between F z ∆ and F z [ n ] , where ∆ ∈U ( [ n ] k ) . This task is equivalent to bounding the discrepancy between the empirical distribution and the distribution of the underlying finite population. When each i ∈ ∆ is uniformly sampled without replacement, we use Lemma F.3 from Anand and Qu [2024] which generalizes the Dvoretzky-Kiefer-Wolfowitz (DKW) concentration inequality for empirical distribution functions.

Using this, we show:

<!-- formula-not-decoded -->

Figure 4: Flow of the algorithm and relevant analyses in learning Q ∗ . Here, (1) follows by performing Algorithm 1 ( SUBSAMPLE-MFQ : Learning) on ˆ Q 0 k,m . (2) follows from Lemma 4.3. (3) follows from the Lipschitz continuity and total variation distance bounds in Theorems D.1 and D.2. Finally, (4) follows from noting that ˆ Q ∗ n = Q ∗ .

Theorem D.2. Given a finite population Z = ( z 1 , . . . , z n ) for Z ∈ Z n l , let ∆ ⊆ [ n ] be a uniformly random sample from Z of size k chosen without replacement. Fix ϵ &gt; 0 . Then, for all x ∈ Z l :

<!-- formula-not-decoded -->

Then, by Theorem D.2 and the definition of total variation distance from Section 2, we have that for δ ∈ (0 , 1] , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

We then apply this result to our MARL setting by studying the rate of decay of the objective function between the learned policy π est k and the optimal policy π ∗ (Theorem 4.4).

Step 3: Performance Difference Lemma to Complete the Proof. As a consequence of the prior two steps and Lemma 4.3, Q ∗ ( s, a ′ ) and ˆ Q est k,m ( s g , F z ∆ , a ′ g ) become similar as k → n . We further prove that the value generated by the policies π ∗ and π est k must also be very close (where the residue shrinks as k → n ). We then use the well-known performance difference lemma [Kakade and Langford, 2002] which we restate in Appendix G.1. A crucial theorem needed to use the performance difference lemma is a bound on Q ∗ ( s ′ , π ∗ ( s ′ )) -Q ∗ ( s ′ , ˆ π est k ( s ′ g , F s ′ ∆ )) .

Therefore, we formulate and prove Theorem D.3 which yields a probabilistic bound on this difference, where the randomness is over the choice of ∆ ∈ ( [ n ] k ) :

Theorem D.3. For a fixed s ′ ∈ S := S g × S n l and for δ ∈ (0 , 1] , with probability at least 1 -2 |A g | k |A l | δ :

<!-- formula-not-decoded -->

We defer the proof of Theorem D.3 and finding optimal value of δ to Lemma G.8 in the Appendix. Using Theorem D.3 and the performance difference lemma leads to Theorem 4.4.

## E Proof of Lipschitz-continuity Bound

This section proves a Lipschitz-continuity bound between ˆ Q ∗ k and Q ∗ and includes a framework to compare 1 ( n k ) ∑ ∆ ∈ ( [ n ] k ) ˆ Q ∗ k ( s g , s ∆ , a g ) and Q ∗ ( s, a g ) in Lemma E.12.

Let z i := ( s i , a i ) ∈ Z l := S l ×A l and z ∆ = { z i : i ∈ ∆ } ∈ Z k l . For z i = ( s i , a i ) , let z i ( s ) = s i , and z i ( a ) = a i . With abuse of notation, note that ˆ Q T k ( s g , a g , s ∆ , a ∆ ) is equal to ˆ Q T k ( s g , a g , z ∆ ) .

The following definitions will be relevant to the proof of Theorem E.3.

Definition E.1 (Empirical Distribution Function) . For all z ∈ Z | ∆ | l and ( s ′ , a ′ ) ∈ S l ×A l , where ∆ ⊆ [ n ] ,

<!-- formula-not-decoded -->

Definition E.2 (Total Variation Distance) . Let P and Q be discrete probability distribution over some domain Ω . Then,

<!-- formula-not-decoded -->

Theorem E.3 ( ˆ Q T k is 2 1 -γ ∥ r l ( · , · ) ∥ ∞ -Lipschitz continuous with respect to F z ∆ in total variation distance) . Suppose ∆ , ∆ ′ ⊆ [ n ] such that | ∆ | = k and | ∆ ′ | = k ′ . Then:

<!-- formula-not-decoded -->

Proof. We prove this inductively. First, note that ˆ Q 0 k ( · , · , · ) = ˆ Q 0 k ′ ( · , · , · ) = 0 from the initialization step, which proves the lemma for T = 0 , since TV( · , · ) ≥ 0 . At T = 1 :

<!-- formula-not-decoded -->

In the first and second equalities, we use the time evolution property of ˆ Q 1 k and ˆ Q 1 k ′ by applying the adapted Bellman operators ˆ T k and ˆ T k ′ to ˆ Q 0 k and ˆ Q 0 k ′ , respectively, and expanding. In the third and fourth equalities, we note that ˆ Q 0 k ( · , · , · ) = ˆ Q 0 k ′ ( · , · , · ) = 0 , and subtract the common 'global component' of the reward function.

Then, noting the general property that for any function f : X → Y for |X| &lt; ∞ we can write f ( x ) = ∑ y ∈X f ( y ) { y = x } , we have:

<!-- formula-not-decoded -->

The second equality follows from the linearity of expectations, and the third equality follows by noting that for any random variable X ∼ X , E X ✶ [ X = x ] = Pr[ X = x ] . The first inequality follows from an application of the triangle inequality and the Cauchy-Schwarz inequality, and the second inequality uses the definition of TV distance. Thus, at T = 1 , ˆ Q is (2 ∥ r l ( · , · ) ∥ ∞ ) -Lipschitz continuous in TV distance, proving the base case.

Assume that for T ≤ t ′ ∈ N :

<!-- formula-not-decoded -->

Then, inductively:

<!-- formula-not-decoded -->

In the first inequality, we rewrite the expectations over the states as the expectation over the joint transition probabilities. The second inequality then follows from Lemma E.14. To apply it to Lemma E.14, we conflate the joint expectation over ( s g , s ∆ ∪ ∆ ′ ) and reduce it back to the original form of its expectation. Finally, the third inequality follows from Lemma E.5.

<!-- formula-not-decoded -->

Definition E.4. [Joint Stochastic Kernels] The joint stochastic kernel on ( s g , s ∆ ) for ∆ ⊆ [ n ] where | ∆ | = k is defined as J k : S g ×S k l ×S g ×A g ×S k l ×A k l → [0 , 1] , where

<!-- formula-not-decoded -->

Lemma E.5. For all T ∈ N , for any a g ∈ A g , s g ∈ S g , s ∆ ∈ S k l , a ∆ ∈ A k l , a ′ ∆ ∈ A k l , and for all joint stochastic kernels J k as defined in Definition E.4:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We prove this inductively. At T = 0 , the statement is true since ˆ Q 0 k ( · , · , · ) = ˆ Q 0 k ′ ( · , · , · ) = 0 and TV( · , · ) ≥ 0 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, we have:

<!-- formula-not-decoded -->

The first equality follows from noting that f ( x ) = ∑ x ′ ∈X f ( x ′ ) ✶ { x = x ′ } and from Fubini-Tonelli's inequality which allows us to swap the order of summations as the summand is finite. The second equality uses the law of total expectation. The first inequality uses Jensen's inequality and the triangle inequality. The second inequality uses ∥ ˜ r l ( · , · ) ∥ ∞ ≤ ∥ r l ( · , · ) ∥ ∞ which holds as ∥ r l ∥ ∞ is the infinite-norm of the local reward functions and is therefore atleast as large any other element in the image of r l . The third inequality follows from the definition of total variation distance, and the final inequality follows from Lemma E.7. This proves the base case.

Then, assume that for T ≤ t ′ ∈ N , for all joint stochastic kernels J k and J k ′ , and for all a ′ g ∈ A g , a ′ ∆ ∈ A k l :

<!-- formula-not-decoded -->

For the remainder of the proof, we adopt the shorthand E ( s ′ g ,s ′ ∆ ) ∼J to denote E ( s ′ g ,s ′ ∆ ) ∼J | ∆ | ( · , ·| s g ,a g ,s ∆ ,a ∆ ) , and E ( s ′′ g ,s ′′ ∆ ) ∼J to denote E ( s ′′ g ,s ′′ ∆ ) ∼J | ∆ | ( · , ·| s ′ g ,a ′ g ,s ′ ∆ ,a ′ ∆ ) . Then, inductively, we have:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Let

<!-- formula-not-decoded -->

and

ˆ a g , ˆ a ∆ ′

<!-- formula-not-decoded -->

Then, define ˜ a ∆ ∪ ∆ ′ ∈ A | ∆ ∪ ∆ ′ | l by

Suppose (I) ≥ (II). Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, define ˆ a ∆ ∪ ∆ ′ ∈ A | ∆ ∪ ∆ ′ | l by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first equality rewrites the equations with their respective maximizing actions. The first inequality upper-bounds this difference by allowing all terms to share the common action ˜ a . Using the Bellman equation, the second equality expands ˆ Q T +1 k and ˆ Q T +1 k ′ . The second inequality follows from the triangle inequality. The third equality follows by subtracting the common r g ( s ′ g , ˜ a g ) terms from the reward and noting that the two expectation terms E s ′ g ,s ′ ∆ ∼ J k ( · , ·| s g , a g , s ∆ , a ∆ ) E s ′′ g ,s ∆ ′′ ∼J k ( · , ·| s g , ˜ a g ,s ∆ , ˜ a ∆ ) can be combined into a single expectation E s ′′ g ,s ′′ ∆ ∼ ˜ J k ( · , ·| s g ,a g ,s ∆ ,a ∆ ) .

In the special case where a ∆ = ˜ a ∆ , we derive a closed form expression for ˜ J k in Lemma E.11. To justify the third inequality, the second term follows from the induction hypothesis and the first term

follows from the following derivation:

<!-- formula-not-decoded -->

The above derivation follows from the same argument as in the base case where r ˜ a l ( s ′ i , s ′ g ) := r l ( s ′ i , s ′ g , ˜ a i ) for any i ∈ ∆ ∪ ∆ ′ . Similarly, if (I) &lt; (II), an analogous argument that replaces ˜ a ∆ with ˆ a ∆ yields the same result.

Therefore, by the induction hypothesis, the claim is proven.

Remark E.6. Given a joint transition probability function J | ∆ ∪ ∆ ′ | as defined in Definition E.4, we can recover the transition function for a single agent i ∈ ∆ ∪ ∆ ′ given by J 1 using the law of total probability and the conditional independence between s i and s g ∪ s [ n ] \ i in Equation (28). This characterization is crucial in Lemma E.7 and Lemma E.5:

<!-- formula-not-decoded -->

Here, we use a conditional independence property proven in Lemma E.10.

Lemma E.7. The TV-distance between the next-step expected empirical distribution functions is bounded by the TV-distance between the existing empirical distribution functions.

<!-- formula-not-decoded -->

Proof. From the definition of total variation distance, we have:

<!-- formula-not-decoded -->

The second equality uses Lemma E.9. The third equality uses the property that each local agent can only have one state/action pair, and the fourth equality vectorizes the indicator variables. Then,

<!-- formula-not-decoded -->

The first inequality uses the triangle inequality. The second inequality and fifth equality follow from Hölder's inequality and the sum of the probabilities from the stochastic transition function J 1 being equal to 1 . The sixth equality uses Fubini-Tonelli's theorem which applies as the total variation distance measure is bounded from above by 1 . The final equality recovers the total variation distance for the variable z = ( s, a ) ∈ S l ×A l across agents ∆ and ∆ ′ , which proves the claim.

Next, recall the data processing inequality.

Lemma E.8 (Data Processing Inequality.) . Let A and B be random variables over some domain S . Let f be some function (not necessarily deterministically) mapping from S to any codomain T . Then, every f -divergence χ satisfies

<!-- formula-not-decoded -->

Remark: We show an analog of the data processing inequality. Under this lens, Lemma E.7 saturates the data-processing relation for the TV distance in our multi-agent setting.

Lemma E.9.

<!-- formula-not-decoded -->

Proof. By expanding on the definition of the empirical distribution function F s ∆ ′ ( x ) = 1 | ∆ | ∑ i ∈ ∆ ✶ { s i = x } , we have:

<!-- formula-not-decoded -->

The second equality follows from the conditional independence of s ′ i from s ∆ ∪ ∆ ′ \ i from Lemma E.10, and the final equality uses the fact that the expectation of an indicator random variable is the probability distribution function of the random variable.

Lemma E.10. The distribution s ′ ∆ ∪ ∆ ′ \ i | s ′ g , s g , a g , s ∆ ∪ ∆ ′ , a ∆ ∪ ∆ ′ is conditionally independent to the distribution s ′ i | s ′ g , s g , a g , s i , a i for any i ∈ ∆ ∪ ∆ ′ .

Proof. We direct the interested reader to the Bayes-Ball theorem in Shachter [2013] for proving conditional independence. For ease of exposition, we restate the two rules in Shachter [2013] that introduce the notion of d -separations which implies conditional independence. Suppose we have a causal graph G = ( V, E ) where the vertex set V = [ p ] for p ∈ N is a set of variables and the edge set E ⊆ 2 V denotes dependence through connectivity. Then, the two rules that establish d -separations are as follows:

1. For x, y ∈ V , we say that x, y are d -connected if there exists a path ( x, . . . , y ) that can be traced without traversing a pair of arrows that point at the same vertex.
2. We say that x, y ∈ V are d -connected, conditioned on a set of variables Z ⊆ V , if there is a path ( x, . . . , y ) that does not contain an event z ∈ Z that can be traced without traversing a pair of arrows that point at the same vertex.

If x, y ∈ V is not d -connected through any such path, then x, y is d -separated, which implies conditional independence.

Let Z = { s ′ g , s g , a g , s i , a i } be the set of variables we condition on. Then, the below figure demonstrates the causal graph for the events of interest.

Figure 5: Causal graph to demonstrate the dependencies between variables.

<!-- image -->

1. Observe that all paths (through undirected edges) stemming from s ′ ∆ ∪ ∆ ′ \ i → s ∆ ∪ ∆ ′ \ i pass through s i ∈ Z which is blocked.
2. All other paths from s ′ ∆ ∪ ∆ ′ \ i to s ′ i pass through s g ∪ s ′ g ∈ Z .

Therefore, s ′ ∆ ∪ ∆ ′ \ i and s ′ i are d -separated by Z . Hence, by Shachter [2013], s ′ ∆ ∪ ∆ ′ \ i and s ′ i are conditionally independent.

Lemma E.11. For any joint transition probability function J k on s g ∈ S g , s ∆ ∈ S k l , a ∆ ∈ A k l where | ∆ | = k , given by J k : S g ×S | ∆ | l ×S g ×A g ×S | ∆ | l ×A | ∆ | l → [0 , 1] , we have:

<!-- formula-not-decoded -->

Proof. By expanding the expectations:

<!-- formula-not-decoded -->

In the second equality, the right-stochasticity of J k implies the right-stochasticity of J 2 k .

Further, observe that J k [ s ′ g , s ′ ∆ , s g , a g , s ∆ , a ∆ ] J k [ s ′′ g , s ′′ ∆ , s ′ g , a g , s ′ ∆ , a ∆ ] denotes the probability of the transitions ( s g , s ∆ ) → ( s ′ g , s ′ ∆ ) → ( s ′′ g , s ′′ ∆ ) with actions a g , a ∆ at each step, where the joint state evolution is governed by J k . Thus,

<!-- formula-not-decoded -->

since ∑ ( s ′ g ,s ′ ∆ ) ∈S g ×S | ∆ | l J k [ s ′ g , s ′ ∆ , s g , a g , s ∆ , a ∆ ] J k [ s ′′ g , s ′′ ∆ , s ′ g , a g , s ′ g , a ∆ ] is the stochastic probability function corresponding to the two-step evolution of the joint states from ( s g , s ∆ ) to ( s ′′ g , s ′′ ∆ ) under actions a g , a ∆ . This can be thought of as an analogously to the fact that a 1 on the ( i, j ) 'th entry on the square of a 0 / 1 adjacency matrix of a graph represents the fact that there is a path of distance 2 between vertices i and j .

Finally, the third equality recovers the definition of the expectation, with respect to the joint probability function J 2 k .

<!-- formula-not-decoded -->

Lemma E.12. The Q ∗ function is the average value of each of the ( n k ) sub-sampled ˆ Q ∗ k -functions.

<!-- formula-not-decoded -->

Proof. We bound the differences between ˆ Q T k at each Bellman iteration of our approximation to Q ∗ . Note that:

<!-- formula-not-decoded -->

To prove this, we write:

<!-- formula-not-decoded -->

In the second equality, we reparameterized the sum to count the number of times each r l ( s i , s g ) was added for each i ∈ ∆ . To do this, we count the number of ( k -1) other agents that could form a k -tuple with agent i , and there are ( n -1)) candidates from which we chooses the ( k -1) agents. In the last equality, we expanded and simplified the binomial coefficients.

So, we have that:

<!-- formula-not-decoded -->

We justify the first inequality by noting the general property that for positive vectors v, v ′ for which v ⪰ v ′ which follows from the triangle inequality:

<!-- formula-not-decoded -->

Thus, applying this bound recursively, we get:

<!-- formula-not-decoded -->

The first inequality follows from the γ -contraction property of the update procedure, and the ensuing equality follows from our bound on the maximum possible value of Q from Lemma C.4 and noting that ˆ Q 0 k := 0 .

Therefore, as T →∞ , which proves the lemma.

Corollary E.13. Since 1 ( n k ) ∑ ∆ ∈ ( [ n ] k ) ˆ Q T k ( s g , s ∆ , a g , a ∆ ) := E ∆ ∈ ( [ n ] k ) ˆ Q T k ( s g , s ∆ , a g , a ∆ ) , we therefore get:

Q

∗

(

s, a

Consequently,

Further, we have that since

ˆ

Q

∗

k

(

s

g

, s

∆

, a

g

, a

∆

)

)

-

T

lim

Q

→∞

∗

(

∆

s, a

Q

∗

≤

(

Q

E

)

s, a

∗

∈

-

)

(

-

s, a

)

.

Lemma E.14. The absolute difference between the expected maximums between ˆ Q k and ˆ Q k ′ is atmost the maximum of the absolute difference between ˆ Q k and ˆ Q k ′ , where the expectations are taken over any joint distributions of states J , and the maximums are taken over the actions.

<!-- formula-not-decoded -->

Proof. Denote:

ˆ

)

(

[

n

]

k

E

∆

E

Q

∈

∆

T

k

(

∈

(

[

[

n

s

(

]

k

n

]

k

)

)

g

, s

ˆ

Q

∆

∗

k

(

ˆ

Q

∗

k

(

s

, a

g

s

g

, s

g

, a

∆

, s

∆

, a

∆

)

g

, a

≤

T

, a

g

∆

, a lim

→∞

)

∆

γ

0

≤

) = 0

,

T

·

˜

r

-

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

1

γ

We extend a ∗ ∆ to a ∗ ∆ ′ by letting a ∗ ∆ ′ \ ∆ be the corresponding ∆ \ ∆ ′ variables from ˜ a ∗ ∆ ′ in A | ∆ ′ \ ∆ | l . For the remainder of this proof, we adopt the shorthand E s ′ g ,s ′ ∆ ∪ ∆ ′ to refer to E ( s ′ g ,s ′ ∆ ∪ ∆ ′ ) ∼J | ∆ ∪ ∆ ′ | ( · , ·| s g ,a g ,s ∆ ∪ ∆ ′ ,a ∆ ∪ ∆ ′ ) . Then, if E s ′ g ,s ′ ∆ ∪ ∆ ′ max a ′ g ∈A g ,a ′ ∆ ∈A k l ˆ Q T k ( s ′ g , F z ′ ∆ , a ′ g ) -E s ′ g ,s ′ ∆ ∪ ∆ ′ max a ′ g ∈A g ,a ′ ′ ∈A k ′ l ˆ Q T k ′ ( s ′ g , F z ′ ∆ ′ , a ′ g ) &gt; 0 , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Weobserve that if the opposite inequality holds (i.e., E s ′ g ,s ′ ∆ ∪ ∆ ′ max a ′ g ∈A g ,a ′ ∆ ∈A k l ˆ Q T k ( s ′ g , s ′ ∆ , a ′ g , a ′ ∆ ) -E s ′ g ,s ′ ∆ ∪ ∆ ′ max a ′ g ∈A g ,a ′ ∆ ′ ∈A k ′ l ˆ Q T k ′ ( s ′ g , s ′ ∆ ′ , a ′ g , a ′ ∆ ′ ) &lt; 0 ), an analogous argument by replacing a ∗ g with ˜ a ∗ g and a ∗ ∆ with ˜ a ∗ ∆ yields an identical bound.

Lemma E.15. Suppose z, z ′ ≥ 1 . Consider functions Γ : Θ 1 × Θ 2 ×··· × Θ z × Θ ∗ → R and Γ ′ : Θ ′ 1 × Θ ′ 2 ×··· × Θ ′ z ′ × Θ ∗ → R , where Θ 1 , . . . , Θ z and Θ ′ 1 , . . . , Θ ′ z ′ are finite sets. Consider a probability distribution function µ Θ i for i ∈ [ z ] and µ ′ Θ i for i ∈ [ z ′ ] . Then:

<!-- formula-not-decoded -->

Proof. Let ˆ θ ∗ := arg max θ ∗ ∈ Θ ∗ Γ( θ 1 , . . . , θ z , θ ∗ ) and ˜ θ ∗ := arg max θ ∗ ∈ Θ ∗ Γ ′ ( θ 1 , . . . , θ z ′ , θ ∗ ) .

If E θ 1 ∼ µ Θ1 ,...,θ z ∼ µ Θ z max θ ∗ ∈ Θ ∗ Γ( θ 1 , . . . , θ z , θ ∗ ) -E θ 1 ∼ µ ′ Θ ′ 1 ,...,θ z ′ ∼ µ ′ Θ ′ z ′ max θ ∗ ∈ Θ ∗ Γ ′ ( θ 1 , . . . , θ z ′ , θ ∗ ) &gt; 0 , then:

<!-- formula-not-decoded -->

Here, we replace each θ ∗ with the maximizers of their corresponding terms, and upper bound them by the maximizer of the larger term. Next, we replace ˆ θ ∗ in both expressions with the maximizer choice θ ∗ from Θ ∗ , and further bound the expression by its absolute value.

<!-- formula-not-decoded -->

is negative, then an analogous argument that replaces θ with θ yields the same result.

<!-- formula-not-decoded -->

## F Bounding Total Variation Distance

As | ∆ |→ n , we prove that the total variation (TV) distance between the empirical distribution of z ∆ and z [ n ] goes to 0 . Here, recall that z i ∈ Z = S l ×A l , and z ∆ = { z i : i ∈ ∆ } for ∆ ∈ ( [ n ] k ) . Before bounding the total variation distance between F z ∆ and F z ∆ ′ , we first introduce Lemma C.5 of Anand and Qu [2024] which can be viewed as a generalization of the Dvoretzky-Kiefer-Wolfowitz concentration inequality, for sampling without replacement. We first make an important remark.

Remark F.1. First, observe that if ∆ is an independent random variable uniformly supported on ( [ n ] k ) , then s ∆ and a ∆ are also independent random variables uniformly supported on the global state ( s [ n ] k ) and the global action ( a [ n ] k ) . To see this, let ψ 1 : [ n ] → S l where ψ 1 ( i ) = s i and ξ 1 : [ n ] →A l where ξ 1 ( i ) = a i . This naturally extends to ψ k : [ n ] k →S k l , where ψ k ( i 1 , . . . , i k ) = ( s i 1 , . . . , s i k ) and ξ k : [ n ] k →A k l , where ξ k ( i 1 , . . . , i k ) = ( a i 1 , . . . , a i k ) for all k ∈ [ n ] . Then, the independence of ∆ implies the independence of the generated σ -algebra. Further, ψ k and ξ k (which are a Lebesgue measurable function of a σ -algebra) are sub-algebras, implying that s ∆ and a ∆ must also be independent random variables.

For reference, we present the multidimensional Dvoretzky-Kiefer-Wolfowitz (DKW) inequality (Dvoretzky et al. [1956], Massart [1990], Naaman [2021]) which bounds the difference between an empirical distribution function for a set B ∆ and B [ n ] when each element of ∆ for | ∆ | = k is sampled uniformly at random from [ n ] with replacement.

Theorem F.2 (Multi-dimensional Dvoretzky-Kiefer-Wolfowitz (DFW) inequality [Dvoretzky et al., 1956, Naaman, 2021]) . Suppose B ⊂ R d and ϵ &gt; 0 . If ∆ ⊆ [ n ] is sampled uniformly with replacement, then

<!-- formula-not-decoded -->

Lemma C.5 of Anand and Qu [2024] generalizes the DKW inequality for sampling without replacement:

Lemma F.3 (Sampling without replacement analogue of the DKW inequality, Lemma C.5 in Anand and Qu [2024]) . Consider a finite population X = ( x 1 , . . . , x n ) ∈ B n l where B l is a finite set. Let ∆ ⊆ [ n ] be a random sample of size k chosen uniformly and without replacement. Then, for all x ∈ B l :

<!-- formula-not-decoded -->

Lemma F.4. With probability atleast 1 -2 |S l ||A l | e -8 knϵ 2 n -k +1 ,

<!-- formula-not-decoded -->

Proof. Recall that z := ( s l , a l ) ∈ S l ×A l . From Lemma F.3, substituting B l = Z l yields:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which yields the proof.

We now present an alternate bound for the total variation distance, where the distance actually goes to 0 as | ∆ | → n . For this, we use the fact that the total variation distance between two product distributions is subadditive.

Lemma F.5 (Lemma B.8.1 of Ghosal and van der Vaart [2017]. Subadditivity of TV distance for Product Distributions) . Let P and Q be product distributions over some domain S . Let α 1 , . . . , α d be the marginal distributions of P and β 1 , . . . , β q be the marginal distributions of Q . Then,

<!-- formula-not-decoded -->

Lemma F.6 (KL-divergence decays too slowly) .

<!-- formula-not-decoded -->

Proof. By the symmetry of the total variation distance metric, we have that

<!-- formula-not-decoded -->

From the Bretagnolle-Huber inequality Tsybakov [2008] we have that

<!-- formula-not-decoded -->

Here, D KL ( f ∥ g ) is the Kullback-Leibler (KL) divergence metric between probability distributions f and g over the sample space, which we denote by X and is given by

<!-- formula-not-decoded -->

Thus, from Equation (29):

<!-- formula-not-decoded -->

In the third line, we note that ∑ z ∈Z l ∑ i ∈ ∆ ✶ { z i = z } = | ∆ | since each local agent contained in ∆ must have some state/action pair contained in Z l . In the last line, we note that ∑ i ∈ ∆ ✶ { z i = z } ≤ ∑ i ∈ [ n ] ✶ { z i = z } , For all z ∈ Z l , and thus the summation of logarithmic terms in the third line is negative. Finally, using this bound in the Bretagnolle-Huber inequality yields the lemma.

Corollary F.7. From Lemma F.6, setting ∆ = [ n ] also recovers TV( F z ∆ , F z [ n ] ) = 0 . Theorem F.8. With probability atleast 1 -δ for δ ∈ (0 , 1) 2 :

<!-- formula-not-decoded -->

Proof. From combining the total variation distance bound in Lemma F.4 and the Lipschitz continuity bound in Theorem E.3 with ∑ T t =0 γ T ≤ 1 1 -γ for γ ∈ (0 , 1) , we have:

<!-- formula-not-decoded -->

Then, reparameterizing 1 -2 |S l ||A l | e -8 knϵ 2 n -k +1 into 1 -δ to get ϵ = √ n -k +1 8 kn ln ( 2 |S l ||A l | δ ) gives that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

proving the claim.

## G Using the Performance Difference Lemma

In general, convergence analysis requires the guarantee that a stationary optimal policy exists. Fortunately, when working with the empirical distribution function, the existence of a stationary optimal policy is guaranteed when the state/action spaces are finite or countably infinite. However, lifting the knowledge of states onto the continuous empirical distribution function space, and designing a policy on the lifted space is still analytically challenging. To circumvent this, Gu et al. [2021] creates lifted ϵ -nets and does kernel regression to obtain convergence guarantees. Moreover, our result has a similar flavor to MDPs with dynamic exogenous inputs from learning theory, [Dietterich et al., 2018, Foster et al., 2022, Anand and Qu, 2024], wherein our subsampling algorithm treats each sampled state as an endogenous state.

Here, our analytic approach bears a stark difference, wherein we analyze the sampled structure of the mean-field empirical distribution function, rather than studying the structure of the lifted space. For this, we leverage the classical Performance Difference Lemma, which we restate below for completeness.

Lemma G.1 (Performance Difference Lemma, Kakade and Langford [2002]) . Given policies π 1 , π 2 , with corresponding value functions V π 1 , V π 2 :

<!-- formula-not-decoded -->

Here, A π 2 ( s ′ , a ′ ) := Q π 2 ( s ′ , a ′ ) -V π 2 ( s ′ ) and d π 1 s ( s ′ ) = (1 -γ ) ∑ ∞ h =0 γ h Pr π 1 h [ s ′ , s ] where Pr π 1 h [ s ′ , s ] is the probability of π 1 reaching state s ′ at time step h starting from state s .

We denote our learned policy π est k,m where:

<!-- formula-not-decoded -->

where π g, est k,m ( s g , s [ n ] ) = ˆ π g, est k,m ( s g , s u , F s ∆ \ u ) is the global agent's action and π i, est k,m ( s g , s [ n ] ) := ˆ π est k,m ( s g , s i , F s ∆ i ) is the action of the i 'th local agent. Here, ∆ i is a random variable supported on ( [ n ] \ i k -1 ) , u is a random variable uniformly distributed on [ n ] , and ∆ is a random variable uniformly distributed on [ n ] \ u . Then, denote the optimal policy π ∗ given by

<!-- formula-not-decoded -->

where π ∗ g ( s g , s [ n ] ) is the global agent's action, and π ∗ i ( s g , s [ n ] ) := π ∗ ( s g , s i , F s ∆ i ) is the action of the i 'th local agent. Next, in order to compare the difference in the performance of π ∗ ( s ) and π est k,m ( s g , s [ n ] ) , we define the value function of a policy π to be the infinite-horizon γ -discounted rewards, (denoted by V π ) as follows:

Definition G.2. The value function V π : S → R of a given policy π , for S := S g ×S n l is:

<!-- formula-not-decoded -->

Theorem G.3. For the optimal policy π ∗ and the learned policy π est k,m , for any state s 0 ∈ S , we have:

<!-- formula-not-decoded -->

Proof. Applying the performance difference lemma to the policies gives us:

<!-- formula-not-decoded -->

Next, by the law of total expectation,

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Therefore, by grouping the equations above, we have:

<!-- formula-not-decoded -->

Lemma G.4 shows a uniform bound on

<!-- formula-not-decoded -->

(independent of ∆ i , ∀ i ∈ [ n ] ), allowing the sums and the counts in the denominator will cancel out. Observe that ˆ π ∗ k,m : S → A and π ∗ : S → A are deterministic functions. Therefore, denote a = π ∗ ( s ) . Then, from Lemma G.6,

<!-- formula-not-decoded -->

Similarly, from Corollary G.7, we have that

<!-- formula-not-decoded -->

Hence, combining the above inequalities, we get:

<!-- formula-not-decoded -->

which yields the claim. We defer parameter optimization to Lemma G.8.

Lemma G.4 (Uniform Bound on Q ∗ with different actions) . For all s ∈ S , ∆ ∈ ( [ n ] k ) and ∆ i ∈ ( [ n ] \ i k -1 ) for i ∈ [ n ] , we have:

<!-- formula-not-decoded -->

Proof. Observe that

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

The first inequality above follows from the triangle inequality, and the second inequality uses

<!-- formula-not-decoded -->

where the ϵ k,m follows from Lemma 4.3. From Theorem F.8, we have that with probability at least 1 -δ t ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which proves the claim.

Lemma G.5. Fix s ∈ S := S g ×S n l . For each j ∈ [ n ] , suppose we are given T -length sequences of random variables { ∆ j 1 , . . . , ∆ j T } j ∈ [ n ] , distributed uniformly over the support ( [ n ] \ j k -1 ) . Further, suppose we are given a fixed sequence δ 1 , . . . , δ T , where each δ t ∈ (0 , 1] for t ∈ [ T ] . Let

<!-- formula-not-decoded -->

Then, for each action a = ( a g , a [ n ] ) = π ∗ ( s ) , for t ∈ [ T ] and j ∈ [ n ] , define deviation events B a g ,a j, ∆ j t t such that:

<!-- formula-not-decoded -->

For i ∈ [ T ] , we define bad-events B t (which is a union over each deviation event):

<!-- formula-not-decoded -->

Next, denote B = ⋃ T i =1 B i . Then, the probability that no bad event B t occurs is:

<!-- formula-not-decoded -->

So, event B a g ,a j, ∆ j t t occurs with probability atmost δ t .

Here, observe that if we union bound across all the events parameterized by the empirical distributions of a g , a j, ∆ j t ∈ A g ×A k l given by F a ∆ j t , this also forms a covering of the choice of variables F s ∆ j t by agents j ∈ [ m ] , and therefore across all choices of ∆ 1 t , . . . , ∆ n t (subject to the permutation invariance of local agents) for a fixed t .

Thus, from the union bound, we get:

<!-- formula-not-decoded -->

Applying the union bound again proves the lemma:

<!-- formula-not-decoded -->

which proves the claim.

Lemma G.6. For any arbitrary distribution D of states S := S g ×S n l , for any ∆ i ∈ ( [ n ] \ i k -1 ) for i ∈ [ n ] and for δ ∈ (0 , 1] we have:

<!-- formula-not-decoded -->

Proof. By the linearity of expectations, observe that:

<!-- formula-not-decoded -->

Let

<!-- formula-not-decoded -->

Then, define the indicator function I : [ n ] ×S × N × (0 , 1] →{ 0 , 1 } by:

I

(

i, s, k, δ

) :=

<!-- formula-not-decoded -->

The expected difference between Q ∗ ( s ′ , π ∗ ( s ′ )) and ˆ Q est k,m ( s g , s i , F s ∆ i , π ∗ ( s ) g , { π ∗ ( s ) j } j ∈{ i, ∆ i } ) is bounded as follows:

<!-- formula-not-decoded -->

Here, we have used the general property for a random variable X and constant c that E [ X ] = E [ X ✶ { X ≤ c } ] + E [(1 -{ X ≤ c } ) X ] . Then,

<!-- formula-not-decoded -->

For the first term in the first inequality, we use E [ X ✶ { X ≤ c } ] ≤ c . For the second term, we trivially bound Q ∗ ( s ′ , π ∗ ( s ′ )) -ˆ Q ∗ k ( s g , s i, ∆ i , π ∗ ( s ) g , π ∗ ( s ) i, ∆ i ) by the maximum value Q ∗ can take, which is ˜ r 1 -γ by Lemma C.4. In the second inequality, we use the fact that the expectation of an indicator function is the conditional probability of the underlying event. The second inequality follows from Lemma G.5.

Since this is a uniform bound that is independent of ∆ j for j ∈ [ n ] and i ∈ [ n ] , we have:

<!-- formula-not-decoded -->

This proves the claim.

Corollary G.7. Changing the notation of the policy to π est k,m in the proofs of Lemmas G.5 and G.6, such that a ∼ ˜ π est k,m then yields that:

<!-- formula-not-decoded -->

Lemma G.8 (Optimizing Parameters) . With probability at least 1 -1 100 e k ,

<!-- formula-not-decoded -->

Proof. Setting δ = (1 -γ ) 2 2˜ rε |A g | k |A l | in the bound for the optimality gap in Theorem G.3 gives:

<!-- formula-not-decoded -->

Setting ε = 10 √ k for any c &gt; 0 recovers a decaying optimality gap of the order

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, using the probabilistic bound of ϵ k,m ≤ ˜ O ( 1 √ k ) from Lemma G.10 yields that with probability at least 1 -1 100 e k , which proves the claim.

## G.1 Bounding the Bellman Error

This section is devoted to the proof of Lemma 4.3.

Theorem G.9 (Theorem 2 of Li et al. [2022]) . If m ∈ N is the number of samples in the Bellman update, there exists a universal constant 0 &lt; c 0 &lt; 2 and a Bellman noise 0 &lt; ϵ k,m ≤ 1 1 -γ such that

<!-- formula-not-decoded -->

where ϵ k,m satisfies

<!-- formula-not-decoded -->

with probability at least 1 -ρ , for any ρ ∈ (0 , 1) . Here, t cover stands for the cover time, which is the time taken for the trajectory to visit all state-action pairs at least once. Formally,

<!-- formula-not-decoded -->

where B t denotes the event that all ( s, a ) ∈ S × A have been visited at least once between time 0 and time t , and P ( B t | s 0 , a 0 ) denotes the probability of B t conditioned on the initial state ( s 0 , a 0 ) .

Lemma G.10. If T = 2 1 -γ log ˜ r √ k 1 -γ , SUB-SAMPLE-MFQ : Learning runs in time ˜ O ( T |S g | 2 |A g | 2 |A l | 2 |S l | 2 k 3 . 5+2 |S l ||A l | ˜ r ) , while accruing a Bellman noise ϵ k,m ≤ ˜ O (1 / √ k ) with probability at least 1 -1 100 e k , .

Proof. We first prove that ∥ ˆ Q T k -ˆ Q ∗ k ∥ ∞ ≤ 1 √ k .

For this, it suffices to show γ T ˜ r 1 -γ ≤ 1 √ k = ⇒ γ T ≤ 1 -γ ˜ r √ k . Then, using γ = 1 -(1 -γ ) ≤ e -(1 -γ ) , it again suffices to show e -(1 -γ ) T ≤ 1 -γ ˜ r √ k . Taking logarithms, we have

<!-- formula-not-decoded -->

Since T = 2 1 -γ log ˜ r √ k 1 -γ &gt; 1 1 -γ log ˜ r √ k 1 -γ , the condition holds and ∥ ˆ Q T k -ˆ Q ∗ k ∥ ∞ ≤ 1 √ k . Then, rearranging Equation (32) and incorporating the convergence error of the ˆ Q k -function, one has that with probability at least 1 -ρ ,

<!-- formula-not-decoded -->

Then, using the naïve bound t cover (since we are doing offline learning), we have

<!-- formula-not-decoded -->

Substituting this in Equation (34) yields that with probability 1 -ρ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, letting ρ = 1 100 e k , setting

<!-- formula-not-decoded -->

attains a Bellman error of ϵ k,m ≤ ˜ O (1 / √ k ) with probability at least 1 -1 100 e k . Finally, the runtime of our learning algorithm is

<!-- formula-not-decoded -->

which is still polynomial in k , proving the claim.

## H Generalization to Stochastic Rewards

Suppose we are given two families of distributions, {G s g ,a g } s g ,a g ∈S g ×A g and {L s i ,s g ,a i } s i ,s g ,a i ∈S l ×S g ×A l . Let R ( s, a ) denote a stochastic reward of the form

<!-- formula-not-decoded -->

where the rewards of the global agent r g emerge from a distribution r g ( s g , a g ) ∼ G s g ,a g , and the rewards of the local agents r l emerge from a distribution r l ( s i , s g , a i ) ∼ L s i ,s g ,a i . For ∆ ⊆ [ n ] , let R ∆ ( s, a ) be defined as:

<!-- formula-not-decoded -->

We make some standard assumptions of boundedness on G s g ,a g and L s i ,s g ,a i .

## Assumption H.1. Define

<!-- formula-not-decoded -->

where for any distribution D , supp( D ) is the support (set of all random variables D that can be sampled with probability strictly larger than 0 ) of D . Then, let ˆ G = sup ( ¯ G ) , ˆ L = sup ( ¯ L ) , ̂ G = inf ( ¯ G ) , and ̂ L = inf ( ¯ L ) . We assume that ˆ G &lt; ∞ , ˆ L &lt; ∞ , ̂ G &gt; -∞ , ̂ L &gt; -∞ , and that ˆ G , ˆ L , ̂ G , ̂ L are all known in advance.

Definition H.2. Let the randomized empirical adapted Bellman operator be ˆ T random k,m such that:

<!-- formula-not-decoded -->

SUBSAMPLE-MFQ: Learning with Stochastic Rewards. The proposed algorithm averages Ξ samples of the adapted randomized empirical adapted Bellman operator, T random k,m and updates the ˆ Q k,m function using with the average. One can show that T random k,m is a contraction operator with module γ . By Banach's fixed point theorem, T random k,m admits a unique fixed point ˆ Q random k,m .

## Algorithm 4 SUBSAMPLE-MFQ : Learning with Stochastic Rewards

Require: A multi-agent system as described in Section 2. Parameter T for the number of iterations in the initial value iteration step. Sampling parameters k ∈ [ n ] and m ∈ N . Discount parameter γ ∈ (0 , 1) . Oracle O to sample s ′ g ∼ P g ( ·| s g , a g ) and s i ∼ P l ( ·| s i , s g , a i ) for all i ∈ [ n ] .

- 1: Set ˜ ∆ = { 2 , . . . , k } .

<!-- formula-not-decoded -->

- 3: Set ˆ Q 0 k,m ( s g , s 1 , F z ˜ ∆ , a g , a 1 ) = 0 , for ( s g , s 1 , F z ˜ ∆ , a 1 , a g ) ∈ S g ×S l × µ k -1 ( Z l ) ×A g ×A l .
- 4: for t = 1 to T do

<!-- formula-not-decoded -->

- 6: ρ = 0
- 7: for ξ ∈ { 1 , . . . , Ξ } do

<!-- formula-not-decoded -->

- 9: ˆ Q t +1 k,m ( s g , s 1 , F z ˜ ∆ , a g , a 1 ) = ρ/ Ξ
- 10: For all ( s g , s i , F s ∆ \ i ) ∈ S g ×S l × µ k -1 ( S l ) , let

<!-- formula-not-decoded -->

Theorem H.3 (Hoeffding's Theorem [Tsybakov, 2008]) . Let X 1 , . . . , X n be independent random variables such that a i ≤ X i ≤ b i almost surely. Then, let S n = X 1 + · · · + X n . Then, for all ϵ &gt; 0 ,

<!-- formula-not-decoded -->

Lemma H.4. When π est k,m is derived from the randomized empirical value iteration operation and applying our online subsampling execution in Algorithm 2, we get

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

Rearranging this, we get:

̂

̂

<!-- formula-not-decoded -->

Then, setting δ = 1 100 √ k , and setting Ξ = 10 | ˆ G + ˆ LG L| k 1 / 4 √ ln(200 √ k ) gives:

<!-- formula-not-decoded -->

Then, applying the triangle inequality to ϵ k,m allows us to derive a probabilistic bound on the optimality gap between ˆ Q est k,m and Q ∗ , where the gap is increased by 1 √ 200 k , and where the randomness is over the stochasticity of the rewards. Then, the optimality gap between V π ∗ and V π est k,m , for when the policy ˆ π est k,m is learned using Algorithm 4 in the presence of stochastic rewards obeying Assumption H.1 follows

<!-- formula-not-decoded -->

which proves the lemma.

Remark H.5. First, note that that through the naïve averaging argument, the optimality gap above still decays to 0 with probability decaying to 1 , as k increases. ̂ ̂

Remark H.6. This method of analysis could be strengthened by obtaining estimates of ˆ G , ˆ L , G , L and using methods from order statistics to bound the errors of the estimates and use the deviations between estimates to determine an optimal online stopping time [Kleinberg, 2005] as is done in the online secretary problem. This, more sophisticated, argument would become an essential step in converting this to a truly online learning, with a stochastic approximation scheme. Furthermore, one could incorporate variance-based analysis in the algorithm, where we use information derived from higher-order moments to do a weighted update of the Bellman update [Jin et al., 2024], thereby taking advantage of the skewedness of the distribution, and allowing us the freedom to assign an optimism/pessimism score to our estimate of the reward.

## I Partially Relaxing the Offline Learning assumption: Off-Policy Learning

A limitation of the planning Algorithm 1 is that it learns ˆ Q ∗ k in an offline manner by assuming a generative oracle access to the transition functions P g , P l , and reward function r ( · , · ) . In certain realistic RL applications, such a generative oracle might not exist, and it is more desirable to perform off-policy learning where the agent continues to learn in an offline manner but from historical data Fujimoto et al. [2019]. In this setting, the agents learn the target policy ˆ π ∗ k using data generated by a different behavior policy (the strategy it uses to explore the environment). There is a significant body of work on the theoretical guarantees in off-policy learning Chen et al. [2021b], Chen and Maguluri [2022b], Chen et al. [2021a, 2025].

In fact, these previous results are amenable to transforming guarantees about offline Q -learning to off-policy Q -learning, at the cost of log |S g ||S l | k |A g ||A l | k factors in the runtime. Therefore, this section is devoted to showing that our previous result satisfy the assumptions of transforming offline Q -learning to off-policy Q -learning for the subsampled ˆ Q k -function, and we further show that, in expectation, can maintain the decaying optimality gap of ˜ O (1 / √ k ) of the learned policy π k , where the randomness is over the heuristic exploration policy π b .

The off-policy ˆ Q k -learning algorithm is an iterative algorithm to estimate the optimal ˆ Q k -function as follows: first, a sample trajectory { ( s g , s ∆ , a g , a ∆ ) } is collected using a suitable behavior policy π k,b . For simplicity of notation, let S ∆ = ( s g , s ∆ ) and A ∆ = ( a g , a ∆ ) . Then, initialize ˆ Q 0 k : |S g ||S l | k |A g ||A l | k → R and let α &gt; 0 be determined later. For each t ≥ 0 and state-action pair ( S ∆ , A ∆ ) that is updated to S ′ ∆ , the iterate ˆ Q t k ( S ∆ , A ∆ ) is updated by

<!-- formula-not-decoded -->

Note that the update in Equation (44) does not include an expectation and can be computed in a single trajectory via historical data. We make the following ergodicity assumption:

Assumption I.1. The behavior policy π b satisfies π b ( A ∆ | S ∆ ) &gt; 0 for all ( S ∆ , A ∆ ) ∈ S g ×S k l × A g × A k l and the Markov chain M S ∆ = { S ( t ) ∆ } t ≥ 0 induced by π b is irreducible and aperiodic with stationary distribution µ and mixing time t δ ( M ) = min { t ≥ 0 : max S ∆ ∈S g ×S k l ∥ P t ( S ∆ , · ) -µ ( · ) ∥ TV ≤ δ } . There are many heuristics of such behavior policies [Fujimoto et al., 2019].

Theorem I.2. Let π k be the policy learned through off-policy ˆ Q k -learning. Then, under Assumption I.1, we have that that with probability at least 1 -1 100 e k ,

<!-- formula-not-decoded -->

where the randomness in the expectation is over the stochasticity of the exploration policy π b .

Proposition I.3. Recall that the following are true:

<!-- formula-not-decoded -->

2. ∥ ˆ Q k ∥ ≤ ˜ r 1 -γ by Lemma C.4,

<!-- formula-not-decoded -->

4. The Markov chain M S enjoys a rapid mixing property from Assumption I.1,

Then, by treating the single trajectory update of the ˆ Q k -function as a noisy addition to the expected update from the ideal Bellman operator, Chen et al. [2021b] uses Markovian stochastic approximation to bound E [ ∥ ˆ Q T k -ˆ Q ∗ k ∥ 2 ∞ ] . We restate their result:

Theorem I.4 (Theorem 3.1 in Chen et al. [2021b] adapted to our setting) . Suppose α t = α for all t ≥ 0 , where α is chosen such that αt α ( M S ∆ ) ≤ c Q, 0 (1 -β 1 ) 2 log |S g ||S k l ||A g |A k l | , where c Q, 0 is a numerical constant. Then, under Proposition I.3, for all t ≥ t α ( M S ∆ ) , we have

<!-- formula-not-decoded -->

where c Q, 1 = 3( ˜ r 1 -γ +1) 2 , c Q, 2 = 912 e ( 3˜ r 1 -γ +1) 2 , and where the randomness in the expectation is over the randomness of the stochasticity of the behavior/exploration policy π b .

Corollary I.5 (Corollary 3.2 in Chen et al. [2021b] adapted to our setting) . To make E [ ∥ ˆ Q t k -ˆ Q ∗ k ∥ ∞ ≤ 1 100 √ k ] for ϵ &gt; 0 , we need

<!-- formula-not-decoded -->

With this sample complexity, by the triangle inequality we also recover an expected-value analog of Theorem F.8.

Corollary I.6. For δ ∈ (0 , 1) 2 , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

where the randomness in the expectation is over the stochasticity of the exploration policy π b .

In turn, following the argument in the proof of Theorem G.3, it is straightforward to verify that this leads to a result on the expected performance difference using off-policy learning:

Corollary I.7. With probability at least 1 -1 ,

<!-- formula-not-decoded -->

where the randomness in the expectation is over the stochasticity of the exploration policy π b .

## J Extension to Continuous State/Action Spaces

Multi-agent settings where each agent handles a continuous state and action space find many applications in optimization, control, and synchronization.

Example J.1 (Quadcopter Swarm Drone [Preiss et al., 2017]) . Consider a system of drones with a global controller, where each drone has to chase the controller, and the controller is designed to follow a bounded trajectory. Here, the state of each local agent i ∈ [ n ] is its position and velocity in the bounded region, and the state of the global agent g is a signal on its position and direction. The action of each local agent is a velocity vector a l ∈ A l ⊂ R 3 which is a bounded subset of R 3 , and the action of the global agent is a vector a g ∈ A g ⊂ R 2 which is a bounded subset of R 2 .

Hence, this section is devoted to extending our algorithm and theoretical results to the case where the state and action spaces can be continuous (and therefore have infinite cardinality).

Preliminaries. For a measurable space ( S , B ) , where B is a σ -algebra on S , let R S denote the set of all real-valued B -measurable functions on S . Let ( X,d ) be a metric space, where X is a set and d is a metric on X . A set S ⊆ X is dense in X if every element of X is either in S or a limit point of S . A set is nowhere dense in X if the interior of its closure in X is empty. A set S ⊆ X is of Baire first category if S is a union of countably many nowhere dense sets. A set S ⊆ X is of Baire second category if X \ S is of first category.

Theorem J.2 (Baire Category Theorem [Jin et al., 2020]) . Let ( X,d ) be a complete metric space. Then any countable intersection of dense open subsets of X is dense.

Definition J.3 (Linear MDP) . MDP( S, A, P , r ) is a linear MDP with feature map ϕ : S × A → R d if there exist d unknown (signed) measures µ = ( µ 1 , . . . , µ d ) over S and a vector θ ∈ R d such that for any ( s, a ) ∈ S × A , we have

<!-- formula-not-decoded -->

Assumption J.4 (Bounded features) . Without loss of generality, we assume boundedness of the features: ∥ ϕ ( s, a ) ∥ ≤ 1 , for all ( s, a ) ∈ S × A and max {∥ µ ( S ) ∥ , ∥ θ ∥} ≤ √ d .

We motivate our analysis by reviewing representation learning in RL via spectral decompositions. For instance, if P ( s ′ | s, a ) admits a linear decomposition in terms of some spectral features ϕ ( s, a ) and µ ( s ′ ) , then the Q ( s, a ) -function can be linearly represented in terms of the spectral features ϕ ( s, a ) .

Then, through a reduction from Ren et al. [2024] that uses function approximation to learn the spectral features ϕ k for ˆ Q k , we derive a performance guarantee for the learned policy π est k , where the optimality gap decays with k :

Theorem J.5. When π est k is derived from the spectral features ϕ k learned in ˆ Q k , and M is the number of samples used in the function approximation, then

<!-- formula-not-decoded -->

We assume the system is a linear MDP, where S g and S l are infinite compact sets. By a reduction from Ren et al. [2024] and using function approximation to learn the spectral features ϕ k for ˆ Q k , we derive a performance guarantee for the learned policy π est k , where the optimality gap decays with k .

Assumption J.6. Suppose S g ⊂ R σ g , S l ⊂ R σ l , A g ⊂ R α g , A l ⊂ R α l are bounded compact sets. From the Baire category theorem, the underlying field R can be replaced to any set of Baire first category, which satisfies the property that there exists a dense open subset. In particular, replace Assumption 2.1 with a boundedness assumption.

Multi-agent settings where each agent handles a continuous state/action space find many applications in optimization, control, and synchronization.

Lemma J.7 (Proposition 2.3 in Jin et al. [2020]) . For any linear MDP, for any policy π , there exist weights { w π } such that for any ( s, a ) ∈ S × A , we have Q π ( s, a ) = ⟨ ϕ ( s, a ) , w π ⟩ .

Lemma J.8 (Proposition A.1 in Jin et al. [2020]) . For any linear MDP, for any ( s, a ) ∈ S × A , and for any measurable subset B ⊆ S , we have that

<!-- formula-not-decoded -->

Property J.9. Suppose that there exist spectral representation µ g ( s ′ g ) ∈ R d and µ l ( s ′ i ) ∈ R d such that the probability transitions P g ( s ′ g , | s g , a g ) and P l ( s ′ i | s i , s g , a i ) can be linearly decomposed by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some features ϕ g ( s g , a g ) ∈ R d and ϕ l ( s i , s g , a i ) ∈ R d . Then, the dynamics are amenable to the following spectral factorization, as in Ren et al. [2024].

Lemma J.10. ˆ Q k admits a linear representation.

Proof. Given the factorization of the dynamics of the multi-agent system, we have:

<!-- formula-not-decoded -->

Similarly, for any ∆ ⊆ [ n ] where | ∆ | = k , the subsystem consisting of k local agents ∆ has a subsystem dynamics given by

<!-- formula-not-decoded -->

Therefore, ˆ Q k admits the linear representation:

<!-- formula-not-decoded -->

proving the claim.

Therefore, ¯ ϕ k serves as a good representation for the ˆ Q k -function making the problem amenable to the classic linear functional approximation algorithms, consisting of feature generation and policy gradients .

In feature generation [Ren et al., 2024], we generate the appropriate features ϕ , comprising of the local reward functions and the spectral features coming from the factorization of the dynamics. In applying the policy gradient, we perform a gradient step to update the local policy weights θ i and update the new policy.

For this, we update the weight w via the TD(0) target semi-gradient method (with step size α ), to get

<!-- formula-not-decoded -->

Definition J.11. Let the complete feature map be denoted by Φ : S × d → R , and let the subsampled feature map be denoted by ˆ Φ k : S g ×S k l × d ∈ R , where we let

<!-- formula-not-decoded -->

Here, the system's stage rewards are given by

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where d is a low-dimensional embedding we wish to learn.

Agnostically, the goal is to approximate the value function through the approximation

<!-- formula-not-decoded -->

This manner of updating the weights can be considered via the projected Bellman equation ˆ Φ k w = Π µ T ( ˆ Φ k w ) , where Π µ ( v ) = arg min z ∈ ˆ Φ k w ∥ z -v ∥ 2 µ . Notably, the fixed point of the the projected Bellman equation satisfies

<!-- formula-not-decoded -->

where D µ is a diagonal matrix comprising of µ 's. In other words, D µ = diag( µ 1 , . . . , µ n ) . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, the problem is amenable to Algorithm 1 in Ren et al. [2024]. To bound the error of using linear function approximation to learn ˆ Q k , we use a result from Ren et al. [2024].

Lemma J.12 (Policy Evaluation Error, Theorem 6 of Ren et al. [2024]) . Suppose the sample size M ≥ log ( 4 n δ ) , where n is the number of agents and δ ∈ (0 , 1) is an error parameter. Then, with probability at least 1 -2 δ , the ground truth ˆ Q π k ( s, a ) function and the approximated ˆ Q k -function ˆ Q LFA k ( s, a ) satisfies, for any ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

where ϵ P is the error in approximating the spectral features ϕ g , ϕ l .

Corollary J.13. Therefore, when π k,m is derived from the spectral features learned in ˆ Q LFA k , applying the triangle inequality on the Bellman noise and setting δ = ϵ P = 1 √ k yields 5

<!-- formula-not-decoded -->

Using ∥ ˜ ϕ k ∥ ≤ 1 and simplifying gives

<!-- formula-not-decoded -->

Remark J.14. Hence, in the continuous state/action space setting, as k → n and M → ∞ , V ˜ π k ( s 0 ) → V ˜ π n ( s 0 ) = V π ∗ ( s 0 ) . Intuitively, as k → n , the optimality gap diminishes following the arguments in Corollary G.7. As M →∞ , the number of samples obtained allows for more effective learning of the spectral features.

## K Towards Deterministic Algorithms: Sharing Randomness

In large distributed systems, the random sampling in our communication network may be a bottleneck for 'decentralized execution'. In light of this, we have provided a practical derandomized heuristic where the agents can share some randomness by only sampling within pre-defined fixed blocks of size.

5 Following Ren et al. [2024], the result easily generalizes to any positive-definite transition Kernel noise (e.g. Laplacian, Cauchy, Matérn, etc.

In turn, this implies

## Algorithm 5 SUB-SAMPLE-MFQ : Execution with weakly shared randomness

- Require: A multi-agent system as described in Section 2. A distribution s 0 on the initial global state s 0 = ( s g , s [ n ] ) . Parameter T ′ for the number of iterations for the decision-making sequence. Hyperparameter k ∈ [ n ] . Discount parameter γ . Policy ˆ π est ( s , s ) .
- 1: Sample ( s g (0) , s [ n ] (0)) ∼ s 0 .
- 2: Define groups h 1 , . . . , h x of agents where x := ⌈ n k ⌉ and | h 1 | = | h 2 | = · · · = | h x -1 | = k and | h x | = n mod k .
- 3: for t = 0 to T ′ -1 do
- 4: for i ∈ [ x ] do
- 5: Let ∆ i be a uniform sample of ( [ n ] \ g i k -1 ) , and let a g i ( t ) = [ˆ π est k,m ( s g ( t ) , s ∆ i ( t ))] 1: k .
- 6: Let a g ( t ) = majority ( { [ˆ π est k,m ( s g ( t ) , s ∆ i ( t ))] g } i ∈ [ x ] ) .
- 7: Let s g ( t +1) ∼ P g ( ·| s g ( t ) , a g ( t ))
- 8: Let s i ( t +1) ∼ P l ( ·| s i ( t )
- 9: s g ( t ) , a i ( t )) , for all i ∈ [ n ] .
- 10: Get reward r ( s, a ) = r g ( s g , a g ) + 1 n ∑ i ∈ [ n ] r l ( s i , a i , s g )

## Algorithm 6 SUB-SAMPLE-MFQ : Execution with strongly shared randomness

- Require: A multi-agent system as described in Section 2. A distribution s 0 on the initial global state s 0 = ( s g , s [ n ] ) . Parameter T ′ for the number of iterations for the decision-making sequence. Hyperparameter k ∈ [ n ] . Discount parameter γ . Policy ˆ π est k,m ( s g , s ∆ ) .
- 1: Sample ( s g (0) , s [ n ] (0)) ∼ s 0 .
- 2: Define groups h 1 , . . . , h x of agents where x := ⌈ n k ⌉ and | h 1 | = | h 2 | = · · · = | h x -1 | = k and | h x | = n mod k .
- 3: for t = 0 to T ′ -1 do
- 4: for i ∈ [ x -1] do
- 5: Let a h i ( t ) = [ˆ π est k,m ( s g ( t ) , s h i ( t ))] 1: k .
- 6: Let a g h i ( t ) = [ˆ π est k,m ( s g ( t ) , s h i ( t ))] g .
- 7: Let ∆ residual := ( [ n ] \ h x k -( n -⌈ n/k ⌉ k ) ) .
- 8: Let a h x ( t ) = [ˆ π est k,m ( s g ( t ) , s h x ∪ ∆ residual ( t ))] 1: n -⌈ n/k ⌉ k
- 10: Let a g ( t ) = majority ( { a g h i ( t ) : i ∈ [ x ] } ) .
- 9: Let a g h x ( t ) = [ˆ π est k,m ( s g ( t ) , s h x ∪ ∆ residual ( t ))] g
- 11: Let s g ( t +1) ∼ P g ( ·| s g ( t ) , a g ( t )) , s i ( t +1) ∼ P l ( ·| s i ( t ) , s g ( t ) , a i ( t )) , for all i ∈ [ n ] .
- 12: Get reward r ( s, a ) = r g ( s g , a g ) + 1 n ∑ i ∈ [ n ] r l ( s i , a i , s g )

In this section, we propose algorithms RANDOM-SUB-SAMPLE-MFQ and RANDOM-SUB-SAMPLE-MFQ+ , which shares randomness between agents through various careful groupings and using shared randomness within each groups. By implementing these algorithms, we derive simulation results, and establish progress towards derandomizing inherently stochastic approximately-optimal policies.

Algorithm 5 (Execution with weakly shared randomness). The local agents are heuristically divided into arbitrary groups of size k . For each group h i , the k -1 local agents sampled are the same at each iteration. The global agent's action is then the majority global agent action determined by each group of local agents. At each round, this requires O ( ⌈ n 2 /k ⌉ ( n -k )) random numbers.

Algorithm 6 (Execution with strongly shared randomness). The local agents are randomly divided in to groups of size k . For each group, each agent uses the other agents in the group as the k -1 other sampled states. Similarly, the global agent's action is then the majority global agent action determined by each group of local agents. Here, at each round, this requires O ( ⌈ n 2 /k ⌉ ) random numbers.

The probability of a bad event (the policy causing the Q -function to exceed the Lipschitz bound) scales with the O ( n k ) independent random variables, which is polynomially large. Agnostically, some randomness is always needed to avoid periodic dynamics that may occur when the underlying k,m g ∆

Markov chain is not irreducible. In this case, an adversarial reward function could be designed such that the system does not minimize the optimality gap, thereby penalizing excessive determinism.

This ties into an open problem of interest, which has been recently explored [Larsen et al., 2024]. What is the minimum amount of randomness required for SUBSAMPLE-MFQ (or general mean-field or multi-agent RL algorithms) to perform well? Can we derive a theoretical result that demonstrates and balances the tradeoff between the amount of random bits and the performance of the subsampled policy when using the greedy action from the k -agent subsystem derived from ˆ Q ∗ k ? We leave this problem as an exciting direction for future research.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We provide clear and formal mathematical proofs to support each claim.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We explicitly discuss the limitations of our results in the paper: the lack of matching lower bounds to verify if the scaling with the number of sampled agents k is optimal in the final bound on the optimality gap of ˜ O (1 / √ k ) , and that we do not provide any online regret bounds.

Guidelines:

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

Justification: We provide clear and formal mathematical proofs to support each claim.

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

Justification: We provide a full description of the algorithms and experimental setup.

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

Justification: Our simulations are all on synthetic datasets, but the code we provide in https://github.com/emiletimothy/Mean-Field-Subsample-Q-Learning , and our algorithm design, is sufficient to faithfully reproduce our experimental results.

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

Justification: We run an explicit sweep of the parameter k .

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We provide error bars that report over the min → max values.

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

Answer: [Yes]

Justification: We provide sufficient information on the compute resources in Appendix B: all our experiments were run on a 2-code CPU server with 12GB RAM.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research conducted in this work conforms, in every respect, with the NeurIPS code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: This work is theoretical and foundational in nature. As such, it is not tied toa ny specific applications or deployments.

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

Answer: [NA]

Justification: The paper does not use any existing assets.

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

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve any crowd-sourcing nor any research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve any crowd-sourcing nor any research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.