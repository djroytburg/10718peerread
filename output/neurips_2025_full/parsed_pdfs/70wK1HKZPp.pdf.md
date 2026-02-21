## Oracle-Efficient Combinatorial Semi-Bandits

## Jung-hun Kim

CREST, ENSAE, IP Paris FairPlay joint team, France junghun.kim@ensae.fr

## Milan Vojnovi´ c

London School of Economics United Kingdom m.vojnovic@lse.ac.uk

## Abstract

We study the combinatorial semi-bandit problem where an agent selects a subset of base arms and receives individual feedback. While this generalizes the classical multi-armed bandit and has broad applicability, its scalability is limited by the high cost of combinatorial optimization, requiring oracle queries at every round. To tackle this, we propose oracle-efficient frameworks that significantly reduce oracle calls while maintaining tight regret guarantees. For the worst-case linear reward setting, our algorithms achieve ˜ O ( √ T ) regret using only O (log log T ) oracle queries. We also propose covariance-adaptive algorithms that leverage noise structure for improved regret, and extend our approach to general (nonlinear) rewards. Overall, our methods reduce oracle usage from linear to (doubly) logarithmic in time, with strong theoretical guarantees.

## 1 Introduction

The combinatorial semi-bandit problem extends the classical multi-armed bandit (MAB) model to settings where an agent selects a subset of base arms (a combinatorial action) and receives individual feedback for each. This general framework captures many real-world scenarios, such as product recommendation, where a set of items is recommended to a user [16]; ad slot allocation, where multiple ads are displayed on a webpage [12]; and network routing, where a path comprising several links is selected in a communication network [27].

Due to its broad applicability, the combinatorial semi-bandit problem has been extensively studied in the literature [5, 7, 18, 9, 22, 32]. However, a central challenge lies in the computational complexity of solving the combinatorial optimization problem, which is often NP-hard. As a result, most existing algorithms assume access to an oracle that returns a solution to the combinatorial problem. These algorithms rely on querying the oracle at every round, leading to excessive oracle usage and substantial computational overhead in practice.

In this work, following the computational complexity notions introduced in Balkanski and Singer [1], Fahrbach et al. [11], we distinguish between two measures of oracle efficiency: adaptivity complexity and query complexity , which are defined later. Our goal is to improve oracle efficiency by substantially reducing the overall oracle adaptivity and query complexities in decision-making over a time horizon T , while maintaining tight gap-free regret guarantees that do not depend on the suboptimality gaps. Our main contributions are summarized below and compared with prior work on gap-free combinatorial semi-bandits in Table 1.

- Oracle-efficient algorithms for worst-case linear rewards: We propose two frameworks that significantly reduce oracle query usage while maintaining tight regret guarantees. Using an adaptive oracle query framework, AROQ-CMAB achieves near-optimal regret of ˜ O ( √ mdT ) , with both adaptivity and query complexity bounded by O ( d log log( Tm/d )) , where d denotes the number of base arms and m is the maximum number of activated base arms per action. To further improve computational practicality by reducing adaptivity complexity,

Min-hwan Oh South Korea

Seoul National University minoh@snu.ac.kr

Table 1: Gap-free regret bounds for combinatorial semi-bandit algorithms.

| Combinatorial Reward Model     | Algorithm                                                    | Regret                                                                                          | Adaptivity Complexity                                  | Query Complexity                                                |
|--------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------|-----------------------------------------------------------------|
| Linear (Worst-case)            | CUCB [5] CUCB [18] AROQ-CMAB (our work) SROQ-CMAB (our work) | ˜ O ( m √ dT ) ˜ O ( √ mdT ) ˜ O ( √ mdT ) ˜ O ( m √ dT )                                       | Θ( T ) Θ( T ) O ( d log log( Tm d )) Θ(log log T )     | Θ( T ) Θ( T ) O ( d log log( Tm d )) O ( d log log T )          |
| Linear (Covariance -dependent) | OLS-UCB-C [32] AROQ-C-CMAB (our work)                        | (√∑ i ∈ [ d ] max a ∈A s.t. i ∈ a σ 2 i ( a ) T (√∑ i ∈ [ d ] max a ∈A s.t. i ∈ a σ 2 i ( a ) T | Θ( T ) O ( d 2 log( Tm ))                              | Θ( T )                                                          |
| General                        | SROQ-C-CMAB (our work) SDCB [4] AROQ-GR-CMAB (our work)      | ˜ O ( √ d max a ∈A ∑ i ∈ a σ 2 i ( a ) T ) ˜ O ( L √ mdT ) ˜ O ( L √ mdT ) ˜ O ( Lm √ dT )      | Θ(log log T ) Θ( T ) O ( d log log( Tm d Θ(log log T ) | O ( d 2 log( Tm )) O ( d 2 log log T ) Θ( T ) ( d log log( Tm d |
|                                | SROQ-GR-CMAB (our work)                                      |                                                                                                 | ))                                                     | O )) O ( d log log T )                                          |

we propose a scheduled oracle query framework that executes multiple independent oracle queries in parallel. Under this framework, SROQ-CMAB achieves regret ˜ O ( m √ dT ) with adaptivity complexity of Θ(log log T ) and query complexity of O ( d log log T ) .

- Covariance-adaptive oracle-efficient algorithms for linear rewards: Utilizing our proposed frameworks, we design oracle-efficient algorithms that leverage the estimated covariance structure of the reward noise. AROQ-C-CMAB achieves near-optimal regret ˜ O ( √ ∑ i ∈ [ d ] max a ∈A s.t. i ∈ a σ 2 i ( a ) T ) , where σ 2 i ( a ) denotes the variance contribution of base arm i under action a , with adaptivity and query complexities of O ( d 2 log( Tm )) . SROQ-C-CMAB achieves regret ˜ O ( √ d max a ∈A ∑ i ∈ a σ 2 i ( a ) T ) with adaptivity complexity of Θ(log log T ) and query complexity of O ( d 2 log log T ) .
- Oracle-efficient algorithm for general reward models: We extend our frameworks to general (non-linear) reward functions. AROQ-GR-CMAB achieves regret ˜ O ( L √ mdT ) with adaptivity and query complexities of O ( d log log( Tm/d )) , where L denotes the maximum possible value of the reward. SROQ-GR-CMAB achieves regret ˜ O ( Lm √ dT ) with adaptivity complexity of O (log log T ) and query complexity of Θ( d log log T ) .

Related Work. The combinatorial semi-bandit problems have been extensively studied, starting from the foundational work of Chen et al. [5]. Kveton et al. [18] established tight regret bounds that are near-optimal. Further improvements were made by Combes et al. [7], who derived better bounds under the assumption that the feedback from selected arms is independent.

More recently, a unified framework that accounts for both dependent and independent feedback through covariance analysis was introduced by Degenne and Perchet [9], assuming knowledge of the covariance matrix. This line of research has been further advanced by Perrault et al. [22] and Zhou et al. [32], who developed covariance-adaptive algorithms based on confidence ellipsoids. In addition to linear reward structures, generalized linear reward functions have also been studied in the combinatorial semi-bandit setting by Chen et al. [4].

Despite these advances, all of the aforementioned works require solving a combinatorial optimization problem frequently at every round using an offline oracle, which is generally NP-hard [8]. To alleviate the computational burden, Cuvelier et al. [8] proposed an approximation-based approach that achieves polynomial-time complexity. However, their method introduces a trade-off between regret and computational cost, as achieving near-optimal regret necessitates increasingly accurate

approximations-leading to potentially unbounded computational time. 1 Similarly, Chen et al. [5] considered approximation oracles but focused on minimizing approximate regret, rather than the original regret, and their method is limited to cases where such approximation oracles are available. Neu and Bartók [20] studied efficient algorithms in the adversarial semi-bandit setting, but their method still requires solving the optimization problem at every round and does not attain optimal regret in the stochastic setting. Similarly, Zhou et al. [32] proposed an adaptive covariance-based algorithm using ellipsoidal confidence regions, yet it also incurs oracle calls at every round. Lastly, Tzeng et al. [26] studied matroid semi-bandits with sublinear per-round computational complexity by exploiting matroid structure, a direction that is complementary to our focus on reducing the number of oracle calls for combinatorial semi-bandits with arbitrary action sets.

As a related line of research, oracle-efficient algorithms have been studied for submodular function optimization problems [1, 2, 11]. However, these approaches do not involve latent models that can be learned from stochastic sequential feedback, as in bandit learning. As a result, they differ fundamentally in formulation and are not applicable to the bandit setting. Oracle-efficient bandit algorithms have also been proposed for bandit linear optimization [17], achieving O (poly( d, log T )) oracle complexity. However, these approaches assume linear rewards with full-arm decisions and do not handle combinatorial action spaces or semi-bandit feedback, which are central to our setting. To the best of our knowledge, rare oracle queries in combinatorial semi-bandit problems have only been empirically explored by Combes et al. [7], who proposed a heuristic using O (log T ) oracle calls but without theoretical regret guarantees. Furthermore, several variants of combinatorial bandits heavily rely on frequent oracle queries, including Thompson Sampling methods [28], maximum-reward feedback settings [29], and pure exploration problems [3].

## 2 Problem Formulation

There are d base arms, and let A ⊆ { 0 , 1 } d denote the set of available actions, where each action a ∈ A is a binary vector indicating the activated base arms. We allow A to be an arbitrary subset of { 0 , 1 } d . Then, we define m = max a ∈A ∥ a ∥ 0 as the maximum number of activated base arms across all actions. At each time t ∈ [ T ] , the environment samples a vector of rewards y t ∈ [0 , 1] d from a fixed distribution D that is unknown to the agent, and the agent chooses an action a t ∈ A . For any vector x ∈ R d , we use x i to denote its i -th entry. Then, the agent receives a reward r ( a t , y t ) where r : A× [0 , 1] d → R , and observes the values of y t,i for each i ∈ [ d ] such that a t,i = 1 (semi-bandit feedback). The mean of the latent distribution D is denoted by µ = ( µ 1 , . . . , µ d ) . We first focus on the standard linear reward setting studied in prior work [18, 7, 9, 22, 32], where the reward is given by r ( a, y t ) := ⟨ a, y t ⟩ . We will discuss generalizations beyond the linear case later.

Regret. Let a ∗ be an optimal action, a ∗ ∈ argmax a ∈A ¯ r ( a ) , where ¯ r ( a ) = E y ∼D [ r ( a, y )] represents the expected reward function (e.g., ¯ r ( a ) = ⟨ a, µ ⟩ in the linear case). The goal is to minimize the cumulative regret over horizon T , defined as R ( T ) = E [ ∑ T t =1 (¯ r ( a ∗ ) -¯ r ( a t ))] .

Combinatorial Optimization. For finding an optimal action, it is required to solve the combinatorial optimization problem argmax a ∈A ¯ r ( a ) , whose computational cost, in general with arbitrary A ⊆ { 0 , 1 } d , is proportional to the size of A which is O ( d m ) . To address this computational complexity, the previous work on combinatorial semi-bandits [7, 5, 18] assumed access to an oracle, which returns a solution for the combinatorial optimization. However, these methods require querying the oracle at every round. In this work, we aim to substantially reduce the number of oracle queries while achieving tight regret. Formally, as in the previous work, we assume access to an oracle that returns a † ∈ argmax a ∈A f ( a ) for given f : A → R . Furthermore, our oracle-efficient approach can be incorporated with an approximation oracle, which will be discussed later.

Oracle Efficiency. Following the computational complexity notions introduced in [1, 11], we evaluate the oracle efficiency of our algorithms using two key measures, which are described as follows and illustrated in Figure 1: Query complexity refers to the total number of individual oracle queries made over the entire time horizon. This reflects the standard computational workload of the

1 To reach optimal regret, the approximation level δ t must satisfy lim t →∞ δ t = 0 , and each round incurs cost O (1 /δ t ) →∞

Proof.

Figure 1: Oracle adaptivity and query complexities.

<!-- image -->

algorithm. Adaptivity complexity captures the number of sequential rounds of oracle queries, where each round may consist of a set of queries that can be executed in parallel without depending on each other.

While query complexity is important in general, adaptivity complexity often dominates the actual runtime cost in parallel or distributed environments, where parallelizing queries within a round is easy, but synchronizing between rounds incurs overhead. Our goal is to reduce both measures of oracle complexity while maintaining tight regret guarantees.

Additional Notation. With a slight abuse of notation, we write i ∈ a for a ∈ A if the i -th coordinate of a satisfies a i = 1 . For any b, c ∈ R , we write b ≲ c to indicate that b is upper bounded by c up to a constant factor. We use b + for max { b, 0 } .

## 3 Oracle-Efficient Algorithms and Regret Analysis

We present two frameworks for combinatorial semi-bandit problems with rare oracle queries: one based on an adaptive oracle calls, and the other based on scheduled oracle queries, both inspired by batch learning [21, 25, 13, 6, 10, 31, 15, 23, 30, 14].

## 3.1 Adaptive Rare Oracle Queries

We propose an algorithm (Algorithm 1) that leverages adaptive oracle queries. The algorithm employs a UCB-based strategy with adaptive epoch-based updates, enabling efficient exploration despite infrequent oracle access. We use the UCB indices defined as: for some constant C &gt; 0 ,

<!-- formula-not-decoded -->

where ˆ µ t,i = (1 /n t,i ) ∑ t -1 s =1 y s,i ✶ ( i ∈ a s ) and n t,i = ∑ t -1 s =1 ✶ ( i ∈ a s ) . The UCB index for each action is updated when there exists a base arm i whose selection count exceeds a specified threshold. More precisely, the indices are updated when the number of rounds in which arm i has been selected in the current epoch, denoted by |T i ( τ i ) | , satisfies |T i ( τ i ) | ≥ 1 + √ Tm · |T i ( τ i -1) | /d , where τ i is the epoch index for arm i , and T i ( τ ) is the set of rounds in epoch τ where i was selected.

The intuition for the threshold condition of oracle queries is as follows: for i ∈ [ d ] with τ i , the instance regret is bounded by √ 1 / |T i ( τ i -1) | so that with the bound for |T i ( τ i ) | from the update condition, the overall regret for τ i epochs is bounded by |T i ( τ i ) | √ 1 / |T i ( τ i -1) | ≲ √ Tm |T i ( τ i -1) | /d √ 1 / |T i ( τ i -1) | = √ Tm/d . By considering all τ i for i ∈ [ d ] , we can obtain the near-optimal regret bound with oracle efficiency.

Theorem 1 With oracle adaptivity and query complexities of O ( d log log( Tm/d )) , respectively, Algorithm 1 achieves a regret bound of

<!-- formula-not-decoded -->

The full version of the proof is provided in Appendix A.2.

```
Algorithm 1 Adaptive Rare Oracle Queries for Combinatorial MAB ( AROQ-CMAB ) Initialize: τ i = 1 for all i ∈ [ d ] for t = 1 , 2 ..., T do for i ∈ [ d ] s.t. |T i ( τ i ) | ≥ 1 + √ Tm · |T i ( τ i -1) | /d do τ i ← τ i +1 , T i ( τ i ) ←∅ Update ← True if Update = True then a t ← arg max a ∈A r UCB t ( a ) with (1) // Oracle Query Update ← False else a t ← a t -1 Play a t and observe feedback y t,i for i ∈ a t T i ( τ i ) ←T i ( τ i ) ∪ { t } for all i ∈ a t
```

Comparison to Previous Work. The worst-case regret lower bound for this problem is known to be Ω( √ mdT ) [5], and a near-optimal regret was achieved by the algorithm of Chen et al. [5], Kveton et al. [18]. However, their approach requires Θ( T ) oracle adaptivity and query complexity, respectively. In contrast, our proposed algorithm, AROQ-CMAB (Algorithm 1), achieves a near-optimal regret bound while significantly reducing both oracle adaptivity and query complexities to O ( d log log( Tm/d )) .

α -Approximation Oracle. Prior work on combinatorial bandits [5] mitigates the cost of exact oracles by using α -approximation oracles, which return an action a † satisfying f ( a † ) ≥ α max a ∈A f ( a ) for some approximation factor α ∈ [0 , 1] , when such oracles are available. This leads to analyzing the regret relative to the best α -approximate reward, rather than the true optimal reward. In contrast, our method targets the original (non-approximate) regret while reducing the frequency of oracle queries. Nevertheless, our framework can naturally incorporate α -approximation oracles, achieving the same regret guarantee in terms of α -approximate regret with only infrequent approximate oracle calls. See Appendix A.3 for details.

## 3.2 Scheduled Rare Oracle Queries

To further reduce adaptivity complexity and enable more efficient distributed computation, we propose an algorithm that performs scheduled batched oracle queries at predetermined epochs (Algorithm 2), allowing these oracle queries to be synchronized and executed in parallel. At each epoch τ , we define UCB and LCB indices as: for some constant C &gt; 0

<!-- formula-not-decoded -->

where ˆ µ τ,i = (1 /n τ,i ) ∑ t τ -1 t =1 y t,i ✶ ( i ∈ a t ) , n τ,i = ∑ t τ -1 t =1 ✶ ( i ∈ a t ) , and t τ denotes the start time of the τ -th epoch.

To schedule oracle queries, we adopt an elimination-based bandit strategy [25]. However, in combinatorial bandits, the exponentially large number of suboptimal actions poses a significant challenge for efficient exploration. To address this, we construct a representative action a ( i ) τ for each base arm i ∈ [ d ] in epoch τ . Using elimination conditions applied to these representative actions, we can efficiently eliminate a large number of suboptimal actions by focusing on suboptimal base arms-those that are not part of the optimal action a ∗ . In each epoch, these representative actions are selected for exploration, after which the estimators and representative actions are updated. This process requires oracle queries but only at the batch level, keeping the query frequency low.

Let T = { t 1 , . . . , t M } denote the set of time steps at which oracle queries are made, where M &gt; 0 , t 1 = 1 , t M = T , and for 1 &lt; τ &lt; M , the sequence is defined recursively as t τ = η √ t τ -1 . The scaling factor is set to η = T 1 / (2 -2 1 -M ) . We choose M = Θ(log log T ) to ensure doubly logarithmic adaptivity and query complexity. Parallel execution of oracle queries is discussed in Appendix A.1.

```
Algorithm 2 Scheduled Rare Oracle Queries for Combinatorial MAB ( SROQ-CMAB ) Input: T // Oracle Queries with (2) // Oracle Query
```

```
1 for τ = 1 , 2 , . . . , M do 2 Update ˆ µ τ := (ˆ µ τ, 1 , . . . , ˆ µ τ,d ) 3 a ( i ) τ := argmax a ∈A τ -1 : i ∈ a r UCB τ ( a ) for all i ∈ N τ -1 with (2) 4 N τ ←{ i ∈ N τ -1 | r UCB τ ( a ( i ) τ ) ≥ max a ∈A τ -1 r LCB τ ( a ) } 5 A τ ←{ a ∈ A τ -1 | a i = 0 for all i ∈ [ d ] / N τ } 6 T τ ← [ t τ , t τ +1 -1] 7 for t ∈ T τ do 8 i ← ( ( t -1) mod |N τ | +1 ) -th element of N τ . 9 Play a t = a ( i ) τ and observe feedback y t,i for i ∈ a t
```

Theorem 2 With oracle adaptivity complexity of Θ(log log T ) and oracle query complexity of O ( d log log T ) , Algorithm 2 achieves a regret bound of

<!-- formula-not-decoded -->

Proof. The full version of the proof is provided in Appendix A.4.

We observe that Algorithm 2 improves the adaptivity complexity from O ( d log log( Tm/d )) in Algorithm 1 to Θ(log log T ) , at the cost of an additional √ m factor in the regret.

Remark 1 In practice, Algorithm 2 can be more computationally efficient than Algorithm 1. First, due to its reduced adaptivity complexity, Algorithm 2 enables more efficient parallel execution of oracle queries. Second, the elimination process progressively discards suboptimal base arms, reducing the oracle query complexity per round from O ( d m ) to O ( |N τ | m ) , where |N τ | ≤ d denotes the number of remaining base arms at epoch τ . These computational advantages are further supported by our experimental results presented later.

Our proposed frameworks for combinatorial semi-bandits with rare oracle queries can be extended to variants of the combinatorial semi-bandit, including covariance-dependent CMAB and general-reward CMAB. In the following, we examine each of these settings in turn.

## 4 Extension to Covariance-dependent CMAB

In this section, instead of targeting worst-case regret, we consider covariance-dependent regret, inspired by [9, 22, 32]. The covariance-dependent analysis can cover independent or dependent (worst-case) semi-bandit rewards of arms in an action. Here, the covariance matrix for the reward distribution D is denoted by Σ ∈ R d × d , which is assumed to be unknown to the agent.

For simplicity, we assume that for any 1 ≤ i ≤ j ≤ d , there exists an action a ∈ A such that a i = a j = 1 . When this assumption does not hold, our algorithms and analysis extend naturally by restricting attention to observable pairs , that is, pairs ( i, j ) for which there exists an action a ∈ A with a i = a j = 1 . All results continue to hold under this extension, with the same regret guarantees.

In the following, we propose covariance-adaptive algorithms based on our two frameworks-adaptive and scheduled rare oracle queries-to handle this setting.

## 4.1 Adaptive Rare Oracle Queries for Covariance-adaptive Approach

We first propose an algorithm (Algorithm 3) based on the adaptive rare oracle query framework. Recall that n t,i = ∑ t -1 s =1 ✶ ( i ∈ a s ) and n t := ( n t, 1 , . . . , n t,d ) . Let D x and D X denote diagonal matrices, where D x has the entries of vector x on its diagonal, and D X has the diagonal entries of matrix X .

We define the estimated means as ˆ µ t = D -1 n t ∑ t -1 s =1 D a s y s . We also define covariance estimator ˆ Σ t = ˆ S t -ˆ µ t ˆ µ ⊤ t where ˆ S t, ( i,j ) = 1 n t, ( i,j ) ∑ t -1 s =1 a s,i a s,j y s,i y s,j , and confidence bound Σ t, ( i,j ) =

```
Algorithm 3 Adaptive Rare Oracle Queries for Covariance-adaptive CMAB ( AROQ-C-CMAB ) Initialize: τ i,j = 0 for all i, j ∈ [ d ] × [ d ] for t = 1 , 2 ..., T do if t ≤ ⌈ d ( d +1)log 3 ( T ) / 2 ⌉ then Let ( i, j ) be the ( ( t -1) mod d ( d +1) 2 +1 ) -th pair in a fixed enumeration of all pairs ( i, j ) with 1 ≤ i ≤ j ≤ d a t ← any a ∈ A s.t. i ∈ a, j ∈ a if t = ⌈ d ( d +1)log 3 ( T ) / 2 ⌉ then τ i,j ← τ i,j +1 for all i, j ∈ [ d ] × [ d ] else for i, j ∈ [ d ] × [ d ] s.t. |T i,j ( τ i,j ) | ≥ 1 + 2 |T i,j ( τ i,j -1) | do τ i,j ← τ i,j +1 , T i,j ( τ i,j ) ←∅ Update ← True if Update = True then a t ← arg max a ∈A r UCB t ( a ) with (3) // Oracle Query Update ← False else a t ← a t -1 Play a t and observe feedback y i,t for i ∈ a t T i,j ( τ i,j ) ←T i,j ( τ i,j ) ∪ { t } for all i ∈ a t and j ∈ a t ,
```

ˆ Σ t, ( i,j ) + 1 4 ( 5 h t √ n t, ( i,j ) + h 2 t n t, ( i,j ) + 1 n 2 t, ( i,j ) ) , where n t, ( i,j ) = ∑ t -1 s =1 ✶ ( i ∈ a s ) ✶ ( j ∈ a s ) with n t, ( i,i ) = n t,i , and h t = O ( √ log t +log d ) . Define the gram matrix G t = ∑ t -1 s =1 D a s Σ t D a s + D Σ t D n t + I . Then, we utilize the UCB index, defined as

<!-- formula-not-decoded -->

where f t = O (log t + d log log t ) .

To initialize, the algorithm uniformly explores actions as a warm-up phase for the stability of the covariance estimator. Then in the main stage, for the adaptive update condition, we adopt a stricter criterion than that of (worst-case) Algorithm 1, resulting in more frequent updates-though oracle queries remain rare-while handling the covariance-dependent bound to achieve a tighter guarantee. For the covariance dependent regret bound, we define σ 2 i ( a ) = ∑ j ∈ a (Σ i,j ) + . The algorithm achieves a near-optimal regret bound asymptotically as follows.

Theorem 3 With oracle adaptivity and query complexities of O ( d 2 log( Tm )) , respectively, Algorithm 3 achieves an asymptotic regret bound of

<!-- formula-not-decoded -->

Proof. The full version of the proof is provided in Appendix A.5.

In the worst case of dependent base arm rewards, our regret bound becomes ˜ O ( √ mdT ) , which is the same as that of Algorithm 1. However, for the independent reward case across all base arms (i.e., Σ = I ), the regret bound becomes ˜ O ( √ dT ) , which is tighter by a factor of √ m .

Comparison to Previous Work. As discussed in [32], the proposed algorithms in [9, 22] achieved gap-dependent asymptotic regret bound, which is not tight for the gap-free bound with respect to T because of the additional 1 / ∆ 2 min factor. The regret lower bound of this problem is Ω( √ T ∑ i ∈ [ d ] max a ∈A s.t. i ∈ a σ 2 i ( a )) [32] and [32] propose an algorithm achieving near-optimal gapfree asymptotic regret bound of ˜ O ( √ T ∑ i ∈ [ d ] max a ∈A s.t. i ∈ a σ 2 i ( a )) with Θ( T ) oracle adaptivity

Algorithm 4 Scheduled Rare Oracle Queries for Covariance-adaptive CMAB ( SROQ-C-CMAB )

̸

```
Input: T 10 for t ∈ [1 , ⌈ d ( d +1) / 2 ⌉ ] do 11 Let ( i, j ) be the ( ( t -1) mod d ( d +1) 2 +1 ) -th pair in a fixed enumeration of all pairs ( i, j ) with 1 ≤ i ≤ j ≤ d 12 a t ← any a ∈ A s.t. i ∈ a, j ∈ a 13 for τ = 1 , 2 , . . . , M do 14 Update ˆ µ τ,i , ˆ S τ, ( i,j ) for all ( i, j ) ∈ [ d ] × [ d ] 15 a ( i ) τ := argmax a ∈A τ -1 : i ∈ a r UCB τ ( a ) for all i ∈ N τ -1 with (4) // Oracle Queries 16 a ( i,j ) τ := argmax a ∈A τ -1 : i,j ∈ a r UCB τ ( a ) for all ( i, j ) ∈ N (2) τ -1 : i = j // Oracle Queries 17 N τ ←{ i ∈ N τ -1 | r UCB τ ( a ( i ) τ ) ≥ max a ∈A τ -1 r LCB τ ( a ) } with (4) // Oracle Query 18 A ′ τ ←{ a ∈ A τ -1 | a i = 0 for all i ∈ [ d ] / N τ } 19 N (2) τ ←{ ( i, j ) ∈ N τ ×N τ | r UCB τ ( a ( i,j ) τ ) ≥ max a ∈A ′ τ r LCB τ ( a ) , i = j } // Oracle Query 20 A τ ←{ a ∈ A ′ τ | a i = 0 or a j = 0 for all ( i, j ) ∈ [ d ] × [ d ] / N (2) τ , i = j } 21 T (1) τ ← [ t τ , t τ + T τ -( d 2 m 2 T τ log T ) 2 / 3 -1] , T (2) τ ← [ t τ + T τ -( d 2 m 2 T τ log T ) 2 / 3 , t τ +1 -1] 22 for t ∈ T (1) τ do 23 i ← ( t mod |N τ | ) -th element in N τ 24 Play a t = a ( i ) τ 25 Receive reward ⟨ a t , y t ⟩ and observe feedback y t,i for i ∈ [ d ] s.t. a t,i = 1 26 for t ∈ T (2) τ do 27 ( i, j ) ← ( t mod |N (2) τ | ) -th element in N (2) τ 28 Play a t = a ( i,j ) τ 29 Receive reward ⟨ a t , y t ⟩ and observe feedback y t,i for i ∈ [ d ] s.t. a t,i = 1
```

̸

and query complexity, respectively. Our algorithm achieves the near-optimal asymptotic regret bound with reduced oracle adaptivity and query complexity of O ( d 2 log( mT )) , respectively.

## 4.2 Scheduled Rare Oracle Queries for Covariance-adaptive Approach

Here, we propose a covariance-adaptive algorithm (Algorithm 4) by utilizing the framework of scheduled rare oracle queries. Recall that for each epoch τ , ˆ µ τ,i = (1 /n τ,i ) ∑ t τ -1 t =1 y t,i ✶ ( i ∈ a t ) where n τ,i = ∑ t τ -1 t =1 ✶ ( i ∈ a t ) and t τ is the start time of epoch τ in the algorithm. For the covariance estimator, we define ˆ Σ τ = ˆ S τ -ˆ µ τ ˆ µ ⊤ τ where ˆ S τ, ( i,j ) = (1 /n τ,i,j ) ∑ t τ -1 t =1 a t,i a t,j y t,i y t,j , and confidence bound Σ τ, ( i,j ) = ˆ Σ τ, ( i,j ) + 1 4 ( 5 h T √ n τ, ( i,j ) + h 2 T n τ, ( i,j ) + 1 n 2 τ, ( i,j ) ) , where n τ, ( i,j ) = ∑ t τ -1 t =1 ✶ ( i ∈ a t ) ✶ ( j ∈ a t ) , n τ, ( i,i ) = n τ,i , and h T = O ( √ log T +log d ) . Define the gram matrix G τ = ∑ t τ -1 s =1 D a s Σ τ D a s + D Σ τ D n τ + I . Then, for the confidence bounds, we utilize

<!-- formula-not-decoded -->

where f T = O (log T + d log log T ) . For scheduled oracle queries, we employ the same time grid T = { t 1 , . . . , t M } , as in Algorithm 2, shifted by ⌈ d ( d +1) / 2 ⌉ time steps to account for the warm-up phase, that is, t k ← t k + ⌈ d ( d +1) / 2 ⌉ for all k ∈ [ M ] . Parallel execution of oracle is discussed in Appendix A.1.

Theorem 4 With oracle adaptivity complexity of Θ(log log T ) and oracle query complexity of O ( d 2 log log T ) , Algorithm 4 achieves an asymptotic regret bound of

<!-- formula-not-decoded -->

̸

Proof. The full version of the proof is provided in Appendix A.6.

In the worst case, the scheduled query-based Algorithm 4 achieves a regret bound of ˜ O ( m dT ) , which matches that of the scheduled query-based Algorithm 2 but is larger than the bounds achieved by the adaptive oracle query framework of Algorithm 1 and Algorithm 3 by a factor of √ m . This gap arises from the inefficiency of using a fixed-time framework compared to an adaptive-time framework. However, the oracle complexities of Algorithm 4 are significantly lower than the O ( d 2 log( mT )) complexities of Algorithm 3, achieving Θ(log log T ) adaptivity and O ( d 2 log log T ) query complexity.

## 5 Extension to General-Reward CMAB

In this section, beyond a linear reward function we explored in the previous sections, we consider general reward functions r ( a, y ) defined on A× [0 , 1] d → [0 , L ] for L &gt; 0 . We adopt the same setting with assumptions as in [4]. Specifically, at each time t , the arms produce stochastic outcomes ( y t,i ) d i =1 ∈ [0 , 1] d drawn i.i.d. over time from a distribution D with finite support 2 and the expected reward is denote by ¯ r ( a ) = E y ∼D [ r ( a, y )] . In this setting, we consider the following assumption.

Assumption 1 (Monotone reward function) For any y, y ′ ∈ [0 , 1] d satisfying y i ≤ y ′ i for all i ∈ [ d ] and any a ∈ A , we have r ( a, y ) ≤ r ( a, y ′ ) .

The assumption for the monotone reward function is commonly observed in various combinatorial problems such as K -MAX [24], K -SUM [5], and Expected Utility Maximization [19].

By adopting our adaptive and scheduled frameworks for rare oracle queries, we propose oracleefficient algorithms for general reward CMAB; Algorithms 6 and 7, respectively. The details of the algorithms are provided in Appendix A.7. In what follows, we provide theorems for oracle complexities and regret of each algorithm. The proofs are provided in Appendices A.8, A.9.

Theorem 5 With oracle adaptivity and query complexities of O ( d log log( Tm/d )) , respectively, Algorithm 6 achieves a regret bound of R ( T ) = ˜ O ( L √ dmT ) .

Theorem 6 With oracle adaptivity complexity of Θ(log log T ) and oracle query complexity of O ( d log log T ) , Algorithm 7 achieves a regret bound of R ( T ) = ˜ O ( Lm √ dT ) .

Comparison to Previous Work. Chen et al. [4] proposed an algorithm achieving a regret bound of ˜ O ( L √ mdT ) with oracle adaptivity and oracle query complexities of Θ( T ) . However, our algorithms achieve ˜ O ( L √ mdT ) and ˜ O ( Lm √ dT ) , respectively, requiring significantly reduced oracle complexities of of order log log T .

## 6 Experiments

We compare our algorithms to benchmarks in terms of oracle efficiency and regret using synthetic datasets 3 . We begin with the linear reward setting, where the mean vector is sampled from Unif [0 , 1] with d = 20 and m = 3 , and stochastic rewards are uniformly generated around these means at each round. As shown in Figure 2 (a,b), our algorithms ( AROQ-CMAB , SROQ-CMAB ) achieve significantly lower oracle adaptivity and query complexities than CUCB [5], consistent with Theorems 1 and 2. Importantly, as shown in Figure 2 (d), our algorithms achieve faster runtime than the benchmark. In particular, SROQ-CMAB outperforms AROQ-CMAB in runtime, benefiting from a lower total adaptivity complexity up to T (Figure 2 (e)), which enables more efficient parallel oracle execution (Remark 1). Figure 2(c) demonstrates that AROQ-CMAB incurs slightly higher regret and SROQ-CMAB incurs somewhat larger regret than CUCB , which is consistent with our theoretical predictions: the regret bounds involve an additional logarithmic factor for AROQ-CMAB and a √ m

2 The finite-support assumption simplifies the algorithms and analysis but is not essential. As noted in Chen et al. [4], the results can extend to Lipschitz-continuous reward functions by discretization techniques that preserve the same regret bound (see Appendix A.10).

3 Source Code: https://github.com/junghunkim7786/OracleEfficientCombinatorialBandits

√

Figure 2: Experimental results for linear rewards with d = 20 and m = 3 .

<!-- image -->

factor for SROQ-CMAB . Additional results for the covariance-adaptive variants and general reward functions are provided in Appendix A.11.

## 7 Conclusion

In this work, we proposed oracle-efficient algorithms for semi-combinatorial bandits. We introduced two algorithmic frameworks for handling rare oracle queries-adaptive and scheduled-and demonstrated that our algorithms significantly improve oracle efficiency while maintaining tight regret guarantees for worst-case linear rewards, covariance-dependent linear rewards, and general (non-linear) reward functions.

Societal Impact. The research is primarily theoretical and does not engage with human subjects, sensitive data, or domains with identifiable risks of negative societal impact.

## Acknowledgements

J. Kim acknowledges the support of ANR through the PEPR IA FOUNDRY project (ANR-23PEIA-0003) and the Doom project (ANR-23-CE23-0002), as well as the ERC through the Ocean project (ERC-2022-SYG-OCEAN-101071601). M. Oh was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2022-NR071853 and RS-2023-00222663), by the Global-LAMP Program of the NRF grant funded by the Ministry of Education (No. RS-2023-00301976), and by AI-Bio Research Grant through Seoul National University.

## References

- [1] Balkanski, E. and Singer, Y. (2018). The adaptive complexity of maximizing a submodular function. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing , STOC 2018, page 1138-1151, New York, NY, USA. Association for Computing Machinery.
- [2] Chekuri, C. and Quanrud, K. (2019). Parallelizing greedy for submodular set function maximization in matroids and beyond. In Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing , pages 78-89.
- [3] Chen, S., Lin, T., King, I., Lyu, M. R., and Chen, W. (2014). Combinatorial pure exploration of multi-armed bandits. Advances in neural information processing systems , 27.

- [4] Chen, W., Hu, W., Li, F., Li, J., Liu, Y., and Lu, P. (2016). Combinatorial multi-armed bandit with general reward functions. Advances in Neural Information Processing Systems , 29.
- [5] Chen, W., Wang, Y., and Yuan, Y. (2013). Combinatorial multi-armed bandit: General framework and applications. In International conference on machine learning , pages 151-159. PMLR.
- [6] Chen, X., Krishnamurthy, A., and Wang, Y. (2024). Robust dynamic assortment optimization in the presence of outlier customers. Operations Research , 72(3):999-1015.
- [7] Combes, R., Talebi Mazraeh Shahi, M. S., Proutiere, A., et al. (2015). Combinatorial bandits revisited. Advances in neural information processing systems , 28.
- [8] Cuvelier, T., Combes, R., and Gourdin, E. (2021). Statistically efficient, polynomial-time algorithms for combinatorial semi-bandits. Proceedings of the ACM on Measurement and Analysis of Computing Systems , 5(1):1-31.
- [9] Degenne, R. and Perchet, V. (2016). Combinatorial semi-bandit with known covariance. Advances in Neural Information Processing Systems , 29.
- [10] Dong, K., Li, Y., Zhang, Q., and Zhou, Y. (2020). Multinomial logit bandit with low switching cost. In International Conference on Machine Learning , pages 2607-2615. PMLR.
- [11] Fahrbach, M., Mirrokni, V., and Zadimoghaddam, M. (2019). Non-monotone submodular maximization with nearly optimal adaptivity and query complexity. In International Conference on Machine Learning , pages 1833-1842. PMLR.
- [12] Feldman, J., Henzinger, M., Korula, N., Mirrokni, V. S., and Stein, C. (2010). Online stochastic packing applied to display ad allocation. In European Symposium on Algorithms , pages 182-194. Springer.
- [13] Gao, Z., Han, Y., Ren, Z., and Zhou, Z. (2019). Batched multi-armed bandits problem. Advances in Neural Information Processing Systems , 32.
- [14] Hanna, O., Yang, L., and Fragouli, C. (2023a). Efficient batched algorithm for contextual linear bandits with large action space via soft elimination. Advances in Neural Information Processing Systems , 36:56772-56783.
- [15] Hanna, O. A., Yang, L., and Fragouli, C. (2023b). Contexts can be cheap: Solving stochastic contextual bandits with linear bandit algorithms. In The Thirty Sixth Annual Conference on Learning Theory , pages 1791-1821. PMLR.
- [16] Hao, J., Zhao, T., Li, J., Dong, X. L., Faloutsos, C., Sun, Y ., and Wang, W. (2020). P-companion: A principled framework for diversified complementary product recommendation. In Proceedings of the 29th ACM International Conference on Information &amp; Knowledge Management , pages 2517-2524.
- [17] Ito, S., Hatano, D., Sumita, H., Takemura, K., Fukunaga, T., Kakimura, N., and Kawarabayashi, K.-I. (2019). Oracle-efficient algorithms for online linear optimization with bandit feedback. Advances in Neural Information Processing Systems , 32.
- [18] Kveton, B., Wen, Z., Ashkan, A., and Szepesvari, C. (2015). Tight regret bounds for stochastic combinatorial semi-bandits. In Artificial Intelligence and Statistics , pages 535-543. PMLR.
- [19] Li, J. and Deshpande, A. (2011). Maximizing expected utility for stochastic combinatorial optimization problems. In 2011 IEEE 52nd Annual Symposium on Foundations of Computer Science , pages 797-806. IEEE.
- [20] Neu, G. and Bartók, G. (2013). An efficient algorithm for learning with semi-bandit feedback. In International Conference on Algorithmic Learning Theory , pages 234-248. Springer.
- [21] Perchet, V., Rigollet, P., Chassang, S., and Snowberg, E. (2015). Batched bandit problems. In Conference on Learning Theory , pages 1456-1456. PMLR.
- [22] Perrault, P., Valko, M., and Perchet, V . (2020). Covariance-adapting algorithm for semi-bandits with application to sparse outcomes. In Conference on Learning Theory , pages 3152-3184. PMLR.

- [23] Ren, X., Jin, T., and Xu, P. (2024). Optimal batched linear bandits. In International Conference on Machine Learning , pages 42391-42416. PMLR.
- [24] Simchowitz, M., Jamieson, K., and Recht, B. (2016). Best-of-k-bandits. In Conference on Learning Theory , pages 1440-1489. PMLR.
- [25] Slivkins, A. et al. (2019). Introduction to multi-armed bandits. Foundations and Trends® in Machine Learning , 12(1-2):1-286.
- [26] Tzeng, R.-C., Ohsaka, N., and Ariu, K. (2024). Matroid semi-bandits in sublinear time. In Proceedings of the 41st International Conference on Machine Learning , pages 48855-48877.
- [27] Vesselinova, N., Steinert, R., Perez-Ramirez, D. F., and Boman, M. (2020). Learning combinatorial optimization on graphs: A survey with applications to networking. IEEE Access , 8:120388-120416.
- [28] Wang, S. and Chen, W. (2018). Thompson sampling for combinatorial semi-bandits. In International Conference on Machine Learning , pages 5114-5122. PMLR.
- [29] Wang, Y., Chen, W., and Vojnovi´ c, M. (2023). Combinatorial bandits for maximum value reward function under max value-index feedback. arXiv preprint arXiv:2305.16074 .
- [30] Yu, S. and Oh, M.-h. (2025). Optimal and practical batched linear bandit algorithm. In Proceedings of the 42nd International Conference on Machine Learning , volume 267 of Proceedings of Machine Learning Research , pages 73262-73285. PMLR.
- [31] Zhang, Z., Ji, X., and Zhou, Y. (2021). Almost optimal batch-regret tradeoff for batch linear contextual bandits. arXiv preprint arXiv:2110.08057 .
- [32] Zhou, J., Gaillard, P., Rahier, T., Zenati, H., and Arbel, J. (2024). Towards efficient and optimal covariance-adaptive algorithms for combinatorial semi-bandits. Advances in Neural Information Processing Systems , 37:30474-30521.

## A Appendix

## A.1 Details for Parallel Execution of Oracle Queries in the Scheduled Framework

This section provides further details on the parallel execution of oracle queries within the scheduled framework. Specifically, we describe how our proposed algorithms (Algorithms 2, 4, and 7) support parallel execution of oracle queries to reduce adaptivity complexity.

Worst-Case Linear Rewards. In Algorithm 2, at the beginning of each epoch, the oracle queries in Line 3 can be executed in parallel. Moreover, although Line 4 depends on the result of Line 3, the oracle queries in Lines 3 and 4 can still be executed in parallel, as they involve independent computations. Specifically, evaluating max a ∈A r LCB τ ( a ) in Line 4 requires only a single independent oracle call that returns the maximizer a † = argmax a ∈A r LCB τ ( a ) , allowing us to directly retrieve the maximum value r LCB τ ( a † ) .

Covariance-Dependent Linear Rewards. In Algorithm 4, the oracle queries in Lines 15, 16, and 17 can be executed in parallel, following the same principle as in the worst-case linear rewards setting. In contrast, the query in Line 19 must be performed sequentially afterward since the oracle relies on the updated set A ′ τ resulting from the preceding computation.

General Rewards. In Algorithm 7, following the same principle as in the worst-case linear rewards setting, the oracle queries in Lines 32 and 33 can be executed in parallel.

## A.2 Proof of Theorem 1

Let τ i ( t ) be the value of τ i at time t in the algorithm. Then τ i ( T ) represents the number of updates, each update requiring oracle queries, up to T from arm i . Then, for the bound of the total oracle queries up to T , by adopting the proof techniques in [10], we have the following lemma.

Lemma 1 (Oracle Queries Bound) We have

<!-- formula-not-decoded -->

Proof. We first provide the following lemma.

Lemma 2 For M ≥ 0 and a sequence x 0 , x 1 , . . . such that x i ≥ 1 + √ Mx i -1 for all i ≥ 1 , we have that x τ ≥ M 1 -2 -τ +1 for all τ ≥ 1 .

Proof. For τ = 1 , we have

<!-- formula-not-decoded -->

which satisfies the desired inequality. We now proceed by induction. Suppose that for some τ ≥ 1 , the inequality x τ ≥ M 1 -2 -τ +1 holds. Then, using the recurrence, we have

<!-- formula-not-decoded -->

Thus, by induction, we conclude that

<!-- formula-not-decoded -->

Let τ 0 = log log( Tm/d ) . From Lemma 2, if τ ≥ τ 0 +1 and τ is not the last stage, for any i ∈ [ d ] , we have

<!-- formula-not-decoded -->

Therefore, from the fact that ∑ T t =1 ∥ a t ∥ 0 ≤ mT , there are at most 2 d + d (including the last stages for all i ∈ [ d ] ) pairs of ( i, τ ) for i ∈ [ d ] and τ ∈ [ τ i ( T )] satisfying τ ≥ τ 0 +1 . This implies that for i ∈ [ d ] s.t. τ i ( T ) ≥ τ 0 +1 , we have ∑ i ∈ [ d ] ✶ ( τ i ( T ) ≥ τ 0 +1) τ i ( T ) ≤ dτ 0 +3 d . Therefore, we have which concludes the proof.

<!-- formula-not-decoded -->

For the proof of regret bound, we utilize the Azuma-Hoeffding inequality provided in the following lemma.

Lemma 3 (Azuma-Hoeffding Inequality) For a martingale difference sequence X 1 , . . . , X n with support of size 1 for all X i , for γ &gt; 0 we have

<!-- formula-not-decoded -->

From Lemma 3, we can show that the event E t = {| µ i -ˆ µ s,i | ≤ √ 1 . 5 log s n s,i ∀ 1 ≤ s ≤ t ∀ i ∈ [ d ] } holds with probability of at least 1 -O ( d t 2 ) . For ease of presentation, we define auxiliary variables n ′ t,i and ˆ µ ′ t,i for each i ∈ [ d ] and time step t ∈ [ T ] for the analysis on the rare updated indexes as follows: If the selected action at time t is newly updated in the algorithm (i.e., Update = True ), then

<!-- formula-not-decoded -->

Otherwise, if the previously selected action is maintained (i.e., Update = False ), then

<!-- formula-not-decoded -->

Using these adjusted statistics, we define the UCB-based optimistic reward estimate for any action a as

<!-- formula-not-decoded -->

Now we provide a bound for the regret as follows:

<!-- formula-not-decoded -->

where the second and last inequalities are obtained from E t . For bounding the last term, we have

<!-- formula-not-decoded -->

≲

log log(

Tm/d

)

dmT

log

T, where the first inequality is obtained from n ′ t,i ≥ |T i ( τ -1) | for t ∈ T i ( τ ) , the second equality is obtained from the condition of updates in the algorithm, and the last inequality is obtained from Lemma 1.

Oracle Complexity Bounds. Based on Lemma 1, we can show that the oracle query complexity is bounded by O ( d log log( Tm/d )) . Since the epochs for each base arm are updated separately, the adaptivity complexity is also bounded by O ( d log log( Tm/d )) .

## A.3 α -Approximation Oracle

In this section, we provide a detailed explanation for α -approximation oracle. We focus on the adaptive rare oracle query framework, noting that similar results can be derived for the scheduled framework, which we omit to avoid redundancy. Instead of obtaining the exact solution, the α -approximation oracle, denoted by O α t , outputs action a α t ∈ A satisfying

<!-- formula-not-decoded -->

for α &gt; 0 . We investigate the α -regret, which is defined as

<!-- formula-not-decoded -->

Algorithm 5 α -approximated Adaptive Rare Oracle Queries for Combinatorial MAB ( α -AROQ-CMAB )

<!-- formula-not-decoded -->

Theorem 7 With oracle adaptivity and query complexities of O ( d log log( Tm/d )) , respectively, Algorithm 5 achieves a α -regret bound of

<!-- formula-not-decoded -->

Proof. Here we provide the part that is different from the proof of Theorem 1. We provide a bound for the regret as follows:

<!-- formula-not-decoded -->

where the second last inequality is obtained from O α t . The other parts of the proof are the same as those of Theorem 1.

## A.4 Proof of Theorem 2

From Lemma 3, we can show that the event E = {| µ i -ˆ µ i ( τ ) | ≤ √ 1 . 5 log T n τ,i ∀ τ ∈ [ M ] , ∀ i ∈ [ d ] }

holds with probability of at least 1 -d T 2 . Then by adopting the proof technique in [6], under the event E , we show that A τ , activated arm set for τ -th epoch, always contains optimal arm a ∗ in the following lemma.

Lemma 4 Under E , we can show that for all τ ∈ [ M ] , a ∗ ∈ A τ .

Proof. This can be shown by induction. Suppose a ∗ ∈ A τ for fixed τ ∈ [ M ] . Under E , we have ¯ r ( a ) ≤ r UCB τ +1 ( a ) and ¯ r ( a ) ≥ r LCB τ +1 ( a ) for any a ∈ A τ . Then for any fixed i ∈ a ∗ , for any a ∈ A τ , we can show that

<!-- formula-not-decoded -->

where the first inequality is obtained from the definition of a ( i ) τ +1 with a ∗ ∈ A τ . This implies that

<!-- formula-not-decoded -->

which implies that i ∈ a ∗ is not eliminated from the elimination condition at the τ +1 -th epoch. This holds for all i ∈ a ∗ so that a ∗ ∈ A τ +1 . With a ∗ ∈ A 0 = A , we can conclude the induction.

Under E , we have

<!-- formula-not-decoded -->

where the first inequality is obtained from E , the second inequality comes from the fact that a ∗ ∈ A τ -1 from Lemma 4, and max a ∈A τ -1 r LCB τ ( a ) ≤ r UCB τ ( a ( i ) τ ) from the algorithm, and the third inequality is obtained from n τ,i ≥ ∑ t ∈T τ -1 ✶ ( i ∈ a t ) ≳ |T τ -1 | /d from the exploration in the algorithm at the τ -1 -th epoch. Finally, we can show that

<!-- formula-not-decoded -->

which concludes the proof for the regret bound.

Oracle Complexity Bounds. Based on the oracle calls in Lines 3 and 4 of Algorithm 2, we observe that each epoch involves at most d independent oracle queries (Line 3) and one sequential oracle query (Line 4). Since the total number of epochs is M = Θ(log log T ) , the overall adaptivity complexity is bounded by Θ(log log T ) , and the query complexity is bounded by O ( d log log T ) .

## A.5 Proof of Theorem 3

Let τ i,j ( t ) be the value of τ i,j at time t in the algorithm. Then τ i,j ( T ) represents the number of updates, each update requiring Oracle queries, up to T from a pair of arms i, j . For the bound of the Oracle queries up to T for each arm, we have the following lemma.

Lemma 5 (Oracle Queries Bound for Each Arm) For ( i, j ) ∈ [ d ] × [ d ] , we always have

<!-- formula-not-decoded -->

Proof. For i, j ∈ [ d ] × [ d ] , if τ i,j is not the last stage for i, j , it holds that |T i,j ( τ i,j ) | ≥ 2 τ i,j -1 . This can be derived from the update condition in the algorithm so that |T i,j ( τ i,j ) | ≥ 2 |T i,j ( τ i,j -1) | . Let τ 0 = log( Tm ) . If τ ≥ τ 0 +1 , for any i, j ∈ [ d ] × [ d ] , we have

<!-- formula-not-decoded -->

Therefore, given the fact that the total number of selected bases over T is at most mT , if τ 0 +1 ≤ τ i,j ( T ) , there is always at most 1 pair of (( i, j ) , τ ) for the fixed ( i, j ) ∈ [ d ] × [ d ] and for all τ ∈ [ τ 0 + 1 , τ i,j ( T )] . This implies that for ( i, j ) ∈ [ d ] × [ d ] s.t. τ i,j ( T ) ≥ τ 0 + 1 , we have τ i,j ( T ) ≤ τ 0 +2 , which concludes the proof.

From the above lemma, we can show that the oracle adaptivity complexity and query complexity are bounded by

<!-- formula-not-decoded -->

Let G t = ∑ t -1 s =1 D a s Σ D a s + D Σ N t + I . For the regret bound, we first provide lemmas to define a favorable event of concentration bounds.

Lemma 6 (Proposition 1 in [32]) Let t ≥ d ( d + 1) log 3 ( T ) / 2 . With probability at least 1 -1 / ( t log( t )) 2 , for all a ∈ A ,

<!-- formula-not-decoded -->

From lemma 6, we define event E t, 1 = {|⟨ a, ˆ µ s -µ ⟩| ≤ f s ∥ D -1 n s a ∥ Z s ∀ s ∈ [ ⌈ d ( d + 1) log 3 ( T ) / 2 ⌉ , t ] ∀ a ∈ A} , which holds with probability of at least 1 -1 /t log 2 ( t ) .

Lemma 7 (Proposition 5 in [32]) Let t ≥ d ( d + 1) log 3 ( T ) / 2 . With probability at least 1 -1 / ( t log( t )) 2 , for all ( i, j ) ∈ [ d ] × [ d ] , we have

<!-- formula-not-decoded -->

From the above lemma, we define event E t, 2 = {| ˆ Σ s, ( i,j ) -Σ i,j | ≤ 1 4 ( 5 h s √ n s, ( i,j ) + h 2 s n s, ( i,j ) + 1 n 2 s, ( i,j ) ) , ∀ s ∈ [ ⌈ d ( d + 1) log 3 ( T ) / 2 ⌉ , t ] ∀ ( i, j ) ∈ [ d ] 2 } , which holds with probability of at least 1 -1 /t log 2 ( t ) .

Recall Σ t, ( i,j ) = ˆ Σ t, ( i,j ) + 1 4 ( 5 h t √ n t, ( i,j ) + h 2 t n t, ( i,j ) + 1 n 2 t, ( i,j ) ) . Under E t, 2 , we have Σ t, ( i,j ) ≥ Σ i,j for all ( i, j ) ∈ [ d ] 2 . This implies that G t ⪰ G t ( ⪰ 0) so that ∥ D -1 n t a ∥ G t ≤ ∥ D -1 n t a ∥ G t . Therefore, under E t := E t, 1 ∩ E t, 2 , we have |⟨ a, ˆ µ t -µ ⟩| ≤ f t ∥ D -1 n t a ∥ G t , which implies r UCB t ( a ) ≥ ¯ r ( a ) .

Under E t , we can show that

<!-- formula-not-decoded -->

where the first inequality is obtained from ˆ Σ t, ( i,j ) ≤ Σ i,j + 1 4 ( 5 h t √ n t, ( i,j ) + h 2 t n t, ( i,j ) + 1 n 2 t, ( i,j ) ) under E t, 2 and the second inequality is obtained from n t, ( i,j ) ≤ n t,j .

For ease of presentation, we define auxiliary variables n ′ t, ( i,j ) , ˆ µ ′ t, ( i ) , G ′ t , and f ′ t for each i ∈ [ d ] and time step t ∈ [ T ] as follows: If the selected action at time t is newly updated in the algorithm ( Update = True ), then

<!-- formula-not-decoded -->

Otherwise, if the previously selected action is maintained ( Update = False ), then

<!-- formula-not-decoded -->

Using these adjusted statistics, we define the UCB-based optimistic reward estimate for any action a as

<!-- formula-not-decoded -->

Now we provide a bound for the regret as follows:

<!-- formula-not-decoded -->

For bounding the last term, we have

<!-- formula-not-decoded -->

where the first inequality is obtained from n ′ t,i ≥ |T i ( τ -1) | , the third inequality is obtained from the condition of updates in the algorithm, and the last inequality is obtained from Lemma 5, |T i,j ( τ -1) | ≥ log 3 ( T ) from warm-up stage in the algorithm, and large enough T . This concludes the proof with the fact that f T = O (log( T )) when T is large enough.

Oracle Complexity Bounds. Based on Lemma 5, we can show that the oracle query complexity is bounded by O ( d 2 log( Tm )) . Since the epochs for each base arm are updated separately, the adaptivity complexity is also bounded by O ( d 2 log( Tm )) .

## A.6 Proof of Theorem 4

From lemma 6, we define event E 1 = {|⟨ a, ˆ µ τ -µ ⟩| ≤ f T ∥ D -1 n τ a ∥ Z τ ∀ a ∈ A , ∀ τ ∈ [ M ] } , which holds with probability of at least 1 -1 / ( T log 2 ( T )) . From Lemma 7, we define event E 2 = {| ˆ Σ τ, ( i,j ) -Σ i,j | ≤ 1 4 ( 5 h T √ n τ, ( i,j ) + h 2 T n τ, ( i,j ) + 1 n 2 τ, ( i,j ) ) , ∀ ( i, j ) ∈ [ d ] 2 , ∀ τ ∈ [ M ] } , which holds with probability of at least 1 -1 / ( T log 2 ( T )) .

Under E 2 , we have Σ τ, ( i,j ) ≥ Σ i,j for all ( i, j ) ∈ [ d ] 2 . This implies that G τ ⪰ G τ ( ⪰ 0) so that ∥ D -1 n τ a ∥ G τ ≤ ∥ D -1 n τ a ∥ G τ . Therefore, under E := E 1 ∩ E 2 , we have |⟨ a, ˆ µ τ -µ ⟩| ≤ f τ ∥ D -1 n τ a ∥ G τ , which implies

<!-- formula-not-decoded -->

Similar to Lemma 4, we then have the following lemma.

Lemma 8 Under E , we can show that for all τ ∈ [ M ] , a ∗ ∈ A τ .

Proof. This can be shown by induction. Suppose a ∗ ∈ A τ for fixed τ ∈ [ M -1] . Under E , we have ¯ r ( a ) ≤ r UCB τ +1 ( a ) and ¯ r ( a ) ≥ r LCB τ +1 ( a ) for any a ∈ A τ . Then for any fixed i ∈ a ∗ , for any a ∈ A τ , we can show that

<!-- formula-not-decoded -->

where the first inequality is obtained from the definition of a ( i ) τ +1 with a ∗ ∈ A τ . This implies that

<!-- formula-not-decoded -->

which implies that i ∈ a ∗ is not eliminated from the elimination condition at the τ +1 -th epoch. This holds for all i ∈ a ∗ so that a ∗ ∈ A ′ τ +1 .

Then for any fixed i ∈ a ∗ and j ∈ a ∗ / { i } , for any a ∈ A ′ τ +1 , we can show that

<!-- formula-not-decoded -->

where the first inequality is obtained from the definition of a ( i,j ) τ +1 with a ∗ ∈ A ′ τ +1 . This implies that

<!-- formula-not-decoded -->

which implies that i ∈ a ∗ and j ∈ a ∗ / { i } are not eliminated from the elimination condition at the τ +1 -th epoch. This holds for all i ∈ a ∗ and j ∈ a ∗ / { i } so that a ∗ ∈ A τ +1 . With a ∗ ∈ A 0 = A , we can conclude the induction.

Under E , we can show that

<!-- formula-not-decoded -->

where the first inequality is obtained from ˆ Σ τ, ( i,j ) ≤ Σ i,j + 1 4 ( 5 h T √ n τ, ( i,j ) + h 2 T n τ, ( i,j ) + 1 n 2 τ, ( i,j ) ) under E 2 and the second inequality is obtained from n τ, ( i,j ) ≤ n τ, ( j,j ) .

Then, under E , for t ∈ T (1) τ ∪ T (2) τ we have

<!-- formula-not-decoded -->

̸

where the first inequality is obtained from E , the second inequality comes from the fact that a ∗ ∈ A τ -1 from Lemma 8 and elimination conditions from the algorithm, and the third inequality is obtained from n τ, ( i,i ) ≥ ∑ t ∈T (1) τ -1 ✶ ( i ∈ a t ) ≳ ( T τ -1 -( dm 2 T τ -1 log( T )) 2 / 3 ) /d and n τ, ( i,j ) ≥ ∑ t ∈T (2) τ -1 ✶ ( i ∈ a t ) ✶ ( j ∈ a t ) ≳ ( dm 2 T τ -1 log( T )) 2 / 3 /d 2 for i = j from the exploration in the algorithm at the τ -1 -th epoch, and the last inequality is obtained from large enough T .

Finally, we can show that

<!-- formula-not-decoded -->

which concludes the proof with the fact that f T = O (log( T )) when T is large enough.

Oracle Complexity Bounds. Based on the oracle calls in Lines 15,16, 17, and 19 of Algorithm 4, we observe that each epoch involves at most d + d 2 independent oracle queries (Lines 15,16) and two sequential oracle queries (Lines 17,19). Since the total number of epochs is M = Θ(log log T ) , the overall adaptivity complexity is bounded by Θ(log log T ) , and the query complexity is bounded by O ( d 2 log log T ) .

## A.7 Rare Oracle Queries for General-Reward CMAB

## A.7.1 Adaptive Rare Oracle Queries for General Reward CMAB

We first propose an algorithm (Algorithm 6) for rare oracle queries for general reward CMAB using the adaptive framework as in Algorithm 1. For i ∈ [ d ] , let ˆ F τ,i ( x ) be the fraction of the observed feedback from arm i that is no longer than 0 ≤ x ≤ 1 before time t . By inspired by [4], for i ∈ [ d ] , we define D t,i to be the distribution whose CDF is, for some constant C &gt; 0 ,

<!-- formula-not-decoded -->

Then we construct UCB for each action based on D t as follows:

<!-- formula-not-decoded -->

## A.7.2 Scheduled Rare Oracle Queries for General Reward CMAB

For i ∈ [ d ] , let ˆ F τ,i ( x ) be the fraction of the observed feedback from arm i that is no longer than 0 ≤ x ≤ 1 before epoch τ . Then, for some constant C &gt; 0 , we define D τ,i to be the distribution whose CDF is

<!-- formula-not-decoded -->

and define D τ,i to be the distribution whose CDF is

<!-- formula-not-decoded -->

```
Algorithm 6 Adaptive Rare Oracle Queries for General-Reward CMAB ( AROQ-GR-CMAB ) Init: τ i = 1 for all i ∈ [ d ] for t = 1 , 2 ..., T do for i ∈ [ d ] s.t. |T i ( τ i ) | ≥ 1 + √ Tm · |T i ( τ i -1) | /d do τ i ← τ i +1 , T i ( τ i ) ←∅ Update ← True if Update = True then a t ← arg max a ∈A r UCB t ( a ) with (6) // Oracle Query Update ← False else a t ← a t -1 Play a t and observe feedback y t,i for i ∈ a t T i ( τ i ) ←T i ( τ i ) ∪ { t } for all i ∈ a t
```

We construct UCB and LCB for each action based on D τ and D τ , respectively, as follows:

<!-- formula-not-decoded -->

Let grid T = { t 1 (= 1) , . . . , t M (= T ) } , where t τ = η √ t τ -1 and η = T 1 2 -2 1 -M for M &gt; 0 . We set M = Θ(log log( T )) .

```
Input: T 30 for τ = 1 , 2 , . . . , M do 31 Update ˆ F τ ( x ) 32 a ( i ) τ := argmax a ∈A τ -1 : i ∈ a r UCB τ ( a ) for all i ∈ N τ -1 with (7) // Oracle Queries 33 N τ ←{ i ∈ N τ -1 | r UCB τ ( a ( i ) τ ) ≥ max a ∈A τ -1 r LCB τ ( a ) } with (7) // Oracle Query 34 A τ ←{ a ∈ A τ -1 | a i = 0 for all i ∈ [ d ] / N τ } 35 T τ ← [ t τ , t τ +1 -1] 36 for t ∈ T τ do 37 i ← ( ( t -1) mod |N τ | +1 ) -th element in N τ 38 Play a t = a ( i ) τ and observe feedback y t,i for i ∈ a t
```

Algorithm 7 Scheduled Rare Oracle Queries for General-Reward CMAB ( SROQ-GR-CMAB )

## A.8 Proof of Theorem 5

Let τ i ( t ) be the value of τ i at time t in the algorithm. Then τ i ( T ) represents the number of updates, each update requiring Oracle queries, up to T from arm i . For a slight abuse of notation, we use T i ( τ i ) for the set T i ( τ i ) in the algorithm at the last time step T . For the bound of the total Oracle queries up to T , from Lemma 1, we have

<!-- formula-not-decoded -->

For ease of presentation, we use ¯ r D ( a ) = E X ∼D [ r ( a, X )] .

Lemma 9 (Lemma 3 in [4]) Let P = P 1 ×··· × P d and P ′ = P ′ 1 ×··· × P ′ d be two probability distributions D and D ′ , respectively, over [0 , 1] d . Let F i and F ′ i be the CDFs of P i and P ′ i , respectively for i ∈ [ d ] . Suppose each P i is a discrete distribution with finite support.

(a) If we have F ′ i ( x ) ≤ F i ( x ) for any i ∈ [ d ] , x ∈ [0 , 1] , then for any a ∈ A , we have ¯ r D ′ ( a ) ≥ ¯ r D ( a ) .

- (b) If we have F i ( x ) -F ′ i ( x ) ≤ z i with z i &gt; 0 for any i ∈ [ d ] , x ∈ [0 , 1] , then for any a ∈ A , we have ¯ r D ′ ( a ) -¯ r D ( a ) ≤ 2 L ∑ i ∈ a z i .

Lemma 10 (Dvoretzky-Kiefer-Wolfowitz inequality) For i.i.d. samples of X 1 , . . . , X n drawn from a distribution D , let empirical CDF ˆ F n ( x ) = 1 n ∑ n i =1 ✶ ( X i ≤ x ) . Then, for any ϵ &gt; 0 and any n ∈ N , we have

<!-- formula-not-decoded -->

From the above lemma, we define favorable event E t = { sup x ∈ [0 , 1] | ˆ F i,n s,i ( x ) -F i ( x ) | ≤ √ 3 ln s 2 n s,i ∀ s ∈ [1 , t ] ∀ i ∈ [ d ] } , which holds with probability at least 1 -O ( d/t 2 ) . Recall that r UCB t ( a ) = E y ∼D t [¯ r ( y, a )] .

For ease of presentation, we define auxiliary variables D ′ t and n ′ t,i for i ∈ [ d ] and time step t ∈ [ T ] as follows: If the selected action at time t is newly updated in the algorithm ( Update = True ), then D ′ t = D t and n ′ t,i = n t,i . Otherwise, if the previously selected action is maintained ( Update = False ), then D ′ t = D ′ t -1 and n ′ t,i = n ′ t -1 ,i . Using these adjusted statistics, we define the UCB-based optimistic reward estimate for any action a as

<!-- formula-not-decoded -->

Now we provide a bound for the regret as follows:

<!-- formula-not-decoded -->

where the second inequality is obtained from (a) in Lemma 9 and the last inequality comes from E t and (b) in Lemma 9. For bounding the last term, we have

<!-- formula-not-decoded -->

where the first inequality is obtained from n ′ t,i ≥ |T i ( τ -1) | for t ∈ T i ( τ ) , the second equality is obtained from the condition of updates in the algorithm, and the last inequality is obtained from (8).

Oracle Complexity Bounds. Based on (8), we can show that the oracle query complexity is bounded by O ( d log log( Tm/d )) . Since the epochs for each base arm are updated separately, the adaptivity complexity is also bounded by O ( d log log( Tm/d )) .

## A.9 Proof of Theorem 6

From Lemma 10, we define the event E = { sup x ∈ [0 , 1] | ˆ F τ,i ( x ) -F i ( x ) | ≤ √ 3 ln T 2 n τ,i ∀ τ ∈ [ M ] , ∀ i ∈ [ d ] } . Then, similar to Lemma 4, under the event E , we show that A τ , activated arm set for τ -th epoch, always contains the optimal arm a ∗ .

Lemma 11 Under E , we can show that for all τ ∈ [ M ] , a ∗ ∈ A τ .

Proof. The proof is the same as that of Lemma 4. This can be shown by induction. Suppose a ∗ ∈ A τ for fixed τ ∈ [ M ] . Under E , we have ¯ r ( a ) ≤ r UCB τ +1 ( a ) and ¯ r ( a ) ≥ r LCB τ +1 ( a ) for any a ∈ A τ . Then for any fixed i ∈ a ∗ , for any a ∈ A τ , we can show that

<!-- formula-not-decoded -->

where the first inequality is obtained from the definition of a ( i ) τ +1 with a ∗ ∈ A τ . This implies that

<!-- formula-not-decoded -->

which implies that i ∈ a ∗ is not eliminated from the elimination condition at the τ +1 -th epoch. This holds for all i ∈ a ∗ so that a ∗ ∈ A τ +1 . With a ∗ ∈ A 0 = A , we can conclude the induction.

Under E , we have

<!-- formula-not-decoded -->

where the first inequality is obtained from E , the second inequality comes from the fact that a ∗ ∈ A τ -1 from Lemma 11, and max a ∈A τ -1 r LCB τ ( a ) ≤ r UCB τ ( a ( i ) τ ) from the algorithm, and the third inequality is obtained from n τ,i ≥ ∑ t ∈T τ -1 ✶ ( i ∈ a t ) ≳ |T τ -1 | /d from the exploration in the algorithm at the τ -1 -th epoch. Finally, we can show that

<!-- formula-not-decoded -->

which concludes the proof.

Oracle Complexity Bounds. Based on the oracle calls in Lines 32 and 33 of Algorithm 7, we observe that each epoch involves at most d independent oracle queries (Line 32) and one sequential oracle query (Line 33). Since the total number of epochs is M = Θ(log log T ) , the overall adaptivity complexity is bounded by Θ(log log T ) , and the query complexity is bounded by O ( d log log T ) .

## A.10 Extension to Continuous Distributions for General Reward Functions

We now consider the setting where each y t,i ∈ [0 , 1] is drawn from a continuous distribution D . In this case, we impose the additional Lipschitz-continuity assumption on the reward function as follows.

Assumption 2 There exists C &gt; 0 such that for any a ∈ A and any y, y ′ ∈ [0 , 1] m , we have | r ( a, y ) -r ( a, y ′ ) | ≤ C ∑ i ∈ a | y i -y ′ i | .

Here we provide a regret bound for the discretization of Algorithm 6, and that of Algorithm 7 is omitted due to its redundancy.

Theorem 8 Algorithm 8 with Algorithm 6 achieves a regret bound of R ( T ) = ˜ O ( L √ mdT )

## Algorithm 8 Discretizations [4]

Set the number of intervals s ←⌈ C √ mT ⌉

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Invoke Algorithm 6 or Algorithm 7 for T

for i ∈ a t do

Upon observing an outcome y t,i ∈ [0 , 1] , identify j ∈ [ s ] such that y t,i ∈ I j

Treat the observation as j s

Proof. We define ˜ D to be the discretized distribution of D . For ease of presentation, we use ¯ r D ( a ) = E X ∼D [ r ( a, X )] .

Lemma 12 (Lemma 7 in [4]) For any a ∈ A , we have | ¯ r D ( a ) -¯ r ˜ D ( a ) | ≤ √ m T .

Then, from the above lemma, we have

<!-- formula-not-decoded -->

For the regret bound of the first term in the above, we treat ˜ D as the true distribution. By following the proof steps in Theorem 5, we provide a bound for the regret as follows:

<!-- formula-not-decoded -->

rounds with the following modification:

where the second inequality is obtained from (a) in Lemma 9 and the last inequality comes from E t and (b) in Lemma 9. For bounding the last term, we have

<!-- formula-not-decoded -->

where the first inequality is obtained from n ′ t,i ≥ |T i ( τ -1) | for t ∈ T i ( τ ) , the second equality is obtained from the condition of updates in the algorithm, and the last inequality is obtained from (8). This concludes the proof with (9).

## A.11 Additional Experiments

Here, we present additional experimental results for the covariance-adaptive algorithms and for the setting with a general reward function. The following results confirm that, consistent with the observations in the worst-case linear reward setting (Section 6), our algorithms achieve significantly reduced oracle usage and improved computational efficiency, while maintaining tight regret performance.

## A.11.1 Covariance-adaptive

Here, we present experiments (Figure 3) on covariance-adaptive frameworks under linear reward settings. The mean reward of each base arm is independently sampled from a uniform distribution over [0 , 1] , with d = 10 base arms and cardinality constraint m = 3 . The reward noise is correlated according to a d × d positive semi-definite covariance matrix Σ , constructed as AA ⊤ + I d with normalization, where A ∈ R d × d is a randomly generated matrix. The stochastic rewards are then sampled from a multivariate Gaussian distribution with the specified mean vector and covariance matrix Σ .

## A.11.2 General Reward

Next, we present experiments (Figure 4) on general (non-linear) reward settings with d = 5 and m = 2 . For each arm i ∈ [ d ] , the reward is sampled from a discrete distribution supported on the finite set { 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 , 1 } . The probability distribution for each arm is generated as follows: one value is randomly assigned to each arm. Then the value is assigned a large probability mass of 0 . 99 , while the remaining values share the remaining 0 . 01 probability mass equally. The reward for an action is defined as the square root of the sum of the sampled rewards from the selected arms.

Figure 3: Experimental results for covariance-adaptivity under linear reward with d = 10 and m = 3 : (a) cumulative oracle adaptivity complexity, (b) cumulative oracle query complexity, (c) regret, (d) runtime, (e) overall oracle adaptivity complexity, and (f) overall oracle query complexity of algorithms.

<!-- image -->

Figure 4: Experimental results for non-linear reward with d = 5 and m = 2 : (a) cumulative oracle adaptivity complexity, (b) cumulative oracle query complexity, (c) regret, (d) runtime, (e) overall oracle adaptivity complexity, and (f) overall oracle query complexity of algorithms.

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We explain our contributions and scope in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We compare our algorithms to the previous methods. Please see Table 1, lines 141 ∼ 145 , 217 ∼ 223 , and 262 ∼ 265 .

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

Justification: All assumptions are clearly stated, and complete proofs are provided in the main text and appendix.

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

Justification: The paper includes details to reproduce the experimental results.

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

Justification: The paper provides a link to the source code in the experimental section.

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

Justification: The paper provides details to understand and reproduce the experimental results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The experimental results include appropriate error bars to indicate statistical variation across repeated trials.

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

Answer: [No]

Justification: The experiments are simple and do not require significant computational resources.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research does not involve human subjects, sensitive data, or applications with foreseeable negative societal impact.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Please see Section 7.

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

Justification: This work is primarily theoretical.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This work is primarily theoretical.

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

Justification: This work is primarily theoretical.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This work is primarily theoretical.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This work is primarily theoretical.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA]

Justification: LLMs were not used in the development of core methods; they were used only for improving writing.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.