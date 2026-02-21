## Optimal Single-Policy Sample Complexity and Transient Coverage for Average-Reward Offline RL

## Matthew Zurek

Department of Computer Sciences University of Wisconsin-Madison matthew.zurek@wisc.edu

## Guy Zamir

Department of Computer Sciences University of Wisconsin-Madison gzamir@wisc.edu

## Yudong Chen

Department of Computer Sciences University of Wisconsin-Madison yudongchen@cs.wisc.edu

## Abstract

We study offline reinforcement learning in average-reward MDPs, which presents increased challenges from the perspectives of distribution shift and non-uniform coverage, and has been relatively underexamined from a theoretical perspective. While previous work obtains performance guarantees under single-policy data coverage assumptions, such guarantees utilize additional complexity measures which are uniform over all policies, such as the uniform mixing time. We develop sharp guarantees depending only on the target policy, specifically the bias span and a novel policy hitting radius, yielding the first fully single-policy sample complexity bound for average-reward offline RL. We are also the first to handle general weakly communicating MDPs, contrasting restrictive structural assumptions made in prior work. To achieve this, we introduce an algorithm based on pessimistic discounted value iteration enhanced by a novel quantile clipping technique, which enables the use of a sharper empirical-span-based penalty function. Our algorithm also does not require any prior parameter knowledge for its implementation. Remarkably, we show via hard examples that learning under our conditions requires coverage assumptions beyond the stationary distribution of the target policy, distinguishing single-policy complexity measures from previously examined cases. We also develop lower bounds nearly matching our main result.

## 1 Introduction

Reinforcement learning (RL) has achieved impressive results for many control problems where it is possible to collect large amounts of experience through online interaction with the environment. However, many real-world application areas where we would like to apply RL methods, such as robotics, education, or healthcare, there may not exist simulators and data collection can be expensive or dangerous. Offline RL is a subfield of RL which seeks to address these issues by learning from historical data without online interaction, and hence achieving the maximum possible statistical efficiency is the paramount concern. The lack of online experience collection poses many related challenges to offline RL methods. One issue, often termed distribution shift , is that improving a policy's performance will inherently change the distribution of states and actions it experiences, potentially moving it away from the distribution of the historical dataset. Another closely related issue, sometimes referred to as non-uniform coverage , is that our dataset may generally be unevenly

concentrated so that it is impossible to estimate the performance of all policies to uniform accuracy, and instead we must balance exploitation with varying degrees of confidence.

Recent research has made significant progress on the theoretical limits of offline RL by addressing these issues. However, many of these advances have been confined to the finite horizon setting, or the discounted infinite horizon setting, which can also behave like a finite horizon due to the irrelevance of distant future rewards. In this paper we focus on the challenging average-reward setting where the goal is to maximize the long-term average of rewards, which has been underexplored from a theoretical perspective. We briefly argue that the two aforementioned difficulties are amplified in the average-reward setting, and have not been satisfactorily addressed by previous work. First, since the average-reward objective captures performance in the long-horizon limit, we must contend with distribution shifts that occur after arbitrarily long time scales. Secondly, the issue of nonuniform coverage is magnified because while the (effective) horizon can serve as an extrinsic upper bound on the complexity of a particular policy, in the average-reward setting different policies can have arbitrarily different intrinsic complexities (as measured by parameters such as the span of the policy's relative value function). Existing work has developed algorithms which succeed under single-policy data coverage assumptions/concentrability coefficients, but has only done so when also using parameters that upper bound the complexity of all policies. Such large uniform-policy complexity measures can lead to vacuous bounds and overall fail to fully address both of the above issues. Additionally, algorithms from prior work fail to obtain optimal statistical efficiency and require foreknowledge of unlearnable parameters (such as coverage coefficients or environmental complexity parameters) for their implementation.

## 1.1 Our contributions

We address all of these challenges, developing an algorithm for (single-policy coverage) offline average-reward RL which is the first to handle the weakly communicating setting where not all policies have constant gains, as well as the first to obtain a convergence rate dependent on the bias span of only the target policy (as opposed to uniform complexity measures). Informally, our main theorem provides a high-probability guarantee on the suboptimality of the output policy ̂ π of the form

<!-- formula-not-decoded -->

where ∥ h π ⋆ ∥ span is the bias-span of the target policy π ⋆ and S is the number of states. This holds whenever the sample size n ( s, a ) per state-action pair ( s, a ) satisfies n ( s, π ⋆ ( s )) ≥ mµ π ⋆ ( s ) + ˜ O ( T hit ( P, π ⋆ ) 2 ) for all states s . Here µ π ⋆ is the stationary distribution of the target policy, m is the 'effective dataset size,' and T hit ( P, π ⋆ ) is a novel policy hitting radius that measures the time for π ⋆ to reach a particular state in the support of its stationary distribution, and is thus also a single-policy complexity measure.

Interestingly, this condition requires data even for state-action pairs ( s, π ⋆ ( s )) for which s is transient ( µ π ⋆ ( s ) = 0 ) under the target policy, and we show via a hard example that this requirement is nearly unimprovable. In particular, this implies two surprising findings: i) with a fully 'single-policy' sample complexity, learning a near-optimal policy is impossible under coverage conditions with respect to only the stationary distribution of the target policy, even with arbitrarily large amounts of data; ii) on the other hand, only a bounded amount of data from the transient state-action pairs of the target policy is sufficient to achieve vanishing suboptimality. We also show another lower bound which implies the optimality of the guarantee (1) in terms of its dependence on m , making our result the first among offline average-reward RL approaches to achieve an optimal rate for large m .

Our algorithm is based upon a pessimistic discounted value iteration procedure, involving a very large and prior-knowledge-free choice of discount factor. Most notably we develop a quantile clipping technique which enables the use of a sharper empirical-span-based penalty function.

## 1.2 Related work

First we discuss prior work on average-reward offline RL. To the best of our knowledge the only works with explicit results for this setting are Ozdaglar et al. [2024] and Gabbianelli et al. [2023]. Ozdaglar et al. [2024] assume that the MDP is unichain, and obtain guarantees with a constrained linear

programming (LP) algorithm in terms of the uniform mixing time τ unif (defined in Section 2), for both general function approximation and tabular settings. We also discuss quantitative comparisons to the tabular results from Ozdaglar et al. [2024] after presenting our main theorem. Gabbianelli et al. [2023] assume that all policies in the MDP have constant (state-independent) gain, which is more general than unichain MDPs but does not hold in weakly communicating MDPs. Gabbianelli et al. [2023] consider the linear MDP setting, develop an algorithm based on primal-dual methods for solving LPs, and obtain guarantees in terms of a uniform bound on the span of all policies H unif . The algorithms in both of these works require knowledge of certain concentrability coefficients.

Next we briefly discuss related work for offline RL outside of the average-reward setting. Our algorithm is essentially a careful refinement of the pessimistic value iteration approach of Li et al. [2023] for the discounted tabular setting, which in turn is a refinement of Rashidinejad et al. [2022]. Many works (e.g., Liu et al. [2020], Jin et al. [2021], Xie et al. [2021], Uehara and Sun [2021], Rashidinejad et al. [2022]) have demonstrated the ability for pessimistic approaches to address the distribution shift/non-uniform coverage challenges of offline RL and achieve near-optimal performance under single-policy concentrability assumptions.

Finally we discuss prior work on average-reward RL under uniform coverage assumptions. Many papers on average-reward RL considering the tabular generative model setting [Kearns and Singh, 1998] actually only require a dataset with an equal number of samples from all state-action pairs (e.g., Wang et al. [2022, 2023], Zurek and Chen [2024, 2025a,b]), and hence we believe such papers could be easily extended to the uniform coverage setting, obtaining a guarantee dependent on the smallest number of samples for any state-action pair. While such works might be considered offline RL, we reserve this term for guarantees involving only single-policy coverage assumptions. Achieving instance-dependent guarantees in terms of the bias span of an optimal policy (e.g., Zhang and Xie [2023], Wang et al. [2022], Zurek and Chen [2025b]) and removing the need for prior knowledge of complexity parameters (e.g., Jin et al. [2024], Neu and Okolo [2024], Tuynman et al. [2024], Zurek and Chen [2025a]) have been the objectives of extensive research in the uniform coverage setting.

## 2 Background and problem setup

## 2.1 Background

A Markov decision process (MDP) is a tuple ( S , A , P, r ) where S and A respectively denote the finite state and action spaces, P : S × A → ∆( S ) is the transition kernel (with ∆( S ) denoting the probability simplex on S ), and r : [0 , 1] S×A is the reward function. We let S = |S| and A = |A| . We generally omit the explicit reference to S and A when defining MDPs. A (Markovian/stationary) policy is a mapping π : S → ∆( A ) . We call a policy deterministic if for all s ∈ S , π ( s ) only places probability mass on one action, and in this case we also treat π as a mapping S → A . Let Π denote the set of all stationary deterministic policies. An initial state s 0 ∈ S and policy π induce a distribution over trajectories ( s 0 , A 0 , S 1 , A 1 , . . . ) where A t ∼ π ( S t ) , S t +1 ∼ P ( · | S t , A t ) , and we let E π s 0 denote the expectation with respect to this distribution. We often treat P as an ( S × A ) -byS matrix where P sa,s ′ = P ( s ′ | s, a ) , and let P sa denote the sa -th row of this matrix (treated as a 'row vector', so P sa X = ∑ s ′ P sa ( s ′ ) X ( s ′ ) for X ∈ R S ). For X ∈ R S and s ∈ S , a ∈ A , define the next-state value variance V P sa [ X ] = ∑ s ′ ∈S P ( s ′ | s, a ) X ( s ′ ) 2 -( ∑ s ′ ∈S P ( s ′ | s, a ) X ( s ′ )) 2 .

A discounted MDP is a tuple ( S , A , P, r, γ ) where γ ∈ [0 , 1) is the discount factor. For a policy π , the discounted value function V π γ ∈ [0 , 1 1 -γ ] S is defined V π γ ( s ) = E π s [ ∑ ∞ t =0 γ t R t ] where R t = r ( S t , A t ) , and the gain ρ π ∈ [0 , 1] S , is ρ π ( s ) = C-lim t →∞ E π s [ R t ] = lim T →∞ 1 T E π s [ ∑ T -1 t =0 R t ] where C-lim is the Cesaro limit. We define the optimal gain ρ ⋆ = sup π ∈ Π ρ π , and we say a policy π is gain-optimal if ρ π = ρ ⋆ . A gain-optimal policy always exists [Puterman, 1994]. The bias function of a policy π , h π ∈ R S , is h π ( s ) = C-lim T →∞ E π s [ ∑ T -1 t =0 ( R t -ρ π ( S t ))] .

M : R S×A → R S denotes the action maximization operator where M ( Q )( s ) = max a ∈A Q ( s, a ) , and M π denotes the policy matrix where M π ( Q )( s ) = ∑ a ∈A π ( s )( a ) Q ( s, a ) , for any Q ∈ R S×A , s ∈ S , and policy π . We often drop the parenthesis and write MQ := M ( Q ) . For any Q ∈ R S×A , the discounted (action-value) Bellman operator T : R S×A → R S×A is T ( Q ) := r + γPM ( Q ) , and the policy-evaluation Bellman operator T π is T π ( Q ) := r + γPM π Q , for any policy π .

Let N = { 1 , 2 , . . . } denote the set of natural numbers. Define 0 , 1 as the all-zero and all-one vectors, respectively. For X ∈ R S , let ∥ X ∥ span = max s ∈S X ( s ) -min s ∈S X ( s ) denote the span semi-norm. We use ˜ O ( · ) , ˜ Θ( · ) , ˜ Ω( · ) notation to ignore constants as well as logarithmic factors in S, A, 1 1 -γ , 1 δ , and n tot , where δ and n tot are the failure probability and the total dataset size, to be defined below. Let e s ∈ R S denote the vector which is all zero except for a 1 in entry s ∈ S . For two vectors v, v ′ ∈ R d , v ≥ v ′ denotes the elementwise inequality v ( i ) ≥ v ′ ( i ) for all i .

Under the transition kernel P , a policy π induces a Markov chain over state S , whose transition matrix is denoted by P π . The policy π is said to be unichain if it induces a unichain Markov chain, meaning that the chain consists of a single (irreducible) recurrent class plus a possibly empty set of transient states. An MDP is unichain if all deterministic policies in the MDP are unichain. An MDP is communicating (aka strongly connected) if for any pair of states s, s ′ ∈ S , s ′ is accessible from s , meaning there exists some policy π and some k ∈ N such that E π s I ( S k = s ′ ) &gt; 0 . An MDP is weakly communicating if it consists of a set of states S c such that, for any s, s ′ ∈ S c , s ′ is accessible from s , plus a set of states S t = S \ S c which are transient under all policies. All unichain and communicating MDPs are weakly communicating.

Aunichain policy π has constant (state-independent) ρ π , and thus in unichain MDPs, all policies have constant gains. In weakly communicating MDPs, the optimal gain ρ ⋆ is constant, but sub-optimal policies π may have non-constant ρ π . For any unichain policy π , we write its (unique) stationary distribution as µ π ∈ R S (which we treat as a 'row vector'). For any unichain policy π , we define its mixing time τ ( π ) = inf { t ≥ 0 : ∥ ∥ e ⊤ s P t π -µ π ∥ ∥ 1 ≤ 1 2 } . Define the uniform mixing time as τ unif = sup π ∈ Π τ ( π ) . Also define the uniform span bound H unif = sup π ∈ Π ∥ h π ∥ span . For any s ∈ S , let η s := inf { t ≥ 0 : S t = s } be the first hitting time of state s . Define the diameter D = max s,s ′ ∈S min π ∈ Π E π s [ η s ′ ] , and we sometimes write D P to emphasize the dependence on P .

## 2.2 Offline RL setting

We assume a sample size function n : S × A → N is fixed a priori, and for each s ∈ S , a ∈ A , we assume that we have n ( s, a ) samples S 1 s,a , . . . , S n ( s,a ) s,a sampled independently from the next-state transition distribution P ( · | s, a ) . We define the dataset D = ( ( s, a, S i s,a ) ) s ∈S ,a ∈A , 1 ≤ i ≤ n ( s,a ) and let n tot = ∑ s ∈S ,a ∈A n ( s, a ) denote the total dataset size. We assume the reward function r is known.

We introduce a new quantity which plays a key role in both our main theorem and our lower bounds. For any transition kernel matrix P and policy π , we define the policy hitting radius

<!-- formula-not-decoded -->

where again η s is the first hitting time of state s . In words, T hit ( P, π ) measures the largest expected amount of time required to hit the 'center' state s ⋆ , for the optimal choice of s ⋆ (which will always be a recurrent state). As shown in Lemma B.10, T hit ( P, π ) is always finite if P π is unichain. We also always have that ∥ h π ∥ span ≤ 4 T hit ( P, π ) for any π (Lemma B.13). There is generally no relationship between T hit ( P, π ) and τ ( π ) ; see the discussion in Appendix B.5.1.

## 3 Main results

## 3.1 Algorithm

First we describe the algorithm used to obtain our main result. We employ a discounted reduction approach, i.e., approximating the average-reward MDP by a discounted MDP with an appropriate choice of discount factor. The main component of our approach, Algorithm 1, is a pessimistic value iteration subroutine which can be understood as solving a discounted MDP.

Now we define the pessimistic Bellman operator ̂ T pe : R S×A → R S×A used in Algorithm 1. ̂ T pe is a function of γ as well as the dataset D , utilizing the empirical transition matrix ̂ P where ̂ P ( s ′ | s, a ) = 1 n ( s,a ) ∑ n ( s,a ) i =1 I ( S i sa = s ′ ) . If n ( s, a ) = 0 for some s, a then for concreteness we

## Algorithm 1 Pessimistic Value Iteration With Quantile Clipping

input: Dataset D , reward function r , discount factor γ ∈ (0 , 1) , failure probability δ ∈ (0 , 1)

- 1: Form empirical transition matrix ̂ P used in ̂ T pe from D
- 2: Let ̂ Q 0 = 0 and K = ⌈ log( 2 n tot 1 -γ ) 1 -γ
- 3: for t = 1 , . . . , K do
- 4: Let ̂ Q t = ̂ T pe ( ̂ Q t -1 )
- 5: end for
- 6: Let ̂ Q = ̂ Q K and for each s ∈ S , let ̂ π ( s ) ∈ argmax a ∈A ̂ Q ( s, a )
- 7: return ̂ π, ̂ Q

define ̂ P ( s ′ | s, a ) = 1 /S , although any default probability distribution over S would be fine, since our construction of ̂ T pe does not depend on rows ̂ P sa such that n ( s, a ) = 0 . 1

For any Q ∈ R S×A and any s ∈ S , a ∈ A , we define

<!-- formula-not-decoded -->

Here MQ ∈ R S takes the maximum over actions of the Q-function Q (and thus should be understood as the corresponding value function). The term b ( s, a, MQ ) ≥ 0 is a certain Bernstein-style penalty, which is chosen below to ensure that ̂ T pe ( Q ) lower-bounds the true (unknown) Bellman operator T ( Q ) for any Q . The expression ̂ P sa T β ( s,a ) ( ̂ P sa , MQ ) denotes the inner product of the probability distribution ̂ P sa with the vector T β ( s,a ) ( ̂ P sa , MQ ) ∈ R S , which is a 'quantile-clipped' version of MQ to be defined momentarily. For β ∈ [0 , 1] , the quantile clipping operator T β : R S × R S → R S is defined as follows: for any V ∈ R S , s ∈ S , and probability distribution µ ∈ R S , let

<!-- formula-not-decoded -->

In words, all entries of V larger than the (largest) 1 -β quantile with respect to µ are clipped down to this quantile. To extend the definition to β &gt; 1 , we set T β ( µ, V )( s ) = min s ′ ∈S V ( s ′ ) , that is all entries will be clipped to the minimum entry of V . Finally we define the penalty term

<!-- formula-not-decoded -->

where α = 8log ( 6 S 2 An tot (1 -γ ) δ ) and β ( s, a ) = α max { n ( s,a ) -1 , 1 } . Note that β ( s, a ) = ˜ O ( 1 n ( s,a ) ) (whenever n ( s, a ) &gt; 0 ).

The pessimistic Bellman operator ̂ T pe has several nice properties that are crucial to our analysis.

Lemma 3.1. ̂ T pe satisfies the following:

1. Monotonicity: If Q ≥ Q ′ then ̂ T pe ( Q ) ≥ ̂ T pe ( Q ′ ) .
2. Constant shift: For any c ∈ R , ̂ T pe ( Q + c 1 ) = ̂ T pe ( Q ) + γc 1 .
3. γ -contractivity: ̂ T pe is a γ -contraction and has a unique fixed point ̂ Q ⋆ pe ∈ [0 , 1 1 -γ ] S .

See Lemma B.1 for a more complete statement. In summary, like previous pessimistic value iteration approaches [Li et al., 2023, Rashidinejad et al., 2022], our pessimistic Bellman operator shares key properties with usual Bellman operators enabling us to find an approximate fixed point in ˜ O ( 1 1 -γ ) value iteration steps, and then we will choose policy ̂ π to be greedy with respect to this fixed point.

Now we discuss the motivation for quantile clipping, and the differences from prior work. In particular we highlight the constant shift property enjoyed by ̂ T pe . This is highly desirable for the average-reward

1 If n ( s, a ) = 0 then β ( s, a ) = α &gt; 1 and T β ( s,a ) ( ̂ P sa , MQ ) = (min s ′ ( MQ )( s ′ )) 1 , causing the max in (3) to equal min s ′ ( MQ )( s ′ ) .

- ⌉ ▷ initialization and number of iterations

setting, and more generally any weakly communicating MDPs, since in such MDPs the optimal value function behaves as V ⋆ γ ≈ 1 1 -γ ρ ⋆ + h ⋆ and ρ ⋆ is a multiple of 1 . The constant shift property essentially guarantees that we only penalize the variability in the relative value differences between states, not the overall horizon-dependent scale 1 1 -γ of the cumulative rewards. The ∥·∥ span-based second term in our penalty function definition (5) of b is essential for this constant-shift property, since the span seminorm is invariant to translation by multiples of 1 . Previous 'Bernstein-style' penalty functions [Li et al., 2023] use a larger term like β ( s, a ) 1 1 -γ ≈ 1 n ( s,a ) 1 1 -γ , which breaks the constant shift property and can dominate the first (variance-based) term in (5) when used with large horizons. Naively using β ( s, a ) ∥ V ∥ span in the second term of (5) actually fails to ensure the monotonicity and contractivity properties of ̂ T pe , for reasons that we elaborate upon in Section 4. Fortunately, the introduction of quantile clipping remedies these issues, and only introduces small additional bias: since only entries representing at most β ( s, a ) = ˜ O ( 1 n ( s,a ) ) of the probability mass with respect to ̂ P sa have their values clipped, we have ̂ P sa T β ( s,a ) ( ̂ P sa , V ) ≤ ̂ P sa V ≤ ̂ P sa T β ( s,a ) ( ̂ P sa , V ) + β ( s, a ) ∥ V ∥ span , and introducing quantile clipping within the two terms of the penalty function b in (5) only reduces the penalty value, relative to instead using V ̂ P sa [ V ] and ∥ V ∥ span. (See Lemma B.14.)

## 3.2 Main theorem

Now we present our main theorem on the performance of Algorithm 1. We will apply Algorithm 1 with a very large discount factor γ such that the effective horizon is 1 1 -γ = n tot .

Theorem 3.2. There exist absolute constants C 1 , C 2 such that the following holds: Fix δ &gt; 0 . Let γ = 1 -1 n tot and α = 8log ( 6 S 2 An tot (1 -γ ) δ ) . Let π ⋆ be a deterministic gain-optimal policy which is unichain with stationary distribution µ π ⋆ . Suppose there exists some m ∈ N such that

<!-- formula-not-decoded -->

Then letting ̂ π be the policy returned by Algorithm 1 with inputs D , r , γ = 1 -1 n tot , and δ , we have with probability at least 1 -5 δ that

<!-- formula-not-decoded -->

We prove Theorem 3.2 in Appendix B. Theorem 3.2 demonstrates that as the 'effective dataset size' m increases, the suboptimality of ̂ π decreases at a rate of ˜ O ( √ S ∥ h π ⋆ ∥ span /m ) , which matches our lower bound Theorem 3.4. Our coverage assumption is qualitatively different than previous works on average-reward RL, since even for states s which are transient under π ⋆ (and thus have µ π ⋆ ( s ) = 0 ), we still require ˜ O ( T hit ( P, π ⋆ ) 2 ) samples from the state-action pair ( s, π ⋆ ( s )) . Note that up to a log factor this transient state coverage assumption is independent of m , meaning that vanishing suboptimality is possible with only an essentially bounded amount of data from transient states. (In the absence of this additional term we could treat n tot /m as a 'concentrability coefficient' similar to prior work, but we believe our results are stated more clearly in terms of the effective dataset size m .) As shown in Theorem 3.3, this transient data requirement is necessary to obtain a ∥ h π ⋆ ∥ span-based guarantee, and our dependence on T hit ( P, π ⋆ ) is nearly optimal. Theorem 3.2 requires π ⋆ to be unichain, which is a mild assumption, since even in weakly communicating MDPs where not all policies are unichain, there always exists a unichain gain-optimal policy [Bertsekas, 2018].

No prior parameter knowledge, such as of ∥ h π ⋆ ∥ span or the value of m (or equivalently a coverage coefficient) is needed for Algorithm 1 to be implemented and enjoy the above guarantee. In particular γ is set so that the effective horizon is n tot . Actually our theorem would hold for arbitrarily larger choices of the effective horizon, and the guarantee would not degrade except for a logarithmic dependence on the effective horizon, but this would be suboptimal from a computational perspective, since ˜ O (1 / (1 -γ )) iterations are required for convergence in Algorithm 1. Also see Theorem B.20 for a version of Theorem 3.2 allowing π ⋆ to be gain-suboptimal.

In the unichain tabular setting, Ozdaglar et al. [2024] obtain a suboptimality bound like ˜ O ( √ C 2 τ 2 unif S/n tot ) where C ≥ 1 is a certain coverage coefficient roughly equivalent to n tot /m . With this substitution their bound becomes ˜ O ( √ Cτ 2 unif S/m ) , which interestingly degrades with the

coverage coefficient C even as the effective dataset size m is held constant, while our bound has no such issue. We also have ∥ h π ⋆ ∥ span ≤ O ( τ unif ) , and qualitatively ∥ h π ⋆ ∥ span is much sharper since it depends only on π ⋆ rather than all policies.

## 3.3 Lower bounds

In this subsection we present two lower bounds implying the near-optimality of our Theorem 3.2. Below, for an MDP ( P θ , r ) , ρ π θ , h π θ and µ π θ denote the gain, bias and stationary distribution of a policy π , respectively; ρ ∗ θ and D θ denote the optimal gain and the diameter of the MDP, respectively; and P θ,n denotes the distribution of the dataset D under this MDP when the sample size function is n .

First, we present the surprising fact that, to obtain convergence rates dependent on certain singlepolicy complexity measures including ∥ h π ⋆ ∥ span and T hit ( P, π ⋆ ) , coverage assumptions with respect to only the stationary distribution of the target policy are insufficient to learn a near-optimal policy, even with an arbitrarily large amount of data.

Theorem 3.3. For any T ≥ 4 and any m ∈ N , there exist a finite index set Θ , transition matrices P θ for each θ ∈ Θ , and a reward function r , such that for all δ ∈ ( 0 , 1 e 9 ] , there exists a function n : S × A → N satisfying the following:

1. For each θ ∈ Θ , the MDP ( P θ , r ) is unichain and communicating, with A ≤ O (⌈ m T ⌉) actions and diameter T .
2. For each θ ∈ Θ , the MDP ( P θ , r ) has a unique deterministic gain-optimal policy π ⋆ θ such that T hit ( P θ , π ⋆ θ ) ≤ T and n ( s, π ⋆ θ ( s )) ≥ mµ π ⋆ θ θ ( s ) + T 6 log ( 1 δ ) for all s ∈ S .
3. For any algorithm A that maps the dataset D to a stationary policy, we have

<!-- formula-not-decoded -->

Note that the 'effective dataset size' parameter m can be taken arbitrarily large, meaning that learning better than a 1 2 -suboptimal policy is impossible even with arbitrarily large amounts of data from the stationary distribution of the target policy. This does not contradict the error bounds from prior work which make stationary-distribution-based coverage assumptions and involve uniform complexity measures τ unif , H unif [Ozdaglar et al., 2024, Gabbianelli et al., 2023], since the parameters τ unif , H unif scale with m in our hard instances in such a way as to render such bounds vacuous. In contrast, the parameters ∥ h π ⋆ θ θ ∥ span , T hit ( P θ , π ⋆ θ ) , and D θ remain bounded, implying that a convergence rate involving any of these parameters is impossible without data coverage beyond the stationary distribution, revealing a qualitatively different behavior of such parameters. While oftentimes results for average-reward setups can be predicted/derived by taking appropriate largeγ limits of results for discounted settings, taking the limit as γ → 1 of usual discounted occupancy coverage assumptions (e.g., C ⋆ in Rashidinejad et al. [2022, Theorem 6]) only leads to requirements on covering the stationary distribution.

The setup in Theorem 3.3 even provides the learner with ˜ Ω( T hit ( P θ , π ⋆ θ )) samples from state-action pairs which are transient under the target policy ( µ π ⋆ θ θ ( s ) = 0) , and this is still insufficient for learning near-optimal policies. This implies that the transient state dataset coverage requirement of Theorem 3.2 is nearly unimprovable, up to an additional factor of ˜ O ( T hit ( P, π ⋆ )) . A complete proof of Theorem 3.3 is provided in Appendix C and a sketch is provided in Section 4, but we briefly summarize the key idea: even with an arbitrarily large (but finite) amount of data from the recurrent class of the target policy, we may inevitably learn a policy with a small probability of leaving these well-covered states. Without any data we cannot learn how to recover from such a transition and navigate back to highly-rewarding regions quickly enough. This unfavorable but rare transition has negligible impact for finite horizon/discounted RL objectives (if the starting state is within the highly-rewarding region). In unichain MDPs all policies are guaranteed to eventually return to the recurrent class of the optimal policy eventually (because all recurrent classes must overlap, otherwise it would be possible to construct a multichain policy), but the fact that some policies take a long time to do so means that the uniform mixing time τ unif is very large, even if the optimal policy can recover quickly. Despite being unichain, such MDPs are qualitatively close to being non-unichain (but weakly communicating).

Next, we present a lower bound which demonstrates that dependence on m in Theorem 3.2 is tight.

Theorem 3.4. There exist absolute constants c 1 , c 2 , c 3 &gt; 0 such that for any T ≥ c 1 , S ≥ c 2 , k ≥ 0 , and m ≥ max { TS,kS } , one can construct a finite index set Θ , transition matrices P θ for each θ ∈ Θ , a reward function r , and a function n : S × A → N such that the following hold:

1. For each θ ∈ Θ , the MDP ( P θ , r ) is unichain and communicating, with S states and diameter T .
2. For each θ ∈ Θ , the MDP ( P θ , r ) has a unique stationary gain-optimal policy π ⋆ θ such that T hit ( P θ , π ⋆ θ ) ≤ T and n ( s, π ⋆ θ ( s )) ≥ mµ π ⋆ θ θ ( s ) + k for all s ∈ S .
3. For any algorithm A that maps the dataset D to a stationary policy, we have

<!-- formula-not-decoded -->

Since generally T hit ( P, π ) ≥ ∥ h π ∥ span / 4 (see Lemma B.13), Theorem 3.2 implies a lower bound in terms of ∥ h π ⋆ ∥ span ( ∥ h π ⋆ θ θ ∥ span and T hit ( P θ , π ⋆ θ ) are on the same order in the instances of Theorem 3.4). We add the parameter k to demonstrate that a coverage requirement in the form of Theorem 3.2 does not affect the dependence on m in (6) for sufficiently large m . In particular after setting k = ˜ Θ( T 2 ) to match Theorem 3.2, its dependence on ∥ h π ⋆ ∥ span , S, and m matches (6) and thus is unimprovable up to ˜ O ( · ) factors as long as m ≥ ˜ Θ( T 2 S ) . Theorem 3.4 is proven in Appendix D.

## 4 Proof sketches

## 4.1 Main theorem

First we discuss the proof of Theorem 3.2, including the motivation for quantile clipping. The key idea of pessimistic value iteration is to choose ̂ T pe so that ̂ T pe ( ̂ Q ⋆ pe ) ≤ T ( ̂ Q ⋆ pe ) , and then letting ̂ π be greedy with respect to ̂ Q ⋆ pe (meaning T ( ̂ Q ⋆ pe ) = T ̂ π ( ̂ Q ⋆ pe ) ), we have

<!-- formula-not-decoded -->

so by standard monotonicity arguments we have ̂ Q ⋆ pe ≤ Q ̂ π . The challenge is then to choose ̂ T pe as 'close' to T as possible, so that ̂ Q ⋆ pe is as close as possible to Q ⋆ (while ensuring ̂ T pe ( ̂ Q ⋆ pe ) ≤ T ( ̂ Q ⋆ pe ) ), in order to maximize Q ̂ π . Using α to hide ˜ O ( · ) terms, an empirical Bernstein-like bound [Maurer and Pontil, 2009] for the quantity ̂ V ⋆ pe = M ̂ Q ⋆ pe , and upper-bounding a sum by max , yields

<!-- formula-not-decoded -->

This sharp span-based form of penalty function ˜ b is crucial for the constant shift property described in Lemma 3.1, since both V ̂ P sa [ · ] and ∥·∥ span are invariant to shifts by multiples of 1 . As discussed there this property is essential for the average-reward setting, and the Bernstein-style penalty used in Li et al. [2023] replaces the second term from the max in (7) with 1 1 -γ ≥ ∥ ̂ V ⋆ pe ∥ span and hence does not enjoy this property. However, we cannot simply use an operator like ˜ T ( Q )( s, a ) := r ( s, a ) + γ ̂ P sa ̂ V ⋆ pe -γ ˜ b ( s, a, ̂ V ⋆ pe ) , because the span term within ˜ b would lead to non-monotonicity of ˜ T and disrupt many other essential properties (like γ -contractivity). To see the non-monotonicity, suppose some s ′ has ̂ P ( s ′ | s, a ) &lt; α n ( s,a ) . Then, for V ∈ R S where V ( s ′ ) is the largest entry, ignoring non-differentiability edge cases, we have

<!-- formula-not-decoded -->

However, if we replace V with the quantile-clipped quantity T α/n ( s,a ) ( ̂ P s,a , V ) , then increasing V ( s ′ ) (when it is the largest entry of V ) will only increase T α/n ( s,a ) ( ̂ P s,a , V ) if ̂ P ( s ′ | s, a ) has at least α/n ( s, a ) probability mass. Hence, by fixing the overpenalization caused by ∥·∥ span , quantile clipping is essential to define our empirical-span-based pessimistic Bellman operator.

Now we discuss a few other aspects of the proof of Theorem 3.2. Obtaining the Bernstein-style inequality (7) is nontrivial due to statistical dependence between ̂ P sa and ̂ V ⋆ pe . We remedy this with an argument based on leave-one-out/absorbing MDP techniques [Agarwal et al., 2020], which requires additional covering steps due to the presence of quantile clipping. (See Lemmas B.6 and B.5.)

It is somewhat surprising that Theorem 3.2 is able to obtain a bias-span-based guarantee without requiring any prior bias-span knowledge, since prior work in related uniform coverage settings has shown this is impossible when the effective horizon is large/on the same order as the size of the dataset [Zurek and Chen, 2024]. This is closely related to the issue that the bias span ∥ h π ∥ span of a policy π is not estimable to multiplicative error with a sample complexity polynomial in only S, A , and ∥ h π ∥ span [Zurek and Chen, 2025b, Tuynman et al., 2024]. However, our proof suggests that ∥ h π ∥ span is estimable if we allow a dependence on the policy hitting radius T hit ( P, π ) , which we believe is an independently interesting finding. (See Lemma B.18.) This fact plays a key role in bounding the suboptimality in terms of ∥ h π ∥ span .

## 4.2 Transient lower bound

Next we briefly describe the idea behind the hard instances within Theorem 3.3, which implies that transient coverage is required for offline RL with single-policy complexity parameters. Consider the MDP P in Figure 1, which is parameterized by m , which we imagine as arbitrarily large, and T , which we imagine as measuring the complexity of P . There are two states with two actions each, an absorbing stay action and a leave action which has a small chance of leading to the other state. State 1 has reward 1 for both actions and state 2 has reward 0 for both actions, so clearly the optimal policy π ⋆ is to take leave in state 2 and take stay in state 1 , and the associated stationary distribution has all its mass on state 1 . Also, assuming m ≥ T , T hit ( P, π ⋆ ) = T , since this is the expected amount of time to hit state 1 starting from state 2 . Therefore to satisfy the coverage assumption n ( s, π ⋆ ( s )) ≥ mµ π ⋆ ( s ) + T hit ( P, π ⋆ ) , it would suffice to provide m samples for both state 1 actions, and T samples for both state 2 actions.

<!-- image -->

̂

Figure 1: An MDP P parameterized by m,T , and an empirical MDP ̂ P which has constant probability of being sampled from P . Each solid arrow indicates an action and is annotated with its reward. Arrows which split into multiple dashed arrows indicate possible stochastic transitions, and each dashed arrow is annotated with the associated probabilities.

For this sample size function n , with constant probability we will not observe any transitions to the other state from either of the leave actions (that is, the samples from each of these state-action pairs would all be of the form ( s, leave , s ) ). Under such an event, illustrated by the empirical MDP ̂ P , no algorithm could distinguish between the leave and stay actions in either state better than random guessing. If an algorithm is forced to return a deterministic policy, then there would be a constant probability of choosing the policy π where ( π (1) , π (2)) = ( leave , stay ) , which will remain in state 2 (and hence have gain 0 ). To generalize to algorithms which may choose randomized policies,

we add more copies of the stay action to state 2 , so that a 'guessed' randomized policy has a low chance of returning to state 1 quickly enough for good performance. Also P is not unichain, but we can add an arbitrarily small ( O ( m -2 ) ) probability for the stay actions in state 2 to return to state 1 , which ensures unichainedness without meaningfully changing the story. We emphasize that the hardness is not due to the inability to identify the stay action in state 1 , since in general we cannot expect to perfectly match the stationary distribution of the target policy (and in this example, the policy ( leave , leave ) still has suboptimality only O ( T/m ) ). Rather, the hardness is due to the fact that it is nontrivial to navigate (quickly) back to the target policy's stationary distribution after leaving it, and learning to do so requires data coverage beyond said stationary distribution.

## 5 Conclusion

We developed the first average-reward offline RL algorithms for MDPs where not all policies have constant gain, and also the first convergence rates depending only on the bias span of a single policy. A main limitation of our work is its focus on the tabular setting, hence an important direction is to extend these improvements to function approximation setups to avoid dependence on S in the results. While Theorem 3.3 demonstrates the necessity of data from the target policy from all states, this may be limiting in practice, so an interesting future direction is to explore additional assumptions or information that could be provided to the algorithm to circumvent this requirement.

## Acknowledgments and Disclosure of Funding

Y. Chen and M. Zurek acknowledge support by National Science Foundation grants CCF-2233152 and DMS-2023239.

## References

- Alekh Agarwal, Sham Kakade, and Lin F. Yang. Model-Based Reinforcement Learning with a Generative Model is Minimax Optimal, April 2020. URL http://arxiv.org/abs/1906.03804 . arXiv:1906.03804 [cs, math, stat] version: 3.
- Dimitri P. Bertsekas. Approximate dynamic programming . Number volume 2 in Dynamic programming and optimal control / Dimitri P. Bertsekas, Massachusetts Institute of Technology. Athena Scientific, Belmont, Massachusetts, fourth edition, updated printing edition, 2018. ISBN 978-1-886529-08-3 978-1-886529-44-1.
- David Cheikhi and Daniel Russo. On the Statistical Benefits of Temporal Difference Learning. In Proceedings of the 40th International Conference on Machine Learning , pages 4269-4293. PMLR, July 2023. URL https://proceedings.mlr.press/v202/cheikhi23a.html . ISSN: 2640-3498.
- Richard Durrett. Probability: theory and examples . Cambridge series in statistical and probabilistic mathematics. Cambridge University Press, Cambridge, fifth edition edition, 2019. ISBN 978-1108-47368-2.
- Germano Gabbianelli, Gergely Neu, Nneka Okolo, and Matteo Papini. Offline Primal-Dual Reinforcement Learning for Linear MDPs, May 2023. URL http://arxiv.org/abs/2305.12944 . arXiv:2305.12944 [cs].
- Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is Pessimism Provably Efficient for Offline RL? In Proceedings of the 38th International Conference on Machine Learning , pages 5084-5096. PMLR, July 2021. URL https://proceedings.mlr.press/v139/jin21e.html . ISSN: 2640-3498.
- Ying Jin, Ramki Gummadi, Zhengyuan Zhou, and Jose Blanchet. Feasible $Q$-Learning for Average Reward Reinforcement Learning. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics , pages 1630-1638. PMLR, April 2024. URL https: //proceedings.mlr.press/v238/jin24b.html . ISSN: 2640-3498.

- Michael Kearns and Satinder Singh. Finite-Sample Convergence Rates for Q-Learning and Indirect Algorithms. In Advances in Neural Information Processing Systems , volume 11. MIT Press, 1998. URL https://proceedings.neurips.cc/paper/1998/hash/ 99adff456950dd9629a5260c4de21858-Abstract.html .
- John G. Kemeny and J. Laurie Snell. Finite Markov chains . Undergraduate texts in mathematics. Springer-Verlag, New York, 1976. ISBN 978-0-387-90192-3.
- Gen Li, Laixi Shi, Yuxin Chen, Yuejie Chi, and Yuting Wei. Settling the Sample Complexity of Model-Based Offline Reinforcement Learning, February 2023. URL http://arxiv.org/abs/ 2204.05275 . arXiv:2204.05275 [cs, eess, math, stat].
- Yao Liu, Adith Swaminathan, Alekh Agarwal, and Emma Brunskill. Provably Good Batch Off-Policy Reinforcement Learning Without Great Exploration. In Advances in Neural Information Processing Systems , volume 33, pages 1264-1274. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc//paper\_files/paper/2020/hash/ 0dc23b6a0e4abc39904388dd3ffadcd1-Abstract.html .
- Pascal Massart. Concentration inequalities and model selection: École d'Ete de Probabilites de Saint-Flour XXXIII - 2003 . Number 1896 in Lecture notes in mathematics. Springer-Verlag, Berlin New York, 2007. ISBN 978-3-540-48503-2.
- Andreas Maurer and Massimiliano Pontil. Empirical Bernstein Bounds and Sample Variance Penalization, July 2009. URL http://arxiv.org/abs/0907.3740 . arXiv:0907.3740 [stat] version: 1.
- Gergely Neu and Nneka Okolo. Dealing with unbounded gradients in stochastic saddle-point optimization, June 2024. URL http://arxiv.org/abs/2402.13903 . arXiv:2402.13903 [cs, math, stat] version: 2.
- Asuman Ozdaglar, Sarath Pattathil, Jiawei Zhang, and Kaiqing Zhang. Offline Reinforcement Learning via Linear-Programming with Error-Bound Induced Constraints, December 2024. URL http://arxiv.org/abs/2212.13861 . arXiv:2212.13861 [cs].
- Charles Chapman Pugh. Real mathematical analysis . Undergraduate texts in mathematics. Springer, Cham Heidelberg, 2. ed edition, 2015. ISBN 978-3-319-17770-0.
- Martin L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming . Wiley Series in Probability and Statistics. Wiley, 1 edition, April 1994. ISBN 978-0-471-61977-2 9780-470-31688-7. doi: 10.1002/9780470316887. URL https://onlinelibrary.wiley.com/ doi/book/10.1002/9780470316887 .
- Paria Rashidinejad, Banghua Zhu, Cong Ma, Jiantao Jiao, and Stuart Russell. Bridging Offline Reinforcement Learning and Imitation Learning: A Tale of Pessimism. IEEE Transactions on Information Theory , 68(12):8156-8196, December 2022. ISSN 1557-9654. doi: 10.1109/TIT. 2022.3185139. URL https://ieeexplore.ieee.org/document/9803237 .
- Slavko Simic. On a global upper bound for Jensen's inequality. Journal of Mathematical Analysis and Applications , 343(1):414-419, July 2008. ISSN 0022247X. doi: 10.1016/j.jmaa.2008.01.060. URL https://linkinghub.elsevier.com/retrieve/pii/S0022247X08000814 .
- Adrienne Tuynman, Rémy Degenne, and Emilie Kaufmann. Finding good policies in average-reward Markov Decision Processes without prior knowledge, May 2024. URL http://arxiv.org/ abs/2405.17108 . arXiv:2405.17108 [cs].
- Masatoshi Uehara and Wen Sun. Pessimistic Model-based Offline Reinforcement Learning under Partial Coverage. October 2021. URL https://openreview.net/forum?id=tyrJsbKAe6 .
- Martin J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint . Cambridge University Press, 1 edition, February 2019. ISBN 978-1-108-62777-1 978-1-108-49802-9. doi: 10.1017/9781108627771. URL https://www.cambridge.org/core/product/identifier/ 9781108627771/type/book .

- Jinghan Wang, Mengdi Wang, and Lin F. Yang. Near Sample-Optimal Reduction-based Policy Learning for Average Reward MDP, December 2022. URL http://arxiv.org/abs/2212. 00603 . arXiv:2212.00603 [cs].
- Shengbo Wang, Jose Blanchet, and Peter Glynn. Optimal Sample Complexity for Average Reward Markov Decision Processes. October 2023. URL https://openreview.net/forum?id= jOm5p3q7c7 .
- Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, and Alekh Agarwal. Bellman-consistent Pessimism for Offline Reinforcement Learning. In Advances in Neural Information Processing Systems , volume 34, pages 6683-6694. Curran Associates, Inc., 2021. URL https://proceedings. neurips.cc/paper/2021/hash/34f98c7c5d7063181da890ea8d25265a-Abstract.html .
- Zihan Zhang and Qiaomin Xie. Sharper Model-free Reinforcement Learning for Averagereward Markov Decision Processes, June 2023. URL http://arxiv.org/abs/2306.16394 . arXiv:2306.16394 [cs].
- Matthew Zurek and Yudong Chen. The Plug-in Approach for Average-Reward and Discounted MDPs: Optimal Sample Complexity Analysis, October 2024. URL http://arxiv.org/abs/ 2410.07616 . arXiv:2410.07616 [cs].
- Matthew Zurek and Yudong Chen. Span-Agnostic Optimal Sample Complexity and Oracle Inequalities for Average-Reward RL, February 2025a. URL http://arxiv.org/abs/2502.11238 . arXiv:2502.11238 [cs].
- Matthew Zurek and Yudong Chen. Span-Based Optimal Sample Complexity for Weakly Communicating and General Average Reward MDPs. Advances in Neural Information Processing Systems , 37:33455-33504, January 2025b. URL https://proceedings.neurips.cc/paper\_files/ paper/2024/hash/3acbe9dc3a1e8d48a57b16e9aef91879-Abstract-Conference.html .

## A Additional notation and guide to appendices

Let π be some stationary policy. Note that P π (defined above as the Markov chain over states induced by policy π on the transition kernel P ) is equal to M π P . We also define r π = M π r . Then we have V π γ = ( I -γP π ) -1 r π . ∥·∥ ∞ and ∥·∥ 1 denote the usual ℓ ∞ / ℓ 1 -norms, respectively. ∥ W ∥ ∞→∞ denotes the ∥·∥ ∞ →∥·∥ ∞ operator norm of a matrix W . In particular ∥ ∥ ( I -γP π ) -1 ∥ ∥ ∞→∞ = 1 1 -γ . Wenote that the action maximization operator M and the policy matrix M π both satisfy monotonicity: V ≥ V ′ (elementwise, for Q,Q ′ ∈ R S×A ) implies M ( Q ) ≥ M ( Q ′ ) , and likewise that M π Q ≥ M π Q ′ . These two operators also both satisfy the 'constant-shift' property, that for any c ∈ R and any Q ∈ R S×A , we have M ( Q + c 1 ) = c 1 + M ( Q ) and M π ( Q + c 1 ) = c 1 + M π ( Q ) . Also we note that M and M π are both 1 -Lipschitz with respect to ∥∥ ∞ , that is ∥ MQ -MQ ′ ∥ ∞ ≤ ∥ Q -Q ′ ∥ ∞ and ∥ M π Q -M π Q ′ ∥ ∞ ≤ ∥ Q -Q ′ ∥ ∞ . For any vector x we let x ◦ k denote its elementwise k th power. We let I denote the usual indicator function used in probability where I ( E ) is a random variable with value 1 if the event E holds and 0 otherwise.

In Appendix B we prove the main theorem, Theorem 3.2. In Appendix C we prove Theorem 3.3 and in Appendix D we prove Theorem 3.4. Appendix E contains additional supporting results.

## B Proof of main theorem

## B.1 Well-definedness

We also define a fixed-policy/policy evaluation version of ̂ T pe which will be useful within the analysis. For any fixed stationary policy π , we let

<!-- formula-not-decoded -->

We also define ̂ V π pe := M π ̂ Q π pe , where ̂ Q π pe is the unique fixed point of ̂ T π pe (justified in the below lemma).

The following is a more comprehensive variant of Lemma 3.1.

Lemma B.1. 1. ̂ T pe satisfies the following properties:

- (a) Monotonicity: If Q ≥ Q ′ then ̂ T pe ( Q ) ≥ ̂ T pe ( Q ′ ) .
- (b) Constant shift: For any c ∈ R , ̂ T pe ( Q + c 1 ) = ̂ T pe ( Q ) + γc 1 .
- (c) γ -contractivity: ̂ T pe is a γ -contraction and has a unique fixed point ̂ Q ⋆ pe .
- (d) Boundedness: 0 ≤ ̂ Q ⋆ pe ≤ 1 1 -γ 1 .
2. For any fixed stationary deterministic policy π , the analogous statements hold for ̂ T π pe :
- (a) Monotonicity: If Q ≥ Q ′ then ̂ T π pe ( Q ) ≥ ̂ T π pe ( Q ′ ) .
- (b) Constant shift: For any c ∈ R , ̂ T π pe ( Q + c 1 ) = ̂ T π pe ( Q ) + γc 1 .
- (c) γ -contractivity: ̂ T π pe is a γ -contraction and has a unique fixed point ̂ Q π pe .
- (d) Boundedness: 0 ≤ ̂ Q π pe ≤ 1 1 -γ 1 .
3. For any fixed stationary deterministic policy π , we have ̂ Q ⋆ pe ≥ ̂ Q π pe .

Proof. We note that a few steps are similar to Li et al. [2023, Lemma 1], but our new choice of penalty requires much more involved analysis.

We define an auxiliary operator T pe : R S → R SA by, for any V ∈ R S ,

<!-- formula-not-decoded -->

We defer the verification of the following fact, which involves somewhat lengthy calculations, to Appendix E.1.

Lemma B.2. Let V, V ′ ∈ R S be arbitrary and suppose that V ≥ V ′ . Then (elementwise)

<!-- formula-not-decoded -->

Given Lemma B.2, we can relatively easily verify Lemma B.1. We note that Lemma B.2 makes use of the quantile clipping in an essential way.

Now we will show item 1 except for the boundedness property. Notice that ̂ T pe ( Q ) = T pe ( MQ ) (for any Q ∈ R SA ). Therefore letting Q,Q ′ ∈ R SA with Q ≥ Q ′ , we have by monotonicity of M that MQ ≥ MQ ′ , and thus by monotonicity of T pe we conclude that

<!-- formula-not-decoded -->

as desired. Next we check the constant shift property of T pe . Fix c ∈ R , V ∈ R S , and s ∈ S , a ∈ A . Then we have that T β ( s,a ) ( ̂ P sa , V + c 1 ) = T β ( s,a ) ( ̂ P sa , V )+ c 1 , regardless of whether β ( s, a ) ∈ [0 , 1] or β ( s, a ) &gt; 1 , since when β ( s, a ) &gt; 1 we have T β ( s,a ) ( ̂ P sa , V + c 1 ) = min s ∈S ( V + c 1 ) 1 = (min s ∈S ( V ) + c ) 1 , and when β ( s, a ) ≤ 1 , by (4) we have

<!-- formula-not-decoded -->

Therefore and

<!-- formula-not-decoded -->

and therefore we have that b ( s, a, V ) = b ( s, a, V + c 1 ) . Additionally we have that

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

(since ̂ P sa 1 = 1 ). Using (9) and the fact that M ( Q + c 1 ) = MQ + c 1 we can show that ̂ T pe satisfies the constant shift property as well:

<!-- formula-not-decoded -->

as desired. Finally we can check contractivity of ̂ T pe . We note that it suffices to show that T pe is γ -Lipschitz, since then we would have for any Q 1 , Q 2 ∈ R SA that

<!-- formula-not-decoded -->

as desired, where the first inequality is due to the (assumed) Lipschitzness of T pe and the second inequality is due to the 1 -Lipschitzness of M . Now we verify that T pe is indeed γ -Lipschitz. For any V 1 , V 2 ∈ R S we have V 1 ≤ V 2 + ∥ V 1 -V 2 ∥ ∞ 1 (elementwise), so by monotonicity of T pe (Lemma

B.2), and then using the fact that T pe satisfies the constant shift property (shown in (9)) in the next inequality, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By reversing the roles of V 1 and V 2 we also have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining these two inequalities involving T pe ( V 2 ) - T pe ( V 1 ) we conclude that ∥ ∥ T pe ( V 1 ) -T pe ( V 2 ) ∥ ∥ ∞ ≤ γ ∥ V 1 -V 2 ∥ ∞ as desired and thus ̂ T pe is a γ -contraction. By the Banach fixed-point theorem (e.g. [Pugh, 2015, Chapter 4.5]) this implies the existence of a unique fixed point of ̂ T pe , which we call ̂ Q ⋆ pe . (We check that 0 ≤ ̂ Q ⋆ pe ≤ 1 1 -γ 1 later.)

Now we will show item 2 except for the boundedness property. Notice that similarly to the previous case, ̂ T π pe ( Q ) = T pe ( M π Q ) (for any Q ∈ R SA ). The only properties of M used in the proofs for the previous case were monotonicity (that Q ≥ Q ′ = ⇒ MQ ≥ MQ ′ ), that M ( Q + c 1 ) = MQ + c 1 , and that M is 1 -Lipschitz. All of these properties are also true with M π in place of M , so in fact all proofs used to verify item 1 can immediately be applied (with this minor modification) to also verify item 2.

Next, item 3 would follow by showing that for any fixed Q ∈ R SA we have

<!-- formula-not-decoded -->

since then by a standard argument we can show for any integer k ≥ 0 that

<!-- formula-not-decoded -->

(where ( k ) denotes k compositions of an operator) and therefore that

<!-- formula-not-decoded -->

So now we focus on showing (10), but this follows immediately from the fact that MQ ≥ M π Q and that T pe is monotone (Lemma B.2), since we have

<!-- formula-not-decoded -->

[Matthew: should be β ( s, a ) not β in the below paragraph?] [Guy: yes] Finally, we check both boundedness properties. Since we already have that ̂ Q π pe ≤ ̂ Q ⋆ pe , it suffices to show that 0 ≤ ̂ Q π pe and that ̂ Q ⋆ pe ≤ 1 1 -γ 1 . First, note that we have ̂ T π pe ( 0 ) ≥ 0 , since for any s ∈ S , a ∈ A ,

<!-- formula-not-decoded -->

Then by monotonicity of ̂ T π pe we have for any integer k ≥ 0 that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so by rearranging or equivalently

and so

as desired. Similarly, we have that ̂ T pe ( 1 / (1 -γ )) ≤ 1 / (1 -γ ) , since for any s ∈ S , a ∈ A , ̂ T pe ( 1 / (1 -γ ))( s, a )

<!-- formula-not-decoded -->

By an analogous argument to the previous bound, we have from monotonicity of ̂ T pe that ( ̂ T pe ) ( k ) ( 1 / (1 -γ )) ≤ 1 / (1 -γ ) for all positive integers k and thus that ̂ Q ⋆ pe ≤ 1 / (1 -γ ) .

In the above proof we defined the operator T pe and verified its Lipshitzness, which we state in the following lemma as T pe will appear again later.

Lemma B.3. T pe is γ -Lipschitz.

## B.2 Optimization

In this subsection we establish the basic properties of the outputs of Algorithm 1.

Lemma B.4. Algorithm 1 returns ̂ Q such that

<!-- formula-not-decoded -->

Proof. First we note that ̂ T pe ( 0 ) ≥ 0 , which follows easily from the definition (3) since (for arbitrary s ∈ S , a ∈ A )

<!-- formula-not-decoded -->

̂ T pe ( ̂ Q ) ≥ ̂ Q follows from this fact and monotonicity of ̂ T pe by standard arguments, since if for any t ∈ N we have that ̂ T pe ( ̂ Q t ) ≥ ̂ Q t then

<!-- formula-not-decoded -->

so by induction (since ̂ Q 0 = 0 ) ̂ T pe ( ̂ Q t ) ≥ ̂ Q t holds for t = K , and we have ̂ Q K = ̂ Q by definition. Now we argue that ̂ Q ≤ ̂ Q ⋆ pe , which follows from ̂ T pe ( ̂ Q ) ≥ ̂ Q and monotonicity of ̂ T pe by standard arguments, since assuming for some t ≥ 1 that ̂ T ( t ) pe ( ̂ Q ) ≥ ̂ Q , then we have by monotonicity that

<!-- formula-not-decoded -->

and so by induction ( ̂ T pe ) ( t ) ( ̂ Q ) ≥ ̂ Q for all t ≥ 1 , and thus

<!-- formula-not-decoded -->

as desired.

Finally we check that ̂ Q ⋆ pe ≤ ̂ Q + 1 2 n tot 1 . Again note that ̂ Q = ̂ Q K . By the definition of K = ⌈ log ( 2 n tot 1 -γ ) 1 -γ ⌉ , as well as the fact that log(1 /γ ) ≥ 1 -γ for any γ , we have

<!-- formula-not-decoded -->

Using this bound, γ -contractivity, and the fact that 0 ≤ ̂ Q ⋆ pe ≤ 1 1 -γ 1 from Lemma B.1, we have

<!-- formula-not-decoded -->

2 n tot

## B.3 Concentration

In this subsection we establish the key concentration inequalities, given in Lemmas B.7 and B.8, using leave-one-out techniques. We start with two helper lemmas which abstractly handle the leave-one-out-based covering steps before proving Lemmas B.7 and B.8.

Lemma B.5. Fix some δ ′ &gt; 0 and some s ∈ S , a ∈ A . Suppose that for some random vector X ∈ R S , there exists a (deterministic) set U and some random variables X u ∈ R S for each u (that is, for each u ∈ U , X u is a random vector in R S ) such that

1. For all u ∈ U , X u is independent of all samples S 1 sa , . . . , S n ( s,a ) sa drawn from P ( · | s, a ) .
2. Almost surely there exists some u ⋆ ∈ U such that ∥ X -X u ⋆ ∥ ∞ ≤ 1 n tot .

Also assume n ( s, a ) ≥ 2 . Then with probability at least 1 -6 δ ′ , we have that

<!-- formula-not-decoded -->

Proof. We start by showing that

<!-- formula-not-decoded -->

where the final inequality is because ∥ ∥ ∥ ̂ P sa -P sa ∥ ∥ ∥ 1 ≤ 2 and ∥ X -X u ⋆ ∥ ∞ ≤ 1 n tot .

Then since for any fixed u ∈ U we have ( ̂ P sa -P sa ) X u = ∑ n ( s,a ) i =1 ( X u ( S i sa ) -P sa X u ) , by Hoeffding's inequality conditioned on X u (since by assumption X u is independent from the S i sa and each term in the above sum is contained within the interval [min X u , max X u ] which has length ∥ X u ∥ span ) we have that

<!-- formula-not-decoded -->

and so

<!-- formula-not-decoded -->

Taking a union bound, the above inequality holds for all u ∈ U with probability at least 1 -2 δ ′ . Finally, since

<!-- formula-not-decoded -->

combining with (15) we have that

<!-- formula-not-decoded -->

Next we would like to apply the concentration inequalities of Maurer and Pontil [2009]. To apply their theorems as stated, we must shift and normalize to define (for each u ∈ U )

<!-- formula-not-decoded -->

so that X ′ u ∈ [0 , 1] almost surely. Fixing some u ∈ U and applying Maurer and Pontil [2009, Theorem 10], assuming n ( s, a ) ≥ 2 , we have with probability at least 1 -2 δ ′ / | U | that

<!-- formula-not-decoded -->

using the facts that by standard calculations, abbreviating ˜ n = n ( s, a ) for convenience,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

(since Maurer and Pontil [2009, Theorem 10] as stated involves the quantity 1 2˜ n (˜ n -1) ∑ ˜ n i =1 ∑ ˜ n j =1 ( X ′ u ( S i sa ) -X ′ u ( S j sa ) ) 2 and its expectation). Taking a union bound and undoing the normalization and shifting, we have for all u ∈ U that

<!-- formula-not-decoded -->

with probability at least 1 -2 δ ′ . For any arbitrary probability distribution µ ∈ R S we have that since

<!-- formula-not-decoded -->

(where the inequality step follows from triangle inequality since Y ↦→ √ E Y 2 is a norm on random variables Y ) and then we have

<!-- formula-not-decoded -->

Thus combining (17) with (18) we conclude that

<!-- formula-not-decoded -->

using (16) again in the final inequality. To obtain the slightly simplified bound (13) we use that by assumption n ( s, a ) ≥ 2 , so √ n ( s,a ) n ( s,a ) -1 ≤ √ 2 ≤ 2 .

Now, similarly to our use of Hoeffding's inequality, using Bernstein's inequality (e.g., Maurer and Pontil [2009, Theorem 3]), as well as a union bound over all u ∈ U , we have that with probability at least 1 -2 δ ′ , for all u ∈ U ,

<!-- formula-not-decoded -->

Combining this inequality (for u = u ⋆ ) along with (15), (16), and (18), we obtain that

<!-- formula-not-decoded -->

Combining this with (20) we furthermore obtain that

<!-- formula-not-decoded -->

Now we develop several leave-one-out constructions which satisfy the conditions of Lemma B.5.

- Lemma B.6. 1. (LOO construction for ̂ V ⋆ pe ) For each s, a , there exists a set U 1 sa ⊆ R with | U 1 sa | ≤ n tot 1 -γ and random vectors ( X 1 u ) u ∈ U 1 sa such that 1) for all u ∈ U 1 sa , X 1 u is independent from S 1 sa , . . . , S n ( s,a ) sa , and 2) almost surely there exists some u ∈ U 1 sa such that ∥ ∥ ∥ ̂ V ⋆ pe -X 1 u ∥ ∥ ∥ ∞ ≤ 1 n tot .
2. (LOO constructions for T β ( s,a ) ( ̂ P sa , ̂ V ⋆ pe ) ) For each s, a , there exists a set U 2 sa ⊆ R with | U 2 sa | ≤ S n tot 1 -γ and random vectors ( X 2 u ) u ∈ U 2 sa such that 1) for all u ∈ U 2 sa , X 2 u is independent from S 1 sa , . . . , S n ( s,a ) sa , and 2) almost surely there exists some u ∈ U 2 sa such that ∥ ∥ ∥ T β ( s,a ) ( ̂ P sa , ̂ V ⋆ pe ) -X 2 u ∥ ∥ ∥ ∞ ≤ 1 n tot .
3. (LOO construction for ̂ V π pe ) Fix any policy π . For each s, a , there exists a set U 3 sa ⊆ R with | U 3 sa | ≤ n tot 1 -γ and random vectors ( X 3 u ) u ∈ U 3 sa such that 1) for all u ∈ U 3 sa , X 3 u is independent from S 1 sa , . . . , S n ( s,a ) sa , and 2) almost surely there exists some u ∈ U 3 sa such that ∥ ∥ ∥ ̂ V π pe -X 3 u ∥ ∥ ∥ ∞ ≤ 1 n tot .
4. (LOO constructions for T β ( s,a ) ( ̂ P sa , ̂ V π pe ) ) Fix any policy π . For each s, a , there exists a set U 4 sa ⊆ R with | U 4 sa | ≤ S n tot 1 -γ and random vectors ( X 4 u ) u ∈ U 4 sa such that 1) for all u ∈ U 4 sa , X 4 u is independent from S 1 sa , . . . , S n ( s,a ) sa , and 2) almost surely there exists some u ∈ U 4 sa such that ∥ ∥ ∥ T β ( s,a ) ( ̂ P sa , ̂ V π pe ) -X 4 u ∥ ∥ ∥ ∞ ≤ 1 n tot .

Proof. We start by showing item 1. Fix arbitrary s ∈ S , a ∈ A . For any u ∈ R we define the reward function r s,u ∈ R SA , (random) transition matrix ̂ P s ∈ R SA×S , and (random) operator

T s,u pe : R S → R SA by (for arbitrary s ′ ∈ S , a ′ ∈ S , V ∈ R S )

̸

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Note e ⊤ s is a vector which is all 0 except for a 1 in state s , meaning that state s is absorbing in ̂ P s,u , for all actions. Also all actions receive reward u in this state. All other state-action pairs have the same rewards and transition distributions as in the MDP ( ̂ P,r ) . Also, we have defined b s and T s,u pe in an identical manner to b and T pe , except we now use r s,u and ̂ P s in place of r and ̂ P . Since all of the properties of T pe verified above only required ̂ P to be a valid transition matrix and for r to be a vector in [0 , 1] SA , the properties hold identically for T s,u pe , and thus by Lemma B.3 we have that T s,u pe is γ -Lipschitz.

Now we define ̂ L s,u : R S → R S as ̂ L s,u ( V ) := M T s,u pe ( V ) (for any V ∈ R S ). By the γ -Lipschitzness of T s,u pe and the 1 -Lipschitzness of M , we immediately have that ̂ L s,u is a γ -contraction, since

<!-- formula-not-decoded -->

for any V 1 , V 2 ∈ R S . Therefore contractivity implies that there exists a unique fixed point of ̂ L s,u (e.g. [Pugh, 2015, Chapter 4.5]), which we call X 1 u . Note that since ̂ L s,u is defined without using ̂ P sa , it is independent of all samples S 1 sa , . . . , S n ( s,a ) sa drawn from P ( · | s, a ) .

Now, as intermediate steps, we show the following two properties:

<!-- formula-not-decoded -->

For A, letting u, u ′ ∈ R , we can calculate that

<!-- formula-not-decoded -->

where the key equality step was that T s,u pe ( X 1 u ) = r s,u -r s,u ′ + T s,u ′ pe ( X 1 u ) , and in the final inequality we used γ -Lipschitzness of T s,u ′ pe . Rearranging we obtain that ∥ ∥ X 1 u -X 1 u ′ ∥ ∥ ∞ ≤ | u -u ′ | 1 -γ as desired, verifying A.

̸

<!-- formula-not-decoded -->

For B we first check that X 1 U ⋆ = ̂ V ⋆ pe . It suffices to check that M T s,U ⋆ pe ( ̂ V ⋆ pe ) = M T pe ( ̂ V ⋆ pe ) , because then we would have that

<!-- formula-not-decoded -->

̸

thus showing that ̂ V ⋆ pe is a fixed point of ̂ L s,U ⋆ , and by uniqueness of this fixed point we must have X 1 U ⋆ = ̂ V ⋆ pe . Comparing the definitions of T pe ( ̂ V ⋆ pe ) and T s,U ⋆ pe ( ̂ V ⋆ pe ) , it is immediate that M ( T s,U ⋆ pe ( ̂ V ⋆ pe ) ) ( s ′ ) = M ( T pe ( ̂ V ⋆ pe ) ) ( s ′ ) for all s ′ = s , so it remains to check the equality for s ′ = s .

First we argue that for all a ′ ∈ A , we have b s ( s, a ′ , ̂ V ⋆ pe ) = 5 n tot . If β ( s, a ′ ) &gt; 1 then we have T β ( s,a ′ ) ( ̂ P s sa ′ , ̂ V ⋆ pe ) = ( min s ′ ̂ V ⋆ pe ( s ′ ) ) 1 , and if β ( s, a ′ ) ≤ 1 then we have T β ( s,a ′ ) ( ̂ P s sa ′ , ̂ V ⋆ pe ) = ̂ V ⋆ pe ( s ) 1 , since ̂ P s sa ′ = e ⊤ s ( ̂ P s sa ′ transitions to state s with probability 1 ). Either way T β ( s,a ′ ) ( ̂ P s sa ′ , ̂ V ⋆ pe ) is a multiple of the all-ones vector, which implies V ̂ P s s ′ a ′ [ T β ( s ′ ,a ′ ) ( ̂ P s s ′ a ′ , ̂ V ⋆ pe ) ] = 0 and ∥ ∥ ∥ T β ( s ′ ,a ′ ) ( ̂ P s s ′ a ′ , ̂ V ⋆ pe ) ∥ ∥ ∥ span = 0 , and thus that b s ( s, a ′ , ̂ V ⋆ pe ) = 5 n tot . Therefore by the construction of U ⋆ we have that

<!-- formula-not-decoded -->

as desired, so we have checked that X 1 U ⋆ = ̂ V ⋆ pe .

Now it remains to verify that U ⋆ ∈ [0 , 1] . Given our calculation of T β ( s,a ′ ) ( ̂ P s sa ′ , ̂ V ⋆ pe ) (for any a ′ ∈ A ) above, we have the alternate expression for U ⋆

<!-- formula-not-decoded -->

We consider the two cases in the above expression. If ∃ a ′ ∈ A : β ( s, a ′ ) ≤ 1 , then we can upper bound U ⋆ as

<!-- formula-not-decoded -->

where the last inequality is due to the fact that ̂ V ⋆ pe = M ̂ Q ⋆ pe ≤ M 1 1 -γ 1 = 1 1 -γ 1 (by Lemma B.1). For the lower bound in this case, we have

<!-- formula-not-decoded -->

which is clearly ≥ 0 (note the first term within the min is ≥ 0 by Lemma B.1).

Now we consider the case that there does not exist a ′ ∈ A such that β ( s, a ′ ) ≤ 1 , that is, the case that β ( s, a ′ ) &gt; 1 for all a ′ ∈ A . Then as argued above we have for all a ′ ∈ A that T β ( s,a ′ ) ( ̂ P sa ′ , ̂ V ⋆ pe ) =

( min s ′′ ̂ V ⋆ pe ( s ′′ ) ) 1 , and so by the definition of ̂ T pe and the fact that ̂ Q ⋆ pe is its fixed point and ̂ V ⋆ pe = M ̂ Q ⋆ pe , we have

<!-- formula-not-decoded -->

(using the fact that b ( s, a ′ , ̂ V ⋆ pe ) ≥ 0 to compute the max). Hence in this case

<!-- formula-not-decoded -->

which is clearly in [0 , 1] . We have thus verified B.

Now unfix u and let U 1 sa be a set of n tot 1 -γ points chosen by dividing [0 , 1] into n tot 1 -γ intervals and placing a point at the midpoint of each such interval. Note this guarantees that for any x ∈ [0 , 1] there exists some u ∈ U such that | x -u | ≤ 1 -γ 2 n tot . Therefore, letting ˜ U ⋆ ∈ U be this closest point in U to the value U ⋆ , we have by A and B that

<!-- formula-not-decoded -->

Therefore we have confirmed item 1.

Now we continue to item 2. Fix s ∈ S , a ∈ A , and define U 2 sa = U 1 sa ×S . For each u, s ′ ∈ U 2 sa , we define

<!-- formula-not-decoded -->

that is, we clip all entries of the vector X 1 u constructed in the previous part so that they are ≤ X 1 u ( s ′ ) . Since X 1 u was independent of all samples S 1 sa , . . . , S n ( s,a ) sa drawn from P ( · | s, a ) , the same is true of X 2 u,s ′ . Define S ⋆ ( s, a ) to be a state such that Q β ( s,a ) ( ̂ P sa , ̂ V ⋆ pe ) = ̂ V ⋆ pe ( S ⋆ ( s, a )) (if multiple states satisfy this, we can break ties in some consistent manner). Then for any u, s ′ ∈ U 2 sa we have

<!-- formula-not-decoded -->

From item 1 we know there exists some u ∈ U 1 sa such that ∥ ∥ ∥ ̂ V ⋆ pe -X 1 u ∥ ∥ ∥ ∞ ≤ 1 2 n tot , and furthermore if s ′ = S ⋆ ( s, a ) then

<!-- formula-not-decoded -->

Combining these with (22) we conclude that almost surely there exists some ( u, s ′ ) ∈ U 1 sa ×S = U 2 sa such that ∥ ∥ ∥ T β ( s,a ) ( ̂ P sa , ̂ V ⋆ pe ) -X 2 u,s ′ ∥ ∥ ∥ ∞ ≤ 1 n tot as desired. Therefore we have confirmed item 2.

For item 3 and item 4, we can use nearly identical constructions, with the only difference being that for item 3 we define X 3 u to be the fixed point of the operator ̂ L π,s,u : R S → R S as ̂ L π,s,u ( V ) := M π T s,u pe ( V ) (and otherwise use the same construction as for X 1 u ), and then for item 4 we use X 3 u in place of X 1 u in the construction for X 2 u . Thus, the key difference is replacing M with M π within the construction for X 3 u , and since the only properties of M used were 1 -Lipschitzness and that M 1 = 1 , which both hold with M π in place of M , and also the fact that ̂ V ⋆ pe = M ̂ Q ⋆ pe which is analogous to the fact that ̂ V π pe = M π ̂ Q π pe , all steps work in an analogous manner.

Now we can prove the key concentration inequalities needed for the rest of the proof.

Lemma B.7. With probability at least 1 -δ , for all s ∈ S , a ∈ A , if n ( s, a ) ≥ 1+8log ( 6 S 2 An tot (1 -γ ) δ ) , then

<!-- formula-not-decoded -->

where α = 8log ( 6 S 2 An tot (1 -γ ) δ ) and β ( s, a ) = α max { n ( s,a ) -1 , 1 } .

Proof. Fix some s ∈ S and a ∈ A . If n ( s, a ) &lt; 1 + 8 log ( 6 S 2 An tot (1 -γ ) δ ) then we have nothing to check. Otherwise, we can immediately combine item 2 of Lemma B.6 (which gives | U | ≤ S n tot 1 -γ ) with Lemma B.5 (since our condition on n ( s, a ) clearly implies n ( s, a ) ≥ 2 ) to conclude that with probability at least 1 -6 δ ′ ,

<!-- formula-not-decoded -->

Taking a union bound over all s ∈ S , a ∈ A , and setting δ ′ = δ 6 SA , we obtain that with probability at least 1 -δ , for all s ∈ S , a ∈ A where n ( s, a ) ≥ 1 + 8 log ( 6 S 2 An tot (1 -γ ) δ ) , we have

<!-- formula-not-decoded -->

where the second inequality uses the assumption that n ( s, a ) ≥ 1+8log ( 6 S 2 An tot (1 -γ ) δ ) and the fact that 2+3 √ 1 4 + 14 3 1 8 &lt; 4 . 5 , and then we bounded a + b ≤ 2 max { a, b } . Wealso note that since we are in the case that n ( s, a ) ≥ 1+8log ( 6 S 2 An tot (1 -γ ) δ ) ≥ 9 , we have that n ( s, a ) -1 = max { n ( s, a ) -1 , 1 } . ⋆

Lemma B.8. Fix any policy π . With probability at least 1 -2 δ , for all s ∈ S , a ∈ A , if n ( s, a ) ≥ 1 + 8 ln ( 6 S 2 An tot (1 -γ ) δ ) , then

<!-- formula-not-decoded -->

and and

<!-- formula-not-decoded -->

Proof. The first statement is analogous to Lemma B.7 but uses the construction of item 4 of Lemma B.6 in place of item 2. Thus combining item 4 of Lemma B.6 with Lemma B.5, taking a union bound and performing the same simplifications, we obtain that with probability at least 1 -δ , for all s ∈ S , a ∈ A , if n ( s, a ) ≥ 1 + 8 ln ( 6 S 2 An tot (1 -γ ) δ ) , then

<!-- formula-not-decoded -->

Now we establish the second two properties. We will show that they both hold with probability 1 -δ , after which we are done since we can then use a union bound to combine with the above. Fixing some s ∈ S and a ∈ A , if n ( s, a ) &lt; 1 + 8 log ( 6 S 2 An tot (1 -γ ) δ ) then we have nothing to check. Otherwise, we can immediately combine item 3 of Lemma B.6 (which gives | U | ≤ n tot 1 -γ ≤ S n tot 1 -γ ) with Lemma B.5 (since our condition on n ( s, a ) implies n ( s, a ) ≥ 2 ) to conclude that with probability at least 1 -6 δ ′ , we have both

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking a union bound over all s, a ∈ S , A and setting δ ′ = δ 6 SA , we have that with probability at least 1 -δ , for all s, a such that n ( s, a ) ≥ 1 + 8 ln ( 6 S 2 An tot (1 -γ ) δ ) , both

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where for the first bound we simplified (25) using the condition on n ( s, a ) and the fact that 2+ √ 2 8 + 2 3 1 8 &lt; 3 , and for the second bound we simplified (27) also using the condition on n ( s, a ) and then the fact that 2 √ 2 8 +3 = 4 .

## B.4 Pessimism

In this subsection we establish the following essential pessimism property, making use of the previous concentration results and our construction of ̂ T pe .

Lemma B.9. Under the event in Lemma B.7, we have that

<!-- formula-not-decoded -->

Proof. We will show that T ̂ π ( ̂ Q ) ≥ ̂ Q (where T ̂ π ( Q ) := r + PM ̂ π Q is the Bellman evaluation operator for ̂ π ), which by a standard argument implies that Q ̂ π ≥ ̂ Q , since we can then easily derive (by monotonicity of T ̂ π ) that ( T ̂ π ) ( k ) ( ̂ Q ) ≥ ̂ Q for any integer k ≥ 0 , and thus

<!-- formula-not-decoded -->

Fixing arbitrary s ∈ S , a ∈ A , we will now verify that T ̂ π ( ̂ Q )( s, a ) ≥ ̂ Q ( s, a ) . From Lemma B.4 we have that ̂ T pe ( ̂ Q )( s, a ) ≥ ̂ Q ( s, a ) . We consider two cases based upon the value of ̂ T pe ( ̂ Q )( s, a ) , which by (3) is either 1) equal to r ( s, a ) + γ ̂ P sa T β ( s,a ) ( ̂ P sa , M ̂ Q ) -γb ( s, a, M ̂ Q ) or 2) equal to r ( s, a ) + γ min s ′ ( M ̂ Q )( s ′ ) . In the simpler case 2, we thus have that

<!-- formula-not-decoded -->

using the facts that min s ′ V ( s ′ ) ≤ P sa V for any V ∈ R S (since P sa is a probability distribution) and that M ̂ Q = M ̂ π ̂ Q since ̂ π is greedy with respect to ̂ Q . We therefore have that ̂ Q ( s, a ) ≤ ̂ T pe ( ̂ Q )( s, a ) ≤ T ̂ π ( ̂ Q )( s, a ) in case 2, as desired. Now we consider case 1. Note that since we are in case 1, we must have that β ( s, a ) ≤ 1 , which implies that n ( s, a ) ≥ α +1 (because if we had β ( s, a ) &gt; 1 , then we would have T β ( s,a ) ( ̂ P sa , M ̂ Q ) = min s ′ ( M ̂ Q )( s ′ ) , and b ( s, a, M ̂ Q ) &gt; 0 , so the term T β ( s,a ) ( ̂ P sa , M ̂ Q ) -b ( s, a, M ̂ Q ) could not have achieved the maximum in the definition (3)

of ̂ T pe ). Then we have that

<!-- formula-not-decoded -->

where the first inequality is due to ̂ T pe ( ̂ Q ) ≥ ̂ Q from Lemma B.4, the second inequality is due to monotonicity of ̂ T pe (Lemma B.1) and the fact that ̂ Q ≤ ̂ Q ⋆ pe (Lemma B.4), the third inequality is by triangle inequality, the fourth inequality is from Lemma B.7, the fifth inequality is from the trivial fact that elementwise T β ( s,a ) ( ̂ P sa , M ̂ Q ⋆ pe ) ≤ M ̂ Q ⋆ pe , the sixth inequality follows from ̂ Q ⋆ pe ≤ ̂ Q + 1 2 n tot 1 due to Lemma B.4 (since by monotonicity of M , M ̂ Q ⋆ pe ≤ M ( ̂ Q + 1 2 n tot 1 ) = M ̂ Q + 1 2 n tot 1 ), and the final equality is from the definition of ̂ π (from Algorithm 1) since it is greedy with respect to ̂ Q . Combining the two cases we have shown that T ̂ π ( ̂ Q ) ≥ ̂ Q as desired. Combining the two cases we have shown that T ̂ π ( ̂ Q ) ≥ ̂ Q as desired.

## B.5 Policy hitting radius lemmas

In this subsection we establish some key properties regarding the relationship between T hit and certain discounted policy occupancy measures which will appear in later analysis steps. We also establish some facts about T hit of general interest and compare it to the mixing time.

Recall that η s := inf { t ≥ 0 : S t = s } is the first hitting time of state s . We define an additional useful quantity: for any s ⋆ ∈ S , let

<!-- formula-not-decoded -->

This is the maximum expected hitting time of state s ⋆ in the Markov chain P π (which can be infinite). Then we have

<!-- formula-not-decoded -->

T hit ( P, π ) is finite if and only if P π is unichain:

Lemma B.10. Fix a policy π and an MDP transition kernel P . Then the Markov chain P π is unichain if and only if T hit ( P, π ) is finite.

Proof. First, suppose that T hit ( P, π ) is finite. Then there exists some s ⋆ such that for all s 0 ∈ S , E π s 0 η s ⋆ &lt; ∞ . Therefore s ⋆ is reachable from any state, so all recurrent classes must contain s ⋆ , but since the irreducible closed recurrent classes (along with the transient states) form a partition of S , this implies that there can only be one closed irreducible recurrent class, that is that P π is unichain.

Next, suppose that P π is unichain. Let ˜ s ⋆ be some state in the single closed irreducible recurrent class of P π . Now we argue that E π s 0 [ η ˜ s ⋆ ] &lt; ∞ for any s 0 ∈ S . First, it is a standard fact (in finite Markov chains) that letting C be the recurrent class, we have M := max s 0 ∈ C E π s 0 [ η ˜ s ⋆ ] &lt; ∞ (e.g. Kemeny and Snell [1976], where E π s 0 [ η ˜ s ⋆ ] is referred to as the mean first passage time). Now letting s 0 be any fixed transient state, since there exists a unique irreducible recurrent class C , letting η C = inf { t ≥ 0 : S t ∈ C } be its first hitting time, it is also a standard fact (for finite Markov chains) that E π s 0 η C &lt; ∞ (replacing C with a single absorbing state, the new chain becomes an absorbing chain, and the absorption time formulas in Kemeny and Snell [1976] imply E π s 0 η C &lt; ∞ ). Then a

calculation using the strong Markov property (where F η C is the stopped sigma-algebra associated with the stopping time η C ) implies that

<!-- formula-not-decoded -->

Since there are only a finite number of such transient states s 0 , the maximum of E π s 0 [ η ˜ s ⋆ ] over all such states is finite. Hence T hit ( P, π ) ≤ max s 0 ∈S E π s 0 [ η ˜ s ⋆ ] &lt; ∞ .

Define d π γ,s 0 ∈ R S as and

<!-- formula-not-decoded -->

We often drop the dependence on γ, π and simply write d s 0 . We also define d ⋆ ( s ) = 1 1 -γ µ ⋆ ( s ) .

Lemma B.11. Let s ⋆ ∈ S satisfy T hit ( P, π ) = T hit ( P, π, s ⋆ ) . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Weuse a coupling argument, and these calculations are somewhat inspired by those in [Cheikhi and Russo, 2023, Lemma B.13]. Starting with the first statement, fix some s 0 ∈ S . Let S ⋆ 0 , S ⋆ 1 , . . . , be the stochastic process with distribution given by the Markov chain P π with starting state s ⋆ , and let S 0 , S 1 , . . . , be the stochastic process with distribution given by the Markov chain P π but with starting state s 0 . Let η s ⋆ = inf { t : S t = s ⋆ } be the first hitting time of the state s ⋆ by the process ( S t ) ∞ t =0 . Now define the process S ′ 0 , S ′ 1 , . . . identically to ( S t ) ∞ t =0 but to follow ( S ⋆ t ) ∞ t =0 once it reaches s ⋆ , that is S ′ η s ⋆ = S ⋆ 0 , S ′ η s ⋆ +1 = S ⋆ 1 , and so on. It is a standard fact due to the Markov property that ( S ′ t ) ∞ t =0 has the same distribution as ( S t ) ∞ t =0 . Now add an absorbing terminal state q (which we do not consider as an element of S ) and for all t ≥ 1 let Z t ∼ Bernoulli ( γ ) (independently), and define the processes ( ˜ S ′ t ) ∞ t =0 and ( ˜ S ⋆ t ) ∞ t =0 by ˜ S ′ 0 = S ′ 0 , ˜ S ⋆ 0 = S ⋆ 0 , and for all t ≥ 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Intuitively speaking, ( ˜ S ′ t ) ∞ t =0 and ( ˜ S ⋆ t ) ∞ t =0 will reach the absorbing state q at the same time, and the probability of reaching it on any given timestep is γ if it has not yet been reached. It is a standard fact that d π γ,s 0 ( s ) = E ∑ ∞ t =0 I ( ˜ S ′ t = s ) and that d π γ,s ⋆ ( s ) = E ∑ ∞ t =0 I ( ˜ S ⋆ t = s ) . Hence using the above coupling we can bound d π γ,s 0 ( s ) -d π γ,s ⋆ ( s ) . Specifically we have

<!-- formula-not-decoded -->

where in the final equality we let η q = inf { t ≥ 1 : Z t = 1 } be the first hitting time of the terminal state. Now we consider two cases. On the event that η q ≤ η s ⋆ , we have

̸

<!-- formula-not-decoded -->

On the event that η s ⋆ &lt; η q , we have

̸

<!-- formula-not-decoded -->

using the fact that S ′ η s ⋆ = S ⋆ 0 , S ′ η s ⋆ +1 = S ⋆ 1 , . . . to cancel terms. Combining the bounds for the two cases with (28), we have that

<!-- formula-not-decoded -->

as desired.

The second statement of the lemma follows immediately from the first, since by triangle inequality

<!-- formula-not-decoded -->

Lemma B.12. Let π be a policy such that P π is unichain, and let µ π ∈ R S denote its stationary distribution. Then

<!-- formula-not-decoded -->

Proof. Since µ π is a stationary distribution, we have for any s ∈ S that

<!-- formula-not-decoded -->

(since ( µ π ) ⊤ P π = ( µ π ) ⊤ ). Then we can calculate by Jensen's inequality that for any fixed s ∈ S ,

<!-- formula-not-decoded -->

Therefore

<!-- formula-not-decoded -->

where in the second inequality step we used Lemma B.11.

Lemma B.13. For any policy π , ∥ h π ∥ span ≤ 4 T hit ( P, π ) .

Proof. Note that by Lemma B.10, if P π is not unichain then T hit ( P, π ) = ∞ and so the desired bound holds trivially (note ∥ h π ∥ span is always finite). So we can now focus on the case that P π is unichain. This implies ρ π is a state-independent constant. In this case it is a standard fact (e.g. [Puterman, 1994, Corollary 8.2.4]) that for any s, s ′ ∈ S ,

<!-- formula-not-decoded -->

Therefore

<!-- formula-not-decoded -->

where the inequality steps are by Holder's inequality and Lemma B.11.

## B.5.1 Relationship between policy hitting radius and uniform mixing time

Here we argue that there is generally no relationship between the policy hitting radius and the mixing time. First, if P π is a unichain and periodic Markov chain, then the mixing time will be infinite/undefined whereas T hit ( P, π ) &lt; ∞ by Lemma B.10.

̸

Now we show an example where the mixing time can be arbitrarily smaller than the policy hitting radius. Suppose that P , π are defined so that P π is the random walk on the complete graph on L nodes, where L is any positive integer. Then µ π ( s ) = 1 /L for all s ∈ S , and after just one step from any starting state we have that S 1 has distribution µ π so τ ( π ) = 1 . However, for any fixed starting state s 0 and any state s = s 0 , we have that η s ∼ Geom (1 /L ) , so E π s 0 η s = L , and hence T hit ( P, π ) = L .

## B.6 Error analysis

Now we can continue with analyzing the relationship between ̂ Q ⋆ pe and ρ π ⋆ , for a comparator policy π ⋆ . Having established pessimism (Lemma B.9), which implies an upper bound on ̂ Q ⋆ pe , we now seek to lower-bound this quantity. Since (by Lemma B.1) ̂ Q ⋆ pe ≥ ̂ Q π ⋆ pe , it suffices to lower-bound ̂ Q π ⋆ pe in terms of V π ⋆ , which is then related to ρ π ⋆ .

Lemma B.14. For any probability distribution µ ∈ ∆ S , any V ∈ R S , and any β ∈ [0 , 1] , we have that

<!-- formula-not-decoded -->

Proof. We prove this by showing the more general statement that for any random variable X and any scalar a ,

<!-- formula-not-decoded -->

Let T = min( X,a ) and ∆ = X -T . Then

<!-- formula-not-decoded -->

Thus to show V [ X ] ≥ V [ T ] it suffices to show that Cov ( T, ∆) ≥ 0 . Now we compute

<!-- formula-not-decoded -->

On the event { X &lt; a } we have ∆ = 0 , so E [∆( T -E T ) I { X &lt; a } ] = 0 . On the event { X ≥ a } , ( T -E T ) ≥ 0 since T = a and E T ≤ a , and ∆ ≥ 0 , so E [∆( T -E T ) I { X ≥ a } ] ≥ 0 . Therefore Cov ( T, ∆) ≥ 0 as desired.

Lemma B.15. Fix any deterministic policy π ⋆ . Under the event in Lemma B.8,

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

We also have that

<!-- formula-not-decoded -->

Proof. Fix s ∈ S , a ∈ A . First we handle the case that β ( s, a ) ≤ 1 . This implies that n ( s, a ) ≥ 1 + α = 1 + 8 log ( 6 S 2 An tot (1 -γ ) δ ) . By the definition (8) of ̂ T π ⋆ pe we have that

<!-- formula-not-decoded -->

By the definition of T β ( s,a ) ( ̂ P sa , ̂ V π ⋆ pe ) we have that (elementwise)

<!-- formula-not-decoded -->

where in the final inequality we used that ∑ s ′ : ̂ V π ⋆ pe ( s ′ ) &gt;Q β ( s,a ) ( ̂ P sa , ̂ V π ⋆ pe ) ̂ P sa ( s ′ ) &lt; β ( s, a ) . Using (24) from Lemma B.8 to relate ̂ P sa ̂ V π ⋆ pe to P sa ̂ V π ⋆ pe , we can further bound

<!-- formula-not-decoded -->

To finish lower-bounding (30) we must also lower-bound b ( s, a, ̂ V π ⋆ pe ) . It is immediate to see that ∥ ∥ ∥ T β ( s,a ) ( ̂ P sa , ̂ V π ⋆ pe ) ∥ ∥ ∥ span ≤ ∥ ∥ ∥ ̂ V π ⋆ pe ∥ ∥ ∥ span , and also by Lemma B.14 (since we are in the β ( s, a ) ≤ 1 case) we have that V ̂ P sa [ T β ( s,a ) ( ̂ P sa , ̂ V π ⋆ pe ) ] ≤ V ̂ P sa [ ̂ V π ⋆ pe ] . These two facts yield that

<!-- formula-not-decoded -->

Furthermore, using the bound (23) from Lemma B.8, we can further bound (33) as

<!-- formula-not-decoded -->

(using the definition of β ( s, a ) and the fact that we are in the β ( s, a ) ≤ 1 case).

Combining (34) and (32) with (30) we obtain that

<!-- formula-not-decoded -->

where we define ˜ b ( s, a ) = √ β ( s, a ) V P sa [ ̂ V π ⋆ pe ] +4 β ( s, a ) ∥ ∥ ∥ ̂ V π ⋆ pe ∥ ∥ ∥ span + 12 n tot

Now for the simpler case that β ( s, a ) &gt; 1 , we have that

<!-- formula-not-decoded -->

Combining the two cases of β ( s, a ) , we have for all s, a that ̂ Q π ⋆ pe ( s, a ) ≥ r ( s, a ) + γP sa ̂ V π ⋆ pe -γ ˜ b ( s, a ) . Therefore by monotonicity of M π ⋆ ,

<!-- formula-not-decoded -->

We also have ̂ V π ⋆ pe -γP π ⋆ ̂ V π ⋆ pe + γ ˜ b π ⋆ ≥ r π ⋆ , which will be needed later. By the Bellman equation for π ⋆ we also have that V π ⋆ = r π ⋆ + γP π ⋆ V π ⋆ . Combining these, rearranging, and using the monotonicity of multiplication by ( I -γP π ⋆ ) -1 (since all its entries are nonnegative), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma B.16. Fix a deterministic unichain policy π ⋆ . Suppose that for all s ∈ S , n ( s, π ⋆ ( s )) ≥ mµ π ⋆ ( s ) + 4 + 4 T hit ( P, π ⋆ ) , 1 1 -γ ≥ m , and 1 1 -γ ≥ 2 . Then under the event in Lemma B.8, we have that

<!-- formula-not-decoded -->

Proof. First we note that, using Lemma B.15, we have

<!-- formula-not-decoded -->

We will now fix some arbitrary s 0 ∈ S and try to bound 〈 d π ⋆ γ,s 0 , ˜ b π ⋆ 〉 . By the assumptions in the lemma statement we have that for all s ∈ S ,

<!-- formula-not-decoded -->

where the third inequality is a consequence of Lemma B.12. For convenience we will let C := (1 -γ ) m , and so we have shown that n ( s, π ⋆ ( s )) ≥ Cd π ⋆ γ,s 0 ( s ) for all s ∈ S . Also for convenience abbreviate ℓ = log ( 6 S 2 An tot (1 -γ ) δ ) . Using the fact that n ( s, π ⋆ ( s )) ≥ 4 which implies 1 max { n ( s,π ⋆ ( s )) -1 , 1 } = 1 n ( s,π ⋆ ( s )) -1 ≤ 4 / 3 n ( s,π ⋆ ( s )) ≤ 2 n ( s,π ⋆ ( s )) , we can simplify ˜ b π ⋆ as

<!-- formula-not-decoded -->

Using this and the fact that n ( s, π ⋆ ( s )) ≥ Cd π ⋆ γ,s 0 ( s ) for all s ∈ S , we have

<!-- formula-not-decoded -->

where in the final inequality we used Cauchy-Schwarz to bound the first term.

Now we focus on bounding the quantity ∑ s ∈S d π ⋆ γ,s 0 ( s ) V P sπ ⋆ ( s ) [ ̂ V π ⋆ pe ] . Let c = min s ∈S ̂ V π ⋆ pe ( s ) and V = ̂ V π ⋆ pe -c 1 . Then

<!-- formula-not-decoded -->

where for the first inequality we used that V + γP π ⋆ V ≥ 0 and that ˜ b π ⋆ +(1 -γ ) c 1 ≥ 0 , and for the second inequality we used that V + γP π ⋆ V ≤ 2 ∥ ∥ V ∥ ∥ ∞ 1 and that V -γP π ⋆ V + γ ˜ b π ⋆ +(1 -γ ) c 1 ≥ 0 , which follows from the fact that

<!-- formula-not-decoded -->

using (29) in the inequality step. Thus

<!-- formula-not-decoded -->

In ( i ) we use (36) and in ( ii ) we use that

<!-- formula-not-decoded -->

Combining the bound (37) with (35) (and noting that ∥ ∥

where we simplified by using that a + b ≤ √ a + b and that 1 γ ≤ 2 (since 1 1 -γ ≥ 2 implies that γ ≥ 1 2 ). The above is a quadratic inequality in x := √ 〈 d π ⋆ γ,s 0 , ˜ b π ⋆ 〉 of the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where y = 64 S ∥ ∥ ∥ ̂ V π ⋆ pe ∥ ∥ ∥ span ℓ C . From the quadratic formula we obtain that

<!-- formula-not-decoded -->

and then squaring both sides we obtain that

<!-- formula-not-decoded -->

using that ( a + b ) 2 ≤ 2 a 2 +2 b 2 . Recalling the definitions of C = (1 -γ ) m and ℓ = log ( 6 S 2 An tot (1 -γ ) δ ) , and also since the above bound held for arbitrary s 0 , we have thus shown that

<!-- formula-not-decoded -->

## B.7 Controlling the empirical span

While Lemma B.16 is approaching the desired result, it involves the empirical span term ∥ ∥ ∥ ̂ V π ⋆ pe ∥ ∥ ∥ span which we would like to bound in terms of ∥ ∥ V π ⋆ ∥ ∥ span . Such a bound is the objective of this subsection, and makes crucial use of our assumption of data even for states which are transient under P π ⋆ .

Lemma B.17. Fix a deterministic unichain policy π ⋆ . Suppose that n ( s, π ⋆ ( s )) ≥ 72( T hit ( P, π ⋆ )) 2 log ( 2 S δ ) for all s ∈ S . Then with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof. The proof of this lemma is inspired by that of Zurek and Chen [2024, Lemma 4]. For any MDP M and s ∈ S we let E π s 0 , M denote the expectation with respect to the Markov chain induced by π in the MDP M from starting state s 0 , and similarly we let P π s 0 , M ( E ) = E π s 0 , M [ I ( E )] denote the associated probability measure. Let s ⋆ ∈ S satisfy T hit ( P, π ⋆ ) = T hit ( P, π, s ⋆ ) . Let ̂ M be the MDP ( ̂ P,r ) . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Supposing that k ∈ N satisfies max s 0 ∈S P s 0 , ̂ M ( η s ⋆ ≥ k ) ≤ 1 2 , then we have for any s ′ 0 that

<!-- formula-not-decoded -->

where the final inequality step used that

<!-- formula-not-decoded -->

which follows from the following standard arguments: for any integer i ≥ 1 (since this formula obviously holds for i = 0 ), we have

<!-- formula-not-decoded -->

where F ( i -1) k is the sigma-algebra generated by S 0 , . . . , S ( i -1) k , step ( i ) is the tower property, step ( ii ) is because the event η s ⋆ ̸∈ { 0 , . . . , ( i -1) k -1 } is F ( i -1) k -measurable, step ( iii ) is the Markov property (e.g., [Durrett, 2019, Theorem 5.2.3]), and step ( iv ) is because P S k , ̂ M ( η s ⋆ ̸∈ { 0 , . . . , k -1 } ) = P S k , ̂ M ( η s ⋆ ≥ k ) ≤ 1 2 (this last inequality holding almost surely, due to the assumption that max s 0 ∈S P s 0 , ̂ M ( η s ⋆ ≥ k ) ≤ 1 2 ). Since these arguments held for arbitrary i , we can repeat them to obtain the desired bound.

̸

Now we try to find such a k . Define the reward function r by r ( s, a ) = I ( s = s ⋆ ) and also let P ′ be the same transition matrix as P except with state s ⋆ made to be absorbing for all actions. Then, for some γ to be chosen later, letting V π ⋆ γ, M ′ be the discounted value function for policy π ⋆ in MDP M ′ = ( P ′ , r ) , and letting E π ⋆ s 0 , M ′ , E π ⋆ s 0 , M denote expectations with respect to the MDPs M ′ and M

respectively, we have that

̸

<!-- formula-not-decoded -->

This implies ∥ ∥ V π ⋆ γ, M ′ ∥ ∥ span ≤ T hit ( P, π ⋆ , s ⋆ ) , which will be needed shortly.

Let ̂ P ′ similarly be the same transition matrix as ̂ P except s ⋆ is absorbing for all actions. Let ̂ M ′ be the MDP ( ̂ P ′ , r ) . Then for any k ∈ N we have

̸

<!-- formula-not-decoded -->

Rearranging this implies that

<!-- formula-not-decoded -->

where for the second inequality we set γ = 1 -1 k and used the fact that (1 -1 k ) k -1 ≥ 1 /e ≥ 1 / 3 for all integers k &gt; 1 .

̸

Now we bound V π ⋆ γ, ̂ M ′ ( s 0 ) using concentration inequalities. For concreteness in the following application of Hoeffding we set k = 12 T hit ( P, π ⋆ ) so γ = 1 -1 / (12 T hit ( P, π ⋆ )) . By Hoeffding's inequality, we have for any s = s ⋆ that with probability at least 1 -δ ′

<!-- formula-not-decoded -->

and trivially we have ∣ ∣ ∣ e ⊤ s ⋆ ( ̂ P ′ π ⋆ -P ′ π ⋆ ) V π ⋆ γ, M ′ ∣ ∣ ∣ = 0 . Therefore by a union bound over all s ∈ S and setting δ ′ = δ S , we have with probability at least 1 -δ that

<!-- formula-not-decoded -->

where the second inequality uses the condition that n ( s, π ⋆ ( s )) ≥ 12 2 2 ( T hit ( P, π ⋆ , s ⋆ )) 2 log ( 2 S δ ) = 72( T hit ( P, π ⋆ )) 2 log ( 2 S δ ) for all s ∈ S .

Following standard arguments for the difference between two value functions with different transition matrices we have

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

Combining this with (40), we have that

<!-- formula-not-decoded -->

Using k = 12 T hit ( P, π ⋆ ) in (39) and combining with (38), we conclude that

<!-- formula-not-decoded -->

as desired.

Lemma B.18. Fix a deterministic unichain policy π ⋆ . Suppose that n ( s, π ⋆ ( s )) ≥ 1 + α (576 T hit ( P, π ⋆ )) 2 for all s ∈ S , where α = 8log ( 6 S 2 An tot (1 -γ ) δ ) . Then with probability at least 1 -2 δ ,

<!-- formula-not-decoded -->

Proof. By the definition (8) of ̂ T π ⋆ pe , we have for any s ∈ S that

<!-- formula-not-decoded -->

where we have defined ˜ b ′ ∈ R S as ˜ b ′ ( s ) = -max { ̂ P sπ ⋆ ( s ) ( T β ( s,π ⋆ ( s )) ( ̂ P sπ ⋆ ( s ) , ̂ V π ⋆ pe ) -̂ V π ⋆ pe ) -b ( s, π ⋆ ( s ) , ̂ V π ⋆ pe ) , min s ′ ( ̂ V π ⋆ pe )( s ′ ) -̂ P sπ ⋆ ( s ) ̂ V π ⋆ pe } .

Note that both terms within the max in the definition of ˜ b ′ ( s ) are ≤ 0 , so ˜ b ′ ≥ 0 , and also we can bound

<!-- formula-not-decoded -->

where ( i ) is due to the fact that ̂ P sπ ⋆ ( s ) T β ( s,π ⋆ ( s )) ( ̂ P sπ ⋆ ( s ) , ̂ V π ⋆ pe ) ≥ ̂ P sπ ⋆ ( s ) ̂ V π ⋆ pe -β ( s, π ⋆ ( s )) ∥ ∥ ∥ ̂ V π ⋆ pe ∥ ∥ ∥ span , which holds by an argument identical to that of (31), and ( ii ) holds since

<!-- formula-not-decoded -->

where we used Lemma B.14 and the fact that ∥ ∥ ∥ T β ( s,π ⋆ ( s )) ( ̂ P sπ ⋆ ( s ) , ̂ V π ⋆ pe ) ∥ ∥ ∥ span ≤ ∥ ∥ ∥ ̂ V π ⋆ pe ∥ ∥ ∥ span in the first inequality, then that V ̂ P sπ ⋆ ( s ) [ ̂ V π ⋆ pe ] ≤ ∥ ∥ ∥ ̂ V π ⋆ pe ∥ ∥ ∥ 2 span , and then bounded the max by the sum. (While Lemma B.14 is stated for β ( s, π ⋆ ( s )) ≤ 1 , if β ( s, π ⋆ ( s )) &gt; 1 then T β ( s,π ⋆ ( s )) ( ̂ P sπ ⋆ ( s ) , ̂ V π ⋆ pe ) is a constant vector so the bound is still true.)

Now since ˜ b ′ satisfies ̂ V π ⋆ pe = r π ⋆ -γ ˜ b ′ + γ ̂ P π ⋆ ̂ V π ⋆ pe , we can rearrange to obtain that ̂ V π ⋆ pe = ( I -γ ̂ P π ⋆ ) -1 ( r π ⋆ -γ ˜ b ′ ) . Likewise by the standard Bellman equation we have that V π ⋆ = r π ⋆ + γP π ⋆ V π ⋆ so V π ⋆ = ( I -γP π ⋆ ) -1 r π ⋆ . Then we can calculate that

<!-- formula-not-decoded -->

Now we can bound

<!-- formula-not-decoded -->

Fixing arbitrary s, s ′ ∈ S and letting ξ = e s -e s ′ , and using (42), we have that

<!-- formula-not-decoded -->

Next we bound all the terms in (44). First, ∥ ∥ ∥ ξ ⊤ ( I -γ ̂ P π ⋆ ) -1 ∥ ∥ ∥ 1 ≤ 4 T hit ( ̂ P,π ⋆ ) by Lemma B.11, and furthermore by Lemma B.17, since its conditions are satisfied under the conditions of the present lemma (since α ≥ log( 2 S δ ) ), we have with probability at least 1 -δ that T hit ( ̂ P,π ⋆ ) ≤ 24 T hit ( P, π ⋆ ) . Hence ∥ ∥ ∥ ξ ⊤ ( I -γ ̂ P π ⋆ ) -1 ∥ ∥ ∥ 1 ≤ 96 T hit ( P, π ⋆ ) . Next, for any s ∈ S , by Hoeffding's inequality, with probability at least 1 -δ ′ we have

<!-- formula-not-decoded -->

and so by a union bound over all s ∈ S and setting δ ′ = δ S , we have that with additional failure probability at most δ that

<!-- formula-not-decoded -->

Finally, using the bound (41), we have

<!-- formula-not-decoded -->

because our condition on n ( s, π ⋆ ( s )) guarantees that β ( s, π ⋆ ( s )) ≤ 1 so β ( s, π ⋆ ( s )) ≤ √ β ( s, π ⋆ ( s )) .

Combining these three bounds with (44), using that γ ≤ 1 , and taking the maximum over all s, s ′ , we have that

<!-- formula-not-decoded -->

Combining this with (43) and rearranging, we have that

<!-- formula-not-decoded -->

Noticing that 576 = 3 · 2 · 96 , our condition on n ( s, π ⋆ ( s )) in the lemma statement is chosen exactly so that

<!-- formula-not-decoded -->

Also since for all s ∈ S , β ( s, π ⋆ ( s )) = α max { n ( s,π ⋆ ( s )) -1 , 1 } = α n ( s,π ⋆ ( s )) -1 ≥ log( 2 S δ ) 2 n ( s,π ⋆ ( s )) (since α ≥ 8 log( 2 S δ ) and n ( s, π ⋆ ( s )) ≥ 4 so max { n ( s, π ⋆ ( s )) -1 , 1 } = n ( s, π ⋆ ( s )) -1 ≥ 1 2 n ( s, π ⋆ ( s )) ), we can also simply bound

<!-- formula-not-decoded -->

Wecan also bound 96 T hit ( P, π ⋆ ) 5 n tot ≤ 1 (by lower-bounding n tot by n ( s 0 , π ⋆ ( s 0 )) for one arbitrary s 0 ∈ S ). Combining all these bounds with (45), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies as desired.

## B.8 Average-reward-to-discounted reduction

Now we can combine our previous results and relate the discounted MDP quantities to ρ π ⋆ and h π ⋆ . Lemma B.19. There exist some absolute constants C 1 , C 2 such that the following holds: Fix a deterministic unichain policy π ⋆ . Suppose that n ( s, π ⋆ ( s )) ≥ mµ π ⋆ ( s ) + 4 + α (576 T hit ( P, π ⋆ )) 2 for all s ∈ S , where α = 8log ( 6 S 2 An tot (1 -γ ) δ ) , and that 1 1 -γ ≥ m and 1 1 -γ ≥ 2 . Then with probability at least 1 -5 δ , we have that

<!-- formula-not-decoded -->

Proof. By Lemma B.16 (the conditions of which are met here as α ( s, π ⋆ ( s )) (576 T hit ( P, π ⋆ )) 2 ≥ 4 T hit ( P, π ⋆ ) ), we have under the event of Lemma B.8, which holds with probability at least 1 -2 δ , that

<!-- formula-not-decoded -->

Combining this with the conclusion of Lemma B.18 which implies ∥ ∥ ∥ ̂ V π ⋆ pe ∥ ∥ ∥ span ≤ 3 ( ∥ ∥ V π ⋆ ∥ ∥ span +1 ) and adds additional failure probability at most 2 δ by the union bound, we have that

<!-- formula-not-decoded -->

For convenience abbreviate the right-hand-side of (46) as ε . Then since Q ̂ π ≥ ̂ Q by Lemma B.9 (which holds under the event of Lemma B.7, adding additional failure probability at most δ ) and ̂ Q ≥ ̂ Q ⋆ pe -1 2 n tot 1 by Lemma B.4, we have that

<!-- formula-not-decoded -->

Furthermore we have

<!-- formula-not-decoded -->

where ( i ) is due to Lemma B.1 which gives ̂ Q ⋆ pe ≥ ̂ Q π ⋆ pe , which implies ̂ V ⋆ pe = M ̂ Q ⋆ pe ≥ M π ⋆ ̂ Q ⋆ pe ≥ M π ⋆ ̂ Q π ⋆ pe using monotonicity of M π ⋆ . ( ii ) is due to (46), and ( iii ) uses ∥ ∥ ∥ V π ⋆ -1 1 -γ ρ π ⋆ ∥ ∥ ∥ ∞ ≤ ∥ ∥ V π ⋆ ∥ ∥ span due to Zurek and Chen [2025a, Lemma 6]. Also by Zurek and Chen [2025a, Lemma 6], we have the elementwise inequality ρ ̂ π ≥ (1 -γ ) ( min s ∈S V ̂ π ( s ) ) 1 . Thus

<!-- formula-not-decoded -->

where ( i ) uses (47), ( ii ) uses (48), ( iii ) uses the fact that ρ π ⋆ is assumed to be state-independent and the definition of ε (and canceling/simplifying), and ( iv ) uses that 1 1 -γ ≥ m (so (1 -γ ) ≤ 1 m ), that 1 -γ ≤ 1 , and n tot ≥ m .

Furthermore, using Zurek and Chen [2025a, Lemma 26] we have (since ρ π ⋆ is constant) that ∥ ∥ V π ⋆ ∥ ∥ span ≤ 2 ∥ ∥ h π ⋆ ∥ ∥ span . Combining this with the above bound and letting C 1 = 2 · 6144 / 8 , C 2 = 2 · 1933 / 8 , we obtain the desired bound.

## B.9 Completing the proof

Here we complete the proof of the main Theorem 3.2 by checking conditions and simplifying previous results. The following result is actually more general than Theorem 3.2 because it allows an arbitrary unichain deterministic comparator policy π ⋆ , rather than requiring π ⋆ to be gain-optimal. Theorem 3.2 follows immediately from the below theorem by adding this additional requirement that ρ π ⋆ = ρ ⋆ . Theorem B.20. There exist absolute constants C ′ 1 , C ′ 2 such that the following holds: Fix δ &gt; 0 . Let γ = 1 -1 n tot and α = 8log ( 6 S 2 An tot (1 -γ ) δ ) . Let π ⋆ be a deterministic policy which is unichain with stationary distribution µ π ⋆ . Suppose there exists some m ∈ N such that

<!-- formula-not-decoded -->

Then letting ̂ π be the policy returned by Algorithm 1 with inputs D , r , γ = 1 -1 n tot , and δ , we have with probability at least 1 -5 δ that

<!-- formula-not-decoded -->

Proof. Note that the condition on n implies that n tot ≥ 4 , so setting 1 1 -γ = n tot has 1 1 -γ ≥ 2 . Also we have

<!-- formula-not-decoded -->

using the assumption on n ( s, π ⋆ ( s )) for all s , so setting 1 1 -γ = n tot also ensures 1 1 -γ ≥ m . Therefore we can apply Lemma B.19 to obtain that if n ( s, π ⋆ ( s )) ≥ mµ π ⋆ ( s ) + 4 + α (576 T hit ( P, π ⋆ )) 2 for all s ∈ S , then with probability at least 1 -5 δ , we have

<!-- formula-not-decoded -->

where α = 8log ( 6 S 2 An tot (1 -γ ) δ ) = 8log ( 6 S 2 An 2 tot δ ) . Thus we can set C ′ 2 = 576 . To choose C ′ 1 , note that since trivially ρ π ⋆ ≤ 1 and ρ ̂ π ≥ 0 , if the term C 2 S (∥ ∥ ∥ h π ⋆ ∥ ∥ ∥ span +1 ) α m ≥ 1 then the bound

<!-- formula-not-decoded -->

holds vacuously, and otherwise if it is ≤ 1 then we have

<!-- formula-not-decoded -->

since √ x ≥ x for x ∈ [0 , 1] . Since √ a + √ b ≤ √ 2( a + b ) , we can take C ′ 1 = 2( C 1 + C 2 ) .

## C Proof of Theorem 3.3

Let T ≥ 4 and m ∈ N be arbitrary.

Step 1: MDP construction Define p = 1 3( m + T ) , A = ⌈ 16 pT ⌉ , and q = 1 AT . The set of states is S = { 0 , 1 } , and the set of actions is A = { 0 , 1 , . . . , A -1 } . The reward function r : S × A → [0 , 1] is defined by r (0 , a ) = 1 and r (1 , a ) = 0 for all a ∈ A . We define an index set Θ = { ( i, b ) ∣ ∣ ∣ i ∈ { 0 , 1 } , b ∈ { 0 , 1 , . . . , A -1 } } . For each θ = ( i, b ) ∈ Θ , we define the transition matrix P θ as follows:

̸

<!-- formula-not-decoded -->

Figure 2: Diagram of the MDP ( P (0 , 0) , r ) . Arrows splitting into multiple dashed arrows indicate stochastic transitions, and each dashed arrow is annotated with the associated probability. Blue arrows represent action 0 and red arrows represent action 1. In state 1, the red arrow also represents actions 2 , . . . , A -1 (which are all identical). The reward function does not depend on the action, and is +1 in state 0 and +0 in state 1 . In general, the MDP ( P ( i,b ) , r ) is similar, except that the blue arrow in state 0 represents action i and the blue arrow in state 1 represents action b .

<!-- image -->

See Figure 2 for a diagram of the MDP ( P θ , r ) for θ = (0 , 0) . We now state some easily verifiable facts about the MDP ( P θ , r ):

- The unique deterministic gain-optimal stationary policy π ⋆ θ is the one that takes action i in state 0 and action b in state 1.
- The optimal gain is ρ ∗ θ = 1 .
- µ π ⋆ θ θ (0) = 1 and µ π ⋆ θ θ (1) = 0 .
- The policy hitting radius T hit ( P θ , π ⋆ θ ) , the optimal bias span ∥ ∥ ∥ h π ⋆ θ P θ ∥ ∥ ∥ span , and the diameter are all at most T .
- Suppose a stationary policy π usually makes the wrong decisions - specifically π ( i | 0) &lt; 1 2 and π ( b | 1) &lt; 4 A . Then ρ π θ &lt; 4 A · 1 T +(1 -4 A ) q 4 A · 1 T +(1 -4 A ) q + p 2 ≤ 5 q 5 q + p 2 ≤ 5 p 16 5 p 16 + p 2 &lt; 1 2 . In words, our choice of A is one that is sufficiently large so that randomly guessing the optimal action b in state 1 will not yield a good policy.

Note that action 2 in state 0 is added to keep the diameter bounded by T , and actions 3 , . . . , A -1 in state 0 simply keep the action space independent of the state, consistent with our upper bounds. Since actions 2 , . . . , A -1 in state 0 are always suboptimal, whenever we consider some policy π , we will assume that π ( a | 0) = 0 for a ≥ 2 .

Step 2: dataset construction For any δ ∈ ( 0 , 1 e 9 ] , denote t δ = ⌈ T 6 log ( 1 δ )⌉ . We define n : S × A → N by n (0 , 0) = n (0 , 1) = m + t δ and n (1 , a ) = t δ for all a ∈ A . Observe that this choice of n satisfies the desired requirements. Indeed, since µ π ⋆ θ θ (0) = 1 and µ π ⋆ θ θ (1) = 0 , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Step 3: impossible to do well in all MDPs Suppose towards a contradiction that there exists an algorithm A that maps the dataset D to a stationary policy ˆ π = A ( D ) such that for all θ ∈ Θ , P θ,n ( ρ ˆ π θ &gt; 1 2 ) .

Before proceeding, we define some events. Let B be the bad event that D contains no transitions from state 0 to state 1 and no transitions from state 1 to state 0. Let E 0 be the event that ˆ π (0 | 0) ≥ 1 2 ( ˆ π prefers action 0 in state 0). Similarly, let E 1 be the event that ˆ π (1 | 0) ≥ 1 2 ( ˆ π prefers action 1 in state 0). For each a ∈ A , let F a be the event that ˆ π ( a | 1) ≥ 4 A ( ˆ π gives significant weight to action a in state 1).

A key idea is that under event B , the dataset is the same no matter the underlying MDP. That is, under event B , we always have

<!-- formula-not-decoded -->

It follows that for all θ, θ ′ ∈ Θ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For ease of notation, going forward we will drop the subscript θ, n when it does not matter what the underlying MDP is.

Since P ( E 0 ∪ E 1 | B ) = 1 , we must have P ( E i ′ | B ) ≥ 1 2 for some i ′ ∈ { 0 , 1 } . Furthermore, for some a ′ ∈ A we have P ( F a ′ | B ) ≤ 1 4 , or equivalently, P ( F c a ′ | B ) &gt; 3 4 . Indeed, if this were not the case, we would have

<!-- formula-not-decoded -->

which is a contradiction because we always have ∑ a ∈A ˆ π ( a | 1) = 1 .

We have shown that when the dataset does not contain any useful transitions, there must be at least one MDP where the algorithm is likely to make a poor guess. Our last step will be to combine this fact with Lemma C.1 which tells us that the dataset will be useless with large enough probability. We noted above that when the underlying MDP is ( P ( i ′ ,a ′ ) , r ) and a policy π satisfies π ( i ′ | 0) &lt; 1 2 and π ( a ′ | 1) &lt; 4 A we have ρ π ( i ′ ,a ′ ) &lt; 1 2 . In particular, under the the event E c i ′ ∩F c a ′ we have ρ ˆ π ( i ′ ,a ′ ) &lt; 1 2 . Subsequently, for θ ′ = ( i ′ , a ′ ) , we have

<!-- formula-not-decoded -->

where the final inequality follows from Lemma C.1.

In summary, we have shown that

<!-- formula-not-decoded -->

and as desired.

## C.1 Auxiliary lemmas

Lemma C.1. For all θ ∈ Θ , we have P θ,n ( B ) ≥ 4 δ .

Proof. By symmetry P θ ( B ) are equal for all θ , so for ease of notation we drop the subscript θ . Let B 0 be the event that D contains no transitions from state 0 to state 1, and let B 1 be the event that D contains no transitions from state 1 to state 0. Then

<!-- formula-not-decoded -->

with the last equality following by independence. Now,

<!-- formula-not-decoded -->

Recall that p = 1 3( m + T ) . In the case that m ≥ t δ , we have

<!-- formula-not-decoded -->

with the last inequality following from Lemma C.2 with x = 2 m and c = 3 . Otherwise, when m&lt;t δ , we have

<!-- formula-not-decoded -->

with the last inequality following from claim 3 of Lemma C.3 with x = 2 T . Combining Equations (49) and (50) and the fact that 4 δ 1 / 3 ≤ 1 e , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Claim 2 of Lemma C.3 with x = T gives us that ( 1 -1 T ) t δ ≥ δ 1 / 3 . Moreover, recalling that q = 1 AT , we have

<!-- formula-not-decoded -->

with the last inequality following from claim 2 of Lemma C.3 with x = AT . Hence, P ( B 1 ) ≥ δ 2 / 3 , and consequently, P ( B ) ≥ 4 δ .

Lemma C.2. For all x ≥ 2 and c ≥ 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality follows from log(1 -y ) ≥ -y -y 2 for y ∈ [0 , 0 . 68] . Since log x is monotonically increasing, we are done.

Lemma C.3. For any x ≥ 4 , the following holds:

1. For any δ ∈ ( 0 , 1 e ] , we have ( 1 -1 x ) ⌈ x 2 log ( 1 δ ) ⌉ ≥ δ .
2. For any δ ∈ ( 0 , 1 e 3 ] , we have ( 1 -1 x ) ⌈ x 6 log ( 1 δ ) ⌉ ≥ δ 1 / 3 .
3. For any δ ∈ ( 0 , 1 e 9 ] , we have ( 1 -1 3 x ) ⌈ x 6 log ( 1 δ ) ⌉ ≥ 4 δ 1 / 3 .

Next,

Proof. We have

Proof. We will prove claim 1 by showing that ( 1 -1 x ) x 2 log ( 1 δ ) +1 ≥ δ . For any x ≥ 4 and δ ∈ ( 0 , 1 e ] , we have

<!-- formula-not-decoded -->

where the first inequality follows from log(1 -y ) ≥ -y -y 2 for y ∈ [0 , 0 . 68] . Since log x is monotonically increasing, claim 1 follows.

For claim 2, take x ≥ 4 and δ ∈ ( 0 , 1 e 3 ] . Then δ ′ = δ 1 / 3 ∈ ( 0 , 1 e ] , so by claim 1 we have

<!-- formula-not-decoded -->

Finally, for claim 3, take x ≥ 4 and δ ∈ ( 0 , 1 e 9 ] , and let y = 3 x . Since δ ′ = δ 1 / 3 ∈ ( 0 , 1 e 3 ] , claim 2 gives us that

<!-- formula-not-decoded -->

where the last inequality holds because δ 1 / 3 &lt; 1 8 .

## D Proof of Theorem 3.4

We define the absolute constants c 1 = 4 and c 2 = 33 . Let T ≥ c 1 , S ≥ c 2 , k ≥ 0 , and m ≥ max { TS,kS } be arbitrary.

̸

Step 1: MDP construction Define S ′ = S -1 , D = T -2 , ε = 1 256 √ TS m . Note that ε ≤ 1 256 . Let p = 1 -ε D and q = 1 D . The set of states is S = { 0 , 1 , . . . , S ′ } and the set of actions is A = { 0 , 1 , . . . , S ′ } . The reward function r : S × A → [0 , 1] is defined to be 1 when s = 0 and a ≤ 1 , and 0 otherwise. We define an index set Θ = { 0 , 1 } S ′ . For each θ ∈ Θ , we define the transition matrix P θ as follows:

̸

| s     | a           | P θ ( s ′ &#124; s,a )                                        |
|-------|-------------|---------------------------------------------------------------|
| 0     | 0           | (1 - q ) I ( s ′ = s )+ q S ′ ∑ s ′′ ≥ 1 I ( s ′ = s ′′ )     |
| 0     | a ≥ 1       | (1 - q 2 ) I ( s ′ = s )+ q 2 S ′ ∑ s ′′ ≥ 1 I ( s ′ = s ′′ ) |
| s ≥ 1 | θ s         | (1 - p ) I ( s ′ = s )+ p I ( s ′ = 0)                        |
| s ≥ 1 | 1 - θ s     | (1 - q ) I ( s ′ = s )+ q I ( s ′ = 0)                        |
| s ≥ 2 | s           | 1 2 I ( s ′ = 1)+ 1 2 S ′ ∑ s ′′ ≥ 1 I ( s ′ = s ′′ )         |
| s ≥ 1 | a = s,a ≥ 2 | 1 2 I ( s ′ = a )+ 1 2 S ′ ∑ s ′′ ≥ 1 I ( s ′ = s ′′ )        |

Figure 3: Diagram of the MDP ( P (0 ,..., 0) , r ) only including actions 0 and 1. Arrows splitting into multiple dashed arrows indicate stochastic transitions, and each dashed arrow is annotated with the associated probability. Blue arrows represent action 0 and red arrows represent action 1. The reward is 0 at state 0 and the reward is 1 at all other states. In general, the MDP ( P θ , r ) is similar, except in each state s ≥ 1 , the blue arrow represents the optimal action θ s .

<!-- image -->

Observe that the decision-maker needs to decide between two actions in states 1 , . . . , S ′ . Both actions give an immediate reward of 1 , but one action has a slightly higher probability of transiting to the bad state 0. At state 0, which has a reward of 0, the agent will likely be trapped for a long time before returning to one of states 1 , . . . , S ′ . See Figure 3 for a diagram of the MDP ( P θ , r ) for θ = (0 , . . . , 0) . We now state some easily verifiable facts about the MDP ( P θ , r ) :

- The MDP has S states, is unichain, and has diameter 1 q + 1 1 / 2 = D +2 = T .
- There is a unique gain-optimal policy π ⋆ θ . It takes action 0 in state 0 and action θ s in state s for s ≥ 1 .
- µ π ⋆ θ θ (0) = p p + q = 1 -ε 2 -ε . By symmetry, it follows that µ π ⋆ θ θ ( s ) = 1 S ′ ( 1 -µ π ⋆ θ θ ) = 1 /S ′ 2 -ε for s ≥ 1 .
- The optimal gain is ρ ∗ θ = 1 -µ π ⋆ θ θ (0) = 1 2 -ε .

Note that actions 2 , . . . , S ′ for states s ≥ 1 are always suboptimal, and only exist to keep the diameter bounded by T . Furthermore, actions 1 , . . . , S ′ in state 0 simply keep the action space independent of the state, consistent with our upper bounds. As such, whenever we consider some policy π , we will assume that it may only take actions 0 and 1 in states s ≥ 1 and action 0 in state 0.

Step 2: dataset construction We define n : S × A → N by n (0 , 0) = m and

<!-- formula-not-decoded -->

for all s ≥ 1 and a ∈ { 0 , 1 } . For all other ( s, a ) we set n ( s, a ) = 0 . Observe that this choice of n satisfies n ( s, π ⋆ θ ( s )) = m S ′ + m S ′ ≥ mµ π ⋆ θ θ ( s ) + k for all s ∈ S .

Step 3: reduction to estimation Given a stationary policy π and some θ ∈ Θ , let L π θ ( s ) be the proportion of incorrect actions π takes in state s . To be precise, we define L π θ ( s ) = π (1 -θ s | s ) . We also set L π θ = ∑ S ′ s =1 L π θ ( s ) . By Lemma D.1, we can upper bound the gain of a policy π in terms of L π θ :

<!-- formula-not-decoded -->

Subsequently, for any stationary policy π ,

<!-- formula-not-decoded -->

Now, suppose the underlying MDP is ( P θ , r ) . Let A be an algorithm that maps the dataset to a stationary policy ˆ π = A ( D ) , and consider the estimator ˆ θ A whose s th coordinate is ˆ π (1 | s ) . By the definition of L ˆ π θ , we have L ˆ π θ = ∥ ∥ ∥ ˆ θ A -θ ∥ ∥ ∥ 1 . Our next step is to show that no estimator can achieve low ℓ 1 error uniformly over Θ with high probability, a result which will lower bound L ˆ π θ and consequently also the sub-optimality of ˆ π for some θ .

̸

Step 4: Fano's method We will achieve such a lower bound with Fano's method. First, by the Gilbert-Varshamov Lemma (Lemma D.2), there exists some subset Θ ′ ⊂ Θ such that | Θ ′ | ≥ 2 S ′ / 8 and ∥ θ -θ ′ ∥ 1 ≥ S ′ / 8 for any θ = θ ′ ∈ Θ ′ . Since max θ,θ ′ ∈ Θ ′ KL ( P θ,n ∥ P θ ′ ,n ) ≤ ( S ′ / 16 -1) log 2 by Lemma D.3, Local Fano's (Lemma D.4) gives us that for any estimator ˆ θ ,

<!-- formula-not-decoded -->

which implies that with c 3 = 2 -17 .

## D.1 Auxiliary lemmas

Lemma D.1. Let π be a stationary policy on MDP M θ . Then

<!-- formula-not-decoded -->

Proof. A routine computation (see Lemma D.7) yields

<!-- formula-not-decoded -->

where κ s = L π θ ( s ) q +(1 -L π θ ( s )) p = 1 -ε (1 -L π θ ( s )) D is the probability of transiting from state s to state 0 under π . Since x 1+ x is monotonically increasing for x &gt; -1 , to achieve the desired upper bound for ρ π θ it suffices to find an acceptable upper bound for λ := q S ′ ∑ S ′ s =1 1 κ s = 1 S ′ ∑ S ′ s =1 1 1 -ε (1 -L π θ ( s )) .

Defining f ( x ) = 1 1 -x and λ s = ε (1 -L π θ ( s )) , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the above holds for estimator of the dataset, it of course holds for ˆ θ A , where A is any algorithm that maps the dataset to a stationary policy. Therefore,

<!-- formula-not-decoded -->

Now, by Equation 51, in the event that L A ( D ) θ &gt; S ′ 64 ,

<!-- formula-not-decoded -->

with the second inequality holding by ε ≤ 1 256 . Thus, plugging back into Equation 52 yields

<!-- formula-not-decoded -->

We would like to get a bound that looks like λ ≤ f ( 1 S ′ ∑ S ′ s =1 λ s ) . This goal suggests applying Jensen's inequality, but since f is convex for x &lt; 1 it gives us an inequality in the wrong direction. It turns out, however, that because f is nearly linear in the sufficiently small interval of interest, we can obtain an inequality in the right direction with some error term of lower order.

Since λ s ∈ [0 , ε ] for all s ∈ { 1 , . . . , S ′ } , Lemma D.6 give us

<!-- formula-not-decoded -->

where the last inequality holds for ε &lt; 1 3 . Consequently,

<!-- formula-not-decoded -->

̸

Lemma D.2 (Gilbert-Varshamov Lemma [Massart, 2007, Lemma 4.7]) . Let d ≥ 8 . There exists Ω d ⊂ { 0 , 1 } d such that | Ω d | ≥ 2 d/ 8 and ∥ ω -ω ′ ∥ 1 ≥ d/ 8 for all ω = ω ′ ∈ Ω d .

Lemma D.3. For any θ, θ ′ ∈ Θ , we have

<!-- formula-not-decoded -->

Proof. Let θ, θ ′ ∈ Θ . By the construction of P θ,n and P θ ′ ,n , we can decompose

<!-- formula-not-decoded -->

Recalling our choice of n , we can further simplify

<!-- formula-not-decoded -->

where we remove the s = 0 term from the sum because the data coming from state 0 has the same distribution for all possible MDPs. Observing that

<!-- formula-not-decoded -->

we can apply Lemma D.5 to further simplify

<!-- formula-not-decoded -->

The final inequality holds due to the assumption that S ≥ 33 = ⇒ S ′ ≥ 32 .

̸

Lemma D.4 (Local Fano's inequality [Wainwright, 2019, Proposition 15.12, Equation 15.34]) . Let P be a class of distributions with parameter space Θ , and let { P 1 , . . . , P N } ⊂ P . Letting θ ( P ) ∈ Θ denote the parameters of P , define δ = min j = k ∥ θ ( P j ) -θ ( P k ) ∥ 1 . For any estimator ˆ θ , we have

<!-- formula-not-decoded -->

Lemma D.5. For any p, q ∈ ( 0 , 1 2 ] satisfying p &lt; q , we have

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

Proof. By Lemma 10 in Li et al. [2023], we have

<!-- formula-not-decoded -->

for any p ′ , q ′ ∈ [ 1 2 , 1 ) satisfying p ′ &gt; q ′ . The desired result follows immediately by taking p ′ = 1 -p and q ′ = 1 -q , along with the observation that KL ( Ber (1 -p ) ∥ Ber (1 -q )) = KL ( Ber ( p ) ∥ Ber ( q )) .

Lemma D.6 (Theorem 1 in Simic [2008]) . Let I = [ a, b ] be a closed interval with a, b ∈ R , a &lt; b . For some n ∈ Z + , let x 1 , . . . , x n ∈ I , and let p 1 , . . . , p n &gt; 0 satisfy ∑ n i =1 p i = 1 . If f : [ a, b ] → R is convex, then

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

Lemma D.7. Suppose the underlying MDP is ( P θ , r ) . Let π be a stationary policy such that for each s = 0 , if the current state is s then the probability of transiting to state 0 after taking action according to π is κ s . Then

̸

Proof. We first solve for µ π θ (0) by considering the balance equations for the MDP ( P θ , r ) . For each s = 0 , we have

Rearranging gives us

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ∑ S ′ s =0 µ π θ ( s ) = 1 , we have

<!-- formula-not-decoded -->

We then solve for µ π θ (0) to obtain

<!-- formula-not-decoded -->

Since the reward is 0 in state 0 and 1 in all other states, we conclude that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E Deferred proofs and auxiliary lemmas

## E.1 Proof of Lemma B.2

Proof of Lemma B.2. Letting V, V ′ ∈ R S satisfy V ≥ V ′ elementwise, we seek to show that

<!-- formula-not-decoded -->

Since this is an elementwise bound, we can fix arbitrary s ∈ S , a ∈ A and show that T pe ( V )( s, a ) ≥ T pe ( V ′ )( s, a ) . From here on, since s, a are fixed, we abbreviate β ( s, a ) ∈ R as β for notational convenience.

Consider the simpler function ˜ T : R S → R (which depends on our fixed s, a ) defined as

<!-- formula-not-decoded -->

for any V ′′ ∈ R S . Note that

<!-- formula-not-decoded -->

Therefore, if we could show that

<!-- formula-not-decoded -->

then since clearly V ≥ V ′ implies min s ′ ( V )( s ′ ) ≥ min s ′ ( V ′ )( s ′ ) , we could immediately conclude that

<!-- formula-not-decoded -->

as desired.

Thus we now focus on showing (53). First we can quickly handle the case that β &gt; 1 , since in this case for any V ′′ ∈ R S we have T β ( ̂ P sa , V ′′ ) = (min s ′ V ′′ ( s ′ )) 1 , and then

<!-- formula-not-decoded -->

confirming (53). Now we can focus on the case that β ≤ 1 .

The fact that β ≤ 1 means that the following expression for T β holds: for any s ′ ∈ S and V ′′ ∈ R S , we have

<!-- formula-not-decoded -->

where Q β ( ̂ P sa , V ′′ ) = sup { V ′′ ( x ) : x ∈ S , ∑ x ′ ∈S : V ( x ′ ) ≥ V ( x ) ̂ P sa ( x ′ ) ≥ β } is the 1 -β quantile of V ′′ with respect to ̂ P sa (in words, we choose the largest V ′′ ( x ) such that ̂ P sa places probability at

least β on states x ′ with V ′′ ( x ′ ) ≥ V ′′ ( x ) ). We will make use of the function Q β shortly. We also make the useful definitions

<!-- formula-not-decoded -->

so that we can decompose ˜ T as ˜ T ( V ) = min { ˜ T 1 ( V ) , ˜ T 2 ( V ) } . To show (53), it suffices to show that this holds when V and V ′ differ in only one coordinate, since then we could decompose V = V ′ + ∑ s ′ ∈S e s ′ e ⊤ s ′ ( V -V ′ ) and apply the inequalities ˜ T ( V ′ + ∑ k -1 s ′ =1 e s ′ e ⊤ s ′ ( V -V ′ ) ) ≤ ˜ T ( V ′ + ∑ k s ′ =1 e s ′ e ⊤ s ′ ( V -V ′ ) ) for each k = 1 , . . . , S . Therefore we fix one state x ∈ S and try to show ˜ T ( V ) is montonically non-decreasing as V ( x ) increases (with the other entries of V held constant). We will show this by using Lemma E.1, which says that if a univariate function is continuous and at all but a finite number of points has a non-negative right derivative, then it must be non-decreasing.

First we justify that ˜ T is continuous. Since we have decomposed ˜ T as the composition of many continuous functions, it suffices to check that Q β ( ̂ P sa , V ) is a continuous function of V ( x ) . This follows immediately from Lemma E.3, which shows 1 -Lipschitzness. (We remark that the 1 -β quantile is well-known to be discontinuous in β , a fact which is irrelevant here since β is fixed and we instead vary V ( x ) .)

̸

We will now compute the right derivative at all values of V ( x ) such that V ( x ) is not equal to V ( s ′ ) for some other s ′ ∈ S with s ′ = x (which is a finite set). We define some new notation for this purpose. With respect to this fixed value of V ( x ) , let S &gt; = { s ′ ∈ S : V ( s ′ ) &gt; V ( x ) } and S &lt; = { s ′ ∈ S : V ( s ′ ) &lt; V ( x ) } . Define a neighborhood of V ( x ) , the open interval U := (max s ′ ∈S &lt; V ( s ′ ) , min s ′ ∈S &gt; V ( s ′ )) . Let V ′ ∈ R S have V ′ ( s ′ ) = V ( s ′ ) for all s ′ = x , and we vary V ′ ( x ) within the neighborhood U of V ( x ) in order to compute the (full/two-sided) derivatives d ˜ T 1 ( V ′ ) dV ′ ( x ) ∣ ∣ ∣ V ′ ( x )= V ( x ) and d ˜ T 2 ( V ′ ) dV ′ ( x ) ∣ ∣ ∣ V ′ ( x )= V ( x ) . Once we have computed these two derivatives, we will be able to compute the right derivative of ˜ T ( V ′ ) , since if both ˜ T 1 ( V ′ ) and ˜ T 2 ( V ′ ) are differentiable at a point V ( x ) , then by Lemma E.2 the right derivative of ˜ T ( V ′ ) satisfies

̸

<!-- formula-not-decoded -->

To compute the derivatives of ˜ T 1 ( V ′ ) and ˜ T 2 ( V ′ ) , we also analyze the functions Q β ( ̂ P sa , V ′ ) and T β ( ̂ P sa , V ′ ) on the set U (all considered as functions of V ′ ( x ) ). For any set S ′ ⊆ S , let ̂ P sa ( S ′ ) = ∑ s ′ ∈S ′ ̂ P sa ( s ′ ) . We define three possible cases depending on the (fixed) state x :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

1. In case (55), we have Q β ( ̂ P sa , V ′ ) = Q β ( ̂ P sa , V ) on the entire interval U and also that for any V ′ ( x ) ∈ U , Q β ( ̂ P sa , V ) &gt; V ′ ( x ) (since the (1 -β ) -percentile is achieved at some

̸

state s ′ ∈ S &gt; ), so T β ( ̂ P sa , V ′ )( x ) = V ′ ( x ) and T β ( ̂ P sa , V ′ )( s ′ ) = T β ( ̂ P sa , V )( s ′ ) for all s ′ = x . Therefore

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

2. In case (56), we have Q β ( ̂ P sa , V ′ ) = V ′ ( x ) on the entire interval U . Thus T β ( ̂ P sa , V ′ )( s ′ ) = V ′ ( x ) if s ′ ∈ S &gt; ∪ { x } , and T β ( ̂ P sa , V ′ )( s ′ ) = V ′ ( s ′ ) = V ( s ′ ) for s ′ ∈ S &lt; . Thus

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

3. In case (57), we have Q β ( ̂ P sa , V ′ ) = Q β ( ̂ P sa , V ) and also that T β ( ̂ P sa , V ′ )( x ) = Q β ( ̂ P sa , V ) &lt; V ′ ( x ) (since V ′ ( x ) &lt; Q β ( ̂ P sa , V ) in this case), so T β ( ̂ P sa , V ′ ) = T β ( ̂ P sa , V ) on the interval U . Also min s ′ V ′ ( s ′ ) &lt; V ′ ( x ) on U , so min s ′ V ′ ( s ′ ) = min s ′ V ( s ′ ) on U . Thus

<!-- formula-not-decoded -->

for all s ′ ∈ S , and

<!-- formula-not-decoded -->

̸

Next we calculate d ˜ T 2 ( V ′ ) dV ′ ( x ) ∣ ∣ ∣ ∣ V ′ ( x )= V ( x ) . First, letting T ∈ R S , if V ̂ P sa [ T ] = 0 then (recalling ̂ P sa is a row vector so ̂ P ⊤ sa is a column vector)

̸

<!-- formula-not-decoded -->

where the final inequality is elementwise and uses the fact that for any s ′ , T ( s ′ ) -̂ P sa T ≤ max s ′′ T ( s ′′ ) -min s ′′ T ( s ′′ ) = ∥ T ∥ span . Now we will combine this calculation with the chain rule to lower bound d ˜ T 2 ( V ′ ) dV ′ ( x ) ∣ ∣ ∣ V ′ ( x )= V ( x ) . Note that in light of (54), we only need to bound d ˜ T 2 ( V ′ ) dV ′ ( x ) ∣ ∣ ∣ V ′ ( x )= V ( x ) when ˜ T 1 ( V ) &gt; ˜ T 2 ( V ) or equivalently when our fixed value of V ( x ) satisfies

<!-- formula-not-decoded -->

Since we have already excluded the finite set of values of V ( x ) where V ( x ) is equal to V ( s ′ ) for some other state s ′ = x , the only way for V ̂ P sa [ T β ( ̂ P sa , V ) ] = 0 is if ̂ P sa ( x ) = 1 , but in that case we have ∥ ∥ ∥ T β ( ̂ P sa , V ) ∥ ∥ ∥ span = 0 which contradicts (59). Therefore we can calculate that if V ( x ) satisfies (59), we have

<!-- formula-not-decoded -->

where the first inequality step is using the fact that dT β ( ̂ P sa ,V ′ )( s ′ ) dV ′ ( x ) ∣ ∣ ∣ ∣ V ′ ( x )= V ( x ) ≥ 0 for all s ′ (verified above in all three cases) and inequality (58), and the second inequality step uses (59).

## E.2 Auxiliary lemmas

Lemma E.1. If f : R → R is a continuous function that has a nonnegative right derivative for all but finitely many points, then f is monotonically non-decreasing.

Proof. We make the following claim: for a, b ∈ R with a &lt; b , if f : [ a, b ] → R is continuous on [ a, b ] and has a nonnegative right derivative on ( a, b ) , then f is monotonically non-decreasing on [ a, b ] .

We first prove the lemma assuming that the claim holds. Let f : R → R be a continuous function that has a nonnegative right derivative for all but finitely many points. Let x, y ∈ R satisfy x &lt; y , and denote by a 1 , . . . , a n -1 the points in ( x, y ) where f either is not right-differentiable or has negative right derivative. Also denote a 0 = x and a n = y . By the claim, f is monotonically increasing on [ a i -1 , a i ] for each i = 1 , . . . , n . Hence f ( x ) = f ( a 0 ) ≤ f ( a 1 ) ≤ · · · ≤ f ( a n ) = f ( y ) . Since x and y were arbitrary, we conclude that f is monotonically increasing.

It remains to prove the claim. Let a, b ∈ R with a &lt; b , and let f : [ a, b ] → R be continuous on [ a, b ] with a nonnegative right derivative on ( a, b ) . Suppose towards a contradiction that there exist x, y ∈ [ a, b ] such that x &lt; y and f ( x ) &gt; f ( y ) . Since f is continuous, we can assume that x &gt; a (if x = a we have x + δ &lt; y and f ( x + δ ) &gt; f ( y ) for sufficiently small δ &gt; 0 ).

Now, set r := f ( y ) -f ( x ) y -x &lt; 0 and

<!-- formula-not-decoded -->

Consider the case where z = x . f has a nonnegative right derivative at x , so there exists w ∈ ( x, y ] such that f ( t ) -f ( x ) t -x &gt; r 2 for all t ∈ ( x, w ] . However, this implies a contradiction:

<!-- formula-not-decoded -->

We next consider the case where z &gt; x . Note that by continuity of f , the function g ( t ) := f ( t ) -f ( x ) t -x is continuous on ( x, y ] . It follows that g ( z ) = f ( z ) -f ( x ) z -x = r 2 . Indeed, if we had g ( z ) &gt; r 2 , then by continuity of g there would exist δ &gt; 0 such that g ( t ) &gt; r 2 for t ∈ [ z, z + δ ] , which would imply that z ≥ z + δ . And by a similar argument, g ( z ) &lt; r 2 would imply z ≤ z -δ .

At z the right-derivative is nonnegative, so there exists w ∈ ( z, y ] such that f ( t ) -f ( z ) t -z &gt; r 2 for all t ∈ ( z, w ] . Consequently, for all t ∈ ( z, w ] , we have

<!-- formula-not-decoded -->

which implies the following contradiction:

<!-- formula-not-decoded -->

Lemma E.2. Let f, g : R → R be differentiable at some x ∈ R , and suppose f ( x ) = g ( x ) . Then ϕ : R → R defined by ϕ ( t ) = min { f ( t ) , g ( t ) } is right-differentiable at x , and its right derivative satisfies ϕ ′ + ( x ) = min { f ′ ( x ) , g ′ ( x ) } .

Proof. We first consider the case where f ′ ( x ) &lt; g ′ ( x ) . Since lim h → 0 f ( x + h ) -f ( x ) h &lt; lim h → 0 g ( x + h ) -g ( x ) h , there exists some δ &gt; 0 such that f ( x + h ) -f ( x ) h &lt; g ( x + h ) -g ( x ) h for all h ∈ (0 , δ ) . Subsequently, since f ( x ) = g ( x ) , we have f ( x + h ) &lt; g ( x + h ) for all h ∈ (0 , δ ) . It follows that ϕ ( x + h ) = f ( x + h ) for all h ∈ (0 , δ ) , and thus

<!-- formula-not-decoded -->

Next, the case where f ′ ( x ) &gt; g ′ ( x ) is identical to the previous case except we swap the roles of f and g .

Finally, we consider the case where f ′ ( x ) = g ′ ( x ) . Here we can even show that ϕ is differentiable at x . Let { h n } n ∈ N be a sequence such that h n → 0 . To show that ϕ ( x + h n ) -ϕ ( x ) h n → f ′ ( x ) , fix ε &gt; 0 . Since f ( x + h n ) -f ( x ) h n → f ′ ( x ) and g ( x + h n ) -g ( x ) h n → g ′ ( x ) , there exist N 1 , N 2 ∈ N such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Taking N = max { N 1 , N 2 } , we have for all n ≥ N ,

<!-- formula-not-decoded -->

where the first inequality holds due to f ( x ) = g ( x ) , f ′ ( x ) = g ′ ( x ) , and the fact that for each n , either ϕ ( x + h n ) = f ( x + h n ) or ϕ ( x + h n ) = g ( x + h n ) . Thus, we have that ϕ ( x + h n ) -ϕ ( x ) h n → f ′ ( x ) . Since the sequence { h n } n ∈ N was arbitrary, we conclude that

<!-- formula-not-decoded -->

Lemma E.3. For any probability distribution µ ∈ R S and any β ∈ [0 , 1] , the largest(1 -β ) -quantile function

<!-- formula-not-decoded -->

satisfies

<!-- formula-not-decoded -->

for any V, V ′ ∈ R S .

Proof. First, we note that the definition of Q β can be written equivalently as

<!-- formula-not-decoded -->

Without loss of generality we can assume that Q β ( µ, V ) ≥ Q β ( µ, V ′ ) , so it suffices to lower-bound Q β ( µ, V ′ ) . By the definition of Q β ( µ, V ) (and the fact that S is finite so the supremum within its definition is attained exactly), there exists some set S ′ ⊆ S such that

<!-- formula-not-decoded -->

and ∑ s ′ ∈S ′ µ ( s ′ ) ≥ β . Therefore since

<!-- formula-not-decoded -->

for all s ′ ∈ S ′ , we have that as desired.

<!-- formula-not-decoded -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: All claims made in the introduction and abstract are substantiated in Section 3.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations of the work are discussed in the conclusion in Section 5.

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

Justification: All theorems state all required assumptions, and proofs for all formal results are provided in the appendices.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: There are no experimental results.

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

Answer: [NA]

Justification: There are no experiments requiring code.

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

Answer: [NA]

Justification: There are no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: There are no experiments.

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

Justification: There are no experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our paper does not have any direct negative societal impacts nor any potential harms caused by the research process.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our paper is focused on theoretical aspects of offline RL and therefore there are no immediate negative societal impacts.

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

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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