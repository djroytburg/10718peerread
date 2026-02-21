## Incentive-Aware Dynamic Resource Allocation under Long-Term Cost Constraints

Yan Dai Operations Research Center MIT yandai20@mit.edu

Negin Golrezaei Sloan School of Management MIT golrezae@mit.edu

## Abstract

Motivated by applications such as cloud platforms allocating GPUs to users or governments deploying mobile health units across competing regions, we study the constrained dynamic allocation of a reusable resource to a group of strategic agents. Our objective is to simultaneously (i) maximize social welfare, (ii) satisfy multidimensional long-term cost constraints, and (iii) incentivize truthful reporting. We begin by numerically evaluating primal-dual methods widely used in constrained online optimization and find them to be highly fragile in strategic settings - agents can easily manipulate their reports to distort future dual updates for future gain. To address this vulnerability, we develop an incentive-aware framework that makes primal-dual methods robust to strategic behavior. Our primal-side design combines epoch-based lazy updates - discouraging agents from distorting dual updates - with dual-adjust pricing and randomized exploration techniques that extract approximately truthful signals for learning. On the dual side, we design a novel online learning subroutine to resolve a circular dependency between actions and predictions; this makes our mechanism achieve ˜ O ( √ T ) social welfare regret (where T is the number of allocation rounds), satisfies all cost constraints, and ensures incentive alignment. This ˜ O ( √ T ) performance matches that of non-strategic allocation approaches while additionally exhibiting robustness to strategic agents.

## 1 Introduction

Modern platforms and public agencies often face the challenge of allocating limited, reusable resources over time to self-interested agents, who may hide their true desire in sake of more favorable allocations. For example, cloud providers must decide how to distribute scarce GPUs to competing jobs under compute and energy constraints (Buyya et al., 2008; Nejad et al., 2014). Governments may deploy mobile health units or medical devices such as ventilators across regions, where needs vary over time and access is constrained by staffing or capacity (Stummer et al., 2004; Adan et al., 2009). In these settings, a same unit of resource is reallocated across time, and the allocation must respect some multi-dimensional long-term cost constraints - such as energy or staffing - while accounting for the strategic behavior of agents with private information.

A central challenge in these dynamic environments is being both efficient -i.e. , maximizing social welfare subject to constraints - and robust to agents' strategic manipulation. Focusing on efficiency, primal-dual methods serve as a powerful tool in online resource allocation, offering principled ways to handle constraints while adapting to changing demand (Devanur and Hayes, 2009; Golrezaei et al., 2014; Molinaro and Ravi, 2014; Balseiro et al., 2023). These methods maintain dual variables that act as shadow prices on resource usage, guiding allocations based on both values and costs. However, these approaches typically assume truthful agents and ignore any strategic responses agents make.

Patrick Jaillet Department of EECS MIT jaillet@mit.edu

Figure 1: We simulate a T -round game for 1000 trials, during which agents use Q-learning to optimize their reporting strategy. Under the vanilla primal-dual algorithm of Balseiro et al. (2023), agents learn to frequently misreport their values, resulting in reduced social welfare and low budget utilization (blue). In contrast, under our incentive-aware mechanism, agents gradually learn to report truthfully, leading to significantly improved social welfare while adhering to cost constraints (green).

<!-- image -->

Indeed, as we illustrate numerically in Figure 1, classical primal-dual mechanisms are highly vulnerable to manipulation (see Section 5 for the detailed setup): Strategic agents game the learning process by distorting their current reports to influence future dual updates, thereby improving their individual utility but giving low budget utilization and less welfare. This fragility raises a natural question:

With strategic agents, is it still possible to optimize social welfare subject to long-term constraints?

To our knowledge, the only prior work addressing strategic agents in constrained online allocation is that of Yin et al. (2022). While their framework is a valuable step, it has two key limitations. First, it focuses on homogeneous agents with identical value distributions - an assumption critical for their equilibrium argument, but unrealistic in many applications. Second, their mechanism focuses on a specific type of cost constraint based on some 'fair share' per agent, which requires knowing ideal allocation proportions in advance. These assumptions limit the practical applicability of their results. Due to space limitations, more discussions on related works are postponed to Appendix A.

This paper goes beyond these limitations and yields an incentive-aware primal-dual framework - one that is robust to strategic manipulation. To limit agents' influence on the future, we stabilize dual updates through epoch-based lazy updates (fixing dual variables within each epoch), which reduce the impact of any individual report on future duals and allocations. To further deter manipulation, we combine dual-adjusted pricing rounds with randomized exploration which imposes immediate utility loss on untruthful agents, thereby creating localized incentives for truthful reporting. We show that, when the dual variables are updated via the classical Follow-the-Regularized-Leader (FTRL) algorithm, our mechanism (i) achieves sublinear regret of ˜ O ( T 2 / 3 ) w.r.t. offline optimal allocations, where T is the number of rounds, (ii) satisfies all resource constraints exactly, and (iii) admits a Perfect Bayesian Equilibrium (PBE) in which agents have no incentives to misreport in most rounds.

We show that FTRL is unable to get o ( T 2 / 3 ) regret; however, we observe that property (iii) allows the planner to treat historical reports as reliable estimates of true values, thus enabling optimistic predictions of future outcomes. Building on this, we introduce a novel online learning algorithm - Optimistic FTRL with Fixed Points (O-FTRL-FP) - which solves a small number of fixed-point problems across the time horizon to incorporate such predictive structure. 1 This gives an improved regret bound of ˜ O ( √ T ) , matching the Ω( √ T ) lower bound for non-strategic constrained allocation (Arlotto and Gurvich, 2019). In doing so, we bridge the gap between online constrained optimization and dynamic mechanism design, enabling robust decision-making in complex, strategic environments.

## 2 Preliminaries

Notations. For an integer n ≥ 1 , [ n ] denotes the set { 1 , 2 , . . . , n } . For a set X , the probabilistic simplex △ ( X ) contains all probability distributions over X . We use bold letters like v to denote

1 Due to a circular dependency between actions and predictions, the O-FTRL framework (Rakhlin and Sridharan, 2013) is not applicable. Our newly proposed O-FTRL-FP framework resolves this issue, and we expect it to be of independent interest to the online learning literature; see Section 4.2.4 for more context.

| Protocol 1 Interaction Protocol for Repeated Resource Allocation                                                                                                                                                             | Protocol 1 Interaction Protocol for Repeated Resource Allocation                                                                                                                                                             |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Input: Number of rounds T , number of agents K , value distributions {V i } i ∈ [ K ] , cost distributions {C i } i ∈ [ K ] , mechanism M = ( M t ) t ∈ [ T ] , agents' strategy profile π = ( π t,i ) t ∈ [ T ] , i ∈ [ K ] | Input: Number of rounds T , number of agents K , value distributions {V i } i ∈ [ K ] , cost distributions {C i } i ∈ [ K ] , mechanism M = ( M t ) t ∈ [ T ] , agents' strategy profile π = ( π t,i ) t ∈ [ T ] , i ∈ [ K ] |
| 1:                                                                                                                                                                                                                           | Initialize public history: H 1 , 0 ← ∅                                                                                                                                                                                       |
| 2:                                                                                                                                                                                                                           | Initialize private history for each agent i : H 1 ,i ←{ ( V j , C j ) } j ∈ [ K ]                                                                                                                                            |
| 3:                                                                                                                                                                                                                           | for each round t = 1 , 2 , . . .,T do                                                                                                                                                                                        |
| 4:                                                                                                                                                                                                                           | Each agent i ∈ [ K ] observes: • Private value: v t,i ∼ V i                                                                                                                                                                  |
| 5:                                                                                                                                                                                                                           | • Public cost vector: c t,i ∼ C i Agent i submits report: u t,i ∼ π t,i ( v t,i , c t ; H t,i )                                                                                                                              |
| 6:                                                                                                                                                                                                                           | Planner applies mechanism: ( i t , p t,i t ) ∼ M t ( u t , c t ; H t, 0 )                                                                                                                                                    |
| 7:                                                                                                                                                                                                                           | Update public history: add ( u t , c t , i t , p t,i t ) to H t +1 , 0                                                                                                                                                       |
| 8:                                                                                                                                                                                                                           | Each agent i updates private history: add v t,i and ( u t , c t , i t , p t,i t ) to H t +1 ,i                                                                                                                               |

a vector, and use normal letters like v i for an element therein. For a random variable X , we use PDF( X ) to denote its probability density function (PDF). We use O to hide all absolute constants, ˜ O to additionally hide all logarithmic factors, and ˜ O T to focus on the polynomial dependency of T .

Setup. We consider the problem of allocating indivisible resources over T rounds from a central planner to K strategic agents, indexed by 1 , 2 , . . . , K . In each round, the planner allocates a single indivisible resource to one of the agents, aiming to maximize social welfare while satisfying d long-term cost constraints simultaneously; more details on this objective can be found in Section 2.3.

## 2.1 Agents' Values and Costs, Planner's Allocation and Payment

In each round t ∈ [ T ] , agent i ∈ [ K ] has a private scalar value v t,i ∈ [0 , 1] and a public d -dimensional cost vector c t,i ∈ [0 , 1] d . Allocating the resource to agent i in round t yields a value of v t,i to the agent and incurs c t,i,j units of cost along dimension j for all j ∈ [ d ] . We assume that values and costs are independent across agents and rounds; specifically, v t,i and c t,i are i.i.d. samples from fixed but unknown distributions V i ∈ △ ([0 , 1]) and C i ∈ △ ([0 , 1] d ) , respectively, for all t ∈ [ T ] and i ∈ [ K ] .

Every agent i ∈ [ K ] , after observing their own private value v t,i , strategically generates a report u t,i ∈ [0 , 1] which may differ from v t,i . We defer the generation rule of such reports to Section 2.2. After observing agents' strategic reports u t and cost vectors c t (but without access to the true values v t ), the planner either irrevocably allocates the resource to one of the agents i t ∈ [ K ] or forfeits it.

̸

After the allocation, the planner decides a payment charged from the winner i t , denoted by p t,i t . For all remaining agents i = i t , the payment p t,i = 0 . The planner maximizes the T -round cumulative social welfare ∑ T t =1 v t,i t subject to long-term constraints that the T -round average costs are no more than a pre-specified threshold ρ ∈ [0 , 1] d , i.e. , 1 T ∑ T t =1 c t,i t ≤ ρ , where ≤ is compared element-wise.

## 2.2 History, Planner's Mechanism, and Agents' Strategies

At the beginning of round t ∈ [ T ] , the public history is given by H t, 0 := { ( u τ , c τ , i τ , p τ,i τ ) } τ&lt;t . Each agent i ∈ [ K ] additionally has access to their own past values and all agents' value and cost distributions. Thus, the private history available to agent i at the beginning of round t is 2

<!-- formula-not-decoded -->

In each round t ∈ [ T ] , the planner determines the allocation and payment ( i t , p t,i t ) based on agents' reports u t , cost vectors c t , public history H t, 0 , and possibly some internal randomness used to break ties or randomize decisions. We write i t = p t,i t = 0 when the allocation is forfeited. Formally,

<!-- formula-not-decoded -->

2 We assume the distributional information is known across the agents since such information (or at least some prior) is necessary for the definition of Perfect Bayesian Equilibrium (PBE) in Definition 2. We adopt this mutually known setup as it is most challenging for the planner in terms of information asymmetry.

The collection of decision rules M = ( M t ) t ∈ [ T ] is referred to as the planner's mechanism . We emphasize that the planner does not know the agents' value or cost distributions, whereas the mechanism M is publicly known to the agents, as is standard in the literature.

For each agent i ∈ [ K ] , their report u t,i is determined based on their private value v t,i , the cost vector c t , their private history H t,i , and potentially some internal randomness. Formally, we write

<!-- formula-not-decoded -->

Agent i 's decision rules collectively form their strategy π i := ( π t,i ) t ∈ [ T ] . The agents' strategies together constitute a joint strategy π := ( π i ) i ∈ [ K ] . We summarize the interaction as Protocol 1.

## 2.3 Agents' Behavior and Planner's Regret

To model agents' behavior in a dynamic environment, we adopt the γ -impatient agent framework introduced by Golrezaei et al. (2021a, 2023) which captures the idea that agents often prioritize immediate rewards over long-term gains - due to bounded rationality, uncertainty about future rounds, or limited planning horizons - while the planner is more patiently optimizing the long-run welfare.

Assumption 1 ( γ -Impatient Agents) . For some fixed constant γ ∈ (0 , 1) unknown to the planner, every agent i ∈ [ K ] is γ -impatient in the sense that they maximize their γ -discounted T -round gain 3

<!-- formula-not-decoded -->

In this paper, we study the equilibrium concept of Perfect Bayesian Equilibrium (PBE):

Definition 2 (Perfect Bayesian Equilibrium) . Fix a mechanism M = ( M 1 , M 2 , . . . , M T ) . An agents' joint strategy π is a Perfect Bayesian Equilibrium (PBE) under M , if any single agent's unilateral deviation from π does not increase their own gain. Formally, a joint strategy π ∈ Π is a PBE if

<!-- formula-not-decoded -->

The planner aims to maximize social welfare, namely the expected total value yielded from allocations E [ ∑ T t =1 v t,i t ] , while satisfying the d long-term cost constraints 1 T ∑ T t =1 c t,i t ≤ ρ simultaneously. To evaluate a mechanism's performance, we compare the allocations against the following offline optimal benchmark , which performs a hindsight optimization using agents' true values and costs:

<!-- formula-not-decoded -->

We remark that benchmark in Eq. (2) depends on the full sequence of true values { v t } T t =1 and costs { c t } T t =1 , which are not observable to the planner. This distinguishes it from typical online learning benchmarks, which fixes a policy before the game; see (Balseiro et al., 2023) for a related discussion. Since Eq. (2) relies on information unavailable at decision time, it cannot be matched exactly. Instead, we assess the mechanism's performance by measuring its regret relative to the offline optimal:

<!-- formula-not-decoded -->

where the expectation is over the randomness in the mechanism, agent strategies, and value/cost realizations. This regret notion generalizes the one studied in the non-strategic setting of Balseiro et al. (2023), where agents report truthfully ( i.e. , π = TRUTH such that u t,i = v t,i for all t and i ). In that setting, their mechanism M 0 achieved regret R T ( TRUTH , M 0 ) = ˜ O ( √ T ) . However, if agents are strategic, they may deviate from truthful reporting. In contrast, our mechanism in Algorithm 2 guarantees the existence of a Perfect Bayesian Equilibrium (PBE) strategy profile π such that no agent benefits from unilateral deviation, and under which the regret remains ˜ O ( √ T ) (Theorem 3.2).

3 Two special cases of Assumption 1 are 0 -impatient agents, who only care about their gains in the current round (often referred to as myopic agent), and 1 -impatient agents, who care about their total gains over the entire T -round game (as is typical in extensive-form games).

```
Input: Number of rounds T , agents K , resources d , cost constraint ρ ∈ [0 , 1] d Epochs {E ℓ } L ℓ =1 , learning rates { η ℓ } L ℓ =1 , regularizer Ψ: R d ≥ 0 → R , sub-routine A Output: Allocations i 1 , . . . , i T , where i t = 0 denotes no allocation 1: Define dual region Λ := { λ ∈ R d : λ j ∈ [0 , ρ -1 j ] } 2: for epoch ℓ = 1 , 2 , . . . , L do 3: Update dual variable λ ℓ ∈ Λ via the sub-routine A : λ ℓ ←A ( ℓ, η ℓ , Ψ) . 4: for each round t ∈ E ℓ do 5: Each agent i ∈ [ K ] observes their value v t,i ∼ V i and costs c t ∼ C 6: Agent reports u t,i ∈ [0 , 1] according to Protocol 1; u t and c t become public. 7: if round t is selected for exploration ( w.p. 1 / |E ℓ | independently) then 8: Sample tentative agent i ∼ Unif ([ K ]) and payment p ∼ Unif ([0 , 1]) 9: If u t,i ≥ p , set i t = i and p t,i t = p . Otherwise, set i t = 0 10: else 11: Compute adjusted cost ˜ c t,i = λ ⊤ ℓ c t,i and adjusted report ˜ u t,i := u t,i -˜ c t,i , ∀ i ∈ [ K ] 12: Allocate to agent i t := arg max i ˜ u t,i with payment p t,i t = ˜ c t,i t +max j = i t ˜ u t,j 13: if cost constraint is violated: ∑ s<t c s,i s + c t,i t ̸≤ T ρ then 14: Reject allocation by setting i t = 0
```

Algorithm 2 Primal-Dual Mechanism Robust to Strategic Manipulations

̸

In parallel, the planner also aims to minimize constraint violations, which we define as

<!-- formula-not-decoded -->

where ( · ) + is the coordinate-wise maximum with zero, i.e. , x + := [max( x i , 0)] i ∈ [ d ] . Our mechanism M , presented in Algorithm 2, guarantees B T ( π , M ) = 0 ; that is, cost constraints are satisfied.

We conclude this section with a smoothness assumption on cost distributions. The idea is to ensure that projected costs (linear combinations of the cost vector) do not place excessive probability mass on any single value. This smoothness condition prevents pathological behaviors where a small change in an agent's report could drastically alter outcomes due to spiky distributions. Such assumptions are common in strategic settings to ensure robustness to perturbations, including in bilateral trades (Cesa-Bianchi et al., 2024a), first-price auctions (Cesa-Bianchi et al., 2024b), second-price auctions with reserves (Golrezaei et al., 2021a), and smoothed revenue maximization (Durvasula et al., 2023).

Assumption 3 (Smooth Costs) . For any agent i ∈ [ K ] , the cost distribution C i satisfies the following: for all λ ∈ Λ := { λ ∈ R d | λ j ∈ [0 , ρ -1 j ] } , the density of the projected cost PDF c t ∼C i ( λ ⊤ c i ) is uniformly bounded above by some universal constant ϵ c &gt; 0 . We do not assume this ϵ c to be known.

## 3 Primal-Dual Mechanism Robust to Strategic Manipulation

Weintroduce our incentive-aware primal-dual mechanism, Algorithm 2, which overcomes the fragility of standard primal-dual methods to strategic agents. As demonstrated in Figure 1, vanilla primal-dual methods allow agents to manipulate future dual updates by misreporting, leading to misaligned incentives and degraded performance. Our mechanism addresses this challenge through three key innovations: epoch-based lazy dual updates, dual-adjusted allocation and payments, and randomized exploration rounds, which we describe below before presenting the algorithm and its guarantees.

Epoch-Based Lazy Updates. We divide the game horizon [ T ] into L epochs as [ T ] = E 1 ∪··· ∪ E L and fix a single dual variable λ ℓ within each epoch E ℓ . These dual variables act as implicit prices on resource consumption and adjust reported values accordingly. By holding λ ℓ constant within each epoch, we reduce agents' ability to manipulate future allocations through misreports. This 'lazy update' scheme is a central ingredient in limiting intertemporal strategic behavior.

Dual-Adjusted Allocation and Payments. In each round t ∈ [ T ] , agents submit reports based on their private values and observed costs. With high probability, the mechanism enters a standard round, allocating the resource to the agent with the highest dual-adjusted report (reported value minus dual-weighted cost), and charging them their cost plus the second-highest dual-adjusted report.

## Algorithm 3 Dual Update Sub-Routine using Follow-the-Regularized-Leader (FTRL)

Input: Current epoch number ℓ , learning rate η ℓ &gt; 0 , regularizer Ψ: R d ≥ 0 → R

1: Solve the following optimization problem for λ ℓ ∈ Λ and return λ ℓ .

<!-- formula-not-decoded -->

Algorithm 4 Dual Update Sub-Routine using Optimistic FTRL with Fixed Points (O-FTRL-FP)

Input: Current epoch number ℓ , learning rate η ℓ &gt; 0 , regularizer Ψ: R d ≥ 0 → R

1: Solve the following fixed point problem for ( λ ℓ , ˜ g ℓ ) ∈ Λ × R d and return λ ℓ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This payment rule, inspired by boosted second-price auctions (Golrezaei et al., 2021b), is incentivecompatible in static (one-shot) settings and encourages truthful reporting in our dynamic setup. If the allocation would violate the cumulative cost constraint, it is rejected to ensure feasibility.

Randomized Exploration Rounds. With a small probability, the mechanism initiates an exploration round , offering a random price to a randomly selected agent. This structure penalizes misreports by imposing a direct utility loss when the reported value deviates from the true value (see Theorem 4.2). These rounds act as incentive-compatible signal extractors and are essential for maintaining the accuracy of dual updates based on strategic reports. The idea of randomized pricing has been explored in repeated second-price auctions (Amin et al., 2013; Golrezaei et al., 2021a, 2023), but to our knowledge, our work is the first to leverage it for robust primal-dual learning.

Dual Updates via Online Learning. In the beginning of each epoch, the planner updates the dual variable λ ℓ via an online learning approach. Equipped with Follow-the-Regularized-Leader (FTRL), we prove ˜ O ( T 2 / 3 ) is attainable; however, the epoch-based structure also poses an Ω( T 2 / 3 ) online learning barrier (Theorem 4.4). To go beyond the ˜ O ( T 2 / 3 ) regret of FTRL, we leverage the neartruthfulness induced by our mechanism to make predictions about future behavior. Our Optimistic FTRL with Fixed Points (O-FTRL-FP) augments classical FTRL with a forward-looking term that estimates how the current dual variable λ ℓ would perform if agent behavior remains consistent.

Specifically, O-FTRL-FP solves a fixed-point problem: it chooses λ ℓ to minimize a combination of past constraint violations ∑ τ ( ρ -c τ,i τ ) T λ , a prediction term ˜ g ℓ ( λ ℓ ) T λ based on simulated allocations using prior reports, and a regularization term Ψ( λ ) η ℓ . Since ˜ g ℓ ( λ ℓ ) itself depends on λ ℓ , the optimization forms a self-consistent loop. We show that this method achieves ˜ O ( √ T ) regret while maintaining feasibility and incentive alignment.

Main Results. We now state our two main theoretical guarantees, corresponding to different dual update strategies. Full formal statements and proofs are provided in Appendix B.

Theorem 3.1 (Algorithm 2 with FTRL) . Under appropriate choice of epoch lengths and learning rates, Algorithm 2 using FTRL in Eq. (5) as the dual-update sub-routine A guarantees the existence of a PBE π ∗ such that R T ( π ∗ , Algorithm 2 ) = ˜ O ( T 2 / 3 ) and B T ( π ∗ , Algorithm 2 ) = 0 .

Theorem 3.2 (Algorithm 2 with O-FTRL-FP) . Under appropriate choice of epoch lengths and learning rates, Algorithm 2 using O-FTRL-FP in Eq. (6) as the dual-update sub-routine A guarantees the existence of a PBE π ∗ such that R T ( π ∗ , Algorithm 2 ) = ˜ O ( √ T ) and B T ( π ∗ , Algorithm 2 ) = 0 .

Remarkably, Arlotto and Gurvich (2019) proved that even when agents are non-strategic, with unknown value and cost distributions, it is unavoidable to suffer Ω( √ T ) social welfare regret in the worst case. Therefore, our mechanism - while additionally being robust to strategic agents - matches this lower bound when focusing on poly ( T ) dependencies.

## 4 Analysis Sketch of Theorems 3.1 and 3.2

The safety property of Algorithm 2, namely that B T = 0 , follows directly from Line 14. Thus, we focus on analyzing the regret R T = E [ ∑ T t =1 ( v t,i ∗ t -v t,i t ) ] , which measures the expected difference between our allocation { i t } t ∈ [ T ] and the offline optimal allocation { i ∗ t } t ∈ [ T ] defined in Eq. (2). To help control this regret, we introduce an intermediate allocation that maximizes the dual-adjusted values (rather than the actual allocation i t maximizing dual-adjusted reports, which may be strategic):

<!-- formula-not-decoded -->

Define a stopping time T v := min { t ∈ [ T ] | ∑ t τ =1 c τ,i τ + 1 ̸≤ T ρ } ∪ { T +1 } as the last round where it is impossible for Line 14 to reject i t . We decompose the regret R T as

<!-- formula-not-decoded -->

## 4.1 Misallocations Due to Agents' Strategic Behavior (PRIMALALLOC)

As discussed in Section 3, agents' strategic reports in epoch ℓ can influence future dual variables λ ℓ +1 , which in turn affect subsequent allocations - potentially creating feedback loops that lead to misallocation. We give an example of this scenario:

Example 4 (Agents are able to strategically affect λ ℓ +1 ) . Consider two agents with identical value distributions, i.e., V 1 = V 2 . Suppose the cost vectors are fixed as c t,i ≡ e i for all t and i , and the cost budget is ρ = (1 / 2 , 1 / 2) . If both agents report truthfully during epoch E 1 , the resource is allocated approximately equally. As a result, the dual vector λ 2 , computed via Eq. (6), has similar values across its coordinates. However, suppose that agent 1 strategically under-reports their value throughout epoch 1, causing all allocations to go to agent 2 (i.e., i t = 2 for all t ∈ E 1 ). This skews the observed cost consumption toward the second type, causing λ 2 , 2 ≫ λ 2 , 1 . Consequently, agent 2 whose actions incur the second cost type - will face significantly higher penalties in future epochs.

To understand this effect, we decompose the total inefficiency due to agents' strategic behavior, namely the PRIMALALLOC in Eq. (7), into two parts: (i) INTRAEPOCH measuring misallocations arisen due to agents' incentives for immediate or short-term gains, and (ii) INTEREPOCH measuring misallocations caused by agents influencing dual updates for future-epoch benefits. To isolate them, we introduce two behavioral models for agents, where Model 1 is exactly Assumption 1:

<!-- formula-not-decoded -->

Model 2 essentially assumes agents only optimize over the current epoch E ℓ , ignoring long-term impact. Let { i h t } t ∈ [ T ] be the allocations that would occur under Algorithm 2 if agents followed Model 2. Our goal is to first analyze this hypothetical setting to understand INTRAEPOCH, then examine the deviation introduced by agents following the realistic Model 1, which leads to INTEREPOCH effects.

## 4.1.1 INTRAEPOCH: Misalignment Within Epochs

Under Model 2, each epoch ℓ ∈ [ L ] can be treated independently, which we call an 'epochℓ game'. The planner selects allocations to maximize the total dual-adjusted value ∑ t ∈E ℓ ˜ v t,i h t , where ˜ v t,i := v t,i -λ T ℓ c t,i . In contrast, agents care about their true value v t,i rather than the dual-adjusted objective. This introduces a mismatch between the planner's and agents' optimization criteria. Despite this mismatch, we show that our mechanism's dual-adjusted allocation rule in Line 12 incentivizes truthful reporting within each epoch:

Theorem 4.1 (INTRAEPOCH Guarantee; Informal Theorem C.2) . In the epochℓ game under Model 2, the allocation rule i h t = arg max i ˜ u t,i with payment p t,i h t = λ T ℓ c t,i h t + 2nd-highest ˜ u t,i ensures that truthful reporting is a PBE. Under this equilibrium, E [ INTRAEPOCH ] = 0 .

## 4.1.2 INTEREPOCH: Strategic Manipulation of Future Duals

We now return to Model 1, where agents optimize over the entire horizon t ∈ [ T ] . In this setting, an agent can misreport in the current epoch to influence the next epoch's dual variable λ ℓ +1 , thereby improving their chances of allocation in future rounds. This creates a new avenue for strategic behavior that was not captured under Model 2.

To mitigate this, our mechanism introduces randomized exploration rounds , which penalize deviations from truthful reporting through stochastic pricing: Reporting u t,i &gt; v t,i means paying a price higher than value when p ∈ ( v t,i , u t,i ) ; reporting u t,i &lt; v t,i means missing an opportunity to make profits when p ∈ ( v t,i , u t,i ) . These rounds - by ensuring misreports carry immediate utility losses that do not outweigh the future gains - reduce agents' willingness to manipulate dual updates.

Theorem 4.2 (PRIMALALLOC Guarantee; Informal Theorem C.5) . There exists a PBE π ∗ such that for any epoch ℓ ∈ [ L ] , the number of round-agent ( t, i ) -pairs where | u t,i -v t,i | ≥ 1 |E ℓ | is ˜ O (1) with high probability. That is, agent reports under Models 1 and 2 rarely differ. Moreover, the resulting allocations { i t } t ∈E ℓ and { i h t } t ∈E ℓ differ in at most ˜ O (1) rounds with high probability. Consequently, this equilibrium ensures E [ PRIMALALLOC ] = ˜ O ( L ) .

## 4.2 Inaccurate Dual Variables Due to Incomplete Information (DUALVAR)

The second source of inefficiency comes from sub-optimal dual variables - or more precisely, the gap between the dual-adjusted allocation ˜ i ∗ t := arg max i ( v t,i -λ T ℓ c t,i ) that our mechanism maximizes and the offline optimal benchmark i ∗ t defined in Eq. (2) - which we call DUALVAR in Eq. (7).

## 4.2.1 Translating DUALVAR to Online Learning Regret

Using primal-dual analysis similar to that of Balseiro et al. (2023), we relate this gap to the online learning regret over dual variables λ 1 , λ 2 , . . . , λ L ∈ Λ (where Λ := { λ ∈ R d : λ j ∈ [0 , ρ -1 j ] } is chosen such that ρ -1 j e j ∈ Λ for all j ∈ [ d ] ) as follows, which is an informal version of Lemma D.1:

Lemma 4.3 (DUALVAR as Online Learning Regret) . Let ˜ i ∗ t = argmax i ∈ [ K ] ( v t,i -λ T ℓ c t,i ) denote the best allocation under dual prices and i ∗ t denote the offline optimal benchmark. Then,

<!-- formula-not-decoded -->

That is, DUALVAR reduces to an online learning problem where the planner selects a dual vector λ ℓ ∈ Λ at the beginning of each epoch ℓ and incurs a linear loss based on constraint violations of { i t } t ∈E ℓ . Specifically, the loss function in epoch ℓ is given by F ℓ ( λ ) := ∑ t ∈E ℓ ( ρ -c t,i t ) T λ .

## 4.2.2 Achievements and Limitations of FTRL

Using FTRL, the planner selects λ ℓ by minimizing a regularized sum of historical losses, namely λ ℓ = argmin λ ∈ Λ ∑ ℓ ′ &lt;ℓ F ℓ ′ ( λ ) + 1 η ℓ Ψ( λ ) where Ψ is a strongly convex regularizer. This choice yields regret R λ L = ˜ O ( √ ∑ L ℓ =1 |E ℓ | 2 ) . Choosing L = T 1 / 3 epochs each of size T 2 / 3 minimizes R λ + L , giving total regret O ( T 2 / 3 ) as shown in Theorem 3.1.

L ˜

However, vanilla FTRL is fundamentally limited due to epoch-based structures: Frequent updates to λ ℓ would enable faster learning, but break incentive compatibility. Conversely, fewer updates protect incentives, but slow learning. This tradeoff is formalized in the following hardness result:

Theorem 4.4 (Hardness for Low-Switching Online Learning (Dekel et al., 2014)) . Consider an online learning algorithm that guarantees regret R λ L = O ( T α ) for some α ∈ [1 / 2 , 1) . Then, it must switch decisions at least Ω( T 2(1 -α ) ) times. In our setting, this means that the number of dual updates - i.e., the number of epochs L - must satisfy L = Ω( T 2(1 -α ) ) . This infers that the R λ L + L term in Lemma 4.3 suffers a worst-case bound of Ω ( min α ∈ [1 / 2 , 1) T α + T 2(1 -α ) ) = Ω( T 2 / 3 ) .

Figure 2: From dual-optimal to actual allocations: fixing dual λ ℓ , our mechanism ensures that the actual allocations { i t } closely follow the dual-optimal { ˜ i ∗ t } via an intermediate myopic model { i h t } .

<!-- image -->

## 4.2.3 Exploiting Incentive Alignments for Boosted Regret

The hardness result in Theorem 4.4 applies to the worst-case senario, i.e. , when the loss functions F 1 , F 2 , . . . , F L can be any arbitrary linear functions. But in our setting, due to the incentivecompatible primal allocations (Theorems 4.1 and 4.2), the loss F ℓ ( λ ) has an almosti.i.d. structure:

Claim 4.5 (Loss Structure; Informal) . Fix any λ ℓ ∈ Λ . Then for all but ˜ O (1) rounds in epoch E ℓ , the actual allocations i t match the dual-optimal choices ˜ i ∗ t = arg max i ( v t,i -λ T ℓ c t,i ) . Thus,

<!-- formula-not-decoded -->

which behaves is the sum of |E ℓ | i.i.d. samples and thus has a variance of order ˜ O ( |E ℓ | ) .

Proof Idea. To understand how our mechanism enables accurate dual updates despite incomplete information and strategic behavior, we illustrate in Figure 2 the connection between three key allocation sequences within a fixed epoch. First, the sequence { ˜ i ∗ t } represents the dual-optimal allocations computed in hindsight, assuming access to true valuations and a fixed dual vector λ ℓ . Second, { i h t } denotes the hypothetical allocations made under Model 2, where agents are myopic and only optimize within a single epoch. By Theorem 4.1, these allocations align with { ˜ i ∗ t } under our incentive-compatible subroutine. Lastly, the actual sequence { i t } , generated by strategic agents under Model 1, is shown to be close to { i h t } via Theorem 4.2. Together, these approximations establish that { i t } behaves almost like an i.i.d. sample from the dual-adjusted best responses.

This low-variance structure is essential for achieving ˜ O ( √ T ) regret. In general, linear losses with gradients of norm ˜ O ( |E ℓ | ) can have variances as large as ˜ O ( |E ℓ | 2 ) , which explains the T 2 / 3 regret scaling of FTRL in Theorem 3.1 and the lower bound in Theorem 4.4. Even more importantly, because this neari.i.d. structure holds across previous epochs as well, it enables us to accurately estimate losses associated with new dual choices using only historical data - without requiring access to agents' true values or distributions. These two insights provides predictability ahead of time, which enables the application of optimistic online learning algorithms, for example, Optimistic FTRL (O-FTRL) by Rakhlin and Sridharan (2013). O-FTRL ensures that if the actual loss F ℓ is well-predicted by some predicted loss ̂ F ℓ , such that the expected squared error in gradients is of order ˜ O ( |E ℓ | ) , then one can break the ˜ O ( T 2 / 3 ) regret barrier and attain ˜ O ( √ T ) performance (Lemma E.2).

## 4.2.4 Resolving Circular Dependencies between Actions and Predictions

The final issue stopping us from obtaining Theorem 3.2 is that, O-FTRL framework requires us to construct the predicted loss ̂ F ℓ ( λ ) before deciding action λ ℓ ; however, in our case, recall that

<!-- formula-not-decoded -->

In words, to construct a good ̂ F ℓ ( λ ) ≈ F ℓ ( λ ) and decide λ ℓ , we need to know λ ℓ first because F ℓ ( λ ) depends on λ ℓ . This circular dependency between action λ ℓ and prediction ̂ F ℓ ( λ ) stops us from applying O-FTRL. To circumvent this issue, we instead allow the prediction to have a form ̂ F ℓ ( λ ; λ ℓ ) - such that ̂ F ℓ ( · ; λ ℓ ) ≈ F ℓ ( · ) if we really chose λ ℓ as our action for epoch ℓ - and decide the action via a fixed point problem (as in Eq. (6)). We call this novel online learning algorithm O ptimistic FTRL with F ixed P oints (O-FTRL-FP in short). In Lemma D.4, we prove that if a small perturbation in λ ℓ doesn't change ∇ λ ̂ F ℓ ( λ ; λ ℓ ) by a lot, O-FTRL-FP always admits an approximate fixed point.

To get the ˜ O ( √ T ) social welfare regret claimed in Theorem 3.2, it only remains to i) verify our ̂ F ℓ in Eq. (6) indeed makes Lemma D.4 applicable; ii) show that E [ ∥∇ ℓ F ℓ ( λ ) -∇ ℓ ̂ F ℓ ( λ ; λ ℓ ) ∥ 2 2 ] = ˜ O ( |E ℓ | ) despite agents' misreports in the current epoch E ℓ , the statistical barrier in reconstructing { ( V i , C i ) } i , and agents' historical misreports; and iii) properly tune all hyper-parameters. Due to space limitations, these steps are deferred to Theorem D.3 in the appendix.

## 5 Numerical Study

Basic Setup. We simulate a game with T = 1000 rounds, K = 3 agents, a single resource dimension d = 1 , and a discount factor γ = 0 . 9 . Each agent's valuation is drawn from V i = Unif [0 , 1] , and their cost is drawn from C i = Unif [0 . 7 ρ, 1 . 3 ρ ] , for all i = 1 , 2 , 3 .

Agents' Model. To reflect agents' strategic behaviors, we assume that every agent i = 1 , 2 , 3 models the game as a Markov Decision Process (MDP) with state defined as ( t, λ t , v t,i ) -i.e. , round number t , current dual variable λ t , 4 and their own private value v t,i - and the action defined as u t,i . The reward for agent i after playing a report u t,i is γ t ( v t,i -p t,i ) ✶ [ i t = i ] as in Assumption 1. We discretize all the values, duals, and reports to the nearest multiple of 0 . 1 . We repeat the same game for N = 1000 independent trails, where every agent keeps refining their strategy via Q-learning (Watkins and Dayan, 1992). In the n -th trial, every agent uses ϵ n -greedy with a geometrically decaying schedule of ϵ = 0 . 995 n . We update Q-tables using Q ( s, a ) ← Q ( s, a ) + α ( r + γ (max a ′ Q ( s ′ , a ′ )) -Q ( s, a )) where α = 0 . 1 , r is the instantaneous reward, and s ′ is the new state ( t +1 , λ t +1 , v t +1 ,i ) .

Mechanisms. The vanilla primal-dual algorithm we use is Algorithm 1 of Balseiro et al. (2023), which ensures constraint satisfaction by rejecting any allocation that would violate the budget. When implementing the O-FTRL-FP update rule in Eq. (6), instead of solving the fixed-point problem λ ℓ = argmin λ ( · · · + ˜ g ℓ ( λ ℓ ) T λ + Ψ( λ ) η ℓ ) , we solve argmin λ ( · · · + ˜ g ℓ ( λ ) T λ + Ψ( λ ) η ℓ ) for numerical simplicity. The validity of this approximation is due to Lemma D.5, which says ∀ λ 1 , λ 2 ∈ Λ s.t. ∥ λ 1 -λ 2 ∥ 1 ≤ ϵ , ˜ g ℓ ( λ 1 ) ≈ ˜ g ℓ ( λ 2 ) w.h.p. Thus the two objectives agree locally around the true λ ℓ .

Results. Figure 1 demonstrates that our mechanism significantly outperforms the standard primaldual method in the presence of strategic agents, both adhering to cost constraints. Under the primaldual approach of Balseiro et al. (2023, Algorithm 1) which is plotted in blue, agents systematically over-report their valuations (left), hurting the overall social welfare (middle), and resulting in lower budget utilization - when agents submit similar reports, the impact of cost minimization is amplified (right). In contrast, our mechanism equipped with O-FTRL-FP (in green) incentivizes near-truthful reporting, as shown by the alignment between reported and true values, resulting in a substantial increase in long-term social welfare. These findings highlight the fragility of standard primal-dual methods in strategic settings and the robustness of our proposed incentive-aware mechanism.

Computational Resources. Every illustration executes on a M1 MacBook Air 2020 in 10 minutes.

## 6 Conclusion and Future Directions

This paper investigates the dynamic allocation of reusable resources to strategic agents under multidimensional long-term cost constraints. We show that standard primal-dual methods, though effective in non-strategic settings, are vulnerable to manipulation when agents act strategically. To address this, we introduce a novel incentive-aware mechanism that stabilizes dual updates via an epoch-based structure and leverages randomized exploration rounds to extract truthful signals. Equipped with a computationally efficient FTRL dual update rule, our mechanism guarantees sublinear regret with respect to an offline benchmark, satisfies all cost constraints, and admits a PBE; further leveraging a novel O-FTRL-FP framework for dual updates, we boost the regret to ˜ O ( √ T ) - which is nearoptimal even in non-strategic constrained dynamic resource allocation settings. Looking ahead, several promising research directions remain open. For example, while our mechanism uses monetary transfers to ensure incentive compatibility, many real-world applications, e.g. , organ matching, school admissions, or vaccine distribution, operate under non-monetary constraints. Extending our framework to such settings is an important step toward more broadly applicable mechanism design.

4 Since mechanism M is public and every agent knows every information the planner has, i.e. , H t, 0 ⊆ H t,i , agents are able to calculate λ t on their end.

## Acknowledgements

P.J. and Y.D. are partially funded by the Office of Naval Research (ONR) under Award ID N0001424-1-2470. N.G. and Y.D. are partially supported by the MIT Junior Faculty Research Assistance Grant and by the Office of Naval Research (ONR) under Award ID N00014-23-1-2584. The authors thank the anonymous reviewers for their constructive feedback.

## References

- Ivo Adan, Jos Bekkers, Nico Dellaert, Jan Vissers, and Xiaoting Yu. Patient mix optimisation and stochastic resource requirements: A case study in cardiothoracic surgery planning. Health care management science , 12:129-141, 2009.
- Shipra Agrawal, Zizhuo Wang, and Yinyu Ye. A dynamic near-optimal algorithm for online linear programming. Operations Research , 62(4):876-890, 2014.
- Kareem Amin, Afshin Rostamizadeh, and Umar Syed. Learning prices for repeated auctions with strategic buyers. Advances in neural information processing systems , 26, 2013.
- Kareem Amin, Afshin Rostamizadeh, and Umar Syed. Repeated contextual auctions with strategic buyers. Advances in Neural Information Processing Systems , 27, 2014.
- Alessandro Arlotto and Itai Gurvich. Uniformly bounded regret in the multisecretary problem. Stochastic Systems , 9(3):231-260, 2019.
- Kenneth J Arrow. A difficulty in the concept of social welfare. Journal of political economy , 58(4): 328-346, 1950.
- Ashwinkumar Badanidiyuru, Robert Kleinberg, and Aleksandrs Slivkins. Bandits with knapsacks. Journal of the ACM (JACM) , 65(3):1-55, 2018.
- Santiago R Balseiro and Yonatan Gur. Learning in repeated auctions with budgets: Regret minimization and equilibrium. Management Science , 65(9):3952-3968, 2019.
- Santiago R Balseiro, Huseyin Gurkan, and Peng Sun. Multiagent mechanism design without money. Operations Research , 67(5):1417-1436, 2019.
- Santiago R Balseiro, Haihao Lu, and Vahab Mirrokni. The best of many worlds:: Dual mirror descent for online allocation problems. Operations Research , 71(1):101-119, 2023.
- Siddhartha Banerjee, Giannis Fikioris, and Eva Tardos. Robust pseudo-markets for reusable public resources. In Proceedings of the 24th ACM Conference on Economics and Computation , pages 241-241, 2023.
- Damien Berriaud, Ezzat Elokda, Devansh Jalota, Emilio Frazzoli, Marco Pavone, and Florian Dörfler. To spend or to gain: Online learning in repeated karma auctions. arXiv preprint arXiv:2403.04057 , 2024.
- Dimitri Bertsekas, Angelia Nedic, and Asuman Ozdaglar. Convex analysis and optimization , volume 1. Athena Scientific, 2003.
- Moise Blanchard and Patrick Jaillet. Near-optimal mechanisms for resource allocation without monetary transfers. arXiv preprint arXiv:2408.10066 , 2024.
- Rajkumar Buyya, Chee Shin Yeo, and Srikumar Venugopal. Market-oriented cloud computing: Vision, hype, and reality for delivering it services as computing utilities. In 2008 10th IEEE international conference on high performance computing and communications , pages 5-13. Ieee, 2008.
- Matteo Castiglioni, Andrea Celli, and Christian Kroer. Online learning with knapsacks: the best of both worlds. In International Conference on Machine Learning , pages 2767-2783. PMLR, 2022.

- Nicolò Cesa-Bianchi, Tommaso Cesari, Roberto Colomboni, Federico Fusco, and Stefano Leonardi. Regret analysis of bilateral trade with a smoothed adversary. Journal of Machine Learning Research , 25(234):1-36, 2024a.
- Nicolò Cesa-Bianchi, Tommaso Cesari, Roberto Colomboni, Federico Fusco, and Stefano Leonardi. The role of transparency in repeated first-price auctions with unknown valuations. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing , pages 225-236, 2024b.
- Edward H Clarke. Multipart pricing of public goods. Public choice , pages 17-33, 1971.
- Richard Cole, Vasilis Gkatzelis, and Gagan Goel. Positive results for mechanism design without money. In Proceedings of the 2013 international conference on Autonomous agents and multi-agent systems , pages 1165-1166, 2013.
- Yan Dai, Moise Blanchard, and Patrick Jaillet. Non-monetary mechanism design without distributional information: Using scarce audits wisely. arXiv preprint arXiv:2502.08412 , 2025.
- Ofer Dekel, Jian Ding, Tomer Koren, and Yuval Peres. Bandits with switching costs: T 2/3 regret. In Proceedings of the forty-sixth annual ACM symposium on Theory of computing , pages 459-467, 2014.
- Nikhil R Devanur and Thomas P Hayes. The adwords problem: online keyword matching with budgeted bidders under random permutations. In Proceedings of the 10th ACM conference on Electronic commerce , pages 71-78, 2009.
- Nikhil R Devanur, Kamal Jain, Balasubramanian Sivan, and Christopher A Wilkens. Near optimal online algorithms and fast approximation algorithms for resource allocation problems. Journal of the ACM (JACM) , 66(1):1-41, 2019.
- John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research , 12(7), 2011.
- Naveen Durvasula, Nika Haghtalab, and Manolis Zampetakis. Smoothed analysis of online nonparametric auctions. In Proceedings of the 24th ACM Conference on Economics and Computation , pages 540-560, 2023.
- Jon Feldman, Monika Henzinger, Nitish Korula, Vahab S Mirrokni, and Cliff Stein. Online stochastic packing applied to display ad allocation. In European Symposium on Algorithms , pages 182-194. Springer, 2010.
- Giannis Fikioris, Siddhartha Banerjee, and Éva Tardos. Online resource sharing via dynamic max-min fairness: efficiency, robustness and non-stationarity. arXiv preprint arXiv:2310.08881 , 2023.
- Rigel Galgana and Negin Golrezaei. Learning in repeated multiunit pay-as-bid auctions. Manufacturing &amp; Service Operations Management , 27(1):200-229, 2025.
- A. Gibbard. Manipulation of voting schemes: A general result. Econometrica , 41(4):587-601, 1973.
- Negin Golrezaei, Hamid Nazerzadeh, and Paat Rusmevichientong. Real-time optimization of personalized assortments. Management Science , 60(6):1532-1551, 2014.
- Negin Golrezaei, Patrick Jaillet, and Jason Cheuk Nam Liang. No-regret learning in price competitions under consumer reference effects. Advances in Neural Information Processing Systems , 33:2141621427, 2020.
- Negin Golrezaei, Adel Javanmard, and Vahab Mirrokni. Dynamic incentive-aware learning:: Robust pricing in contextual auctions. Operations Research , 69(1):297-314, 2021a.
- Negin Golrezaei, Max Lin, Vahab Mirrokni, and Hamid Nazerzadeh. Boosted second price auctions: Revenue optimization for heterogeneous bidders. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining , pages 447-457, 2021b.
- Negin Golrezaei, Patrick Jaillet, and Jason Cheuk Nam Liang. Incentive-aware contextual pricing with non-parametric market noise. In International Conference on Artificial Intelligence and Statistics , pages 9331-9361. PMLR, 2023.

- Artur Gorokh, Siddhartha Banerjee, and Krishnamurthy Iyer. The remarkable robustness of the repeated fisher market. In Proceedings of the 22nd ACM Conference on Economics and Computation , pages 562-562, 2021a.
- Artur Gorokh, Siddhartha Banerjee, and Krishnamurthy Iyer. From monetary to nonmonetary mechanism design via artificial currencies. Mathematics of Operations Research , 46(3):835-855, 2021b.
- Theodore Groves. Incentives in teams. Econometrica: Journal of the Econometric Society , pages 617-631, 1973.
- Mingyu Guo and Vincent Conitzer. Strategy-proof allocation of multiple items between two agents without payments or priors. In AAMAS , pages 881-888, 2010.
- Anupam Gupta and Marco Molinaro. How the experts algorithm can help solve lps online. Mathematics of Operations Research , 41(4):1404-1431, 2016.
- Li Han, Chunzhi Su, Linpeng Tang, and Hongyang Zhang. On strategy-proof allocation without payments or priors. In International Workshop on Internet and Network Economics , pages 182-193. Springer, 2011.
- Devansh Jalota, Matthew Tsao, and Marco Pavone. Catch me if you can: Combatting fraud in artificial currency based government benefits programs. arXiv preprint arXiv:2402.16162 , 2024.
- Yash Kanoria and Hamid Nazerzadeh. Dynamic reserve prices for repeated auctions: Learning from bids. In Web and Internet Economics: 10th International Conference , volume 8877, page 232. Springer, 2014.
- Thomas Kesselheim, Klaus Radke, Andreas Tonnis, and Berthold Vocking. Primal beats dual on online packing lps in the random-order model. SIAM Journal on Computing , 47(5):1939-1964, 2018.
- Jonas Moritz Kohler and Aurelien Lucchi. Sub-sampled cubic regularization for non-convex optimization. In International Conference on Machine Learning , pages 1895-1904. PMLR, 2017.
- Christos Koufogiannakis and Neal E Young. A nearly linear-time ptas for explicit fractional packing and covering linear programs. Algorithmica , 70:648-674, 2014.
- Xiaocheng Li, Chunlin Sun, and Yinyu Ye. Simple and fast algorithm for binary integer and online linear programming. Mathematical Programming , 200(2):831-875, 2023.
- Antonio Miralles. Cardinal bayesian allocation mechanisms without transfers. Journal of Economic Theory , 147(1):179-206, 2012.
- Marco Molinaro and Ramamoorthi Ravi. The geometry of online packing linear programs. Mathematics of Operations Research , 39(1):46-59, 2014.
- J.R. Munkres. Topology . Featured Titles for Topology. Prentice Hall, Incorporated, 2000. ISBN 9780131816299. URL https://books.google.com/books?id=XjoZAQAAIAAJ .
- Mahyar Movahed Nejad, Lena Mashayekhy, and Daniel Grosu. Truthful greedy mechanisms for dynamic virtual machine provisioning and allocation in clouds. IEEE transactions on parallel and distributed systems , 26(2):594-603, 2014.
- Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213 , 2019.
- Alexander Rakhlin and Karthik Sridharan. Online learning with predictable sequences. In Conference on Learning Theory , pages 993-1019. PMLR, 2013.
- N.A. Satterthwaite. Strategy-proofness and Arrow's conditions: Existence and correspondence theorems for voting procedures and social welfare functions. Journal of Economic Theory , 10(2): 187-217, 1975.

- Christian Stummer, Karl Doerner, Axel Focke, and Kurt Heidenberger. Determining location and size of medical departments in a hospital network: A multiobjective decision support approach. Health care management science , 7:63-71, 2004.
- Rui Sun, Xinshang Wang, and Zijie Zhou. Near-optimal primal-dual algorithms for quantity-based network revenue management. arXiv preprint arXiv:2011.06327 , 2020.
- William Vickrey. Counterspeculation, auctions, and competitive sealed tenders. The Journal of finance , 16(1):8-37, 1961.
- Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning , 8:279-292, 1992.
- David Williams. Probability with martingales . Cambridge university press, 1991.
- Zongjun Yang, Luofeng Liao, Yuan Gao, and Christian Kroer. Online fair allocation with best-ofmany-worlds guarantees. arXiv preprint arXiv:2408.02403 , 2024.
- Steven Yin, Shipra Agrawal, and Assaf Zeevi. Online allocation and learning in the presence of strategic agents. Advances in Neural Information Processing Systems , 35:6333-6344, 2022.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In Figure 1, we illustrated the fragility of primal-dual framework to agents' strategic behaviors. In Theorems 3.1 and 3.2 (with formal versions appearing as Theorems B.1 and B.2), we prove the claimed performance guarantee of our proposed mechanism in Algorithm 2.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: All assumptions are clearly stated. These limitations are discussed in the Conclusion. Guidelines:

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

Justification: Our main claims, Theorems 3.1 and 3.2, have their sketched proofs in Section 4 and their full proofs in Appendices B to D. Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The numerical illustration setup is clearly explained in Section 5, with codes attached as supplementary materials.

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

Justification: The code used for simulation is attached in the supplementary materials.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.

- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https://nips.cc/public/ guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: They are described in Section 5.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The claim regarding agents' truth-reporting, e.g. , the first plot in Figure 1, is plotted using repeated sampling and error bars.

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

Justification: See Section 5.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Work of purely theoretical nature.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Work of purely theoretical nature.

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

Justification: Work of purely theoretical nature.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: In our numerical illustrations, we reproduced the algorithm designed by Balseiro et al. (2023) with proper citations.

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

Justification: No new assets released.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No crowdsourcing and research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No crowdsourcing and research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Work of purely theoretical nature without LLM involvement.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendices

| A   | More Discussions on Related Work                     | More Discussions on Related Work                              |   21 |
|-----|------------------------------------------------------|---------------------------------------------------------------|------|
| B   | Proof of Main Theorems                               | Proof of Main Theorems                                        |   22 |
|     | B.1                                                  | Main Theorem for Algorithm 2 with FTRL . . . . . . . . . . .  |   22 |
|     | B.2                                                  | Main Theorem for Algorithm 2 with O-FTRL-FP . . . . . . . .   |   24 |
| C   | PRIMALALLOC: Regret due to Agents' Strategic Reports | PRIMALALLOC: Regret due to Agents' Strategic Reports          |   26 |
|     | C.1                                                  | INTRAEPOCH: Agents Lie to Affect Current-Epoch Allocations    |   26 |
|     | C.2                                                  | INTEREPOCH: Agents Lie to Affect Next-Epoch Dual Variables    |   27 |
| D   | DUALVAR: Regret due to Primal-Dual Framework         | DUALVAR: Regret due to Primal-Dual Framework                  |   31 |
|     | D.1                                                  | DUALVAR and Online Learning Regret . . . . . . . . . . . . .  |   31 |
|     | D.2                                                  | DUALVAR Guarantee for FTRL in Eq. (5) . . . . . . . . . . . . |   34 |
|     | D.3 DUALVAR Guarantee for O-FTRL-FP in Eq. (6) .     | . . . . . . . .                                               |   36 |
|     | D.3.1                                                | O-FTRL-FP Framework for Non-Continuous Predictions            |   40 |
|     | D.3.2                                                | Approximate Continuity of Predictions in O-FTRL-FP .          |   43 |
|     | D.3.3                                                | Stability Term Bounds in O-FTRL-FP . . . . . . . .            |   44 |
| E   | Auxiliary Lemmas                                     | Auxiliary Lemmas                                              |   49 |

## A More Discussions on Related Work

Dynamic Resource Allocation with Non-Strategic Agents. With non-strategic agents, dynamic resource allocation, or more generally, online linear programming, was first studied under the random permutation model where an adversary selects a set of requests that are presented in a random order (Devanur and Hayes, 2009; Feldman et al., 2010; Gupta and Molinaro, 2016), which is more general than our i.i.d. model where all values and costs are identically distributed. For the i.i.d. model, various primal-dual-based algorithms were proposed with the main focus of refining computational efficiency (Agrawal et al., 2014; Devanur et al., 2019; Kesselheim et al., 2018; Li et al., 2023); specifically, Li et al. (2023) proposed a fast O ( T 1 / 2 ) -regret algorithm for the online linear programming problem, which matches the Ω( T 1 / 2 ) online resource allocation lower bound (Arlotto and Gurvich, 2019).

Recent progress on this problem includes o ( T 1 / 2 ) regret with known distributions (Sun et al., 2020) or better robustness and adaptivity to adversarial corruptions (Balseiro et al., 2023; Yang et al., 2024). Another closely related problem is Bandits with Knapsacks (BwK), where the planner makes allocations without observing the values or costs in advance but only learns via post-decision feedback (Badanidiyuru et al., 2018; Castiglioni et al., 2022). Nevertheless, none of them considers strategic behaviors of the agents but assume the values and costs are fully truthful. In contrast, our main focus is to be robust to agents' strategic behaviors while remaining efficient and obeying all the constraints.

Dynamic Resource Allocation with Strategic Agents. When agents are strategic, ensuring both efficiency and incentive-compatibility provide a foundation for static truthful allocation with monetary transfers. In static (one-shot) allocations when money can be redistributed, the celebrated VCG mechanism (Vickrey, 1961; Clarke, 1971; Groves, 1973) provides a foundation way to achieve both. We study the repeated setup where money can only be charged but not redistributed, the so-called 'money-burning' setup. A related problem is learning prices in repeated auctions (Amin et al., 2013, 2014; Kanoria and Nazerzadeh, 2014; Golrezaei et al., 2021a, 2023). These works focus on a seller learning prices to maximize revenue where the agents strategically react to these prices, sometimes

subject to buyer's budget constraints. In our work, we study the different social welfare maximization task and consider more general multi-dimensional cost constraints.

We also briefly discuss the settings where monetary transfers are completely disallowed. In static setups, incentive-compatibility is in general hard due to the Arrow's impossibility theorem (Arrow, 1950; Gibbard, 1973; Satterthwaite, 1975), though some positive results exist under restricted assumptions (Miralles, 2012; Guo and Conitzer, 2010; Han et al., 2011; Cole et al., 2013). Many recent efforts have been made to ensure efficiency and compatibility in repeated non-monetary allocations, which have very different setups from ours, for example when agents' value distributions are known (Balseiro et al., 2019; Gorokh et al., 2021b; Blanchard and Jaillet, 2024), when a pre-determined 'fair share' is revealed to the planner (Gorokh et al., 2021a; Yin et al., 2022; Banerjee et al., 2023; Fikioris et al., 2023), or when the planner has extra power like audits (Jalota et al., 2024; Dai et al., 2025). We also remark that they either do not consider constraints or only have a specific 'fair share' constraint that we discuss later. In contrast, we consider general multi-dimensional constraints.

Multi-Agent Learning. While our planner learns for better allocation mechanisms, the agents are also learning in reaction to it ( e.g. , in our numerical illustration in Figure 1, we consider agents who use Q-learning to learn the best reporting strategies under different dual variables). There is a rich literature investigating this dynamics as well, for example Balseiro and Gur (2019); Golrezaei et al. (2020); Berriaud et al. (2024); Galgana and Golrezaei (2025) studying the convergence to equilibria when multiple agents deploy no-regret online learning algorithms in reaction to some mechanism at the same time. While such results are stronger than our existence of equilibrium results in the sense that agents find such an equilibrium on their own, we remark that the main focus of this work is designing robust mechanisms for the planner instead of designing learning algorithms for the agents.

Comparison with (Yin et al., 2022). Yin et al. (2022) also study the problem of ensuring efficiency and incentive-compatibility in the dynamic constrained resource allocation problem. The first critical difference is that they assume all the agents have identical value distributions, which is crucial for incentive-compatibility: By comparing one agents' reports to all the opponents', unilateral deviation from TRUTH is easily caught and thus TRUTH is a PBE even without using monetary transfers. In contrast, we need more delicate algorithmic components, including epoch-based lazy updates, random exploration rounds, and dual-adjusted allocation and payment plans, to ensure the near-truthfulness of agents. Another main difference is the type of constraints. They study a specific 'fair share' type resource constraint, which says given some p ∈ △ ([ K ]) , the number of allocations that agent i receive should be roughly Tp i , ∀ i ∈ [ K ] . In contrast, our multi-dimensional long-term constraint is strictly more general than theirs, which can be written as 1 T ∑ T t =1 e i t ≤ p in our language.

## B Proof of Main Theorems

## B.1 Main Theorem for Algorithm 2 with FTRL

Theorem B.1 (Formal Version of Theorem 3.1: Algorithm 2 with FTRL) . In Algorithm 2, let

<!-- formula-not-decoded -->

and the sub-routine A chosen as FTRL (Eq. (5)). Then under Assumptions 1 and 3, there exists a PBE of agents' joint strategies under Algorithm 2, denoted by π ∗ , such that

R T ( π ∗ , Algorithm 2 with Eq. (5) )

<!-- formula-not-decoded -->

Specifically, when only focusing on polynomial dependencies on d, K , and T , we have

<!-- formula-not-decoded -->

Proof. As mentioned in the main text, we introduce an intermediate allocation:

<!-- formula-not-decoded -->

Define stopping time T v as the last round where no constraint violations can happen:

<!-- formula-not-decoded -->

Thus in rounds t ≤ T v , Line 14 never rejects our allocation i t . Decompose the regret R T as

<!-- formula-not-decoded -->

From Theorem C.5 presented later in Appendix C, there exists a PBE π ∗ such that

<!-- formula-not-decoded -->

This is the first main technical contribution of our paper, namely justifying that Algorithm 2 equipped with epoch-based lazy updates, uniform exploration rounds, and dual-adjusted allocation and payment plans - is robust to agents' strategic manipulations.

From Theorem D.2 presented later in Appendix D, when setting

<!-- formula-not-decoded -->

and under the same π ∗ , the E [ DUALVAR ] term is bounded by

<!-- formula-not-decoded -->

Thus it only remains to balance the ∑ L ℓ =1 N ℓ = ˜ O ( L ) term and √ ∑ L ℓ =1 |E ℓ | 2 term. Setting L = T 2 / 3 and |E ℓ | = T 1 / 3 for all ℓ ∈ [ L ] ensures √ ∑ L ℓ =1 |E ℓ | 2 = √ T 2 / 3 × T 2 / 3 = T 2 / 3 and thus

<!-- formula-not-decoded -->

Plugging our specific epoching rule that L = T 2 / 3 and |E ℓ | = T 1 / 3 into N ℓ , we get

<!-- formula-not-decoded -->

which gives our claimed bound after rearrangement. For B T , since Line 14 rejects every infeasible allocation, we trivially have B T ( π ∗ , Algorithm 2 with Eq. (5) ) = 0 .

## B.2 Main Theorem for Algorithm 2 with O-FTRL-FP

Theorem B.2 (Formal Version of Theorem 3.2: Algorithm 2 with O-FTRL-FP) . In Algorithm 2, let

<!-- formula-not-decoded -->

and the sub-routine A chosen as O-FTRL-FP (Eq. (6)). Then under Assumptions 1 and 3, there exists a PBE of agents' joint strategies under Algorithm 2, denoted by π ∗ , such that

R T ( π ∗ , Algorithm 2 with Eq. (6) )

<!-- formula-not-decoded -->

and B T ( π ∗ , Algorithm 2 with Eq. (6) ) = 0 , where N ℓ and M ℓ is defined as follows for all ℓ ∈ [ L ] .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Specifically, when only focusing on polynomial dependencies on d, K , and T , we have

<!-- formula-not-decoded -->

Proof. The proof follows the same structure as the previous one, but the treatment of the E [ DUALVAR ] term is extremely challenging and requests delicate analytical tools; we refer the readers to Theorem D.3 for more details. We still include a full proof of this theorem for completeness.

As mentioned in the main text, we introduce an intermediate allocation:

<!-- formula-not-decoded -->

Define stopping time T v as the last round where no constraint violations can happen:

<!-- formula-not-decoded -->

Thus in rounds t ≤ T v , Line 14 never rejects our allocation i t . Decompose the regret R T as

<!-- formula-not-decoded -->

For the PRIMALALLOC term, we still use Theorem C.5 presented later in Appendix C (which holds regardless to the dual update rules). Theorem C.5 asserts that there exists a PBE π ∗ such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

For the DUALVAR term, the O-FTRL-FP analysis is substantially harder. This is the second main technical contribution of our paper. We first recall the O-FTRL-FP update rule from Eq. (6):

<!-- formula-not-decoded -->

We highlight the main technical challenges here and refer the readers to Theorem D.3 for more details:

1. Since ˜ i τ ( λ ℓ ) is not continuous w.r.t. λ ℓ due to the argmax, the predicted loss function ˜ F ℓ ( λ ; λ ℓ ) := ˜ g ℓ ( λ ℓ ) T λ is non-continuous w.r.t. λ ℓ as well - which means an exact fixed point may not exist. In Lemma D.4, we prove that our O-FTRL-FP framework only requires an approximate continuity, which we show ensures the existence of an approximate fixed point.
2. In Lemma D.5, utilizing the smooth cost condition in Assumption 3 and an ϵ -based uniform smoothness argument, we prove that our ˜ g ℓ ( λ ℓ ) T λ is indeed approximately continuous.
3. As agents can misreport in epoch E ℓ , the actual epochℓ loss function F ℓ ( λ ) := ∑ t ∈E ℓ ( ρ -c t,i t ) T λ may differ from the predicted ˜ F ℓ ( λ ; λ ℓ ) . We control this effect in Lemma D.6.
4. Due to the unknown distributions, namely {V i } i ∈ [ K ] and {C i } i ∈ [ K ] , the planner can only use a finite number of samples (more preciously, ∑ ℓ ′ &lt;ℓ |E ℓ ′ | ones) in ˜ i τ ( λ ℓ ) . We control the statistical error in Lemma D.7; however, since λ ℓ is not measurable in round τ , we develop an ϵ -net based uniform smoothness analysis to ensure the statistical error is small for every possible λ ℓ ∈ Λ .
5. Finally, since agents could also misreport in the past, namely epoch E ℓ ′ where ℓ ′ &lt; ℓ , the reports used in ˜ i τ ( λ ℓ ) can also be very different from the true values. In Lemma D.8, we analyze this type of error, again incorporating an ϵ -net based uniform smoothness analysis.

Via careful investigation, Theorem D.3 proves that when configuring

<!-- formula-not-decoded -->

we can control the E [ DUALVAR ] as

<!-- formula-not-decoded -->

where (the definition of N ℓ is the same as that in PRIMALALLOC)

<!-- formula-not-decoded -->

Putting two parts together, we get

<!-- formula-not-decoded -->

Then:

- (i) For any agent i ∈ [ K ] aiming to maximize their expected utility ( v i -p i ) · ✶ [ i t = i ] , truthful reporting u i = v i is a weakly dominant strategy.
- (ii) When all agents report truthfully ( u i = v i ), the allocation i t = argmax i ∈ [ K ] ( v i -c i ) , i.e., it maximizes the value-minus-cost v i -c i across all the agents.

̸

Proof. Define each agent's pseudo-report as ˜ u i := u i -c i , and let ˜ u ∗ i := v i -c i be the truthful pseudo-report. Fix any agent i ∈ [ K ] and suppose all other agents' reports { u j } j = i are fixed. We evaluate the utility U i that agent i obtains under various reporting strategies.

Case 1: Truthfully report u i = v i so that ˜ u i = ˜ u ∗ i :

̸

- If ˜ u ∗ i &lt; max j = i ˜ u j , then i t = i , and U i = 0 .
- If ˜ u ∗ i &gt; max j = i ˜ u j , then i t = i , and U i = v i -c i -max j = i ˜ u j = ˜ u ∗ i -max j = i ˜ u j .

̸

̸

Case 2: Over-reporting u i &gt; v i so ˜ u i &gt; ˜ u ∗ i :

̸

̸

- If max j = i ˜ u j &gt; ˜ u i &gt; ˜ u ∗ i , U i = 0 (same as truthful).

̸

- If ˜ u i &gt; ˜ u ∗ i &gt; max j = i ˜ u j , U i = ˜ u ∗ i -max j = i ˜ u j (same as truthful).

̸

- Otherwise if ˜ u i &gt; max j = i ˜ u j &gt; ˜ u ∗ i , U i = ˜ u ∗ i -max j = i ˜ u j &lt; 0 (worse than truthful).

Case 3: Under-reporting u i &lt; v i so ˜ u i &lt; ˜ u ∗ i :

̸

̸

- If max j = i ˜ u j &gt; ˜ u ∗ i &gt; ˜ u i , U i = 0 (same as truthful).

̸

- If ˜ u ∗ i &gt; ˜ u i &gt; max j = i ˜ u j , U i = ˜ u ∗ i -max j = i ˜ u j (same as truthful).

̸

- Otherwise, if ˜ u ∗ i &gt; max j = i ˜ u j &gt; ˜ u i , U i = 0 . But they originally get ˜ u ∗ i -max j = i ˜ u j ≥ 0 .

̸

In all cases, deviating from truth-telling does not improve agent i 's utility, and may strictly reduce it. Hence, truthful reporting is a weakly dominant strategy, proving claim (i) .

For claim (ii) , when all agents report truthfully ( u i = v i ), the planner allocates the resource to i t = arg max i ( v i -c i ) , which maximizes net social value.

Theorem C.2 (Intra-Epoch Truthfulness; Formal Theorem 4.1) . Fix any epoch E ℓ ⊆ [ T ] and dual variable λ ℓ ∈ Λ . Suppose that all the agents, when crafting their reports only consider their discounted gains within the current epoch (Model 2 in Eq. (8)), namely Eq. (11). To highlight the

<!-- formula-not-decoded -->

The bound that B T ( π ∗ , Algorithm 2 with Eq. (6) ) = 0 directly follows from Line 14 of Algorithm 2.

<!-- formula-not-decoded -->

## C PRIMALALLOC: Regret due to Agents' Strategic Reports

## C.1 INTRAEPOCH: Agents Lie to Affect Current-Epoch Allocations

Lemma C.1 (Truthfulness of a Cost-Adjusted Second-Price Auction) . Consider a one-shot monetary allocation setting with K agents. Each agent i ∈ [ K ] privately observes their value v i ∼ V i and publicly incurs a known cost c i ∼ C i . Agents submit scalar reports u i ∈ [0 , 1] , and the planner allocates the item to one agent i t ∈ [ K ] and charges payment p i t . The utility of the selected agent is v i t -p i t , while all others receive zero utility.

Suppose the planner implements the following mechanism:

<!-- formula-not-decoded -->

̸

̸

̸

̸

different reports, allocations, and payments due to the different agent model, we add a superscript h .

<!-- formula-not-decoded -->

The planner, on the other hand, uses the allocation and payment rule in Line 12. Formally, in round t ∈ E ℓ the planner allocates to the agent with maximal dual-adjusted report - which we denote by i h t to highlight its difference with i t due to the different agent model - and sets payments accordingly:

̸

<!-- formula-not-decoded -->

Then, truthful reporting u t,i = v t,i for all t ∈ E ℓ and all i ∈ [ K ] - a joint strategy denoted as TRUTH - constitutes a Perfect Bayesian Equilibrium (PBE). Furthermore, under this PBE, the planner always chooses the optimal agent according to dual-adjusted value:

<!-- formula-not-decoded -->

and the regret due to intra-epoch misallocations is zero, i.e., E [ INTRAEPOCH ] = 0 .

Proof. For any round t ∈ E ℓ , we observe that the planner's allocation and payment plan ( i h t , p h t ) depends only on the current reports u h t , current costs c t , and the fixed dual variable λ ℓ . Specifically, it does not depend on historical reports { u h τ } τ&lt;t , past allocations { i h τ } τ&lt;t , or payments { p h τ } τ&lt;t .

̸

Likewise, for any fixed agent i ∈ [ K ] , suppose that all the opponents follow truthful reporting TRUTH -i , i.e. , ( u t,j = v t,j for all t ∈ E ℓ and j = i ). In this case, for any round t ∈ E ℓ whose value is v t,i ∈ [0 , 1] , the expected gain of any tentative report u t,i ∈ [0 , 1] does not depend on the history but only on u h t , c t , λ ℓ , and v t,i . Therefore, agent i does not benefit from unilaterally deviating to a 'history-dependent' strategy that depends on previous public or private information, namely { u h τ } τ&lt;t , { i h τ } τ&lt;t , { p h τ } τ&lt;t , and { v τ,i } τ&lt;t .

Therefore we only need to consider agent i 's potential unilateral deviation to history-independent policies, which means we can isolate each round t ∈ E ℓ . From Lemma C.1, we know that in any such round t , given fixed costs and fixed dual λ ℓ , truthful reporting maximizes an agent's expected utility regardless of opponents' actions. Hence, no agent can benefit from deviating - whether using a history-dependent strategy or a history-independent one - and thus TRUTH is a PBE.

Finally, under PBE TRUTH, the planner allocates to the agent with maximal v t,i -λ T ℓ c t,i , i.e. , i h t = ˜ i ∗ t . Therefore, there is no misallocation in the epoch and thus INTRAEPOCH = 0 .

## C.2 INTEREPOCH: Agents Lie to Affect Next-Epoch Dual Variables

The key ideas of Lemmas C.3 and C.4 are largely motivated by Golrezaei et al. (2021a, 2023). The main difference is due to the costs { c t } t ∈ [ T ] , which forbids us from using their results as black-boxes.

For epoch ℓ ∈ [ L ] , consider the 'epochℓ game with exploration rounds' induced by Lines 4 to 12 in Algorithm 2. Theorem C.2 proves that under Model 2 in Eq. (8) ( i.e. , agents only care about current-epoch gains) and when there are no exploration rounds, TRUTH is a PBE. Now, we claim that under Model 1 in Eq. (8) ( i.e. , agents optimize over the whole future) and the actual mechanism with exploration rounds (Lines 4 to 12 in Algorithm 2), there exists a PBE of agents' joint strategies π that is not too far from TRUTH. Formally, we present Lemma C.3.

Lemma C.3 (Large Misreport Happens Rarely) . For an epoch ℓ ∈ [ L ] , consider the agent Model 1 in Eq. (8) and the 'epochℓ game with exploration rounds' specified by Lines 4 to 12 in Algorithm 2. There exists a PBE of agents' joint strategies π , such that for all i ∈ [ K ] ,

<!-- formula-not-decoded -->

where { u t,i } t ∈E ℓ are the reports made by agent i under π .

Proof. Consider the history-independent auxiliary game defined in the proof of Theorem C.2, where for every round t ∈ E ℓ , agent i ∈ [ K ] is only allowed to craft their reports u t,i based on current-epoch dual variable λ ℓ , round number t , current-round private value v t,i , and current-round public costs c t .

Let π be a PBE in this history-independent auxiliary game. Using same arguments from Theorem C.2, unilaterally deviating to a history-dependent strategy is not beneficial as all opponents' strategies π -i and the mechanism (Lines 4 to 12 in Algorithm 2) are history-independent. Hence, π remains a PBE in the actual 'epochℓ game with exploration rounds' where history dependency is allowed.

To prove the claim for this PBE π , consider the unilateral deviation of any agent i ∈ [ K ] to the truth-telling policy, i.e. , π i := TRUTH i ◦ π -i . Since π is a PBE, π i is no better than π under Model 1 (the actual model). That is, for s ℓ := min { t | t ∈ E ℓ } and any history H s ℓ , we always have

<!-- formula-not-decoded -->

For the second term, fix any ℓ ′ &gt; ℓ and τ ∈ E ℓ ′ . Since the values v τ , reports u τ , and costs c τ are all bounded by [0 , 1] , and that λ ℓ ′ ∈ Λ = ⊗ d j =1 [0 , ρ -1 j ] (which infers ∥ λ ℓ ′ ∥ 1 ≤ ∥ ρ -1 ∥ 1 ), we have

̸

<!-- formula-not-decoded -->

for all ℓ ′ &gt; ℓ, τ ∈ E ℓ ′ . Since v t,i ∈ [0 , 1] , this suggests that v τ,i -p τ,i ∈ [ -1 -2 ∥ ρ -1 ∥ 1 , 1+2 ∥ ρ -1 ∥ 1 ] for all i ∈ [ K ] , and thus

<!-- formula-not-decoded -->

where the second inequality uses the fact that ∑ T τ = s ℓ +1 γ τ ≤ γ s ℓ +1 1 -γ .

We now focus on the current-epoch difference part. Since both π and π i are history independent (recall that π is a joint strategy from the history-independent auxiliary game and π i = TRUTH i ◦ π -i ), the conditioning on H s ℓ is redundant and consequently

<!-- formula-not-decoded -->

For any round t ∈ E ℓ , we control this difference utilizing the exploration rounds:

- Suppose that round t ∈ E ℓ is an exploration round for agent i (which happens with probability 1 |E ℓ | · 1 K ), the expected gain of reporting u t,i rather than v t,i ( i.e. , under π versus under π i ) is
- If round t is an exploration round but not for agent i , then reporting u t,i and v t,i both give 0 gain.
- Finally, suppose that round t ∈ E ℓ is not an exploration round. Notice that i) both π and π -i are history-independent, and ii) if round t ∈ E ℓ is not an exploration round, then the epochℓ game in Lines 4 to 12 in Algorithm 2 coincides with Line 12. Therefore, via Lemma C.1, the gain of reporting u t,i is no larger than that of reporting v t,i , i.e. , the expectation difference is non-positive.

<!-- formula-not-decoded -->

.

Let M ℓ,i := { t ∈ E ℓ | | u t,i -v t,i | ≥ 1 |E ℓ | } be the set of large misreports from agent i (regardless of whether t turns out to be an exploration round, since reports happen before it). Let the last round in epoch E ℓ be e ℓ := max { t | t ∈ E ℓ } . Since ∑ t ∈M ℓ,i γ t ≥ ∑ e ℓ t = e ℓ -|M ℓ,i | +1 γ t , we have

<!-- formula-not-decoded -->

In order for π i to be inferior when compared with π , i.e. , Eq. (12) holds, we therefore must have

<!-- formula-not-decoded -->

where the last step uses the fact that s ℓ +1 = e ℓ +1 .

Picking c such that 2(1 + 2 ∥ ρ -1 ∥ 1 ) 2 K |E ℓ | 3 γ -c -1 = 1 |E ℓ | , we reach our conclusion that

<!-- formula-not-decoded -->

This completes the proof.

Still focusing on a fixed epoch ℓ ∈ [ L ] , Lemma C.3 bounds the number of rounds with large misreports. We now turn to the remaining rounds, i.e. , those t ∈ E ℓ such that | u t,i -v t,i | ≤ 1 |E ℓ | for all i ∈ [ K ] . We claim that misallocations are rare among these rounds, formalized in Lemma C.4.

Lemma C.4 (Misallocation with Small Misreports Happens Rarely) . Consider an epoch ℓ ∈ [ L ] with dual variable λ ℓ ∈ Λ . We have

̸

<!-- formula-not-decoded -->

Proof. For any round t ∈ E ℓ , we bound the probability that the allocation based on reported utilities differs from that based on true values, even when reports are close to truthful. Specifically, consider the event

̸

<!-- formula-not-decoded -->

̸

Such a mismatch can only happen if there exists a pair of indices i = j whose true dual-adjusted values are very close - within 2 |E ℓ | - so that small deviations in reported utilities (bounded by 1 |E ℓ | ) are able to flip the argmax decision. We apply a union bound over all such pairs to upper bound this probability:

̸

<!-- formula-not-decoded -->

where the second inequality comes from Assumption 3: Since it assumes that PDF( λ T ℓ c t,i ) is uniformly upper bounded by ϵ c where λ ℓ ∈ Λ , we know

<!-- formula-not-decoded -->

Now consider the martingale difference sequence { X t -E [ X t | F t -1 ] } t ∈E ℓ where

̸

<!-- formula-not-decoded -->

We apply the multiplicative Azuma-Hoeffding inequality restated as Lemma E.4 (Koufogiannakis and Young, 2014, Lemma 10) with Y t = E [ X t | F t -1 ] , ϵ = 1 2 , and A = 2log |E ℓ | . Since X t ∈ [0 , 1] a.s. and E [ X t -Y t | F t -1 ] = E [ X t -E [ X t | F t -1 ] | F t -1 ] = 0 , the two conditions in Lemma E.4 hold and thus

<!-- formula-not-decoded -->

From Eq. (14), we know E [ X t | F t -1 ] = Pr { X t | F t -1 } ≤ 2 K 2 |E ℓ | ϵ c ∥ λ ℓ ∥ 1 . Therefore, rearranging the above concentration result gives

<!-- formula-not-decoded -->

Plugging back the definition of X t completes the proof.

Putting the previous two parts together for all ℓ ∈ [ L ] gives the following theorem.

Theorem C.5 (INTEREPOCH Guarantee; Formal Theorem 4.2) . Under the mechanism specified in Algorithm 2, there exists a PBE of agents' strategies π ∗ , such that the PRIMALALLOC term (which is the sum of INTRAEPOCH and INTEREPOCH terms) is bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Applying Lemma C.3 to every epoch ℓ ∈ [ L ] , we get a PBE π ℓ for every 'epochℓ game with exploration rounds' (Lines 4 to 12 in Algorithm 2). By definition of T v , the safety constraint is never violated before that and thus Line 14 has no effect. Furthermore, since Algorithm 2's allocations and payments within every epoch ℓ ∈ [ L ] only directly depend on the current dual variable λ ℓ but not anything else from the past, using the same auxiliary game arguments as in Theorem C.2, there is a PBE π ∗ for the whole game under mechanism Algorithm 2 that matches ( π ℓ ) ℓ ∈ [ L ] up to T v .

For every epoch t ∈ E ℓ , it can be either i) an exploration round, which happens w.p. 1 |E ℓ | independently, ii) a standard round with large misreports: ∃ i ∈ [ K ] such that | u t,i -v t,i ≥ 1 |E ℓ | , or iii) a standard round with only small misreports. For ii) , we apply Lemma C.3; for iii) , we apply Lemma C.4. For i) , applying Chernoff inequality,

<!-- formula-not-decoded -->

Setting c = log |E ℓ | so that the RHS is no more than 1 |E ℓ | , we get

<!-- formula-not-decoded -->

Now we put the aforementioned three cases together:

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

where (a) uses the fact that v t,i ∈ [0 , 1] for all t and i , (b) uses the above discussions of (i) , (ii) , and (iii) , and (c) applied Line 9 to (i) , Lemma C.3 to (ii) , Lemma C.4 to (iii) , and the trivial bound that ∑ t ∈E ℓ ✶ [ ˜ i ∗ t = i t ] ≤ |E ℓ | if any of the conclusions in Line 9 or Lemmas C.3 and C.4 do not hold (every conclusion holds with probability 1 -1 |E ℓ | , and thus a Union Bound controls the overall failure probability by 3 |E ℓ | ). This completes the proof.

## D DUALVAR: Regret due to Primal-Dual Framework

## D.1 DUALVAR and Online Learning Regret

Lemma D.1 (DUALVAR and Online Learning Regret; Formal Lemma 4.3) . Under Algorithm 2, the DUALVAR term can be controlled as follows:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. This lemma largely adopts Theorem 1 of Balseiro et al. (2023), but we incorporate some arguments of Castiglioni et al. (2022) to fix a measurability issue in the original proof.

Slightly abusing the notations, for any round t belonging to an epoch E ℓ , we define λ t := λ ℓ . Let ( F t ) t ≥ 0 be the filtration specified as F t = σ ( λ 1 , . . . , λ t , v 1 , . . . , v t , c 1 , . . . , c t ) .

For any t ∈ [ T ] , let v ∗ t be the convex conjugate (Fenchel dual) function of v t , namely

<!-- formula-not-decoded -->

which is convex in λ since it's the maximum of convex functions. We further define v ∗ ( λ ) = E v ∼V , c ∼C [max i ∈ [ K ] ( v i -λ T c i )] , which is an expectation of a convex function and thus also convex.

Step 1: Lower bound the values collected by { ˜ i ∗ t } t ∈ [ T v ] . By definition of ˜ i ∗ t from Eq. (9) that ˜ i ∗ t = argmax i ∈ [ K ] ( v t,i -λ T t c t,i ) , we have

<!-- formula-not-decoded -->

Since λ t is F t -1 -measurable but v t and c t are sampled from V and C independently to F t -1 ,

<!-- formula-not-decoded -->

Put this equality in another way, the stochastic process ( X t ) t ≥ 1 adapted to ( F t ) t ≥ 0 defined as

<!-- formula-not-decoded -->

ensures E [ X t | F t -1 ] = 0 and is thus a martingale difference sequence. Since T v ≤ T +1 a.s. by definition from Eq. (10), Optional Stopping Time theorem (Williams, 1991, Theorem 10.10) gives

<!-- formula-not-decoded -->

We now utilize the convexity of v ∗ , which is the expectation of a convex function, and conclude that

<!-- formula-not-decoded -->

where (a) uses Jensen's inequality and (b) uses the fact that λ = 1 T v ∑ T v τ =1 λ τ ∈ Λ because λ τ ∈ Λ for all τ and the fact that Λ = ⊗ d j =1 [0 , ρ -1 j ] is convex.

Step 2: Upper bound the offline optimal social welfare ∑ T t =1 v t,i ∗ t . We now work on the other term in E [ DUALVAR ] , namely E [ ∑ T t =1 v t,i ∗ t ] . For any fixed λ ∈ Λ , since v ∗ t ( λ ) = max i ∈ [ K ] ( v t,i -λ T c t,i ) ≥ v t,i ∗ t -λ T c t,i ∗ t for all t ∈ [ T ] , we have

<!-- formula-not-decoded -->

Further recall from Eq. (2) that { i ∗ t } t ∈ [ T ] is the optimum of the offline optimization problem

<!-- formula-not-decoded -->

it ensures 1 T ∑ T t =1 c t,i ∗ t ≤ ρ . Therefore, for any fixed λ ∈ Λ , we further know

<!-- formula-not-decoded -->

where (a) uses Eq. (18), (b) rearranges the terms, (c) uses 1 T ∑ T t =1 c t,i ∗ t ≤ ρ , (d) uses the definition that v ∗ ( λ ) = E v ∼V , c ∼C [max i ∈ [ K ] ( v i -λ T c i )] (and thus E [ v t ( λ )] = v ∗ ( λ ) for any fixed λ ∈ Λ ).

Taking infimum of λ ∈ Λ and further recalling that all the values v t,i ∗ t are [0 , 1] -bounded, we get

<!-- formula-not-decoded -->

Step 3: Combine Steps 1 and 2. Putting Eqs. (17) and (19) together, we get

<!-- formula-not-decoded -->

Consider the following stochastic process ( Y t ) t ≥ 1 adapted to ( F t ) t ≥ 0 :

<!-- formula-not-decoded -->

we must have E [ Y t | F t -1 ] ≤ 0 since λ t ∈ Λ , which means ( Y t ) t ≥ 1 is a super-martingale difference sequence. Again utilizing the fact that T v ≤ ( T +1) a.s. and the Optional Stopping Time theorem (Williams, 1991, Theorem 10.10), we know

<!-- formula-not-decoded -->

This reveals that

<!-- formula-not-decoded -->

Comparing Eq. (20) to our conclusion, it only remains to associate λ T t c t, ˜ i ∗ t with λ T t c t,i t and control E [ T -T v ] . We first focus on the former objective.

Step 4: Relate λ T t ( ρ -c t, ˜ i ∗ t ) to λ T t ( ρ -c t,i t ) . Using Eq. (16) from Theorem C.5, with probability 1 -3 |E ℓ | , the sequence { ˜ i ∗ t } t ∈E ℓ and { i t } t ∈E ℓ only differs by no more than N ℓ , where

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

That is, we have shown than E [ ∑ T v t =1 ✶ [ ˜ i ∗ t = i t ] ] ≤ ∑ L ℓ =1 ( N ℓ +3) where the 3 comes from the 3 |E ℓ | failure probability and the fact that ∑ t ∈E ℓ [ ˜ i ∗ t = i t ] ≤ |E ℓ | . Combining it with the observation that

̸

where we shall recall that c t,i , c t,j ∈ [0 , 1] d and that λ t ∈ Λ = ⊗ d j =1 [0 , ρ -1 j ] , Eq. (20) gives

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

Step 5: Control E [ T -T v ] . We recall the definition of T v from Eq. (10):

<!-- formula-not-decoded -->

If T v = T +1 , then ( T -T v ) is trivially bounded. Otherwise, suppose that ∑ T v τ =1 c τ,i τ + 1 ̸≤ T ρ is violated for the j ∈ [ d ] -th coordinate (if there are multiple j 's, pick one arbitrarily). We have

<!-- formula-not-decoded -->

Let λ ∗ = 1 ρ j e j where e j is the one-hot vector over coordinate j , we know λ ∗ ∈ Λ and that

<!-- formula-not-decoded -->

Rearranging gives

<!-- formula-not-decoded -->

Final Bound. Taking expectation and plugging it back to Eq. (21), we yield

<!-- formula-not-decoded -->

This completes the proof.

## D.2 DUALVAR Guarantee for FTRL in Eq. (5)

Theorem D.2 (DUALVAR Guarantee with FTRL) . When using Follow-the-Regularized-Leader (FTRL) in Eq. (5) to decide { λ ℓ } ℓ ∈ [ L ] , the online learning regret is no more than

<!-- formula-not-decoded -->

Specifically, when setting

<!-- formula-not-decoded -->

the E [ DUALVAR ] term is bounded by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Proof. We apply the FTRL guarantee stated as Lemma E.1 with their decision region X as our dual decision region Λ = ⊗ d j =1 [0 , ρ -1 j ] , their norm ∥·∥ as ℓ 2 -norm, their round number R as our epoch number L , their roundℓ loss function f ℓ as our observed loss F ℓ ( λ ) := ∑ t ∈E ℓ ( ρ -c t,i t ) T λ , the FTRL decisions { λ ℓ } ℓ ∈ [ L ] suggested by Lemma E.1 recover our dual-decision rule in Eq. (5), i.e. ,

<!-- formula-not-decoded -->

Since ∇ λ F ℓ ( λ ) = ∑ t ∈E ℓ ( ρ -c t,i t ) and the dual norm of ℓ 2 -norm is still ℓ 2 -norm, Lemma E.1 gives

<!-- formula-not-decoded -->

Recalling that ρ , c t,i ∈ [0 , 1] d , we have ∥ ∑ t ∈E ℓ ( ρ -c t,i t ) ∥ 2 2 ≤ d |E ℓ | 2 and hence

<!-- formula-not-decoded -->

This gives the first conclusion in this theorem.

We now move on to the second conclusion in this theorem, namely the online learning regret under the given configuration of Ψ( λ ) = 1 2 ∥ λ ∥ 2 2 and η ℓ = ∥ ρ -1 ∥ 2 √ 2 d ( ∑ ℓ ℓ ′ =1 |E ℓ ′ | 2 ) -1 / 2 . First of all,

<!-- formula-not-decoded -->

Plugging in the specific choice that η ℓ = ∥ ρ -1 ∥ 2 √ 2 d ( ∑ ℓ ℓ ′ =1 |E ℓ ′ | 2 ) -1 / 2 , we therefore have

<!-- formula-not-decoded -->

where (a) uses the folklore summation lemma that ∑ T t =1 x t √ ∑ t s =1 x s ≤ 2 √ ∑ T t =1 x t for all x 1 , x 2 , . . . , x T ∈ R ≥ 0 (Duchi et al., 2011, Lemma 4) and (b) follows from rearranging the terms.

Plugging the online learning regret R λ L into Lemma D.1, we therefore get

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

This finishes the proof.

## D.3 DUALVAR Guarantee for O-FTRL-FP in Eq. (6)

Theorem D.3 (DUALVAR Guarantee with O-FTRL-FP) . When using Optimistic Follow-theRegularized-Leader with Fixed-Points (O-FTRL-FP) in Eq. (6) to decide { λ ℓ } ℓ ∈ [ L ] , the online learning regret is no more than 5

<!-- formula-not-decoded -->

where for any ℓ ∈ [ L ] , N ℓ and M ℓ are defined as follows:

<!-- formula-not-decoded -->

Specifically, when setting

<!-- formula-not-decoded -->

the E [ DUALVAR ] term is bounded by

<!-- formula-not-decoded -->

Proof. We would like to apply the O-FTRL-FP guarantee stated as Lemma D.4. One main challenge we face is the discontinuity of our predictions; recall the O-FTRL-FP dual update rule from Eq. (6):

<!-- formula-not-decoded -->

Step 1: Make sure Lemma D.4 is applicable. Since predicted loss term ˜ g ℓ ( λ ℓ ) T λ - where ˜ g ℓ ( λ ℓ ) is the estimated gradient from historical reports and costs - is not continuous w.r.t. λ ℓ due to the argmax in ˜ i τ ( λ ℓ ) , we cannot directly apply the Brouwer's Fixed Point theorem (Munkres, 2000,

5 For the readers to better interpret this long inequality, we provide an ˜ O T version as Eq. (29) in the proof.

Theorem 55.6) to conclude the existence of an exact fixed point λ ℓ . Fortunately, in Lemma D.4, we prove the existence of an ( η ℓ L ℓ ϵ ℓ ) -approximate fixed point given ( ϵ ℓ , L ℓ ) -approximate continuity, which requires the existence of two constants ( ϵ ℓ , L ℓ ) such that

<!-- formula-not-decoded -->

We refer the readers to Lemma D.4 for a more general version of the ( ϵ ℓ , L ℓ ) -approximate continuity condition that we propose, which generalizes to non-linear predicted losses. In Lemma D.5, we prove

<!-- formula-not-decoded -->

for any fixed constant ϵ ℓ &gt; 0 and δ ℓ ∈ (0 , 1) . Roughly speaking, Lemma D.5 first picks an ϵ ℓ 2 -net of Λ and then utilizes the smooth cost condition (Assumption 3) to conclude that close λ 's give similar ˜ i τ ( λ ) 's for all τ , which consequently give similar ˜ g ℓ ( λ ) 's within every small ball in the net.

Properly tuning ϵ ℓ to minimize the RHS in Eq. (23), it translates to the ( ϵ ℓ , L ℓ ) -approximate continuity of prediction ˜ g ℓ ( λ ℓ ) T λ w.r.t. λ ℓ where

<!-- formula-not-decoded -->

Under this specific configuration of ϵ ℓ and L ℓ , we call the event defined in in Eq. (23) G ℓ . Conditioning on G 1 , . . . , G L which happens with probability at least 1 -∑ L ℓ =1 δ ℓ (the conditioning is valid because G ℓ only depends on a fixed ϵ ℓ 2 -net of Λ and historical reports and costs, and is thus measurable before the start of epoch E ℓ ), we can apply the O-FTRL-FP guarantee stated as Lemma D.4.

Specifically, set their decision region X as our dual decision region Λ = ⊗ d j =1 [0 , ρ -1 j ] , their norm ∥·∥ as ℓ 2 -norm, their round number R as our epoch number L , their roundℓ loss function f ℓ as our observed loss F ℓ ( λ ) := ∑ t ∈E ℓ ( ρ -c t,i t ) T λ , their prediction ˜ f ℓ ( λ ℓ , λ ) as our ˜ g ℓ ( λ ℓ ) T λ . The O-FTRL-FP decisions { λ ℓ } ℓ ∈ [ L ] suggested by Lemma D.4 recover our dual-decision rule in Eq. (5):

<!-- formula-not-decoded -->

where the ≈ means the ( η ℓ L ℓ ) -approximate fixed point suggested by Lemma D.4.

Further noticing that ∇ λ F ℓ ( λ ) = ∑ t ∈E ℓ ( ρ -c t,i t ) , ∇ λ ( ˜ g ℓ ( λ ℓ ) T λ ) = ˜ g ℓ ( λ ℓ ) , the loss F ℓ ( λ ) is √ d |E ℓ | -Lipschitz since ( ρ -c t,i t ) ∈ [0 , 1] d , the loss and predictions are 0 -smooth w.r.t. λ since they are linear, and the dual norm of ℓ 2 -norm is still ℓ 2 -norm, Lemma D.4 gives

<!-- formula-not-decoded -->

where the last term considers the failure probability of G 1 , . . . , G L , in which case we use the trivial bound that ∑ T t =1 ( ρ -c t,i t ) T ( λ t -λ ∗ ) ≤ T · (max t,i ∥ ρ -c t,i ∥ ∞ )(2 sup λ ∈ Λ ∥ λ ∥ 1 ) ≤ 2 T ∥ ρ -1 ∥ 1 , where the last inequality is due to ρ -c t,i ∈ [ -1 , 1] 2 and λ ∈ ⊗ d j =1 [0 , ρ -j ] .

Step 2: Control the Stability terms. For analytical convenience, we add a superscript u to the notation ˜ g ℓ ( λ ) to highlight it is yielded from reports { u τ } ℓ ′ &lt;ℓ,τ ∈E ℓ ′ . Analogous to ˜ g u ℓ ( λ ) , which

is computed from agents' strategic reports { u τ } ℓ ′ &lt;ℓ,τ ∈E ℓ ′ , we define ˜ g v ℓ ( λ ) using the true values { v τ } ℓ ′ &lt;ℓ,τ ∈E ℓ ′ , and ˜ g ∗ ℓ ( λ ) using the underlying true distributions V = {V i } i ∈ [ K ] and C = {C i } i ∈ [ K ] :

<!-- formula-not-decoded -->

We now decompose the Stability ℓ term above as

<!-- formula-not-decoded -->

In Lemmas D.6 to D.8 presented immediately after this theorem, we control these three terms one by one. Specifically, Lemma D.6 relates ˜ g ∗ ℓ ( λ ℓ ) first to ∑ t ∈E ℓ ( ρ -c t, ˜ i ∗ t ) and then to ∑ t ∈E ℓ ( ρ -c t,i t ) , which gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Meanwhile, in Lemmas D.7 and D.8, by an ϵ -net argument, we first prove ˜ g ∗ ℓ ( λ ) ≈ ˜ g v ℓ ( λ ) ( resp. ˜ g v ℓ ( λ ) ≈ ˜ g u ℓ ( λ ) ) for all λ 's in the ϵ -net and then extend this similarity to all λ ∈ Λ using the smooth cost condition; we refer the readers to corresponding proofs for more details. Summing up their conclusions and taking Union Bound, they together ensure that for any ϵ &gt; 0 and δ ∈ (0 , 1) , with probability 1 -6 δ , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Step 3: Plug Stability back to O-FTRL-FP guarantee. Now we plug the Stability bounds from Eqs. (26) and (27) into the online learning regret bound derived in Eq. (24). For every ℓ ∈ [ L ] , Eq. (27) happen with probability 1 -6 δ ; in case it does not hold, we use the trivial bound that ∥ ˜ g ∗ ℓ ( λ ) -˜ g v ℓ ( λ ) ∥ 2 2 + ∥ ˜ g v ℓ ( λ ) -˜ g u ℓ ( λ ) ∥ 2 2 ≤ 2 d |E ℓ | 2 . Therefore, Eq. (24) translates to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ϵ ℓ = √ d K 2 ϵ c ∑ ℓ ′ &lt;ℓ |E ℓ ′ | and { δ ℓ } ℓ ∈ [ L ] , ϵ , and δ are some parameters that we can tune. We also recall the definitions of N ℓ and M ℓ :

<!-- formula-not-decoded -->

Step 4: Deriving first half of this theorem. We now configure ϵ , δ , and δ ℓ 's as follows:

<!-- formula-not-decoded -->

We remark that we did not make every effort to make the overall online learning regret as small as possible. Instead, the above tuning mainly focuses on polynomial dependencies on T and L .

Under this specific tuning and simplifying using the fact that ∑ L ℓ =1 |E ℓ | ≤ T , we get

<!-- formula-not-decoded -->

For the readers to better interpret, we annotate the order of every term in terms of ˜ O T , which only highlights the polynomial dependency on T , and consequently, also L , {|E ℓ |} ℓ ∈ [ L ] , and { η ℓ } ℓ ∈ [ L ] :

<!-- formula-not-decoded -->

Step 5: Plug in specific tuning of Ψ , L, {E ℓ } ℓ ∈ [ L ] , and { η ℓ } ℓ ∈ [ L ] . We now plug in the following specific configuration:

<!-- formula-not-decoded -->

Due to the doubling epoch length structure, we observe that

<!-- formula-not-decoded -->

Therefore the informal bound in Eq. (29) becomes ˜ O T ( η -1 L + ∑ L ℓ =1 η ℓ |E ℓ | ) . This explains our choice that η ℓ = ( ∑ ℓ ℓ ′ =1 |E ℓ ′ | ) -1 / 2 . To make it formal, substituting |E ℓ | ∑ ℓ -1 ℓ ′ =1 |E ℓ ′ | ≤ 2 into Eq. (28):

<!-- formula-not-decoded -->

Under the specific choice of Ψ( λ ∗ ) = 1 2 ∥ λ ∥ 2 2 and that η ℓ = ( ∑ ℓ ℓ ′ =1 |E ℓ ′ | ) -1 / 2 , we get

<!-- formula-not-decoded -->

where the second inequality again uses the folklore summation lemma that ∑ T t =1 x t √ ∑ t s =1 x s ≤

2 √ ∑ T t =1 x t for all x 1 , x 2 , . . . , x T ∈ R ≥ 0 (Duchi et al., 2011, Lemma 4); we also used the trivial bound that η ℓ ≤ 1 for the non-dominant terms. Once again, we remark that the final bound is only optimized w.r.t. poly ( T ) dependencies.

To translate the online learning regret R λ L to the E [ DUALVAR ] guarantee, we use Lemma D.1 (we also usd the fact that ∑ L ℓ =1 |E ℓ | = T ):

<!-- formula-not-decoded -->

This finishes the proof.

## D.3.1 O-FTRL-FP Framework for Non-Continuous Predictions

Lemma D.4 (O-FTRL-FP Guarantee) . For a convex and compact region X within an Euclidean space R d , an 1 -strongly-convex regularizer Ψ: X → R w.r.t. some norm ∥·∥ with min x ∈X Ψ( x ) = 0 , a sequence of continuous, differentiable, convex, L -Lipschitz, and L f -smooth losses f 1 , f 2 , . . . , f R , a sequence of learning rates η 1 ≥ η 2 ≥ · · · ≥ η R ≥ 0 , a (stochastic) action-dependent prediction sequence { ˜ f r : X × X → R } r ∈ [ R ] such that the following conditions hold:

1. ˜ f r is F r -1 -measurable where ( F r ) t is the natural filtration that F r = σ ( f 1 , f 2 , . . . , f r ) ,
2. For any fixed y ∈ X , ˜ f r ( y , · ) is continuous, differentiable, convex, and L f -smooth, and
3. ˜ f r ( y , x ) is ( ϵ r , L r ) -approximately-continuous w.r.t. its first parameter y in the sense that

<!-- formula-not-decoded -->

where ∇ 2 ˜ f r is the gradient of ˜ f r ( y , x ) taken only w.r.t. the second parameter x , and ∥·∥ ∗ is the dual norm of ∥·∥ .

Then, for all r = 1 , 2 , . . . , R , consider the following fixed-point system:

<!-- formula-not-decoded -->

First of all, the argmin in the RHS of Eq. (30) exists and is unique. Furthermore, Eq. (30) allows an ( η r L r ) -approximate fixed point x r such that

<!-- formula-not-decoded -->

Using this { x r } r ∈ [ R ] , we have the following guarantee for all x ∗ ∈ X :

<!-- formula-not-decoded -->

Proof. We first study the system Eq. (30) for any fixed r ∈ [ R ] . For simplicity, we drop the subscripts r from F ( x ) and G ( y ) defined soon. Since Ψ( x ) is 1-strongly-convex and f r ( x ) is convex, ∀ r &lt; r ,

<!-- formula-not-decoded -->

Fix any y 1 , y 2 ∈ X such that ∥ y 1 -y 2 ∥ &lt; ϵ r . From Condition 2 of ˜ f r , H 1 ( x ) := F ( x ) + ˜ f r ( y 1 , x ) and H 2 ( x ) := F ( x ) + ˜ f r ( y 2 , x ) are continuous, differentiable, and η -1 r -strongly-convex. Hence x 1 := argmin x ∈X H 1 ( x ) and x 2 := argmin x ∈X H 2 ( x ) exist and are unique. Further utilizing the η -1 r -strong-convexity of H 1 , we have (Bertsekas et al., 2003, Exercise 1.10)

<!-- formula-not-decoded -->

where (a) uses the first-order condition of H 1 ( x 1 ) that ⟨∇ H 1 ( x 1 ) , x 2 -x 1 ⟩ ≥ 0 , (b) uses the first-order condition of H 2 ( x 2 ) that ⟨∇ H 2 ( x 2 ) , x 1 -x 2 ⟩ ≥ 0 , and (c) applies Cauchy-Schwartz inequality. Using Condition 3 of ˜ f r and rearranging terms, we further have

<!-- formula-not-decoded -->

For any y ∈ X , due to strong convexity of F ( x ) + ˜ f r ( y , x ) , the following G ( y ) is well-defined:

<!-- formula-not-decoded -->

Eq. (30) translates to x = G ( x ) , and the aforementioned x = G ( y ) , x = G ( y ) . Thus

<!-- formula-not-decoded -->

We now utilize the partitions of unity tool in topology. Consider an ϵ r 2 -net of X , whose size is finite since X is a compact subset of the Euclidean space. Denote it by X ⊆ ⋃ M i =1 B i where B i is a ball with radius ϵ r 2 centered at some ˜ y i ∈ X . It induces a continuous partition of unity { ϕ i : B i → [0 , 1] } i ∈ [ M ] such that ∑ M i =1 ϕ i ( y ) = 1 for all y ∈ X (Munkres, 2000, Theorem 36.1). Consider

<!-- formula-not-decoded -->

which is continuous since every ϕ i is. Furthermore, as G ( ˜ y i ) ∈ X for all i ∈ [ M ] , we know ˜ G is a continuous map from X to X . As X is a non-empty, convex, and compact set, the Brouwer's Fixed Point theorem (Munkres, 2000, Theorem 55.6) suggests the existence of y ∗ ∈ X such that ˜ G ( y ∗ ) = y ∗ . This y ∗ ∈ X then ensures

<!-- formula-not-decoded -->

̸

where (a) uses the fact that if ϕ i ( y ∗ ) = 0 , then y ∗ ∈ B i and hence ∥ y ∗ -˜ y i ∥ &lt; ϵ r , which gives ∥ G ( y ∗ ) -G ( ˜ y i ) ∥ ≤ η r L r from Eq. (31). Therefore, for any round r ∈ [ R ] , the system Eq. (30) indeed allows an ( η r L r ) -approximate fixed point x r ∈ X .

After proving the existence of approximate fixed points in Eq. (30), we utilize the vanilla O-FTRL result stated as Lemma E.2. Since every term in Eq. (30) is F r -1 -measurable, the approximate fixed point x r and its induced prediction ̂ ℓ r ( · ) := ˜ ℓ r ( x r , · ) are F r -1 -measurable. Therefore, the { ̂ ℓ r } r ∈ [ R ] serves as a valid prediction required by Lemma E.2. Applying Lemma E.2, for the sequence

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

Since x r is an ( η r L r ) -approximate fixed point of G , we know ∥ x r -x ∗ r ∥ = ∥ x r -G ( x r ) ∥ ≤ η r L r ϵ r . Further realizing that ∇ ̂ f r ( x ) = ∇ 2 ˜ f r ( x r , x ) by definition of ̂ f r ( · ) = ˜ f r ( x r , · ) , we get

<!-- formula-not-decoded -->

where the first inequality uses the L -Lipschitzness of f r , and the second and third inequalities use the L f -smoothness of f r and ˜ f r ( x r , · ) . Plugging back gives our conclusion that for any x ∗ ∈ X ,

<!-- formula-not-decoded -->

This finishes the proof.

## D.3.2 Approximate Continuity of Predictions in O-FTRL-FP

Lemma D.5 (Approximate Continuity of Predictions) . Recall the definition of ˜ g ℓ ( λ ) from Eq. (6):

<!-- formula-not-decoded -->

For any fixed ϵ &gt; 0 and δ ∈ (0 , 1) , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof. Take an ϵ 2 -net of Λ defined in Lemma E.3, and denote it by Λ ϵ (which we slightly abused the notation; note that ϵ 2 -nets are also ϵ -nets since ϵ 2 &lt; ϵ ). Then we have | Λ ϵ | ≤ ∏ d j =1 (2 d/ρ j ϵ ) .

Fix a λ ϵ ∈ Λ ϵ and consider the stochastic process ( X τ ) τ ≥ 1 adapted to ( F τ ) τ ≥ 0 : 6

̸

X τ := ✶ [ ∃ λ ∈ B ϵ ( λ ϵ ) s.t. ˜ i τ ( λ ) = ˜ i τ ( λ ϵ )] , ∀ ℓ ′ &lt; ℓ, τ ∈ E ℓ ′ , where ( F τ ) τ ≥ 0 is defined as F τ = σ ( ⋃ i ∈{ 0 }∪ [ K ] H τ +1 ,i ) , i.e. , the smallest σ -algebra containing all revealed history up to the end of round τ . Then X τ is F τ -measurable. Using the Multiplicative Azuma-Hoeffding inequality given as Lemma E.4 with Y t = E [ X t | F t -1 ] and ϵ = 1 2 ,

<!-- formula-not-decoded -->

For any ℓ ′ &lt; ℓ and τ ∈ E ℓ ′ , the distribution of u τ conditional on all previous history, namely ⋃ i ∈{ 0 }∪ [ K ] H τ,i , is F τ -1 -measurable ( i.e. , it follows a F τ -1 -measurable joint distribution U ∈ △ ([0 , 1] d ) ). λ ϵ ∈ Λ ϵ is fixed before the game and thus also F τ -1 -measurable. Lemma E.3 gives

̸

<!-- formula-not-decoded -->

where the Pr is taken w.r.t. the randomness of generating u τ according to the conditional joint distribution u τ | ⋃ i ∈{ 0 }∪ [ K ] H τ,i and the independent sampling of c τ ∼ C .

Hence, for any failure probability δ &gt; 0 that we determine later, with probability 1 -δ ,

̸

Taking Union Bound over λ ϵ ∈ Λ ϵ , with probability 1 -δ ∏ d j =1 (2 d/ρ j ϵ ) , the above good event holds for all λ ϵ ∈ Λ at the same time. Consider any λ 1 , λ 2 ∈ Λ such that ∥ λ 1 -λ 2 ∥ 2 ≤ ϵ 2 √ d , which immediately gives ∥ λ 1 -λ 2 ∥ 1 ≤ ϵ 2 . Take λ ϵ ∈ Λ ϵ such that λ 1 ∈ B ϵ/ 2 ( λ ϵ ) (recall that Λ ϵ is in fact a ϵ 2 -net), we therefore have ∥ λ 2 -λ ϵ ∥ 1 ≤ ϵ , which means λ 1 , λ 2 ∈ B ϵ ( λ ϵ ) . Thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

6 This ∃ λ ∈ B ϵ ( λ ϵ ) is pivotal because we cannot afford to take a Union Bound over all λ ∈ B ϵ ( λ ϵ ) . We call this step 'uniform smoothness', since it ensures the similarity holds uniformly in the neighborhood of λ ϵ .

<!-- formula-not-decoded -->

where we used the fact that c τ,i ∈ [0 , 1] d for any i ∈ [ K ] . This ensures that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -δ ∏ d j =1 (2 d/ρ j ϵ ) . Substituting ϵ ′ = ϵ 2 √ d and δ ′ = δ ∏ d j =1 (2 d/ρ j ϵ ) gives the conclusion.

## D.3.3 Stability Term Bounds in O-FTRL-FP

Lemma D.6 (Difference between { i t } t ∈E ℓ and { ˜ i ∗ t } t ∈E ℓ ) . For any epoch ℓ ∈ [ L ] ,

<!-- formula-not-decoded -->

where N ℓ is defined as in Lemma D.1:

<!-- formula-not-decoded -->

and we recall that

<!-- formula-not-decoded -->

Proof. In this proof, we first control E [ ∥ ∑ t ∈E ℓ ( ρ -c t, ˜ i ∗ t ) -˜ g ∗ ℓ ( λ ℓ ) ∥ 2 2 ] , i.e. , the squared ℓ 2 -error of |E ℓ | random vectors from their mean - which is of order |E ℓ | because they are i.i.d. We then relate it to E [ ∥ ∑ t ∈E ℓ ( ρ -c t,i t ) -˜ g ∗ ℓ ( λ ℓ ) ∥ 2 2 ] by utilizing the similarity between { ˜ i ∗ t } t ∈E ℓ and { i t } t ∈E ℓ that we derived in Theorem C.5.

Step 1: Control E [ ∥ ∑ t ∈E ℓ ( ρ -c t, ˜ i ∗ t ) -˜ g ∗ ℓ ( λ ℓ ) ∥ 2 2 ] . Recall the definition of { ˜ i ∗ t } t ∈ [ T ] from Eq. (9):

<!-- formula-not-decoded -->

and noticing that v t and c t are i.i.d. samples from V and C , we have PDF( ˜ i ∗ t ) = PDF( ˜ i ∗ ( λ ℓ )) for all t ∈ E ℓ . Therefore,

<!-- formula-not-decoded -->

where the last equation is precisely the definition of ˜ g ∗ ℓ ( λ ℓ ) . Since for a d -dimensional random vector X , E [ ∥ X -E [ X ] ∥ 2 2 ] = E [ ∑ d i =1 ( X i -E [ X i ]) 2 ] = ∑ d i =1 Var ( X i ) = Tr ( Cov ( X )) where Tr is the trace and Cov is the covariance matrix, we have

<!-- formula-not-decoded -->

using the fact that ˜ i ∗ t 's are independent from each other and that ρ and c ∗ ,i are all within [0 , 1] d .

Step 2: Relate ∑ t ∈E ℓ ( ρ -c t, ˜ i ∗ t ) to ∑ t ∈E ℓ ( ρ -c t,i t ) . Recall Eq. (16) from Theorem C.5:

̸

<!-- formula-not-decoded -->

where N ℓ := 1 + 4 K 2 ϵ c ∥ ρ -1 ∥ 1 +5log |E ℓ | +log γ -1 (1 + 4(1 + ∥ ρ -1 ∥ 1 ) K |E ℓ | 4 ) .

As ( ρ -c t,i ) ∈ [ -1 , 1] d , we have

<!-- formula-not-decoded -->

̸

̸

where the last term considers the failure probability of Eq. (32), in which case we use the trivial bound ( ∑ t ∈E ℓ ✶ [ i t = ˜ i ∗ t ]) 2 ≤ |E ℓ | 2 . Rearranging gives the desired conclusion. Lemma D.7 (Empirical Estimation) . For any ℓ ∈ [ L ] , ϵ &gt; 0 , and δ ∈ (0 , 1) , with probability 1 -2 δ ,

<!-- formula-not-decoded -->

where we recall that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. In contrast to Lemma D.6 where we directly applied concentration bounds at the realized dual iterate λ ℓ , here we cannot proceed in the same way. The reason is that the value samples v τ used to compute ˜ g v ℓ ( λ ℓ ) were drawn in the past, and λ ℓ itself is computed based on reports dependent on these values. As a result, conditioning on the event that ˜ g ∗ ℓ ( λ ℓ ) ≈ ˜ g v ℓ ( λ ℓ ) introduces a dependence on future information from the perspective of those past realizations, violating valid conditioning.

̸

Step 1: Cover Λ with an ϵ -net. From Lemma E.3, for any ϵ &gt; 0 , there exists an ϵ -net Λ ϵ ⊆ Λ of size O (( d/ϵ ) d ) , such that every λ ∈ Λ has some λ ϵ ∈ Λ ϵ with ∥ λ -λ ϵ ∥ 1 ≤ ϵ . We remark that our final guarantee does not have a d -exponent, because the dependency on | Λ ϵ | is logarithmic.

To overcome this, we establish uniform concentration over all λ ∈ Λ by discretizing the domain. Specifically, we construct an ϵ -net Λ ϵ ⊆ Λ and first show that for every λ ϵ ∈ Λ ϵ , the approximation ˜ g ∗ ℓ ( λ ϵ ) ≈ ˜ g v ℓ ( λ ϵ ) holds with high probability. We then extend this guarantee to all λ ∈ Λ by considering the stochastic process { ✶ [ ∃ λ ∈ B ϵ ( λ ϵ ) s.t. ˜ i v τ ( λ ) = ˜ i v τ ( λ ϵ )] } ℓ ′ &lt;ℓ,τ ∈E ℓ ′ where B ϵ ( λ ϵ ) is the ϵ -radius ball centered at λ ϵ . The proof goes in three steps.

Step 2: Yield concentration for any fixed λ ϵ ∈ Λ ϵ . Fix λ ϵ ∈ Λ ϵ . Let

<!-- formula-not-decoded -->

Since x τ only depends on v τ and c τ which are i.i.d. samples from V and C , these vectors are i.i.d. , zero-mean, and ensures ∥ x τ ∥ 2 ≤ √ d a.s. since ρ -c τ ∈ [ -1 , 1] d . Applying the vector Bernstein inequality (Kohler and Lucchi, 2017, Lemma 18) restated as Lemma E.5 gives:

<!-- formula-not-decoded -->

Taking a union bound over all λ ϵ ∈ Λ ϵ , we obtain that

<!-- formula-not-decoded -->

Therefore, for the given failure probability δ , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Step 3: Extend the similarity to all λ ∈ Λ . We now fix a λ ϵ ∈ Λ ϵ and try to ensure a uniform concentration guarantee for all λ ∈ B ϵ ( λ ϵ ) . Using boundedness ∥ ρ -c τ,i ∥ 2 ≤ √ d , we have:

̸

<!-- formula-not-decoded -->

From Lemma E.3, we know

<!-- formula-not-decoded -->

̸

where we remark that the ∃ λ ∈ B ϵ ( λ ϵ ) clause is important since it ensures 'uniform smoothness' in the neighborhood of λ ϵ . If we instead fix a λ and its corresponding λ ϵ and apply concentration to this specific λ , we need to do an prohibitively expensive Union Bound afterwards; see also Footnote 6. Hence the error between ˜ g ∗ ℓ ( λ ) and ˜ g ∗ ℓ ( λ ϵ ) is bounded by

<!-- formula-not-decoded -->

For the error between ˜ g v ℓ ( λ ) and ˜ g v ℓ ( λ ϵ ) , consider a stochastic process ( X τ ) τ ≥ 1 adapted to ( F τ ) τ ≥ 0 :

̸

X τ := ✶ [ ∃ λ ∈ B ϵ ( λ ϵ ) s.t. ˜ i v τ ( λ ) = ˜ i v τ ( λ ϵ )] , ∀ ℓ ′ &lt; ℓ, τ ∈ E ℓ ′ , where F τ = σ ( ⋃ i ∈{ 0 }∪ [ K ] H τ +1 ,i ) = σ ( v 1 , . . . , v τ , u 1 , . . . , u τ , c 1 , . . . , c τ , i 1 , . . . , i τ , p 1 , . . . , p τ ) is the smallest σ -algebra containing all generated history up to the end of round τ . 7 Then X τ is F τ -measurable, and we have

̸

<!-- formula-not-decoded -->

where the last step uses Eq. (34). Applying Azuma-Hoeffding to martingale difference sequence { X τ -E [ X τ | F τ -1 ] } ℓ ′ &lt;ℓ,τ ∈E ℓ ′ , we know for any c &gt; 0 ,

̸

<!-- formula-not-decoded -->

Applying a Union Bound over all λ ϵ ∈ Λ ϵ and recalling the expression of ∥ ˜ g v ℓ ( λ ) -˜ g v ℓ ( λ ϵ ) ∥ 2 2 ,

<!-- formula-not-decoded -->

7 In fact, the X τ 's here are i.i.d. variables since v τ ∼ V and c τ ∼ C are independent. However, when controlling ∥ ˜ g u ℓ ( λ ) -˜ g u ℓ ( λ ϵ ) ∥ 2 2 in Lemma D.8, since reports u τ can be history-dependent, to make Lemma E.3 still applicable we need to make sure the conditional distribution of reports u τ |H τ is F τ -1 -measurable.

̸

Therefore, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Final Bound. Via Union Bound, with probability 1 -2 δ , Eqs. (33), (35) and (36) are all true and

<!-- formula-not-decoded -->

This finishes the proof.

Lemma D.8 (Untruthful Reports) . For any ℓ ∈ [ L ] , ϵ &gt; 0 , and δ ∈ (0 , 1) , with probability 1 -4 δ ,

<!-- formula-not-decoded -->

where we recall that

<!-- formula-not-decoded -->

Proof. We now bound the impact of untruthful reporting, specifically the difference between past reported values u t and true values v t . Similar to the reason in Lemma D.7, we cannot directly apply concentration inequalities to λ ℓ which is unmeasurable when the reports are generated. Therefore, we still consider an ϵ -net Λ ϵ of Λ , ensure that ˜ g v ℓ ( λ ϵ ) ≈ ˜ g u ℓ ( λ ϵ ) for all λ ϵ ∈ Λ ϵ , and then extend it to all λ ∈ Λ via Chernoff-Hoeffding inequalities.

Step 1: Cover Λ with an ϵ -net. From Lemma E.3, for any ϵ &gt; 0 , there exists an ϵ -net Λ ϵ ⊆ Λ of size O (( d/ϵ ) d ) , such that every λ ∈ Λ has some λ ϵ ∈ Λ ϵ with ∥ λ -λ ϵ ∥ 1 ≤ ϵ . We remark that our final guarantee does not have a d -exponent, because the dependency on | Λ ϵ | is logarithmic.

Step 2: Yield concentration for any fixed λ ϵ ∈ Λ ϵ . Fix λ ϵ ∈ Λ ϵ . From Lemma C.3, which underpins the INTEREPOCH analysis in Theorem C.5, we know that for any previous epoch ℓ ′ &lt; ℓ , the event | u τ,i -v τ,i | ≥ 1 |E ℓ ′ | occurs in only ˜ O (1) rounds (with τ ∈ E ℓ ′ ) with probability at least 1 -1 |E ℓ ′ | . Following the approach in Lemma C.4, we now leverage the smoothness of costs in Assumption 3 in combination with Azuma-Hoeffding inequalities to conclude that ˜ g v ℓ ( λ ϵ ) ≈ ˜ g u ℓ ( λ ϵ ) .

̸

Formally, to compare ˜ g u ℓ ( λ ϵ ) and ˜ g v ℓ ( λ ϵ ) , we need only to control the number of previous rounds τ such that ˜ i u τ ( λ ϵ ) = ˜ i v τ ( λ ϵ ) . We decompose such events by whether large misreports happen:

̸

<!-- formula-not-decoded -->

̸

where the second term plugs | u τ,i -v τ,i | ≥ 1 |E ℓ ′ | , ∀ i ∈ [ K ] into definitions of ˜ i u τ ( λ ϵ ) and ˜ i v τ ( λ ϵ ) . For the first term, we use the following inequality which appeared as Eq. (13) in Lemma C.3:

<!-- formula-not-decoded -->

For any fixed failure probability δ ∈ (0 , 1) , for every ℓ ′ &lt; ℓ , picking c so that the RHS is δ ℓ gives

<!-- formula-not-decoded -->

For the second term, under Assumption 3 that PDF( λ T ϵ c τ,i ) is uniformly bounded by ϵ c , ∀ i ∈ [ K ] ,

̸

<!-- formula-not-decoded -->

Although we are now standing at epoch ℓ , the ϵ -nets are fixed before the game (as it only depends on Λ ). Hence, the indicator X τ := ✶ [( v τ,i -v τ,j ) -λ T ϵ ( c τ,i -c τ,j ) ∈ [0 , 2 |E ℓ ′ | ]] is indeed F τ -measurable back in the past when τ ∈ E ℓ ′ and ℓ ′ &lt; ℓ , where ( F τ ) τ ≥ 0 is the natural filtration F τ = σ ( X 1 , . . . , X τ ) . Thus, applying multiplicative Azuma-Hoeffding inequality in Lemma E.4 to the martingale difference sequence { X τ -E [ X τ | F τ -1 ] } ℓ ′ &lt;ℓ,τ ∈E ℓ ′ , we get

<!-- formula-not-decoded -->

Since X τ only involves v τ ∼ V and c τ ∼ C which are i.i.d. , we know E [ X τ | F τ -1 ] = E [ X τ ] ≤ 2 |E ℓ ′ | ϵ c ∥ λ ϵ ∥ 1 . Setting the RHS as δ | Λ ϵ | K 2 and taking a Union Bound over all λ ϵ ∈ Λ ϵ , we have

̸

<!-- formula-not-decoded -->

̸

Using another Union Bound over all i = j ∈ [ K ] and plugging it back into Eq. (37), we get

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -2 δ . Therefore, with probability at least 1 -2 δ , we have

̸

<!-- formula-not-decoded -->

where the first inequality uses the boundedness of ∥ ρ -c τ,i ∥ 2 2 ≤ d .

Step 3: Extend the similarity to all λ ∈ Λ . After yielding the similarity that ˜ g u ℓ ( λ ϵ ) ≈ ˜ g v ℓ ( λ ϵ ) for all λ ϵ ∈ Λ ϵ , we extend it to all λ ∈ Λ using the arguments already derived in Lemma D.7. Recall from Eq. (36) that we already proved that with probability 1 -δ ,

<!-- formula-not-decoded -->

Using exactly the same arguments (see Footnote 7 for the reason why Lemma E.3 is still applicable when reports { u τ } ℓ ′ &lt;ℓ,τ ∈E ℓ ′ can be history-dependent), with probability 1 -δ ,

<!-- formula-not-decoded -->

Final Bound. Putting the three inequalities together and taking Union Bound,

<!-- formula-not-decoded -->

This completes the proof.

## E Auxiliary Lemmas

We first include several classical online learning guarantees.

LemmaE.1 (FTRL Guarantee (Orabona, 2019, Corollary 7.7)) . For a convex region X , an 1 -stronglyconvex regularizer Ψ: X → R w.r.t. some norm ∥·∥ with min x ∈X Ψ( x ) = 0 , a sequence of convex and differentiable losses f 1 , f 2 , . . . , f R , a sequence of learning rates η 1 ≥ η 2 ≥ · · · ≥ η R ≥ 0 ,

<!-- formula-not-decoded -->

we have the following regret guarantee where ∥·∥ ∗ is the dual norm of ∥·∥ :

<!-- formula-not-decoded -->

Lemma E.2 (O-FTRL Guarantee (Orabona, 2019, Theorem 7.39)) . For a convex region X , an 1 -strongly-convex regularizer Ψ: X → R w.r.t. some norm ∥·∥ with min x ∈X Ψ( x ) = 0 , a sequence of convex and differentiable losses f 1 , f 2 , . . . , f R , a sequence of learning rates η 1 ≥ η 2 ≥ · · · ≥ η R ≥ 0 , a (stochastic) prediction sequence { ̂ f r : X → R } r ∈ [ R ] that is F r -1 -measurable (where ( F r ) t is the natural filtration that F r = σ ( f 1 , f 2 , . . . , f r ) ), and

<!-- formula-not-decoded -->

we have the following regret guarantee where ∥·∥ ∗ is the dual norm of ∥·∥ :

<!-- formula-not-decoded -->

We now present a covering of the dual variable space Λ .

Lemma E.3 (Covering of Dual Variables) . For any fixed constant ϵ &gt; 0 , there exists a subset Λ ϵ of Λ := ⊗ d j =1 [0 , ρ -1 j ] that has a size no more than ∏ d j =1 ( d/ρ j ϵ ) and ensures

<!-- formula-not-decoded -->

Furthermore, for any ϵ -net Λ ϵ such that Λ ⊆ ⋃ λ ϵ ∈ Λ ϵ B ϵ ( λ ϵ ) where B ϵ ( λ ϵ ) = { λ ∈ Λ | ∥ λ -λ ϵ ∥ 1 ≤ ϵ } is the neighborhood of λ ϵ ∈ Λ ϵ , under Assumption 3, we have for all λ ϵ ∈ Λ ϵ and any distribution U ∈ △ ([0 , 1] K ) that

̸

<!-- formula-not-decoded -->

̸

Proof. The first claim is standard from covering arguments and the fact that Λ = ⊗ d j =1 [0 , ρ -1 j ] is bounded. For the second part, we make use of Assumption 3: For any fixed i = j ∈ [ K ] ,

<!-- formula-not-decoded -->

where the first inequality uses |⟨ λ -λ ϵ , c i -c j ⟩| ≤ ∥ λ -λ ϵ ∥ 1 · ∥ c i -c j ∥ ∞ ≤ ϵ for all λ ∈ B ϵ ( λ ϵ ) , while the second uses Assumption 3 and the independence of c i and c j : If two independent realvalued random variables X ⊥ Y have their PDFs f X and f Y uniformly bounded by ϵ c , then

<!-- formula-not-decoded -->

Applying Union Bound to the K 2 pairs of ( i, j ) ∈ [ K ] × [ K ] gives the second conclusion.

Finally, we introduce some martingale or random variable concentration inequalities.

Lemma E.4 (Multiplicative Azuma-Hoeffding Inequality (Koufogiannakis and Young, 2014, Lemma 10)) . Let T be a stopping time such that E [ T ] &lt; ∞ . Let { X t } t ≥ 1 and { Y t } t ≥ 1 be two sequence of random variables such that

<!-- formula-not-decoded -->

then for any ϵ ∈ [0 , 1] and A ∈ R , we have

<!-- formula-not-decoded -->

LemmaE.5 (Vector Bernstein Inequality (Kohler and Lucchi, 2017, Lemma 18)) . Let x 1 , x 2 , . . . , x n be independent random d -dimensional vectors such that they are zero-mean E [ x i ] = 0 , uniformly bounded ∥ x i ∥ 2 ≤ C a.s. for some C &gt; 0 , and have bounded variance E [ ∥ x i ∥ 2 ] ≤ σ 2 for some σ &gt; 0 . Let z = 1 n ∑ n i =1 x i , then

<!-- formula-not-decoded -->