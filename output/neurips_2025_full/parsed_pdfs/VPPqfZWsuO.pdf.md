## Improved Confidence Regions and Optimal Algorithms for Online and Offline Linear MNL Bandits

Yuxuan Han New York University yh6061@stern.nyu.edu

Jose Blanchet Stanford University jose.blanchet@stanford.edu

## Abstract

In this work, we consider the data-driven assortment optimization problem under the linear multinomial logit (MNL) choice model. We first establish an improved confidence region for the maximum-likelihood-estimator (MLE) of the d -dimensional linear MNL likelihood function that removes the explicit dependency on a problem-dependent parameter κ -1 in previous result [42], which scales exponentially with the radius of the parameter set. Building on the confidence region result, we investigate the data-driven assortment optimization problem in both offline and online settings. In the offline setting, the previously best-known result scales as ˜ O (√ d κn S ⋆ ) , where n S ⋆ denote the number of times that optimal assortment S ⋆ is observed [26]. We propose a new pessimistic-based algorithm that, under a burn-in condition, removes the dependency on d, κ -1 in the leading order bound and works under a more relaxed coverage condition, without requiring the exact observation of S ⋆ . In the online setting, we propose the first algorithm to achieve ˜ O ( √ dT ) regret without a multiplicative dependency on κ -1 . In both settings, our results nearly achieve the corresponding lower bound when reduced to the canonical N -item MNL problem, demonstrating their optimality.

## 1 Introduction

In modern data-driven decision-making problems, a key challenge for sellers, often managing a large inventory, is determining a subset of products, also referred to as an assortment , to display to customers. For instance, when customers search for "laptops" on an e-commerce platform, the system must select an assortment from thousands of options to present. Due to space constraints or cognitive overload, only a limited number of items-at most K -can be displayed at a time. This motivates the study of assortment optimization with cardinality constraints , where the seller aims to identify the optimal assortment of K items to display in order to maximize revenue.

The revenue of an assortment S depends on both the revenue of each item i ∈ S and the final choice made by the customer after observing S . While the revenue of individual items is often known to the seller, the customer's choice behavior is unknown and must be estimated from data. A large body of work has focused on modeling customer choice behavior in the context of assortment optimization [40, 52, 41, 7, 22, 25, 6, 12]. Among these models, the multinomial logit (MNL) model and its linearly parametrized variant stands out as a widely used approach and serves as a foundation for understanding more complex models due to its clear mathematical structure, computational simplicity, and ease of calibration [11].

In the linear MNL choice model with N items, each item i ∈ [ N ] is associated with an d -dimensional feature x i , and its utility, presenting its attraction to customers, is given by v i = x ⊤ i θ ⋆ for some

Zhengyuan Zhou New York University zzhou@stern.nyu.edu

Table 1: Comparison of offline and online assortment optimization results, where p i ( S ) denote the choice probability of item i under assortment S ; only leading-order terms are presented for notational simplicity.

<!-- image -->

| Offline Learning   | Offline Learning                                                                                                                             | Offline Learning                                                  | Offline Learning                                                                           |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
|                    | Leading-Order Upper Bound                                                                                                                    | Lower Bound                                                       | Remark                                                                                     |
| [26]               | ˜ O ( √ d/ ( κ ′ n S ⋆ )) †                                                                                                                  | -                                                                 | n S ⋆ = ∑ n m =1 1 { S m = S ⋆ }                                                           |
| [29]               | ˜ O ( K/ √ n ⋆ )                                                                                                                             | Ω( K/ √ n ⋆ )                                                     | N -item setting, n ⋆ = min i ∈ S ⋆ ∑ n m =1 1 { i ∈ S ⋆ }                                  |
| Our work           | ˜ O (√∑ i ∈ S ⋆ p i ( S ⋆ ) p 0 ( S ⋆ ) ∥ x i ∥ 2 H - 1 D ( θ ⋆ ) ) ˜ O (√ d · ∑ i ∈ S ⋆ p i ( S ⋆ ) p 0 ( S ⋆ ) ∥ x i ∥ 2 H - 1 D ( θ ⋆ ) ) | Ω( √∑ i ∈ S ⋆ p i ( S ⋆ ) p 0 ( S ⋆ ) ∥ x i ∥ 2 H - 1 D ( θ ⋆ ) ) | Burn-in condition max j ∈∪ n m =1 S n ∥ x j ∥ H - 1 D ( θ ⋆ ) ≲ 1 √ d No burn-in condition |
| Online Learning    | Online Learning                                                                                                                              | Online Learning                                                   | Online Learning                                                                            |
|                    | Leading-Order Upper Bound                                                                                                                    | Lower Bound                                                       | Remark                                                                                     |
| [5]                | ˜ O ( √ NT )                                                                                                                                 | Ω( √ NT/K )                                                       | N -item setting                                                                            |
| [17]               | -                                                                                                                                            | Ω( √ NT )                                                         | N -item setting                                                                            |
| [18]               | ˜ O ( d √ T )                                                                                                                                | Ω( d √ T/K )                                                      | Adversarial context with initial exploration                                               |
| [42]               | ˜ O ( κ - 1 √ dT log N )                                                                                                                     | -                                                                 | Adversarial context with initial exploration                                               |
| [44]               | ˜ O ( Kd √ T )                                                                                                                               | -                                                                 | Adversarial context with uniform reward                                                    |
| [35]               | ˜ O ( d √ T )                                                                                                                                | Ω( d √ T )                                                        | Adversarial context                                                                        |
| Our work           | ˜ O ( √ dT log N )                                                                                                                           | Ω( √ dT ) ‡                                                       | Fixed design Adversarial context with initial exploration ∗                                |

† The κ ′ notation in [26] defined in a different way as those in other works, but the still suffers from the exponential dependency on ∥ θ ⋆ ∥ in the worst case. √

‡ When consider the canonical N -item setting with x i = e i , d = N, the Ω( NT ) result in [17] implies this result.

∗ We leave the related algorithm design and proof to Appendix E.4 due to space limitation.

underlying θ ⋆ ∈ R d . After given an S , the choice of the customer then follows the standard multinomial distribution as described as in (1). While data-driven assortment optimization with the linear MNL model has been extensively studied [14, 46, 50, 4, 5, 19, 48], most of the existing literature focuses on the online learning setting. The current best-known regret bounds are given as ˜ O ( d √ T ∧ κ -1 √ dT log N ) . When specialized to the canonical N -item setting ( d = N , x i = e i being the canonical basis), the regret either exhibits an additional √ d dependence or incurs a multiplicative dependence on κ -1 , a problem-dependent quantity defined in (2), which may scales exponentially with ∥ θ ⋆ ∥ . Besides the online setting, the only known works in the offline learning setting [26, 29] either focus on the canonical N -item setting or rely on restrictive assumptions about the coverage of the data.

Motivated by these gaps, we propose new algorithms in this work that achieve improved online regret and offline sample complexity guarantees. Our methods build on a sharper analysis of the linear MNL likelihood function, which serves as the foundation for most existing approaches under the linear MNL model. We compare our results with previous works in Table 1 and summarize our contributions as follows:

## 1.1 Our Contributions

Sharper Confidence Region result for Maximum Likelihood Estimator. Our first result is an improved confidence region for the linear MNL maximum likelihood estimator (MLE), which incorporates variance information and avoids explicit dependency on κ . More precisely, given the conditional independence of the observed choice dataset D := { i k , S k } n k =1 , we show that the corresponding maximizer ̂ θ of the linear MNL likelihood function, under a burn-in condition, satisfies that with high probability, | x ⊤ ( ̂ θ -θ ⋆ ) | = ˜ O ( ∥ x ∥ H -1 D ( θ ⋆ ) ) , ∀∥ x ∥ ≤ 1 , with H D the Hessian matrix of the log-likelihood function given D . Our result provides an non-asymptotic variant of the large-sample asymptotic x ⊤ ( ̂ θ -θ ⋆ ) = ⇒ N (0 , ∥ x ∥ H -1 D ( θ ⋆ ) ) which holds for general M -estimators under certain conditions [37], and improves the best previous non-asymptotic result under the same assumptions, stated as | x ⊤ ( ̂ θ -θ ⋆ ) | = ˜ O ( κ -1 ∥ x ∥ V -1 ) [42], since it always holds that H D ( θ ⋆ ) ⪰ κV. Our improvement stems from a novel variance-aware analysis combined with a careful exploration of the self-concordant-like properties of the MNL likelihood function, which is inspired by recent developments for MNL likelihood with adaptively collected data [44, 3, 35].

Offline Assortment Optimization with Item-wise Coverage. Based on the sharp confidence region result, we then consider the offline assortment optimization problem, where the seller can access to a dataset D = { i k , S k } n k =1 and aim to find the assortment that maximize the revenue. In this setting, we provide an pessimistic-based algorithm and show that under a basic coverage number of each items, it can achieve the sub-optimality gap that scales with ˜ O ( ∑ i ∈ S ⋆ p i ( S ⋆ ) p 0 ( S ⋆ ) ∥ x i ∥ H -1 ( θ ⋆ ) )

D

in the leading-order term. Notably, we can show that p i ( S ⋆ ) p 0 ( S ⋆ ) ∥ x i ∥ H -1 D ( θ ⋆ ) ≳ n -1 / 2 i for n i := ∑ n k =1 1 { i ∈ S k } , thus it suffices for each i ∈ S ⋆ to be covered by S k sufficiently many times in order to ensure that ∥ x i ∥ H -1 ( θ ⋆ ) becomes small. In contrast, the best-known result for linear MNL

D

model prior to ours, presented in [26], scales as ˜ O (√ d κn S ⋆ ) , where n S ⋆ := ∑ n k =1 1 { S ⋆ = S k } . -1

This result has an additional d -dependency and a multiplicative κ -dependency compared to ours. More importantly, their approach requires the optimal assortment S ⋆ to be exactly observed sufficiently many times, which imposes a restrictive coverage requirement on D . Finally, we show that, when reduced to the canonical N -item setting with uniform item-wise rewards, our result matches the Ω ( max i √ K/n i ) lower bound recently developed in [29]. This demonstrates that the proposed item-wise coverage measure, p i ( S ⋆ ) p 0 ( S ⋆ ) ∥ x i ∥ H -1 D ( θ ⋆ ) , is an appropriate metric for sample complexity in the offline setting.

Improved Regret for Online Assortment Optimization. In the online assortment optimization setting, where the seller starts without prior knowledge but can interact with arriving customers over T rounds, we design an algorithm based on the SupCB framework [8] that achieves a regret of ˜ O ( √ dT log N + κ -1 d ) . This result improves upon the previous regret bound of ˜ O ( κ -1 √ dT log N ) in [42] by reducing the dependency on κ -1 . Our result also improves the ˜ O ( d √ T + κ -1 d ) result in [44, 35] on the dependency of d when N = o (2 d ) . Especially, our result is nearly optimal in the sense that it nearly matches the Ω( √ dT ) lower bound in [17] when reduced to the canonical N -item setting .

## 1.2 Related Works

Data-Driven Assortment Optimization. The online assortment optimization problem under the MNL choice model has been extensively studied in the literature [14, 46, 50, 4, 5, 19, 48]. Among these works, [4, 5] and [17] were the first to close the ˜ Θ( √ NT ) minimax optimal regret for the canonical N -item setting. In the linear MNL setting, [18, 44] proposed algorithms achieving a regret of d √ T + Poly ( κ -1 , d ) , but these algorithms are computationally intractable. While [42] proposed computationally tractable algorithms for the same setting with a regret of ˜ O ( κ -1 d √ T ∧ κ -1 √ dT log N ) , their results depend on κ -1 in a multiplicative manner. The only known computationally tractable algorithm achieving ˜ O ( d √ T ) regret with additive dependency on κ -1 is that of [35] 1 . Our result contributes to this direction by improving the best-known regret bound with additive κ -1 dependency from ˜ O ( d √ T ) to ˜ O ( d √ T ∧ √ dT log N ) . For the offline assortment optimization problem, the only known works are [26] and [29]. [26] were the first to design a pessimistic-based algorithm for the linear MNL setting. Their algorithm is based on the assortment-wise pessimistic principle, and its performance bound scales to n -1 / 2 S ⋆ , with n S ⋆ the number of times the optimal assortment S ⋆ is exactly observed in the dataset. On the other hand, [29] studied the canonical N -item setting using an item-wise pessimistic principle. They showed that the minimax rate in this setting is K √ min i ∈ S ⋆ n i , where n i is the covering time of item i by the observed assortments, thus relaxing the requirement in [26]. The algorithm design in our work can be seen as a generalization of the item-wise pessimistic principle in [29] to the linear setting, with a new concept of item-wise covering introduced. Finally, beyond the MNL setting, learning problems involving additional constraints [20, 10, 15] or other choice models [43, 16, 39, 56, 55] have also been explored.

Offline Learning via Pessimistic Principle. Our design of offline algorithms follows the same spirit as the pessimistic (conservative) methods [54, 33] in offline bandit and reinforcement learning. The sample efficiency of pessimistic algorithms under partial coverage in the offline setting has been

1 [3] also proposed a computationally tractable algorithm with a similar regret; but their proof contains a technical error, as discussed in Appendix L of [35].

demonstrated in a series of studies [30, 45, 57]. Unfortunately, the setting considered in these works is not directly applicable to the assortment problem, due to differences in the feedback structure.

Online Learning with Bandit Feedback. The algorithm design in most online MNL works draws from the optimistic principle, a concept extensively explored in online bandit learning [8, 23, 1, 51, 34]. In particular, the logistic bandit problem can be viewed as a special case of the online assortment problem with K = 1 , which faces a similar challenge of eliminating the multiplicative dependency on a problem dependent parameter similar to κ -1 , motivating a number of studies [2, 27, 28, 31]. In particular, the confidence region result in our work can be regarded as a generalization of [31] to the MNL setting and our improvement over [42] parallels that of [31] for [38].

## 2 Preliminary

Revenue Maximization under the Linear MNL Model. We study the assortment optimization problem, which models the interaction between a seller and a customer . Let { 1 , . . . , N } denote the set of N available products/items. An assortment S ⊆ { 1 , . . . , N } represents the subset of products that the seller offers to the customer. When presented with assortment S , the customer chooses a product from the choice set S + = { 0 } ∪ S , where { 0 } represents the no-purchase option. In the N -item MNL model, each item has an attraction value v i ≥ 0 . When a customer encounters assortment S , the probability that he/she will choose product i ∈ S + is given by

<!-- formula-not-decoded -->

Each item i generates a revenue r i when purchased, while the no-purchase option generates no revenue: r 0 = 0 . The seller's goal is to maximize the expected revenue from the selected assortment, defined as

<!-- formula-not-decoded -->

In the d -dimensional linear MNL model with fixed design , each item i ∈ [ N ] is further associated with a vector x i ∈ R d , and there exists a underlying parameter θ ⋆ ∈ R d so that v i = exp( x ⊤ i θ ⋆ ) . With X := ( x 1 , . . . , x N ) ∈ R N × d , we also abuse the notation p i ( S | θ ⋆ ) := p i ( S | exp( X ⊤ θ ⋆ )) and R ( S | θ ⋆ ) := R ( S | exp( X ⊤ θ ⋆ )) to denote their dependence on θ ⋆ when there is no ambiguity.

Following standard research conventions, we consider the assortments S where | S | ≤ K . In the following context, we denote S K the set consist of all K -sized assortments and S ⋆ = argmax S ∈S K R ( S | θ ⋆ ) the optimal assortment.

Throughout the paper, we assume that the attraction values v i and the underlying parameter norm ∥ θ ⋆ ∥ are bounded by constants V and W , respectively. We also introduce the problem parameter

<!-- formula-not-decoded -->

it can be seen that κ -1 can scale with exp( W ) even when V is small. It is worth noting that a series of previous works on MNL bandits [18, 44, 3, 35] focus on improving the dependency on κ -1 in the sample complexity.

Offline Assortment Optimization. In the offline assortment optimization setting, the seller does not know the underlying parameter θ ⋆ but can access to a pre-collected dataset { i t , S t } n t =1 consisting of the choice-assortment pairs, where for each given S j , the corresponding i j is sampled independently from the linear MNL choice model with parameter θ ⋆ . The seller's goal is to approximate the optimal assortment based on those observed data, the learning objective is the sub-optimality gap, defined as

<!-- formula-not-decoded -->

which measures the gap between the revenue of assortment S and the best-possible revenue.

Online Assortment Optimization. In the online learning setting, the seller can interact with the coming customers for T rounds. At each time step t , the seller provides an assortment S t ∈ S K to the customer and then receives a feedback i t drawn according to the distribution specified in (1). The goal of the seller is to design a policy π = ( π 1 , . . . , π T ) , where each π t adaptively selects the

assortment S t based on the historical observations, to minimize the cumulative regret over T, defined as

<!-- formula-not-decoded -->

which measures the total sub-optimality of the policy π over T rounds.

Notations Through out the paper, for any real numbers a, b we use the notion a ∨ b to denote max { a, b } , and we use the notation a ≲ b if there exists some absolute constant c &gt; 0 so that a &lt; cb . For matrices and vectors, given any vector x ∈ R d and A ∈ R d × d , we denote ∥·∥ the ℓ 2 -norm, and ∥ x ∥ A := √ x ⊤ Ax, we also denote A † the pseudo inverse of A .

## 3 Improved Confidence Region for the Linear MNL MLE

In this section, we present our main result on the improved confidence region for the linear MNL MLE. While we focus on the fixed design setting in the main text, where each item's feature x i remains unchanged throughout t , we allow the observed feature x ti to vary across rounds in the dataset to ensure generality in this section's result, and we denote X t := { x ti } i ∈ [ N ] throughout this section. Given any dataset D := { i t , X t , S t } n t =1 , the linear MNL log-likelihood function is defined as

<!-- formula-not-decoded -->

where y tj = 1 { i t = j } . In the following context, for any λ ≥ 0 we denote ̂ θ λ D the λ -regularized MLE, i.e.,

<!-- formula-not-decoded -->

Our first result establishes a confidence interval for x ⊤ ̂ θ λ D with any ∥ x ∥ ≤ 1 given the following conditional independence assumption:

Assumption 1. Condition on { ( X t , S t ) } n t =1 , the observed choices { i t } n t =1 are mutually independent.

Theorem 1. Given D := { i t , X t , S t } n t =1 and ̂ θ λ D the λ -regularized MLE under D . Suppose Assumption 1, then for any x ∈ R d with ∥ x ∥ ≤ 1 , if we denote H D ( θ ) the Hessian matrix of ℓ D at θ , H λ D ( θ ) := H D ( θ ) + λI , and N eff as the total number of distinct vectors appeared in { x kj } j ∈ S k ,k ≤ t , we have condition on

<!-- formula-not-decoded -->

it holds that with probability at least 1 -δ i)

<!-- formula-not-decoded -->

ii)

<!-- formula-not-decoded -->

Technically, our proof of Theorem 1, detailed in Appendix B, builds on a extended framework for proving Theorem 1 of [31]. But several new ideas that rely on the geometry and self-concordant structure of the K -MNL loss are introduced. These structural properties have recently attracted much attention in the theoretical study of MNL bandits and are essential to our argument. The main technical contribution appears in Lemma 6, where we use the curvature of the MNL loss to design a confidence region that does not depend explicitly on K or κ . This result improves upon [42], which is the only previous work giving d -free confidence bounds under the independence condition, and parallels the refinement technique in [36], which is developed for obtaining sharp variance-dependent bounds.

Comparison to Previous Dimension-Free Result. Prior to our result, the only confidence interval for the linear MNL MLE achieving a √ d -free rate is stated in [42]. Their confidence region result scales as ˜ O ( κ -1 ∥ x ∥ V -1 t ) under the burn-in condition

<!-- formula-not-decoded -->

By the relation H t ( θ ⋆ ) ⪰ κV t and

<!-- formula-not-decoded -->

our result provides an improvement over [42] in both the confidence interval and the burn-in condition.

Comparison to other Confidence Bound Results. Besides [42], a line recent works have focused on deriving confidence regions for the MLE that are independent of κ [44, 3, 35]. It is worth mentioning that results in [44, 3, 35] no longer require the burn-in condition or the conditional independence assumption, with a price of introducing an additional factor of √ d in the resulting confidence interval. The requirement of a burn-in condition to achieve sharp dependency on d first appears in the logistic linear bandit literature, where such a condition arises as [31, 38] refine the results of [27, 2, 28] by improving the dependency on a √ d factor. This corresponds to a special case of our setting with K = 1 . In Appendix F.1, we present several comparison experiments to those burn-in time free bounds to discuss the sensitivity to burn-in condition of Theorem 1.

## 4 Offline Assortment Optimization with Linear MNL Choices

In this section, based on the confidence interval result in Theorem 1, we present the algorithm for offline assortment optimization problem and the corresponding theoretical guarantee.

Throughout this section, we assume W.L.O.G. that H D ( θ ⋆ ) is invertible to maintain notational clarity; otherwise, we consider the λ -regularized MLE and its corresponding confidence region for λ very close to 0 , which does not affect the theoretical results.

## 4.1 The LCB-LinearMNL Algorithm

For the offline assortment optimization problem, we propose the LCB-LinearMNL algorithm, as detailed in Algorithm 1. The design of Algorithm 1 is inspired by the lower-confidence-bound technique, also known as the pessimistic principle, which has been shown to be theoretically effective in the offline policy learning literature [30, 45, 57]. The LCB-LinearMNL algorithm consists of two steps, the pessimistic estimation step and the revenue maximization step .

## Algorithm 1 LCB-LinearMNL

- 1: Input: Dataset D = { ( i j , S j ) } n j =1 , feature set X

- 2: Compute the MLE ̂ θ = argmax θ ℓ n ( θ )

- 3: Compute the pessimistic value v LCB i as in (3) for each i ∈ [ N ] .

- 4: Select the pessimistic assortment: S LCB = argmax S ∈S K R ( S | v LCB ) .

- 5: Return:

- S

LCB

Pessimistic Estimation In the pessimistic estimation step, with the MLE ̂ θ D , the algorithm first compute

<!-- formula-not-decoded -->

then takes the pessimistic value estimation for each item i ∈ [ N ] via

<!-- formula-not-decoded -->

The confidence region result provided in Theorem 1 ensures the pessimistic property holds for v LCB i i.e. v LCB i ≤ v i := e x ⊤ i θ ⋆ for all i ∈ [ N ] with high probability.

Revenue Maximization After computing the pessimistic values for each item, the algorithm proceeds to the revenue maximization step, where it selects the assortment that maximizes revenue under v LCB i . This step can be efficiently solved using several well-studied polynomial-time algorithms [46, 24, 9].

## 4.2 Sub-optimality Gap Guarantee

Now we present the sub-optimality gap guarantee of Algorithm 1.

Theorem 2. Suppose the burn-in condition 64 max j ∈ S k ,k ∈ [ n ] ∥ x j ∥ H -1 D ( θ ⋆ ) ≤ ( ( d log( N eff /δ )) ) -1 / 2 holds, then with probability at least 1 -δ, we have

<!-- formula-not-decoded -->

up to logarithmic factors.

Theorem 2 shows that the leading-order complexity of the sub-optimality gap depends on the summation of p j ( S ⋆ | θ ⋆ ) p 0 ( S ⋆ | θ ⋆ ) ∥ x j ∥ H -1 D ( θ ⋆ ) over j ∈ S ⋆ , which measures how well the items in the optimal assortment are covered by the observed assortment data. Here we provide a upper bound of the this quantity with respect to item-wise coverage numbers to make the Theorem 2 more transparent:

Proposition 1. Denote n i := ∑ k ∈ [ n ] 1 { i ∈ S k } and n ⋆ := min i ∈ S ⋆ n i , then it holds that

<!-- formula-not-decoded -->

Based on Proposition 1, we can further obtain the following sub-optimality guarantee of Algorithm 1. Corollary 1. Denote n i := ∑ k ∈ [ n ] 1 { i ∈ S k } and n ⋆ := min i ∈ S ⋆ n i , then it holds that under the same condition on Theorem 2, we have with probability at least 1 -δ ,

<!-- formula-not-decoded -->

up to logarithmic factors

Results under Fixed Data Collecting Policy. In the setting that { S t } n t =1 is sampled i.i.d. from some fixed exploration policy π off , the burn-in condition in Theorem 2 always holds as n → + ∞ . Then Theorem 2 implies the following large-sample asymptotic

<!-- formula-not-decoded -->

with H off := E S ∼ π off [ ∑ i ∈ S p 0 ( S | θ ⋆ ) p i ( S | θ ⋆ ) x i x ⊤ i ] as n → + ∞ .

Comparison to Previous Results. Prior to our work, the sample complexity of offline assortment optimization under the linear MNL model was largely unexplored. The only known result, from [26], considers a fixed policy S t ∼ π off and yields a complexity of ˜ O (√ d/K κnπ off ( S ⋆ ) ) , which depends on how often the optimal assortment S ⋆ is sampled. In contrast, our result avoids multiplicative dependence on d and κ -1 in the leading term and does not require full coverage of S ⋆ . Instead, it suffices for S ⋆ to be explored through assortments with non-orthogonal feature overlap.

Result in the Canonical N -item setting and the Optimality. In the canonical N -item MNL setting , a special case of the linear MNL setting with d = N and x j = e j , a very recent work [29] shows a similar result to Corollary 1 where the coverage condition for each item is sufficient for learning of the optimal assortment. Their algorithm design shares the same spirit as ours in employing an itemwise pessimistic principle. However, their algorithm and analysis rely heavily on the rank-breaking technique [48, 49, 32] , which is challenging to extend to the general linear MNL setting. More specifically, they propose a ˜ Θ( K/ √ n ⋆ ) complexity result and a ˜ Θ( √ K/n ⋆ ) complexity result in the

uniform reward setting ( r i ≡ 1 ). In particular, Proposition 1 implies the a ˜ O ( K 3 / 2 / √ n ⋆ ) upper bound in general setting and an improved O ( √ K/n ⋆ ) bound in the uniform reward setting, since S ⋆ consists of the topK items by value. The uniform reward setting result, together with lower bound established

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Eliminating the Burn-in Condition. The burn-in condition in Theorem 2 stems from the confidence region requirement in Theorem 1. The item-wise pessimistic estimation in Algorithm 1 is flexible and can incorporate alternative confidence bounds by replacing (3). In particular, using confidence regions for adaptively collected data (e.g., [42, 44]) yields a burn-in-free guarantee at the cost of an additional √ d factor. We state the main result below and defer the proof to Appendix D.2.

Proposition 2. There exists a plug-in confidence region result so that when replacing the pessimistic estimation step in (3) by this result, the output Algorithm 1 satisfies

<!-- formula-not-decoded -->

with probability at least 1 -δ up to logarithmic factors.

## 5 Online Linear MNL Bandits

## 5.1 The SupLinearMNL algorithm

In this section, we propose a linear MNL bandit algorithms by leveraging the confidence region result established in Theorem 1. A detailed description of our algorithm is provided in Algorithm 2. To effectively balance the exploration-exploitation trade-off while keeping the independence of collected samples, we apply the SupCB framework of [8], incorporating a sophisticated action set elimination procedure and sample splitting procedure, as in [21, 38, 31, 13, 42]. However, several modifications have been made to our algorithm to ensure the burn-in condition of Theorem 1 and to utilize the first-order geometry of the revenue functions, as described below.

The SupCB Framework. Similar to the standard framework in [8], Algorithm 2 divides the collected samples into S bins, denoted as Ψ 1 , . . . , Ψ M +1 . To maintain independence while accounting for the first-order geometry, we add an additional bin, Ψ S +1 , as in [31], which contains a rough estimator for approximating the first-order coefficients of R . After an initial pure-exploration period of length τ -ensuring that each bin collects sufficient samples, as explained in the next paragraph-the algorithm enters an adaptive elimination phase conducted through a multi-layer procedure that loops over the first S bins.

Initial Exploration Phase. The initial exploration phase is designed to ensure the burn-in condition in Theorem 1. More precisely, the initial exploration phase ensures that max j ∈ [ N ] ∥ x j ∥ H † τ,ℓ ( θ ⋆ ) is well controlled for every ℓ ∈ [ S +1] with high probability. Note that since H τ,ℓ ( θ ⋆ ) ⪰ κV τ,ℓ , with V τ,ℓ := ∑ s ≤ τ,s ∈ Ψ ℓ ∑ k ∈ S s x k x ⊤ k , it suffices to ensure that max j ∈ [ N ] κ -1 / 2 ∥ x j ∥ V † τ,ℓ is well bounded for every ℓ . To achieve this goal with finite sample complexity, the algorithm repeatedly taking the exploration assortments containing uncertain items until

<!-- formula-not-decoded -->

holds for all ℓ. Careful readers may notice that to compute (4), we need prior knowledge of κ as an input to Algorithm 2, which is a problem-dependent quantity. When the exact value of κ is unknown, we can instead use the parameter radius W (or its upper bound) to provide a conservative estimate. In particular, exp( -KW ) can be used as a worst-case bound for κ . It should be noted that there may exist problem instances where κ is strictly smaller than its worst-case bound. Designing algorithms that achieve the same regret guarantee while fully adapting to an unknown κ , as in [44, 35], is left for future work.

Adaptive Elimination Phase. After entering the adaptive elimination phase, the algorithm stops allocating samples to Ψ S +1 . The MLE

<!-- formula-not-decoded -->

```
Algorithm 2 SupLinearMNL 1: Input: Time horizon T, problem-dependent factor κ . 2: initialize M = log 2 T, λ = 1 , τ = 1 , Ψ 1 = · · · = Ψ M +1 = ∅ . 3: while (4) is not satisfied for some ℓ ∈ [ M ] and j ∈ [ N ] do 4: Select arbitrary assortment that contains item j , add τ into Ψ ℓ 5: τ ← τ +1 6: end while 7: Ψ 0 ←∅ , compute ̂ θ 0 as in (5) 8: for t = τ +1 , . . . , T do 9: set A 1 = S K , S t = ∅ , ℓ = 1 10: while S t = ∅ do 11: Compute W ℓ t,S , R UCB t,ℓ ( S ) , ∀ S ∈ A ℓ as in (7), (8). 12: if W ℓ t,S > 2 -ℓ for some S ∈ A ℓ then 13: select such S ∈ A ℓ . 14: Ψ ℓ ← Ψ ℓ ∪ { t } 15: else if W ℓ t,S ≤ 1 /T for all S ∈ A ℓ then 16: take the action S t = argmax S ∈A ℓ R UCB t,ℓ ( S ) 17: Ψ 0 ← Ψ 0 ∪ { t } 18: else 19: ̂ R ← max S ∈A ℓ R UCB t,ℓ ( S ) 20: A ℓ +1 ← { S ∈ A ℓ , R UCB t,ℓ ( S ) ≥ ̂ R -2 -ℓ +7 } 21: ℓ ← ℓ +1 22: end if 23: end while 24: end for
```

is computed based on the data in Ψ S +1 collected during the first τ rounds and is fixed in all subsequent time steps.

At each t &gt; τ , during the ℓ -th loop, the algorithm calculates the confidence level term

<!-- formula-not-decoded -->

for i ∈ ∪ S ∈A ℓ S , with D t,ℓ the data collected in the ℓ -th bin up to time t and we simply denote H λ D t,ℓ by H λ t,ℓ . Then, based on the value of w t,i for each i , the assortment-wise confidence level for S , written as

<!-- formula-not-decoded -->

is calculated for each S ∈ A ℓ . Then the algorithm takes one of the following steps based on { W ℓ t,S } , :

Step (a). Exploration of Uncertain Items: If there exists some uncertain assortment (i.e., W ℓ t,S &gt; 2 -ℓ ), the algorithm selects this assortment to explore it further.

Step (b). Output UCB Assortment: If all assortments are sufficiently certain (i.e., W ℓ t,S &lt; 1 /T for all S ∈ A ℓ ), the algorithm outputs the assortment with highest UCB value, computed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step (c). Assortment Elimination: Otherwise, the algorithm performs assortment elimination over A ℓ . Specifically, it first computes the maximum UCB value ̂ R . Next, it eliminates all assortments S such that the UCB value gap, ̂ R -R UCB t,ℓ ( S ) , exceeds 2 -ℓ . Finally, the algorithm loops to the ( ℓ +1) -th bin with the eliminated item set A ℓ +1 .

## 5.2 Regret Guarantee of Algorithm 2

Now we show the regret guarantee of Algorithm 2:

Theorem 3. With probability at least 1 -1 /T, we have Algorithm 2 satisfies

<!-- formula-not-decoded -->

Theorem 3 establishes a ˜ O ( √ dT + κ -1 d 2 K 2 ) regret bound. Compared to the ˜ O ( κ -1 √ dT ) in [42] and ˜ O ( d √ T + κ -1 d 2 ) in [35], our result is the first to achieve ˜ O ( √ dT log N ) regret in the linear MNL setting with a second-order dependence on κ -1 . However, due to the need to enumerate over A ℓ in the elimination (Line 20) and confidence computation (Line 11) steps-each takes Ω( N K ) times of computation in worst-case, similar to [18, 42]-the algorithm is computationally inefficient 2 . Thus, Theorem 3 serves primarily as a theoretical benchmark, and we leave designing efficient algorithms with similar guarantees as a future direction.

Problem-Dependent Regret with Uniform Revenue. While above results consider the non-uniform revenue setting for general r ∈ [0 , 1] N . We can show that in the uniform revenue setting where r i ≡ 1 as in [44, 35], an improved problem-dependent rate is possible: Denote

<!-- formula-not-decoded -->

we have the following guarantee of Algorithm 2

Theorem 4. In the uniform revenue setting r i = 1 , ∀ i ∈ [ N ] , Algorithm 2 satisfies

<!-- formula-not-decoded -->

Noticing that in the good scenario where all v i = Θ(1) as in [35], we have the above bound simply turns to a ˜ O ( √ dT/K ) result, improving a √ K factor than previous best known results. In Appendix, we further show a Ω( κ ⋆ √ dT ) problem-dependent lower bound for a large range of κ ⋆ , implying the optimality of Theorem 4.

Acknowledgments. This work is generously supported by NSF CCF-2419564, NSF CCF-2312204, NSF CCF-2312205, ONR-13983263, ONR-13983111, and 2026 New York University Center for Global Economy and Business grant.

## References

- [1] Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems , 24, 2011.
- [2] Marc Abeille, Louis Faury, and Clément Calauzènes. Instance-wise minimax-optimal algorithms for logistic bandits. In International Conference on Artificial Intelligence and Statistics , pages 3691-3699. PMLR, 2021.
- [3] Priyank Agrawal, Theja Tulabandhula, and Vashist Avadhanula. A tractable online learning algorithm for the multinomial logit contextual bandit. European Journal of Operational Research , 310(2):737-750, 2023.
- [4] Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. Thompson sampling for the mnl-bandit. In Conference on learning theory , pages 76-78. PMLR, 2017.
- [5] Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. Mnl-bandit: A dynamic learning approach to assortment selection. Operations Research , 67(5):1453-1485, 2019.

2 A natural question is whether heuristic algorithms as in [18] can be developed for our setting. The main challenge lies in the assortment space A ℓ , which is exponentially large and unstructured. Unlike [18], where feasibility can be verified simply by checking the size of S , determining whether S ∈ A ℓ may require enumerating or storing all |A ℓ | elements, making heuristic design particularly difficult.

- [6] Aydın Alptekino˘ glu and John H Semple. The exponomial choice model: A new alternative for assortment and price optimization. Operations Research , 64(1):79-93, 2016.
- [7] Arthur Asuncion, David Newman, et al. Uci machine learning repository, 2007.
- [8] Peter Auer. Using confidence bounds for exploitation-exploration trade-offs. Journal of Machine Learning Research , 3(Nov):397-422, 2002.
- [9] Vashist Avadhanula, Jalaj Bhandari, Vineet Goyal, and Assaf Zeevi. On the tightness of an lp relaxation for rational optimization and its applications. Operations Research Letters , 44(5):612-617, 2016.
- [10] Abdellah Aznag, Vineet Goyal, and Noemie Perivier. Mnl-bandit with knapsacks. arXiv preprint arXiv:2106.01135 , 2021.
- [11] Gerardo Berbeglia, Agustín Garassino, and Gustavo Vulcano. A comparative empirical study of discrete choice models in retail operations. Management Science , 68(6):4005-4023, 2022.
- [12] Jose Blanchet, Guillermo Gallego, and Vineet Goyal. A markov chain approximation to choice modeling. Operations Research , 64(4):886-905, 2016.
- [13] Jose Blanchet, Renyuan Xu, and Zhengyuan Zhou. Delay-adaptive learning in generalized linear contextual bandits. Mathematics of Operations Research , 49(1):326-345, 2024.
- [14] Felipe Caro and Jérémie Gallien. Dynamic assortment with demand learning for seasonal consumer goods. Management science , 53(2):276-292, 2007.
- [15] Xi Chen, Mo Liu, Yining Wang, and Yuan Zhou. A re-solving heuristic for dynamic assortment optimization with knapsack constraints. arXiv preprint arXiv:2407.05564 , 2024.
- [16] Xi Chen, Chao Shi, Yining Wang, and Yuan Zhou. Dynamic assortment planning under nested logit models. Production and Operations Management , 30(1):85-102, 2021.
- [17] Xi Chen and Yining Wang. A note on a tight lower bound for capacitated mnl-bandit assortment selection models. Operations Research Letters , 46(5):534-537, 2018.
- [18] Xi Chen, Yining Wang, and Yuan Zhou. Dynamic assortment optimization with changing contextual information. Journal of machine learning research , 21(216):1-44, 2020.
- [19] Xi Chen, Yining Wang, and Yuan Zhou. Optimal policy for dynamic assortment planning under multinomial logit models. Mathematics of Operations Research , 46(4):1639-1657, 2021.
- [20] Wang Chi Cheung and David Simchi-Levi. Assortment optimization under unknown multinomial logit choice models. arXiv preprint arXiv:1704.00108 , 2017.
- [21] Wei Chu, Lihong Li, Lev Reyzin, and Robert Schapire. Contextual bandits with linear payoff functions. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics , pages 208-214. JMLR Workshop and Conference Proceedings, 2011.
- [22] Carlos Daganzo. Multinomial probit: the theory and its application to demand forecasting . Elsevier, 2014.
- [23] Varsha Dani, Thomas P Hayes, and Sham M Kakade. Stochastic linear optimization under bandit feedback. In COLT , volume 2, page 3, 2008.
- [24] James Davis, Guillermo Gallego, and Huseyin Topaloglu. Assortment planning under the multinomial logit model with totally unimodular constraint structures. Work in Progress , 2013.
- [25] James M Davis, Guillermo Gallego, and Huseyin Topaloglu. Assortment optimization under variants of the nested logit model. Operations Research , 62(2):250-273, 2014.
- [26] Juncheng Dong, Weibin Mo, Zhengling Qi, Cong Shi, Ethan X Fang, and Vahid Tarokh. Pasta: pessimistic assortment optimization. In International Conference on Machine Learning , pages 8276-8295. PMLR, 2023.

- [27] Louis Faury, Marc Abeille, Clément Calauzènes, and Olivier Fercoq. Improved optimistic algorithms for logistic bandits. In International Conference on Machine Learning , pages 3052-3060. PMLR, 2020.
- [28] Louis Faury, Marc Abeille, Kwang-Sung Jun, and Clément Calauzènes. Jointly efficient and optimal algorithms for logistic bandits. In International Conference on Artificial Intelligence and Statistics , pages 546-580. PMLR, 2022.
- [29] Yuxuan Han, Han Zhong, Miao Lu, Jose Blanchet, and Zhengyuan Zhou. Learning an optimal assortment policy under observational data. arXiv preprint arXiv:2502.06777 , 2025.
- [30] Ying Jin, Zhuoran Yang, and Zhaoran Wang. Is pessimism provably efficient for offline rl? In International Conference on Machine Learning , pages 5084-5096. PMLR, 2021.
- [31] Kwang-Sung Jun, Lalit Jain, Blake Mason, and Houssam Nassif. Improved confidence bounds for the linear logistic model and applications to bandits. In International Conference on Machine Learning , pages 5148-5157. PMLR, 2021.
- [32] Ashish Khetan and Sewoong Oh. Data-driven rank breaking for efficient rank aggregation. Journal of Machine Learning Research , 17(193):1-54, 2016.
- [33] Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems , 33:11791191, 2020.
- [34] Tor Lattimore and Csaba Szepesvári. Bandit algorithms . Cambridge University Press, 2020.
- [35] Joongkyu Lee and Min-hwan Oh. Nearly minimax optimal regret for multinomial logistic bandit. arXiv preprint arXiv:2405.09831 , 2024.
- [36] Joongkyu Lee and Min-hwan Oh. Improved online confidence bounds for multinomial logistic bandits. arXiv preprint arXiv:2502.10020 , 2025.
- [37] Erich L Lehmann and George Casella. Theory of point estimation . Springer Science &amp; Business Media, 2006.
- [38] Lihong Li, Yu Lu, and Dengyong Zhou. Provably optimal algorithms for generalized linear contextual bandits. In International Conference on Machine Learning , pages 2071-2080. PMLR, 2017.
- [39] Shukai Li, Qi Luo, Zhiyuan Huang, and Cong Shi. Online learning for constrained assortment optimization under markov chain choice model. Available at SSRN 4079753 , 2022.
- [40] R Duncan Luce. Individual choice behavior , volume 4. Wiley New York, 1959.
- [41] Daniel McFadden and Kenneth Train. Mixed mnl models for discrete response. Journal of applied Econometrics , 15(5):447-470, 2000.
- [42] Min-hwan Oh and Garud Iyengar. Multinomial logit contextual bandits: Provable optimality and practicality. In Proceedings of the AAAI conference on artificial intelligence , volume 35, pages 9205-9213, 2021.
- [43] Mingdong Ou, Nan Li, Shenghuo Zhu, and Rong Jin. Multinomial logit bandit with linear utility functions. arXiv preprint arXiv:1805.02971 , 2018.
- [44] Noemie Perivier and Vineet Goyal. Dynamic pricing and assortment under a contextual mnl demand. Advances in Neural Information Processing Systems , 35:3461-3474, 2022.
- [45] Paria Rashidinejad, Banghua Zhu, Cong Ma, Jiantao Jiao, and Stuart Russell. Bridging offline reinforcement learning and imitation learning: A tale of pessimism. Advances in Neural Information Processing Systems , 34:11702-11716, 2021.
- [46] Paat Rusmevichientong, Zuo-Jun Max Shen, and David B Shmoys. Dynamic assortment optimization with a multinomial logit choice model and capacity constraint. Operations research , 58(6):1666-1680, 2010.

- [47] Daniel Russo and Benjamin Van Roy. Eluder dimension and the sample complexity of optimistic exploration. Advances in Neural Information Processing Systems , 26, 2013.
- [48] Aadirupa Saha and Pierre Gaillard. Stop relying on no-choice and do not repeat the moves: Optimal, efficient and practical algorithms for assortment optimization. arXiv preprint arXiv:2402.18917 , 2024.
- [49] Aadirupa Saha and Aditya Gopalan. Active ranking with subset-wise preferences. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 3312-3321. PMLR, 2019.
- [50] Denis Sauré and Assaf Zeevi. Optimal dynamic assortment planning with demand learning. Manufacturing &amp; Service Operations Management , 15(3):387-404, 2013.
- [51] Aleksandrs Slivkins et al. Introduction to multi-armed bandits. Foundations and Trends® in Machine Learning , 12(1-2):1-286, 2019.
- [52] Kenneth E Train. Discrete choice methods with simulation . Cambridge university press, 2009.
- [53] Dong Yin, Botao Hao, Yasin Abbasi-Yadkori, Nevena Lazi´ c, and Csaba Szepesvári. Efficient local planning with linear function approximation. In International Conference on Algorithmic Learning Theory , pages 1165-1192. PMLR, 2022.
- [54] Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Y Zou, Sergey Levine, Chelsea Finn, and Tengyu Ma. Mopo: Model-based offline policy optimization. Advances in Neural Information Processing Systems , 33:14129-14142, 2020.
- [55] Mengxiao Zhang and Haipeng Luo. Contextual multinomial logit bandits with general value functions. arXiv preprint arXiv:2402.08126 , 2024.
- [56] Yu-Jie Zhang and Masashi Sugiyama. Online (multinomial) logistic bandit: Improved regret and constant computation cost. Advances in Neural Information Processing Systems , 36, 2024.
- [57] Han Zhong, Wei Xiong, Jiyuan Tan, Liwei Wang, Tong Zhang, Zhaoran Wang, and Zhuoran Yang. Pessimistic minimax value iteration: Provably efficient equilibrium learning from offline datasets. In International Conference on Machine Learning , pages 27117-27142. PMLR, 2022.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Contributions of the paper are accurately stated in the abstract and introduction. Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Potential limitations are discussed below the main theorems.

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

Justification: Assumptions are claimed in Section 2

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

Justification: No experiment.

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

Justification: No experiment.

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

Justification: No experiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: No experiment.

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

Justification: No experiment.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This theoretical work involves no sensitive data or human subjects and fully complies with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This is a pure theoretical paper and there is no societal impact of the work performed.

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

Justification:

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: This work does not use any existing code, data, or other external assets.

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

Justification: This work does not involve crowdsourcing or research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

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

## A Technical Overview and Proof of Main Results

## A.1 New Perturbation Results of Revenue Functions

As in previous works, to eliminate the multiplicative dependency on κ -1 , we follow an improved analysis that uses a second-order expansion of the revenue function R ( S | e u ) with respect to u := X ⊤ θ . However, when K ≥ 2 , the presence of multiple items in S introduces significant challenges. To handle this, a careful perturbation analysis is required to bound the resulting error. For example, as noted in [35], Equation (16) in [3] incorrectly extends the perturbation result from the logistic bandit( K = 1 ) setting, causing their analysis to fail. On the other hand, [44] provides a promising direction for handling such perturbations, though their result only applies to the uniform revenue case where r i ≡ 1 . The only promised development for non-uniform revenue setting is given by the recent work [35], where an approach based on centralizing the context features are provided in the proof of their Theorem 4. However, this argument is somewhat complex and requires a careful auxiliary analysis.

In this section, we present several simpler and general perturbation results for revenue functions for the purpose of analyzing Algorithm 1 and Algorithm 2. As a byproduct, we can show that a straightforward plug-in argument based on our lemma allows us to extend the regret analysis in [44]-which previously applied only to the uniform revenue case-to the general non-uniform revenue setting, achieving the optimal ˜ O ( d √ T + κ -1 d 2 ) regret, we leave the detail to Section E.5.

Proposition 3. For any fixed revenue vector r and utility vectors u ′ ∈ R N , denote u := log( v ) and w := u ′ -u , v ′ := e u ′ , then i) it holds for any S 0 ∈ S K that

<!-- formula-not-decoded -->

ii) In addition, if S satisfies r ≥ R ( S | v ) for all j ∈ S , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Based on Proposition 3, we can further present the following result for the analysis of eliminationbased Algorithm 2.

Proposition 4. Suppose for some ˜ v ≥ v it holds that R ( S 0 | ˜ v ) ≥ R ( S ⋆ | ˜ v ) -ε , then for w = ˜ u -u we have

<!-- formula-not-decoded -->

## A.2 Proof of Theorem 2

Proof. Throughout the proof, we denote ξ i := 72 ∥ x i ∥ H -1 D ( ̂ θ ) √ log( N eff /δ ) , ̂ θ D by ̂ θ for simplicity. With above notation, we have

<!-- formula-not-decoded -->

Now we have for v = exp( X ⊤ θ ⋆ ) , it holds by Theorem 1 that with probability at least 1 -δ that v LCB ≤ v . Under such a condition, we have then

<!-- formula-not-decoded -->

Where (i) is by the monotone property of v at its the revenue maximizer(see e.g. Lemma A.3 of [5] (ii) is by R ( S ⋆ | v LCB ) ≤ R ( S | v LCB ) , (iii) is by the statement ii) of Proposition 3 and the fact r j ≥ R ( S ⋆ | v ) , ∀ j ∈ S ⋆ . This finishes the proof.

## A.3 Proof of Theorem 3

We first show the following exploration length upper bound result:

Lemma 1. In Algorithm 2, there exits some absolute constant c 0 so that the exploration phase will stop after at most C 0 Mκ -1 d (√ d log( NT ) ∨ W ) 2 log( dNTW ) iterations, and it holds that

<!-- formula-not-decoded -->

for every ℓ ∈ [ M +1] .

In particular, Lemma 1 verifies the burn-in condition in Theorem 1 with λ = 1 . Consequently, it can be applied in subsequent time steps as long as the independence assumption is satisfied, which is guaranteed by the SupCB elimination framework.

Based on our exploration-phase analysis in Lemma 1, we have the exploration phase incurs a regret of order ˜ O ( κ -1 d 2 K 2 log( NT )) , and the burn-in condition in Theorem 1 is satisfied for all bins. So it remains to bound the regret incurred from τ +1 to T under the event

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

which holds with probability at least 1 -O (1 /T ) .

We would also note that (9) and (10) together with (4) also implies

<!-- formula-not-decoded -->

thus for any S and i ∈ S,

<!-- formula-not-decoded -->

with ̂ u i := x ⊤ j ̂ θ 0 . As a result,

<!-- formula-not-decoded -->

for any ℓ, S and t ≥ τ.

Lemma 2. For every ℓ and S ∈ A ℓ it holds under (9) and (10) that

<!-- formula-not-decoded -->

Proof of Lemma 2. We divide the proof into two steps:

Step 1: S ⋆ ∈ A ℓ for each ℓ. At each t ≥ τ , we first show that S ⋆ ∈ A ℓ for all ℓ : The claim holds for ℓ = 0 since A 0 = S K . Now suppose S ⋆ ∈ A ℓ -1 for some ℓ ≥ 1 , then suppose the algorithm enters the step (c) at the ℓ -th loop, it holds that by v UCB t,ℓ ≥ exp( X ⊤ θ ⋆ ) under (9) and (10),

<!-- formula-not-decoded -->

for ̂ S ℓ := argmax S ∈A ℓ R UCB t,ℓ ( S ) . On the other hand, we cannot directly apply Proposition 3 to obtain a perturbation bound for ̂ S ℓ , since it maximizes the optimistic revenue over an unstructured set A ℓ rather than the structured set S K .

For ̂ S ℓ , it holds that by statement i) of Proposition 3,

<!-- formula-not-decoded -->

Now if we denote then it holds that

<!-- formula-not-decoded -->

where in the last second inequality we have used

<!-- formula-not-decoded -->

by S ∈ A ℓ . As a consequence, for ∆ := R UCB t,ℓ ( ̂ S ℓ ) -R ( ̂ S ℓ | θ ⋆ ) we get

<!-- formula-not-decoded -->

Using the elementary inequalities

<!-- formula-not-decoded -->

for A,B,z ≥ 0 , we get then

<!-- formula-not-decoded -->

where the last second inequality is by (11) and the last inequality is by Algorithm 2 does not enter step (a) in ℓ -th loop. Now we get

<!-- formula-not-decoded -->

thus then R UCB t,ℓ ( S ⋆ ) ∈ A ℓ , as desired.

Step 2: Bound the regret of assortments in A ℓ Noticing that by the selection of A ℓ , we have

<!-- formula-not-decoded -->

where in the last line we have used Lemma 4, as desired.

Similar to the proofs in [38, 31], we can bound the regret of Algorithm 2 in a layer-wise approach:

Lemma 3. For each round t &gt; τ , let ℓ t denote the value of ℓ when S t is selected. Then, under (9) and (10) , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We have by S t ∈ A ℓ t ,

<!-- formula-not-decoded -->

this shows the first inequality.

To show the second inequality, noticing that by Lemma 2, the condition of step (b) implies

<!-- formula-not-decoded -->

Then by

<!-- formula-not-decoded -->

the desired result holds.

Now with Lemma 3, we can bounding the regret of Algorithm 2 for each bin Ψ ℓ separately. More precisely, we have

Lemma 4. For each ℓ ∈ [ S ] , we have condition on E 1 , E 2 ,

<!-- formula-not-decoded -->

Proof. We divide the time indices in Ψ ℓ into

<!-- formula-not-decoded -->

By Lemma 3 and 2 -ℓ ≤ W ℓ t,S t for all j ∈ S t and t ∈ T a , we have

<!-- formula-not-decoded -->

To bound the above summation over T a,ℓ , we apply the following linear MNL version elliptical potential lemma:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Applying this Lemma and Cauchy-Schwartz inequality in (13) then leads to

<!-- formula-not-decoded -->

this provides the regret bound for the summation over T a,ℓ .

On the other hand, by Lemma 2 we have taking summation over t ∈ T b,ℓ naturally leads to

<!-- formula-not-decoded -->

combining the upper bounds for T a,ℓ , T b,ℓ and taking summation over ℓ then finishes the proof.

## B Proof of Theorem 1

We begin with the following Lemma:

Lemma 6. Suppose the same independence structure as in Theorem 1 and denote

<!-- formula-not-decoded -->

Proof of Lemma 6. Since in this formulation the feature vector x m,j may change for each m , we introduce the notation p mj ( θ ) to represent the probability that item j is chosen given S m and x m . We also denote H λ D ( θ ⋆ ) by H λ t , H D ( θ ⋆ ) by H t for simplicity.

Noticing that by ̂ θ λ D maximizes ℓ λ D , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now if we denote then for any θ it holds that

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

Now notice that H λ t = ∑ t m =1 DL m ( θ ⋆ ) + λI and for

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## The first term J 1

The first term can be bounded by variance-aware concentration results using conditional independence as in previous works obtaining sharp bounds for logistic setting [31] , the main difference between the logistic setting is the dependency across η m,j for j ∈ S m :

For every m , we have is centered and bounded as

<!-- formula-not-decoded -->

With variance

By we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

now by ∑ t m =1 U m = H t , we have ∑ m E [ Z 2 m ] = x ⊤ ( H λ t ) -1 H t ( H λ t ) -1 x ≤ x ⊤ ( H λ t ) -1 x , now we can apply the following Bernstein inequality:

Lemma 7 (Bernstein's Inequality) . Let X 1 , . . . , X n be independent zero-mean random variables. Suppose that | X i | ≤ M almost surely, for all i . Then, for all positive t ,

<!-- formula-not-decoded -->

to obtain that

<!-- formula-not-decoded -->

or equivalently, with probability at least 1 -δ,

<!-- formula-not-decoded -->

## The second term J 2

For the second term, we have denote z t := ∑ t m =1 ∑ j ∈ S m x m,j η m,j , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Bounding ∥ ( H λ t ) -1 E t ( H λ t ) -1 ∥ . Given any vector w , if we denote

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

for j ∈ S m and set a m 0 = b m 0 = u m 0 = 0 , v m 0 = 1 for convenience. Then we have as shown in [35],

<!-- formula-not-decoded -->

It can be verified by calculation that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now we have for any w ∈ R d , by taking summation over m ,

<!-- formula-not-decoded -->

this leads to

<!-- formula-not-decoded -->

On the other hand, we have by

<!-- formula-not-decoded -->

where the last inequality is by the elementary inequality 1 -e -x x ≥ 1 1+ x , x &gt; 0 . With (17) and (18), we arrive at

<!-- formula-not-decoded -->

and thus

<!-- formula-not-decoded -->

Bounding ∥ ( H λ t + E ) -1 z t ∥ H λ t : By (18), we have

H t ⪯ (1 + ζ )( H t + E t ) = ⇒ ( H λ t + E t ) -1 H λ t ( H λ t + E t ) -1 ⪯ (1 + ζ )( H λ t + E ) -1 ⪯ (1 + ζ ) 2 H λ t . thus it holds that

<!-- formula-not-decoded -->

To control ∥ z t ∥ ( H λ t ) -1 , we have for any unit vector w , it holds that

<!-- formula-not-decoded -->

which has the same form as J 1 when we replace ˜ x by x, thus by the same argument as in bounding J 1 , we have (15) still holds: with probability at least 1 -δ ′ ,

<!-- formula-not-decoded -->

now taking w over the 1 / 2 -net of the unit ball and taking union bound with selecting δ ′ ≍ 4 d δ leads to with probability at least 1 -δ,

<!-- formula-not-decoded -->

Further introduce the notation ξ = max m ≤ t,j ∈ S m ∥ x m,j ∥ ( H λ t ) -1 in Lemma 6, we have then

<!-- formula-not-decoded -->

Now we arrive at

<!-- formula-not-decoded -->

Remark 1. In the above proof of (17) , we borrow the calculation in [35], which was used to verify the self-concordant-like property of the linear MNL-likelihood function. The key distinction in our approach is that we retain the b mj term throughout the calculation, whereas [35] directly apply the bound | b mj | ≲ ∥ ̂ θ λ D -θ ⋆ ∥ 2 . This difference allows us to derive a bound on E t that depends only on max m,j | x ⊤ mj ( ̂ θ λ D -θ ⋆ ) | , rather than on ∥ ̂ θ λ D -θ ⋆ ∥ 2 . It is worth noting that a very recent work [36] also uses a similar argument to establish a refined self-concordant-like property for the MNL likelihood function (specifically, the ℓ ∞ self-concordant-like property in their Proposition B.3) to obtain sharper confidence bounds, and the bound in (17) can also be derived from their Proposition B.3.

While the above proof primarily focuses on the relation between H D ( ̂ θ λ D ) and H D ( θ ⋆ ) , the inequality in (16) can also imply a dominance relation between H D ( θ ) and H D ( θ ⋆ ) for general θ . We summarize this general result in the following proposition for future applications.

Proposition 5. Given any θ ∈ R d , λ ≥ 0 , denoting ζ θ := 3 √ 2 max m ≤ t,j ∈ S m | x ⊤ m,j ( θ -θ ⋆ ) | , then it holds that

<!-- formula-not-decoded -->

Proof of Proposition 5. By taking s = 1 in (16), we can get

<!-- formula-not-decoded -->

for any unit vector w , thus it holds that

<!-- formula-not-decoded -->

Now by

<!-- formula-not-decoded -->

## The last term J 3

For the last term in (14), we have by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Combining all bounds

Now combining all above bounds, we get

<!-- formula-not-decoded -->

as desired.

Lemma 6 provides an upper bound on the ratio between x ⊤ ( ̂ θ λ D -θ ⋆ ) and ∥ x ∥ ( H λ ) -1 . However, the upper bound involves both ξ and ζ simultaneously. Our next lemma shows that when ξ is well-controlled, the ζ factor can be bounded by ξ .

Lemma 8. Under the condition

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

Proof. Applying Lemma 6 to each x m,j , we have then with probability at least 1 -δ,

<!-- formula-not-decoded -->

Now by the elementary inequality

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we get when ζ ≤ 3 5 , the above inequality can be reduced to

<!-- formula-not-decoded -->

As a result, if it holds simultaneously that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then

On the other hand, by (19), it always holds that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

it holds that

<!-- formula-not-decoded -->

As a consequence,

<!-- formula-not-decoded -->

That verifies ξ ≤ min { 1 64 √ d log( N eff /δ ) , 1 64 √ λW } is sufficient for (20) holds, as desired.

Now combining the results in Lemma 6 and Lemma 8, we have ξ ≤ min { 1 64 √ d log( N eff /δ ) , 1 64 √ λW } implies that

<!-- formula-not-decoded -->

This finishes the proof of the confidence bound result under the burn-in condition.

Moreover we have by φ ( ζ ) ≤ ζ ≤ 3 / 5 , Proposition 5 implies

<!-- formula-not-decoded -->

This finishes the proof of Theorem 1.

## C Perturbation Results for Revenue Functions

Before proving results in Section 4 and Section 5, we first introduce several perturbation results on the revenue function R with respect to the utility function u .

To emphasis its dependency on the utility variable u , we use the following notation throughout this section for given S and r :

<!-- formula-not-decoded -->

Since r is fixed throughout the whole paper, we denote Q ( u ; r , S ) by Q ( u ; S ) for simplicity. We first recall the following first-order and second-order derivative results, originally proved in [44] and refined in [35], we provide the detailed proof only for completeness.

Proposition 6 (Lemma E.3 of [35]) . Given r ∈ [0 , 1] N and S ⊂ [ N ] fixed, if we denote

<!-- formula-not-decoded -->

for w ∈ R N . Then for any u ∈ R N , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Proposition 6. During the proof S is fixed, thus we simply omit the notation and simply denote Q ( u ; S ) as Q ( u ) .

To prove (23), noticing that for every i / ∈ S, ∂ u i Q ( u ) = 0 and for every i ∈ S, we have

Thus

<!-- formula-not-decoded -->

as desired.

To prove (24), noticing that ∂ 2 u i ,u j Q ( u ) = 0 if i / ∈ S or j / ∈ S. When i, j ∈ S , we have

<!-- formula-not-decoded -->

Noticing that for i, j ∈ S,

<!-- formula-not-decoded -->

Thus

̸

and for j = i,

As a result,

<!-- formula-not-decoded -->

This finishes the proof.

Based on the first and second-order derivatives of Q , we now introduce the following perturbation result:

Proposition 7 (Proposition 3, restated) . For any fixed r and u , u ′ ∈ R N , denote w := u ′ -u , then i) it holds for any S 0 ∈ S that

<!-- formula-not-decoded -->

ii) In addition, if S 0 is the maximizer of R ( S ; e ˜ u ) over S for some ˜ u ≥ u element-wisely, then

<!-- formula-not-decoded -->

iii) In the uniform-reward setting, where r i ≡ 1 , ∀ i ∈ [ N ] , we have

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Proof. We have by Proposition 6,

<!-- formula-not-decoded -->

On the other hand, we have

<!-- formula-not-decoded -->

This finishes the proof of the statement (i).

To prove the second inequality, it suffices to provide upper bound of ∑ j ∈ S 0 v j | r j -Q ( u ; S 0 ) | : We first recall the following structural result of the revenue maximizer:

Proposition 8 (Lemma A.3, [5]) . For any given u denote ¯ S := argmax S ∈S K Q ( u ; S ) , then it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By S 0 is the maximizer of R ( S ; ˜ v ) , we have r j ≥ Q ( ˜ u ) for all j ∈ S 0 by Proposition 8. We further have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

thus then by we have

<!-- formula-not-decoded -->

This finishes the proof of statement (ii).

Finally for statement (iii), we have by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

this finishes the proof.

The proof of the above result relies on the condition that r j ≥ R ( S ; e u ) for all j ∈ S , which holds only when S is the best assortment under some ˜ u ≥ u . However, this does not apply to Algorithm 2, which is based on elimination. To handle this, we extend the result to a class of near-optimal assortments, which can be applied for analyzing elimination-based algorithms.

Proposition 9 (Proposition 4 restated) . Under the same notation of Proposition 7 and suppose for some ˜ u ≥ u it holds

<!-- formula-not-decoded -->

then for v = e u , w = ˜ u -u and S ⋆ = argmax S ∈S R ( S ; v ) we have

<!-- formula-not-decoded -->

Proof. We have by statement (i) of Proposition 7,

<!-- formula-not-decoded -->

Denote then by

we have

<!-- formula-not-decoded -->

where in the last line we have used

<!-- formula-not-decoded -->

and v j ≤ 1 + ∑ i ∈ S 0 v i = p -1 0 ( S 0 | v ) .

<!-- formula-not-decoded -->

Now using the elementary inequality for quadratic equation solutions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all x ≥ 0 . We have

<!-- formula-not-decoded -->

as desired.

## D Proof of Results in Section 4

## D.1 Proof of Proposition 1 and Corollary 1

Proof. For each j ∈ S ⋆ and any λ &gt; 0 , assume W.L.O.G. that j is exactly contained in S 1 , . . . , S n j , then by

<!-- formula-not-decoded -->

we have then

<!-- formula-not-decoded -->

Taking limit as λ j → 0 and using the continuity of ∥ x j ∥ 2 ( H D ( θ ⋆ )+ λI ) -1 leads to ∥ x j ∥ 2 H -1 ( θ ⋆ ) ≤ γ -1 j .

D Finally, by

<!-- formula-not-decoded -->

This finishes the proof of Corollary 1, and the second inequality implies Proposition 1.

## D.2 Details of Proposition 2

In this section we provide the detail on how to plug other confidence region results in Algorithm 1 to achieve a burn-in-free offline learning guarantee. In [44], confidence region results with the radius ˜ O ( √ d ∥ x j ∥ H -1 D ( θ ⋆ ) ) are available 3 , and we take the result in [44] to establish the confidence region for example. Here we first restate the confidence region bound of [44]:

3 It should be noted that the original result in [44] includes an additional K -dependency due to the selfconcordant coefficient they established for the MNL likelihood function. This coefficient was later refined by [35], and incorporating their result eliminates the K -dependency in [44].

Lemma 9 (Proposition 3.3 and Lemma C.4 in [44]) . With the selection λ = √ d/W and denoting

<!-- formula-not-decoded -->

it holds that with probability at least 1 -δ,

<!-- formula-not-decoded -->

With Lemma 9 and (26), we now assign the LCB values for each j ∈ [ N ] as

<!-- formula-not-decoded -->

It can be seen from Lemma 9 that with probability at least 1 -δ,

<!-- formula-not-decoded -->

Now replacing the term u ⋆ -̂ u j by u ⋆ j -˜ u LCB j the right-hand-side upper bound our analysis in Section A.3 leads to Proposition 2, as desired.

## E Proof of Results in Section 5

## E.1 Proof of Lemma 1

Lemma 1 is a direct corollary of the following result from [47, 53]:

Lemma 10. For any given X ⊂ { x ∈ R d : ∥ x ∥ 2 ≤ 1 } threshold µ &gt; 0 and regularizer λ &gt; 0 , the following procedure starting with C = ∅ :

while ∃ x ∈ X so that ∥ x ∥ 2 ( V λ C ) -1 &gt; µ for V λ C := ∑ z ∈C zz ⊤ + λI do: add this x to C .

will stop in at most

<!-- formula-not-decoded -->

steps.

Applying this lemma with

<!-- formula-not-decoded -->

to each ℓ separately then leads to the desired result.

## E.2 Proof of Theorem 4

Based on the burn-in condition established in Lemma 1, which incurs at most poly-logarithmic regret in T , it remains to bound the regret incurred from τ +1 to T.

Again, it suffices to perform regret analysis under the inequalities (9) and (10), under which we can show that

<!-- formula-not-decoded -->

holds using the same argument for establishing (11).

Now noticing that with the uniform revenue assumption r i ≡ ¯ r for some 0 ≤ ¯ r ≤ 1 , statement i) in Proposition 3 can be refined to

<!-- formula-not-decoded -->

where the last inequality is by

<!-- formula-not-decoded -->

With this result, we have the following analogue of Lemma 2 directly in this setting: For every ℓ and S ∈ A ℓ it holds under (9) and (10) that

<!-- formula-not-decoded -->

Plugging (27) to the subsequent analysis in the proof of Theorem 3 then leads to

<!-- formula-not-decoded -->

where the last line is by Lemma 11 of [44]. Now if we denote ∆ := ∑ t ∈T a,ℓ R ( S ⋆ ) -R ( S t ) and apply the fact

<!-- formula-not-decoded -->

with A = ( d + W ) log( NT ) and B = √ κ ⋆ ( d + W ) T log( NT ) + d log( NT ) κ , the desired result holds.

## E.3 Proof of Theorem 5

In this section, we show the following problem-dependent lower bound result:

Theorem 5 (Problem-Dependent Lower Bound) . For any given 0 &lt; ¯ κ &lt; 1 , we can find a class of problem instances V with uniform revenue and d -dimensional linear MNL choice feedback so that for any instance in V the corresponding parameter κ ⋆ ≤ ¯ κ and for any policy π , there exists some instance in V so that over such instance.

The construction of hard instance for proving Theorem 5 relies only on a minor modification of the proof in [17], and we just provide the detailed proof here for completeness.

Step 1: Construction of Hard Instances. We let r 1 = · · · = r N = 1 for all instances constructed later, and given any K -sized assortment S, we consider the corresponding attraction value construction as

<!-- formula-not-decoded -->

And we define the problem instance set as

<!-- formula-not-decoded -->

It can be seen that we always have S ⋆ ( v S ) = S for every v, and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

holds for every v S by definition.

Since there exists a one-to-one correspondence between S K and V K via S → v S , we interchangeably use the notations v S and S for convenience.

Step 2: Verifying the Separation Condition. We have the following separation result over S Proposition 10. For every fixed S 0 ∈ S and corresponding v S 0 ∈ V K , it holds for any S ∈ S that

<!-- formula-not-decoded -->

as desired.

Proof of Proposition 10. Denoting R ( ·| v S 0 ) by R ( · ) thorough the proof for simplicity, we have

<!-- formula-not-decoded -->

and for any S,

<!-- formula-not-decoded -->

thus

<!-- formula-not-decoded -->

Step 3: Decomposing the Regret. Based on Proposition 10, we have

<!-- formula-not-decoded -->

Now for each i, we define S ( i ) K -1 . = S K -1 ∩ { S ⊂ [ N ] : i / ∈ S } , then it holds that

<!-- formula-not-decoded -->

Now for the second term, we have

<!-- formula-not-decoded -->

For the first term, by Pinsker's inequality, denoting the probability induced by environment S ′ ∪ { i } as P and induced by S ′ as Q , then

<!-- formula-not-decoded -->

As a result, we have

<!-- formula-not-decoded -->

Where the last line is by when 4 K ≤ N.

Step 4: Upper Bounding the KL Divergence. We have for every S ′ ∈ S ( i ) K -1 , it holds that

<!-- formula-not-decoded -->

Where the last line is by KL ( P S ′ ( ·| ¯ S ) ∥ Q S ′ ( ·| ¯ S ) ) = 0 as long as i / ∈ ¯ S.

For every ¯ S containing i , we have denote K ′ = | ¯ S | and p j . = P S ′ ( j | ¯ S ) , q j . = Q S ′ ( j | ¯ S ) for j ∈ ¯ S + , then i) For j = 0 ,

<!-- formula-not-decoded -->

̸

ii) For j = i, iii) For j = i,

<!-- formula-not-decoded -->

Now applying the following Proposition from [17]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition 11. Let P and Q be two categorical distributions on J items, with parameters p 1 , · · · , p J and q 1 , · · · , q J respectively. Denote also ε j . = p j -q j . Then KL( P ∥ Q ) ≤ ∑ J j =1 ε 2 j /q j .

we get

<!-- formula-not-decoded -->

Final Step: Putting All Together. Using the above upper bound of KL divergence, we get

<!-- formula-not-decoded -->

Selecting ε 2 = 1 512 √ N Tκ ⋆ then leads to the Ω( √ κ ⋆ NT ) lower bound, as desired.

## E.4 Extension to Time-Varying Contexts

Another problem setting widely adopted in linear MNL and generalized linear bandit frameworks is the adversarial context setting with an initial exploration period [38, 18, 31, 42]. In this setting, instead of always using the same feature map x j , we allow the feature vector x t,j to vary with t . Furthermore, during the initial exploration period, the observed features x t,j are drawn i.i.d. from some distribution P 0 with λ min ( E x ∼ P 0 [ xx ⊤ ]) ≥ σ 0 for some σ 0 &gt; 0 . In general, The fixed action setting does not fit this framework as it violates the eigenvalue lower bound assumption, making previous exploration-phase designs inapplicable. While our algorithmic description focuses on the fixed feature set setting, it can be readily extended to the time-varying context setting, achieving a regret of ˜ O ( √ dT + κ -1 d 2 K ) . This holds under the same stochastic context assumption during the initial exploration phase and allows for an even simpler design in the exploration phase, thanks to the generality of Theorem 1.

In this section, we extend Algorithm 2 to the setting that the observed context at each round can be different vectors in R d , more precisely, at each time t , the i -th item is associated with a feature map x t,i ∈ R d . And we impose the following eigenvalue assumption in the exploration phase, as in [42, 18]:

Assumption 2. For some given t 0 , { x t,i } i ∈ [ N ] ,t ∈ [ t 0 ] are generated i.i.d. from some unknown distribution P supported on the d -dimensional unit ball, with λ min ( E x ∼ P [ xx ⊤ ]) ≥ σ 0 for some σ 0 &gt; 0 .

We first present the modified algorithm for time-varying contexts in Algorithm 3, with modifications highlighted in blue. In the elimination phase, the only change is that item-wise uncertainty levels are computed over x t,i instead of x i for each i , which then affects W ℓ t,S and the UCB revenues. Thus the same analysis as in Section A.3 can be conducted to derive the same regret bound once the burn-in condition can be verified.

The main difference lies in the initial exploration phase, where any K -sized assortment can be selected for efficient exploration due to Assumption 2.

## Algorithm 3 SupLinearMNL with Time-Varying Contexts

```
1: Input: Time horizon T, exploration length n 0 , τ . initialize S = log T, , Ψ 1 = · · · = Ψ S +1 = ∅ . 2: for t = 1 , . . . , ( S +1) τ do 3: Select arbitrary assortment in S K , add t into Ψ ⌈ t/τ ⌉ 4: end for 5: Ψ 0 ←∅ , compute ̂ θ 0 as in (5). 6: for t = ( S +1) τ +1 , . . . , T do 7: set A 1 = S K , S t = ∅ , ℓ = 1 8: while S t = ∅ do 9: Compute W ℓ t,S , R UCB t,ℓ ( S ) , ∀ S ∈ A ℓ as in (7), (8) , with w ℓ t,i := 72 ∥ x t,i ∥ H -1 t,ℓ ( ̂ θ 0 ) √ log( NT ) 10: if W ℓ t,S > 2 -ℓ for some S ∈ A ℓ then 11: select such S ∈ A ℓ . 12: Ψ ℓ ← Ψ ℓ ∪ { t } 13: else if W ℓ t,S ≤ 1 / √ T for all S ∈ A ℓ then 14: take the action S t = argmax S ∈A ℓ R UCB t,ℓ ( S ) 15: Ψ 0 ← Ψ 0 ∪ { t } 16: else 17: ̂ R ← max S ∈A ℓ R UCB t,ℓ ( S ) 18: A ℓ +1 ← { S ∈ A ℓ , R UCB t,ℓ ( S ) ≥ ̂ R -2 -ℓ +2 } 19: ℓ ← ℓ +1 20: end if 21: end while 22: end for
```

Now it sufficient to prove the following burn-in condition guarantee:

Lemma 11. With the selection τ = Ω( σ -1 0 [ d log T/σ 0 + √ d log( NT ) ∨ K -1 √ W ) , it holds that with probability at least 1 -2 /T , the event ˜ E 1 ∩ ˜ E 2 , with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

holds.

Proof. We need only show that with probability at least 1 -1 /T, after the exploration phase, it holds for every ℓ that

<!-- formula-not-decoded -->

From this, we obtain

<!-- formula-not-decoded -->

which then allows the result to follow naturally from Theorem 1 and Proposition 5.

Now to prove (28), we simply recall the following result in [42] and [38]:

Proposition 12 (Proposition 1 in [42]&amp; Proposition 1 in [38]) . For any constant B &gt; 0 , there exists some absolute constant c 1 , c 2 &gt; 0 so that with the selection

<!-- formula-not-decoded -->

it holds with probability at least 1 -1 /T 2 that H t,ℓ ( θ ⋆ ) ⪰ BI.

Now selecting τ as in Proposition 12 with B = 64 K √ d log( NT ) ∨ 64 √ W then finishes our proof.

## E.5 Extension of OFU-MNL [44]

In this section, we show that a simple plug-in argument based on the developed perturbation result Proposition 3 can extend the result in Theorem 3.4 of [44] for uniform revenue setting to the general revenue setting:

Theorem 6. The OFU-MNL algorithm (Algorithm 2, [44]) satisfies the regret up to logarithmic factors.

Proof of Theorem 6. Simply noticing that the OFU-MNL algorithm always taking the optimistic action

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with C t specified as in Lemma 9.

Proposition 3.3 in [44] has shown that v UCB t ≥ v , ∀ t ∈ [ T ] with probability at least 1 -O (1 /T ) , thus then we have under such event, it holds that r j ≥ R ( S t | v UCB t ) ≥ R ( S ⋆ | v ) . Now denote w tj := v UCB tj -v tj , we have then

<!-- formula-not-decoded -->

where (i) is by statement ii) of Proposition 3, (ii) is by Proposition 3.3 and Lemma 9. Finally, applying the elliptic potential lemma presented in Lemma 5 leads to the desired result.

## F Experiment Results

In this section, we present additional numerical results to illustrate the effect of the burn-in condition and to compare Algorithm 1 with existing offline assortment optimization benchmarks.

## F.1 Sensitivity to Burn-in Condition

To better understand the trade-off between the tightness of our confidence intervals (CIs) and the burn-in condition, we conduct a set of numerical experiments comparing our bound in Theorem 2 with that of [44]. Our CIs are theoretically tighter than those in [44, 35, 3] in that the confidence radius is smaller by a √ d factor, but requires an additional burn-in condition and a conditional independence assumption. The following simulation demonstrates how violating the burn-in condition may lead to the failure of our CI guarantee, while satisfying it yields improved tightness.

## F.1.1 Evaluation and Results

We compare the empirical tightness of the two confidence bounds using the following ratio:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for

where the denominator corresponds to the theoretical confidence radius at coordinate e i . A smaller CI ratio indicates a tighter alignment between the theoretical CI and the actual estimation error, and thus a lower likelihood of CI violation.

We evaluate the CI ratio under different burn-in constants, defined as

<!-- formula-not-decoded -->

Theoretically, our CI guarantee in Theorem 2 requires ζ ≲ 1 / √ d , whereas the bound in [44] imposes no such restriction.

Table 2: Comparison of CI ratios under varying burn-in constants.

| Value of ζ        |   0 . 130 |   0 . 134 |   0 . 141 |   0 . 160 |   0 . 335 |
|-------------------|-----------|-----------|-----------|-----------|-----------|
| CI ratio (Theorem |     2.563 |     2.831 |     3.147 |     4.052 |     5.617 |
| CI ratio ([44])   |     3.495 |     3.48  |     3.42  |     3.27  |     2.852 |

We present the result in Table 2, and defer the detailed experiment setup in the next section. As ζ increases, our CI ratio grows noticeably, reflecting a higher risk of CI violation when the burn-in condition is not met. In contrast, the ratio for [42] remains stable across ζ , consistent with its theory-independent nature. This comparison highlights a fundamental trade-off: our CI achieves tighter bounds under well-conditioned data but is more sensitive to initialization.

## F.1.2 Experimental Setup

We describe below how the feature map, parameter, and assortment sets are constructed to obtain results in Table 2.

Feature Set and Parameters. For given ( d, K ) (assuming d even) and a parameter radius W &gt; 0 , we define the item features as follows:

- Type-1 items: For i ≤ d -1 , x i = e i (canonical basis vectors).
- Type-2 item: x d = (1 , 1 , . . . , 1) / √ d .

The true parameter is set as θ ⋆ = ( W,W,...,W ) / √ d .

Assortment Sets. We consider two types of assortments:

- Type-1 assortment: S 0 = { d } , containing only the Type-2 item.
- Type-2 assortments: Let m = ⌈ d/K ⌉ . For each 1 ≤ k ≤ m , define

<!-- formula-not-decoded -->

Observed Data. Given integers n 1 , n 2 , we generate n 1 copies of S 1 , . . . , S m and n 2 copies of S 0 , with choices sampled under the linear MNL model parameterized by ( X,θ ⋆ ) as above.

Relation Between W and Burn-in Constant. Under this construction, ζ increases monotonically with W . The mapping between W and ζ used in our experiments is shown below.

Table 3: Mapping between W and burn-in constant ζ .

| Value of W   |   0 . 5 |   0 . 841 |   1 . 41 |   2 . 37 |   4 . 0 |
|--------------|---------|-----------|----------|----------|---------|
| Value of ζ   |    0.13 |     0.134 |    0.141 |     0.16 |   0.335 |

All results reported in Table 2 are based on d = 6 , K = 3 , and n 1 = n 2 = 500 . This setup allows us to systematically vary W and thus ζ , thereby illustrating the relationship between burn-in strength and confidence interval validity.

## F.2 Results for Offline Assortment Optimization

Since in online setting, Algorithm 2 is computationally inefficient. We focus on Algorithm 1 in the offline setting. We consider two variants of our approach based on different confidence intervals: one derived from Theorem 2 and the other from Lemma 9, denoted by LCB-MLE and LCB-MLE-v2 , respectively. These experiments are designed to examine the trade-off between statistical conservativeness and empirical performance, and to compare our results with the PASTA algorithm [26].

We evaluate the empirical performance of our proposed confidence-based algorithms under four representative settings that differ in coverage and model specification conditions. Settings 1 and 2 examine performance under varying coverage levels in the standard linear MNL model, while Settings 3 and 4 study robustness under misspecified mixture-MNL environments.

Setting 1: Full Coverage. In this setting, both the item features X and the true parameter θ ⋆ are independently drawn from spherical uniform distributions. The observed assortments are uniformly sampled from all K -sized subsets of [ N ] , ensuring that every item is well covered. The experimental parameters are N = 30 , d = K = 10 , and W = 1 , and results are averaged over 10 repetitions.

Table 4: Comparison of SubOpt Gap (smaller is better) under full coverage, N = 30 , d = K = 10 , W = 1 , over 10 repetitions.

|    n |   LCB-MLE |   LCB-MLE-v2 |   PASTA |
|------|-----------|--------------|---------|
|  100 |    0.0073 |       0.0087 |  0.0077 |
|  300 |    0.0056 |       0.0088 |  0.0048 |
|  500 |    0.0051 |       0.0082 |  0.0041 |
| 1000 |    0.0051 |       0.0069 |  0.0046 |

Under full coverage, PASTA slightly outperforms both LCB-MLE variants. Moreover, LCB-MLEv2-with its larger lower-confidence penalty-shows higher conservativeness and thus lower empirical efficiency. This observation aligns with our theoretical insight: when all items are well explored, excessive conservativeness can lead to under-selection and higher SubOpt gaps.

Setting 2: Partial Coverage. Wenext consider a setting with partial coverage to examine robustness when the observation set does not fully span the item space. We use the same setup for ( X,θ ⋆ ) , N , K , and d as in Setting 1, and fix the total number of observations at n = 500 . For each n ⋆ , assortments are constructed by including n ⋆ items from the optimal set and filling the remaining K -n ⋆ items with randomly sampled non-optimal ones. Increasing n ⋆ therefore improves the effective coverage of the optimal set.

Table 5: Comparison of SubOpt Gap (smaller is better) under partial coverage.

|   n ⋆ per S |   LCB-MLE |   LCB-MLE-v2 |   PASTA |
|-------------|-----------|--------------|---------|
|           2 |    0.005  |       0.0057 |  0.0047 |
|           4 |    0.0004 |       0.0004 |  0.0021 |
|           6 |    0.0007 |       0      |  0.0022 |
|           8 |    0.0003 |       0      |  0.0019 |

Under partial coverage, both LCB-MLE variants consistently outperform PASTA . In particular, LCB-MLE-v2 achieves the smallest SubOpt Gap, suggesting that stronger conservativeness improves robustness when data coverage is limited. These results reveal a complementary pattern: LCB-MLE performs well in well-covered settings, whereas LCB-MLE-v2 is more effective in partially observed regimes.

Setting 3: Misspecification under mixture of MNL I. In this setting, we consider a mixture MNL model with two subgroups, where the second subgroup is selected with probability µ , and the total sample size is fixed at n = 500 . The generation of ( X,θ ⋆ ) and parameters ( N,K,d ) follows

Table 6: Comparison of Expected Revenue (larger is better) under mixed MNL model I.

|    µ |   LCB-MLE |   LCB-MLE-v2 |   PASTA |
|------|-----------|--------------|---------|
| 0.05 |    0.9312 |       0.928  |  0.9316 |
| 0.2  |    0.9279 |       0.9248 |  0.9277 |
| 0.5  |    0.9246 |       0.9234 |  0.9246 |
| 0.7  |    0.9263 |       0.9267 |  0.9262 |
| 0.9  |    0.9312 |       0.9312 |  0.9309 |

the same configuration as in Setting 1. Each assortment consists of 5 items sampled from the first subgroup's optimal set and 5 from the rest.

Since computing the true optimal assortment under a mixture MNL is intractable, we evaluate performance using the expected revenue instead of SubOpt Gap. As µ approaches 0 . 5 , model misspecification becomes more severe and the performance of all methods slightly degrades. Overall, all three algorithms perform comparably, showing that our confidence-based methods remain stable under mild model mismatch.

Table 7: Comparison of Expected Revenue (larger is better) under mixed MNL model II.

|   δ |   LCB-MLE |   LCB-MLE-v2 |   PASTA |
|-----|-----------|--------------|---------|
| 0.1 |    0.9332 |       0.9338 |  0.9326 |
| 0.2 |    0.9328 |       0.933  |  0.9322 |
| 0.5 |    0.9313 |       0.9316 |  0.9308 |
| 0.7 |    0.9311 |       0.9314 |  0.9307 |

Setting 4: Misspecification under mixture of MNL II. This experiment uses the same parameter setup as in Setting 3, with the mixing probability fixed at µ = 0 . 5 . The distance between the two subgroups' parameters θ ⋆ is controlled by a parameter δ . As δ increases, model misspecification becomes more pronounced and overall performance declines. In this case, LCB-MLE-v2 slightly outperforms LCB-MLE and PASTA, indicating that stronger conservativeness may improve robustness under severe model mismatch.