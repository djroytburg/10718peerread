## Parameter-free Algorithms for the Stochastically Extended Adversarial Model

Shuche Wang

1 Adarsh Barik 2 Peng Zhao 3 , 4 Vincent Y. F. Tan 1 , 5 , 6

1 Institute of Operations Research and Analytics, National University of Singapore

2 Department of Computer Science and Engineering, Indian Institute of Technology Delhi

3 National Key Laboratory for Novel Software Technology, Nanjing University 4 School of Artificial Intelligence, Nanjing University

5 Department of Mathematics, National University of Singapore

6 Department of Electrical and Computer Engineering, National University of Singapore shuche.wang@u.nus.edu, adarshbarik1@iitd.ac.in, zhaop@lamda.nju.edu.cn, vtan@nus.edu.sg

## Abstract

We develop the first parameter-free algorithms for the Stochastically Extended Adversarial (SEA) model, a framework that bridges adversarial and stochastic online convex optimization. Existing approaches for the SEA model require prior knowledge of problem-specific parameters, such as the diameter of the domain D and the Lipschitz constant of the loss functions G , which limits their practical applicability. Addressing this, we develop parameter-free methods by leveraging the Optimistic Online Newton Step (OONS) algorithm to eliminate the need for these parameters. We first establish a comparator-adaptive algorithm for the scenario with unknown domain diameter but known Lipschitz constant, achieving an expected regret bound of ˜ O ( ∥ u ∥ 2 2 + ∥ u ∥ 2 ( √ σ 2 1: T + √ Σ 2 1: T ) ) , where u is the comparator vector and σ 2 1: T and Σ 2 1: T represent the cumulative stochastic variance and cumulative adversarial variation, respectively. We then extend this to the more general setting where both D and G are unknown, attaining the comparatorand Lipschitz-adaptive algorithm. Notably, the regret bound exhibits the same dependence on σ 2 1: T and Σ 2 1: T , demonstrating the efficacy of our proposed methods even when both parameters are unknown in the SEA model.

## 1 Introduction

We focus on online convex optimization (OCO) [1, 2, 3], a broad framework for sequential decisionmaking. In each round t ∈ [ T ] , a learner chooses a point x t from a convex set X ⊆ R d . The environment then discloses a convex function f t : X → R , after which the learner incurs a loss f t ( x t ) and updates their decision. The standard way to show the performance is via the regret , the total loss relative to a comparator u ∈ X , defined as R T ( u ) = ∑ T t =1 f t ( x t ) -∑ T t =1 f t ( u ) .

For convex problems, the regret can be bounded by O ( √ T ) [4], which is known to be minimax optimal [5]. OCO encompasses two primary frameworks: adversarial OCO [4, 6], which aims to minimize regret against arbitrarily chosen loss functions, and stochastic OCO (SCO) [6, 7], which minimizes excess risk under i.i.d. losses. While both frameworks are well-studied, realworld scenarios typically fall between these theoretical extremes of purely adversarial or stochastic settings. The Stochastically Extended Adversarial (SEA) model proposed in [8] bridges the gap between traditional adversarial and stochastic frameworks in OCO. This hybrid approach serves as an intermediate formulation that captures aspects of both adversarial OCO and SCO settings.

Table 1: Comparison of the regret bounds of existing results and our proposed algorithms.

| Algorithm                                      | Free of D   | Free of G   | Bound on Expected Regret E [ R T ( u )]                                      |
|------------------------------------------------|-------------|-------------|------------------------------------------------------------------------------|
| OFTLR,OMD (Sachs et al. [8], Chen et al. [12]) | ✗           | ✗           | O (√ σ 2 1: T + √ Σ 2 1: T )                                                 |
| OONS (Theorem 3.2)                             | ✗           | ✗           | ˜ O (√ σ 2 1: T + √ Σ 2 1: T )                                               |
| CA-OONS (Theorem 4.1)                          | ✓           | ✗           | ˜ O ( ∥ u ∥ 2 2 + ∥ u ∥ 2 ( √ σ 2 1: T + √ Σ 2 1: T ) )                      |
| CLA-OONS (Theorem 4.5)                         | ✓           | ✓           | ˜ O ( ∥ u ∥ 2 2 ( √ σ 2 1: T + √ Σ 2 1: T )+ ∥ u ∥ 4 2 + √ σ 1: T + G 1: T ) |

Optimal performance in OCO, SCO, and SEA models typically relies on careful step-size tuning, which requires prior knowledge of problem parameters such as the diameter of the decision set and Lipschitz constants. However, these parameters are often unknown in practice, motivating the development of parameter-free algorithms that achieve comparable regret without requiring such oracle information. Specifically, parameter-free algorithms include comparator-adaptive algorithms (unknown diameter D ) and Lipschitz-adaptive algorithms (unknown Lipschitz constant G ). A related challenge arises when the decision set X is unbounded, allowing adversaries to induce arbitrarily large losses for linear functions. Traditional methods often circumvent this by assuming bounded domains, where sup x,y ∈X ∥ x -y ∥ 2 ≤ D . Consequently, developing OCO algorithms that remain effective under both unknown parameters and unbounded domains is significantly more challenging than in classical settings [9, 10, 11].

To address these challenges, we propose 'parameter-free' algorithms for the SEA model, accommodating potentially unbounded decision sets. Using the Optimistic Online Newton Step as our base algorithm, we systematically relax assumptions: first tackling the case of an unknown domain diameter D (potentially infinite) with a known Lipschitz constant G , and then extending to the more complex scenario where both D and G are unknown. In the SEA model, at each time step t , the learner selects a distribution D t over functions and incurs a loss f t ( x t ) , where f t is sampled from D t . The expected gradient is denoted as ∇ F t ( x ) = E f t ∼D t [ ∇ f t ( x )] .

Main Contributions. Our main results and contributions are summarized as follows.

- (1) We begin by introducing the Optimistic Online Newton Step (OONS) as our foundational algorithm. The OONS algorithm is inspired by [10]; however, we incorporate an adaptive step-size η t rather than a fixed step-size throughout the learning process. When the parameters D and G are known, we demonstrate that OONS achieves an expected regret bound of ˜ O ( √ σ 2 1: T + √ Σ 2 1: T ) , matching the state-of-the-art results in terms of dependence on the cumulative stochastic variance σ 2 1: T and the cumulative adversarial variation Σ 2 1: T [8, 12]. This establishes a solid foundation for our subsequent extensions to parameter-free algorithms.
- (2) We introduce the first parameter-free (comparator-adaptive) algorithm for the SEA model that remains effective when the domain diameter D is unknown, provided the Lipschitz constant G is known. This is achieved through a meta-framework wherein each base learner operates within a distinct bounded domain, complemented by the Multi-scale Multiplicative-Weight with Correction (MsMwC) algorithm [10] for the meta-algorithm's weight updates. This construction yields an expected regret bound of ˜ O ( ∥ u ∥ 2 2 + ∥ u ∥ 2 ( √ σ 2 1: T + √ Σ 2 1: T )) , where the bound scales with the ℓ 2 -norm of the comparator u without requiring prior knowledge of the domain diameter D .
- (3) We further consider a setting in which both the domain diameter D and the Lipschitz constant G are unknown. By devising appropriate update rules for the estimation of the domain diameter, we establish an expected regret bound of ˜ O ( ∥ u ∥ 2 2 ( √ σ 2 1: T + √ Σ 2 1: T ) + ∥ u ∥ 4 2 + √ σ 1: T + G 1: T ) where σ 1: T captures the deviation of the stochastic gradients (excluding squared norms), and G 1: T denotes the sum of the maximum expected gradients over the sequence.

Asummary of our results and the best existing results are included in Table 1. Due to space limitations, we hide the Lipschitz constant G in the ˜ O ( · ) -notation in the regret bound of CLA-OONS algorithm.

## 1.1 Related Work

The SEA model [8] is motivated by foundational insights from the gradual-variation online learning . The study of gradual variation can be traced back to the works of [13] and [14], and it has gained significant traction in recent years [15, 16, 17, 18, 19]. Notably, the SEA model has emerged as a practical application of the gradual variation principle [16, 18, 19]. Furthermore, this model serves as a bridge between adversarial OCO and SCO. This intermediate framework is comprehensively understood in the context of expert prediction [20, 21] and the bandit setting [22, 23].

Parameter-free online learning has emerged as a fundamental advancement in machine learning, offering solutions to the critical challenge of parameter tuning in practice. In the baseline scenario, when both the diameter parameter D and the gradient bound G are known, algorithms leveraging Follow the Regularized Leader or Mirror Descent principles achieve the minimax optimal regret bound of R T ( u ) ≤ O ( GD √ T ) [2]. The field has subsequently progressed to address more practical scenarios where complete parameter knowledge is unavailable. Notably, in the Lipschitz-adaptive setting, it is still possible to attain the same optimal regret bound, differing only by constant factors [24, 25]. Xie et al. [18] extended these principles to gradient-variation online learning.

In the comparator-adaptive setting, the online learning problem becomes substantially more challenging due to the unknown comparator's magnitude, which could cause the algorithm's predictions to significantly deviate from the optimal solution, leading to a large regret. For this challenging scenario, a key result has been established as R T ( u ) ≤ ˜ O ( ∥ u ∥ 2 G √ T ) [26, 24, 9, 27]. For scenarios where both parameters D and G are unknown, significant progress has been made recently. Cutkosky [25] developed an algorithm with R T ( u ) ≤ ˜ O ( G ∥ u ∥ 3 2 + ∥ u ∥ 2 G √ T ) , while Mhammedi &amp; Koolen [28] achieve R T ( u ) ≤ ˜ O ( G ∥ u ∥ 3 2 + G √ max t ≤ T ( ∑ t s =1 ∥ g s ∥ 2 / max s ≤ t ∥ g s ∥ 2 )) . An alternative approach by [29] presented the regret bound R T ( u ) ≤ ˜ O ( ∥ u ∥ 2 2 G √ T ) . More recent advances including [11] achieve the regret R T ( u ) ≤ ˜ O ( G ∥ u ∥ 2 √ T + L ∥ u ∥ 2 2 √ T ) under the condition that subgradients satisfy ∥ g t ∥ 2 ≤ G + L ∥ x t ∥ 2 . Cutkosky &amp; Mhammedi [30] further improve it to ˜ O ( G ∥ u ∥ 2 √ T + ∥ u ∥ 2 2 + G 2 ) .

Besides parameter-free algorithms for OCO, [31] and [32] studied the parameter-free stochastic gradient descent (SGD) algorithms. Khaled et al. [33] introduced the concept of 'tuning-free' algorithms, which achieve performance comparable to optimally-tuned SGD within polylogarithmic factors, requiring only approximate estimates of the relevant problem parameters.

Although this series of works on parameter-free algorithms in OCO provides valuable insights, these approaches cannot be directly applied to attain the optimal regret bounds for the SEA model without prior knowledge of parameters. This limitation stems from the fact that the desired bounds for the SEA model should be expressed in terms of the variance-like quantities σ 2 1: T and Σ 2 1: T , rather than the time horizon T . While Sachs et al. [8] have attempted to address this issue by proposing an algorithm that adapts to an unknown strong convexity parameter, their step-size search range still depends on both D and G , thereby restricting its fully parameter-free adaptivity.

## 2 Problem Setup and Preliminaries

In this section, we formulate the problem setup of the Stochastically Extended Adversarial (SEA) model, present the existing results, and discuss the key challenges.

## 2.1 Problem Setup of the SEA Model

In iteration t ∈ [ T ] , the learner selects a decision x t from a convex feasible domain X ⊆ R d , and nature chooses a distribution D t from a set of distributions over functions. Then, the learner suffers a loss f t ( x t ) , where f t is a random function sampled from the distribution D t . The distributions are allowed to vary over time, and by choosing them appropriately, the SEA model reduces to the adversarial OCO, SCO, or other intermediate settings. Additionally, for each t ∈ [ T ] , the (conditional) expected function is defined as F t ( x ) = E f t ∼D t [ f t ( x )] and the expected gradient is defined as ∇ F t ( x ) = E f t ∼D t [ ∇ f t ( x )] . We define G t := sup x ∈X ∥∇ F t ( x ) ∥ 2 to be the largest norm of the expected gradient, and use the shorthand G 1: T to denote the sum ∑ T t =1 G t .

Due to the randomness in the online decision-making process, our goal in the SEA model is to bound the expected regret with respect to the randomness in the loss functions f t drawn from the distribution D t against any fixed comparator u ∈ X , defined as E [ R T ( u )] ≜ E [ ∑ T t =1 f t ( x t ) -∑ T t =1 f t ( u )] . To capture the characteristics of the SEA model, we introduce the following quantities. For each t ∈ [ T ] , define the (conditional) variance of the gradients and cumulative stochastic variance respectively as

<!-- formula-not-decoded -->

which reflect the stochasticity of the online process. Additionally, we introduce the concepts of stochastic gradient deviation and cumulative gradient deviation to characterize the stochastic variation of gradients, without the squared norm. The stochastic gradient deviation is defined as σ t = sup x ∈X E f t ∼D t [ ∥∇ f t ( x ) -∇ F t ( x ) ∥ 2 ] , and the cumulative gradient deviation is defined as σ 1: T = E [ ∑ T t =1 σ t ] . The cumulative adversarial variation is defined as

<!-- formula-not-decoded -->

where ∇ F 0 ( x ) = 0 , reflecting the adversarial difficulty. This work aims to provide expected regret bounds that depend on problem-dependent quantities such as σ 2 1: T , Σ 2 1: T , and G 1: T instead of T .

Below, we present several assumptions. Note that our results do not rely on all of these assumptions; rather, specific assumptions are required for each result, which will be explicitly stated in the theorem.

Assumption 2.1 (Boundedness of gradient norms) . The gradient norms of all loss functions are bounded by G , i.e., max t ∈ [ T ] max x ∈X ∥∇ f t ( x ) ∥ 2 ≤ G .

Assumption 2.2 (Boundedness of domain) . The diameter of the convex set X (the feasible domain) is bounded by D i.e., sup x,y ∈X ∥ x -y ∥ 2 ≤ D .

Assumption 2.3 (Smoothness) . For all t ∈ [ T ] , the expected function F t is L -smooth over X , i.e., ∥∇ F t ( x ) -∇ F t ( y ) ∥ 2 ≤ L ∥ x -y ∥ 2 for all x, y ∈ X .

Assumption 2.4 (Convexity) . For all t ∈ [ T ] , the expected function F t is convex on X .

Notations. Given a positive definite matrix A , the norm induced by A is ∥ x ∥ A = √ x ⊤ Ax . ∆ N denotes the ( N -1) -dimensional simplex. Let ψ : X → R be a continuously differentiable and strictly convex function, the associated Bregman divergence is defined as D ψ ( x, y ) := ψ ( x ) -ψ ( y ) -⟨∇ ψ ( y ) , x -y ⟩ . The notation O ( · ) hides constants and ˜ O ( · ) additionally hides polylog factors.

## 2.2 Existing Results for the SEA Model

Bounded domain and gradient norm. Sachs et al. [8] established a regret bound for the SEA model using both Optimistic Follow-The-Regularized-Leader (OFTRL) and Optimistic Mirror Descent (OMD), given by E [ R T ( u )] = O ( √ σ 2 1: T + √ Σ 2 1: T ) , achieved by setting the step-size as η t = D 2 ∑ t -1 s =1 min { ηs 2 ∥ g s -m s ∥ 2 2 ,D ∥ g s -m s ∥ 2 } , where g t = ∇ f t ( x t ) and m t = g t -1 . Similarly, Chen et al. [12] derived the same bound by Optimistic Online Mirror Descent (OMD) with the step-size η t = D √ δ +4 G 2 + ¯ V t -1 , where ¯ V t -1 = ∑ t -1 s =1 ∥ g s -m s ∥ 2 2 and δ &gt; 0 .

In all of the above settings, the optimal step-size η t is dependent on the parameters D (the diameter of decision set X ) and G (Lipschitz constant), so there has been a natural motivation to develop algorithms that achieve similar regret bounds without knowing such parameters a priori . We term such algorithms as 'parameter-free' algorithms for the SEA model.

Parameter-free algorithm for the SEA model. Theorem 5 in [27] demonstrates that the parameterfree mirror descent algorithm can be extended to enjoy a gradient-variation regret of R T ( u ) ≤

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Algorithm 1 Optimistic Online Newton Step (OONS)

Input:

learning rate η t &gt; 0 , x ′ 1 = 0 .

- 1: for t = 1 , . . . , T do
- 2: Receive optimistic prediction m t and range hint z t .
- 4: Receive g t = ∇ f t ( x t ) and construct ∇ t = g t +32 η t ⟨ x t , g t -m t ⟩ ( g t -m t ) .
- 3: Update x t = arg min x ∈X {⟨ x, m t ⟩ + D ψ t ( x, x ′ t ) } where ψ t ( x ) = 1 2 ∥ x ∥ 2 A t and A t = 4 z 2 1 I + ∑ t -1 s =1 η s ( ∇ s -m s )( ∇ s -m s ) ⊤ +4 η t z 2 t I.
- 5: Update x ′ t +1 = arg min x ∈X {⟨ x, ∇ t ⟩ + D ψ t ( x, x ′ t ) } .
- 6: end for

Akin to σ 2 1: T , ˜ σ 2 1: T defined in [12] also captures the stochastic nature of the SEA model. Furthermore, in the worst case, the bound in (2) reduces to ˜ O ( ∥ u ∥ 2 √ T ) , matching the best available problemindependent bound. The outer expectation in the definition of ˜ σ 2 1: T accounts for the randomness in the choice of the distribution D t at each step. Refer to Appendix B.1 for a self-contained proof of (2).

Key Challenge. However, we emphasize that our goal is to obtain regret bounds scaling with σ 2 1: T , as defined in (1). As pointed out in previous work on the SEA model [12, Remark 9], σ 2 1: T is more favorable than ˜ σ 2 1: T . First, from a mathematical perspective, the latter is generally larger due to the convexity of the supremum operator. The difference between σ 2 1: T and ˜ σ 2 1: T can, in fact, be arbitrarily large. The detailed comparison is provided in Appendix A. Second, from an algorithmic perspective, an algorithm with a regret bound involving ˜ σ 2 1: T typically involves an implicit update, which operates on the original function and is significantly more costly than standard first-order methods (see Remark 10 in [12]). Third, achieving regret bounds scaling with σ 2 1: T typically requires leveraging the Regret Bounded by Variation in Utilities (RVU) property [34], which captures the regret to be bounded not only by the gradient variations but also an additional negative stability term. Formally, an algorithm satisfies the RVU property if its regret upper bound is in the form of ∑ T t =1 ⟨ x ∗ -x t , u t ⟩ ≤ α + β ∑ T t =1 ∥ u t -u t -1 ∥ 2 -γ ∑ T t =1 ∥ x t -x t -1 ∥ 2 , for some constants α, β, γ &gt; 0 . This structure enables finer control over the regret by explicitly analyzing trajectory stability, establishing profound connections to game theory [34, 35] and accelerations in smooth optimization [36]. Consequently, the key challenge lies in how to achieve this preferred σ 2 1: T -scaling without knowledge of D and G for unbounded domains.

## 3 Optimistic Online Newton Step (ONS) for the SEA Model

In this section, different from the Optimistic follow-the-regularized-leader (OFTRL) [8] and Optimistic mirror descent (OMD) [12], we first introduce the Optimistic Online Newton Step (OONS) algorithm as the base algorithm for the 'parameter-free' algorithms to be introduced later. This algorithm is summarized in Algorithm 1.

The ONS algorithm [6] is an iterative algorithm that adaptively updates a second-order (Hessianbased) model of the loss, allowing more efficient gradient-based updates and improved regret bounds. OONS also maintains two sequences { x t } T t =1 and { x ′ t } T t =1 like OMD and OFTRL, which is achieved by introducing the optimistic prediction m t . Chen et al. [10] also considered combining their Multiscale Multiplicative-weight with Correction (MsMwC) algorithm with this variant of the ONS algorithm. However, the step-size η is fixed in their algorithm and the MsMwC algorithm is applied to learn the optimal η ⋆ . Different from it, in OONS, we consider adaptive step-sizes η t .

Theorem 3.1. Suppose that ∥ g t -m t ∥ 2 ≤ z t , z t is non-decreasing in t , 64 η t Dz T ≤ 1 for all t ∈ [ T ] , and η t is non-increasing in t . Then, OONS guarantees that

<!-- formula-not-decoded -->

where r is the rank of ∑ T t =1 ( g t -m t )( g t -m t ) ⊤ .

Next, we verify that OONS also works for the case with known parameters D,G , and we can also obtain a similar regret bound as [8] and [12]. The regret bound of OONS for the SEA model with

Algorithm 2 Comparator-adaptive algorithm for the SEA model (CA-OONS)

Input:

Lipschitz constant G .

- 1: for t = 1 , . . . , T do
- 2: Create N = ⌈ log T ⌉ base-learners. Each base-learner j ∈ [ N ] runs OONS with stepsize η j t .
- 3: Each base-learner j provides x j t .
- 4: Run Algorithm 3 to obtain w t ∈ ∆ N .
- 5: The final decision is x t = ∑ N j =1 w t,j x j t .
- 6: end for

known parameters D and G is presented below. We specify the adaptive step-size for all t ∈ [ T ] as

<!-- formula-not-decoded -->

Since G is known, we have ∥ g t -m t ∥ 2 ≤ z t = 2 G, ∀ t ∈ [ T ] and η t is defined in terms of z T here.

Theorem 3.2. Under Assumptions 2.1, 2.2, 2.3, and 2.4, OONS with step-size η t given in (4) , m t = ∇ f t -1 ( x t -1 ) and z t = 2 G for all t ∈ [ T ] ensures E [ R T ( u )] = ˜ O ( √ σ 2 1: T + √ Σ 2 1: T ) .

Remark 3.3 . Theorem 3.2 achieves the same (up to logarithmic terms) dependence on σ 2 1: T and Σ 2 1: T as in [8] and [12]. The primary reason to use OONS as the base algorithm instead of OMD [12] is that the final regret bound for OMD typically depends on D √ ∑ T t =1 ∥ g t -m t ∥ 2 2 . In scenarios when D is unknown or potentially infinite, like in Section 4.2, this might lead to O ( T ) regret bounds. By contrast, OONS leverages adaptive second-order information, which helps remove (or substantially reduce) explicit dependence on D . In (3), the only term relevant to D is D ( z T -z 1 ) , which solely depends on the starting and ending points, z 1 and z T .

## 4 Parameter-free Algorithms for the SEA Model

In this section, we develop parameter-free algorithms for the SEA model, building on OONS which we use as the base algorithm. Moreover, we allow the decision set X to be potentially unbounded throughout this section, i.e. D = ∞ in Assumption 2.2. In Section 4.1, we present a comparatoradaptive algorithm for the SEA model for unknown D but known G , and then, we develop the comparator- and Lipschitz-adaptive algorithm where both D and G are unknown in Section 4.2.

## 4.1 Comparator-adaptive algorithm

We now propose the Comparator-Adaptive Optimistic Online Newton Step (CA-OONS) algorithm for the unknown D (potentially infinite) but known G case by using a meta-base algorithm framework. Recent works in [10] and [11] addressed the challenges associated with unbounded domains by developing a base-learner framework. Building on this philosophy, we propose CA-OONS(Algorithm 2) where we adopt the MsMwC-Master algorithm [10] as the meta algorithm.

The algorithm uses N base-learners. For any base-learner i ∈ [ N ] , the regret can be decomposed as

<!-- formula-not-decoded -->

Let A i denote the base algorithm for the i -th base-learner. We denote R A i T ( u ) = ∑ T t =1 f t ( x i t ) -∑ T t =1 f t ( u ) as the base regret by taking A i as the base algorithm. Moreover, the final decision x t is a weighted-average of all the base-learners' decisions: x t = ∑ N j =1 w t,j x j t with w t ∈ ∆ N . As such,

<!-- formula-not-decoded -->

## Algorithm 3 Meta Algorithm

Input: Additional expert set S defined in (7). Initialization: ′ ′ 2

p 1 ∈ ∆ S such that p 1 ,k ∝ β k for all k ∈ S .

- 1: for t = 1 , . . . , T do
- 2: Construct h t ∈ R N with h j t = ⟨∇ f t -1 ( x j t -1 ) , x j t ⟩ .
- 3: Each expert k ∈ S runs MsMwC with step-size β k t,j and plays w k t ∈ ∆ N .
- 4: Receive w k t for each k ∈ S and compute H k t = 〈 w k t , h t 〉 .
- 5: Compute p t = arg min p ∈ ∆ S ⟨ p, H t ⟩ + D ϕ ( p, p ′ t ) .
- 6: Play w t = ∑ k ∈S p t,k w k t ∈ ∆ N .
- 7: Receive ℓ t ∈ R N . Define L k t = 〈 w k t , ℓ t 〉 and b k t = 32 β k ( L k t -H k t ) .
- 9: end for
- 8: Compute p ′ t +1 = arg min p ∈ ∆ S ⟨ p, L t + b t ⟩ + D ϕ ( p, p ′ t ) .

Table 2: Three-layer hierarchy of CA-OONS

| Layer       | Algorithm   | Loss            | Optimism                | Decision              | Output                |
|-------------|-------------|-----------------|-------------------------|-----------------------|-----------------------|
| Top Meta    | MsMwC       | ( L k t ) k ∈S  | ( H k t ) k ∈S          | p t ∈ ∆ S             | w t = ∑ k p t,k w k t |
| Middle Meta | MsMwC       | ℓ t ∈ R N       | h t ∈ R N               | ( w k t ) k ∈S ∈ ∆ N  | w k t                 |
| Base        | OONS        | ∇ f t ( x j t ) | ∇ f t - 1 ( x j t - 1 ) | ( x j t ) j ∈ N ∈ X j | x j t                 |

where ℓ t ∈ R N with ℓ j t = ⟨∇ f t ( x j t ) , x j t ⟩ and w i ⋆ is a vector in ∆ N whose j -th component is ( w i ⋆ ) j = 1 if j = i and 0 otherwise. Refer to Appendix D.2 for the proof of (5).

We first consider the base algorithm. Specifically, for each base-learner j ∈ [ N ] , we impose a constraint that it operates within X j = { x : ∥ x ∥ 2 ≤ D j and x ∈ X} , where D j = 2 j . Then, we define g j t = ∇ f t ( x j t ) and m j t = ∇ f t -1 ( x j t -1 ) . Each base-learner j ∈ [ N ] runs OONS with step-size

<!-- formula-not-decoded -->

which depends on D j instead of D in OONS. Hence, each base-learner j can update x j t via OONS with step-size η j t . Since the final decision is x t = ∑ N j =1 w t,j x j t , we need to adopt a meta-algorithm to learn the weight parameter w t ∈ ∆ N .

As mentioned above, we introduce a constraint that each base-learner j ∈ [ N ] operates within a D j -bounded domain. We can consider this as a 'multi-scale' base-learner problem [37, 9, 10] where each base-learner j has a different loss range such that | ℓ j t | ≤ GD j since ℓ j t = ⟨∇ f t ( x j t ) , x j t ⟩ . We choose the Multi-scale Multiplicative-weight with Correction (MsMwC)-Master algorithm (Algorithm 2 in [10]) as the meta-algorithm to learn w t , which is implemented based on the MsMwC [10]. Details of MsMwC are presented in Appendix D.1.Specifically, we define a new expert set

<!-- formula-not-decoded -->

For all k ∈ S , the step-size of the MsMwC-Master algorithm is set to β k = 1 32 · 2 k . Each expert k ∈ S runs the MsMwC algorithm with w ′ 1 being uniform over Z ( k ) , where Z k = { j ∈ [ N ] : GD j ≤ 2 k -2 } . Moreover, each base MsMwC algorithm only works in the subset Z ( k ) , i.e., w t ∈ ∆ N with w t,j = 0 for all j / ∈ Z ( k ) . We can view CA-OONS (Algorithm 2) as a three-layer structure, where the meta-algorithm (Algorithm 3) itself consists of two layers, which we refer to as meta top and meta middle . Also, the base layer is OONS algorithm. For clarity, we summarize the notations of the three-layer hierarchy for CA-OONS in Table 2.

In the following, we provide the expected regret guarantee for CA-OONS.

Theorem 4.1. Let D be unknown (potentially infinite). Under Assumptions 2.1, 2.3 and 2.4, CAOONS provides the following regret

<!-- formula-not-decoded -->

This regret guarantee is referred to as 'comparator-adaptive' because it depends directly on the norm of the comparator, ∥ u ∥ 2 , rather than explicitly relying on the diameter of the decision set, D . Notably, when considering the constrained decision set with a diameter D , our regret bound immediately recovers the result E [ R T ( u )] = ˜ O ( D 2 + D ( √ σ 2 1: T + √ Σ 2 1: T )) established in [8, 12].

One limitation of our regret bound (8) is that when particularizing to adversarial OCO, it achieves only an ˜ O ( ∥ u ∥ 2 2 + ∥ u ∥ 2 √ T ) worst-case regret bound, which falls short of the best-known ˜ O ( ∥ u ∥ 2 √ T ) regret bound [24, 27, 28]. However, in the following two remarks, we will justify the ∥ u ∥ 2 -dependence in the gradient-variation regret and emphasize the fundamental challenge of achieving adaptivity from the gradient-variation bound (for smooth functions) to the worst-case bound (for the non-smooth case) when the decision set of online learning is unconstrained .

Remark 4.2 (Dependency on ∥ u ∥ 2 2 ) . Recent studies have established the connection between gradientvariation online learning and accelerated offline optimization through advanced online-to-batch conversions [36, 38]. Specifically, let d 0 = ∥ x 0 -x ∗ ∥ 2 denote the distance of an initial point x 0 to the optimum x ∗ . For an L -smooth function, gradient-variation online algorithms using first-order information correspond to an accelerated convergence rate of O ( Ld 2 0 /T 2 ) via the stabilized online-tobatch conversion [39]. For a G -Lipschitz function, the problem-independent regret bounds translate to an O ( Gd 0 / √ T ) rate through the standard conversion [1]. In this context, we hypothesize that the ∥ u ∥ 2 2 term may be unavoidable in gradient-variation regret for unconstrained online learning, paralleling how the d 2 0 term also appears in the accelerated rate of unconstrained offline optimization. Remark 4.3 (Adaptivity between gradient-variation bound and worst-case bound) . We argue that achieving adaptivity between the gradient-variation bound and the problem-independent worst-case bound in unconstrained online learning may be as challenging as achieving universality in offline optimization over unconstrained domains, where the method must adapt to both smooth and Lipschitz functions. To the best of our knowledge, the best-known universal method for offline unconstrained optimization is by [40], which combines UNIXGRAD [39] with the DOG step size [31]. Nonetheless, this method is complex and still relies on a predefined range of parameters, highlighting both the difficulty of the problem and the fact that it remains only partially solved. Consequently, designing a single unconstrained online learning algorithm that adaptively bridges the gradient-variation regret bound for smooth functions and the worst-case bound for Lipschitz functions is non-trivial, which could provide new insights into universal offline optimization methods. We leave this for future work. Remark 4.4 (On dependence on the time horizon) . The use of T in CA-OONS (via N = ⌈ log T ⌉ experts) is only for convenience and not fundamental. An anytime variant is obtained by the standard doubling trick: restart the algorithm at epochs of lengths 1 , 2 , 4 , . . . , and in epoch k set N k = ⌈ log 2 k -1 ⌉ . This introduces at most an additional logarithmic factor already hidden in ˜ O ( · ) [41, Section 4.3]. A restart-free alternative is a sleeping (awakening) expert grid of learning rates as in the multi-rate construction of [42], which activates only those experts whose scale becomes relevant.

## 4.2 Comparator- and Lipschitz-adaptive Algorithm

The algorithm in the previous subsection requires prior knowledge of the Lipschitz constant G . Due to practical limitations such knowledge may not be available in real applications. A comparator- and Lipschitz-adaptive algorithm would instead adapt to an unknown Lipschitz constant G .

A simple approach to handling the unknown gradient norms, proposed by [25], relies on a gradientclipping reduction. The key idea is to design an algorithm A that achieves appropriate regret when given prescient 'hints' h t ≥ ∥ g t ∥ 2 at the start of round t . Since such hints are impractical (as g t is not observed beforehand), we instead approximate them using a clipped gradient, inspired by [25]. We start with an initial guess B 0 on the range of max t ∥ g t -m t ∥ 2 , where g t = ∇ f t ( x t ) . We define B t = max 0 ≤ s ≤ t ∥ g s -m s ∥ 2 as the predicted error range up to iteration t . The truncated gradient is then defined as ˜ g t = m t + B t -1 B t ( g t -m t ) . The truncated gradient satisfies ∥ ˜ g t -m t ∥ 2 ≤ B t -1 , allowing the learner to assume that the range of predicted error in iteration t is known at the start.

Next, we initialize the decision set diameter guess as D 1 = 1 . For each iteration t ∈ [ T ] , we first play x t and receive g t = ∇ f t ( x t ) . To update D t , we consider the condition D t &lt;

Algorithm 4 Comparator and Lipschitz-Adaptive (or CLA-OONS) for the SEA model

Input:

Initial scale B 0 .

Initialize:

D 1 = 1 .

- 1: for t = 1 , . . . , T do
- 2: Run OONS in D t -bounded domain and obtain x t . Play x t and receive g t = ∇ f t ( x t ) .
- 3: Construct ˜ g t = m t + B t -1 B t ( g t -m t ) , where B t = max 0 ≤ s ≤ t ∥ g s -m s ∥ 2 .
- 4: if D t &lt; √ ∑ t s =1 ∥ g s ∥ 2 max { 1 , max k ≤ s ∥ g k ∥ 2 } then
- 5: Update D t +1 = 2 √ ∑ t s =1 ∥ g s ∥ 2 max { 1 , max k ≤ s ∥ g k ∥ 2 } and reset A t +1 as (9) and x ′ t +1 = 0 .
- 6: end if
- 7: Feed ˜ g t to OONS running in the D t +1 -bounded domain and get x t +1 , where z t +1 = B t . 8: end for

√ ∑ t s =1 ∥ g s ∥ 2 max { 1 , max k ≤ s ∥ g k ∥ 2 } . If this condition holds, we update D t +1 using the doubling trick. This ensures that we need to update D t a maximum of M = O (log T ) times. We divide the total T iterations into disjoint subsets of M iterations. If the 'doubling' occurs at the t -th iteration, we update t a ← t and reset x ′ t +1 = 0 and the matrix A t +1 in OONS as follows

<!-- formula-not-decoded -->

Then, we feed ˜ g t to OONS running in the D t +1 -bounded domain X t +1 = { x : ∥ x ∥ 2 ≤ D t +1 ∧ x ∈ X} and obtain x t +1 . We summarize the ideas in Algorithm 4 and term it as Comparator and Lipschitz-Adaptive Optimistic Online Newton Step (or CLA-OONS) algorithm.

Theorem 4.5. Let both D (potentially infinite) and G be unknown. Under Assumptions 2.1 (but G is unknown), 2.3 and 2.4, the proposed FPF-OONS algorithm satisfies

<!-- formula-not-decoded -->

where σ 1: T captures the stochastic gradient deviation (without the squared norm) and G 1: T denotes the sum of maximum expected gradients.

Remark 4.6 (Discussion and challenges) . In Theorem 4.5, the regret includes ∥ u ∥ 2 2 ( √ σ 2 1: T + √ Σ 2 1: T ) . Ideally, we aim to achieve a dependence of ˜ O ( ∥ u ∥ 2 ) , consistent with [25] and [43]. However, achieving this within the SEA framework presents significant challenges. As mentioned in Section 2, obtaining regret bounds that scale with σ 2 1: T in the SEA framework is difficult. These challenges are compounded in the comparator and Lipschitz-adaptive setting. Below, we outline some of the main technical challenges associated with achieving the desired bound of ˜ O ( ∥ u ∥ 2 ( √ σ 2 1: T + √ Σ 2 1: T )) .

As stated in Section 2, the methods such as those proposed by [25, 27, 11] cannot be applied to obtain the expected regret bound in terms of σ 2 1: T . The work [44] also looks promising; however, it remains unclear to us whether the approach proposed in the paper can be directly extended to the SEA framework. Their results, presented in Theorems 3 and 5 of [44], are not Lipschitz-adaptive. Specifically, they operate under the assumptions that ∥ g t ∥ 2 ≤ 1 and ∥ m t ∥ 2 ≤ 1 .

One could also use a large number of base-learners to achieve a regret of ˜ O ( ∥ u ∥ 2 √ r ∑ t ∥ g t -m t ∥ 2 2 + ∥ u ∥ 3 2 ) , similar to [10]. However, this approach presents a subtle yet significant challenge. Following a similar analysis [10], we get the following decomposition: ∑ t ⟨ g t , x t -u ⟩ = ∑ t ⟨ g t , x t -x k ∗ t ⟩ + ∑ t ⟨ g t , x k ∗ t -u ⟩ . By leveraging Theorem 23 in [10], we can write ∑ t ⟨ g t , x k ∗ t -u ⟩ as ∑ t ⟨ g t , x k ∗ t -u ⟩ ≤ ˜ O ( ∥ u ∥ √ r ∑ t ∥ g t -m t ∥ 2 2 -∑ t ∥ x k ∗ t -x k ∗ t -1 ∥ 2 2 ) . Observe that expressing the first term, √ r ∑ t ∥ g t -m t ∥ 2 2 , in terms of σ 1: T and Σ 1: T introduces additional terms involving ∑ t ∥ x t -x t -1 ∥ 2 2 (See Lemma 2 in [12]). The only way to address this term is through the negative term -∑ t ∥ x k ∗ t -x k ∗ t -1 ∥ 2 2 , which becomes tricky. This challenge is reminiscent of the problem encountered by [17]. Their solution, as outlined in Equation (17) of [17], relies on the bounded domain assumption, which is not applicable in our setting. Consequently, this limitation prevents us from improving the ∥ u ∥ 2 2 dependence in the leading term by using additional base-learners.

√

The bound in Theorem 4.5 also includes an additive term involving G 1: T , which reflects the sum of the maximum expected gradients' norms over T rounds, and arises because the domain is potentially unbounded. Note that this term does not have a ∥ u ∥ dependence. Hence, the comparator having a large norm in an unbounded setting (potentially dependent on T ) does not affect its growth. In the worst case, √ G 1: T = O ( √ T ) , which underscores that this additive term does not have a significant adverse effect on the regret as √ σ 2 1: T and √ Σ 2 1: T also scale as √ T [8].

## 5 Conclusions and Future Work

This paper presents novel parameter-free algorithms for the SEA model, addressing critical challenges in online optimization where traditional approaches require prior knowledge of parameters such as the diameter of the domain D and the Lipschitz constant of the loss functions G . Our proposed algorithms: CA-OONS and CLA-OONS are designed to operate effectively even when D and G are unknown, demonstrating their adaptability and practicality.

There are several avenues for future research. First, we would like to improve the regret's dependence on ∥ u ∥ 2 when both D and G are unknown. Another promising direction is to reduce the number of gradient queries in CA-OONS from O (log T ) to O (1) , thus enhancing its efficiency. An intriguing question in the comparator-adaptive setting is whether it is possible to design a single, simple online algorithm that simultaneously achieves two types of bounds: ˜ O ( ∥ u ∥ 2 2 + ∥ u ∥ 2 ( √ σ 2 1: T + √ Σ 2 1: T )) and ˜ O ( ∥ u ∥ 2 G √ T ) . As discussed in Remark 4.3, it remains an open challenge to construct an adaptive parameter-free online algorithm that can interpolate between these bounds. An additional open direction is to move beyond expected regret and derive high-probability (or variance-sensitive) regret guarantees for the SEA model in the parameter-free setting. Current SEA analyses, including [8, 12], bound only E [ R T ( u )] ; developing concentration results that retain the fine σ 2 1: T and Σ 2 1: T dependence without incurring suboptimal logarithmic inflation appears non-trivial and is left for future work.

## Acknowledgments and Disclosure of Funding

This research is funded by the Singapore Ministry of Education Academic Research Fund Tier 2 under grant number A-8000423-00-00 and three Singapore Ministry of Education Academic Research Funds Tier 1 under grant numbers A-8000189-01-00, A-8000980-00-00 and A-8002934-00-00.

## References

- [1] Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games . Cambridge university press, 2006.
- [2] Elad Hazan et al. Introduction to online convex optimization. Foundations and Trends® in Optimization , 2(3-4):157-325, 2016.
- [3] Francesco Orabona. A modern introduction to online learning. arXiv preprint arXiv:1912.13213 , 2019.
- [4] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th international conference on machine learning , pages 928-936, 2003.
- [5] Jacob Abernethy, Peter L Bartlett, Alexander Rakhlin, and Ambuj Tewari. Optimal strategies and minimax lower bounds for online convex games. In Proceedings of the 21st annual conference on learning theory , pages 414-424, 2008.
- [6] Elad Hazan, Amit Agarwal, and Satyen Kale. Logarithmic regret algorithms for online convex optimization. Machine Learning , 69(2):169-192, 2007.
- [7] Hao Yu, Michael Neely, and Xiaohan Wei. Online convex optimization with stochastic constraints. Advances in Neural Information Processing Systems , 30, 2017.
- [8] Sarah Sachs, Hedi Hadiji, Tim van Erven, and Cristobal Guzman. Accelerated rates between stochastic and adversarial online convex optimization. arXiv preprint arXiv:2303.03272 , 2023.
- [9] Ashok Cutkosky and Francesco Orabona. Black-box reductions for parameter-free online learning in banach spaces. In Conference On Learning Theory , pages 1493-1529. PMLR, 2018.
- [10] Liyu Chen, Haipeng Luo, and Chen-Yu Wei. Impossible tuning made possible: A new expert algorithm and its applications. In Conference on Learning Theory , pages 1216-1259. PMLR, 2021.
- [11] Andrew Jacobsen and Ashok Cutkosky. Unconstrained online learning with unbounded losses. In International Conference on Machine Learning , pages 14590-14630. PMLR, 2023.
- [12] Sijia Chen, Yu-Jie Zhang, Wei-Wei Tu, Peng Zhao, and Lijun Zhang. Optimistic online mirror descent for bridging stochastic and adversarial online convex optimization. Journal of Machine Learning Research , 25(178):1-62, 2024.
- [13] Elad Hazan and Satyen Kale. Extracting certainty from uncertainty: Regret bounded by variation in costs. Machine learning , 80:165-188, 2010.
- [14] Chao-Kai Chiang, Tianbao Yang, Chia-Jung Lee, Mehrdad Mahdavi, Chi-Jen Lu, Rong Jin, and Shenghuo Zhu. Online optimization with gradual variations. In Conference on Learning Theory , pages 6-1. JMLR Workshop and Conference Proceedings, 2012.
- [15] Peng Zhao, Yu-Jie Zhang, Lijun Zhang, and Zhi-Hua Zhou. Dynamic regret of convex and smooth functions. Advances in Neural Information Processing Systems , 33:12510-12520, 2020.
- [16] Yu-Hu Yan, Peng Zhao, and Zhi-Hua Zhou. Universal online learning with gradient variations: A multi-layer online ensemble approach. Advances in Neural Information Processing Systems , 36:37682-37715, 2023.
- [17] Peng Zhao, Yu-Jie Zhang, Lijun Zhang, and Zhi-Hua Zhou. Adaptivity and non-stationarity: Problem-dependent dynamic regret for online convex optimization. Journal of Machine Learning Research , 25(98):1-52, 2024.
- [18] Yan-Feng Xie, Peng Zhao, and Zhi-Hua Zhou. Gradient-variation online learning under generalized smoothness. Advances in Neural Information Processing Systems , 37:37865-37899, 2024.
- [19] Yu-Hu Yan, Peng Zhao, and Zhi-Hua Zhou. A simple and optimal approach for universal online learning with gradient variations. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.

- [20] Idan Amir, Idan Attias, Tomer Koren, Yishay Mansour, and Roi Livni. Prediction with corrupted expert advice. Advances in Neural Information Processing Systems , 33:14315-14325, 2020.
- [21] Shinji Ito. On optimal robustness to adversarial corruption in online decision problems. Advances in Neural Information Processing Systems , 34:7409-7420, 2021.
- [22] Julian Zimmert and Yevgeny Seldin. An optimal algorithm for stochastic and adversarial bandits. In The 22nd International Conference on Artificial Intelligence and Statistics , pages 467-475. PMLR, 2019.
- [23] Chung-Wei Lee, Haipeng Luo, Chen-Yu Wei, Mengxiao Zhang, and Xiaojin Zhang. Achieving near instance-optimality and minimax-optimality in stochastic and adversarial linear bandits simultaneously. In International Conference on Machine Learning , pages 6142-6151. PMLR, 2021.
- [24] Francesco Orabona and Dávid Pál. Coin betting and parameter-free online learning. Advances in Neural Information Processing Systems , 29, 2016.
- [25] Ashok Cutkosky. Artificial constraints and hints for unbounded online learning. In Conference on Learning Theory , pages 874-894. PMLR, 2019.
- [26] Dylan J Foster, Alexander Rakhlin, and Karthik Sridharan. Adaptive online learning. Advances in Neural Information Processing Systems , 28, 2015.
- [27] Andrew Jacobsen and Ashok Cutkosky. Parameter-free mirror descent. In Conference on Learning Theory , pages 4160-4211. PMLR, 2022.
- [28] Zakaria Mhammedi and Wouter M Koolen. Lipschitz and comparator-norm adaptivity in online learning. In Conference on Learning Theory , pages 2858-2887. PMLR, 2020.
- [29] Francesco Orabona and Dávid Pál. Scale-free online learning. Theoretical Computer Science , 716:50-69, 2018.
- [30] Ashok Cutkosky and Zak Mhammedi. Fully unconstrained online learning. Advances in Neural Information Processing Systems , 37:10148-10201, 2024.
- [31] Maor Ivgi, Oliver Hinder, and Yair Carmon. Dog is sgd's best friend: A parameter-free dynamic step size schedule. In International Conference on Machine Learning , pages 14465-14499. PMLR, 2023.
- [32] Ahmed Khaled, Konstantin Mishchenko, and Chi Jin. Dowg unleashed: An efficient universal parameter-free gradient descent method. Advances in Neural Information Processing Systems , 36:6748-6769, 2023.
- [33] Ahmed Khaled and Chi Jin. Tuning-free stochastic optimization. In International Conference on Machine Learning , pages 23622-23661. PMLR, 2024.
- [34] Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E Schapire. Fast convergence of regularized learning in games. Advances in Neural Information Processing Systems , 28, 2015.
- [35] Mengxiao Zhang, Peng Zhao, Haipeng Luo, and Zhi-Hua Zhou. No-regret learning in timevarying zero-sum games. In International Conference on Machine Learning , pages 2677226808. PMLR, 2022.
- [36] Peng Zhao. Lecture Notes for Advanced Optimization, 2025. Lecture 9. Optimism for Acceleration.
- [37] Sebastien Bubeck, Nikhil R Devanur, Zhiyi Huang, and Rad Niazadeh. Online auctions and multi-scale online learning. In Proceedings of the 2017 ACM Conference on Economics and Computation , pages 497-514, 2017.
- [38] Yuheng Zhao, Yu-Hu Yan, Kfir Yehuda Levy, and Peng Zhao. Gradient-variation online adaptivity for accelerated optimization with hölder smoothness. In Advances in Neural Information Processing Systems 38 (NeurIPS) , page to appear, 2025.

- [39] Ali Kavis, Kfir Y Levy, Francis Bach, and Volkan Cevher. UniXGrad: A universal, adaptive algorithm with optimal guarantees for constrained optimization. Advances in neural information processing systems , 32, 2019.
- [40] Itai Kreisler, Maor Ivgi, Oliver Hinder, and Yair Carmon. Accelerated parameter-free stochastic optimization. In The Thirty Seventh Annual Conference on Learning Theory , pages 3257-3324. PMLR, 2024.
- [41] Peng Zhao, Guanghui Wang, Lijun Zhang, and Zhi-Hua Zhou. Bandit convex optimization in non-stationary environments. Journal of Machine Learning Research , 22(125):1-45, 2021.
- [42] Zakaria Mhammedi, Wouter M Koolen, and Tim Van Erven. Lipschitz adaptivity with multiple learning rates in online learning. In Conference on Learning Theory , pages 2490-2511. PMLR, 2019.
- [43] H Brendan McMahan and Matthew Streeter. Adaptive bound optimization for online convex optimization. arXiv preprint arXiv:1002.4908 , 2010.
- [44] Ashok Cutkosky. Combining online learning guarantees. In Conference on Learning Theory , pages 895-913. PMLR, 2019.
- [45] Haipeng Luo, Alekh Agarwal, Nicolo Cesa-Bianchi, and John Langford. Efficient second order online learning by sketching. Advances in Neural Information Processing Systems , 29, 2016.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations of our regret bound are discussed in Remarks 4.2, 4.3, and 4.5.

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

Justification: The problem setting is clearly introduced in Section 2 and all theoretical results are accompanied by proofs.

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

Justification: This paper is a fully theoretical paper without an experimental section.

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

Justification: This paper is a fully theoretical paper without an experimental section.

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

Justification: This paper is a fully theoretical paper without an experimental section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: This paper is a fully theoretical paper without an experimental section.

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

Justification: This paper is a fully theoretical paper without an experimental section.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The authors confirm with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The work is mainly theoretical, thus there is no identifiable societal impact.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to

generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The work is mainly theoretical.

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

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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

## A Separation between σ 2 1: T and ˜ σ 2 1: T

Recall (Section 2) the two stochastic variance measures:

<!-- formula-not-decoded -->

and their cumulative versions σ 2 1: T = ∑ T t =1 σ 2 t , ˜ σ 2 1: T = ∑ T t =1 ˜ σ 2 t . Always σ 2 t ≤ ˜ σ 2 t by Jensen Inequality, but the gap can be arbitrarily large . The following simple 1-dimensional construction (with a growing number of disjoint regions) shows an order difference.

Proposition A.1 (Linear separation between σ 2 1: T and ˜ σ 2 1: T ) . Let X = ⋃ n i =1 X i ⊂ R with disjoint cells X i = [ i -1 , i ) for i ∈ { 1 , . . . , n } . For each round t ∈ { 1 , . . . , T } and each cell i ∈ { 1 , . . . , n } , draw independently

<!-- formula-not-decoded -->

and define the (stochastic) gradient field by

<!-- formula-not-decoded -->

with arbitrary values on cell boundaries. Let F t be the expected loss, defined up to an additive constant by ∇ F t ( x ) = E [ ∇ f t ( x )] (we fix the constant so that F t ≡ 0 ). Consider the two gradient-noise proxies

<!-- formula-not-decoded -->

and their sums σ 2 1: T = ∑ T t =1 σ 2 t and ˜ σ 2 1: T = ∑ T t =1 ˜ σ 2 t .

Then, for every n, T ≥ 1 ,

<!-- formula-not-decoded -->

hence, for all large n ,

<!-- formula-not-decoded -->

In particular, taking n = n ( T ) = T yields σ 2 1: T = 1 while ˜ σ 2 1: T ≥ (1 -e -1 ) T , i.e., a linear (in T ) separation between the two quantities.

Proof. By construction and independence, for any x ∈ X i ,

<!-- formula-not-decoded -->

so F t ≡ 0 (up to an additive constant). Therefore

<!-- formula-not-decoded -->

Summing over t gives σ 2 1: T = T/n .

For ˜ σ 2 t , note that s 2 t,i ≡ 1 and ( ∇ f t ( x )) 2 = c t,i for all x ∈ X i . Thus

<!-- formula-not-decoded -->

Since ( 1 -1 n ) n ≤ e -1 for all n , we have ˜ σ 2 t ∈ [ 1 -e -1 , 1] . Summing over t yields ˜ σ 2 1: T = T ( 1 -(1 -1 n ) n ) ≥ (1 -e -1 ) T . Finally, with n = T we obtain σ 2 1: T = 1 and ˜ σ 2 1: T ≥ (1 -e -1 ) T , which proves the claimed linear separation.

Remark. This example shows that regret bounds expressed in terms of ˜ σ 2 1: T = ∑ t E [sup x ∥ · ∥ 2 ] can be a factor Θ( T ) looser than bounds in terms of σ 2 1: T = ∑ t sup x E [ ∥ · ∥ 2 ] on the same instance.

## B Omitted Details of Section 2

## B.1 Proof of (2)

Proof. From Theorem 5 in [27], we have

<!-- formula-not-decoded -->

From Lemma 8 in [12], we also have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking expectations with Jensen's inequality and the definition of ˜ σ 2 1: T and Σ 2 1: T , we obtain

<!-- formula-not-decoded -->

## C Omitted Details of Section 3

## C.1 Auxiliary Lemmas

Lemma C.1 (Bregman Proximal Inequality) . The Bregman Proximal update in the form of x t +1 = arg min x ∈X {⟨ x, g t ⟩ + D ψ ( x, x t ) } satisfies

<!-- formula-not-decoded -->

Proof. By the first-order optimality condition at x t +1 , for any u ∈ X , we have

<!-- formula-not-decoded -->

On the RHS of (10), we expand each term by the definition of Bregman divergence

<!-- formula-not-decoded -->

Hence, the proof is finished by rearranging the terms.

Lemma C.2. Let x t = arg min x ∈X {⟨ x, m t ⟩ + D ψ t ( x, x ′ t ) } and x ′ t +1 = arg min x ∈X {⟨ x, g t ⟩ + D ψ t ( x, x ′ t ) } . Then, it holds for any u in X

<!-- formula-not-decoded -->

Proof. We have

<!-- formula-not-decoded -->

We apply Lemma C.1 twice, i.e., ⟨ a -u, f ⟩ ≤ D ψ ( u, b ) -D ψ ( u, a ) -D ψ ( a, b ) since a = arg min x ∈X ⟨ x, f ⟩ + D ψ ( x, b ) . Then, we have

<!-- formula-not-decoded -->

Substitute these two back to (11) and sum over T

providing the desired result.

<!-- formula-not-decoded -->

Lemma C.3. In OONS , we have 0 ≤ ⟨ x t -x ′ t +1 , ∇ t -m t ⟩ ≤ 2 ∥∇ t -m t ∥ 2 A -1 t and ∑ T t =1 ⟨ x t -x ′ t +1 , ∇ t -m t ⟩ ≤ O ( r ln( Tη 1 z T /z 1 ) η T ) .

Proof. We define

<!-- formula-not-decoded -->

By OONS, we have

<!-- formula-not-decoded -->

Since ∇ 2 D ψ t = A t and by the first-order optimality at x ′ t +1 , we have

<!-- formula-not-decoded -->

Also, we can write F ∇ t ( x t ) -F ∇ t ( x ′ t +1 ) as

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

where the second inequality comes from x t = arg min F m t ( x ) . Therefore, we have

<!-- formula-not-decoded -->

Since x ′ t +1 minimizes F m t ( x ) and x t minimizes F ∇ t ( x ) , we have

<!-- formula-not-decoded -->

By the definition of ∇ t = g t +32 η t ⟨ x t , g t -m t ⟩ ( g t -m t ) , we have ∥∇ t -m t ∥ 2 = ∥ g t -m t +32 η t ⟨ x t , g t -m t ⟩ ( g t -m t ) ∥ 2 ≤ ∥ g t -m t ∥ 2 +32 η t D ∥ g t -m t ∥ 2 2 ≤ 3 2 ∥ g t -m t ∥ 2 . (12)

Next, we define

<!-- formula-not-decoded -->

Hence, A t ⪰ ¯ A t since ∥∇ t -m t ∥ 2 2 ≤ 4 ∥ g t -m t ∥ 2 2 ≤ 4 z 2 t . Also, we have

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

For | ¯ A T | :

<!-- formula-not-decoded -->

where r is the rank of ∑ T t =1 ( ∇ t -m t )( ∇ t -m t ) ⊤ . Therefore, we have

<!-- formula-not-decoded -->

Lemma C.4. Let s t , ∀ t ∈ [ T ] be non-negative. Then,

<!-- formula-not-decoded -->

Proof. Let S t = ∑ t j =1 s j . Then,

<!-- formula-not-decoded -->

Lemma C.5 (Theorem 5 in [8], Lemma 3 in [12]) . Under Assumptions 2.1 and 2.3, we have

<!-- formula-not-decoded -->

## C.2 Proof of Theorem 3.1

Proof. By Lemma C.1, we have

<!-- formula-not-decoded -->

Term D ψ 1 ( u, x ′ 1 ) : Since the initialization of x ′ 1 = 0 and A 1 = O ( z 2 1 I ) , we have

<!-- formula-not-decoded -->

Term ∑ T -1 t =1 D ψ t +1 ( u, x ′ t +1 ) -D ψ t ( u, x ′ t +1 ) : First, we have

<!-- formula-not-decoded -->

Also,

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2

where the last inequality comes from A t ⪰ A t -1 ⪰ 4 z 1 I . Since c t ( x ) is convex in x , we have

<!-- formula-not-decoded -->

Therefore, the final regret bound here is

<!-- formula-not-decoded -->

## C.3 Proof of Theorem 3.2

Proof. In this case, we have ∥ g t -m t ∥ 2 ≤ 2 G, ∀ t ∈ [ T ] . Then, we set z t = 2 G, ∀ t ∈ [ T ] and step-size η t as

<!-- formula-not-decoded -->

By substituting η t and z t into (3), we have

<!-- formula-not-decoded -->

By Lemma C.4, we have

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

By Lemma C.5 and the definition of g t and m t , we have

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

## C.4 Computational Complexity of OONS

The primary computational bottleneck in our proposed OONS algorithm (Algorithm 1) is the management of the d × d matrix A t . A naive implementation would involve storing this dense matrix and performing a full matrix inversion at each step, leading to prohibitive costs in high-dimensional settings.

- Storage: Storing the dense d × d matrix A t requires O ( d 2 ) memory.
- Computation: Anaive matrix inversion A -1 t would cost O ( d 3 ) per step, and the subsequent matrix-vector products would cost O ( d 2 ) .

In practice, this complexity can be significantly reduced. Since the matrix A t is constructed by a sum of outer products ( A t = cI + ∑ η s v s v ⊤ s ), its inverse can be efficiently computed and updated at each step using the Sherman-Morrison-Woodbury formula. This reduces the update complexity from O ( d 3 ) to O ( d 2 ) per step.

However, an O ( d 2 ) complexity per step can still be prohibitive in high-dimensional scenarios. To address this, existing research has explored several techniques:

- Matrix Sketching: This technique approximates the original d × d matrix A t with a much smaller "sketched" matrix, thereby significantly reducing both storage and computational requirements. For instance, Luo et al. [45] have successfully applied sketching to the Online Newton Step (ONS) algorithm, creating matrix-free updates that avoid direct manipulation of the high-dimensional matrix.
- Sparsity: If the gradient vectors are sparse across most iterations, specialized sparse data structures and algorithms can be utilized. This allows update computations to be performed much more efficiently, avoiding the full O ( d 2 ) cost of dense matrix-vector multiplications.

Integrating these high-dimensional adaptation techniques into our proposed algorithms for the SEA model and analyzing their theoretical guarantees is an interesting direction for future work.

Algorithm 5 Multi-scale Multiplicative-weight with Correction (MsMwC)

Input:

w ′ 1 ∈ ∆ N .

- 1: for t = 1 , . . . , T do
- 2: Receive the prediction h t ∈ R N .
- 3: Compute w t = arg min w ∈ ∆ N ⟨ w,h t ⟩ + D ϕ ( w,w ′ t ) , where ϕ t ( w ) = ∑ N j =1 w j β t,j ln w j .
- 4: Play w t , receive ℓ t and construct correction term a t ∈ R N with a t,j = 32 β t,j ( ℓ j t -m j t ) 2 .
- 6: end for
- 5: Compute w ′ t +1 = arg min w ∈ ∆ N ⟨ w,ℓ t + a t ⟩ + D ϕ ( w,w ′ t ) .

## D Omitted Details of Section 4.1

## D.1 Multi-scale Multiplicative-weight with Correction (MsMwC)

We rephrase the MsMwC algorithm [10] as the following Algorithm 5.

## D.2 Proof of equation

## (5)

The final decision x t is a weighted-average of base-learners' decisions: x t = ∑ N j =1 w t,j x j t . Then,

<!-- formula-not-decoded -->

where ℓ t ∈ R N with ℓ j t = ⟨∇ f t ( x j t ) , x j t ⟩ and w i ⋆ is a vector in ∆ N whose j -th component is ( w i ⋆ ) j = 1 if j = i and 0 otherwise.

## D.3 Auxiliary Lemma

Lemma D.1. (Theorem 6 in [10]) Suppose for all t ∈ [ T ] and j ∈ [ N ] , | ℓ j t | ≤ GD j and | h j t | ≤ GD j , where ℓ j t = ⟨∇ f t ( x j t ) , x j t ⟩ and h j t = ⟨∇ f t -1 ( x j t -1 ) , x j t ⟩ . Define Γ j = ln( NTD j D 1 ) and the set E as

<!-- formula-not-decoded -->

where G k is the MsMwC algorithm with w ′ 1 being uniform over Z ( k ) , S = { k ∈ Z : ∃ j ∈ [ N ] , GD j ≤ 2 k -2 ≤ GD j √ T } and Z k = { j ∈ [ N ] : GD j ≤ 2 k -2 } . We have the following regret

bound

<!-- formula-not-decoded -->

Proof. The regret ∑ T t =1 ⟨ ℓ t , w t -w i ⋆ ⟩ can also be decomposed as

<!-- formula-not-decoded -->

where e k ⋆ is the k ⋆ -th standard basis vector and the second equality is from the definition of L t .

For any i ∈ [ N ] , there exists a k ⋆ such that η k ⋆ ≤ min { 1 128 GD i , √ Γ i ∑ T t =1 ( ℓ i t -h i t ) 2 } ≤ 2 η k ⋆ . By Lemma 1 and Theorem 4 in [10], we have

<!-- formula-not-decoded -->

## D.4 Proof of Theorem 4.1

Proof. We begin with considering the first and second cases that ∥ u ∥ 2 ≤ D 1 and ∥ u ∥ 2 ≤ D i ≤ 2 ∥ u ∥ 2 .

Here, we define g j t = ∇ f t ( x j t ) and m j t = ∇ f t -1 ( x j t -1 ) for all t ∈ [ T ] and j ∈ [ N ] . For the meta regret, we define ℓ t ∈ R N with ℓ j t = ⟨∇ f t ( x j t ) , x j t ⟩ and h t ∈ R N with h j t = ⟨∇ f t -1 ( x j t -1 ) , x j t ⟩ . Then, | ℓ j t | ≤ GD j and | h j t | ≤ GD j for all t ∈ [ T ] and j ∈ [ N ] . By applying Lemma D.1, we have

<!-- formula-not-decoded -->

By the definition of ℓ i t and h i t , we have

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Then, we investigate the expert regret part. Here, we set the step-size for the expert i as

<!-- formula-not-decoded -->

By substituting the step-size specified in (14) to (3), we have

<!-- formula-not-decoded -->

Therefore, by combining (13) and (15), we have

<!-- formula-not-decoded -->

Then, by applying Lemma C.5, we have

<!-- formula-not-decoded -->

Since the expert i runs OONS within the set X i = { x : ∥ x ∥ 2 ≤ D i } ⊆ X , we have

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Case 1 : ∥ u ∥ 2 ≤ D 1 . We take i = 1 and substitute D i with D 1 into (16). Hence, we have

<!-- formula-not-decoded -->

Case 2 : In this case, let i be the smallest integer such that ∥ u ∥ 2 ≤ D i = 2 i . We have ∥ u ∥ 2 ≤ D i ≤ 2 ∥ u ∥ 2 since D i +1 = 2 D i . Then, we substitute D i with 2 ∥ u ∥ 2 into the regret bound (16). Then,

<!-- formula-not-decoded -->

Case 3 : ∥ u ∥ 2 &gt; D max .

Next, we consider the case when ∥ u ∥ 2 &gt; D max = 2 N . Then, we have

<!-- formula-not-decoded -->

We take N = ⌈ log T ⌉ , then T ≤ ∥ u ∥ 2 . Therefore,

<!-- formula-not-decoded -->

By combining these two cases above, the desirable regret bound is achieved.

## E Omitted Details of Section 4.2

In this section, we also denote g t = ∇ f t ( x t ) and m t = ∇ f t -1 ( x t -1 ) . We first note that

<!-- formula-not-decoded -->

̸

Thus, we need to update D t at most O (log T ) times. Let M be the number of total updates in D t , where M = O (log T ) . We split the T iterations into M intervals I m with m ∈ [ M ] , where the last iteration of I m (denoted by t m ) either equals to T or D t m +1 = D t m .

## E.1 Proof of Theorem 4.5

Proof. We have

<!-- formula-not-decoded -->

where we define u t = min { 1 , D t ∥ u ∥ 2 } u .

We first consider T m as

<!-- formula-not-decoded -->

Note that iteration t within interval I m , i.e., t ∈ I m , the domain has a bounded diameter D t . When t ∈ I m , we take

<!-- formula-not-decoded -->

where t 1 is first index in I m . Also, we denote the last index in I m as t m , respectively. From the Line 5 or 8 in FPF-OONS we need to reset x ′ t 1 = 0 and A t for all t ∈ I m at iteration t 1 as follows

<!-- formula-not-decoded -->

Similar to the proof of Theorem 3.1, we have

<!-- formula-not-decoded -->

where ∇ t = ˜ g t +32 η t ⟨ x t , ˜ g t -m t ⟩ ( ˜ g t -m t ) .

We first consider the term ∑ t m t = t 1 ⟨ x t -x ′ t +1 , ∇ t -m t ⟩ . By Lemma C.3, we have 0 ≤ ⟨ x t -x ′ t +1 , ∇ t -m t ⟩ ≤ 2 ∥∇ t -m t ∥ 2 A -1 t .

Also, we have

<!-- formula-not-decoded -->

By the definition of ∇ t = g t +32 η t ⟨ x t , g t -m t ⟩ ( g t -m t ) and ∥ g t -m t ∥ 2 ≤ ∥ g t -m t ∥ 2 , we have

<!-- formula-not-decoded -->

For t ∈ I m , we also redefine

<!-- formula-not-decoded -->

Hence, A t ⪰ ¯ A t since ∥∇ t -m t ∥ 2 2 ≤ 4 ∥ ˜ g t -m t ∥ 2 2 ≤ 4 z 2 t . Also, for t ∈ [ t 1 +1 , t m ] , we have ( ∇ t -m t )( ∇ t -m t ) ⊤ = 1 η t [ η t ( ∇ t -m t )( ∇ t -m t ) ⊤ ] = 1 η t ( ¯ A t -¯ A t -1 )

Then,

<!-- formula-not-decoded -->

For | ¯ A t m | :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we have

Term D ψ t 1 ( u, x ′ t 1 ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Term ∑ t m -1 t = t 1 D ψ t +1 ( u t , x ′ t +1 ) -D ψ t ( u t , x ′ t +1 ) :

By the definition of u t = min { 1 , D t ∥ u ∥ 2 } u , we have

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality comes from A t ⪰ A t -1 ⪰ 4 z t 1 I when t ∈ [ t 1 +1 , t m ] . Then, we have

<!-- formula-not-decoded -->

Furthermore, by ∥ ˜ g t -m t ∥ 2 ≤ ∥ g t -m t ∥ 2 , we have

<!-- formula-not-decoded -->

At the last iteration T , we have

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

̸

Now, we bound T extra . We observe that u t is either u or D t ∥ u ∥ 2 u . When u t = u , ∥ u ∥ 2 ≥ D t &gt; √ ∑ t s =1 ∥ g s ∥ 2 max k ≤ s ∥ g k ∥ 2 . Once u t = u , it stays there. Let t ∗ be the last round when u t = u .

̸

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

By Lemma C.5, we have

<!-- formula-not-decoded -->

Therefore, we can conclude that

<!-- formula-not-decoded -->

where σ 1: T captures the stochastic gradient deviation (without the squared norm) and G 1: T denotes the sum of maximum expected gradients.