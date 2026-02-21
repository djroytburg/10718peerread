## Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization

## Xiaochuan Gong

George Mason University xgong2@gmu.edu

## Jie Hao

George Mason University jhao6@gmu.edu

## Abstract

Hierarchical optimization refers to problems with interdependent decision variables and objectives, such as minimax and bilevel formulations. While various algorithms have been proposed, existing methods and analyses lack adaptivity in stochastic optimization settings: they cannot achieve optimal convergence rates across a wide spectrum of gradient noise levels without prior knowledge of the noise magnitude. In this paper, we propose novel adaptive algorithms for two important classes of stochastic hierarchical optimization problems: nonconvex-strongly-concave minimax optimization and nonconvex-strongly-convex bilevel optimization. Our algorithms achieve sharp convergence rates of ˜ O (1 / √ T + √ ¯ σ/T 1 / 4 ) in T iterations for the gradient norm, where ¯ σ is an upper bound on the stochastic gradient noise. Notably, these rates are obtained without prior knowledge of the noise level, thereby enabling automatic adaptivity in both low and high-noise regimes. To our knowledge, this work provides the first adaptive and sharp convergence guarantees for stochastic hierarchical optimization. Our algorithm design combines the momentum normalization technique with novel adaptive parameter choices. Extensive experiments on synthetic and deep learning tasks demonstrate the effectiveness of our proposed algorithms.

## 1 Introduction

Hierarchical optimization refers to a class of optimization problems characterized by nested structures in their objectives or constraints, such as minimax optimization [61, 64, 50] and bilevel optimization [5, 13]. These problems have wide applications in machine learning. For example, minimax optimization is the foundation for adversarial learning [27] and AUC maximization [75, 53], while bilevel optimization is central to meta-learning [19] and hyperparameter optimization [20, 18]. In this paper, we are interested in solving two classes of stochastic hierarchical optimization problems. The first class is the nonconvex-strongly-concave minimax problem in (1):

<!-- formula-not-decoded -->

where D is an unknown distribution where one can sample from, and f ( x, y ) is nonconvex in x and strongly concave in y . The second class is the nonconvex-strongly-convex bilevel problem in (2):

<!-- formula-not-decoded -->

where D x and D y are unknown distributions where one can sample from, f ( x, y ) := E ξ ∼D x [ F ( x, y ; ξ )] is nonconvex in x and g ( x, y ) := E ξ ∼D y [ G ( x, y ; ζ )] is strongly convex in y . We call x the upper-level variable and y the lower-level variable. Note that the bilevel problem in (2) degenerates to the minimax problem in (1) when g = -f and then Φ( x ) = max y ∈ R dy f ( x, y ) .

Mingrui Liu

George Mason University mingruil@gmu.edu

There are various algorithms designed for the minimax problem (1) and the bilevel problem (2) in the stochastic setting [64, 50, 47, 22, 34, 40, 8, 33, 45, 29, 28]. However, existing algorithms and analyses lack adaptivity to various levels of stochastic gradient noise: their convergence rates remain suboptimal in various noise regimes unless the noise level is known a priori (see Appendix H for discussion and details). In contrast, such an adaptivity guarantee is achieved in single-level stochastic optimization [48, 67, 42, 17, 56, 2] by AdaGrad-type algorithms [60, 14]. This naturally motivates us to study the following question:

How can we design adaptive gradient algorithms for stochastic hierarchical optimization problems (1) and (2) that achieve convergence rates automatically adapting to the level of stochastic gradient noise, without requiring prior knowledge of this noise?

Designing such algorithms in stochastic hierarchical optimization presents significant challenges. In particular, applying AdaGrad-type algorithms (e.g., AdaGrad-Norm [67]) simultaneously to the upper- and lower-level variables will introduce complicated randomness dependency issues due to AdaGrad stepsizes. These dependencies are difficult to handle analytically without imposing strong assumptions such as bounded stochastic gradients or bounded function values [47]. However, such assumptions undermine the algorithm's ability to adapt effectively in various noise regimes.

In this paper, we address these challenges by developing novel adaptive algorithms for solving (1) and (2), respectively. Unlike standard AdaGrad-type algorithms [67], the key innovation of our approach lies in combining the momentum normalization technique [11] with novel adaptive parameter choices. A distinctive feature of our method is the dynamic adjustment of the momentum parameter based on online estimates of the stochastic gradient variance. This adaptive momentum directly informs our stepsize scheme, enabling improved convergence across both high- and low-noise regimes without requiring prior knowledge of the noise level. The primary challenge in analyzing the convergence of our proposed algorithms is simultaneously controlling the upper-level and lower-level errors under time-varying parameters, including adaptive momentum and stepsizes, while maintaining adaptivity in the presence of unknown stochastic noise. Our main contributions are summarized as follows.

- We propose two new adaptive algorithms, namely Ada-Minimax and Ada-BiO, for solving the nonconvex-strong-concave minimax optimization problem (1) and the nonconvexstrongly-convex bilevel optimization problem (2) respectively. Both algorithms leverage the momentum normalization technique and adaptively set the momentum parameter, along with carefully designed adaptive stepsizes for both upper- and lower-level variables. To our knowledge, adaptive algorithms of this type that distincts from standard AdaGrad approaches are novel within both stochastic single-level and hierarchical optimization problems.
- We obtain a high probability convergence rate of ˜ O (1 / √ T + √ ¯ σ/T 1 / 4 ) in T iterations for the gradient norm (here ˜ O ( · ) compresses poly-logarithmic factors of T and the failure probability δ ∈ (0 , 1) ), where ¯ σ denotes an upper-bound on the stochastic gradient noise. Notably, our algorithms automatically adapt to both high- and low-noise regimes without requiring prior knowledge of the noise levels.
- We empirically validate our theoretical results through a synthetic experiment and various deep learning tasks, including deep AUC maximization and hyperparameter optimization. Our results demonstrate that our proposed algorithms consistently outperform existing adaptive gradient methods as well as well-tuned non-adaptive baselines.

## 2 Related Work

Minimax Optimization. Early works on minimax optimization focused on convex-concave settings and developed first-order algorithms with convergence guarantees [61, 63, 41, 62]. The study of first-order algorithms for nonconvex-concave minimax optimization was pioneered by [64]. Subsequent works improved convergence rates under various assumptions [53, 68, 59], proposed single-loop algorithms [50, 29, 71], and relaxed the concavity requirement on the maximization variable [69, 51, 52, 4]. Some recent efforts have incorporated adaptive gradient methods into minimax optimization [51, 47, 38, 70]. However, none of these approaches provide convergence guarantees that adapt across different levels of stochastic gradient noise in nonconvex-strongly-concave settings.

Bilevel Optimization. Bilevel optimization [5, 13] is a type of hierarchical optimization problem where one optimization task (i.e., upper-level problem) is constrained by another optimization task

(i.e., lower-level problem). The first nonasymptotic convergence guarantees for bilevel optimization problems were established by [22], followed by a growing body of work that established improved complexity bounds under various assumptions [34, 40, 8, 43, 9, 12, 28, 72, 33, 25, 24, 12, 36, 45, 65, 57, 58]. More recently, a few studies have explored bilevel optimization algorithms with adaptive step sizes [15, 1, 73, 37]. However, these methods are either restricted to the deterministic setting [1, 73] or fail to adapt to a broad range of stochastic gradient noise levels [15, 37, 26] in nonconvex-strongly-convex bilevel optimization problems.

Adaptive Gradient Algorithms. Adaptive gradient algorithms [60, 14, 66, 44] refer to a class of first-order algorithms where the stepsizes are computed based on the historical stochastic gradients, and they are empirically effective for training deep neural networks. The theoretical guarantees of adaptive gradient algorithms for single-level optimization problems are extensively studied and well-understood in the literature [48, 67, 42, 17, 56, 2, 46, 16]. Extensions of adaptive methods to minimax [51, 47, 38, 70] and bilevel optimization [37, 26, 1, 73] have also been proposed. However, none of these works establish theoretical guarantees for adaptivity to unknown stochastic gradient noise levels in nonconvex-strongly-concave minimax or nonconvex-strongly-convex bilevel optimization problems, as achieved by our proposed algorithms in this work.

## 3 Preliminaries

Notations. Denote ∥ · ∥ as the Euclidean norm. We use the standard O ( · ) , Θ( · ) , Ω( · ) notations, with ˜ O ( · ) , ˜ Θ( · ) , ˜ Ω( · ) hiding logarithmic factors. Throughout, with slight abuse of notation, we use F t to denote the filtration (i.e., σ -algebra) generated by stochastic queries up to iteration t , and E t [ · ] = E [ · | F t ] to denote the conditional expectation with respect to F t , for all algorithms. A function h is said to be L -smooth if ∥∇ h ( x ) - ∇ h ( y ) ∥ ≤ L ∥ x -y ∥ for all x, y ∈ R d . We additionally assume that all objective functions are bounded from below, i.e., f ∗ := inf x f ( x ) &gt; -∞ (Section 5.1) and Φ ∗ := inf x Φ( x ) &gt; -∞ (Sections 3.1 and 3.2).

Settings. Let y ∗ ( x ) = arg max y ∈ R dy f ( x, y ) for (1) and y ∗ ( x ) = arg min y ∈ R dy g ( x, y ) for (2). Define the objective function Φ( x ) = f ( x, y ∗ ( x )) for both minimax and bilevel optimization. Recall from Section 1 that the bilevel problem (2) reduces to the minimax problem (1) when g = -f . The goal of this paper is to minimize Φ .

## 3.1 Assumptions for Nonconvex-Strongly-Concave Minimax Optimization

Assumption 3.1. The objective function f is L -smooth in ( x, y ) and f ( x, · ) is µ -strongly concave. Assumption 3.2. (i) The gradient oracle is unbiased, i.e., E [ ∇ F ( x, y ; ξ ) | x, y ] = ∇ f ( x, y ) . (ii) With probability one, the following holds: ¯ σ x ≤ ∥∇ x F ( x, y ; ξ ) -∇ x f ( x, y ) ∥ ≤ ¯ σ x with ¯ σ x ≥ 0 and ∥∇ y F ( x, y ; ξ ) -∇ y f ( x, y ) ∥ ≤ σ y .

Remark: Assumptions 3.1 and 3.2(i) are standard in the minimax optimization literature [50, 74, 29]. The main extra assumption we make is Assumption 3.2(ii): the stochastic gradient noise is lower bounded and upper-bounded (with probability one), which may appear somewhat unusual. However, this assumption holds naturally in the additive noise setting used in certain nonconvex optimization scenarios, such as escaping saddle points with isotropic noise [21], where the stochastic gradient noise is sampled uniformly from the unit sphere and therefore has a nonzero magnitude with probability one. We also empirically validate this assumption, as shown in Appendix L. In the noiseless case, we have ¯ σ x = ¯ σ x = 0 , and σ y = 0 .

## 3.2 Assumptions for Nonconvex-Strongly-Convex Bilevel Optimization

Assumption 3.3. The objective functions f and g satisfy: (i) f is L -smooth in ( x, y ) ; for every x, ξ , ∥∇ y f ( x, y ) ∥ ≤ l f, 0 and ∥∇ y F ( x, y ; ξ ) ∥ ≤ l f, 0 . (ii) For every x , g ( x, · ) is µ g -strongly convex for µ g &gt; 0 and g is l g, 1 -smooth in ( x, y ) . (iii) g is twice continuously differentiable, and ∇ 2 xy g, ∇ 2 yy g are l g, 2 -Lipschitz in ( x, y ) .

Remark: Assumption 3.3 is standard in the bilevel optimization literature [40, 45, 22, 33, 7]. Notably, the condition ∥∇ y F ( x, y ; ξ ) ∥ ≤ l f, 0 is essential for deriving ∥ ¯ ∇ f ( x, y ; ¯ ξ ) -E [ ¯ ∇ f ( x, y ; ¯ ξ )] ∥ ≤ ¯ σ ϕ in Lemma E.3 (see Appendix E.1 for the definition of ¯ ∇ f ( x, y ; ¯ ξ ) ), where ¯ σ ϕ plays a similar role to

¯ σ x in Assumption 3.2 for minimax optimization. Under these assumptions, the objective function Φ is L F -smooth; please refer to Lemma E.1 in Appendix E for the definition of L F and further details.

Assumption 3.4. All stochastic estimators are unbiased, and almost surely satisfy (i) ∥∇ x F ( x, y ; ξ ) -∇ x f ( x, y ) ∥ ≤ σ f ; (ii) ∥∇ y F ( x, y ; ξ ) - ∇ y f ( x, y ) ∥ ≤ σ f ; (iii) ∥∇ y G ( x, y ; ζ ) - ∇ y g ( x, y ) ∥ ≤ σ g, 1 ; (iv) ∥∇ 2 xy G ( x, y ; ζ ) -∇ 2 xy g ( x, y ) ∥ ≤ σ g, 2 ; (v) ∥∇ 2 yy G ( x, y ; ζ ) -∇ 2 yy g ( x, y ) ∥ ≤ σ g, 2 ; (vi) ∥ ¯ ∇ f ( x, y ; ¯ ξ ) -E [ ¯ ∇ f ( x, y ; ¯ ξ )] ∥ ≥ ¯ σ ϕ , where ¯ ∇ f ( x, y ; ¯ ξ ) is defined in Equation (39).

Remark: Assumption 3.4(i)-(v) assumes the noise in the stochastic gradient and Hessian/Jacobian is bounded with probability one. This is a commonly used assumption to establish high probability guarantees or handle generalized-smooth objective functions in the single-level optimization literature [46, 2, 39, 78, 77], as well as for stochastic bilevel optimization under the unbounded smoothness setting [33, 25]. Assumption 3.4(vi) is a stochastic gradient noise lower bound for the bilevel optimization problem, sharing a similar spirit to Assumption 3.2(ii). Note that Assumption 3.2 is empirically verified in Appendix L. Under Assumption 3.4, we also have ∥ ¯ ∇ f ( x, y ; ¯ ξ ) -E [ ¯ ∇ f ( x, y ; ¯ ξ )] ∥ ≤ ¯ σ ϕ , where the definition of ¯ σ ϕ can be found in Equation (44). See the detailed proof in Lemma E.3.

Additional Notations. In the subsequent analysis, we denote κ σ := ¯ σ/ ¯ σ in Section 5.1 (single-level optimization), κ σ := ¯ σ x / ¯ σ x in Section 4.2 (minimax optimization), and κ σ := ¯ σ ϕ / ¯ σ ϕ in Section 4.3 (bilevel optimization). We also adopt the convention 0 / 0 := 1 .

## 4 Algorithms and Main Results

## 4.1 Main Challenges and Algorithm Design

Main Challenges. While numerous adaptive gradient algorithms with adaptivity to stochastic gradient noise are developed in single-level optimization [48, 67, 42, 17, 56, 2], designing algorithms with such an adaptive guarantee in hierarchical optimization is nontrivial. The main challenges lies in the following two aspects. First, designing such an algorithm in hierarchical optimization requires a careful balance between the upper- and lower-level update [29, 34, 10], which is difficult to achieve without the knowledge of the noise magnitude of stochastic gradient. Second, applying AdaGrad-type algorithms (e.g., AdaGrad-Norm [67]) simultaneously to the upper- and lower-level variables will introduce complicated randomness dependency issues due to AdaGrad stepsizes [2], which are difficult to handle unless strong assumptions (e.g., bounded stochastic gradient, bounded function value) are imposed as in [47].

Algorithm Design. To address these main challenges, our proposed algorithms leverage the normalized stochastic gradient descent (SGD) with momentum for the upper-level variable [11] with a noise-aware adaptive momentum parameter and carefully crafted adaptive stepsize schemes for both levels. The momentum parameter automatically estimates the level of noise in the stochastic gradients on the fly, and this estimate is used to construct the stepsizes to maintain a balanced progress across both levels. These adaptive mechanisms, together with the momentum normalization technique, not only improve optimization stability but also make the theoretical convergence analysis more tractable. In particular, our proposed adaptive algorithms, namely Ada-Minimax and Ada-BiO, are designed for the minimax problem (1) and the bilevel problem (2) respectively. Both algorithms achieve sharp and adaptive convergence rates of ˜ O (1 / √ T + √ ¯ σ/T 1 / 4 ) for the gradient norm, where ¯ σ denotes an upper bound on the stochastic gradient noise. We describe our methods in Algorithms 1 and 2 with novel parameter choices in Equations (3) and (4). Their respective convergence guarantees are stated in Theorems 4.1 and 4.2.

Adaptive Parameter Choices. For simplicity, let α t = 1 -β t . In particular, for both Algorithms 1 and 2, we set α t , α ′ t , η x,t , η y,t as follows:

<!-- formula-not-decoded -->

Algorithm 1 Adaptive Algorithm for Minimax Optimization (Ada-Minimax)

- 1: Input: x 1 , y 1 , m 1 = ∇ x F ( x 1 , y 1 ; ξ 1 ) 2: for t = 1 , . . . , T do 3: m t = β t m t -1 +(1 -β t ) g x,t 4: x t +1 = x t -η x,t m t ∥ m t ∥ 5: y t +1 = y t + η y,t g y,t 6: end for

Algorithm 2 Adaptive Algorithm for Bilevel Optimization (Ada-BiO)

- 1: Input: x 1 , y 1 , m 1 = ¯ ∇ f ( x 1 , y 1 ; ¯ ξ 1 ) 2: for t = 1 , . . . , T do 3: m t = β t m t -1 +(1 -β t ) g x,t 4: x t +1 = x t -η x,t m t ∥ m t ∥ 5: y t +1 = y t -η y,t g y,t 6: end for

In the above formulas, the terms g x,t , ˜ g x,t , and g y,t carry different meanings; see the subsequent sections (Sections 4.2 and 4.3) for their precise definitions. For simplicity, we set η x = η y = η in analysis of Algorithms 1 and 2 (see Theorems 4.1 and 4.2). It is worth noting that this condition is not necessary for establishing convergence, as it only affects the universal constants in the rate.

## 4.2 Adaptive Algorithm for Minimax Optimization

Our proposed algorithm Ada-Minimax is presented in Algorithm 1. The algorithm updates the upper-level variable using normalized SGD with momentum [11] with adaptive and parameter-free choices for the momentum parameter and learning rates. The lower-level variable is updated by AdaGrad-Norm. In Equations (3) and (4), g x,t = ∇ x F ( x t , y t ; ξ t ) , ˜ g x,t = ∇ x F ( x t , y t ; ξ ′ t ) , and g y,t = ∇ y F ( x t , y t ; ξ t ) , with ξ t , ξ ′ t being independent samples.

Intuitively, the term ∑ t k =1 ∥ g x,k -˜ g x,k ∥ 2 in the denominator of α t is designed to approximate the variance term σ 2 T as in [11], and this choice is partly inspired by AdaGrad-Norm [14, 2]. Additionally, using α ′ t instead of α t in the design of η x,t effectively controls the ratio η x,t /η y,t and facilitates establishing Lemma 5.7. It is worth noting that Assumption 3.2 plays a crucial role in deriving tight, high-probability upper and lower bounds for both ∑ t k =1 ∥ g x,k -˜ g x,k ∥ 2 and α t , see Lemma 5.5 for details.

Theorem 4.1. Under Assumptions 3.1 and 3.2 and the parameter choices in Equations (3) and (4) , let ¯ σ x = σ y , then for any δ ∈ (0 , 1 / 7) , it holds with probability at least 1 -7 δ that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C m = ˜ O ( κ 4 σ ) and D are defined in Equations (24) and (38) , respectively.

Remark: Theorem 4.1 demonstrates that Ada-Minimax achieves a rate of ˜ O ( √ ¯ σ x /T 1 / 4 ) in the stochastic setting ( ¯ σ x &gt; 0 ) and ˜ O (1 / √ T ) rate in the deterministic setting ( ¯ σ x = 0 ). More importantly, our bound achieves the same bound of normalized SGD with momentum under known stochastic gradient variance [11]: it automatically interpolates between sharp rates in both high-noise and lownoise regimes without the knowledge of noise level . Specifically, the convergence rate improves from ˜ O (1 /T 1 / 4 ) to a faster ˜ O (1 / √ T ) when ¯ σ x is sufficiently small, namely ¯ σ x = O (1 / √ T ) . Notably, this automatic rate interpolation does not require prior knowledge of any problem-dependent parameters, and our proposed Ada-Minimax algorithm is fully parameter-free. In contrast, TiAda [47] does not exhibit such a bound in the low-noise regime (e.g., ¯ σ x = O (1 / √ T ) ), and its convergence rate is not optimal with respect to ¯ σ x in the stochastic setting since their convergence rate (e.g., Theorem 3.2 in [47]) does not explicitly depend on ¯ σ x . See detailed proof of Theorem 4.1 in Appendix D. A comparison of adaptive methods for minimax optimization is also presented in Table 1.

## 4.3 Adaptive Algorithm for Bilevel Optimization

Our proposed algorithm Ada-Bio is presented in Algorithm 2. The overall framework closely resembles that of Algorithm 1. The upper-level variable is updated using normalized SGD with

Table 1: Comparison of Adaptive Methods for Minimax Optimization

| Method      | Setting                                              | Assumptions                          | High Probability   | Complexity √                                                        |
|-------------|------------------------------------------------------|--------------------------------------|--------------------|---------------------------------------------------------------------|
| TiAda [47]  | Deterministic [47, Theorem 3.1]                      | Assumptions 3.1 to 3.3 in [47]       |                    | O (1 / T )                                                          |
| TiAda [47]  | Stochastic [47, Theorem 3.2]                         | Assumptions 3.1 to 3.6 in [47]       | ✗                  | O ( poly ( G ) · ( T α - 1 2 + T - α 2 + T β - 1 2 + T - β 2 )) 1 √ |
| Ada-Minimax | Deterministic &Stochastic (Theorem 4.1 in this work) | Assumptions 3.1 and 3.2 in this work | ✓                  | ˜ O (1 / T + ¯ σ 1 / 4 x /T 3 / 8 + √ ¯ σ x /T 1 / 4 )              |

Table 2: Comparison of Adaptive Methods for Bilevel Optimization

| Method      | Setting                                              | Assumptions                          | High Probability   | Complexity √                                                          |
|-------------|------------------------------------------------------|--------------------------------------|--------------------|-----------------------------------------------------------------------|
| S-TFBO [73] | Deterministic [73, Theorem 2]                        | Assumptions 1 to 3 in [73]           |                    | ˜ O (1 / T )                                                          |
| Ada-Bio     | Deterministic &Stochastic (Theorem 4.2 in this work) | Assumptions 3.3 and 3.4 in this work | ✓                  | ˜ O (1 / √ T + σ 1 / 4 g, 1 /T 3 / 8 +( √ ¯ σ ϕ + √ σ g, 1 ) /T 1 / 4 |

momentum [11], employing adaptive choices for the momentum parameter and learning rate. This approach differs from those of [33, 25], where fixed, non-adaptive momentum parameters and learning rates are used. The lower-level variable is updated via AdaGrad-Norm. Here in Equations (3) and (4), g x,t = ¯ ∇ f ( x t , y t ; ¯ ξ t ) , ˜ g x,t = ¯ ∇ f ( x t , y t ; ¯ ξ ′ t ) , and g y,t = ∇ y G ( x t , y t ; ζ t ) , where ¯ ∇ f ( x, y ; ¯ ξ ) denotes the Neumann series approximation (see Appendix E.1 for further details), with ¯ ξ t , ¯ ξ ′ t being independent samples.

Theorem 4.2. Under Assumptions 3.3 and 3.4 and the parameter choices in Equations (3) and (4) , for any δ ∈ (0 , 1 / 7) , choose N ≥ 3 log T 2 log(1 / (1 -µ g /l g, 1 )) , it holds with probability at least 1 -7 δ that

<!-- formula-not-decoded -->

where C b = ˜ O ( κ 4 σ ) , D , and ¯ σ ϕ are defined in Equations (24) , (44) and (46) , respectively.

Remark: Theorem 4.2 shows that Ada-Bio achieves a sharp rate of ˜ O ((¯ σ 2 ϕ + σ 2 g, 1 ) 1 / 4 /T 1 / 4 ) in the stochastic setting, where all noise terms introduced in Assumption 3.4 are positive. Moreover, it is obvious that Ada-Bio implicitly adapts to the noise level; in the noiseless case (where all noise parameters in Assumption 3.4 vanish), Ada-Bio automatically recovers the near-optimal ˜ O (1 / √ T ) rate. To the best of our knowledge, Theorem 4.2 provides the first sharp and adaptive convergence guarantee for stochastic bilevel optimization without any prior knowledge of the noise parameters specified in Assumption 3.4. In fact, we only require the knowledge of µ g , l g, 1 and T due to the construction of Neumann series. See detailed proof of Theorem 4.2 in Appendix E. A comparison of adaptive methods for bilevel optimization is also presented in Table 2.

## 5 Theoretical Analysis

In this section, we provide the convergence analysis for Algorithms 1 and 2 with the adaptive parameter choices in Equations (3) and (4). We begin in Section 5.1 by analyzing an adaptive version of normalized SGD with momentum (Algorithm 3) in the nonconvex stochastic (single-level) optimization setting, where we establish a convergence rate of ˜ O (1 / √ T + √ ¯ σ/T 1 / 4 ) , where ¯ σ is an upper bound on the stochastic gradient noise. We then extend this novel framework for the upper-level analysis in both minimax and bilevel optimization, combining it with a generalized AdaGrad-Norm analysis in the (strongly) convex case [2] under time shift for the lower-level variables, presented in Section 5.2. Due to space limitations, we defer the full proofs to Appendices C to E.

## 5.1 Adaptive Normalized SGD with Momentum

With a slight abuse of notation, we consider minimizing an objective function f ( x ) = E [ F ( x ; ξ )] . We start with analyzing adaptive normalized SGD with momentum presented in Algorithm 3, where

1 G denotes the upper bound on the stochastic gradient norm, and α, β satisfy 0 &lt; β &lt; α &lt; 1 .

## Algorithm 3 Adaptive Normalized SGD with Momentum (Ada-NSGDM)

- 1: Input: x 1 , m 1 = ∇ F ( x 1 ; ξ 1 )
- 2: for t = 1 , . . . , T do
- 3: m t = β t m t -1 +(1 -β t ) g t
- 4: x t +1 = x t -η t m t ∥ m t ∥
- 5: end for

g t = ∇ F ( x t ; ξ t ) . This algorithm builds on the method introduced by [11], with the key difference being that we incorporate both an adaptive momentum parameter β t and an adaptive stepsize η t , each of which varies across iterations. In particular, let α t = 1 -β t and we set α t and η t as

<!-- formula-not-decoded -->

where ˜ g t = ∇ F ( x t ; ξ ′ t ) and ξ t , ξ ′ t are independent samples. We will make the following assumptions. Assumption 5.1. The objective function f is L -smooth.

Assumption 5.2. The gradient oracle is unbiased, i.e., E [ ∇ F ( x ; ξ ) | x ] = ∇ f ( x ) , and with probability one, satisfies ¯ σ ≤ ∥∇ F ( x ; ξ ) -∇ f ( x ) ∥ ≤ ¯ σ .

Before proceeding, we introduce the definition of κ σ and t 0 , which will be frequently used throughout the subsequent analysis. Specifically, we define (with the convention 0 / 0 := 1 )

<!-- formula-not-decoded -->

where A T ( · ) and B T ( · ) are logarithmic factors (double-log in T ) defined in Lemma A.1.

We now present the main lemmas necessary to establish Theorem 5.6. All of these lemmas rely on Assumptions 5.1 and 5.2, unless explicitly stated otherwise. The full proof of these lemmas are deferred to Appendix B. The following lemma is a standard recursion for the momentum deviation.

Lemma5.3. Define ˆ ϵ t = m t -∇ f ( x t ) and ϵ t = g t -∇ f ( x t ) . Further, let S t = ∇ f ( x t -1 ) -∇ f ( x t ) . For all t ≥ 1 , it holds that

<!-- formula-not-decoded -->

In order to obtain a high probability bound for ∥ ˆ ϵ t ∥ , we need the following technical lemma, which leverages the concentration bound introduced in [55, Lemma 2.4] and tools from linear programming (see Lemma F.5 in Appendix F) to resolve the difficulties arising from statistical dependency among α t , β t , and ϵ t .

Lemma5.4. Let 0 ≤ ¯ α t ≤ α t ≤ ¯ α t and 0 ≤ ¯ β t ≤ β t ≤ ¯ β t , where ¯ α t , ¯ α t , ¯ β t , and ¯ β t are independent of F t . Then with probability at least 1 -2 δ , it holds for all t ≤ T that

<!-- formula-not-decoded -->

Next, we provide high-probability lower and upper bounds for α t and β t , which help us to derive tight upper bound for the right-hand side of Equation (7) (see Lemma G.2 in Appendix G). Our analysis relies on the martingale technique developed by [6], which uses an empirical Bernstein concentration bound introduced by [35]. Recall the definition of t 0 as in Equation (6). Lemma 5.5 indicates that α t and β t reliably approximate the optimal momentum parameter settings after t 0 iterations: they are both upper- and lower-bounded by quantities of the same order, even without prior knowledge of the noise level σ .

Lemma 5.5. With probability at least 1 -δ , for all t ≤ T ,

<!-- formula-not-decoded -->

Now we are ready the present our main theorem regarding Algorithm 3.

Theorem 5.6. Under Assumptions 5.1 and 5.2 and the parameter choices in Equation (5) , for any δ ∈ (0 , 1 / 3) , it holds with probability at least 1 -3 δ that

<!-- formula-not-decoded -->

where C = ˜ O ( κ 4 σ ) is defined in Equation (19) .

Remark: To our knowledge, this is the first adaptive convergence guarantee for normalized SGD with momentum. Algorithm 3 achieves a rate of ˜ O (1 /T 1 / 4 ) in the stochastic setting and ˜ O (1 / √ T ) in the deterministic setting. This rate-interpolation occurs automatically without requiring any prior knowledge of problem-dependent parameters. We emphasize that Theorem 5.6 builds a general analytical framework for proving Theorems 4.1 and 4.2. See Appendix B for detailed proofs.

## 5.2 Proof Sketch of Theorems 4.1 and 4.2

In this section, we present a unified lower-level analysis applicable to both minimax and bilevel optimization. Recall from Sections 1 and 3 that the bilevel optimization problem (2) reduces to the minimax optimization problem (1) when g = -f . Therefore, we analyze line 5 of Algorithm 1 using the function g and stochastic gradient descent (instead of the original f and stochastic gradient ascent): y t +1 = y t -η y,t g y,t , where g y,t = ∇ y G ( x t , y t ; ξ t ) = -∇ y F ( x t , y t ; ξ t ) . Note that the following lemma (Lemma 5.7) as well as Lemmas C.1 and C.2 in Appendix C are applicable to the proofs of both Theorems 4.1 and 4.2.

The following lemma provides high-probability guarantees for the lower-level estimation error, which are crucial for controlling and bounding the (hyper)gradient bias. The core of our result generalizes the AdaGrad-Norm analysis developed for convex settings by [2], accommodating iteration-dependent shifts induced by the upper-level variable and incorporating our novel adaptive parameter choices as detailed in Equations (3) and (4) and Sections 4.2 and 4.3.

Lemma 5.7. With probability at least 1 -4 δ , for all t ≤ T +1 , ¯ d t := max k ≤ t ∥ y k -y ∗ k ∥ ≤ D , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where D is defined in Equation (24) , and σ = σ y for Algorithm 1 and σ = σ g, 1 for Algorithm 2.

Combining the upper-level analysis framework introduced in Section 5.1 with the lower-level estimation error bounds (i.e., bounds on the gradient/hypergradient estimation bias) established in Lemma 5.7, we can prove Theorems 4.1 and 4.2 similarly to how we derived Theorem 5.6. The complete proofs are deferred to Appendices C to E.

## 6 Experiments

In this section, we empirically evaluate our proposed algorithms on three tasks, including synthetic test functions (Section 6.1), deep AUC maximization (Section 6.2), and hyperparameter optimization (Appendix K). In addition, we further test the robustness of our algorithms by varying several key parameters (e.g., initial learning rates, initial momentum parameter), which is included in Section 6.3. The code is available at https://github.com/MingruiLiu-ML-Lab/ adaptive-hierarchical-optimization .

## 6.1 Synthetic Experiments

We conduct synthetic experiments on a simple one-dimensional function f ( x, y ) = cos x + xy -1 2 y 2 , which satisfies the nonconvex-strongly-concave minimax optimization setting. It is straightforward to

Figure 1: Synthetic experiments on a 1-dimensional function for minimax optimization.

<!-- image -->

Figure 2: 2-layer Transformer for deep AUC maximization on imbalanced Sentiment140 dataset.

<!-- image -->

verify that y ∗ ( x ) = x , Φ( x ) = f ( x, y ∗ ( x )) = cos x + 1 2 x 2 , and ∇ Φ( x ) = x -sin x . To simulate stochastic gradients, we add Gaussian noise sampled from N (0 , σ 2 ) to the ground-truth gradients. As demonstrated in Figure 1, our proposed method, Ada-Minimax , consistently outperforms TiAda [47] across various noise magnitudes. These results clearly illustrate our algorithm's adaptivity to noise levels: specifically, as the noise magnitude decreases, our algorithm automatically achieves faster convergence. Notably, under high-noise regimes (e.g., σ = 100 ), TiAda fails to converge even after extensive parameter tuning, whereas our algorithm successfully converges. The hyperparameter settings are included in Appendix I.

## 6.2 Deep AUC Maximization

The Area Under the ROC Curve (AUC) is a performance measure of classifiers [31, 32], which is widely used in the imbalanced data classification setting. Deep AUC Maximization (DAM) [79, 76] is a new paradigm for learning a deep neural network by maximizing the AUC score of the model on a dataset. Recent studies [75, 76, 53] have shown great success of deep AUC maximization in various domains (e.g., medical image classification and drug discovery). Following [75, 54, 53], AUC maximization can be formulated as a minimax problem,

<!-- formula-not-decoded -->

where F ( w , a, b, α ; z ) = (1 -p )( h ( w ; x ) -a ) 2 I [ y =1] + p ( h ( w ; x ) -b ) 2 I [ y = -1] + 2(1 + α )( ph ( w ; x ) I [ y = -1] -(1 -p ) h ( w ; x ) I [ y =1] ) -p (1 -p ) α 2 , w is the parameter of a deep neural network (e.g., a two-layer transformer as the predictive model), h ( w ; x ) is the score function parameterized by w with the input data x , ξ = ( x , y ) is a random sample from training set D with input x and a binary label y ∈ {-1 , 1 } . The imbalanced ratio p is the proportion of the positive samples in the training set. Therefore, ( w , a, b ) and α are primal and dual variables respectively.

To verify the effectiveness of our proposed Algorithm 1, we run a practical variant (refer to Appendix J) of our algorithm in deep AUC maximization experiments on imbalanced text classification, and compare with other minimax baselines, including SGDA [50], PDSM [30]), and an adaptive minimax algorithm TiAda [47]. We first construct the imbalanced binary classification dataset Sentiment140 [23] (under Creative Commons Attribution 4.0 License). The practical variant of our algorithm replaces the term ∑ t k =1 ∥ g x,k -˜ g x,k ∥ 2 in Equation (3) with ∑ t k =1 ∥ g x,k -g x,k -1 ∥ 2 , where g x,k -1 denotes the gradient of x computed at the previous iteration (i.e., ( k -1) -th iteration). Additionally, we modify the step size from η x,t = η x √ α ′ t / √ t to η x,t = η x √ α ′ t / √ T (note that this change does not affect the convergence of Algorithm 1). In this subsection, with a slight abuse of notation, we use η x to denote η x / √ T . Following the data setting in [76], we randomly remove

Figure 3: Robustness of hyperparameters.

<!-- image -->

positive samples (labeled as 1) from the training set until the proportion of positive samples is exactly 0 . 9 (i.e., p = 0 . 9 ). In the experiment, we adopt a two-layer transformer as the classifier with the hidden size of 4096 and the output dimension of 2. The hyperparameter settings of each baseline and experimental details are described in Appendix J. The comparison results of training and test curve over 50 epochs are shown in Figure 2. From subfigures (a) and (b), our algorithm Ada-Minimax shows 20% higher training AUC and 2% higher test AUC than the best compared algorithm PDSM. From running time curve (c) and (d), our algorithm demonstrates the fastest convergence rate than other baselines.

## 6.3 Hyperparameter Robustness Analysis

We investigate the robustness of our method to the hyperparameters α , η y , η x , and γ by varying each parameter independently while keeping others fixed, as shown in Figure 3. Specifically, Figure 3(a) indicates that changing α within the range [0 . 1 , 2 . 0] has minimal impact on convergence speed and final AUC performance. In Figure 3(b), varying η y between 0 . 001 and 0 . 05 yield nearly overlapping curves after the initial training stage. Similarly, Figure 3(c) shows that varying η x from 0 . 001 to 0 . 01 affects only early-stage training dynamics without compromising the final performance; however, increasing η x to 0 . 05 results in a noticeable decline in the final training AUC. Lastly, Figure 3(d) illustrates that the algorithm maintains consistent performance across a wide range of γ values [0 . 01 , 2 . 0] . Therefore, these results demonstrate that our algorithm exhibits strong robustness across broad ranges of these hyperparameters, significantly reducing the time required for hyperparameter tuning in practice.

## 7 Conclusion

We introduced two novel adaptive algorithms for nonconvex-strongly-concave minimax optimization and nonconvex-strongly-convex bilevel optimization. Both algorithms achieve sharp and adaptive convergence rates: they automatically adapt to unknown variance in stochastic gradient estimates. Our approach leverages the momentum normalization framework along with novel adaptive schemes for jointly setting the momentum parameter and the learning rate. Experimental results validate and support our theoretical analyses. One limitation of our work is the assumption that the stochastic gradient noise is lower-bounded. In future work, we aim to remove this assumption while maintaining the sharp convergence guarantees.

## Acknowledgements

We would like to thank the anonymous reviewers for their helpful comments. We would like to thank Francesco Orabona and El Mehdi Saad for helpful discussions about the concentration inequalities. This work has been supported by the Presidential Scholarship, the ORIEI seed funding, and the IDIA P3 fellowship from George Mason University, and NSF award #2436217, #2425687. The Computations were run on Hopper, a research computing cluster provided by the Office of Research Computing at George Mason University (URL: https://orc.gmu.edu).

## References

- [1] Kimon Antonakopoulos, Shoham Sabach, Luca Viano, Mingyi Hong, and Volkan Cevher. Adaptive bilevel optimization, 2024.
- [2] Amit Attia and Tomer Koren. Sgd with adagrad stepsizes: full adaptivity with high probability to unknown parameters, unbounded gradients and affine variance. In International Conference on Machine Learning , pages 1147-1171. PMLR, 2023.
- [3] Dimitris Bertsimas and John N Tsitsiklis. Introduction to linear optimization , volume 6. Athena scientific Belmont, MA, 1997.
- [4] Radu Ioan Bo¸ t and Axel Böhm. Alternating proximal-gradient steps for (stochastic) nonconvexconcave minimax problems. SIAM Journal on Optimization , 33(3):1884-1913, 2023.
- [5] Jerome Bracken and James T McGill. Mathematical programs with optimization problems in the constraints. Operations research , 21(1):37-44, 1973.
- [6] Yair Carmon and Oliver Hinder. Making sgd parameter-free. In Conference on Learning Theory , pages 2360-2389. PMLR, 2022.
- [7] Lesi Chen, Jing Xu, and Jingzhao Zhang. On bilevel optimization without lower-level strong convexity. arXiv preprint arXiv:2301.00712 , 2023.
- [8] Tianyi Chen, Yuejiao Sun, Quan Xiao, and Wotao Yin. A single-timescale method for stochastic bilevel optimization. In International Conference on Artificial Intelligence and Statistics , pages 2466-2488. PMLR, 2022.
- [9] Tianyi Chen, Yuejiao Sun, and Wotao Yin. Closing the gap: Tighter analysis of alternating stochastic gradient methods for bilevel problems. Advances in Neural Information Processing Systems , 34:25294-25307, 2021.
- [10] Xuxing Chen, Tesi Xiao, and Krishnakumar Balasubramanian. Optimal algorithms for stochastic bilevel optimization under relaxed smoothness conditions. arXiv preprint arXiv:2306.12067 , 2023.
- [11] Ashok Cutkosky and Harsh Mehta. Momentum improves normalized sgd. In International Conference on Machine Learning , pages 2260-2268. PMLR, 2020.
- [12] Mathieu Dagréou, Pierre Ablin, Samuel Vaiter, and Thomas Moreau. A framework for bilevel optimization that enables stochastic and global variance reduction algorithms. Advances in Neural Information Processing Systems , 35:26698-26710, 2022.
- [13] Stephan Dempe. Foundations of bilevel programming . Springer Science &amp; Business Media, 2002.
- [14] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research , 12(Jul):2121-2159, 2011.
- [15] Chen Fan, Gaspard Choné-Ducasse, Mark Schmidt, and Christos Thrampoulidis. Bisls/sps: Auto-tune step sizes for stable bi-level optimization. Advances in Neural Information Processing Systems , 36:50144-50172, 2023.
- [16] Matthew Faw, Litu Rout, Constantine Caramanis, and Sanjay Shakkottai. Beyond uniform smoothness: A stopped analysis of adaptive sgd. arXiv preprint arXiv:2302.06570 , 2023.
- [17] Matthew Faw, Isidoros Tziotis, Constantine Caramanis, Aryan Mokhtari, Sanjay Shakkottai, and Rachel Ward. The power of adaptivity in sgd: Self-tuning step sizes with unbounded gradients and affine variance. In Conference on Learning Theory , pages 313-355. PMLR, 2022.
- [18] Matthias Feurer and Frank Hutter. Hyperparameter optimization. Automated machine learning: Methods, systems, challenges , pages 3-33, 2019.
- [19] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning , pages 1126-1135. PMLR, 2017.

- [20] Luca Franceschi, Michele Donini, Paolo Frasconi, and Massimiliano Pontil. Forward and reverse gradient-based hyperparameter optimization. In International Conference on Machine Learning (ICML) , pages 1165-1173, 2017.
- [21] Rong Ge, Furong Huang, Chi Jin, and Yang Yuan. Escaping from saddle points-online stochastic gradient for tensor decomposition. In Conference on learning theory , pages 797-842. PMLR, 2015.
- [22] Saeed Ghadimi and Mengdi Wang. Approximation methods for bilevel programming. arXiv preprint arXiv:1802.02246 , 2018.
- [23] Alec Go, Richa Bhayani, and Lei Huang. Twitter sentiment classification using distant supervision. CS224N project report, Stanford , 1(12):2009, 2009.
- [24] Xiaochuan Gong, Jie Hao, and Mingrui Liu. An accelerated algorithm for stochastic bilevel optimization under unbounded smoothness. arXiv preprint arXiv:2409.19212 , 2024.
- [25] Xiaochuan Gong, Jie Hao, and Mingrui Liu. A nearly optimal single loop algorithm for stochastic bilevel optimization under unbounded smoothness. In Forty-first International Conference on Machine Learning , 2024.
- [26] Xiaochuan Gong, Jie Hao, and Mingrui Liu. On the convergence of adam-type algorithm for bilevel optimization under unbounded smoothness. arXiv preprint arXiv:2503.03908 , 2025.
- [27] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572 , 2014.
- [28] Zhishuai Guo, Quanqi Hu, Lijun Zhang, and Tianbao Yang. Randomized stochastic variance-reduced methods for multi-task stochastic bilevel optimization. arXiv preprint arXiv:2105.02266 , 2021.
- [29] Zhishuai Guo, Yi Xu, Wotao Yin, Rong Jin, and Tianbao Yang. On stochastic moving-average estimators for non-convex optimization. arXiv preprint arXiv:2104.14840 , 2021.
- [30] Zhishuai Guo, Yi Xu, Wotao Yin, Rong Jin, and Tianbao Yang. Unified convergence analysis for adaptive optimization with moving average estimator. Machine Learning , 114(4):1-51, 2025.
- [31] James A Hanley and Barbara J McNeil. The meaning and use of the area under a receiver operating characteristic (roc) curve. Radiology , 143(1):29-36, 1982.
- [32] James A Hanley and Barbara J McNeil. A method of comparing the areas under receiver operating characteristic curves derived from the same cases. Radiology , 148(3):839-843, 1983.
- [33] Jie Hao, Xiaochuan Gong, and Mingrui Liu. Bilevel optimization under unbounded smoothness: Anewalgorithm and convergence analysis. In The Twelfth International Conference on Learning Representations , 2024.
- [34] Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A two-timescale stochastic algorithm framework for bilevel optimization: Complexity analysis and application to actorcritic. SIAM Journal on Optimization , 33(1):147-180, 2023.
- [35] Steven R Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics , 49(2):1055-1080, 2021.
- [36] Quanqi Hu, Yongjian Zhong, and Tianbao Yang. Multi-block min-max bilevel optimization with applications in multi-task deep AUC maximization. arXiv preprint arXiv:2206.00260 , 2022.
- [37] Feihu Huang, Junyi Li, and Shangqian Gao. Biadam: Fast adaptive bilevel optimization methods. arXiv preprint arXiv:2106.11396 , 2021.

- [38] Feihu Huang, Xidong Wu, and Zhengmian Hu. Adagda: Faster adaptive gradient descent ascent methods for minimax optimization. In International Conference on Artificial Intelligence and Statistics , pages 2365-2389. PMLR, 2023.
- [39] Maor Ivgi, Oliver Hinder, and Yair Carmon. Dog is sgd's best friend: A parameter-free dynamic step size schedule. In International Conference on Machine Learning , pages 14465-14499. PMLR, 2023.
- [40] Kaiyi Ji, Junjie Yang, and Yingbin Liang. Bilevel optimization: Convergence analysis and enhanced design. In International conference on machine learning , pages 4882-4892. PMLR, 2021.
- [41] Anatoli Juditsky, Arkadi Nemirovski, and Claire Tauvel. Solving variational inequalities with stochastic mirror-prox algorithm. Stochastic Systems , 1(1):17-58, 2011.
- [42] Ali Kavis, Kfir Yehuda Levy, and Volkan Cevher. High probability bounds for a class of nonconvex algorithms with adagrad stepsize. arXiv preprint arXiv:2204.02833 , 2022.
- [43] Prashant Khanduri, Siliang Zeng, Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A near-optimal algorithm for stochastic bilevel optimization via double-momentum. Advances in neural information processing systems , 34:30271-30283, 2021.
- [44] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR) , 2014.
- [45] Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert D Nowak. A fully first-order method for stochastic bilevel optimization. In International Conference on Machine Learning , pages 18083-18113. PMLR, 2023.
- [46] Haochuan Li, Ali Jadbabaie, and Alexander Rakhlin. Convergence of adam under relaxed assumptions. arXiv preprint arXiv:2304.13972 , 2023.
- [47] Xiang Li, Junchi Yang, and Niao He. Tiada: A time-scale adaptive algorithm for nonconvex minimax optimization. arXiv preprint arXiv:2210.17478 , 2022.
- [48] Xiaoyu Li and Francesco Orabona. On the convergence of stochastic gradient descent with adaptive stepsizes. In The 22nd international conference on artificial intelligence and statistics , pages 983-992. PMLR, 2019.
- [49] Xin Li and Dan Roth. Learning question classifiers. In COLING 2002: The 19th International Conference on Computational Linguistics , 2002.
- [50] Tianyi Lin, Chi Jin, and Michael Jordan. On gradient descent ascent for nonconvex-concave minimax problems. In International conference on machine learning , pages 6083-6093. PMLR, 2020.
- [51] Mingrui Liu, Youssef Mroueh, Jerret Ross, Wei Zhang, Xiaodong Cui, Payel Das, and Tianbao Yang. Towards better understanding of adaptive gradient algorithms in generative adversarial nets. In International Conference on Learning Representations , 2020.
- [52] Mingrui Liu, Hassan Rafique, Qihang Lin, and Tianbao Yang. First-order convergence theory for weakly-convex-weakly-concave min-max problems. Journal of Machine Learning Research , 22(169):1-34, 2021.
- [53] Mingrui Liu, Zhuoning Yuan, Yiming Ying, and Tianbao Yang. Stochastic auc maximization with deep neural networks. International Conference on Learning Representations , 2020.
- [54] Mingrui Liu, Xiaoxuan Zhang, Zaiyi Chen, Xiaoyu Wang, and Tianbao Yang. Fast stochastic auc maximization with o (1/n)-convergence rate. In International Conference on Machine Learning , pages 3195-3203, 2018.
- [55] Zijian Liu, Srikanth Jagabathula, and Zhengyuan Zhou. Near-optimal non-convex stochastic optimization under generalized smoothness. arXiv preprint arXiv:2302.06032 , 2023.

- [56] Zijian Liu, Ta Duy Nguyen, Thien Hang Nguyen, Alina Ene, and Huy Nguyen. High probability convergence of stochastic gradient methods. In International Conference on Machine Learning , pages 21884-21914. PMLR, 2023.
- [57] Songtao Lu. Slm: A smoothed first-order lagrangian method for structured constrained nonconvex optimization. Advances in Neural Information Processing Systems , 36:80414-80454, 2023.
- [58] Zhaosong Lu and Sanyou Mei. First-order penalty methods for bilevel optimization. SIAM Journal on Optimization , 34(2):1937-1969, 2024.
- [59] Luo Luo, Haishan Ye, Zhichao Huang, and Tong Zhang. Stochastic recursive gradient descent ascent for stochastic nonconvex-strongly-concave minimax problems. Advances in Neural Information Processing Systems , 33:20566-20577, 2020.
- [60] H Brendan McMahan and Matthew Streeter. Adaptive bound optimization for online convex optimization. arXiv preprint arXiv:1002.4908 , 2010.
- [61] Arkadi Nemirovski. Prox-method with rate of convergence o (1/t) for variational inequalities with lipschitz continuous monotone operators and smooth convex-concave saddle point problems. SIAM Journal on Optimization , 15(1):229-251, 2004.
- [62] Arkadi Nemirovski, Anatoli Juditsky, Guanghui Lan, and Alexander Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on optimization , 19(4):15741609, 2009.
- [63] Yurii Nesterov. Dual extrapolation and its applications to solving variational inequalities and related problems. Mathematical Programming , 109(2):319-344, 2007.
- [64] Hassan Rafique, Mingrui Liu, Qihang Lin, and Tianbao Yang. Non-convex min-max optimization: Provable algorithms and applications in machine learning. Optimization Methods and Software , 2020.
- [65] Han Shen and Tianyi Chen. On penalty-based bilevel gradient descent method. arXiv preprint arXiv:2302.05185 , 2023.
- [66] Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop, coursera: Neural networks for machine learning. University of Toronto, Technical Report , 6, 2012.
- [67] Rachel Ward, Xiaoxia Wu, and Leon Bottou. Adagrad stepsizes: Sharp convergence over nonconvex landscapes. Journal of Machine Learning Research , 21(219):1-30, 2020.
- [68] Yan Yan, Yi Xu, Qihang Lin, Wei Liu, and Tianbao Yang. Optimal epoch stochastic gradient descent ascent methods for min-max optimization. Advances in Neural Information Processing Systems , 33:5789-5800, 2020.
- [69] Junchi Yang, Negar Kiyavash, and Niao He. Global convergence and variance reduction for a class of nonconvex-nonconcave minimax problems. Advances in Neural Information Processing Systems , 33:1153-1165, 2020.
- [70] Junchi Yang, Xiang Li, and Niao He. Nest your adaptive algorithm for parameter-agnostic nonconvex minimax optimization. Advances in Neural Information Processing Systems , 35:1120211216, 2022.
- [71] Junchi Yang, Antonio Orvieto, Aurelien Lucchi, and Niao He. Faster single-loop algorithms for minimax optimization without strong concavity. In International Conference on Artificial Intelligence and Statistics , pages 5485-5517. PMLR, 2022.
- [72] Junjie Yang, Kaiyi Ji, and Yingbin Liang. Provably faster algorithms for bilevel optimization. Advances in Neural Information Processing Systems , 34:13670-13682, 2021.
- [73] Yifan Yang, Hao Ban, Minhui Huang, Shiqian Ma, and Kaiyi Ji. Tuning-free bilevel optimization: New algorithms and convergence analysis. arXiv preprint arXiv:2410.05140 , 2024.

- [74] Zhenhuan Yang, Yan Lok Ko, Kush R Varshney, and Yiming Ying. Minimax AUC fairness: Efficient algorithm with provable convergence. arXiv preprint arXiv:2208.10451 , 2022.
- [75] Yiming Ying, Longyin Wen, and Siwei Lyu. Stochastic online auc maximization. In Advances in Neural Information Processing Systems , pages 451-459, 2016.
- [76] Zhuoning Yuan, Yan Yan, Milan Sonka, and Tianbao Yang. Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 3040-3049, 2021.
- [77] Bohang Zhang, Jikai Jin, Cong Fang, and Liwei Wang. Improved analysis of clipping algorithms for non-convex optimization. Advances in Neural Information Processing Systems , 2020.
- [78] Jingzhao Zhang, Tianxing He, Suvrit Sra, and Ali Jadbabaie. Why gradient clipping accelerates training: A theoretical justification for adaptivity. International Conference on Learning Representations , 2020.
- [79] Peilin Zhao, Rong Jin, Tianbao Yang, and Steven C Hoi. Online auc maximization. In Proceedings of the 28th international conference on machine learning (ICML-11) , pages 233240, 2011.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Every claim made in the abstract is specified a section of the paper, including algorithm design and analysis in Section 5 and experiments in Section 6.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We have discussed the limitations of our work in Section 7.

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

Justification: We provide both assumptions and proofs in Section 3 and Appendices B, D and E.

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

Justification: The experimental details are specified in Section 6 and Appendices J to L.

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

## Answer: [Yes]

Justification: The code and data are attached as supplementary material with instructions for reproducibility.

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

Justification: The experimental details are included in Section 6 and Appendices J to L. Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: We only run once due to limited computational budget and resource.

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

Justification: The hardware specification is described in Section 6.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have read and conformed to the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This paper presents work whose goal is to advance the field of Machine Learning from algorithmic and theoretical aspects. We do not see any direct paths to negative societal impacts.

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

Justification: Our paper does not involve the release of any data or models.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Our paper uses existing text classification datasets and are cited in Section 6 and their licenses are mentioned.

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

Justification: Our paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not involve crowdsourcing nor research with human subjects. Guidelines:

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

## Contents

| 1 Introduction                | 1 Introduction                                         | 1 Introduction                                                           |   1 |
|-------------------------------|--------------------------------------------------------|--------------------------------------------------------------------------|-----|
| 2 Related Work                | 2 Related Work                                         | 2 Related Work                                                           |   2 |
| 3 Preliminaries               | 3 Preliminaries                                        | 3 Preliminaries                                                          |   3 |
|                               | 3.1                                                    | Assumptions for Nonconvex-Strongly-Concave Minimax Optimization          |   3 |
|                               | 3.2                                                    | Assumptions for Nonconvex-Strongly-Convex Bilevel Optimization .         |   3 |
| 4 Algorithms and Main Results | 4 Algorithms and Main Results                          | 4 Algorithms and Main Results                                            |   4 |
|                               | 4.1                                                    | Main Challenges and Algorithm Design . . . . . . . . . . . . . . . .     |   4 |
|                               | 4.2                                                    | Adaptive Algorithm for Minimax Optimization . . . . . . . . . . . .      |   5 |
|                               | 4.3                                                    | Adaptive Algorithm for Bilevel Optimization . . . . . . . . . . . . .    |   5 |
| 5 Theoretical Analysis        | 5 Theoretical Analysis                                 | 5 Theoretical Analysis                                                   |   6 |
|                               | 5.1                                                    | Adaptive Normalized SGD with Momentum . . . . . . . . . . . . . .        |   6 |
|                               | 5.2                                                    | Proof Sketch of Theorems 4.1 and 4.2 . . . . . . . . . . . . . . . . .   |   8 |
| 6                             | Experiments                                            | Experiments                                                              |   8 |
|                               | 6.1                                                    | Synthetic Experiments . . . . . . . . . . . . . . . . . . . . . . . . .  |   8 |
|                               | 6.2                                                    | Deep AUC Maximization . . . . . . . . . . . . . . . . . . . . . . . .    |   9 |
|                               | 6.3                                                    | Hyperparameter Robustness Analysis . . . . . . . . . . . . . . . . .     |  10 |
| 7                             | Conclusion                                             | Conclusion                                                               |  10 |
| A                             | Martingale Concentration Bounds and Basic Inequalities | Martingale Concentration Bounds and Basic Inequalities                   |  24 |
| B                             | Proofs of Section 5.1                                  | Proofs of Section 5.1                                                    |  24 |
|                               | B.1                                                    | Technical Lemmas . . . . . . . . . . . . . . . . . . . . . . . . . . .   |  24 |
|                               | B.2                                                    | Proof of Theorem 5.6 . . . . . . . . . . . . . . . . . . . . . . . . . . |  27 |
| C                             | Proof of Section 5.2                                   | Proof of Section 5.2                                                     |  27 |
| C.1                           |                                                        | Proof of Lemma 5.7 . . . . . . . . . . . . . . . . . . . . . . . . . . . |  30 |
| D                             | Analysis of Algorithm 1                                | Analysis of Algorithm 1                                                  |  33 |
|                               | D.1                                                    | Technical Lemmas . . . . . . . . . . . . . . . . . . . . . . . . . . .   |  33 |
|                               | D.2                                                    | Proof of Theorem 4.1 . . . . . . . . . . . . . . . . . . . . . . . . . . |  34 |
| E                             | Analysis of Algorithm 2                                | Analysis of Algorithm 2                                                  |  35 |
|                               | E.1                                                    | Neumann Series . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |  35 |
|                               | E.2                                                    | Technical Lemmas . . . . . . . . . . . . . . . . . . . . . . . . . . .   |  36 |
|                               | E.3                                                    | Proof of Theorem 4.2 . . . . . . . . . . . . . . . . . . . . . . . . . . |  37 |
| F Linear Programming Basics   | F Linear Programming Basics                            | F Linear Programming Basics                                              |  38 |

- G Useful Algebraic Facts 41

- H Discussion on Existing Algorithms for Minimax Optimization 49

- I Experimental Settings for Synthetic Experiments 49

- J Experimental Settings for Deep AUC Maximization 49

- K Experiments for Hyperparameter Optimization 49

- L Experiments for Verifying Assumptions 50

## A Martingale Concentration Bounds and Basic Inequalities

Lemma A.1 ([6, Corollary 1]) . Let X t be adapted to F t such that | X t | ≤ 1 with probability 1 for all t . Then, for every δ ∈ (0 , 1) and any ˆ X t ∈ F t -1 such that | ˆ X t | ≤ 1 with probability 1,

<!-- formula-not-decoded -->

where A t ( δ ) = 16 log ( 60 log(6 t ) δ ) and B t ( δ ) = 16 log 2 ( 60 log(6 t ) δ ) .

Lemma A.2 ([55, Lemma 2.4]) . Suppose X 1 , . . . , X T is a martingale difference sequence adapted to a filtration F 1 , . . . , F T in a Hilbert space such that ∥ X t ∥ ≤ R t almost surely for some R t ∈ F t -1 . Then for any δ ∈ (0 , 1) , with probability at least 1 -δ , for any fixed t we have

<!-- formula-not-decoded -->

Proof of Lemma A.2. The proof concludes by setting R t ∈ F t -1 in [55, Lemma 2.4].

Lemma A.3 ([2, Lemma 4]) . Let g 1 , . . . , g T ∈ R d be an arbitrary sequence of vectors, and let G 0 &gt; 0 . For all t ≥ 1 , define

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

Lemma A.4 ([2, Lemma 6]) . For Ada-NSGDM (Algorithm 3) we have

<!-- formula-not-decoded -->

where ∆ 1 = f ( x 1 ) -f

## B Proofs of Section 5.1

## B.1 Technical Lemmas

Lemma5.3. Define ˆ ϵ t = m t -∇ f ( x t ) and ϵ t = g t -∇ f ( x t ) . Further, let S t = ∇ f ( x t -1 ) -∇ f ( x t ) . For all t ≥ 1 , it holds that

<!-- formula-not-decoded -->

Proof of Lemma 5.3. The proof follows from a straightforward calculation:

<!-- formula-not-decoded -->

Unrolling the recursion and using α t = 1 -β t yields the result.

Lemma5.4. Let 0 ≤ ¯ α t ≤ α t ≤ ¯ α t and 0 ≤ ¯ β t ≤ β t ≤ ¯ β t , where ¯ α t , ¯ α t , ¯ β t , and ¯ β t are independent of F t . Then with probability at least 1 -2 δ , it holds for all t ≤ T that

<!-- formula-not-decoded -->

Proof of Lemma 5.4. Define γ k,t := β ( k +1): t α k and I t = { ( i, j ) | 3 ≤ i ≤ t, 2 ≤ j &lt; i } . Then for all k ≤ t ,

<!-- formula-not-decoded -->

By Lemma F.5, there exists a set { b ∗ ij,t } ( i,j ) ∈I t with each b ∗ ij,t satisfying either b ∗ ij,t = ¯ γ i,t ¯ γ j,t or b ∗ ij,t = ¯ γ i,t ¯ γ j,t for every pair ( i, j ) , such that

<!-- formula-not-decoded -->

Applying Lemma A.2 with X i = 〈 ϵ i , ∑ i -1 j =2 b ∗ ij,t ϵ j 〉 and R i = ¯ σ ∥ ∥ ∥ ∑ i -1 j =2 b ∗ ij,t ϵ j ∥ ∥ ∥ ∈ F i -1 , and using a union bound over t , with probability at least 1 -δ , for all t ≤ T ,

<!-- formula-not-decoded -->

Applying Lemma A.2 again with X j = b ∗ ij,t ϵ j and R j = b ∗ ij,t ¯ σ ∈ R , and using a union bound over i , with probability at least 1 -δ , for all i ≤ T ,

<!-- formula-not-decoded -->

Combing Equations (13) and (14), with probability at least 1 -2 δ (via a union bound), for all t ≤ T ,

<!-- formula-not-decoded -->

where the second inequality uses b ∗ ij,t ≤ ¯ γ i,t ¯ γ j,t . Hence, with probability at least 1 -2 δ ,

<!-- formula-not-decoded -->

Plugging in the definition of ¯ γ k,t as in Equation (12) yields the result.

Lemma 5.5. With probability at least 1 -δ , for all t ≤ T ,

<!-- formula-not-decoded -->

Proof of Lemma 5.5. Consider the case 0 &lt; ¯ σ ≤ ¯ σ . By Assumption 5.2 and Young's inequality,

<!-- formula-not-decoded -->

We proceed to derive high probability lower bound for ∑ t k =1 ∥ g k -˜ g k ∥ 2 . Denote σ 2 t = E t -1 [ ∥ g t -∇ f ( x t ) ∥ 2 ] . Let Z t = ∥ g t -˜ g t ∥ 2 -2 σ 2 t , then { Z t } t ≥ 1 is a martingale difference sequence since

<!-- formula-not-decoded -->

Using Assumption 5.2 and Young's inequality again, we have

<!-- formula-not-decoded -->

This implies that

<!-- formula-not-decoded -->

where the last equality is due to σ t ≤ ¯ σ almost surely. Define X t = Z t / (4¯ σ 2 -2 σ 2 t ) , then | X t | ≤ 1 with probability 1. Applying Lemma A.1 with the X s we defined and ˆ X s = 0 , for any δ ∈ (0 , 1) , with probability at least 1 -δ , for all t ≤ T ,

<!-- formula-not-decoded -->

where the last inequality uses ∑ t k =1 X 2 k ≤ t since | X k | ≤ 1 . Recall the definition of t 0 and c 0 as in Equation (6) ( t 0 is the solution to the equation A T ( δ ) · t + B T ( δ ) = c 2 0 t 2 ), for all t ≥ t 0 ,

<!-- formula-not-decoded -->

¯

Then, expanding Equation (16) and using the above condition yields that, with probability at least 1 -δ , for all t 0 ≤ t ≤ T ,

<!-- formula-not-decoded -->

We conclude the proof by combining Equations (15) and (17) and noting that the results also hold for the case σ = ¯ σ = 0 .

¯

Lemma B.1 (Descent Lemma) . Under Assumptions 5.1 and 5.2, define ˆ ϵ t := m t -∇ f ( x t ) , then

<!-- formula-not-decoded -->

Further, define ∆ 1 := f ( x 1 ) -f ∗ , taking summation and rearranging we have

<!-- formula-not-decoded -->

## B.2 Proof of Theorem 5.6

Before proving Theorem 5.6, let us define (recall the definition of κ σ and t 0 in Equation (6), here κ σ = ¯ σ/ ¯ σ in single-level optimization)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 5.6. Under Assumptions 5.1 and 5.2 and the parameter choices in Equation (5) , for any δ ∈ (0 , 1 / 3) , it holds with probability at least 1 -3 δ that

<!-- formula-not-decoded -->

where C = ˜ O ( κ 4 σ ) is defined in Equation (19) .

Proof of Theorem 5.6. Without loss of generality, we assume t 0 is an integer (see definition in Equation (6)). By Lemmas 5.3 to 5.5, B.1 and G.2, with probability at least 1 -3 δ ,

<!-- formula-not-decoded -->

where the third inequality uses ∥ ˆ ϵ 1 ∥ = ∥ ϵ 1 ∥ ≤ ¯ σ and ∥ S k ∥ = ∥∇ f ( x k -1 ) -∇ f ( x k ) ∥ ≤ Lη k -1 , and the last inequality is due to the definition of C . Then, using η t ≥ η T for t ≤ T ,

<!-- formula-not-decoded -->

Therefore, with probability at least 1 -3 δ ,

<!-- formula-not-decoded -->

## C Proof of Section 5.2

The core of our result in this section generalizes the AdaGrad-Norm analysis developed for convex settings by [2], accommodating iteration-dependent shifts induced by the upper-level variable x t and

incorporating our novel adaptive parameter choices as detailed in Equations (3) and (4) and Sections 4.2 and 4.3. In particular, Lemmas C.1 and C.2 are direct applications of [2, Lemmas 15 and 16], whereas Lemma 5.7 extends [2, Lemma 13] to account for time shifts.

Recall from Sections 1 and 3 that the bilevel optimization problem (2) reduces to the minimax optimization problem (1) when g = -f . Therefore, we analyze line 5 of Algorithm 1 using the function g and stochastic gradient descent (instead of the original f and stochastic gradient ascent): y t +1 = y t -η y,t g y,t , where g y,t = ∇ y G ( x t , y t ; ξ t ) = -∇ y F ( x t , y t ; ξ t ) . Note that the following lemmas (Lemmas 5.7, C.1 and C.2) are applicable to the proofs of both Theorems 4.1 and 4.2.

Additional Notations. Let y ∗ t = y ∗ ( x t ) and d t = ∥ y t -y ∗ t ∥ . Define ¯ d t := max k ≤ t d t . In the proof below, we use a 'decorrelated step size' given by

<!-- formula-not-decoded -->

Lemma C.1. Let ¯ d ′ t = max { ¯ d t , η } . Then with probability at least 1 -2 δ , it holds that for all t ≤ T ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where A t ( · ) and B t ( · ) are defined in Lemma A.1, and σ = σ y for Algorithm 1 and σ = σ g, 1 for Algorithm 2.

Proof of Lemma C.1. In order to invoke Lemma A.1 we will replace y k -y ∗ k with a version which is scaled and projected to the unit ball. We denote a s = 2 s -1 ¯ d ′ 1 and s t = ⌈ log( ¯ d ′ t / ¯ d ′ 1 ) ⌉ +1 . Thus, ¯ d t ≤ ¯ d ′ t ≤ a s t ≤ 2 ¯ d ′ t . Since ∥ y k +1 -y ∗ k +1 ∥ ≤ ∥ y k -y ∗ k ∥ + η for all s , ¯ d t ≤ d 1 + η ( t -1) and 1 ≤ s t ≤ ⌈ log( t ) ⌉ +1 ≤ log(4 T ) . Defining the projection to the unit ball, Π 1 ( x ) = x/ max { 1 , ∥ x ∥} ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that

Thus,

<!-- formula-not-decoded -->

Let X ( s ) k be defined as

<!-- formula-not-decoded -->

for some s . Then X ( s ) k is a martingale difference sequence since

<!-- formula-not-decoded -->

Also note that X ( s ) k ≤ 1 with probability 1 . Using Lemma A.1 with the X ( s ) k we defined and ˆ X k = 0 , for any k and δ ′ ∈ (0 , 1) , with probability at least 1 -δ ′ , for all t ≤ T ,

<!-- formula-not-decoded -->

We can upper bound ( X ( s ) k ) 2 ,

<!-- formula-not-decoded -->

where the first inequality uses Cauchy-Schwarz inequality, the second inequality is due to ∥ Π 1 ( x ) ∥ ≤ 1 , and the last inequality uses ˆ η y,k ≤ η y,k -1 . Thus, returning to Equation (21) multiplied by η y, 0 σa s t , with probability at least 1 -δ ′ log(4 T ) (union bound for all 1 ≤ s ≤ ⌊ log(4 T ) ⌋ ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, replacing ˆ η y,k by ˆ η 2 y,k and η y, 0 σ by η 2 y, 0 σ in Equation (20), following the analysis above, and using η y,k -1 ≤ η y, 0 yields that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

We conclude by applying a union bound over the two events (Equations (22) and (23)).

Lemma C.2. With probability at least 1 -2 δ , for all 1 ≤ t ≤ T ,

<!-- formula-not-decoded -->

where σ = σ y for Algorithm 1 and σ = σ g, 1 for Algorithm 2.

Proof of Lemma C.2. The proof concludes by applying [2, Lemma 16] with η s -1 = η y,k -1 , ∇ f ( x s ) = ∇ y g ( x k , y k ) , g s = g y,k , σ = σ or σ = σ g, 1 , and η 0 = η y, 0 = η/γ .

## C.1 Proof of Lemma 5.7

Before proving Lemma 5.7, let us define

<!-- formula-not-decoded -->

where A t ( δ ) , B t ( δ ) , C 1 , C 2 are defined in Lemmas A.1, A.4 and C.2, respectively Lemma 5.7. With probability at least 1 -4 δ , for all t ≤ T +1 , ¯ d t := max k ≤ t ∥ y k -y ∗ k ∥ ≤ D , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where D is defined in Equation (24) , and σ = σ y for Algorithm 1 and σ = σ g, 1 for Algorithm 2.

Proof of Lemma 5.7. Rolling a single step of SGD,

<!-- formula-not-decoded -->

Since f ( x k , · ) is µ -strongly convex, then

<!-- formula-not-decoded -->

Hence,

∥ y k +1 -y ∗ k ∥ 2 ≤ (1 -2 µη y,k ) ∥ y k -y ∗ k ∥ 2 +2 η y,k ⟨∇ y g ( x k , y k ) -g y,k , y k -y ∗ k ⟩ + η 2 y,k ∥ g y,k ∥ 2 . By Young's inequality and Lemma D.1,

<!-- formula-not-decoded -->

Summing from k = 1 to t , applying Lemma A.4, and using η x,k ≤ η/ k ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Observe that

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Combing Equations (26) and (28) with Lemma C.1, with probability at least 1 -2 δ ,

<!-- formula-not-decoded -->

Similarly,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combing Equations (30) and (31) with Lemma C.1, with probability at least 1 -2 δ ,

<!-- formula-not-decoded -->

Hence, due to ( A ) ≤ 2( A 1 ) + 2( A 2 ) and Equations (29) and (32), with probability at least 1 -2 δ ,

<!-- formula-not-decoded -->

Using ab ≤ a 2 / 2 + b 2 / 2 ,

<!-- formula-not-decoded -->

Under a union bound with Lemma C.2, with probability at least 1 -4 δ ,

<!-- formula-not-decoded -->

Bounding ( B ) . By the definitions of η x,t and η y,t ,

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

Thus, returning to Equation (25) and using the definition of D , with probability at least 1 -4 δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We use induction to show that with probability at least 1 -4 δ , ¯ d 2 t ≤ D 2 for all 1 ≤ t ≤ T + 1 . Note that for t = 1 , ¯ d 2 1 = d 2 1 ≤ D 2 . Assume ¯ d k ≤ D 2 for all k ≤ t ≤ T ; then for k = t + 1 , d 2 t +1 ≤ ¯ d 2 t / 2 + D 2 / 2 ≤ D 2 due to Equation (35). Thus, ¯ d 2 t +1 = max { d 2 t +1 , ¯ d 2 t } ≤ D 2 .

We proceed to prove Equations (8) and (9). Rearranging Equation (25), using Equation (35) and ¯ d 2 t ≤ D 2 , with probability at least 1 -4 δ ,

<!-- formula-not-decoded -->

Using η y,k ≤ η y,t for k ≤ t ,

<!-- formula-not-decoded -->

where the second inequality uses Young's inequality and Assumption 3.1, and the last inequality is due to √ a + b ≤ √ a + √ b for a, b ≥ 0 . Solving the inequality gives

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

## D Analysis of Algorithm 1

## D.1 Technical Lemmas

Lemma D.1 ([50, Lemma 4.3]) . Under Assumption 3.1, y ∗ ( x ) is L/µ -Lipschitz and Φ( x ) is ( µ + L ) L/µ -smooth.

Lemma D.2. Define ˆ ϵ t = m t -∇ Φ( x t ) , ϵ B t = ∇ x f ( x t , y t ) -∇ Φ( x t ) , and ϵ t = g x,t -∇ x f ( x t , y t ) . Further, let S t = ∇ Φ( x t -1 ) -∇ Φ( x t ) . For all t ≥ 1 , it holds that

<!-- formula-not-decoded -->

Proof of Lemma D.2. The proof follows from a straightforward calculation:

<!-- formula-not-decoded -->

Unrolling the recursion and using α t = 1 -β t yields the result.

Lemma D.3 (Descent Lemma) . Under Assumptions 3.1 and 3.2, define ˆ ϵ t := m t -∇ Φ( x t ) , then

<!-- formula-not-decoded -->

Further, define ∆ 1 := Φ( x 1 ) -Φ ∗ , taking summation and rearranging we have

<!-- formula-not-decoded -->

## D.2 Proof of Theorem 4.1

Before proving Theorem 4.1, let us define (recall the definition of κ σ and t 0 in Equation (6), here κ σ = ¯ σ x / ¯ σ x in minimax optimization)

<!-- formula-not-decoded -->

Theorem 4.1. Under Assumptions 3.1 and 3.2 and the parameter choices in Equations (3) and (4) , let ¯ σ x = σ y , then for any δ ∈ (0 , 1 / 7) , it holds with probability at least 1 -7 δ that

<!-- formula-not-decoded -->

where C m = ˜ O ( κ 4 σ ) and D are defined in Equations (24) and (38) , respectively.

Proof of Theorem 4.1. Without loss of generality, we assume t 0 is an integer (see definition in Equation (6)). By Lemmas D.2, D.3, G.2 and G.3, with probability at least 1 -7 δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, using η x,t ≥ η x,T for t ≤ T ,

<!-- formula-not-decoded -->

Therefore, by Lemma 5.7, with probability at least 1 -7 δ ,

<!-- formula-not-decoded -->

Setting ¯ σ x = σ y completes the proof.

## E Analysis of Algorithm 2

## E.1 Neumann Series

For bilevel optimization problems with lower-level strong convexity, we estimate the hypergradient

<!-- formula-not-decoded -->

via the Neumann series approach [22, 40, 34, 43]:

<!-- formula-not-decoded -->

where the matrix H yy is defined by

<!-- formula-not-decoded -->

and the set of random variables ¯ ξ is defined as

<!-- formula-not-decoded -->

In addition, define the gradient approximation of Φ as

<!-- formula-not-decoded -->

## E.2 Technical Lemmas

Lemma E.1 ([22, Lemma 2.2]) . Under Assumptions 3.3 and 3.4, we have

<!-- formula-not-decoded -->

where the constants L f , L y , L F are defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma E.2 ([22, Lemma 3.2], [34, Lemma 1]) . Under Assumptions 3.3 and 3.4, we have

<!-- formula-not-decoded -->

Lemma E.3. Under Assumptions 3.3 and 3.4, we have

<!-- formula-not-decoded -->

Proof of Lemma E.3. By triangle inequality,

<!-- formula-not-decoded -->

By Assumptions 3.3 and 3.4 and Lemma E.2, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

and and

and

<!-- formula-not-decoded -->

Hence, using the definition of ¯ σ ϕ as in Equation (44) we obtain

̸

<!-- formula-not-decoded -->

̸

Lemma E.4. Define ˆ ϵ t = m t -∇ Φ( x t ) , ϵ B t = ¯ ∇ f ( x t , y t ) -∇ Φ( x t ) , ϵ N t = E t -1 [ g x,t ] -¯ ∇ f ( x t , y t ) , and ϵ t = g x,t -E t -1 [ g x,t ] . Further, let S t = ∇ Φ( x t -1 ) -∇ Φ( x t ) . For all t ≥ 1 , it holds that

<!-- formula-not-decoded -->

Proof of Lemma E.4. The proof follows from a straightforward calculation:

<!-- formula-not-decoded -->

Unrolling the recursion and using α t = 1 -β t yields the result.

Lemma E.5 (Descent Lemma) . Under Assumptions 3.3 and 3.4, define ˆ ϵ t := m t -∇ Φ( x t ) , then

<!-- formula-not-decoded -->

Further, define ∆ 1 := Φ( x 1 ) -Φ ∗ , taking summation and rearranging we have

<!-- formula-not-decoded -->

## E.3 Proof of Theorem 4.2

Before proving Theorem 4.2, let us define (recall the definition of κ σ , t 0 , and ¯ σ ϕ in Equations (6) and (44), here κ σ = ¯ σ ϕ / ¯ σ ϕ in bilevel optimization)

<!-- formula-not-decoded -->

Theorem 4.2. Under Assumptions 3.3 and 3.4 and the parameter choices in Equations (3) and (4) , for any δ ∈ (0 , 1 / 7) , choose N ≥ 3 log T 2 log(1 / (1 -µ g /l g, 1 )) , it holds with probability at least 1 -7 δ that

<!-- formula-not-decoded -->

where C b = ˜ O ( κ 4 σ ) , D , and ¯ σ ϕ are defined in Equations (24) , (44) and (46) , respectively.

Proof of Theorem 4.2. Without loss of generality, we assume t 0 is an integer (see definition in Equation (6)). By Lemmas E.4, E.5, G.2 and G.3, with probability at least 1 -7 δ ,

<!-- formula-not-decoded -->

Then, using η x,t ≥ η x,T for t ≤ T ,

<!-- formula-not-decoded -->

Therefore, by Lemma 5.7, with probability at least 1 -7 δ ,

<!-- formula-not-decoded -->

## F Linear Programming Basics

Definition F.1 (General Form of Linear Programming [3, Section 1.1]) . The linear programming problem can be written as

<!-- formula-not-decoded -->

Definition F.2 ([3, Definition 2.1]) . A polyhedron is a set that can be described in the form { x ∈ R n | Ax ≥ b } , where A ∈ R m × n is a matrix and b ∈ R n is a vector.

Definition F.3 ([3, Definition 2.6]) . Let P be a polyhedron. A vector x ∈ P is an extreme point of P if we cannot find two vectors y, z ∈ P , both different from x , a scalar λ ∈ [0 , 1] , such that x = λy +(1 -λ ) z .

Theorem F.4 ([3, Theorem 2.8]) . Consider the linear programming problem of minimizing c ⊤ x over a polyhedron P . Suppose that P has at least one extreme point. Then, either the optimal cost is equal to -∞ , or there exists an extreme point which is optimal.

Lemma F.5. Assume 0 ≤ ¯ α t ≤ α t ≤ ¯ α t and 0 ≤ ¯ β t ≤ β t ≤ ¯ β t . Further, let ϵ i ∈ R d and denote γ k,t := β ( k +1): t α k , ¯ γ k,t := ¯ β ( k +1): t ¯ α k , and ¯ γ k,t := ¯ β ( k +1): t ¯ α k . There exists a set { b ∗ ij,t } with each b ∗ ij,t satisfying either b ∗ ij,t = ¯ γ i,t ¯ γ j,t or b ∗ ij,t = ¯ γ i,t ¯ γ j,t for every pair ( i, j ) , such that

<!-- formula-not-decoded -->

Proof of Lemma F.5. Consider the following constrained optimization problem:

<!-- formula-not-decoded -->

A relaxed version of problem (48) is:

<!-- formula-not-decoded -->

¯

Moreover, problem (49) is equivalent to:

<!-- formula-not-decoded -->

¯

Now we proceed to verify that

- (a) A relaxed version of Equation (50), namely Equation (53), is a linear programming problem of minimizing c ⊤ t x t over a polyhedron P t for some c t , x t , P t ;
- (b) P t has at least one extreme point;
- (c) The optimal cost of Equation (50) is not equal to -∞ .

Fact (a). We first define a few notations. Define c ij , x ij,t , ¯ b ij,t , ¯ b ij,t , and the index set I t as

<!-- formula-not-decoded -->

Let c t , x, P t be defined as

<!-- formula-not-decoded -->

where A t , b t are defined as

<!-- formula-not-decoded -->

According to Definition F.1, the optimization problem in Equation (50) can be relaxed into the following linear programming formulation (with a potentially higher objective value):

<!-- formula-not-decoded -->

Fact (b). We will show that the set of extreme points of P t is

<!-- formula-not-decoded -->

( = ⇒ ) Let x t = [ b ∗ 32 ,t . . . b ∗ t ( t -1) ,t ] ⊤ ∈ S . Check that A t x t ≥ b t , thus x t ∈ P t . Assume there exists y, z ∈ P t (both different from x t ) and a scalar λ ∈ (0 , 1) , such that x t = λy +(1 -λ ) z . Note that at least one element of y differs from the corresponding element in x t , denote this element by y ij , where ( i, j ) ∈ I t . We consider the following two cases:

- If y ij &gt; x ij,t = ¯ b ij,t , then

<!-- formula-not-decoded -->

This implies that z / ∈ P t since A t z ≱ b t .

- If y ij &lt; x ij,t = ¯ b ij,t , then

<!-- formula-not-decoded -->

This implies that z / ∈ P t since A t z ≱ b t .

Therefore, z / ∈ P t in both cases. By Definition F.3, x t is an extreme point of P t .

̸

( ⇐ = ) Assume there exists some x t ∈ P t such that x t / ∈ S t . Then there must be at least one element of x t , denoted by x ij,t , satisfying x ij,t = ¯ b ij,t and x ij,t = ¯ b ij,t . Let y, z , and x t differ only in the ij -th element, and define y ij , z ij as

̸

<!-- formula-not-decoded -->

Then y, z ∈ P t since A t y ≥ b t and A t z ≥ b t . Note that x t = ( y + z ) / 2 , hence by Definition F.3, x t is not an extreme point of P t .

Fact (c). If t is finite, then

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining Fact (a), Fact (b), Fact (c), and using Theorem F.4, we know that there exists an extreme point x ∗ t ∈ S such that

<!-- formula-not-decoded -->

Therefore, for problem (53),

<!-- formula-not-decoded -->

We conclude the proof by noting that problem (53) is a relaxed version of problem (48).

## G Useful Algebraic Facts

Lemma G.1. Let p ∈ (0 , 1] and q ∈ (0 , 1) . Further, let a, b ∈ N ≥ 2 with a ≤ b , and c, c 1 , c 2 &gt; 0 . (a) We have

<!-- formula-not-decoded -->

(b) If p ≥ q and c 1 ≤ c 2 , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma G.1. We prove the results individually.

Lemma G.1(a). Using 1 -x ≤ exp( -x ) and the monotonicity of (1 + ct ) -q ,

<!-- formula-not-decoded -->

Lemma G.1(b). By Lemma G.1(a),

<!-- formula-not-decoded -->

Using the monotonicity of (1 + c 1 t ) -q/ 2 t -p exp ( -(1+ c 2 t ) 1 -q c 2 (1 -q ) ) and c 1 ≤ c 2 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We continue to bound term ( I ) . By partial integration and p ≥ q ,

<!-- formula-not-decoded -->

Rearranging it yields

<!-- formula-not-decoded -->

Thus, we obtain

<!-- formula-not-decoded -->

Lemma G.1(c). Using 1 -x ≤ exp( -x ) ,

<!-- formula-not-decoded -->

Using the monotonicity of (1 + c 2 k ) -q , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Due to c 1 ≤ c 2 and the monotonicity of ( 1+ c 2 t 1+ c 1 t ) p , we continue to bound

<!-- formula-not-decoded -->

Denote h ( t ) as

By simple calculation,

<!-- formula-not-decoded -->

Define t 1 as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that h ( t ) is monotonically decreasing for t ≤ t 1 and monotonically increasing for t ≥ t 1 . If t 1 ≤ a , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If t 1 ≥ b , then

<!-- formula-not-decoded -->

Therefore, based on the three cases above,

<!-- formula-not-decoded -->

Weproceed to upper bound the integral I ′ . By partial integration and ( p -q )(1+( a -1) c 2 ) q -1 ≤ 1 / 2 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Rearranging it yields

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Lemma G.2. For all t ≥ 1 , let α t , β t , η t , and κ σ be defined as in Equations (5) and (6) :

<!-- formula-not-decoded -->

Then with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma G.2. We prove the results individually. Without loss of generality, we assume t 0 is an integer. By Lemma 5.5, with probability at least 1 -δ , we have ¯ α t ≤ α t ≤ ¯ α t and β t ≤ β t ≤ ¯ β t .

¯ Lemma G.2(a). Consider the case where 0 &lt; ¯ σ ≤ ¯ σ . Apply Lemma G.1(b) with a = a ≥ t 0 , b = T , p = q = 1 / 2 , c 1 = ¯ σ 2 /α 2 , and c 2 = 4¯ σ 2 /α 2 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality uses (1 + x ) 1 / 4 ≤ 1 + x 1 / 4 , and the last inequality is due to a ≥ t 0 ≥ 2 . The bound also holds for the case where ¯ σ = ¯ σ = 0 .

Lemma G.2(b). Apply Lemma G.2(a) with a = t 0 ,

<!-- formula-not-decoded -->

Hence, using η t ≤ η/ √ t and β t ≤ 1 ,

<!-- formula-not-decoded -->

Lemma G.2(c). Consider the case where 0 &lt; ¯ σ ≤ ¯ σ . Apply Lemma G.1(c) with a = t 0 , b = t , p = 1 , q = 1 / 2 , c 1 = ¯ σ 2 /α 2 , and c 2 = 4¯ σ 2 /α 2 ,

<!-- formula-not-decoded -->

Using the definition of η t ,

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

where the first inequality uses η t ≤ η/ √ t , the second inequality is due to √ a + b ≤ √ a + √ b for a, b ≥ 0 , and the third inequality uses Lemma G.2(a).

For the case ¯ σ = ¯ σ = 0 , we have α t = 1 and β t = 0 , hence ∑ T t =1 η t √ ∑ t k =2 ¯ β 2 ( k +1): t ¯ α 2 k = 0 .

Lemma G.2(d). We have

<!-- formula-not-decoded -->

where the first inequality uses η t ≤ η/ √ t , the second inequality is due to η t ≤ η , and the last inequality uses Lemma G.2(a). We continue to bound the last term above:

<!-- formula-not-decoded -->

where the first inequality uses Lemma G.2(a), and the second inequality is due to η t = η √ α t / √ t ≤ η/ √ t . Therefore,

<!-- formula-not-decoded -->

Lemma G.2(e). By the definition of η t and the fact that α t ≤ 1 ,

<!-- formula-not-decoded -->

LemmaG.3. For all t ≥ 1 , let α t , β t , and η x,t be defined as in Equations (3) and (4) (see Sections 4.2 and 4.3), and let ϵ B t be defined as in Lemma D.2 for minimax optimization and in Lemma E.4 for bilevel optimization:

<!-- formula-not-decoded -->

Then with probability at least 1 -4 δ , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (with a slight abuse of notation for L ) L = L, ¯ σ u = ¯ σ x , ¯ σ u = ¯ σ x , σ l = σ y for Algorithm 1, and L = l g, 1 , ¯ σ u = ¯ σ ϕ , ¯ σ u = ¯ σ ϕ , σ l = σ g, 1 for Algorithm 2.

Proof of Lemma G.3. We consider the cases σ = ¯ σ = 0 and 0 &lt; σ ≤ ¯ σ separately.

¯ u u ¯ u u Case ¯ σ u = ¯ σ u = 0 . In this case,

<!-- formula-not-decoded -->

By Assumption 3.1, ∥ ϵ B t ∥ = ∥∇ x f ( x t , y t ) -∇ Φ( x t ) ∥ ≤ L ∥ y t -y ∗ t ∥ . Thus,

<!-- formula-not-decoded -->

Using Cauchy-Schwarz inequality and Equations (33) and (36), with probability at least 1 -4 δ ,

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Case 0 &lt; ¯ σ u ≤ ¯ σ u . By triangle inequality,

<!-- formula-not-decoded -->

Then with probability at least 1 -4 δ ,

<!-- formula-not-decoded -->

Swapping the order of summation for the last term, and applying Lemma G.2(a),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality uses t -3 / 4 -( t +1) -3 / 4 ≤ 3 t -7 / 4 / 4 , and the third inequality is due to Lemma 5.7. Therefore,

<!-- formula-not-decoded -->

Combining Equations (54) and (55), we obtain

<!-- formula-not-decoded -->

Lemma G.4. For all t ≥ 1 , let α t , β t , η x,t , and ϵ N t be defined as in Equations (3) and (4) and Lemma E.4 for bilevel optimization (see Section 4.3):

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma G.4. By triangle inequality and Lemma E.2,

<!-- formula-not-decoded -->

where the third inequality uses η x,t ≤ η/ √ t and α t , β t ≤ 1 .

## H Discussion on Existing Algorithms for Minimax Optimization

Among existing algorithms for nonconvex-strongly-concave minimax optimization, TiAda [47] is the only work that attempts to be noise-adaptive. However, their convergence guarantees in the stochastic setting depend only on upper bounds of the stochastic gradient norm and the function value (e.g., Assumption 3.4, 3.5, and Theorem 3.2 in [47]), rather than the actual noise level of stochastic gradients. Consequently, TiAda does not achieve optimal convergence in terms of the dependency on stochastic gradient variance.

## I Experimental Settings for Synthetic Experiments

For synthetic experiments, we tune hyperparameters for each baseline using a grid search and report their best results. Both the parameter α used in the momentum parameter estimate (3) and the base learning rates ( η x , η y ) are tuned within the set { 0 . 5 , 1 . 0 , 1 . 5 , 2 . 0 , 3 . 0 , 4 . 0 , 5 . 0 } . We use the following parameter choices for various noise magnitude: for σ = 0 , α = 2 . 0 , η x = 3 . 0 , η y = 3 . 0 for Ada-Minimax , and η x = 4 . 0 , η y = 4 . 0 for TiAda; for σ = 20 , α = 2 . 0 , η x = 1 . 5 , η y = 1 . 5 for Ada-Minimax , and η x = 2 . 0 , η y = 2 . 0 for TiAda; for σ = 50 , α = 3 . 0 , η x = 2 . 0 , η y = 2 . 0 for Ada-Minimax , and η x = 2 . 0 , η y = 2 . 0 for TiAda; for σ = 100 , α = 5 . 0 , η x = 3 . 0 , η y = 3 . 0 for Ada-Minimax , and η x = 2 . 5 , η y = 2 . 5 for TiAda. Other hyperparameters in TiAda are set to the default choices as suggested in [47].

## J Experimental Settings for Deep AUC Maximization

For a fair comparison, we tune hyperparameters for each baseline using a grid search and report their best results. The base learning rates ( η x , η y ) are tuned within the range of [0 . 001 , 0 . 1] . Specifically, we select ( η x , η y ) = (0 . 1 , 0 . 05) for SGDA, (0 . 01 , 0 . 1) for PDSM, (0 . 1 , 0 . 05) for TiAda, and (0 . 01 , 0 . 01) for Ada-Minimax. The exponential hyperparameters ( α, β ) for TiAda follow the original settings in their paper, i.e., (0 . 6 , 0 . 4) . For Ada-Minimax, the parameters ( α, γ ) are tuned within α ∈ { 0 . 1 , 0 . 5 , 1 . 0 , 2 . 0 } and γ ∈ { 0 . 01 , 0 . 1 , 1 . 0 , 2 . 0 } , resulting in the optimal choice ( α, γ ) = (0 . 5 , 0 . 1) .

## K Experiments for Hyperparameter Optimization

In this section, we consider hyperparameter optimization on the TREC text classification dataset [49], provided under the Creative Commons Attribution 4.0 License. We formulate the hyperparameter optimization problem as follows:

<!-- formula-not-decoded -->

Figure 4: Comparison of BERT model on hyperparameter optimization.

<!-- image -->

where L ( w ; ξ ) denotes the loss function, w represents model parameters, and λ is the regularization hyperparameter. Here, D tr and D val denote the training and validation datasets, respectively. In our experiments, we employ a BERT model with 4 self-attention layers, each comprising 4 attention heads, followed by a fully-connected layer with an output dimension of 6, corresponding to the six classification categories. The model is trained from scratch for 50 epochs. We compare our algorithm's training and test performance against the tuning-free bilevel optimization (TFBO) method proposed by [73]. For TFBO, we conduct a grid search to select optimal initial values for the upper-level learning rate α 0 , lower-level learning rate β 0 , and linear system learning rate φ within the range [1 . 0 × 10 -5 , 10 . 0] , and set them to { 0 . 01 , 0 . 1 , 0 . 1 } . For Ada-BiO, we similarly perform hyperparameter tuning over the parameters ( η x , η y , α, γ ) within the range [1 . 0 × 10 -5 , 1 . 0] , selecting the optimal values (1 . 0 × 10 -5 , 0 . 5 , 1 . 0 , 0 . 1) for evaluation.

The training and test accuracy curves are illustrated in Figure 4. TFBO fails to converge because it is originally designed for deterministic scenarios, rendering it ineffective for stochastic settings. In contrast, Ada-BiO demonstrates rapid convergence in terms of training accuracy and consistently achieves superior test performance.

## L Experiments for Verifying Assumptions

Weempirically verify Assumption 3.2(ii), which states that the noise of the stochastic gradient satisfies ¯ σ x ≤ ∥∇ x F ( x, y ; ξ ) - ∇ x f ( x, y ) ∥ ≤ ¯ σ x with ¯ σ x ≥ 0 . Specifically, following the experimental setup for deep AUC maximization described in Section 6.2, we compute the exact gradient ∇ x f ( x, y ) after each training epoch by averaging the gradients over the entire validation dataset with fixed model parameters and hyperparameters. Similarly, we compute the stochastic gradient ∇ x F ( x, y ; ξ ) , but using a randomly sampled mini-batch from the validation set. We observe that the empirical maximal and minimal noise levels are ¯ σ x = 210 . 71 and ¯ σ x = 0 . 21 , respectively, thus confirming that practical stochastic gradient noise is indeed bounded from both sides.