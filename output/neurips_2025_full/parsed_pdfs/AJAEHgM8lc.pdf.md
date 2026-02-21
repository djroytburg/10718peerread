## Conditional Gradient Methods with Standard LMO for Stochastic Simple Bilevel Optimization

Khanh-Hung Giang-Tran

Cornell University tg452@cornell.edu

Soroosh Shafiee Cornell University shafiee@cornell.edu

## Abstract

We propose efficient methods for solving stochastic simple bilevel optimization problems with convex inner levels, where the goal is to minimize an outer stochastic objective function subject to the solution set of an inner stochastic optimization problem. Existing methods often rely on costly projection or linear optimization oracles over complex sets, limiting their scalability. To overcome this, we propose an iteratively regularized conditional gradient approach that leverages linear optimization oracles exclusively over the base feasible set. Our proposed methods employ a vanishing regularization sequence that progressively emphasizes the inner problem while biasing towards desirable minimal outer objective solutions. In the one-sample stochastic setting and under standard convexity assumptions, we establish non-asymptotic convergence rates of ˜ O ( t -1 / 4 ) for both the outer and inner objectives. In the finite-sum setting with a mini-batch scheme, the corresponding rates become ˜ O ( t -1 / 2 ) . When the outer objective is nonconvex, we prove nonasymptotic convergence rates of ˜ O ( t -1 / 7 ) for both the outer and inner objectives in the one-sample stochastic setting, and ˜ O ( t -1 / 4 ) in the finite-sum setting. Experimental results on over-parametrized regression and dictionary learning tasks demonstrate the practical advantages of our approach over existing methods, confirming our theoretical findings.

## 1 Introduction

We consider stochastic simple bilevel optimization problems , where the goal is to minimize an outer stochastic objective function subject to the solution set of an inner stochastic convex optimization problem. Formally, the problem is defined as

<!-- formula-not-decoded -->

Here X ⊂ R d is a compact convex set, and both F and G are smooth with G is additionally convex, making the optimization problem (1) a possibly nonconvex problem with convex domain. This framework has broad applicability, including hyper-parameter optimization [16, 17, 39], metalearning [17, 42], reinforcement learning [32] and game theory [50]. In particular, problem (1) arises in learning applications where G is the prediction error of a model x on data, and F is an auxiliary objective (e.g., regularization, or a validation data set). The dependence on data is through the expectation operators and the random variables θ, ξ . The bilevel formulation provides a tuning-free alternative to the usual regularization approach where we optimize σF ( x ) + G ( x ) for a carefully tuned parameter σ .

Despite its wide applications in machine learning there are three primary challenges in solving (1). First, we do not have an explicit representation of the optimal set X opt in general, preventing us

## Nam Ho-Nguyen

The University of Sydney nam.ho-nguyen@sydney.edu.au

from using some common operations in optimization such as projection onto or linear optimization over X opt . An alternative is to reformulate (1) using its value function formulation

<!-- formula-not-decoded -->

However, this leads to our second challenge. Problem (2) is inherently ill-conditioned: by definition of G opt , there exists no x ∈ X such that G ( x ) &lt; G opt , hence (2) does not satisfy Slater's condition. As a result, the Lagrangian dual of (2) may not be solvable, complicating the use of standard primal-dual methods. In addition, we do not know G opt a priori, so an alternative is to approximate it with ¯ G ≤ G opt + ϵ g , and consider the constraint G ( x ) ≤ ¯ G instead. However, this approach does not solve the actual bilevel problem (1), and may still introduce numerical instability. The third challenge comes from the stochastic nature of the objectives. Since F and G are defined as expectations, their exact computation may be intractable when dealing with large-scale datasets. In single-level optimization, this is typically addressed through sampling-based methods that operate on mini-batches drawn from the distributions of θ and ξ instead of the actual distributions themselves. However, such stochastic approximations introduce noise into the optimization process, necessitating new techniques to control the noise and ensure convergence.

## 1.1 Related works

Simple bilevel optimization problems with nonconvex outer objectives This setting has been studied by Jiang et al. [29] in the deterministic case and by Cao et al. [4] in the stochastic case. These works propose conditional gradient methods that require performing linear optimization over intersections of the base domain X with a halfspace. In contrast, our approach removes this requirement by relying solely on linear optimization over X . We note that we do not address the more general case with both nonconvex inner and outer objectives considered in [5, 21, 24]. In the remainder of this section, we will focus on simple bilevel problems with convex inner and outer objectives.

Iterative regularization methods. Simple bilevel optimization extends Tikhonov regularization [52] beyond ℓ 2 penalties by considering a blended objective σF ( x )+ G ( x ) , where σ &gt; 0 modulates the trade-off between inner and outer objectives. As σ → 0 , the problem converges to a bilevel form that prioritizes the inner objective. Friedlander and Tseng [18] established fundamental conditions ensuring solution existence for σ &gt; 0 , forming the basis for iterative regularization , where σ is gradually reduced. This idea was first implemented by Cabot [3] in the unconstrained case X = R d using a positive decreasing sequence σ t , and later extended to general convex constraints by Dutta and Pandit [14]. While these methods use proximal steps, which can be computationally expensive, Solodov [49] proposed a more efficient gradient-based alternative. Helou and Simões [23] further generalized the framework to non-smooth settings via an ϵ -subgradient method. These works show asymptotic convergence but lack convergence rates. Amini and Yousefian [1] analyzed an iterative regularized projected gradient method and established a convergence rate of O ( t -(1 / 2 -b ) ) for the inner objective, assuming non-smooth F and G , compact X , and strong convexity of f . Kaushik and Yousefian [30] later removed the strong convexity assumption and proved rates of O ( t -b ) and O ( t -(1 / 2 -b ) ) for the inner and outer objectives, respectively, for any b ∈ (0 , 1 / 2) . Malitsky [35] studied an accelerated variant based on Tseng's method [53], achieving an o ( t -1 ) rate for the inner problem. More recently, Merchav et al. [37] proposed an accelerated proximal scheme with a O (log( t +1) /t ) rate under smoothness assumption and O ( t -2 ) and O (1 /t 2 -γ ) for inner and outer objectives, respectively, under a Hölderian error bound condition with modulus γ ∈ (1 , 2) . Our work also follows an iterative regularization approach, but avoids expensive projections or proximal operations by leveraging linear optimization oracles.

Projection-free methods. Several important applications have domains X where projection-based operations are inefficient, yet linear optimization over X is easier. To address the limitations of projection-based methods, recent work has focused on projection-free bilevel optimization via linear optimization oracles. Jiang et al. [29] approximated X opt by replacing the constraint G ( x ) ≤ G opt with a linear approximation, which removed the need for Slater's condition, and applying the conditional gradient method, refining the approximation at each step. They established O ( t -1 ) convergence rates for both objectives under smoothness and compactness assumptions in case f is convex. Under non-convex setting, they claimed the convergence rates of O ( t -1 / 2 ) to the stationary point of the problem. However, their method requires a pre-specified tolerance parameter ϵ g , and only

guarantees ( ϵ g / 2) -infeasibility. Doron and Shtern [12] proposed an alternative approach based on sublevel sets of the outer function F . Their method performs conditional gradient updates over sets of the form X ×{ x : F ( x ) ≤ α } , with a surrogate ˆ G t that is updated iteratively. They achieved rates of O ( t -1 ) for the inner and O ( t -1 / 2 ) for the outer objective under composite structure in G and an error bound on F . While linear optimization over X may be efficient, this is not always true for sets like X ∩ H , where H is a halfspace, or { F ( x ) ≤ α } unless F has a special structure. To address this, Giang-Tran et al. [20] introduced a conditional gradient-based iterative regularization scheme that only requires linear optimization over X , albeit with slower rates: O ( t -p ) and O ( t -(1 -p ) ) for the inner and outer problems, respectively, for any p ∈ (0 , 1) . We extend this framework to the stochastic setting, maintaining the same oracle-based reliance on linear optimization over X .

Stochastic methods. Stochastic bilevel optimization remains less explored. Jalilzadeh et al. [27] developed an iterative regularization-based stochastic extragradient algorithm, requiring O (1 /ϵ 4 f ) and O (1 /ϵ 4 g ) stochastic gradient queries to achieve ϵ -optimality for both objectives. Cao et al. [4] extended the projection-free framework of Jiang et al. [29] to the stochastic setting. Their methods achieve ˜ O ( t -1 ) convergence rate when the noise distributions have finite support, and ˜ O ( t -1 / 2 ) convergence rates under sub-Gaussian noise. However, their method requires linear optimization over intersections of X with a halfspace. In contrast, our work relaxes this requirement by relying solely on linear optimization over the base set X .

Other methods and bilevel problem classes. Several alternative approaches exist for simple bilevel optimization, including sequential averaging schemes [36, 44, 46], sublevel set-based methods [2, 6, 21, 58], accelerated algorithms [6, 7, 45, 54] and primal-dual strategies [47]. These methods are less related to our framework and do not address the stochastic setting. More general bilevel problems have seen considerable attention, particularly in Stackelberg game formulations where the upper and lower levels model leader-follower dynamics [8, 10, 19, 28, 31, 33, 34, 56]. Recent extensions include contextual bilevel optimization with exogenous variables [25, 51, 55] and problems with functional constraints at the lower level [30, 40]. Our focus in this paper, however, is restricted to the stochastic simple bilevel problems.

## 1.2 Contributions

Wepresent two iterative regularization methods for solving problem (1) in both stochastic and finitesum settings. The key idea is to introduce a decreasing regularization sequence { σ t } t ≥ 1 , and at each iteration t , perform a gradient-based update on the composite objective σ t F + G . As σ t decreases, the algorithm gradually shifts focus from F to G , thereby steering x t toward points in X opt . At the same time the σ t F term encourages x t towards a point in X opt that minimizes F . In the stochastic formulation (1), computing exact gradients of the upper and lower objectives is computationally expensive. To address this, we employ sample-based gradient estimators. Specifically, we use the STOchastic Recursive Momentum ( STORM ) estimator [9] in the one-sample stochastic setting, and the Stochastic Path-Integrated Differential EstimatoR ( SPIDER ) [15, 38] in the finite-sum setting.

The key contributions of this paper are summarized below.

- ⋄ One-sample stochastic setting : We develop a projection-free method for stochastic simple bilevel optimization that under convexity of F on the base set X achieves high-probability convergence rates of O ( t -1 / 4 √ log( dt 2 /δ )) for the outer-level problem and O ( t -1 / 4 √ log( d/δ )) for the inner-level problem, with probability at least 1 -δ . Without convexity of F , the method enjoys the non-asymptotic convergence rates of O ( t -1 7 log( dt 2 /δ )) for both the outer- and innerlevel objectives. Compared to the state-of-the-art approach by Cao et al. [4, Algorithm 1], which achieves the O ( t -1 / 2 √ log( td/δ )) rate under convex settings and O ( t -1 / 3 √ log( td/δ )) under nonconvex settings for both the outer and inner problems, our method offers several key advantages. First, it eliminates the need for optimization over intersections of the base domain with a halfspace, requiring only linear optimization over the original constraint set. Second, it removes the dependency on their pre-specified tolerance parameter ϵ g , and the associated computationally expensive initialization procedure to find an ϵ g -optimal starting point. Third, our algorithm enjoys anytime guarantees. Overall, the proposed algorithm is a simple, single-loop procedure that is significantly easier to implement than that of [4].

- ⋄ Finite-sum setting : For problems with n component functions, we establish convergence rates of O ( t -1 / 2 √ log( t 2 /δ )) (outer) and O ( t -1 / 2 √ log(1 /δ )) (inner) with probability 1 -δ . Without convexity of F , the method enjoys the non-asymptotic convergence rates of O ( t -1 4 log( dt 2 /δ )) for both the outer- and inner-level objectives. Although Cao et al. [4, Algorithm 2] achieves improved O ( t -1 log( t ) √ log( t/δ )) rates under convex settings and O ( t -1 / 2 √ log( t/δ )) rates under nonconvex settings for both objectives, it again relies on more computationally demanding halfspace-intersection oracles, requires an additional initialization step, depends on a fixed stepsize, and does not offer anytime guarantees.
- ⋄ Numerical validation : We demonstrate the practical efficiency of our methods on overparametrized regression and dictionary learning tasks, showing significant speedups over existing projection-based and projection-free approaches. The experiments validate both the convergence rates and computational advantages of our framework.

The supplementary material is organized in the appendices as follows. Appendix A establishes the preparatory lemmas used throughout our analysis. Appendices B and C contain the detailed proofs for the results presented in Section 2 and Section 3, respectively. Appendix D provides additional material, including analyses of the convergence rate coefficients as well as details of the numerical implementations in Section 4.

## 2 Algorithm and convergence results for the one-sample stochastic setting

In this section, we introduce the Iteratively Regularized Stochastic Conditional Gradient ( IR-SCG ) method, summarized in Algorithm 1. This approach is a conditional gradient method that leverages the STORM estimator to progressively reduce noise variance via momentum-based updates, assuming sub-Gaussian noise. While the use of the STORM estimator in projection-free methods was first explored in [60], our IR-SCG algorithm offers several key advantages. Notably, unlike the method of Cao et al. [4], IR-SCG is an anytime algorithm that does not require careful initialization, preset step sizes, or knowledge of the total number of iterations T in advance.

```
Data: Parameters { α t } t ≥ 0 ⊆ [0 , 1] , { σ t } t ≥ 0 ⊆ R ++ . Result: sequences { x t , z t } t ≥ 1 . Initialize x 0 ∈ X ; for t = 0 , 1 , 2 , . . . do if t = 0 then Compute ̂ ∇ F t := ∇ x f ( x t , θ t ) , ̂ ∇ G t := ∇ x g ( x t , ξ t ) else Compute ̂ ∇ F t := (1 -α t ) ̂ ∇ F t -1 + ∇ x f ( x t , θ t ) -(1 -α t ) ∇ x f ( x t -1 , θ t ) ̂ ∇ G t := (1 -α t ) ̂ ∇ G t -1 + ∇ x g ( x t , ξ t ) -(1 -α t ) ∇ x g ( x t -1 , ξ t ) Compute v t ∈ arg min v ∈ X { ( σ t ̂ ∇ F t + ̂ ∇ G t ) ⊤ v } x t +1 := x t + α t ( v t -x t ) S t +1 := ( t +2)( t +1) σ t +1 + ∑ i ∈ [ t +1] ( i +1) i ( σ i -1 -σ i ) z t +1 := ( t +2)( t +1) σ t +1 x t +1 + ∑ i ∈ [ t +1] ( i +1) i ( σ i -1 -σ i ) x i S t +1
```

Algorithm 1: Iteratively Regularized Stochastic Conditional Gradient ( IR-SCG ) Algorithm.

To establish convergence rates for Algorithm 1, we impose the following standard assumptions on the simple bilevel problem (1). In the following, all norms refer to the Euclidean norm.

Assumption 1. The following hold.

- (a) X ⊆ R d is convex and compact with diameter D &lt; ∞ , i.e., ∥ x -y ∥ ≤ D for any x, y ∈ X .
- (b) Functions F, G are convex over X and continuously differentiable on an open neighborhood of X .
- (c) For any θ , f ( · , θ ) is L f -smooth on an open neighborhood of X , i.e., it is continuously differentiable and its derivative is L f -Lipschitz:

<!-- formula-not-decoded -->

for any x, y ∈ X .

- (d) For any ξ , g ( · , ξ ) is L g -smooth on an open neighborhood of X .
- (e) For any x ∈ X , the stochastic gradients noise is sub-Gaussian, i.e., there exists some σ f , σ g &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

First, we present the result for the case that F is convex.

Theorem 1. Let { z t } t ≥ 1 be the iterates generated by Algorithm 1 with α t = 2 / ( t + 2) for any t ≥ 0 , and regularization parameters σ t := ς ( t +1) -p for some chosen ς &gt; 0 , p ∈ (0 , 1 / 2) . Under Assumption 1, for any t ≥ 1 , with probability at least 1 -δ , it (jointly) holds that

<!-- formula-not-decoded -->

Moreover, with probability 1 , we have

<!-- formula-not-decoded -->

Theorem 1 provides the formal convergence guarantees for the IR-SCG algorithm in the convex setting. If we set p = 1 / 4 , Algorithm 1 achieves an ϵ -level optimality gap for both the outer and inner problems with a sample complexity of O (1 /ϵ 4 ) .

We next extend our analysis to the more general case where F is not necessarily convex. In this setting, we require a different measure for stationarity . We define our stationarity measure using the Frank-Wolfe gap, which for a convex function F ( x ) = 0 if and only if x is optimal. This provides a natural generalization for the non-convex case. Accordingly, for all x ∈ X , we define the functions

<!-- formula-not-decoded -->

To avoid clutter, we present the result using the optimally tuned parameters of Algorithm 1.

Theorem 2. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 1 with stepsizes α t = ( t +1) -ω and regularization parameters σ t = ς ( t +1) -p for any t ≥ 0 , with p = 2 7 and ω = 6 7 . If Assumption 1 holds with the exception that F may be nonconvex, then with probability at least 1 -δ , for every t ≥ 0 , it (jointly) holds that

<!-- formula-not-decoded -->

Moreover, with probability 1 , we have

<!-- formula-not-decoded -->

## 3 Algorithm and convergence results for the finite-sum setting

In this section, we address the case where the expectations in (1) are finite sums over n components. We introduce the Iteratively Regularized Finite-Sum Conditional Gradient ( IR-FSCG ) method, summarized in Algorithm 2, where [ n ] := { 1 , . . . , n } . This approach is a conditional gradient method that leverages the SPIDER estimator, achieving variance reduction through periodic gradient recomputations with mini-batches of size q . While the SPIDER estimator has been previously integrated into projection-free methods [22, 48, 57, 59], here we show how it can be effectively adapted to our regularized setting. The primary advantage is that, unlike the method of Cao et al. [4], IR-FSCG preserves the desirable anytime properties of our approach while achieving a faster convergence rate that capitalizes on the finite-sum structure. Additionally, it does not require careful initialization, preset step sizes, or knowledge of the total number of iterations T in advance.

To establish its convergence, we again rely on Assumption 1, noting that condition Assumption 1(e) holds automatically in this finite-sum setting. First, we present the result when F is convex.

Theorem 3. Suppose F and G in (1) are expectations over uniform distributions on [ n ] . Let { z t } t ≥ 1 be the iterates generated by Algorithm 2 with α t = log( q ) /q for 0 ≤ t &lt; q , α t = 2 / ( t + 2) for any t ≥ q , and σ t := ς (max { t, q } + 1) -p for some chosen ς &gt; 0 , p ∈ (0 , 1) and S = q . Under Assumption 1, with probability at least 1 -δ , for any t ≥ q , it holds that

<!-- formula-not-decoded -->

where c ( t, q ) = max { 1 , q log( q ) /t } . Moreover, with probability 1 , we have

<!-- formula-not-decoded -->

## Algorithm 2: Iteratively Regularized Finite-Sum Conditional Gradient ( IR-FSCG ) Algorithm.

```
Data: Parameters { α t } t ≥ 0 ⊆ [0 , 1] , { σ t } t ≥ 0 ⊆ R ++ , S, q ∈ [ n ] . Result: sequences { x t , z t } t ≥ 1 . Initialize x 0 ∈ X ; for t = 0 , 1 , 2 , . . . do if t = 0 mod q then Compute ̂ ∇ F t := ∇ F ( x t ) , ̂ ∇ G t := ∇ G ( x t ) . else Draw S new i.i.d. samples S f = { θ 1 , . . . , θ S } , S g = { ξ 1 , . . . , ξ S } ; Compute ̂ ∇ F t := ̂ ∇ F t -1 + ∇ f S f ( x t ) -∇ f S f ( x t -1 ) ̂ ∇ G t := ̂ ∇ G t -1 + ∇ g S g ( x t ) -∇ g S g ( x t -1 ) . Compute v t ∈ arg min v ∈ X { ( σ t ̂ ∇ F t + ̂ ∇ G t ) ⊤ v } x t +1 := x t + α t ( v t -x t ) if t ≥ q then Compute S t +1 := ( t +2)( t +1) σ t +1 + ∑ i ∈ [ t +1] \ [ q ] ( i +1) i ( σ i -1 -σ i ) z t +1 := ( t +2)( t +1) σ t +1 x t +1 + ∑ i ∈ [ t +1] \ [ q ] ( i +1) i ( σ i -1 -σ i ) x i S t +1 .
```

## 4 Numerical results

We showcase the performance of our proposed algorithms in an over-parametrized regression and a dictionary learning problems. For performance comparison, we implement four algorithms: SBCGI [4, Algorithm 1], SBCGF [4, Algorithm 2], aR-IP-SeG [27], and the stochastic variant of the dynamic barrier gradient descent ( SDBGD ) [21]. All implementation details follow those in the corresponding papers and available in Appendix D. To ensure reproducibility, all source codes are made available at https://github.com/brucegiang/CG-StoBilvl .

## 4.1 Over-parameterized regression

We first consider a simple bilevel optimization problem with a convex outer-level objective function

<!-- formula-not-decoded -->

where the goal is to minimize the validation loss by choosing among optimal solutions of the training loss constrained by the ℓ 1 -norm ball. We use the same training and validation datasets, ( A tr , b tr ) and ( A val , b val ) , from the Wikipedia Math Essential dataset [43], as in [4]. This dataset consists of n = 1068 samples and d = 730 features.

We evaluate the performance of different algorithms across 10 experiments with random initializations and report their average performance within a 4 -minute execution limit. Figure 1 shows the convergence behavior of different algorithms for the inner-level (left) and outer-level (right) optimality gaps in terms of execution time. Our proposed stochastic methods outperform existing approaches. In particular, IR-FSCG achieves high-accuracy solutions for the outer-level objective while maintaining strong inner-level feasibility, surpassing its mini-batch counterpart, SBCGF . Similarly, IR-SCG outperforms its single-sample counterpart, SBCGI . We also observe that both SBCGI and SBCGF exhibit degradation in their inner-level optimality gaps over time. This behavior aligns with the theoretical guarantees in [4], where the inner-level convergence can only be bounded by ϵ g / 2 without monotonic improvement guarantees. Furthermore, neither aR-IP-SeG nor SDBGD makes significant progress in optimizing the outer-level objective function.

̸

Theorem 3 provides the formal convergence guarantees for the IR-FSCG algorithm in the convex setting. Moreover, when we set S = q = ⌊ √ n ⌋ and p = 1 / 2 , Algorithm 2 achieves an ϵ -level optimality gap for both the outer and inner problems with a sample complexity of O ( √ n/ϵ 2 ) . Besides, Algorithm 2 can be readily extended when the outer- and inner-level problems involve different numbers of component functions ( n f = n g ) by introducing ( S f , q f ) for the outer level and ( S g , q g ) for the inner level. Details are omitted for brevity.

We next extend our analysis to the more general case where F is not necessarily convex, using the auxiliary function (3) to assess stationarity. To avoid clutter, we present the result using the optimally tuned parameters of Algorithm 2.

Theorem 4. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 2 with stepsizes α t = log( q + 1) / ( q + 1) for any 0 ≤ t ≤ q , α t = 1 / ( t + 1) ω for any t &gt; q and S = q , and regularization parameters σ t = ς (max { t, q +1 } +1) -p with p = 1 2 and w = 3 4 . If Assumption 1 holds with the exception that F may be nonconvex, then with probability at least 1 -δ , for every t ≥ q , it (jointly) holds that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Moreover, with probability 1 , we have

<!-- formula-not-decoded -->

Figure 1: The inner-level optimality gap (left) and the outer-level absolute optimality gap (right) over time for different algorithms on the over-parametrized regression problem (5).

<!-- image -->

The table below also reports the average number of iterations and stochastic oracle calls over the 4-minute time limit for each algorithm. Our projection-free method completes more iterations within the fixed time budget, benefiting from a simpler linear minimization oracle compared to the projection-free approach in [4]. We also observe that SDBGD and IR-SCG perform a similar number of iterations, as the ℓ 1 -norm projection oracle has comparable computational cost to the linear minimization oracle.

| Method                 | Average # of iterations   | Average # of oracle calls   |
|------------------------|---------------------------|-----------------------------|
| IR-SCG                 | 1182617 . 0               | 2365234 . 0                 |
| IR-FSCG                | 369952 . 5                | 28240425 . 0                |
| SBCGI                  | 6726 . 9                  | 7428681 . 9                 |
| SBCGF                  | 6713 . 4                  | 7935076 . 2                 |
| aR-IP-SeG (short step) | 807700 . 2                | 3230800 . 8                 |
| aR-IP-SeG (long step)  | 799725 . 0                | 3199300 . 0                 |
| SDBGD (short step)     | 1250392 . 6               | 2500782 . 2                 |
| SDBGD (long step)      | 1243720 . 1               | 2487440 . 2                 |

## 4.2 Dictionary learning

In the dictionary learning problem, the goal is to learn a compact representation of the input data A = { a 1 , . . . , a n } ⊆ R m . Formally, we aim to find a dictionary D = [ d 1 · · · d p ] ∈ R m × p such that each data point a i can be approximated by a linear combination of the basis vectors in D . This leads to the following optimization problem

<!-- formula-not-decoded -->

where we refer to X as the coefficient matrix. In practice, data points usually arrive sequentially and the representation evolves gradually. Hence, the dictionary must be updated sequentially as well. Assume that we already have learned a dictionary ˆ D ∈ R m × p and the corresponding coefficient matrix ˆ X ∈ R p × n for some data set A . As a new dataset A ′ = { a ′ 1 , . . . , a ′ n ′ } ⊆ R m arrives, we intend to enrich our dictionary by learning ˜ D ∈ R m × q ( q &gt; p ) and the coefficient matrix ˜ X ∈ R q × n ′ for the new dataset while maintaining good performance of ˜ D on the old dataset A as well as the learned coefficient matrix ˆ X .

Figure 2: The inner-level optimality gap (left) and the outer-level objective function (right) over time for different algorithms on the dictionary learning problem (6).

<!-- image -->

This leads to the following simple bilevel problem with a nonconvex outer-level objective function

<!-- formula-not-decoded -->

where we denote ˆ x i as the prolonged vector in R q by appending zeros at the end. We consider problem (6) on a synthetic dataset with a similar setup to [4, 29].

We evaluate the performance of different algorithms across 10 experiments with random initializations and report their average performance within a 1 -minute execution limit. Figure 2 shows the convergence behavior of different algorithms for the inner-level (left) and outer-level (right) problems in terms of execution time. Our proposed stochastic methods outperform existing approaches. In particular, IR-FSCG achieves better solutions for the outer-level objective while maintaining strong inner-level feasibility, surpassing its mini-batch counterpart, SBCGI . Similarly, IR-SCG outperforms its single-sample counterpart, SBCGI . We again observe that both SBCGI and SBCGF exhibit degradation in their inner-level optimality gaps over time.

The table below reports the average number of iterations and stochastic oracle calls over the 1 -minute time limit for each algorithm. Our projection-free method completes more iterations within the fixed time budget, benefiting from a simpler linear minimization oracle compared to the projection-free methods in [4]. We also observe that SDBGD performs fewer iterations than IR-SCG , as it relies on a more costly projection oracle.

| Method                 | Average # of iterations   | Average # of oracle calls   |
|------------------------|---------------------------|-----------------------------|
| IR-SCG                 | 32736 . 1                 | 65472 . 2                   |
| IR-FSCG                | 30532 . 6                 | 1872851 . 0                 |
| SBCGI                  | 466 . 3                   | 7440 . 6                    |
| SBCGF                  | 361 . 84                  | 6147916 . 0                 |
| aR-IP-SeG (short step) | 6290 . 0                  | 25160 . 0                   |
| aR-IP-SeG (long step)  | 6278 . 1                  | 25112 . 4                   |
| SDBGD (short step)     | 11441 . 3                 | 22882 . 6                   |
| SDBGD (long step)      | 12239 . 3                 | 24478 . 6                   |

## References

- [1] M. Amini and F. Yousefian. An iterative regularized incremental projected subgradient method for a class of bilevel optimization problems. In American Control Conference , pages 40694074, 2019.
- [2] A. Beck and S. Sabach. A first order method for finding minimal norm-like solutions of convex optimization problems. Mathematical Programming , 147(1):25-46, 2014.
- [3] A. Cabot. Proximal point algorithm controlled by a slowly vanishing term: Applications to hierarchical minimization. SIAM Journal on Optimization , 15(2):555-572, 2005.
- [4] J. Cao, R. Jiang, N. Abolfazli, E. Yazdandoost Hamedani, and A. Mokhtari. Projection-free methods for stochastic simple bilevel optimization with convex lower-level problem. In Advances in Neural Information Processing Systems , pages 6105-6131, 2023.
- [5] J. Cao, R. Jiang, E. Y. Hamedani, and A. Mokhtari. On the complexity of finding stationary points in nonconvex simple bilevel optimization. arXiv:2507.23155 , 2025.
- [6] J. Cao, R. Jiang, E. Yazdandoost Hamedani, and A. Mokhtari. An accelerated gradient method for convex smooth simple bilevel optimization. In Advances in Neural Information Processing Systems , pages 45126-45154, 2024.
- [7] P. Chen, X. Shi, R. Jiang, and J. Wang. Penalty-based methods for simple bilevel optimization under Hölderian error bounds. arXiv:2402.02155 , 2024.
- [8] T. Chen, Y. Sun, and W. Yin. Closing the gap: Tighter analysis of alternating stochastic gradient methods for bilevel problems. In Advances in Neural Information Processing Systems , pages 25294-25307, 2021.
- [9] A. Cutkosky and F. Orabona. Momentum-based variance reduction in non-convex SGD. In Advances in Neural Information Processing Systems , pages 15236-15245, 2019.
- [10] M. Dagréou, P. Ablin, S. Vaiter, and T. Moreau. A framework for bilevel optimization that enables stochastic and global variance reduction algorithms. In Advances in Neural Information Processing Systems , pages 26698-26710, 2022.
- [11] S. Diamond and S. Boyd. CVXPY: A Python-embedded modeling language for convex optimization. Journal of Machine Learning Research , 17(83):1-5, 2016.
- [12] L. Doron and S. Shtern. Methodology and first-order algorithms for solving nonsmooth and non-strongly convex bilevel optimization problems. Mathematical Programming , 201(1):521558, 2023.
- [13] J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra. Efficient projections onto the ℓ 1 -ball for learning in high dimensions. In International Conference on Machine Learning , pages 272-279, 2008.
- [14] J. Dutta and T. Pandit. Algorithms for simple bilevel programming. In A. Zemkohoo and S. Dempe, editors, Bilevel Optimization: Advances and Next Challenges , pages 253-291. Springer, 2020.
- [15] C. Fang, C. J. Li, Z. Lin, and T. Zhang. SPIDER: Near-optimal non-convex optimization via stochastic path-integrated differential estimator. In Advances in Neural Information Processing Systems , pages 687-697, 2018.
- [16] L. Franceschi, M. Donini, P. Frasconi, and M. Pontil. Forward and reverse gradient-based hyperparameter optimization. In International Conference on Machine Learning , pages 11651173, 2017.
- [17] L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In International Conference on Machine Learning , pages 1568-1577, 2018.
- [18] M. P. Friedlander and P. Tseng. Exact regularization of convex programs. SIAM Journal on Optimization , 18(4):1326-1350, 2008.
- [19] S. Ghadimi and M. Wang. Approximation methods for bilevel programming. arXiv:1802.02246 , 2018.
- [20] K.-H. Giang-Tran, N. Ho-Nguyen, and D. Lee. A projection-free method for solving convex bilevel optimization problems. Mathematical Programming , 2024.

- [21] C. Gong, X. Liu, and Q. Liu. Bi-objective trade-off with dynamic barrier gradient descent. In Advances in Neural Information Processing Systems , pages 29630-29642, 2021.
- [22] H. Hassani, A. Karbasi, A. Mokhtari, and Z. Shen. Stochastic conditional gradient++:(non) convex minimization and continuous submodular maximization. SIAM Journal on Optimization , 30(4):3315-3344, 2020.
- [23] E. S. Helou and L. E. A. Simões. ϵ -subgradient algorithms for bilevel convex optimization. Inverse Problems , 33(5):055020, 2017. ISSN 0266-5611.
- [24] Y.-G. Hsieh, J. Thornton, E. Ndiaye, M. Klein, M. Cuturi, and P. Ablin. Careful with that scalpel: improving gradient surgery with an EMA. arXiv:2402.02998 , 2024.
- [25] Y. Hu, J. Wang, Y. Xie, A. Krause, and D. Kuhn. Contextual stochastic bilevel optimization. In Advances in Neural Information Processing Systems , pages 78412-78434, 2023.
- [26] M. Jaggi. Revisiting Frank-Wolfe: Projection-free sparse convex optimization. In International Conference on Machine Learning , pages 427-435, 2013.
- [27] A. Jalilzadeh, F. Yousefian, and M. Ebrahimi. Stochastic approximation for estimating the price of stability in stochastic Nash games. ACM Transactions on Modeling and Computer Simulation , 34(2):1-24, 2024.
- [28] K. Ji, J. Yang, and Y . Liang. Bilevel optimization: Convergence analysis and enhanced design. In International Conference on Machine Learning , pages 4882-4892, 2021.
- [29] R. Jiang, N. Abolfazli, A. Mokhtari, and E. Y. Hamedani. A conditional gradient-based method for simple bilevel optimization with convex lower-level problem. In International Conference on Artificial Intelligence and Statistics , pages 10305-10323, 2023.
- [30] H. D. Kaushik and F. Yousefian. A method with convergence rates for optimization problems with variational inequality constraints. SIAM Journal on Optimization , 31(3):2171-2198, 2021.
- [31] P. Khanduri, S. Zeng, M. Hong, H.-T. Wai, Z. Wang, and Z. Yang. A near-optimal algorithm for stochastic bilevel optimization via double-momentum. In Advances in Neural Information Processing Systems , pages 30271-30283, 2021.
- [32] V. Konda and J. Tsitsiklis. Actor-critic algorithms. In Advances in Neural Information Processing Systems , pages 1008-1014, 1999.
- [33] J. Kwon, D. Kwon, S. Wright, and R. D. Nowak. A fully first-order method for stochastic bilevel optimization. In International Conference on Machine Learning , pages 18083-18113, 2023.
- [34] Z. Lu and S. Mei. First-order penalty methods for bilevel optimization. SIAM Journal on Optimization , 34(2):1937-1969, 2024.
- [35] Y. Malitsky. The primal-dual hybrid gradient method reduces to a primal method for linearly constrained optimization problems. arXiv:1706.02602 , 2017.
- [36] R. Merchav and S. Sabach. Convex bi-level optimization problems with nonsmooth outer objective function. SIAM Journal on Optimization , 33(4):3114-3142, 2023.
- [37] R. Merchav, S. Sabach, and M. Teboulle. A fast algorithm for convex composite bi-level optimization. arXiv:2407.21221 , 2024.
- [38] L. M. Nguyen, J. Liu, K. Scheinberg, and M. Takáˇ c. SARAH: A novel method for machine learning problems using stochastic recursive gradient. In International Conference on Machine Learning , pages 2613-2621, 2017.
- [39] F. Pedregosa. Hyperparameter optimization with approximate gradient. In International Conference on Machine Learning , pages 737-746, 2016.
- [40] I. Petrulionyt˙ e, J. Mairal, and M. Arbel. Functional bilevel optimization for machine learning. In Advances in Neural Information Processing Systems , pages 14016-14065, 2024.
- [41] I. Pinelis. Optimum bounds for the distributions of martingales in Banach spaces. The Annals of Probability , 22(4):1679-1706, 1994.
- [42] A. Raghu, M. Raghu, S. Bengio, and O. Vinyals. Rapid learning or feature reuse? Towards understanding the effectiveness of MAML. In International Conference on Learning Representations , 2020.

- [43] B. Rozemberczki, P. Scherer, Y. He, G. Panagopoulos, A. Riedel, M. Astefanoaei, O. Kiss, F. Beres, G. López, N. Collignon, and R. Sarkar. Pytorch geometric temporal: Spatiotemporal signal processing with neural machine learning models. In International Conference on Information and Knowledge Management , pages 4564-4573, 2021.
- [44] S. Sabach and S. Shtern. A first order method for solving convex bilevel optimization problems. SIAM Journal on Optimization , 27(2):640-660, 2017.
- [45] S. Samadi, D. Burbano, and F. Yousefian. Achieving optimal complexity guarantees for a class of bilevel convex optimization problems. In American Control Conference , pages 2206-2211, 2024.
- [46] Y. Shehu, P. T. Vuong, and A. Zemkoho. An inertial extrapolation method for convex simple bilevel optimization. Optimization Methods and Software , 36(1):1-19, 2021.
- [47] L. Shen, N. Ho-Nguyen, and F. Kılınç-Karzan. An online convex optimization-based framework for convex bilevel optimization. Mathematical Programming , 198(2):1519-1582, 2023.
- [48] Z. Shen, C. Fang, P. Zhao, J. Huang, and H. Qian. Complexities in projection-free stochastic non-convex minimization. In International Conference on Artificial Intelligence and Statistics , pages 2868-2876, 2019.
- [49] M. Solodov. An explicit descent method for bilevel convex optimization. Journal of Convex Analysis , 14(2):227, 2007.
- [50] H. Stackelberg. The Theory of Market Economy . Oxford University Press, 1952.
- [51] V. Thoma, B. Pásztor, A. Krause, G. Ramponi, and Y. Hu. Contextual bilevel reinforcement learning for incentive alignment. In Advances in Neural Information Processing Systems , pages 127369-127435, 2024.
- [52] A. N. Tikhonov and V. Y. Arsenin. Solutions of Ill-Posed Problems . V. H. Winston &amp; Sons, 1977. Translated from the Russian, Preface by translation editor Fritz John, Scripta Series in Mathematics.
- [53] P. Tseng. On accelerated proximal gradient methods for convex-concave optimization. Technical report, Department of Mathematics, University of Washington, 2008.
- [54] J. Wang, X. Shi, and R. Jiang. Near-optimal convex simple bilevel optimization with a bisection method. In International Conference on Artificial Intelligence and Statistics , pages 2008-2016, 2024.
- [55] J. Yang, K. Ji, and Y. Liang. Provably faster algorithms for bilevel optimization. In Advances in Neural Information Processing Systems , pages 13670-13682, 2021.
- [56] Y. Yang, P. Xiao, and K. Ji. Achieving O ( ϵ -1 . 5 ) complexity in Hessian/Jacobian-free stochastic bilevel optimization. In Advances in Neural Information Processing Systems , pages 3949139503, 2023.
- [57] A. Yurtsever, S. Sra, and V. Cevher. Conditional gradient methods via stochastic pathintegrated differential estimator. In International Conference on Machine Learning , pages 7282-7291, 2019.
- [58] H. Zhang, L. Chen, J. Xu, and J. Zhang. Functionally constrained algorithm solves convex simple bilevel problem. In Advances in Neural Information Processing Systems , pages 5759157618, 2024.
- [59] M. Zhang, L. Chen, A. Mokhtari, H. Hassani, and A. Karbasi. Quantized Frank-Wolfe: Faster optimization, lower communication, and projection free. In International Conference on Artificial Intelligence and Statistics , pages 3696-3706, 2020.
- [60] M. Zhang, Z. Shen, A. Mokhtari, H. Hassani, and A. Karbasi. One sample stochastic FrankWolfe. In International Conference on Artificial Intelligence and Statistics , pages 4012-4023, 2020.

## A Preparatory Lemmas

Throughout all appendices, we employ the notation Φ t ( x ) := σ t F ( x )+ G ( x ) and ̂ ∇ Φ t := σ t ̂ ∇ F t + ̂ ∇ G t to denote the gradient estimate at step t . We also denote x opt ∈ X opt as an optimal solution to (1). All norms refer to the Euclidean norm.

For any sequence { x t } t ≥ 0 generated by either Algorithm 1 or Algorithm 2, we observe that by construction, x 0 , v t ∈ X . Hence, under convexity of X , x t +1 ∈ X as it is a convex combination of x t and v t , and X is assumed to be convex. By a simple induction argument, one can easily show that the sequence { x t } t ≥ 0 ∈ X .

## A.1 Convex outer-level

Lemma 1. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 1 or Algorithm 2 with some given stepsizes { α t } t ≥ 0 . If Assumption 1 holds, then for every t ≥ k , we have

<!-- formula-not-decoded -->

Proof of Lemma 1. Recall that

<!-- formula-not-decoded -->

As both functions F and G are assumed to be convex and smooth, it is easy to show that Φ is convex and σ t L f + L g -smooth. By smoothness of Φ t and the assumption that diam( X ) = D , one can show that

<!-- formula-not-decoded -->

Note that for any w t ∈ X and any gradient estimator ̂ ∇ Φ t , we have

<!-- formula-not-decoded -->

where the first inequality follows from the definition of v t , and the second inequality uses the Cauchy-Schwarz inequality, and the assumption that diam( X ) = D . Thus, by smoothness of Φ t , for any w t ∈ X and any gradient estimator ̂ ∇ Φ t , we have

<!-- formula-not-decoded -->

and set w t = x opt . Thanks to the convexity of Φ t , one can easily show that ∇ Φ t ( x t ) ⊤ ( x opt -x t ) ≤ Φ t ( x opt ) -Φ t ( x t ) . Thus, we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof concludes by using the definition of the stepsizes α t .

Lemma 2. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 1 or Algorithm 2 with stepsizes { α t } t ≥ 0 given by α t = 2 / ( t +2) for any t ≥ k . If Assumption 1 holds, then for each t ≥ k , we have

<!-- formula-not-decoded -->

Proof of Lemma 2. By definition of Φ t , we have

<!-- formula-not-decoded -->

To simplify the derivations, we introduce the shorthands h t := Φ t ( x t ) -Φ t ( x opt ) and

<!-- formula-not-decoded -->

Applying Lemma 1 with stepsize α t = 2 / ( t +2) for all t ≥ k gives the recursion h t +1 ≤ t t +2 h t + η t for any t ≥ k . This can be re-expressed as

<!-- formula-not-decoded -->

for any t ≥ k +1 . Thus, for t ≥ k +1 , it holds that

<!-- formula-not-decoded -->

By definition of h t and η t , and using the inequalities i/ ( i +1) &lt; 1 and σ i -1 ≤ σ 0 for i ∈ [ t ] \ [ k ] , we arrive at

<!-- formula-not-decoded -->

which implies (8). This completes the proof.

Proposition 1. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 1 or Algorithm 2 with stepsizes { α t } t ≥ 0 given by α t = 2 / ( t +2) for any t ≥ k . If Assumption 1 holds, then for each t ≥ k , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Proof of Proposition 1. By definition of the function Φ t , the first term in (8) satisfies

<!-- formula-not-decoded -->

Since x t ∈ X , we have min x ∈ X F ( x ) -F opt ≤ F ( x t ) -F opt for all t ≥ 0 . Using this bound together with the identity in (11), we can further lower bound the left-hand side of inequality (8), and arrive at the bound (9). This completes the proof of the first claim.

For the second claim, since { x t } t ≥ 0 ∈ X is feasible in the lower-level problem, it follows directly that G ( x t ) ≥ G opt . Combining this inequality with (11) and (8), we obtain (10), which concludes the proof.

## A.2 Nonconvex outer-level

When F is possibly nonconvex, we start by establishing a bound, akin to Lemma 2.

Lemma 3. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 1 or Algorithm 2 with { α t , σ t } t ≥ 0 given by α t = ( t + 1) -ω and σ t = ς ( t + 1) -p for t ≥ k , with 0 &lt; p ≤ ω . If Assumption 1 holds with the exception that F may be nonconvex, then for every t ≥ k , we have

<!-- formula-not-decoded -->

Proof of Lemma 3. Using the definition of Φ t = σ t F + G , for any t ≥ k , we have

<!-- formula-not-decoded -->

where the inequality follows directly from (7) and by minimizing over w t ∈ X . We emphasize that (7) does not involve any convexity assumptions on Φ t and relies solely on its smoothness. Furthermore, notice that for any x ∈ X ,

<!-- formula-not-decoded -->

where the last equality follows from the definition of F and G . Using the above bound, we may thus conclude that

<!-- formula-not-decoded -->

A simple re-arrangement and using the definition of G then yields the bound

<!-- formula-not-decoded -->

Using the definition of α t and the fact that ( t +1) ω ≤ t ω -1 for any t ≥ 0 , we arrive at

<!-- formula-not-decoded -->

Unwinding this recursion back to i = k , we obtain

<!-- formula-not-decoded -->

Since p ≤ ω , we have ( i +1) ω σ i -i ω σ i -1 ≥ 0 . Therefore, we deduce that

<!-- formula-not-decoded -->

This completes the proof.

To establish asymptotic convergence of the proposed methods under nonconvex outer-level, we rely on an observation that function F is Lipschitz continuous on X .

Lemma 4. If Assumption 1 holds, the function F as defined in (3) is L f D + max z ∈ X ∥∇ F ( z ) ∥ -Lipschitz continuous over X .

Proof of Lemma 4. Given x, y ∈ X , we observe that

<!-- formula-not-decoded -->

where the first and third inequality follows from Cauchy-Schwartz inequality, while the second inequality follows from the L f -smoothness of F .

By interchanging the role of y and x , we deduce that

<!-- formula-not-decoded -->

This concludes the proof.

## B Proofs of Section 2

## B.1 Proof of Theorem 1

We restate Theorem 1 with exact upper bounds on sub-optimality gaps.

Theorem 1. Let { z t } t ≥ 1 be the iterates generated by Algorithm 1 with α t = 2 / ( t + 2) for any t ≥ 0 , and regularization parameters { σ t } t ≥ 0 given by σ t := ς ( t +1) -p for some chosen ς &gt; 0 , p ∈ (0 , 1 / 2) . Under Assumption 1, for any t ≥ 1 , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where c is some absolute constant. Moreover, with probability 1 , we have

<!-- formula-not-decoded -->

The proof proceeds by applying Proposition 1 with k = 0 . In this case, the main random component in (9) and (10) is the term ∑ i ∈ [ t ] 4 iD ∥ ̂ ∇ Φ i -1 -∇ Φ i -1 ( x i -1 ) ∥ . Wefirst derive a probabilistic upper bound for this term, then carefully set the parameters in Algorithm 1 to establish the convergence rate for both the lower- and upper-level problems.

We begin by bounding the gradient estimator used in Algorithm 1.

Lemma 5. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 1 with stepsize α t = 2 / ( t +2) for any t ≥ 0 . If Assumption 1 holds, then for any t ≥ 1 , given δ ∈ (0 , 1) , with probability at least 1 -δ , for some absolute constant c , it (jointly) holds that

<!-- formula-not-decoded -->

Proof of Lemma 5. The proof closely follows the argument in [4, Lemma 4.1]. In particular, the bounds in [4, Lemma B.1] still hold under the modified stepsize α t = 2 / ( t +2) , instead of 1 / ( t +1) . The constant inside the logarithmic term changes from 6 to 4, as we now apply a union bound over two events rather than three. Additionally, the constants of the upper bounds is doubled due to use of the modified stepsize. We omit the details for brevity.

Proposition 2. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 1 with α t = 2 / ( t +2) for any t ≥ 0 . If Assumption 1 holds and the parameters { σ t } t ≥ 0 are non-increasing and positive, then given δ ∈ (0 , 1) , with probability at least 1 -δ , for some absolute constant c and any t ≥ 1 , we have

<!-- formula-not-decoded -->

Proof of Proposition 2. Given i ≥ 1 and δ ∈ (0 , 1) , by Lemma 5, with probability at least 1 -δ/i ( i +1) , we have

<!-- formula-not-decoded -->

Thus, using the union bound, with probability at least 1 -δ ∑ i ≥ 1 1 /i ( i +1) = 1 -δ , for any i ∈ [ t ] , we have

<!-- formula-not-decoded -->

where the second inequality follows from bounding ∑ i ∈ [ t ] √ i ≤ 2 3 ( t +1) 3 / 2 via the Riemann sum approximation, and observing that 2 √ 2 ≤ 3 .

To establish the desired convergence results for Algorithm 1, we need to impose certain conditions on the regularization parameters { σ t } t ≥ 0 , stated below.

Condition 1. The regularization parameters { σ t } t ≥ 0 are non-increasing, positive, and converge to 0 . Condition 2. There exists L ∈ R such that

<!-- formula-not-decoded -->

Condition 3. ( t +1) σ 2 t +1 &gt; tσ 2 t for any t ≥ 0 , and log( t ) = o ( tσ 2 t ) as t →∞ .

The following parameters will appear in our analysis and convergence rates

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

While it is not obvious a priori, Lemma 17 in Appendix D.1 guarantees that these quantities remain finite for any parameter choice satisfying Conditions 1-3. In particular, Conditions 1-3 hold when σ t = ς ( t + 1) -p for any p ∈ (0 , 1) , which is what is used in Theorems 1 and 3, and Lemma 8 provides explicit bounds for this case. We now analyze the sequence { x t } t ≥ 0 in Algorithm 1 for the inner-level problem.

Lemma 6. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 1 with α t = 2 / ( t +2) for any t ≥ 0 , and let C δ be defined as in (14a) . If Assumption 1 and Conditions 1-3 hold, with probability at least 1 -δ , it holds that

<!-- formula-not-decoded -->

for any t ≥ 0 .

Proof of Lemma 6. Combining (9) from Proposition 1 with the probabilistic bound in Proposition 2, and after some straightforward calculations, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

By definition of C δ , the right-hand side is at most C δ σ t for any t ≥ 1 . This completes the proof.

We next analyze the convergence of the sequence { z t } t ≥ 0 in Algorithm 1 for both outer- and innerlevel problems.

Lemma 7. Let { z t } t ≥ 1 denote the iterates generated by Algorithm 1 with α t = 2 / ( t +2) for any t ≥ 0 , and let C δ , V be defined as in (14). If Assumption 1 and Conditions 1-3 hold, then with probability at least 1 -δ , it (jointly) holds that

<!-- formula-not-decoded -->

Proof of Lemma 7. Recall from Algorithm 1 that we have defined

<!-- formula-not-decoded -->

thus z t is simply a convex combination of x 0 , . . . , x t for every t ≥ 1 . Therefore, as F is convex, we can apply Jensen's inequality to the left-hand side of the inequality (10) with k = 0 , and after some tedious calculation, we arrive at

<!-- formula-not-decoded -->

Using the inequality S t ≥ ( t +1) tσ t , which holds thanks to Condition 1, and applying Proposition 2, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

This completes the proof of the first claim.

For the second claim, we follow the same procedure. In particular, applying Jensen's inequality with respect to the convex function G to the left-hand side of (15) and using the inequality S t ≥ ( t +1) tσ t , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

The proof concludes by using the definition of V .

Lemma 7 establishes a convergence result in terms of the regularization parameters { σ t } t ≥ 0 . The next lemma specifies an update rule for these parameters and provides bounds on the quantities introduced in (14).

Lemma 8. Consider the sequence σ t := ς ( t +1) -p for t ≥ 0 and the quantities defined in (14).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(ii) If p ∈ (0 , 1 / 2) , then { σ t } t ≥ 0 satisfies Condition 3, and we have

<!-- formula-not-decoded -->

Proof of Lemma 8. As for assertion (i), it is trivial to see that the sequence { σ t } t ≥ 0 satisfies Conditions 1. To validate Condition 2, observe that

<!-- formula-not-decoded -->

where the last equality holds as the derivative of y p at y = 1 equals p . We next establish the bound for C . For t &gt; 1 , it follows from the mean value theorem that there exists b t ∈ ( t, t +1) such that

<!-- formula-not-decoded -->

Hence, we observe that

<!-- formula-not-decoded -->

where the first inequality is due to t + 1 ≤ 2 t and the second inequality holds because t 1 -p is an increasing function in t . This implies

<!-- formula-not-decoded -->

Since min t ≥ 0 ( t +1) σ t = ς , we obtain

<!-- formula-not-decoded -->

We next establish the bound for V . Using similar arguments, we observe that

<!-- formula-not-decoded -->

If 1 -2 p ≥ 0 , then t 1 -2 p is an increasing function in t and thus

<!-- formula-not-decoded -->

Dividing both sides by t ( t +1) σ 2 t , we deduce that V ≤ 2 p . When 1 -2 p &lt; 0 , we have

<!-- formula-not-decoded -->

where the first inequality is due to the Riemann sum approximation ∑ i ∈ [ t ] i 1 -2 p ≤ 1+ ∫ t i =1 i 1 -2 p d i . Dividing both sides by t ( t +1) σ 2 t , we deduce that V ≤ p/ (1 -p ) , thus the claim follows.

As for assertion (ii), it is trivial to see that Condition 3 holds. We thus focus on bounding ¯ C δ . Using the fact that sup t ≥ 1 log( t ) /t = 1 /e, for any t ≥ 1 , we may conclude that

<!-- formula-not-decoded -->

The claim then follows from the definition of ¯ C δ .

Proof of Theorem 1. The first claim on the non-asymptotic convergence guarantee follows directly from Lemmas 7 and 8. As for the second claim on the asymptotic convergence, we prove a more general result: even if σ t is not set as ς ( t +1) -p , the iterates of Algorithm 1 still converge almost surely, provided the regularization sequence satisfies the conditions in Lemma 6.

Lemma 6 implies that lim t →∞ G ( x t ) = G opt with probability at least 1 -δ . As δ can be arbitrarily small, we may conclude that lim t →∞ G ( x t ) = G opt , almost surely. This implies that any limit point of { x t } t ≥ 0 is in X opt almost surely. Since F is convex, hence lower semi-continuous over X , and by definition of F opt , we have lim inf t →∞ F ( x t ) ≥ F opt almost surely.

Besides, by combining Propositions 1 and 2, we deduce that for any δ ∈ (0 , 1) , with probability at least 1 -δ , we have, for all t ≥ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality holds due to Condition 1. We claim that all terms on the right hand side converge to 0 as t →∞ . This is obvious except for the last term.

Recall that almost surely lim inf t →∞ F ( x t ) ≥ F opt . Therefore we have lim sup t →∞ ( F opt -F ( x t )) ≤ 0 and hence lim t →∞ max { F opt -F ( x t ) , 0 } = 0 almost surely. Applying the Stolz-Cesàro theorem and (33) leads to

<!-- formula-not-decoded -->

almost surely. Then each term on the right hand side of the inequality converges to 0 as t →∞ . It follows that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Thus, since this holds for any δ ∈ (0 , 1) , it holds almost surely. In conclusion, we have shown that lim sup t →∞ ( F ( x t ) -F opt ) ≤ 0 and lim inf t →∞ ( F ( x t ) -F opt ) ≥ 0 , which implies that lim t →∞ F ( x t ) = F opt almost surely. This completes the proof.

## B.2 Proof of Theorem 2

We restate Theorem 2 with explicit upper bounds on the stationary gaps and for a more general choice of p and ω . Toward the end of the proof, we show that the optimal parameters are p = 2 / 7 and ω = 6 / 7 .

Theorem 2. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 1 with stepsizes α t = ( t +1) -ω and regularization parameters σ t = ς ( t + 1) -p for any t ≥ 0 . If Assumption 1 holds with the exception that F may be nonconvex, then with probability at least 1 -δ , for every t ≥ 0 , it (jointly) holds that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where c is some absolute constant.

Proof of Theorem 2. Following the proof of [4, Lemma 4.1], for any i ≥ 0 , one can easily show that, with probability at least 1 -η i , we have

<!-- formula-not-decoded -->

Setting η i = δ ( i +1)( i +2) , applying a union bound, and using the fact that σ i ≤ σ 0 = ς we have that with probability at least 1 -δ ∑ t i =0 1 ( i +1)( i +2) ≥ 1 -δ , the following holds:

<!-- formula-not-decoded -->

Applying Lemma 3 with k = 0 , with probability at least 1 -δ , we thus have

<!-- formula-not-decoded -->

Note that the maximum in the definition of function F is with respect to X opt , not X . Therefore, F ( x i ) can take both positive and negative values since x i ∈ X . However, we can derive the following bound

<!-- formula-not-decoded -->

which further implies that

<!-- formula-not-decoded -->

We next focus on the upper-level problem. Since G ( x t +1 ) ≥ 0 , using (16), we have

<!-- formula-not-decoded -->

Dividing both sides by ∑ t i =0 σ i , and furthermore exploiting the bounds

<!-- formula-not-decoded -->

we arrive at

<!-- formula-not-decoded -->

Both bounds involving G and F hold with probability ≥ 1 -δ . Hence, the first claim on the nonasymptotic convergence guarantee follows. Moreover, by optimizing over the parameters p and ω , we aim to ensure that the right-hand sides of both bounds converge to 0 . To achieve this, we select p and ω to minimize the slowest rate with respect to t :

<!-- formula-not-decoded -->

which is realized by setting p = 2 / 7 , ω = 6 / 7 as required. Substituting these values yields the bound presented in the main body of the paper.

As for the second claim on asymptotic convergence, we prove a more general result: If

<!-- formula-not-decoded -->

then it holds that and

<!-- formula-not-decoded -->

Under this additional assumption and the fact that G ( x t ) ≥ 0 , we deduce that lim t →∞ G ( x t ) = 0 from taking the limit on both sides of (17). From the continuity of G , we deduce that any limit point of { x t } t ≥ 0 is in X opt . the continuity of F from Lemma 4 and the fact that F ( x ) ≥ 0 for any x ∈ X opt , we deduce that lim inf t →∞ F ( x t ) ≥ 0 . Recall that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining these observations and (18), we have

<!-- formula-not-decoded -->

By the Stolz-Cesàro theorem (since p ∈ (0 , 1) , ∑ t ≥ 0 σ t = ∞ ) and taking the limit on both sides of the above inequality, we obtain lim inf t →∞ F ( x t ) ≤ 0 . Since this holds with probability at least 1 -δ for any δ ∈ (0 , 1) , it should hold almost surely. Thus, we conclude the proof.

## C Proofs of Section 3

## C.1 Proof of Theorem 3

We restate Theorem 3 with exact upper bounds on sub-optimality gaps.

Theorem 3. Suppose the expectations defining F and G in (1) are from uniform distributions over finite sets of size [ n ] . Let { z t } t ≥ 1 be the iterates generated by Algorithm 2 with α t = log( q ) /q for 0 ≤ t &lt; q , α t = 2 / ( t +2) for any t ≥ q , and σ t := ς (max { t, q } +1) -p for some chosen ς &gt; 0 , p ∈ (0 , 1) and S = q . Under Assumption 1, with probability at least 1 -δ , for any t ≥ q , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, with probability 1 , it holds that

<!-- formula-not-decoded -->

The proof proceeds by applying Proposition 1 with k = q . Unlike the proof of Theorem 1, here we have two main random components in (9) and (10): ( q + 1) q (Φ q ( x q ) -Φ q ( x opt )) and ∑ i ∈ [ t ] \ [ q ] 4 iD ∥ ̂ ∇ Φ i -1 -∇ Φ i -1 ( x i -1 ) ∥ . We first derive probabilistic upper bounds for these terms, then carefully set the parameters in Algorithm 2 to establish the convergence rate for both lowerand upper-level problems. In the following we use the notation

<!-- formula-not-decoded -->

For simplicity, we assume that F and G have the same size support, though our arguments are easily extended to the more general case. Also, given S ⊆ [ n ] , we write

<!-- formula-not-decoded -->

We begin with a probabilistic bound on the gradient estimator used in Algorithm 2.

Lemma 9. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 2 with α t = log( q ) /q for any 0 ≤ t &lt; q and α t = 2 / ( t + 2) for any t ≥ q . If Assumption 1 holds, then for any t ≥ 0 , given δ ∈ (0 , 1) , with probability at least 1 -δ , it (jointly) holds that

<!-- formula-not-decoded -->

provided that 0 ≤ t &lt; q , and

<!-- formula-not-decoded -->

where s t := q ⌊ t/q ⌋ , provided that t &gt; q .

Proof of Lemma 9. If t = s t , we have ̂ ∇ F t = ∇ F ( x t ) . Otherwise, let S t ⊂ [ n ] be the index set of size S chosen at iteration t , and for i ∈ S t define

<!-- formula-not-decoded -->

From the update rule for x t , we have ∥ x t -x t -1 ∥ = α t -1 ∥ v t -1 -x t -1 ∥ ≤ α t -1 D for any t ≥ 1 . As a result,

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ∑ t s = s t +1 ∑ i ∈S s ϵ s,i is a martingale with bounded increments, we apply a concentration inequality [41, Theorem 3.5] to get

<!-- formula-not-decoded -->

If 1 ≤ t ≤ q , we have

<!-- formula-not-decoded -->

and thus

<!-- formula-not-decoded -->

If t &gt; q , we observe that

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

Arguing similarly for ̂ ∇ G t , we have

<!-- formula-not-decoded -->

if 1 ≤ t ≤ q and

<!-- formula-not-decoded -->

if t &gt; q . Given δ ∈ (0 , 1) and 1 ≤ t ≤ q , setting the right hand side = δ/ 2 and solving for λ , we have and

<!-- formula-not-decoded -->

Applying union bound, we deduce that with probability at least 1 -δ , (19) holds. Arguing similarly for t &gt; q , we also deduce that (20) holds with probability at least 1 -δ .

for any t ∈ { s t +1 , . . . , s t + q } and i ∈ [ S t ] . For any t such that t = s t , we have from the definition of ̂ ∇ F t that

Thus, we deduce that

To establish convergence results for Algorithm 2, we replace Condition 3 with the following. Condition 4. σ t = σ 0 for 0 ≤ t ≤ q , ( t + 1) σ t +1 &gt; tσ t for any t ≥ 0 , and log( t ) = o ( tσ t ) as t →∞ .

Note that σ t is fixed during the initialization phase, and σ 2 t in Condition 3 is replaced by σ 2 t in Condition 4. Unlike the proof of Theorem 1, where we set k = 0 , the proof of Theorem 3 requires setting k = q . We therefore begin by analyzing the sequence { x t } t ≥ 0 generated by Algorithm 2 over the first q iterations.

Lemma 10. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 2 with α t = log( q ) /q for any 0 ≤ t &lt; q and S = q . If Assumption 1 and Condition 4 hold, we have then given δ ∈ (0 , 1) , with probability at least 1 -δ ∑ 0 ≤ i&lt;q 1 / ( i +1)( i +2) , for any t ≥ q , it holds that

<!-- formula-not-decoded -->

Proof of Lemma 10. By Lemma 1, for any 0 ≤ i &lt; q , we have

<!-- formula-not-decoded -->

By Condition 4, we have α = · · · = α and

0 q -1 σ 0 = · · · = σ q = ⇒ Φ 0 = · · · = Φ q , where the implication follows from the definition of function Φ t . Thus, we may conclude that

<!-- formula-not-decoded -->

where both inequalities are direct consequences of Condition 4 and (21). Note that

<!-- formula-not-decoded -->

where the last equality follows from convergence of geometric series. We thus arrive at

<!-- formula-not-decoded -->

Note that by the definition of Φ i and the triangle inequality, we have

<!-- formula-not-decoded -->

Hence, applying Lemma 9 along with a union bound argument implies that, for any δ ∈ (0 , 1) , with probability at least 1 -δ ∑ 0 ≤ i&lt;q 1 / ( i +1)( i +2) , we have

<!-- formula-not-decoded -->

Plugging the above probabilistic bound into (22), noting that t ≥ q and S = q , and using the definition of α 0 complete the proof.

We next analyze the sequence { x t } t ≥ 0 generated by Algorithm 2 after the first q iterations.

Proposition 3. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 2 with α t = log( q ) /q for any 0 ≤ t &lt; q , α t = 2 / ( t + 2) for any t ≥ q and S = q . If Assumption 1 holds and the parameters { σ t } t ≥ 0 are non-increasing and positive, then given δ ∈ (0 , 1) , with probability at least 1 -δ ∑ i ∈ [ t ] \ [ q ] 1 /i ( i +1) , for any t &gt; q , it holds that

<!-- formula-not-decoded -->

Proof of Proposition 3. For any i ≥ 1 , we have

<!-- formula-not-decoded -->

Given δ ∈ (0 , 1) and q &lt; i ≤ 2 q , by Lemma 9 and the inequality (24), with probability at least 1 -δ/i ( i +1) , we have

<!-- formula-not-decoded -->

Hence, if q &lt; t ≤ 2 q , with probability at least 1 -δ ∑ q&lt;i ≤ t 1 /i ( i +1) , we have

<!-- formula-not-decoded -->

where the second inequality holds as 1 q ∑ q&lt;i ≤ t i ≤ 2 t -q/ 2 for all q &lt; t ≤ 2 q . Similarly, given δ ∈ (0 , 1) and i &gt; 2 q , by Lemma 9 and the inequality (24), with probability at least 1 -δ/i ( i +1) , we have

<!-- formula-not-decoded -->

where the second inequality follows since q ⌊ ( i -1) /q ⌋ +1 ≥ i -q , and the third inequality follows since i/ ( i -q ) &lt; 2 when i &gt; 2 q . Thus, if t &gt; 2 q , with probability at least 1 -δ ∑ 2 q&lt;i ≤ t 1 /i ( i +1) = 1 -δ , we have

<!-- formula-not-decoded -->

For t &gt; 2 q , with probability at least 1 -δ ∑ q&lt;i ≤ t 1 /i ( i +1) , the sum over all q &lt; i ≤ t is thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, regardless of whether q &lt; t ≤ 2 q or t &gt; 2 q , (23) holds.

The following parameters will appear in our analysis and convergence rates

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 17 in Appendix D.1 guarantees that these quantities remain finite for any parameter choice satisfying Conditions 1-2 and 4. In particular, these conditions hold when σ t = ς ( t + 1) p for any p ∈ (0 , 1) , which is what is used in Theorem 3. We now analyze the sequence { x t } t ≥ 0 in Algorithm 2 for the inner-level problem.

Lemma 11. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 2 with α t = log( q ) /q for 0 ≤ t &lt; q α t = 2 / ( t +2) for any t ≥ q , S = q , and let C δ,q be defined as in (25c) . If Assumption 1 and Condition 1, 2 and 4 hold, then with probability at least 1 -δ , it holds that

<!-- formula-not-decoded -->

for any t ≥ q .

Proof of Lemma 11. Combining (9) from Proposition 1 with the probabilistic bounds in Lemma 10 and Proposition 3, and after some straightforward calculations, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Here, the right-hand side is at most C δ,q , and therefore, G ( x t ) -G opt ≤ C δ,q σ t for any t ≥ q .

We next analyze the convergence of the sequence { z t } t ≥ 0 in Algorithm 2 for both outer- and innerlevel problems.

Lemma 12. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 2 with α t = log( q ) /q for 0 ≤ t &lt; q α t = 2 / ( t + 2) for any t ≥ q , S = q , and let C δ,q and V q defined as in (25). If Assumption 1 and Condition 1, 2 and 4 hold, then with probability at least 1 -δ , it (jointly) holds that

<!-- formula-not-decoded -->

for any

Proof of Lemma 12. Recall from Algorithm 1 that we have defined

<!-- formula-not-decoded -->

thus z t is simply a convex combination of x 0 , . . . , x t for every t ≥ q . Therefore, as F is convex, we can apply Jensen's inequality to the left-hand side of the inequality (10) with k = q , and after some tedious calculation, for any t ≥ q , we arrive at

<!-- formula-not-decoded -->

Using the inequality S t ≥ ( t +1) tσ t , which holds thanks to Condition 1, and applying Lemma 10 and Proposition 2, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

for every t ≥ q . This completes the proof of the first claim.

For the second claim, we follow the same procedure. In particular, applying Jensen's inequality with respect to the convex function G to the left-hand side of (26) and using the inequality S t ≥ ( t +1) tσ t , with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

The proof concludes by using the definition of V q .

Lemma 13. Consider the sequence σ t := ς ( t +1) -p for t ≥ 0 and the quantities defined in (25). If p ∈ (0 , 1) , then the sequence { σ t } t ≥ 0 satisfies Condition 1, 2 and 4 with L = p . Furthermore,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Proof of Lemma 13. We observe that C q , V q can be bounded from above by C, V defined in (14) under the conditions in Lemma 8. Therefore, we focus on ¯ C δ,q for the remainder of the proof. Using the basic inequality log( x ) /x ≤ 1 /e for any x &gt; 0 , observe that for any t ≥ q

<!-- formula-not-decoded -->

We have

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

We also have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

This concludes the proof.

Proof of Theorem 3. The first part is an immediate consequence of Lemmas 12 and 13. Thus, we devote this proof for the second part by proving that a slightly more general result that as long as Conditions 1-2 and Condition 4 hold, the asymptotic convergence remains.

Lemma 11 implies that lim t →∞ G ( x t ) = G opt with probability at least 1 -δ . As δ can be arbitrarily small, we conclude that lim t →∞ G ( x t ) = G opt , almost surely. This implies that any limit point of

{ x t } t ≥ 0 is in X opt almost surely. Since F is convex, hence lower semi-continuous over X , and by definition of F opt , we have lim inf t →∞ F ( x t ) ≥ F opt almost surely.

Besides, by combining Proposition 1, Lemma 10 and Proposition 3, with probability at least 1 -δ , we have

<!-- formula-not-decoded -->

Similar to the proof of Theorem 1, all terms on the right-hand side converge to 0 as t →∞ , implying that lim sup t →∞ ( F ( x t ) -F opt ) ≤ 0 which holds with probability at least 1 -δ for any δ ∈ (0 , 1) . Thus, lim sup t →∞ ( F ( x t ) -F opt ) ≤ 0 almost surely. Combined with lim inf t →∞ ( F ( x t ) -F opt ) ≥ 0 , this yields lim t →∞ F ( x t ) = F opt almost surely, concluding the proof.

## C.2 Proof of Theorem 4

Here, we restate Theorem 4 with exact upper bounds on stationary gaps.

Theorem 4. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 2 with stepsizes α t = log( q + 1) / ( q + 1) for any 0 ≤ t ≤ q , α t = 1 / ( t + 1) ω for any t &gt; q and S = q , and regularization parameters σ t = ς (max { t, q + 1 } + 1) -p with p = 1 2 and w = 3 4 . If Assumption 1 holds with the exception that F may be nonconvex and the sequence { β t } t ≥ 0 is defined as in (4) , then with probability at least 1 -δ , for every t ≥ q , it (jointly) holds that

<!-- formula-not-decoded -->

where β i 's are defined in (4) and a, b are constants that satisfy

<!-- formula-not-decoded -->

The proof relies on several intermediate results. We begin with the following probabilistic bound.

Lemma14. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 2 with α t = α for any 0 ≤ t ≤ q and α t = ( t + 1) -ω with ω &gt; 1 2 for any t &gt; q . If Assumption 1 holds with the exception that F may be nonconvex, then for any t ≥ 0 , given δ ∈ (0 , 1) , with probability at least 1 -δ , it (jointly) holds that

<!-- formula-not-decoded -->

provided that 0 ≤ t &lt; q , and

<!-- formula-not-decoded -->

where s t := q ⌊ t/q ⌋ , provided that t &gt; q .

Proof of Lemma 14. Recall from the proof of Lemma 9 that, employing the concentration inequality in [41, Theorem 3.5] yields

<!-- formula-not-decoded -->

If 1 ≤ t ≤ q , one can easily show that

<!-- formula-not-decoded -->

If t &gt; q , we observe that

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Arguing similarly for ̂ ∇ G t , we have

<!-- formula-not-decoded -->

if 0 ≤ t ≤ q and

<!-- formula-not-decoded -->

if t &gt; q . Given δ ∈ (0 , 1) and 1 ≤ t ≤ q , setting the right hand side = δ/ 2 and solving for λ yields

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Applying union bound, we deduce that with probability at least 1 -δ , (19) holds. Arguing similarly for t &gt; q , we also deduce that (20) holds with probability at least 1 -δ .

We next analyze the sequence { x t } t ≥ 0 generated by Algorithm 2 over the first q iterations.

Lemma 15. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 2 with α t = α for any 0 ≤ t ≤ q and S = q . If Assumption 1 holds with the exception that F may be nonconvex and Condition 4 is satisfied, then given δ ∈ (0 , 1) , with probability at least 1 -δ ∑ 0 ≤ i ≤ q 1 / ( i +1)( i +2) , for any 0 ≤ t ≤ q +1 , it holds that

<!-- formula-not-decoded -->

Proof of Lemma 15. Applying the bound (12) and after a straightforward re-arrangement, under Condition 4, for any 0 ≤ i ≤ t , we obtain

<!-- formula-not-decoded -->

Given δ ∈ (0 , 1) , we apply Lemma 14 to deduce that with probability at least 1 -δ/ ( i +1)( i +2) , we have

<!-- formula-not-decoded -->

Applying the union bound, for any 0 ≤ t ≤ q , with probability at least 1 -δ ∑ 0 ≤ i ≤ t 1 / ( i +1)( i +2) , we obtain

<!-- formula-not-decoded -->

Using the geometric series bound ∑ t i =0 (1 -α ) t -i ≤ 1 /α and observing that

<!-- formula-not-decoded -->

conclude the proof.

We finally analyze the sequence { x t } t ≥ 0 generated by Algorithm 2 after the first q iterations.

Proposition 4. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 2 with α t = α for any 0 ≤ t &lt; q , α t = ( t +1) -w with ω &gt; 1 2 for any t ≥ q and S = q . If Assumption 1 holds with the exception that F may be nonconvex and the parameters { σ t } t ≥ 0 are non-increasing and positive, then given δ ∈ (0 , 1) , with probability at least 1 -δ ∑ i ∈ [ t ] \ [ q ] 1 ( i +1)( i +2) , for any t &gt; q , it holds that

<!-- formula-not-decoded -->

Proof of Proposition 4. For any i ≥ 0 , we have

<!-- formula-not-decoded -->

Moreover, for any i &gt; q , by the mean value theorem there exists c i ∈ [ s i , i ] such that

<!-- formula-not-decoded -->

Given δ ∈ (0 , 1) and i &gt; q , by Lemma 14 and the inequality (30), with probability at least 1 -δ/ ( i +1)( i +2) , we have

<!-- formula-not-decoded -->

Hence, if t &gt; q , with probability at least 1 -δ ∑ q&lt;i ≤ t 1 / ( i +1)( i +2) , we have

<!-- formula-not-decoded -->

where the last inequality follows from the Riemann sum approximation.

We now instantiate the regularization parameter and analyze Algorithm 2.

Lemma 16. Let { x t } t ≥ 0 denote the iterates generated by Algorithm 2 with α t = α := log( q + 1) / ( q +1) for any 0 ≤ t ≤ q , α t = 1 / ( t +1) ω for any t &gt; q and S = q . If Assumption 1 holds with the exception that F may be nonconvex and Condition 4 is satisfied, then given δ ∈ (0 , 1) , with probability at least 1 -δ ∑ 0 ≤ i&lt;t -1 1 / ( i +1)( i +2) , for any t &gt; q +1 , it holds that

<!-- formula-not-decoded -->

Proof of Lemma 16. By Lemma 3, for any t ≥ q +1 , we have

<!-- formula-not-decoded -->

Applying Lemma 15, with probability at least 1 -δ ∑ 0 ≤ i ≤ q 1 / ( i +1)( i +2) , we have

<!-- formula-not-decoded -->

Combining these two bounds with Proposition 4 then yields

<!-- formula-not-decoded -->

The proof concludes using the simple inequality σ q +1 F ( x q +1 ) ≤ ς ( q +2) sup z ∈ X | F ( z ) | .

We are now well-equipped to prove Theorem 4.

Proof of Theorem 4. We first focus on the terms involving F ( x i ) in Lemma 16. Note that the function F is defined with respect to X opt , not X . Therefore, F ( x i ) can take both positive and negative values since x i ∈ X . However, we can derive the following bound

<!-- formula-not-decoded -->

where α = log( q + 1) / ( q + 1) , and the second and third inequalities follow from the geometric series bound and the definition of σ t . Using this bound together with Lemma 16, we arrive at

<!-- formula-not-decoded -->

We next focus on the upper-level problem. Since G ( x t +1 ) ≥ 0 , using Lemma 16 and the definition of the sequence { β t } t ≥ 0 , we obtain

<!-- formula-not-decoded -->

Observe next that

<!-- formula-not-decoded -->

where a and b are constants defined to make the equality hold. Combining these bounds, we arrive at

<!-- formula-not-decoded -->

Both bounds involving G and F hold with probability at least 1 -δ . We want the right hand side of both bounds → 0 . In order to guarantee this, we choose p, ω to minimize the slowest rate in terms of t :

<!-- formula-not-decoded -->

which is realized by setting p = 1 / 2 , ω = 3 / 4 as required.

As for the second claim on asymptotic convergence, we prove a more general result: If

<!-- formula-not-decoded -->

then the asymptotic convergence holds. Under this additional assumption and the fact that G ( x t ) ≥ 0 , we deduce that lim t →∞ G ( x t ) = 0 from taking the limit on both sides of (31). Recall that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Combining these observations and (32), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the Stolz-Cesàro theorem (since p ∈ (0 , 1) , ∑ t ≥ 0 β t = ∞ ) and taking the limit on both sides of the above inequality, we obtain lim inf t →∞ F ( x t ) ≤ 0 . Since this holds with probability at least 1 -δ for any δ ∈ (0 , 1) , it also holds almost surely. Due to the fact that any limit point of { x t } t is in X opt , the continuity of F from Lemma 4 and the fact that F ( x ) ≥ 0 for any x ∈ X opt , we deduce that lim inf t →∞ F ( x t ) ≥ 0 . Thus, we conclude the proof.

## D Additional Discussions

## D.1 Finiteness of constants

The following lemma establishes that the quantities introduced in (14) are finite.

Lemma 17. If Assumption 1 and Conditions 1-3 hold, then C, ¯ C δ , C δ defined in (14) are finite. Furthermore, if L defined in Condition 2 is strictly less than 1 , then V defined in (14d) is finite.

Proof of Lemma 17. We first show that under Condition 1 and Condition 3, when the limit in Condition 2 exists, the parameter L satisfies 0 ≤ L ≤ 1 . While it is trivial to see L ≥ 0 thanks to Condition 1, L ≤ 1 needs some justifications. For the sake of contradiction, suppose L &gt; 1 . Then for some sufficiently large t , we have

<!-- formula-not-decoded -->

This further implies

<!-- formula-not-decoded -->

which always contradicts Condition 3. Thus, we have 0 ≤ L ≤ 1 under our blanket assumptions.

We now show that C is finite. Observe that

<!-- formula-not-decoded -->

where the first equality is obtained by diving both the numerator and the denominator by tσ t and the second equality follows from Condition 2. Moreover, we have

<!-- formula-not-decoded -->

where the second equality follows from the Stolz-Cesàro theorem. This yields

<!-- formula-not-decoded -->

Using Condition 2, we have min t ≥ 0 ( t + 1) σ t &gt; 0 . Thus, C is finite. It is also straightforward to verify that ¯ C δ is finite by Condition 3. Consequently, C δ is finite as well.

To conclude, we show V is finite. From Condition 3, { tσ t } t ≥ 0 and { ( t +1) tσ 2 t } t ≥ 0 are increasing and diverge to ∞ . Note that

<!-- formula-not-decoded -->

where the first equality is implied by the Stolz-Cesàro theorem, the second equality is obtained from dividing both the denominator and the numerator by ( t +1) σ 2 t +1 , and the third equality comes from Condition 3 and

<!-- formula-not-decoded -->

Note that L/ 2(1 -L ) ∈ [0 , ∞ ) since L ∈ [0 , 1) . Thus, V is finite. This completes the proof.

The following lemma establishes that the quantities introduced in (25) are finite.

Lemma 18. If Assumption 1, Condition 1, 2 and 4 hold, then C q , ¯ C δ,q defined in (25b) , and (25c) are finite. Furthermore, if L defined in Condition 2 is less than 1 , then V q defined in (25d) is finite.

Proof of Lemma 18. The proof follows a similar argument to that of Lemma 17. Details are omitted for brevity.

## D.2 Over-parameterized regression: implementation details

The implementation details are as follows. For IR-SCG , presented in Algorithm 1, we set σ t = ς ( t +1) -1 / 4 , where ς = 10 , along with α t = 2 / ( t +2) . For IR-FSCG , presented in Algorithm 2, we set S = q = ⌊ √ n ⌋ , σ t = ς (max { t, q } + 1) -1 / 2 , where ς = 10 , along with α t = 2 / ( t + 1) for every t ≥ q and α t = log( q ) /q for every t &lt; q . Our linear minimization oracle involves optimization over an ℓ 1 -norm ball, which admits an analytical solution; see [26, Section 4.1]. We compare our proposed methods against SBCGI with stepsize γ t = 0 . 01 / ( t +1) , SBCGF with constant stepsize γ t = 10 -5 (both using K t = 10 -4 / √ t +1 , S = q = ⌊ √ n ⌋ ), aR-IP-SeG with long ( γ t = 10 -2 / ( t +1) 3 / 4 ) and short stepsizes ( γ t = 10 -7 / ( t +1) 3 / 4 , ρ t = 10 3 ( t +1) 1 / 4 , r = 0 . 5 ), and SDBGD with long ( γ t = 10 -2 ) and short ( γ t = 10 -6 ) stepsizes. We initialized all algorithms with randomized starting points. For SBCGI and SBCGF , we generated the required initial point x ′ 0 by running the SPIDER-FW algorithm [57, Algorithm 2] on the inner-level problem in (5), using a stepsize of γ t = 0 . 1 / ( t + 1) . The initialization process terminated either after 10 5 stochastic oracle queries or when the computation time exceeded 100 seconds, with the resulting point serving as x 0 for both SBCGI and SBCGF . The implementation of these algorithms rely on certain linear optimization or projection oracles. For SBCGI and SBCGF , a linear optimization oracle over the ℓ 1 -norm ball intersecting with a half-space is required, which we employ CVXPY [11] to solve the problems, similar to the implementation in [4, Appendix F.1]. To compute the projection onto the feasible set required for aR-IP-SeG , we used [13, Algorithm 1]. To approximate the outer- and inner-level optimal values F opt , G opt , we again employ CVXPY to solve the inner-level problem and the reformulation (2) of the bilevel problem.

## D.3 Dictionary learning: implementation details

In our experiment, we obtain A,A ′ , ˆ X from the code provided by Cao et al. [4] with n = n ′ = 250 and p = 40 , q = 50 . In addition, we also choose δ = 3 . For IR-SCG , presented in Algorithm 1, we set σ t = ς ( t + 1) -2 / 7 and α t = ( t + 1) -6 / 7 , where ς = 0 . 1 . For IR-FSCG , presented in Algorithm 2, we set S = q = ⌊ √ n ⌋ , σ t = ς (max { t, q + 1 } + 1) -1 / 2 , where ς = 0 . 1 , along with α t = ( t + 1) -3 / 4 for every t ≥ q + 1 and α t = log( q + 1) / ( q + 1) for every t ≤ q . In both experiments, the linear optimization oracle over the ℓ 2 -norm ball admits an analytical solution; see [26, Section 4.1]. For performance comparison, we implement four algorithms: SBCGI [4, Algorithm 1] with stepsize γ t = 0 . 1( t + 1) -2 / 3 , SBCGF [4, Algorithm 2] with constant stepsize γ t = 10 -3 (both using K t = 0 . 01( t + 1) -1 / 3 , S = q = ⌊ √ n ⌋ ), aR-IP-SeG [27] with long ( γ t = 10 -2 / ( t +1) 3 / 4 ) and short stepsizes ( γ t = 10 -4 / ( t +1) 3 / 4 , ρ t = ( t +1) 1 / 4 , r = 0 . 5 ), and the stochastic variant of the dynamic barrier gradient descent ( SDBGD ) [21] with long ( γ t = 10 -2 ) and short ( γ t = 5 × 10 -3 ) stepsizes. We initialized all algorithms with randomized starting points. For SBCGI and SBCGF , we generated the required initial point x ′ 0 by running the SPIDER-FW algorithm [57, Algorithm 2] on the inner-level problem in (5), using a stepsize of γ t = 0 . 1 / ( t + 1) . The initialization process terminated either after 10 5 stochastic oracle queries, with the resulting point serving as x 0 for both SBCGI and SBCGF . The implementation of these algorithms rely on certain linear optimization or projection oracles. For SBCGI and SBCGF , a linear optimization oracle over

the ℓ 2 -norm ball intersecting with a half-space is required, which we employ CVXPY [11] to solve the problems, similar to the implementation in [4, Appendix F.1]. To compute the projection onto the feasible set required for aR-IP-SeG , we used [13, Algorithm 1] to compute projection on ℓ 1 norm ball and projection onto ℓ 2 norm ball admits an analytical solution. To approximate inner-level optimal value G opt , we again employ CVXPY to solve the inner-level problem.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We have reviewed all claims made and checked that they accurately reflect the contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We describe explicit limitations throughout the work, and in particular in the experimental section.

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

Justification: All claims are precise, and are either referenced, or proven in the appendix.

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

Justification: Yes, all details are provided in the main text and appendix.

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

Justification: Code is provided in supplementary material.

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

Justification: The experiment details are fully described in the main text and appendix.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Yes, all experiments include sample sizes to assess variability.

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

Justification: This paper only studies relatively small-scale problems compared to other machine learning areas, and each individual experiment can run on an ordinary laptop.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We have checked and believe that we conform to the guidelines.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: This is a theory-and-methods-focused paper with no explicit societal application or impact.

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

Justification: We do not release a dataset or model.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We do not use any assets except publicly-available Python packages, which are documented as dependencies for our code, and their companion papers are cited.

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

Justification: We do not introduce any new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: There are no human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: There are no human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: There are no LLM usage.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.