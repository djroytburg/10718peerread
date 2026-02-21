## Stochastic Regret Guarantees for Online Zeroth- and First-Order Bilevel Optimization

## Parvin Nazari

Amirkabir University of Technology p.nazari17@gmail.com

## Davoud Ataee Tarzanagh ∗

Samsung SDS Research America d.tarzanagh@samsung.com

## Bojian Hou

University of Pennsylvania bojianh@upenn.edu

## Li Shen

University of Pennsylvania li.shen@pennmedicine.upenn.edu

## George Michailidis

University of California, Los Angeles gmichail@ucla.edu

## Abstract

Online bilevel optimization (OBO) is a powerful framework for machine learning problems where both outer and inner objectives evolve over time, requiring dynamic updates. Current OBO approaches rely on deterministic window-smoothed regret minimization, which may not accurately reflect system performance when functions change rapidly. In this work, we introduce a novel search direction and show that both first- and zeroth-order (ZO) stochastic OBO algorithms leveraging this direction achieve sublinear stochastic bilevel regret without window smoothing. Beyond these guarantees, our framework enhances efficiency by: (i) reducing oracle dependence in hypergradient estimation, (ii) updating inner and outer variables alongside the linear system solution, and (iii) employing ZO-based estimation of Hessians, Jacobians, and gradients. Experiments on online parametric loss tuning and black-box adversarial attacks validate our approach.

## 1 Introduction

Bilevel optimization (BO) minimizes an outer objective dependent on an inner problem's solution. Originating in game theory [64] and formalized in mathematical optimization [9], BO finds applications in operations research, engineering, economics [16], and image processing [14]. Recently, BO has gained traction in machine learning, including hyperparameter optimization [22], meta-learning [18], reinforcement learning [65], and neural architecture search [51].

In the offline setting, BO solves the following problem:

<!-- formula-not-decoded -->

where f and g are the outer and inner objectives, with x and y as their respective variables.

OBO [67] addresses dynamic scenarios where objectives evolve over time, requiring the agent to update the outer decision in response to the optimal inner decision. Similar to online singlelevel optimization (OSO) [72], OBO involves iterative decision-making without prior knowledge of outcomes [67, 50, 8]. Let T be the total number of rounds. Define x t ∈ X ⊂ R d 1 as the

∗ Corresponding author

Table 1: Comparison of OBO algorithms based on regret window w , solver iterations, stochastic/constrained regrets, feedback type, and local bounds. κ g denotes the condition number of the inner objective g t . V T , H p,T , ∆ T , Ψ T , ˆ ∆ T , ˆ Ψ T , σ , and ˆ σ are defined in (11), (14), (30), (10), and (28), respectively.

| OBO Method   | Window Size in Regret ( w )   | System Iters.     | Stochastic Regret   | Const. Regret Min.   | Only Func. Feedback   | Local Regret Bound                                                      |
|--------------|-------------------------------|-------------------|---------------------|----------------------|-----------------------|-------------------------------------------------------------------------|
| OAGD [67]    | o ( T )                       | N.A. (Exact)      | ✗                   | ✗                    | ✗                     | T w + H 1 ,T + H 2 ,T                                                   |
| SOBOW [50]   | o ( T )                       | O ( κ g log κ g ) | ✗                   | ✗                    | ✗                     | T w + V T + H 2 ,T                                                      |
| SOBBO [8]    | o ( T )                       | O ( κ g log κ g ) | ✓                   | ✓                    | ✗                     | T w σ 2 + V T + H 2 ,T                                                  |
| SOGD         | 1                             | 1                 | ✓                   | ✓                    | ✗                     | T 1 3 ( σ 2 +∆ T )+ T 2 3 Ψ T                                           |
| ZO-SOGD      | 1                             | 1                 | ✓                   | ✓                    | ✓                     | ( d 1 + d 2 ) 3 4 T 1 3 (ˆ σ 2 + ˆ ∆ T ) +( d 1 + d 2 ) 3 2 T 2 3 ˆ Ψ T |

decision variable and f t : X × R d 2 → R as the outer function. Similarly, define y t ∈ R d 2 and g t : X × R d 2 → R for the inner problem, where y ∗ t ( x ) = argmin y ∈ R d 2 g t ( x , y ) . OBO can be seen as a single-player problem, where the player selects x t without knowing y ∗ t ( x ) , using y t as an estimate based on g t . Alternatively, it can be framed as a two-player game [64], where the leader ( x t ) competes with the follower ( y t ), who selects y ∗ t ( x ) based on limited knowledge of g t . This framework includes online and adversarial variants of (BO), such as online actor-critic algorithms [71], online metalearning [19], and online hyperparameter optimization [50]. The inner and outer functions may be time-varying, adversarial, unavailable a priori , and require nonstationary optimization.

Our Contributions. This paper addresses stochastic OBO, introducing novel first - and zeroth-order methods to minimize stochastic bilevel regret. Key contributions are summarized below.

- Stochastic regret minimization without window-smoothing. Existing OBO methods [67, 50, 40, 8] rely on deterministic window-smoothed regret minimization, which may not accurately reflect system performance when functions change rapidly. We address these limitations by introducing a novel search direction (Section 2) and proving that both first-order and ZO methods achieve sublinear stochastic bilevel regret without window-smoothing ( w = 1 ) ; see Theorems 2.6 and 3.2 and Table 1.
- OBO with function value oracle feedback. In large-scale and black-box settings [11, 58], firstand second-order information is often unavailable or costly. Constructing accurate (hyper)-gradient estimators using only function value oracles is particularly challenging due to BO's nested structure. Existing methods rely on gradient, Hessian, and Jacobian oracles, limiting scalability [21, 27]. We propose Algorithm 2, which estimates Hessians, Jacobians, and gradients using function value oracles, achieving sublinear local regret (Theorem 3.2).
- OBO with one subproblem solver iteration. A major challenge in BO is solving implicit systems to approximate the hypergradient [43, 12]. While efficient offline BO methods exist [43, 15], extending them to OBO is difficult due to time-varying objectives. SOBOW [50] partially addresses this using a conjugate gradient (CG) algorithm with increasing iterations (Table 1). We improve upon SOBOW by introducing Algorithms 1 and 2, which require only a single subproblem solver iteration.

## 2 Stochastic OBO with Access to First- and Inner Second-Order Oracles

Notation. R d is the d -dimensional real space; R d + and R d ++ denote its nonnegative and positive orthants. Bold lowercase letters (e.g., x , y ) represent vectors, ⟨ x , y ⟩ is the inner product, and ∥·∥ is the Euclidean norm. ∇ x denotes the gradient, and ∇ 2 xy = ∇ x ∇ y . A function is L -smooth if its gradient is L -Lipschitz. The projection onto a convex set X is Π X ( z ) = argmin x ∈X 1 2 ∥ x -z ∥ 2 . We use [ T ] for { 1 , . . . , T } , E [ · ] for expectation, and O ( · ) to hide problem-independent constants.

Stochastic OBO Setting . Let T be the total rounds [67]. Define x t ∈ X ⊂ R d 1 as the decision variable and f t : X × R d 2 as the outer objective. The inner decision variable and objective are y t ∈ R d 2 and g t : X × R d 2 , where the optimal inner decision is:

<!-- formula-not-decoded -->

Further, we have

<!-- formula-not-decoded -->

Here, ( D f,t , D g,t ) denote data distributions at time t . Our setting is stochastic, with only noisy evaluations of functions, gradients, and Hessians. Unlike OSO [72], where true losses are revealed, in OBO the outer function f t ( x , y ∗ t ( x )) is inaccessible for updating x t and is generally non-convex in x , making standard regret notions from online convex optimization [33] unsuitable.

Given a sequence { α t ∈ R ++ } T t =1 , we define the following notion of bilevel local regret :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The local regret (2) compares the leader's decision x t to the stationary points x ∗ t satisfying P X ,α t ( x ∗ t ; ∇ f t ( x ∗ t , y ∗ t ( x ∗ t ))) = 0 . This can also be viewed as dynamic local regret, as the baseline corresponds to a stationary point of the leader's objective f t .

Previous work on (nonconvex) OBO examined unconstrained local regret using window-smoothed objectives: F t,w ( x , y ) = (1 /w ) ∑ w -1 i =0 f t -i ( x , y ) . For w = 1 and X = R d 1 , this reduces to (2). [67, 50] showed that w = o ( T ) ensures sublinear regret under slow variations in { F t,w } T t =1 , while rapid changes can lead to deviations. However, smoothing may misrepresent regret (Figure 1). This paper introduces a new projection-based local regret notion (2) without smoothing, and establishes sublinear regret for constrained OBO.

Online Gradient Descent (OGD) . One of the most widely used algorithms for online (single-level) optimization is OGD [72]. The procedure for OGD is as follows: For each t ∈ [ T ] , the algorithm selects x t ∈ X , observes the function f t : X ⊂ R d → R , and updates according to

Figure 1: Smoothly and rapidly changing f t in OBO with g t ( x t , y t ) = ( y t -cos( x t )) 2 , a t = 1 + 0 . 5 sin( t ) , b t = 1 + sin(0 . 5 t ) , and c t = 10 b t .

<!-- image -->

<!-- formula-not-decoded -->

In the following, we adapt OGD to OBO and introduce a novel framework that requires limited feedback and can utilize ZO updates within a single-loop structure.

To adapt OGD to OBO, [67, 50, 8] developed a variant alternating between inner and outer OGD, achieving sublinear bilevel regret bounds. We introduce a new search direction that enables sublinear bilevel regret without window smoothing. To compute the hypergradient ∇ f t ( x , y ∗ t ( x )) where y ∗ t ( x ) is defined in (1), since ∇ y g t ( x , y ∗ t ( x )) = 0 , using the implicit function theorem, yields

<!-- formula-not-decoded -->

where v ∗ t ( x ) ∈ R d 2 is the solution to the following linear system:

<!-- formula-not-decoded -->

As the exact y ∗ t ( x ) is not available, we estimate the hypergradient of f t at ( x , y ) and introduce an auxiliary variable v := v ( x , y ) to effectively decouple the nonlinear structure in ∇ f t ( x , y ∗ t ( x )) , i.e.

<!-- formula-not-decoded -->

where v t serves as an inexact solution to the linear system

<!-- formula-not-decoded -->

## Algorithm 1 SOGD

Require: ( x 1 , y 1 , v 1 ) ∈ X × R d 2 × Z p ; p ∈ R ++ ; T ∈ N ; stepsizes { ( α t , β t , δ t ) ∈ R 3 ++ } T t =1 ; parameters { ( γ t , λ t , η t ) } T t =1 ∈ (0 , 1) ; z t := ( x t , y t ) . For t = 1 to T do :

S1. Draw samples B t and ¯ B t with batch sizes b and ¯ b . Get search directions d y t , d v t , and d x t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- S2. Update inner, system, and outer solutions:

<!-- formula-not-decoded -->

An accurate solution of (5b) is crucial for tight regret bounds. [67] assumes an exact solution, which is restrictive in large-scale settings. To address this, [50] proposed an efficient OBO algorithm with window averaging, using CG methods to solve (5b), which is equivalent to:

<!-- formula-not-decoded -->

Next, we introduce a novel search direction that enables both first- and ZO stochastic OBO algorithms to achieve sublinear bilevel regret without smoothing. We first state the following lemma:

Lemma 2.1. Let w = t , W = 1 /η and ν = 1 -η for η ∈ (0 , 1) in the window-smoothed gradient ∇ F t,ν ( x t , y t ; B t ) = 1 W ∑ w -1 i =0 ν i ∇ f t -i ( x t -i , y t -i ; B t -i ) , where B t := { ξ t, 1 , . . . , ξ t,b } is drawn i.i.d. from D f,t . Then, ∇ F t,ν ( x t , y t ; B t ) = ∑ t j =1 η (1 -η ) t -j ∇ f j ( x j , y j ; B j ) , and we have ∇ F t,ν ( x t , y t ; B t ) = d x t with d x t = η ∇ f t ( x t , y t ; B t ) + (1 -η ) d x t -1 , and d x 1 = 1 W ∇ f 1 ( x 1 , y 1 ; B 1 ) for all t ≥ 2 .

Proof is given in Appendix C.1. As shown in Lemma 2.1, for a specific choice of w and W , the time-smoothed gradient forms a recursive momentum-type search direction. However, achieving sublinear regret in stochastic OBO requires large-window smoothing ( w = o ( T ) ) [67, 50, 8]. To address this, we propose the following search direction:

<!-- formula-not-decoded -->

This direction is used for updating x , with similar updates for y and v , as discussed below and detailed in Algorithm 1. The quadratic formulation of (5b) in (6) motivates single-loop methods such as [15]. Building on this, we propose Simultaneous Online Gradient Descent (SOGD) for constrained OBO, presented in Algorithm 1. At each step, SOGD jointly updates the follower variable y t , auxiliary variable v t , and leader variable x t using batches B t = { ξ t, 1 , . . . , ξ t,b } and ¯ B t := { ζ t, 1 , . . . , ζ t, ¯ b } sampled i.i.d. from D f,t and D g,t . Step S1. only requires computing Hessian-vector products, avoiding explicit computation of ∇ 2 y g t or ∇ 2 xy g t . Step S2. uses the projection:

<!-- formula-not-decoded -->

Unlike OAGD [67] with alternating loops, and SOBOW [50] using CG, SOGD performs a single OGD step for all variables.

Assumption 2.2. g t ( x , y ) is twice continuously differentiable and µ g -strongly convex in y for all x ∈ X , t ∈ [ T ] .

Assumption 2.3. Let z = [ x ; y ] and z ′ = [ x ′ ; y ′ ] , where x , x ′ ∈ X and y , y ′ ∈ R d 2 . For any z , z ′ , and t ∈ [ T ] :

B1. ∃ ℓ f, 0 ∈ R + s.t. ∥ f t ( z ; ξ t ) -f t ( z ′ ; ξ t ) ∥ ≤ ℓ f, 0 ∥ z -z ′ ∥ ;

B2.

∃

ℓ

∈

R

s.t.

∥∇

f

(

z

;

ξ

)

-∇

f

(

z

′

;

ξ

)

∥ ≤

ℓ

∥

z

-

z

′

∥

;

- B3. ∃ ℓ g, 1 ∈ R + s.t. ∥∇ g t ( z ; ζ t ) -∇ g t ( z ′ ; ζ t ) ∥ ≤ ℓ g, 1 ∥ z -z ′ ∥ ; 2 2 ′ ′

B4.

∃

ℓ

f, g,

1

2

∈

R

+

+

s.t.

∥∇

t

g

t

(

z

;

t

ζ

t

)

-∇

g

t

(

z

t

;

ζ

t

)

f,

∥ ≤

1

ℓ

g,

2

∥

z

-

z

∥

.

Assumption 2.4. For any t ∈ [ T ] , | f t ( x , y ∗ t ( x )) | ≤ M for some M ∈ R ++ and any x ∈ X . Assumption 2.5. There exist constants σ g y , σ g yy , σ g xy , σ f y , σ f x such that, for all z = [ x , y ] :

<!-- formula-not-decoded -->

Throughout this paper, we define

<!-- formula-not-decoded -->

Assumptions 2.2 and 2.3 are standard in BO [12, 43] and OBO [67], and hold for many bilevel MLproblems [22]. Assumption 2.4 is typical in non-convex OSO [36, 50], while Assumption 2.5 assumes unbiased stochastic gradient, Hessian, and Jacobian estimators with bounded variance [12].

Achieving sublinear dynamic regret is generally infeasible under arbitrary time variations [7]. Prior analyses [67, 50] bound regret by enforcing regularity on the comparator sequence. To attain sublinear regret, [67] introduces the following regularity metrics for bilevel sequences:

<!-- formula-not-decoded -->

Path-length H p,T measures changes in the follower's costs, while V T captures the leader's objective smoothness. We use path-length for the follower and function variation for the leader due to the follower's strong convexity (Assumption 2.2) versus the leader's nonconvexity. Another regularity is the sequential gradient difference of the outer objective:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As in [41, 31], D x ,T and D y ,T measure the gradient drift of f t relative to f t -1 for x and y , respectively. We define deviations in the gradient, Hessian, and Jacobian of the inner objective as:

<!-- formula-not-decoded -->

We introduce the following notations for simplicity:

<!-- formula-not-decoded -->

where ( V T , H p,T ) are defined in (11), and

<!-- formula-not-decoded -->

By accounting for both D T and G T , we can represent the variations in the environments of OBO.

Theorem 2.6. Let { ( f t , g t ) } T t =1 be the sequence of functions presented to Algorithm 1, satisfying Assumptions 2.2-2.5. For all t ∈ [ T ] , let

<!-- formula-not-decoded -->

t

Here, c , c β , c δ , c γ , c η , and c λ are specified in (107) . Algorithm 1 guarantees:

<!-- formula-not-decoded -->

where σ and (∆ T , Ψ T ) are defined in (10) and (14) .

Remark 2.7 ( Stochastic Regret Guarantee for OBO and OSO with w = 1 ) . Theorem 2.6 bounds the regret of Algorithm 1 without window-smoothing, based on the regularities in (14). We note that the average dynamic regret BL-Reg T /T ≤ O ( T -2 / 3 ( σ 2 +∆ T ) + T -1 / 3 Ψ T ) remains sublinear under suitable conditions on ∆ T , Ψ T , and σ . Specifically, if ∆ T = o ( T 2 / 3 ) , Ψ T = o ( T 1 / 3 ) , and σ = o ( T 1 / 3 ) , then the dynamic regret grows sublinearly, i.e., BL-Reg T = o ( T ) ; see Appendix B.2 for further examples and discussion. This result also yields a sharper T -2 / 3 σ 2 regret-improving over the T -1 / 2 σ 2 bound for stochastic OBO [8]-and removes the need for window-smoothing [8, 67, 50, 40]. For OSO, this result surpasses the T -1 / 2 σ 2 rate in [31].

## 3 Stochastic OBO with Zeroth-Order Oracles

Black-box optimization arises when gradients are unavailable [11]. We study ZO-OBO methods with limited access to leader and follower objectives. Let s ∈ R d 1 and r ∈ R d 2 be vectors uniformly sampled from unit balls B 1 and B 2 . Given smoothing parameters ρ = ( ρ s , ρ r ) , we define Gaussiansmoothed objectives using [59]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To solve stochastic OBO with (18), we need to obtain the hyper-gradient of f t, ρ in (18) at ( x , y ) as

<!-- formula-not-decoded -->

Obtaining ˆ y ∗ t ( x ) in closed-form is usually a challenging task, so it is natural to use the following gradient surrogate. At any ( x , y ) , we introduce an auxiliary variable v = v ( x , y ) and define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To do so, we also introduce d y t, ρ , d v t, ρ and d x t, ρ as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, we approximate these directions using stochastic zeroth-order oracles ( SZO ), which produce the quantities ˆ ∇ y f t ( x , y ; ξ t ) , ˆ ∇ y g t ( x , y ; ζ t ) , ˆ ∇ x f t ( x , y ; ξ t ) , and ˆ ∇ x g t ( x , y ; ζ t ) . These are unbiased estimators of the true gradients ∇ y f t, ρ ( x , y ) , ∇ y g t, ρ ( x , y ) , ∇ x f t, ρ ( x , y ) , and ∇ x g t, ρ ( x , y ) , respectively, as shown in [20], such that the following assumption holds:

<!-- formula-not-decoded -->

Specifically, following [62], we estimate the gradient of a function h : R d → R , querying at x -λ s and x + λ s , yielding an estimator ( d/ 2 λ ) ( h ( x + λ s ) -h ( x -λ s )) s . Using this strategy, the finitedifference estimation of ∇ g t, ρ ( x , y ) , denoted by ˆ ∇ g t ( x , y ) , is constructed for given smoothing

## Algorithm 2 ZO-SOGD

Require: In addition to parameters in SOGD, choose ρ v , ρ r , ρ s ∈ R ++ . For t = 1 to T do :

S1. Draw samples B t and ¯ B t with batch sizes b and ¯ b . Using (24)-(26), get:

<!-- formula-not-decoded -->

ˆ

d

y

t

=

d

y

t

(

z

t

; ¯

B

t

) + (1

-

γ

t

)( ˆ

d

y

t

-

1

-

d

y

t

(

z

t

-

1

; ¯

B

t

))

,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- S2. Update inner, system, and outer solutions:

<!-- formula-not-decoded -->

parameters ρ = ( ρ s , ρ r ) , and a set ¯ B t = { ζ t, 1 , . . . , ζ t, ¯ b } drawn i.i.d. from D g,t , as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, we estimate ∇ y f t, ρ ( x , y ; B t ) and ∇ x f t, ρ ( x , y ; B t ) , respectively, using a batch B t = { ξ t, 1 , . . . , ξ t,b } drawn i.i.d. from D f,t , by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, given a smoothing parameter ρ v &gt; 0 , we approximate the Hessian-vector product ∇ 2 y g t, ρ ( x , y ) v and the Jacobian-vector product ∇ 2 xy g t, ρ ( x , y ) v as the finite difference between two gradients, respectively, as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using (24)-(26), the first-order terms in (9) are approximated by ˆ d y t , ˆ d v t , and ˆ d x t in (27). The approximations in (26a) and (26b) introduce errors in the hypergradient, which must be controlled. (26) depends on the dimension of y , as in ZO optimization [59, 62]. The projection Π Z p in (8) bounds v , controlling variance in v and x updates for convergence.

Assumption 3.1. There exist constants ˆ σ g y , ˆ σ g x , ˆ σ f y , ˆ σ f x such that, for all z = [ x , y ] :

<!-- formula-not-decoded -->

Assumption 3.1 is analogous to the upper bound on the variance of stochastic partial gradients discussed in [54, 68]. We simplify the notation by introducing the following shorthand.

<!-- formula-not-decoded -->

Next, we establish a regret bound for ZO-SOGD. Similar to the previous results, we introduce regularity conditions for the smoothed functions defined in (18) and (19).

Inner Gradient Variations : In ZO setting, we use a set of gradient variations at the perturbed point as follows:

<!-- formula-not-decoded -->

where z + t := ( x t -1 , y t -1 + ρ v v t -1 ) , z -t := ( x t -1 , y t -1 -ρ v v t -1 ) , and

<!-- formula-not-decoded -->

Further, for simplicity of notation, we define

<!-- formula-not-decoded -->

where ( V T , H p,T ) and ( E 1 , D T ) are defined in (11), and (15), respectively. Moreover, G y ,T and ( G v ,T , G x ,T ) are defined in (13) and (29), respectively.

Theorem 3.2. Let { ( f t , g t ) } T t =1 be the sequence of functions presented to Algorithm 2, satisfying Assumptions 2.2-2.4 and 3.1. For all t ∈ [ T ] , let

<!-- formula-not-decoded -->

where c , c β , c δ , c γ , c η , c v , and c λ are specified in (226) . Let p = ℓ f, 0 /µ g for the set Z p defined in (8) . Then, Algorithm 2 guarantees:

<!-- formula-not-decoded -->

where ˆ σ 2 and ( ˆ ∆ T , ˆ Ψ T ) are defined in (28) and (30) .

Theorem 3.2 bounds the regret of Algorithm 2 without window-smoothing, based on the regularities in (30). We note that the average dynamic regret BL-Reg T /T ≤ O (( d 1 + d 2 ) 3 / 4 T -2 / 3 ( ˆ σ 2 + ˆ ∆ T ) + ( d 1 + d 2 ) 3 / 2 T -1 / 3 ˆ Ψ T ) remains sublinear under suitable conditions on ˆ ∆ T , ˆ Ψ T , and ˆ σ .

Remark 3.3 ( Regret Guarantee for Zeroth Order OBO ) . Theorem 3.2 provides the first regret guarantee for OBO with access only to noisy function evaluations of the leader and follower. The dimensional dependence O ( d 1 + d 2 ) in Theorem 3.2 aligns with optimal results for simpler offline min-max problems [39]. The bound also depends on the sample sizes b, ¯ b and smoothing parameters ρ v , ρ r , ρ s at each iteration.

Remark 3.4 ( Improved Regret for OSO ) . Our dynamic regret for single-level non-stationary optimization is O (( d 1 + d 2 ) 3 / 4 T -2 / 3 (ˆ σ 2 + E 1 + V T + D T )) , improving the result in [60], which is O ( T -1 / 2 σ 2 √ d ) . [60] proposed a zeroth-order stochastic gradient descent algorithm for unconstrained, non-convex, time-varying objective functions, achieving a regret bound of O ( T -1 / 2 σ 2 √ dW T ) using a two-point gradient estimator, where W T bounds the nonstationarity. Additionally, [29] showed that the local regret for standard online stochastic gradient descent with the standard two-point gradient estimator [1] is O ( T -1 / 2 d √ V T ) .

## 4 Experimental Results

In this section, we present experimental results for two applications: online black-box attacks on deep neural networks and parametric loss tuning for imbalanced data. Code is available at /github . Additional experiments and details on hyperparameter tuning are provided in Appendix E.

<!-- formula-not-decoded -->

Figure 2: Performance comparison (mean ± std) of optimizers including ZO-O-GD, ZO-O-Adam, ZO-O-SignSGD, ZO-O-ConservSGD, ZO-SOGD, and ZO-SOGD (Adam) on online adversarial attack for MNIST data across five runs.

<!-- image -->

Bilevel Optimization for Black-Box Adversarial Attacks (BBAA) Deep neural networks are vulnerable to adversarial examples-inputs subtly perturbed to mislead classifiers. These examples can fool models without access to their internals, as in [11, 52, 13]. We first review the ZO single-level formulation for BBAA [11]. Let ( a , b ) be a clean image a ∈ R d with label b ∈ { 1 , . . . , J } , and define a ′ = a + y , where y is the adversarial perturbation. Let Y := [ -5 , 5] d , and ℓ : R d → R be the black-box attack loss. For a given hyperparameter λ &gt; 0 , the BBAA problem is:

<!-- formula-not-decoded -->

To adapt (32) to our OBO, consider OBO for supervised learning: at each timestep t , new samples ( a t , b t ) ∈ D t := {D val t , D tr t } are received, where a t ∈ R d 2 is the feature vector (image) and b t ∈ R is the corresponding target. Note that the correct decision can change abruptly. We consider an S -stage scenario where ( x ∗ s , y ∗ s ( x ∗ s )) represents the best decisions for the s -th stage, for all s ∈ [ S ] :

<!-- formula-not-decoded -->

Here, { a ( i ) t } i ∈D tr t and { a ( i ) t } i ∈D val t are batches of training and validation samples at timestep t ; a ( i ) t is the i th sample in that batch; and [ x t ] ι and [ y t ] ι denote the ι th component of x t and y t , respectively.

̸

We normalize the pixel values to Y . For an untargeted attack, the loss in (34) is ℓ ( a ′ t ) = max { Z ( a ′ t ) b t -max j = b t Z ( a ′ t ) j , -κ } , where Z ( a ′ t ) j is the prediction score for class j given input a ′ t = a t + y t , and κ &gt; 0 controls the confidence gap. In our experiments, we set κ = 0 . Eq. (33) introduces the first OBO formulation of BBAA. Using a vector x ∈ R d + for hyperparameters instead of λ ∈ R ++ in (32) enables finer control over model components, enhancing performance for complex models and heterogeneous data [53]. For a fair comparison with single-level BBAA, we replace λ with a fixed vector multiplied by each component of y in (32). We compare our ZO-SOGD and ZO-SOGD (Adam) with the following competing methods in the online setting: ZO-O-GD , a single-level method that updates y t with a fixed x at each timestep using ZO gradient descent [59]; ZO-O-Adam , a single-level method that updates y t with a fixed x at each timestep using ZO Adam [45, 13]; ZO-O-SignSGD , a single-level method that updates y t with a fixed x at each timestep using ZO SignSGD [6]; and ZO-O-ConservSGD , a single-level method that updates y t with a fixed x at each timestep using ZO Conservative SGD [44]. Note that ZO-SOGD (ours, Adam) is a variant of our algorithm with an adaptive stepsize, similar to that of [45].

We evaluated the proposed algorithms based on runtime, test accuracy on perturbed samples, and the infinity norm of y t . Figure 2 compares the methods. The left panel shows that ZO-SOGD has a slower runtime than single-level baselines due to outer-level optimization on x . The middle panel illustrates that accuracy decreases as the adversarial attack y strengthens, with ZO-SOGD outperforming ZO-O-GD and ZO-O-ConservGD, while ZO-SOGD (Adam) surpasses ZO-O-Adam and all baselines. The right panel indicates that the infinity norm of y t increases over time for all methods, reducing accuracy. However, perturbations remain minor, with max y t not exceeding 4, demonstrating that ZO-SOGD achieves effective attacks with superior performance.

Figure 3: Performance (mean ± std) on online parametric loss tuning with distribution shift on MNIST across five runs, comparing OGD [72], OAGD [67], SOBOW [50], and our SOGD.

<!-- image -->

Parametric Loss Tuning for Imbalanced Data Imbalanced datasets are common in modern machine learning, causing challenges in generalization and fairness due to underrepresented classes and sensitive attributes. Deep NNs often overfit, seeming accurate and fair during training but performing poorly during testing. A common solution is designing a parametric training loss that balances accuracy and fairness while preventing overfitting [49]. We consider an optimization problem similar to that in (33). For a new sample ( a t , b t ) , the follower and leader incur a parametric and balanced cross-entropy loss, respectively:

<!-- formula-not-decoded -->

Here, x t := (∆ j , γ j ) J j =1 represents the logit adjustments, with j indexing the J classes, and u j is the reciprocal of the proportion of samples from the j -th class to the total number of samples [49].

In (35), y t ( x t ) is the follower conditioned on the leader, and [ y t ( a t )] b t is the logit for class b t on sample a t . The follower y t uses a 4-layer CNN, inducing a nonconvex bilevel objective. We compare SOGD with OAGD [67], a static method using the Neumann series, and SOBOW [50], a dynamic method using conjugate gradients (CG). Experiments were conducted on MNIST [48] with batch size 64. We evaluated cumulative runtime, test accuracy, and balanced accuracy, defined as 1 J ∑ J j =1 P a t ∼D j [ argmax i ([ y t ( a t )] i ) = j ] , where D j is the classj sample distribution [49]. Learning rates were tuned as β t = δ t = β ∈ { 0 . 001 , 0 . 005 , 0 . 01 , 0 . 05 , 0 . 1 } , α t = α ∈ { 0 . 0001 , 0 . 0005 , 0 . 001 , 0 . 005 , 0 . 01 } , and γ t = λ t = η t = γ ∈ { 0 . 9 , 0 . 99 , 0 . 999 } . Both OAGD and SOBOW used 5 iterations for their respective system solvers.

We evaluated performance over 400 timesteps in four 100-timestep phases, transitioning from an imbalanced ( 0 . 4 i ) to a balanced ( 0 . 8 i ) distribution for each class ( i = 0 , 1 , . . . , 9 ). Figure 3 (left) shows SOBOW's longer runtime due to CG complexity, while SOGD is the fastest with simultaneous updates. Figures 3 (middle, right) show accuracy gains as balance increases, with SOGD achieving competitive accuracy.

## 5 Conclusion

This work introduced a novel online bilevel optimization framework that overcomes the limitations of existing algorithms, which often depend on extensive oracle information and incur high computational costs. Our method leverages limited feedback and zeroth-order updates for efficient hypergradient estimation and simultaneous updates of decision variables, achieving sublinear bilevel regret without window smoothing. Experiments on online parametric loss tuning and black-box adversarial attacks validate its effectiveness. A limitation of this study is that the results focus on nonconvex regret bounds, without extending guarantees to convex settings.

## Acknowledgments and Disclosure of Funding

We thank the reviewers for their valuable comments. The work of DAT was supported by Samsung SDS Research America, Mountain View. The work of GM was supported in part by NSF grants DMS-2348640 and DMS-2319552.

## References

- [1] Alekh Agarwal, Ofer Dekel, and Lin Xiao. Optimal algorithms for online convex optimization with multi-point bandit feedback. In Colt , pages 28-40. Citeseer, 2010.
- [2] Naman Agarwal, Alon Gonen, and Elad Hazan. Learning in non-convex games with an optimization oracle. In Conference on Learning Theory , pages 18-29. PMLR, 2019.
- [3] Alireza Aghasi and Saeed Ghadimi. Fully zeroth-order bilevel programming via gaussian smoothing. arXiv preprint arXiv:2404.00158 , 2024.
- [4] Zeyuan Allen-Zhu and Yuanzhi Li. Neon2: Finding local minima via first-order oracles. Advances in Neural Information Processing Systems , 31, 2018.
- [5] Francis Bach and Vianney Perchet. Highly-smooth zero-th order online optimization. In Conference on Learning Theory , pages 257-283. PMLR, 2016.
- [6] Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Anima Anandkumar. Signsgd: Compressed optimisation for non-convex problems. In International Conference on Machine Learning , pages 560-569. PMLR, 2018.
- [7] Omar Besbes, Yonatan Gur, and Assaf Zeevi. Non-stationary stochastic optimization. Operations research , 63(5):1227-1244, 2015.
- [8] Jason Bohne, David Rosenberg, Gary Kazantsev, and Pawel Polak. Online nonconvex bilevel optimization with bregman divergences. arXiv preprint arXiv:2409.10470 , 2024.
- [9] Jerome Bracken and James T McGill. Mathematical programs with optimization problems in the constraints. Operations Research , 21(1):37-44, 1973.
- [10] Sébastien Bubeck, Gilles Stoltz, Csaba Szepesvári, and Rémi Munos. Online optimization in x-armed bandits. Advances in Neural Information Processing Systems , 21, 2008.
- [11] Pin-Yu Chen, Huan Zhang, Yash Sharma, Jinfeng Yi, and Cho-Jui Hsieh. Zoo: Zeroth order optimization based black-box attacks to deep neural networks without training substitute models. In Proceedings of the 10th ACM workshop on artificial intelligence and security , pages 15-26, 2017.
- [12] Tianyi Chen, Yuejiao Sun, and Wotao Yin. Closing the gap: Tighter analysis of alternating stochastic gradient methods for bilevel problems. Advances in Neural Information Processing Systems , 34, 2021.
- [13] Xiangyi Chen, Sijia Liu, Kaidi Xu, Xingguo Li, Xue Lin, Mingyi Hong, and David Cox. Zoadamm: Zeroth-order adaptive momentum method for black-box optimization. Advances in neural information processing systems , 32, 2019.
- [14] Caroline Crockett, Jeffrey A Fessler, et al. Bilevel methods for image reconstruction. Foundations and Trends® in Signal Processing , 15(2-3):121-289, 2022.
- [15] Mathieu Dagréou, Pierre Ablin, Samuel Vaiter, and Thomas Moreau. A framework for bilevel optimization that enables stochastic and global variance reduction algorithms. arXiv preprint arXiv:2201.13409 , 2022.
- [16] Stephan Dempe. Foundations of bilevel programming . Springer Science &amp; Business Media, 2002.
- [17] John C Duchi, Michael I Jordan, Martin J Wainwright, and Andre Wibisono. Optimal rates for zero-order convex optimization: The power of two function evaluations. IEEE Transactions on Information Theory , 61(5):2788-2806, 2015.
- [18] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning , pages 1126-1135. PMLR, 2017.

- [19] Chelsea Finn, Aravind Rajeswaran, Sham Kakade, and Sergey Levine. Online meta-learning. In International Conference on Machine Learning , pages 1920-1930. PMLR, 2019.
- [20] Abraham D Flaxman, Adam Tauman Kalai, and H Brendan McMahan. Online convex optimization in the bandit setting: gradient descent without a gradient. arXiv preprint cs/0408007 , 2004.
- [21] Luca Franceschi, Michele Donini, Paolo Frasconi, and Massimiliano Pontil. Forward and reverse gradient-based hyperparameter optimization. In International Conference on Machine Learning , pages 1165-1173. PMLR, 2017.
- [22] Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In International Conference on Machine Learning , pages 1568-1577. PMLR, 2018.
- [23] Xiand Gao, Xiaobo Li, and Shuzhong Zhang. Online learning with non-convex losses and non-stationary regret. In International Conference on Artificial Intelligence and Statistics , pages 235-243. PMLR, 2018.
- [24] Xiang Gao, Bo Jiang, and Shuzhong Zhang. On the information-adaptive variants of the admm: an iteration complexity perspective. Journal of Scientific Computing , 76:327-363, 2018.
- [25] Saeed Ghadimi and Guanghui Lan. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM journal on optimization , 23(4):2341-2368, 2013.
- [26] Saeed Ghadimi, Guanghui Lan, and Hongchao Zhang. Mini-batch stochastic approximation methods for nonconvex stochastic composite optimization. Mathematical Programming , 155(12):267-305, 2016.
- [27] Saeed Ghadimi and Mengdi Wang. Approximation methods for bilevel programming. arXiv preprint arXiv:1802.02246 , 2018.
- [28] Gautam Goel, Yiheng Lin, Haoyuan Sun, and Adam Wierman. Beyond online balanced descent: An optimal algorithm for smoothed online optimization. Advances in Neural Information Processing Systems , 32, 2019.
- [29] Ziwei Guan, Yi Zhou, and Yingbin Liang. On the hardness of online nonconvex optimization with single oracle feedback. In The Twelfth International Conference on Learning Representations , 2023.
- [30] Ziwei Guan, Yi Zhou, and Yingbin Liang. Online nonconvex optimization with limited instantaneous oracle feedback. In The Thirty Sixth Annual Conference on Learning Theory , pages 3328-3355. PMLR, 2023.
- [31] Nadav Hallak, Panayotis Mertikopoulos, and Volkan Cevher. Regret minimization in stochastic non-convex learning via a proximal-gradient approach. In International Conference on Machine Learning , pages 4008-4017. PMLR, 2021.
- [32] Pierre Hansen, Brigitte Jaumard, and Gilles Savard. New branch-and-bound rules for linear bilevel programming. SIAM Journal on scientific and Statistical Computing , 13(5):1194-1217, 1992.
- [33] Elad Hazan. Introduction to online convex optimization. Foundations and Trends in Optimization , 2(3-4):157-325, 2016.
- [34] Elad Hazan. Introduction to online convex optimization. Foundations and Trends® in Optimization , 2(3-4):157-325, 2016.
- [35] Elad Hazan, Amit Agarwal, and Satyen Kale. Logarithmic regret algorithms for online convex optimization. Machine Learning , 69(2):169-192, 2007.
- [36] Elad Hazan, Karan Singh, and Cyril Zhang. Efficient regret minimization in non-convex games. In International Conference on Machine Learning , pages 1433-1441. PMLR, 2017.

- [37] Amélie Héliou, Matthieu Martin, Panayotis Mertikopoulos, and Thibaud Rahier. Online nonconvex optimization with imperfect feedback. Advances in Neural Information Processing Systems , 33:17224-17235, 2020.
- [38] Amélie Héliou, Matthieu Martin, Panayotis Mertikopoulos, and Thibaud Rahier. Zeroth-order non-convex learning via hierarchical dual averaging. In International Conference on Machine Learning , pages 4192-4202. PMLR, 2021.
- [39] Feihu Huang, Shangqian Gao, Jian Pei, and Heng Huang. Accelerated zeroth-order and firstorder momentum methods from mini to minimax optimization. Journal of Machine Learning Research , 23(36):1-70, 2022.
- [40] Yu Huang, Yuan Cheng, Yingbin Liang, and Longbo Huang. Online min-max problems with non-convexity and non-stationarity. Transactions on Machine Learning Research , 2023.
- [41] Yu Huang, Yuan Cheng, Yingbin Liang, and Longbo Huang. Online min-max problems with non-convexity and non-stationarity. Transactions on Machine Learning Research , 2023.
- [42] Kaiyi Ji, Zhe Wang, Yi Zhou, and Yingbin Liang. Improved zeroth-order variance reduced algorithms and analysis for nonconvex optimization. In International conference on machine learning , pages 3100-3109. PMLR, 2019.
- [43] Kaiyi Ji, Junjie Yang, and Yingbin Liang. Bilevel optimization: Convergence analysis and enhanced design. In International Conference on Machine Learning , pages 4882-4892. PMLR, 2021.
- [44] Bumsu Kim, HanQin Cai, Daniel McKenzie, and Wotao Yin. Curvature-aware derivative-free optimization. arXiv preprint arXiv:2109.13391 , 2021.
- [45] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations , 2014.
- [46] Robert Kleinberg, Aleksandrs Slivkins, and Eli Upfal. Multi-armed bandits in metric spaces. In Proceedings of the fortieth annual ACM symposium on Theory of computing , pages 681-690, 2008.
- [47] Walid Krichene, Maximilian Balandat, Claire Tomlin, and Alexandre Bayen. The hedge algorithm on a continuum. In International Conference on Machine Learning , pages 824-832. PMLR, 2015.
- [48] Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist , 2, 2010.
- [49] Mingchen Li, Xuechen Zhang, Christos Thrampoulidis, Jiasi Chen, and Samet Oymak. Autobalance: Optimized loss functions for imbalanced data. Advances in Neural Information Processing Systems , 34:3163-3177, 2021.
- [50] Sen Lin, Daouda Sow, Kaiyi Ji, Yingbin Liang, and Ness Shroff. Non-convex bilevel optimization with time-varying objective functions. Advances in Neural Information Processing Systems , 36, 2024.
- [51] Hanxiao Liu, Karen Simonyan, and Yiming Yang. Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055 , 2018.
- [52] Sijia Liu, Jie Chen, Pin-Yu Chen, and Alfred Hero. Zeroth-order online alternating direction method of multipliers: Convergence analysis and applications. In International Conference on Artificial Intelligence and Statistics , pages 288-297. PMLR, 2018.
- [53] Jonathan Lorraine, Paul Vicol, and David Duvenaud. Optimizing millions of hyperparameters by implicit differentiation. In International conference on artificial intelligence and statistics , pages 1540-1552. PMLR, 2020.
- [54] Luo Luo, Haishan Ye, Zhichao Huang, and Tong Zhang. Stochastic recursive gradient descent ascent for stochastic nonconvex-strongly-concave minimax problems. Advances in Neural Information Processing Systems , 33:20566-20577, 2020.

- [55] Yibing Lv, Tiesong Hu, Guangmin Wang, and Zhongping Wan. A penalty function method based on kuhn-tucker condition for solving linear bilevel programming. Applied Mathematics and Computation , 188(1):808-813, 2007.
- [56] Parvin Nazari, Ahmad Mousavi, Davoud Ataee Tarzanagh, and George Michailidis. A penaltybased method for communication-efficient decentralized bilevel programming. Automatica , 173:112039, 2025.
- [57] Parvin Nazari, Davoud Ataee Tarzanagh, and George Michailidis. Adaptive first-and zeroth-order methods for weakly convex stochastic optimization problems. arXiv preprint arXiv:2005.09261 , 2020.
- [58] Yu Nesterov. Smooth minimization of non-smooth functions. Mathematical programming , 103:127-152, 2005.
- [59] Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics , 17(2):527-566, 2017.
- [60] Abhishek Roy, Krishnakumar Balasubramanian, Saeed Ghadimi, and Prasant Mohapatra. Stochastic zeroth-order optimization under nonstationarity and nonconvexity. Journal of Machine Learning Research , 23(64):1-47, 2022.
- [61] Shai Shalev-Shwartz et al. Online learning and online convex optimization. Foundations and trends in Machine Learning , 4(2):107-194, 2011.
- [62] Ohad Shamir. An optimal algorithm for bandit and zero-order convex optimization with two-point feedback. Journal of Machine Learning Research , 18(52):1-11, 2017.
- [63] Daouda Sow, Kaiyi Ji, and Yingbin Liang. On the convergence theory for hessian-free bilevel algorithms. Advances in Neural Information Processing Systems , 35:4136-4149, 2022.
- [64] Heinrich von Stackelberg. Theory of the market economy. Oxford University Press , 1952.
- [65] Bradly Stadie, Lunjun Zhang, and Jimmy Ba. Learning intrinsic rewards as a bi-level optimization problem. In Conference on Uncertainty in Artificial Intelligence , pages 111-120. PMLR, 2020.
- [66] Arun Sai Suggala and Praneeth Netrapalli. Online non-convex learning: Following the perturbed leader is optimal. In Algorithmic Learning Theory , pages 845-861. PMLR, 2020.
- [67] Davoud Ataee Tarzanagh, Parvin Nazari, Bojian Hou, Li Shen, and Laura Balzano. Online bilevel optimization: Regret analysis of online alternating gradient methods. In International Conference on Artificial Intelligence and Statistics , pages 2854-2862. PMLR, 2024.
- [68] Zhongruo Wang, Krishnakumar Balasubramanian, Shiqian Ma, and Meisam Razaviyayn. Zerothorder algorithms for nonconvex minimax problems with improved complexities. arXiv preprint arXiv:2001.07819 , 2020.
- [69] Yifan Yang, Peiyao Xiao, and Kaiyi Ji. Achieving O ( ϵ -1 . 5 ) complexity in hessian/jacobian-free stochastic bilevel optimization. arXiv preprint arXiv:2312.03807 , 2023.
- [70] Yan Zhang, Yi Zhou, Kaiyi Ji, and Michael M Zavlanos. Boosting one-point derivative-free online optimization via residual feedback. arXiv preprint arXiv:2010.07378 , 2020.
- [71] Wei Zhou, Yiying Li, Yongxin Yang, Huaimin Wang, and Timothy Hospedales. Online metacritic learning for off-policy actor-critic methods. Advances in Neural Information Processing Systems , 33:17662-17673, 2020.
- [72] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of the 20th international conference on machine learning (icml-03) , pages 928-936, 2003.

## Contents

| 1                               | Introduction                                                           | Introduction                                                                   |   1 |
|---------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------|-----|
| 2                               | Stochastic OBO with Access to First- and Inner Second-Order Oracles    | Stochastic OBO with Access to First- and Inner Second-Order Oracles            |   2 |
| 3                               | Stochastic OBO with Zeroth-Order Oracles                               | Stochastic OBO with Zeroth-Order Oracles                                       |   6 |
| 4                               | Experimental Results                                                   | Experimental Results                                                           |   8 |
| 5                               | Conclusion                                                             | Conclusion                                                                     |  10 |
| A                               | Related Work                                                           | Related Work                                                                   |  16 |
| B                               | Additional Preliminaries and Notations                                 | Additional Preliminaries and Notations                                         |  16 |
|                                 | B.1                                                                    | Preliminary Lemmas . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   |  16 |
|                                 | B.2                                                                    | Examples Illustrating Regularity Conditions . . . . . . . . . . . . . . . . .  |  17 |
| C                               | Proof of Regret Bounds for Simultaneous Online Gradient Descent (SOGD) | Proof of Regret Bounds for Simultaneous Online Gradient Descent (SOGD)         |  19 |
|                                 | C.1                                                                    | Proof of Lemma 2.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |  19 |
|                                 | C.2                                                                    | Bounds on the Inner Decision Variable . . . . . . . . . . . . . . . . . . . .  |  19 |
|                                 | C.3                                                                    | Bounds on the Linear System Solution . . . . . . . . . . . . . . . . . . . .   |  24 |
|                                 | C.4                                                                    | Bounds on the Gradient Estimation Error of Outer Objective . . . . . . . .     |  30 |
|                                 | C.5                                                                    | Bounds on the Outer Objective and its Projected Gradient . . . . . . . . . .   |  33 |
|                                 | C.6                                                                    | Proof of Theorem 2.6 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |  35 |
| D                               | Proof of Regret Bounds for Zeroth Order SOGD (ZO-SOGD)                 | Proof of Regret Bounds for Zeroth Order SOGD (ZO-SOGD)                         |  43 |
|                                 | D.1                                                                    | Auxiliary Lemmas for Proof of Theorem 3.2 . . . . . . . . . . . . . . . . .    |  43 |
|                                 | D.2                                                                    | Perturbation Bounds for OBO Objectives and Their Smoothing Variants . .        |  44 |
|                                 | D.3                                                                    | Bounds on the Zeroth-Order Inner Solution . . . . . . . . . . . . . . . . .    |  47 |
|                                 | D.4                                                                    | Bounds on the Zeroth-Order System Solution . . . . . . . . . . . . . . . .     |  51 |
|                                 | D.5                                                                    | Bounds on the Zeroth-Order Estimation Error of Outer Objective . . . . . .     |  60 |
|                                 | D.6                                                                    | Bounds on the Zeroth-Order Objective Function and its Projected Gradients      |  64 |
|                                 | D.7                                                                    | Proof of Theorem 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |  70 |
| E Hyperparameter Tuning Results | E Hyperparameter Tuning Results                                        | E Hyperparameter Tuning Results                                                |  80 |

## A Related Work

BO was introduced in game theory by [64] and modeled mathematically in [9]. Initial works [32, 55] reduced it to single-level optimization. Recently, gradient-based approaches have gained popularity for their simplicity and efficacy [21, 27, 43, 12, 56], though they assume offline objectives.

OBO was initiated by [67], proposing the OAGD method with regret bounds. [40] developed algorithms for online minimax optimization, special cases of OBO with local regret guarantees. [50] introduced SOBOW, a single-loop optimizer using window-smoothed functions and multiple CGs for nonconvex-strongly-convex cases. Unlike these works, we propose using projected gradient as a more general performance measure for constrained objectives, focusing on the original functions and their regret; See Table 1 for a comparison.

Single-Level Regret Minimization. Single-level online optimization predominantly focuses on convex problems, either with static or dynamic convex regret minimization [72, 34, 61]. Non-convex online optimization [36, 30, 29] poses greater challenges than its convex counterparts [61, 72, 35, 7]. Notable contributions in this field include adversarial multi-armed bandit algorithms [10, 37, 38, 47] and the Follow-the-Perturbed-Leader approach [2, 46, 66]. Hazan et al. [36] introduced windowsmoothed local regret for gradient averaging in non-convex models, which Hallak et al. [31] extended to non-smooth, non-convex problems. Inspired by their work, we employ local regret for Online Bilevel Optimization (OBO) without window-smoothing.

Zeroth-Order Optimization. Single-Level ZO Optimization has been widely studied in both offline [25, 17, 1, 59, 57] and online settings [52, 29, 30, 70, 5]. We next review closely related work. Liu et al. [52] proposed ZOO-ADMM, a gradient-free online optimization algorithm utilizing ADMM. Guan et al. [30] studied online non-convex optimization with limited oracle feedback. Research on online non-convex optimization with bandit feedback includes work by Heliou et al. [37], which established bounds on global static and dynamic regret using dual averaging, further refined in [38]. Gao et al. [23] extended these ideas to ZO algorithms. Flaxman et al. [20] provided algorithms for bandit online optimization of convex functions using ZO gradient approximation. Our work closely relates to [63], which proposes a Hessian-free method approximating the Jacobian matrix using a ZO method based on finite differences of gradients. In contrast, our method uses function oracles to approximate both the Hessian and gradients and is derivative-free. We also point out the recent work [3] on ZO stochastic algorithms for solving bilevel problems when neither the upper/lower objective values nor their unbiased gradient estimates are available. Their approach, limited to the offline setting, does not include numerical results, thus leaving its practical efficiency unclear.

## B Additional Preliminaries and Notations

## B.1 Preliminary Lemmas

We first provide several useful lemmas for the main proofs.

Definition B.1 ( Projected gradient [26]) . Let X ⊂ R d 1 be a closed convex set. Then, the projected gradient for any α t &gt; 0 and p ∈ R d 1 is defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and Π X [ · ] denotes the orthogonal projection operator onto set X .

Lemma B.2. [28, Lemma 13] If f : X → R is a µ f -strongly convex function with respect to some norm ∥ · ∥ , and x ∗ is the minimizer of f (i.e. x ∗ = arg min x ∈X f ( x ) ), then we have ∀ x ∈ X ,

<!-- formula-not-decoded -->

Lemma B.3. Suppose f ( x ) is L -smooth, and x ∗ ∈ argmin x ∈X f ( x ) . Then, we can upper bound the magnitude of the gradient at any given point x ∈ R d in terms of the objective sub optimality at x , as follows:

<!-- formula-not-decoded -->

where

Lemma B.4. For any x , y ∈ R d , the following holds for any c &gt; 0 :

<!-- formula-not-decoded -->

We also utilize a basic yet important property of the projected-gradient mapping.

Lemma B.5. [26, Proposition 1] Let P X ,α t ( x ; p ) denote the projected gradient as defined in Definition B.1. For any x , p 1 , p 2 ∈ R d and α t &gt; 0 , it holds that

<!-- formula-not-decoded -->

Lemma B.6. [36, Proposition 2.4] Let P X ,α t ( x ; p ) denote the projected gradient as defined in Definition B.1. For any x , p 1 , p 2 ∈ R d and α t &gt; 0 , it holds that

<!-- formula-not-decoded -->

Lemma B.7. Let P X ,α t ( x ; p ) be as given in Definition B.1. Then, for any p ∈ R d and α t &gt; 0 , we have

<!-- formula-not-decoded -->

Proof. By the definition of x + , the optimality condition of (36) is

<!-- formula-not-decoded -->

Letting z = x , we obtain

<!-- formula-not-decoded -->

which can be rearranged to

<!-- formula-not-decoded -->

## B.2 Examples Illustrating Regularity Conditions

Theorem 2.6 achieves sublinear bilevel regret when the variations V T and H 2 ,T are o ( T 2 / 3 ) and o ( T 1 / 3 ) , respectively. Below, we provide some examples of online optimization in both single-level and bilevel settings to illustrate when this occurs.

Example B.8 . Consider function f t ( x ) = ∥ A t x -b t ∥ 2 , where A t = [1 , 0; 0 , 1 + 1 t ] , b t = (1 , 1) . It follows from (11) that V T = ∑ T t =2 max x | f t ( x ) -f t -1 ( x ) | = ∑ T t =2 | ( 1 t ) 2 -( 1 t -1 ) 2 | , and

<!-- formula-not-decoded -->

Then, V T ≤ ∑ T t =2 2 t 3 ≈ ∫ T 2 2 t 3 dt = 1 4 -1 T 2 . As T →∞ , V T becomes bounded and approaches a constant value, indicating that V T grows slower than T itself.

Example B.9 . Let t =2 functions

<!-- formula-not-decoded -->

Then, V T = ∑ T max x | f t ( x ) -f t -1 ( x ) | = O (1) .

Example B.10 . Let x ∈ X = [ -1 , 1] ⊂ R , y ∈ R , and consider a sequence of quadratic cost

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where a (1) t = 1 /t and a (2) t = 1 / √ t for all t ∈ [ T ] .

We have and

<!-- formula-not-decoded -->

Taking the maximum over x and using x ∈ [ -1 , 1] :

<!-- formula-not-decoded -->

Since a (1) t = 1 /t and a (2) t = 1 / √ t for all t ∈ [ T ] , then we have

<!-- formula-not-decoded -->

Then, we get

<!-- formula-not-decoded -->

The series ∑ T t =2 ( 2 t 2 + 1 2 t 3 / 2 + 1 t 3 ) converges, implying V T = O (1) . Moreover, we have

<!-- formula-not-decoded -->

which implies H 2 ,T = O (1) .

To achieve V T = o ( T 2 / 3 ) and H 2 ,T = o ( T 1 / 3 ) , the changes in the cost functions f t ( x , y ∗ t ( x )) and y ∗ t ( x ) should decay to zero faster than O (1 /t a ) with a &gt; 1 / 3 . For example, if the coefficients in the functions change as O (1 /t a ) with a &gt; 1 / 3 , then the cumulative sum over T will be o ( T 2 / 3 ) . When f t ( x , y ∗ t ( x )) and y ∗ t ( x ) decay as O (1 / √ t ) , then the total variation grows at most as O ( √ T ) .

## C Proof of Regret Bounds for Simultaneous Online Gradient Descent (SOGD)

Proof Roadmap . We introduce Lemma C.2, which quantifies the error between the approximated direction of the momentum-based gradient estimator, d y t , and the true direction, ∇ y g t ( x t , y t ) , at each iteration. To bound the error of the lower-level variable, we provide Lemma C.4, which captures the gap ∥ y t +1 -y ∗ t ( x t ) ∥ 2 and incorporates the error introduced in Lemma C.2. Moreover, we provide Lemma C.6, which quantifies the error between the approximated direction of the momentum-based gradient estimator, d v t , and the true direction, ∇ 2 y g t ( z t ) v t + ∇ y f t ( z t ) , at each iteration. To bound the error of the system solution, we provide Lemma C.8, which captures the gap ∥ v t +1 -v ∗ t ( x t ) ∥ 2 and incorporates the error introduced in Lemma C.6. Moreover, we provide Lemma C.9, which quantifies the error between the approximated direction of the momentum-based hypergradient estimator, d x t , and the true direction, ∇ x f t ( z t ) + ∇ 2 xy g t ( z t ) v t , at each iteration. We also present Lemma C.11, which provides an upper bound for the projection mapping and relates to the three errors discussed in Lemmas C.4, C.8, and C.9. Finally, by combining these lemmas and appropriately setting the parameters, we achieve the desired result.

## C.1 Proof of Lemma 2.1

Proof. By letting ν = 1 -η for η ∈ (0 , 1) , the window-smoothed gradient

<!-- formula-not-decoded -->

is equivalent to

<!-- formula-not-decoded -->

Let d x t = ∇ F t,ν ( x t , y t ; B t ) . Then (38) is equivalent to

<!-- formula-not-decoded -->

Since we have

<!-- formula-not-decoded -->

with f i ( · ) = 0 for all i ≤ 0 .

If w = t and W = 1 η then, we have

<!-- formula-not-decoded -->

## C.2 Bounds on the Inner Decision Variable

In the following, inspired by offline BO [69, 15] and OBO [67, 50], we provide a set of lemmas for the analysis of SOGD. We first present a lemma that characterizes the Lipschitz continuity of the approximate gradients, as well as the inner and system solutions.

<!-- formula-not-decoded -->

Lemma C.1. Under Assumptions 2.2 and 2.3, for all x , x ′ ∈ X , and the search directions { d x t } T t =1 and { d v t } T t =1 generated by Algorithm 1, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where M f , M v , and ( L y , L v , L f ) are defined in (42) , (43) , and (44) , respectively.

Proof. We first show (39a).

Using Assumptions 2.2 and 2.3, we have ∇ 2 y g t ( x t , y ∗ t ( x t )) ⪰ µ g , and

<!-- formula-not-decoded -->

Observe that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

the third inequality is by Assumption 2.3, and the last inequality follows from (40).

Next, we establish (39b).

<!-- formula-not-decoded -->

Since ∗

Then, from Assumption 2.3 and (40), we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proofs of Eqs. (39c)-(39e) follow from [67, Lemma 17] by setting

<!-- formula-not-decoded -->

where the other constants are defined in Assumption 2.3.

The following lemma is inspired by [69] and can be viewed as an extension of [69] to the online setting.

Lemma C.2. Suppose Assumptions B3. and C1. hold. Let { ( x t , y t , v t ) } T t =1 be generated according to Algorithm 1. For e g t defined as

<!-- formula-not-decoded -->

we have:

<!-- formula-not-decoded -->

Proof. From Algorithm 1, we have

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

where the second inequality follows from Cauchy-Schwartz inequality and Assumption C1.. Moreover, from Cauchy-Schwartz inequality, we have

<!-- formula-not-decoded -->

From Assumption B3., we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

From the two inequalities above, we have

<!-- formula-not-decoded -->

Since e g t := d y t -∇ y g t ( x t , y t ) , we have

<!-- formula-not-decoded -->

Lemma C.3. Suppose Assumptions 2.2, and B3. hold. Then, for the sequence { ( x t , y t ) } T t =1 generated by Algorithm 1, we have

<!-- formula-not-decoded -->

where e g t defined in (45) , y ∗ t ( x t ) is defined in (1) and a &gt; 0 is a constant.

Proof. From Lemma B.4, we have

<!-- formula-not-decoded -->

Next, we will bound the first term on the RHS of (47). We have

<!-- formula-not-decoded -->

where the inequality results from the strong convexity of g t by Assumption 2.2, which implies

<!-- formula-not-decoded -->

Substituting (48) into (47), gives the desired result.

To simplify the notation in the analysis, we introduce the definitions

<!-- formula-not-decoded -->

The following lemma, inspired by the offline bilevel optimization framework in [69], characterizes the descent behavior of the iterates in the inner problem.

Lemma C.4. Suppose Assumptions 2.2, and B3. hold. Let θ y t be defined as in (49) . Then, for the sequence { ( x t , y t ) } T t =1 generated by Algorithm 1, the following bound is guaranteed:

<!-- formula-not-decoded -->

where L µ g = µ g ℓ g, 1 µ g + ℓ g, 1 , L y = ℓ g, 1 µ g is defined as in (44) ; H 2 ,T is defined in (11) . Moreover, e g t is defined in (45) .

Proof. From Lemma B.4, we have for any ´ c &gt; 0

<!-- formula-not-decoded -->

From Lemma C.3, we have for any a &gt; 0

<!-- formula-not-decoded -->

Substituting (52) into (51), we get

<!-- formula-not-decoded -->

Choose ´ c = β t L µg / 2 1 -β t L µg and a = β t L µg 1 -2 β t L µg . Let L µ g := µ g ℓ g, 1 µ g + ℓ g, 1 . Then, the following equations and inequalities are satisfied.

<!-- formula-not-decoded -->

Based on (53) and (54), we get

<!-- formula-not-decoded -->

Next, we upper-bound the last term of the above inequality.

<!-- formula-not-decoded -->

where the second inequality is by Eq. (39d) in Lemma C.1.

Substituting (56) into (55) and summing over t ∈ [ T ] , give the desired result.

## C.3 Bounds on the Linear System Solution

Lemma C.5. Suppose Assumptions 2.2 and B3. hold. Then, for the sequence { ( x t , y t , v t ) } T t =1 generated by Algorithm 1, we have

<!-- formula-not-decoded -->

for any ´ c &gt; 0 , where v ∗ t ( x t ) is the solution of the system in Eq. (4) , and e v t is defined in (60) .

Proof. From the update rules in Algorithm 1, we have the following:

<!-- formula-not-decoded -->

where ∇ P t ( x t , y ∗ t ( x t ) , v t ) := ∇ 2 y g t ( x t , y ∗ t ( x t )) v t + ∇ y f t ( x t , y ∗ t ( x t )) . For the first term of Eq. (57) above, we have

<!-- formula-not-decoded -->

where the first inequality follows from the strong convexity of the function P t , which is the gradient of the strongly convex quadratic program 1 2 v ⊤ ∇ 2 y g t ( x , y ∗ t ( x )) v + v ⊤ ∇ y f t ( x , y ∗ t ( x )) . Then, we have

<!-- formula-not-decoded -->

The second inequality is derived from the following inequality.

<!-- formula-not-decoded -->

where the second equality follows from (4).

Combining (57) and (58), we get the desired result.

Lemma C.6. Suppose Assumptions B2. , B3. , B4. , C2. and C4. hold. Let { ( x t , y t , v t ) } T t =1 be generated according to Algorithm 1. For e v t +1 defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we have:

<!-- formula-not-decoded -->

for all t ∈ [ T ] and ( θ v t , θ y t ) and e g t are defined in (49) and (45) , respectively.

Proof. Note that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

From Algorithm 1, we have

<!-- formula-not-decoded -->

Let u = [ x ; y ; v ] . Then, we have

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

where the inequality follows from Cauchy-Schwartz inequality. For the first term, from Assumptions C2. and C4., we have

<!-- formula-not-decoded -->

where the last inequality follows from (8).

Then, from the above inequality and ∥ a + b + c ∥ 2 ≤ 3( ∥ a ∥ 2 + ∥ b ∥ 2 + ∥ c ∥ 2 ) , we have

<!-- formula-not-decoded -->

Moreover, from ∥ a + b + c ∥ 2 ≤ 3( ∥ a ∥ 2 + ∥ b ∥ 2 + ∥ c ∥ 2 ) , we have

<!-- formula-not-decoded -->

where the last inequality follows from Assumptions B2., B3. and B4.;

From Eq. (63) and the inequality ∥ a + b ∥ 2 ≤ 2( ∥ a ∥ 2 + ∥ b ∥ 2 ) , we obtain

<!-- formula-not-decoded -->

where the last inequality follows from (59).

Similarly, we have

<!-- formula-not-decoded -->

Substituting (65) and (64) into (62), we have

<!-- formula-not-decoded -->

From ∥ a + b ∥ 2 ≤ 2 ∥ a ∥ 2 +2 ∥ b ∥ 2 and (8), we have

<!-- formula-not-decoded -->

This completes the proof.

As demonstrated in Lemma C.6, the gradient estimation error e v t +1 for the linear system consists of four key components: (1) an iteratively refined error term (1 -λ t +1 ) 2 (1 + 72 ℓ 2 g, 1 δ 2 t ) E ∥ e v t ∥ 2 , which depends on the stepsize δ t ; (2) the error arising from the variation in the Hessian of the lower-level objectiv; (3) the error resulting from the variation in the gradient of the upper-level objective, and (4) approximation error terms of order O ( δ 2 t E [ θ v t ]) and O ( δ 2 t E [ θ y t ]) associated with solving the linear system and the iterates in the inner problem, respectively.

Lemma C.7. Suppose Assumptions 2.2, B1. , B2. and B4. hold. Let v ∗ t ( x ) is a solution of Subproblem (4) . Then, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Based on (4), we have that

<!-- formula-not-decoded -->

In the following steps, we bound the terms (66a) and (66b), respectively.

For (66a), we have:

<!-- formula-not-decoded -->

where the equality holds since for any invertible matrix A and B we have ∥ A -1 -B -1 ∥ = ∥ A -1 ( B -A ) B -1 ∥ , and inequalities are obtained from Assumptions 2.2 and B4..

Thus, from (67) and Assumption B1., we get

<!-- formula-not-decoded -->

For (66b), we have

<!-- formula-not-decoded -->

Combining (68) and (69), we have

<!-- formula-not-decoded -->

By raising both sides of the above inequality to the power 2 and using ( a + b ) 2 ≤ 2 a 2 +2 b 2 , we complete the proof.

The following lemma characterizes the decrease in θ v t defined in (49) and can be viewed as an extension of the offline BO result in [69] to the OBO setting.

Lemma C.8. Suppose Assumptions 2.2 and 2.3 hold. Let θ v t be defined in (49) . Then, for any positive choice of step size δ t as

<!-- formula-not-decoded -->

T

<!-- formula-not-decoded -->

where e v t is defined in (60) , ν and L y , are defined in Lemmas C.7 and C.4, respectively.

Proof. By Lemma B.4, for any a &gt; 0 , we have

<!-- formula-not-decoded -->

From Lemma C.5, we have for any ´ c &gt; 0 :

<!-- formula-not-decoded -->

Substituting (72) into (71), we get

<!-- formula-not-decoded -->

In the following, we provide a bound for the third term on the right-hand side of (73). To this end, we have from Lemma C.7:

<!-- formula-not-decoded -->

where the last inequality follows from Lemma C.1.

Combining this result with (73) gives

<!-- formula-not-decoded -->

Let ´ L µ g := ( ℓ g, 1 + ℓ 3 g, 1 ) µ g µ g + ℓ g, 1 , then we have

<!-- formula-not-decoded -->

where the last inequality follows from δ t ≤ ´ L µg ℓ 2 g, 1 .

Choose a = δ t ´ L µg / 4 1 -δ t ´ Lµg and ´ c = δ t ´ L µg / 2 1 -δ t ´ L µg

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, from (74) and (76) we have

<!-- formula-not-decoded -->

Rearranging the terms and summing from t = 1 to T , gives the desired result.

## C.4 Bounds on the Gradient Estimation Error of Outer Objective

The following lemma, inspired by [69], provides a characterization of the descent of the gradient estimation error for the outer-level function.

Lemma C.9. Suppose Assumptions B2. , B3. , B4. , C3. and C5. hold. Let { ( x t , y t , v t ) } T t =1 be generated according to Algorithm 1. For e f t defined as

<!-- formula-not-decoded -->

we have:

<!-- formula-not-decoded -->

for all t ∈ [ T ] , θ v t , e v t and e g t are defined in (49) , (60) and (45) , respectively.

Proof. Note that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

From Algorithm 1, we have

<!-- formula-not-decoded -->

where d xx t +1 ( x t +1 , y t +1 ; B t +1 ) = ∇ x f t +1 ( x t +1 , y t +1 ; B t +1 ) + ∇ 2 xy g t +1 ( x t +1 , y t +1 ; B t +1 ) v t +1 . Let u = [ x ; y ; v ] . Then, we have

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

2 2 2

where the inequality follows from ∥ a + b ∥ ≤ 2 ∥ a ∥ +2 ∥ b ∥ .

Let us bound the second term in the right-hand side of (80). Based on (79), we have

<!-- formula-not-decoded -->

where the first inequality is by and ∥ a + b ∥ 2 ≤ 2 ∥ a ∥ 2 +2 ∥ b ∥ 2 ; the second inequality follows from Assumptions C3., C5. and (8).

Substituting the above inequality into (80) and using ∥ a + b + c ∥ 2 ≤ 3( ∥ a ∥ 2 + ∥ b ∥ 2 + ∥ c ∥ 2 ) , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the (i) follows from (79); (ii) follows from Assumptions B2., B3. and B4.; (iii) follows from (8); (iv) follows from (45) and (60); (vi) follows from (59). Similarly, we have

<!-- formula-not-decoded -->

Substituting (83) and (82) into (81), we have

<!-- formula-not-decoded -->

This completes the proof.

As demonstrated in Lemma C.9, the hypergradient estimator error e f t +1 comprises five key components: (1) the term (1 -η t +1 ) 2 E ∥ e f t ∥ 2 , representing the per-iteration improvement achieved by the momentum-based update; (2) the error arising from the variation in the Jacobian of the lower-level objectiv; (3) the error caused by the variation in the gradient of the upper-level objective ; (4) the error term O (2 β 2 t E ∥ e g t ∥ 2 +2 β 2 t E ∥∇ y g t ( x t , y t ) ∥ 2 ) , which is due to solving the lower-level problem; and (5) the error term O ( δ 2 t E ∥ e v t ∥ 2 +72(1 -η t +1 ) 2 ℓ 4 g, 1 δ 2 t E [ θ v t ]) , which is introduced by the one-step momentum update in solving the linear system problem.

<!-- formula-not-decoded -->

## C.5 Bounds on the Outer Objective and its Projected Gradient

Lemma C.10. Let Assumption 2.4 holds. Then, for the sequence of functions { f t } T t =1 , we have

<!-- formula-not-decoded -->

where M is defined in Assumption 2.4; V T is defined in (11) .

Proof. Note that, we have

<!-- formula-not-decoded -->

where the inequality follows from Assumption 2.4.

Lemma C.11. Let { f t } T t =1 denote the sequence of functions presented to Algorithm 1, satisfying Assumptions 2.2, 2.3 and 2.4. Let P X ,α t be defined as in Definition B.1. For any positive step size α t such that α t ≤ 1 / 4 L f for all t ∈ [ T ] , Algorithm 1 ensures the following bound:

<!-- formula-not-decoded -->

Here, θ y t and θ v t are defined in (49) ; V T , M , M f and e f t are defined in (11) , Assumption 2.4, Eq. (42) , and (77) .

Proof. It follows from Lemma C.1 that

<!-- formula-not-decoded -->

For the first term on the right hand side of (85), we have that

<!-- formula-not-decoded -->

where the inequality follows from Lemma B.7.

˜ 2

Let d t ( z t , v t ) = ∇ x f t ( z t ) + ∇ xy g t ( z t ) v t . Then, from Lemma C.1, we have

<!-- formula-not-decoded -->

where e f t = d x t -˜ d t ( z t , v t ) . This implies that

<!-- formula-not-decoded -->

Plugging the bound (87) into (85), we have that

<!-- formula-not-decoded -->

which can be rearranged into

<!-- formula-not-decoded -->

In addition, we have

<!-- formula-not-decoded -->

where the second inequaliy follows from non-expansiveness of the projection operator and the last inequality follows from (86).

Combining (88) and (89), we have

<!-- formula-not-decoded -->

where the second inequality is due to Lemma C.10.

Lemma C.12. Let Assumptions 2.2, and 2.3 hold. Let { x t } T t =1 be generated according to Algorithm 1. Then, we have

<!-- formula-not-decoded -->

where θ y t and θ v t are defined in (49) , M f is defined in (42) .

<!-- formula-not-decoded -->

Proof. From the update rule of Algorithm 1, we have

<!-- formula-not-decoded -->

where the first inequality is by ( a + b ) 2 ≤ 2 a 2 + 2 b 2 ; the second inequality follows from nonexpansiveness of the projection operator; and the last inequality follows from Eq. (39a) in Lemma C.1.

## C.6 Proof of Theorem 2.6

Proof. Bounding E ∥ e f t ∥ 2 in (78) . From (78), we have

<!-- formula-not-decoded -->

With respect to the coefficient of the first term on the right-hand side of Eq. (91), it is important to note that we have:

<!-- formula-not-decoded -->

Using the definition of α t in (16), we have

<!-- formula-not-decoded -->

where the (i) follows from ( a + b ) 1 / 3 -a 1 / 3 ≤ b/ (3 a 2 / 3 ) ; (ii) follows from c ≥ 2 in (107); (iii) follows from (16); (iv) follows from α t ≤ 1 / 4 L f in (107).

Substituting (93) into (92) and using δ t = c δ α t and η t +1 = c η α 2 t , we have

<!-- formula-not-decoded -->

where the inequalities follow from c η = 1 6 L f +5Ω in (106).

Then, substituting (94) into (91) yields

<!-- formula-not-decoded -->

Bounding E ∥ e g t ∥ 2 in (46) .

From (46), we have

<!-- formula-not-decoded -->

Let us examine the coefficient of the first term on the right-hand side of Eq. (96). Specifically, for γ t +1 = c γ α 2 t and β t = c β α t , we have:

<!-- formula-not-decoded -->

where the last inequality follows from (93).

From the selected c γ in (107) and the definition of Φ in (106), we have

<!-- formula-not-decoded -->

Combined this with Eq. (97) yields

<!-- formula-not-decoded -->

Substituting Eq. (98) into Eq. (96) yields

<!-- formula-not-decoded -->

Bounding E ∥ e v t ∥ 2 in (61) .

From (61), we get

<!-- formula-not-decoded -->

Let us examine the coefficient of the first term on the right-hand side of equation (100). Specifically, for λ t +1 = c λ α 2 t and δ t = c δ α t , we have:

<!-- formula-not-decoded -->

where the last inequality follows from (93).

From the selected c γ in (107) and the definition of Ψ in (106), we have

<!-- formula-not-decoded -->

Combined this with Eq. (101) yields

<!-- formula-not-decoded -->

Substituting Eq. (102) into Eq. (100) yields

<!-- formula-not-decoded -->

Combining the outcomes . We recall from Lemma C.12 that we have

<!-- formula-not-decoded -->

Let

<!-- formula-not-decoded -->

Here

<!-- formula-not-decoded -->

where L µ g = µ g ℓ g, 1 / ( µ g + ℓ g, 1 ) and ´ L µ g = ( ℓ g, 1 + ℓ 3 g, 1 ) µ g / ( µ g + ℓ g, 1 ) . Here, we have

<!-- formula-not-decoded -->

Using (103), (99), (95), (84), (70), and (50), along with (104) and the fact that α t decreases with respect to t , we obtain:

<!-- formula-not-decoded -->

Here, M is defined in Assumption 2.4, V T and H 2 ,T are defined in (11). Moreover, G y ,T , G xy ,T , and G yy ,T are defined in (13). Let

<!-- formula-not-decoded -->

Moreover, we have

<!-- formula-not-decoded -->

Note that, we have

<!-- formula-not-decoded -->

which together with β t = c β α t and δ t = c δ α t in Eq. (16), we have

<!-- formula-not-decoded -->

where the first inequality follows from Γ = 11 M 2 f L µg c β and Υ = 22 M 2 f ´ L µg c δ in (106); the last inequality follows from c β = √ 880 L 2 y M 2 f L 2 µg , c δ = √ 3520 ν 2 M 2 f ´ L 2 µg µ 2 g (1 + 2 L 2 y ) , in (107) and Φ ≥ 480 ℓ 2 g, 1 , and Ω , Ψ ≥ 1440( ℓ 2 g, 2 p 2 + ℓ 2 f, 1 ) in (106). Moreover, we have

<!-- formula-not-decoded -->

where the last inequality follows from α t ≤ 1 / 4 L f in (107) since α t = 1 / ( c + t ) 1 / 3 in (16). Bounding (108a) . From (109), we have

<!-- formula-not-decoded -->

where the first inequality follows from β t = c β α t , δ t = c δ α t in (16), and Eq. (111); the second inequality is by Υ = 22 M 2 f ´ L µg c δ , and Ψ , Ω ≥ 288 ℓ 4 g, 1 M 2 f c 2 δ in (106); the last inequality follows from in (106). Moreover, using Eq. (109) together with β t = c β α t and δ t = c δ α t in Eq. (16), we have

<!-- formula-not-decoded -->

where the first inequality follows from (111); the last inequality follows from Γ = 11 M 2 f L µg c β and Ψ ≥ 576 M 2 f ℓ 2 g, 1 ( p 2 ℓ 2 g, 2 + ℓ 2 f, 1 ) c 2 δ in (106). Thus, from (113) and (114), we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Bounding (108b) .

From (109), we also have

From Eq. (109), β t = c β α t , and Γ = 11 M 2 f L µg c β , we obtain

<!-- formula-not-decoded -->

where the second inequality follows from Ω , Ψ ≥ 144( ℓ 2 g, 2 p 2 + ℓ 2 f, 1 ) L 2 µg c 2 β M 2 f in (106); and the last equality is by ℏ = 25 M 2 f L 2 µg .

From δ t = c δ α t in (16), we obtain

<!-- formula-not-decoded -->

where the second inequality follows from Υ = 22 M 2 f ´ L µg c δ and Ω ≥ 72 ℓ 2 g, 1 ´ L 2 µg M 2 f c 2 δ ; the last equality follows from ȷ = 90 M 2 f ´ L 2 µg .

Thus, we get

## Bounding (108c) .

From β t = c β α t in (16) and Eq. (110), we have

<!-- formula-not-decoded -->

where the second inequality is by Φ ≥ 192 ℓ 2 g, 1 ( µ g + ℓ g, 1 ) Γ c β , and Ω , Ψ ≥ 576( ℓ 2 g, 2 p 2 + ℓ 2 f, 1 ) ( µ g + ℓ g, 1 ) Γ c β in (106); the last inequality follows from α t ≤ 1 /c β ( µ g + ℓ g, 1 ) in (107).

<!-- formula-not-decoded -->

From β t = c β α t , δ t = c δ α t in (16) and Eq. (110), we obtain

<!-- formula-not-decoded -->

Thus, we get

<!-- formula-not-decoded -->

Bounding (108d) .

From

<!-- formula-not-decoded -->

Bounding (108e) .

We also have

<!-- formula-not-decoded -->

From Eq. (16), we have b = ¯ b = 1 . Moreover, by (10), σ 2 = σ 2 g y + σ 2 g yy + σ 2 f y + σ 2 g xy + σ 2 f x . From (15), we also have

<!-- formula-not-decoded -->

Then, by inequalities (112), (115), (116), (117), (118), (119), we have

<!-- formula-not-decoded -->

From the definition of Λ in (105), we have

<!-- formula-not-decoded -->

Using (121), we get

<!-- formula-not-decoded -->

Since α t = 1 / ( c + t ) 1 / 3 , we get

<!-- formula-not-decoded -->

which, combined with the fact that α t decreases with respect to t and by multiplying both sides by 2 /α T , results in Thus, we have

<!-- formula-not-decoded -->

This completes the proof.

## D Proof of Regret Bounds for Zeroth Order SOGD (ZO-SOGD)

Proof Roadmap . We provide Lemma D.7, which quantifies the error between the approximated direction of the momentum-based gradient estimator, ˆ d y t and the true direction, ∇ y g t, ρ ( x t , y t ) , at each iteration. Lemma D.9 assesses the convergence of the iterative solutions { y t } T t =1 , specifically the gap E [ ∥ y t +1 -ˆ y ∗ t ( x t ) ∥ 2 ] , while accounting for the error introduced in Lemma D.7. To establish Lemma D.13, which quantifies the error between the approximated direction of the momentum-based gradient estimator, ˆ d v t , and the true direction, ∇ y f t, ρ ( x t , y t )+ ∇ 2 y g t, ρ ( x t , y t ) v t , we need to present Lemma D.11. This lemma quantifies the error between ˆ d v t and ∇ y f t, ρ ( x t , y t )+ 1 2 ρ v ( ∇ y g t, ρ ( x t , y t + ρ v v t ) -∇ y g t, ρ ( x t , y t -ρ v v t )) . Then, Lemma D.15 captures the error of the system solution of Problem (18), i.e., gap E [ ∥ v t +1 -ˆ v ∗ t ( x t ) ∥ 2 ] , based on these errors. To establish Lemma D.19, which quantifies the error between the approximated direction of the momentum-based hypergradient estimator, ˆ d x t , and the true direction, ∇ x f t, ρ ( x t , y t )+ ∇ 2 xy g t, ρ ( x t , y t ) v t , it is necessary to introduce Lemma D.17. This lemma quantifies the error between ˆ d x t and ∇ x f t, ρ ( x t , y t )+ 1 2 ρ v ( ∇ x g t, ρ ( x t , y t + ρ v v t ) -∇ x g t, ρ ( x t , y t -ρ v v t )) . Then, Lemma D.20 bounds the projection mapping based on these errors. By combining these lemmas and setting parameters, we achieve the desired result.

## D.1 Auxiliary Lemmas for Proof of Theorem 3.2

Lemma D.1. [4, Lemma A.1.] Suppose Assumption B4. holds. Then, for any x , v ∈ X , we have:

<!-- formula-not-decoded -->

Lemma D.2. Suppose that Assumptions 2.2 and 2.3 hold for all x , x ′ ∈ X , and t ∈ [ T ] , and that d x t, ρ and d v t, ρ are defined in (22) . Then, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, ˆ v ∗ t ( x ) , f t, ρ and ˆ y ∗ t ( x ) are defined in (20) , (18) , and (19) , respectively. Moreover, the constants M f , M v , and ( L y , L v , L f ) are defined as in (42) , (43) , and (44) , respectively.

Proof. We first show Eq. (122a).

Using Assumptions 2.2 and B1., we have ∇ 2 y g t, ρ ( x , ˆ y ∗ t ( x )) ⪰ µ g , and

<!-- formula-not-decoded -->

Observe that we have

<!-- formula-not-decoded -->

where M f is defined as in (42); the third inequality is by Assumption 2.3 and the last inequality is by Eq. (123).

We now show Eq. (122b). Since

<!-- formula-not-decoded -->

Then, from Assumption 2.3 and Eq. (123), we have

<!-- formula-not-decoded -->

where M v is defined as in (43).

The proofs of Eqs. (122c)-(122e) follow from [67, Lemma 17] by setting ( L y , L v , L f ) as in (44).

## D.2 Perturbation Bounds for OBO Objectives and Their Smoothing Variants

The following two lemmas are inspired by [24].

Lemma D.3. Given ρ = ( ρ s , ρ r ) as positive smoothing parameters, let g t, ρ ( x , y ) and f t, ρ ( x , y ) be the functions defined by (18) .

- (a) Suppose Assumption B3. holds. Then, we have

<!-- formula-not-decoded -->

- (b) Suppose Assumption B2. holds. Then, we have

<!-- formula-not-decoded -->

Proof. Let B 1 and B 2 be the unit ball in R d 1 and R d 2 , respectively. Let V ( d 1 ) and V ( d 2 ) be volume of the unit ball in R d 1 and R d 2 , respectively. Then, we have

<!-- formula-not-decoded -->

Thus, we get

<!-- formula-not-decoded -->

where the last equality follows since 1 V ( d ) ∫ s ∈ B ∥ s ∥ p ds = d d + p .

The proof of part (b) follows using similar arguments.

Lemma D.4. Given ρ = ( ρ s , ρ r ) as positive smoothing parameters, let g t, ρ ( x , y ) and f t, ρ ( x , y ) be the functions defined by (18) .

- (a) Suppose Assumption B3. holds. Then, we have

<!-- formula-not-decoded -->

- (b) Suppose Assumption B2. holds. Then, we have

<!-- formula-not-decoded -->

Proof. Let S ( d 1 ) be the surface area of the unit sphere in R d 1 . Moreover, let U B 1 be the unit sphere.

<!-- formula-not-decoded -->

where the second equality follows from ∫ U B 1 ss ⊤ d s = S ( d 1 ) d 1 I .

Similarly, let S ( d 2 ) be the surface area of the unit sphere in R d 2 . Moreover, let U B 2 be the unit sphere.

<!-- formula-not-decoded -->

where the second equality follows from ∫ U B 2 rr ⊤ d r = S ( d 2 ) d 2 I .

Thus, we get

<!-- formula-not-decoded -->

Finally, by a similar argument as in Part (a), we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma D.5. Suppose Assumption B3. holds. Let ˆ ∇ y g t ( x , y ; ¯ B t ) and ˆ ∇ x g t ( x , y ; ¯ B t ) be defined as in (24a) and (24b) , respectively. Then, for any ( x , y ) ∈ R d 1 × R d 2 and ρ r , ρ s ≥ 0 , we have

<!-- formula-not-decoded -->

for all ´ y ∈ R d 2 and ´ x ∈ R d 1 .

Proof. The proof is similar to that of Lemma 5 in [42].

and which implies

Lemma D.6. Suppose Assumptions 2.2 and B3. hold. Let ( ρ s , ρ r ) be positive smoothing parameters. Let y ∗ t ( x ) and ˆ y ∗ t ( x ) be defined in (1) and (19) , respectively. Then, we have

<!-- formula-not-decoded -->

Proof. From (1), we have y ∗ t ( x ) ∈ arg min y ∈ R d 2 g t ( x , y ) . Since, by Assumption 2.2, g t ( x , y ) is µ g -strongly convex with respect to y , it follows from Lemma B.2 that

<!-- formula-not-decoded -->

By setting y = ˆ y ∗ t ( x ) , we have

<!-- formula-not-decoded -->

Similarly, from (19), we have

<!-- formula-not-decoded -->

where ρ = ( ρ s , ρ r ) . By Assumption 2.2, g t, ρ ( x , y ) is µ g -strongly convex with respect to y . Hence, according to Lemma B.2, we obtain

<!-- formula-not-decoded -->

By setting y = y ∗ t ( x ) , we have

<!-- formula-not-decoded -->

Summing up (135) and (136), we get

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

where the last inequality is by Eq. (125).

## D.3 Bounds on the Zeroth-Order Inner Solution

Recall that s ∈ R d 1 and r ∈ R d 2 are vectors uniformly sampled from the unit balls B 1 and B 2 , respectively. Let

<!-- formula-not-decoded -->

be generated from the uniform distributions over the unit spheres ( U B 1 , U B 2 ) . Here, ( U B 1 , U B 2 ) denote the uniform distributions over the ( d 1 , d 2 ) -dimensional unit Euclidean balls ( B 1 , B 2 ) , respectively.

Then, similar to (23), we have

<!-- formula-not-decoded -->

Lemma D.7. Suppose that Assumptions B3. and D1. hold. Consider the sequence { ( x t , y t , v t ) } T t =1 generated by Algorithm 2, and define

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

Proof. From the definition of ˆ d y t +1 in Algorithm 2, we have

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

From (137), we have

<!-- formula-not-decoded -->

then, we have

<!-- formula-not-decoded -->

where the second inequality holds by Cauchy-Schwarz inequality. Then, from E ∥ a -E [ a ] ∥ 2 = E ∥ a ∥ 2 -∥ E [ a ] ∥ 2 and Assumption D1., we have

<!-- formula-not-decoded -->

where the second inequality follows from Young's inequality and Lemma D.5. From Eq. (130), we have

<!-- formula-not-decoded -->

Finally, we get

<!-- formula-not-decoded -->

LemmaD.8. Suppose Assumptions 2.2 and B3. hold. Then, for the sequence { ( x t , y t ) } T t =1 generated by Algorithm 2, we have

<!-- formula-not-decoded -->

where a &gt; 0 is a constant, e g ρ t is defined in (138) , and ˆ y ∗ t ( x t ) is defined in (19) .

Proof. From Lemma B.4, we have

<!-- formula-not-decoded -->

Next, we will separately bound the first term on the RHS of the above inequality. We have

<!-- formula-not-decoded -->

where the inequality results from the strong convexity of g t, ρ by Assumption 2.2, which implies

<!-- formula-not-decoded -->

Substituting (141) into (140), gives the desired result.

For notational brevity in the analysis, we define

<!-- formula-not-decoded -->

where ˆ y ∗ t ( x ) and ˆ v ∗ t ( x ) are defined in (19) and (20), respectively.

Lemma D.9. Suppose Assumptions 2.2 and B3. hold. Let ˆ θ y t be defined in (142) . Then, for the sequence { ( x t , y t ) } T t =1 generated by Algorithm 2 guarantees the following bound:

<!-- formula-not-decoded -->

where L y = ℓ g, 1 µ g is defined as in (44) and L µ g = µ g ℓ g, 1 µ g + ℓ g, 1 .

Proof. From Lemma B.4, we have for any c &gt; 0

<!-- formula-not-decoded -->

From Lemma D.8, we have for any a &gt; 0

<!-- formula-not-decoded -->

Substituting (145) into (144), we get

<!-- formula-not-decoded -->

Choose c = β t L µg / 2 1 -β t L µg and a = β t L µg 1 -2 β t L µg . Then, the following equations and inequalities are satisfied.

<!-- formula-not-decoded -->

where L µ g = µ g ℓ g, 1 µ g + ℓ g, 1 . Based on (146) and (147), we get

<!-- formula-not-decoded -->

Next, we upper-bound the last term of the above inequality.

<!-- formula-not-decoded -->

where the second inequality is by Lemma D.2.

Moreover, from Lemma D.6, we get

<!-- formula-not-decoded -->

Combining (149) and (150) yields

<!-- formula-not-decoded -->

Substituting (151) into (148) and summing over t ∈ [ T ] , give the desired result.

## D.4 Bounds on the Zeroth-Order System Solution

Lemma D.10. Suppose Assumptions B2. and B3. hold. Let

<!-- formula-not-decoded -->

where ˆ ∇ y f t and ˆ ∇ 2 y g t are defined in (25a) and (26a) , respectively. Then, for the sequence { ( x t , y t , v t ) } T t =1 generated by Algorithm 2, we have

<!-- formula-not-decoded -->

Proof. From Lemma D.5, we have

<!-- formula-not-decoded -->

Moreover, from (26a), we have

<!-- formula-not-decoded -->

where the first inequality follows from Lemma D.5.

From ∥ a + b ∥ 2 ≤ 2 ( ∥ a ∥ 2 + ∥ b ∥ 2 ) , we get

<!-- formula-not-decoded -->

where the second inequality follows from (152) and (153).

Lemma D.11. Suppose Assumptions B2. , B3. , D1. , and D3. hold. Consider the sequence { ( x t , y t , v t ) } T t =1 generated by Algorithm 2, and define

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

Proof. According to the definition of ˆ d v t in Algorithm 2, we have

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

Since

<!-- formula-not-decoded -->

then, we have

<!-- formula-not-decoded -->

where the second inequality holds by Cauchy-Schwarz inequality. Note that, for the last term on the right-hand side of (157), from (26a) and (155), we have

<!-- formula-not-decoded -->

where the last inequality follows from Assumption D1..

Then, from E ∥ a -E [ a ] ∥ 2 = E ∥ a ∥ 2 -∥ E [ a ] ∥ 2 and Assumptions D1. and D3., we have

<!-- formula-not-decoded -->

Then, from Young's inequality and Lemma D.10, we obtain

<!-- formula-not-decoded -->

For the third term on the right-hand side of (158), based on (155), we have

<!-- formula-not-decoded -->

For (159a), we get

<!-- formula-not-decoded -->

where the last inequality follows from Eq. (130). Similary, for (159b), we have

<!-- formula-not-decoded -->

Substituting the above inequalities in (159), we have

<!-- formula-not-decoded -->

For the second term on the right-hand side of (158), we have

<!-- formula-not-decoded -->

where the last inequality follows from Eq. (132).

From (160), (161) and (158), we get

<!-- formula-not-decoded -->

Lemma D.12. Suppose Assumption B4. holds. Let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Then, for ( x , y , v ) presented to Algorithm 2, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. For part (a) : From Lemma D.1, We have

<!-- formula-not-decoded -->

where the last inequality follows from (8). For part (b): From Lemma D.1, We have

<!-- formula-not-decoded -->

where the last inequality follows from (8).

Lemma D.13. Suppose Assumption B4. holds. Then, for the directions ˆ d v t and ˆ d x t provided to Algorithm 2, and

(a) for d v t, ρ defined in (22b) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

t t t (a)

<!-- formula-not-decoded -->

(b) and for d x t, ρ defined in (22c) , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

with ˜ ∇ 2 xy g t ( x t , y t ) is defined in (172) .

Proof. For part (a) : Let

<!-- formula-not-decoded -->

According to the definition of d v t, ρ in (22b), we have

<!-- formula-not-decoded -->

Next, we separately bound (168a) and (168b) on the RHS of the above inequality. Bounding (168a) . We have

<!-- formula-not-decoded -->

Bounding (168b) . From Lemmas D.1 and D.12, we have

<!-- formula-not-decoded -->

Combining (169) and (170) yields

<!-- formula-not-decoded -->

For part (b): Let

<!-- formula-not-decoded -->

According to the definition of d x t, ρ in (22c), we have

<!-- formula-not-decoded -->

Next, we separately bound (173a) and (173b) on the RHS of the above inequality. Bounding (173a) . We have

<!-- formula-not-decoded -->

Bounding (173b) . From Lemmas D.1 and D.12, we have

<!-- formula-not-decoded -->

Combining (174)-(175) yields

<!-- formula-not-decoded -->

Lemma D.14. Suppose Assumptions 2.2, B1. , B3. and B4. hold. Set the step size δ t and the parameter p in (8) , as

<!-- formula-not-decoded -->

Then, for the sequence { ( x t , y t , v t ) } T t =1 generated by Algorithm 2 and ˆ v ∗ t ( x t ) in (20) , we have

<!-- formula-not-decoded -->

for some ´ a &gt; 0 , where ˆ θ v t and B t are defined in Eq. (142) and Lemma D.13, respectively.

Proof. By setting the radius p := ℓ f, 0 µ g in (8), we have

<!-- formula-not-decoded -->

where ∇ P t ( x t , ˆ y ∗ t ( x t ) , v t ) := ∇ 2 y g t, ρ ( x t , ˆ y ∗ t ( x t )) v t + ∇ y f t, ρ ( x t , ˆ y ∗ t ( x t )) .; the first inequality follows from non-expansiveness property of a projection operator.

We next bound the I t , and K t terms in (177), respectively.

Bounding I t . We have

<!-- formula-not-decoded -->

where the inequality holds since ∇ P t is the gradient of the strongly convex quadratic program 1 2 v ⊤ ∇ 2 y g t, ρ ( x , ˆ y ∗ t ( x )) v + v ⊤ ∇ y f t, ρ ( x , ˆ y ∗ t ( x )) .

Thus, we have

<!-- formula-not-decoded -->

Since δ t ≤ ( 2 + 1 ℓ 2 g, 1 ) µ g ℓ g, 1 µ g + ℓ g, 1 , then we have

<!-- formula-not-decoded -->

where the second inequality holds since from (20), we have

<!-- formula-not-decoded -->

where the second inequality follows from Assumption B3.. Bounding K t . Let

<!-- formula-not-decoded -->

From Lemma D.13, we have

<!-- formula-not-decoded -->

Putting (178), and (179) together with Eq. (177) yields the desired result.

<!-- formula-not-decoded -->

Lemma D.15. Suppose Assumptions 2.2 and 2.3 hold. Let ˆ θ v t be defined in (142) . Set the parameter p in (8) as p = ℓ f, 0 µ g . Then, for any positive choice of step sizes satisfying

<!-- formula-not-decoded -->

the sequence { ( x t , y t , v t ) } T t =1 generated by Algorithm 2 guarantees the following bound:

<!-- formula-not-decoded -->

where B t , ν and ( L µ g , L y ) are defined in Lemmas D.13, C.7 and D.9, respectively.

Proof. From Lemma B.4, we have, for any ´ c &gt; 0

<!-- formula-not-decoded -->

From Lemma D.14, we have, for any ´ a &gt; 0

<!-- formula-not-decoded -->

Substituting (182) into (181), we get

<!-- formula-not-decoded -->

Choose ´ c = δ t L µg / 4 1 -δ t L µ g 2 and ´ a = δ t L µg / 2 1 -δ t L µg . Then, the following equations and inequalities are satisfied. (1 + ´ c ) (1 + ´ a ) ( 1 -δ t L µ g ) = 1 -δ t L µ g 4 , (1 + ´ c ) ( 1 + 1 ´ a ) ≤ 4 δ t L µ g , 1 + 1 ´ a ≤ 2 δ t L µ g , 1 + 1 ´ c ≤ 4 δ t L µ g , (184)

where L µ g = µ g ℓ g, 1 µ g + ℓ g, 1 .

Thus, we have

<!-- formula-not-decoded -->

We now bound the last term on the right-hand side of (185). By Lemma C.7, we have:

<!-- formula-not-decoded -->

where the last inequality follows from Lemma D.2.

From (150), we have

<!-- formula-not-decoded -->

Plugging (187) into (186), we get

<!-- formula-not-decoded -->

Then, substituting (188) into (185), rearranging the resulting inequality and summing over t ∈ [ T ] , we obtain the desired result.

## D.5 Bounds on the Zeroth-Order Estimation Error of Outer Objective

Lemma D.16. Suppose Assumptions B2. and B3. hold. Let

<!-- formula-not-decoded -->

where ˆ ∇ x f t +1 and ˆ ∇ 2 xy g t +1 are defined in (25b) and (26b) , respectively. Then, for the sequence { ( x t , y t , v t ) } T t =1 generated by Algorithm 2, we have

<!-- formula-not-decoded -->

Proof. From Lemma D.5, we have

<!-- formula-not-decoded -->

Moreover, from (26a), we have

<!-- formula-not-decoded -->

where the first inequality follows from Lemma D.5.

<!-- formula-not-decoded -->

where the second inequality follows from (189) and (190).

Lemma D.17. Suppose Assumptions B2. , B3. , D2. , and D4. hold. Consider the sequence { ( x t , y t , v t ) } T t =1 generated by Algorithm 2. For e L t defined in (166c) , we have

<!-- formula-not-decoded -->

Proof. According to the definition of ˆ d x t in Algorithm 2, we have

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

Since

<!-- formula-not-decoded -->

then, we have

<!-- formula-not-decoded -->

where the second inequality holds by Cauchy-Schwarz inequality. Note that for the last term on the right-hand side of (192), using (172) and (26b), we have

<!-- formula-not-decoded -->

where the last inequality follows from Assumption D2..

Then, from E ∥ a -E [ a ] ∥ 2 = E ∥ a ∥ 2 -∥ E [ a ] ∥ 2 and Assumption D4., we have

<!-- formula-not-decoded -->

Then, from Young's inequality and Lemma D.16, we have

<!-- formula-not-decoded -->

For the third term on the right-hand side of (193), we have

<!-- formula-not-decoded -->

For (195a), we get

<!-- formula-not-decoded -->

where the last inequality follows from Eq. (130). Similary, for (195b), we have

<!-- formula-not-decoded -->

Substituting these inequalities in (195), we have

<!-- formula-not-decoded -->

For the second term on the right-hand side of (193), we have

<!-- formula-not-decoded -->

where the last inequality follows from Eq. (132).

From (196), (197) and (194), we get

<!-- formula-not-decoded -->

## D.6 Bounds on the Zeroth-Order Objective Function and its Projected Gradients

Lemma D.18. Suppose Assumptions 2.2, B2. , B3. , and 2.4 hold. Then, for the sequence of functions { f t, ρ } T t =1 defined in Eq. (18) , we have

<!-- formula-not-decoded -->

Here, V T is defined in (11) ; and M is defined in Assumption 2.4.

Proof. Note that, we have

<!-- formula-not-decoded -->

From (126), we have and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Moreover, from Lemma D.6, we have

<!-- formula-not-decoded -->

For the last term of the above inequality, we have

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

From (201), (202), and (204), we get the desired result.

Lemma D.19. Suppose that Assumptions 2.2 and 2.3 hold. Let f t, ρ be defined as in (18) . Then, for ˆ d x t generated by Algorithm 2, for all t ∈ [ T ] , we have

<!-- formula-not-decoded -->

where e L t is defined in Lemma D.13, and ˆ θ y t , ˆ θ v t are as defined in (142) . Additionally, M f is given in Lemma D.2.

Proof. From ∥ a + b ∥ 2 ≤ 2 ( ∥ a ∥ 2 + ∥ b ∥ 2 ) , we get

<!-- formula-not-decoded -->

where d x t, ρ is defined in (22c). From Lemma D.13, we have

<!-- formula-not-decoded -->

Moreover, from Eq. (122a), we get

<!-- formula-not-decoded -->

Substituting (207) and (208) into (206), we conclude the desired result.

Lemma D.20. Suppose Assumptions 2.2, 2.3, and 2.4 hold. Let the sequence of functions { f t, ρ } T t =1 be defined in (18) , and let P X ,α t be given in Definition B.1. Then, for any positive choice of step sizes satisfying α t ≤ 1 / 4 L f , for all t ∈ [ T ] , Algorithm 2 guarantees the following bound:

<!-- formula-not-decoded -->

where V T and A t are respectively defined in Eq. (11) and Lemma D.19.

Proof. Due to the L f -smoothness of the function f t by Eq. (39c) in Lemma C.1, f t, ρ is also L f -smooth. Hence,

<!-- formula-not-decoded -->

For the first term on the R.H.S of Eq. (210), we have that

<!-- formula-not-decoded -->

where the first inequality follows from Lemma B.7; the last inequality follows from Lemma D.19. Plugging the bound (211) into (210), we have that

<!-- formula-not-decoded -->

which can be rearranged into

<!-- formula-not-decoded -->

In addition, we have

<!-- formula-not-decoded -->

where the second inequaliy follows from non-expansiveness of the projection operator. Then, from Lemma D.19 and Assumption B2., we have

<!-- formula-not-decoded -->

where the last inequality is by Lemma D.6.

Combining (212) and (213) and summing over t = 1 to T , we have

<!-- formula-not-decoded -->

where the second inequality is due to Lemma D.18.

Lemma D.21. Let the sequence { ( x t , y t , v t ) } T t =1 be generated by Algorithm 2.

- (a) Then, we have

<!-- formula-not-decoded -->

where e g ρ t is defined in (138) .

- (b) Suppose Assumptions 2.2, B2. and B3. hold. Then, we have

<!-- formula-not-decoded -->

where A t is defined in (205) .

- (c) Suppose Assumptions B1. , B2. and B3. hold. Then, we have

<!-- formula-not-decoded -->

where e t M and ˆ θ y t are defined in (154) and (142) , respectively.

- Proof. For part (a): From Algorithm 2, we have

<!-- formula-not-decoded -->

## For part (b):

From the update rule in Algorithm 2, we obtain

<!-- formula-not-decoded -->

where the first inequality is by ( a + b ) 2 ≤ 2 a 2 + 2 b 2 ; the second inequality follows from nonexpansiveness of the projection operator; and the last inequality follows from Lemma D.19.

The first term in the above inequality can be bounded as

<!-- formula-not-decoded -->

where the last inequality follows from Lemma D.6.

Based on (217) and (216), we get

<!-- formula-not-decoded -->

For part (c): From the nonexpansiveness of projection, we have

<!-- formula-not-decoded -->

where the second equality follows from (154).

From Assumption B3., Lemma B.3 and (8), we have

<!-- formula-not-decoded -->

Similarly, we get

<!-- formula-not-decoded -->

Moreover, from Eq. (132) and Assumption B1., we have

<!-- formula-not-decoded -->

Substituting (219), (220) and (221), into (218), we get

<!-- formula-not-decoded -->

## D.7 Proof of Theorem 3.2

Proof. Since (1 -γ t +1 ) 2 ≤ 1 -γ t +1 and γ t +1 = c γ α t in (31), from (139), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Combining the outcomes .

Let

Here, we have

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By adding (223), (222), (224), (143), and (180), along with (209) and considering the fact that α t decreases with respect to t , and by applying Lemma D.21, we obtain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here,

<!-- formula-not-decoded -->

Moreover,

<!-- formula-not-decoded -->

We then provide bounds for the terms in (227a)-(227i).

Note that, we have

<!-- formula-not-decoded -->

which together with β t = c β α t , δ t = c δ α t in (31), we have

<!-- formula-not-decoded -->

where the first inequality is by Γ = 11 M 2 f L µg c β , Υ = 52 M 2 f L µg c δ in (225), ρ 2 v = c v α t and α t ≤ 1 / 4 L f in (31); the second inequality follows from c β ≥ √ 1760 L 2 y M 2 f L 2 µg , c δ ≥ √ 33280 ν 2 M 2 f L 2 µg µ 2 g (1 + 2 L 2 y ) , in (226); and Φ = 240 d 2 ℓ 2 g, 1 L f , Ψ = 720 d 2 ℓ 2 f, 1 L f , Ω = 720 d 1 ℓ 2 f, 1 L f and c v ≥ 1080 ℓ 2 g, 1 ( d 2 Ψ + d 1 Ω ) in (225). Moreover, we have

<!-- formula-not-decoded -->

where the last inequality is by α t ≤ 1 / 4 L f in (226). Bounding (227a) .

From δ t = c δ α t in (31), we have

<!-- formula-not-decoded -->

where the first inequality follows from (230); the last inequality is by Υ = 52 M 2 f L µg c δ in (225). From (228), we obtain

<!-- formula-not-decoded -->

Thus, from β t = c β α t , δ t = c δ α t and ρ 2 v = c v α t in (31), we have

<!-- formula-not-decoded -->

where the first inequality follows from (230); the second equality follows from Γ = 11 M 2 f L µg c β in (225); the last inequality is by c v ≥ 324 M 2 f ℓ 4 g, 1 ( d 2 Ψ + d 1 Ω ) c 2 δ . Thus, from (232) and (233), we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From (229) and η t +1 = c η α t in (31), we have

<!-- formula-not-decoded -->

## Bounding (227b) .

From (230), we also obtain

where the last inequality is by c η ≥ 26Ω and (230). Thus, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Bounding (227c) .

From δ t = c δ α t in (31) and Eq. (230), we have

<!-- formula-not-decoded -->

Thus, from ρ 2 v = c v α t in (31), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From (229), λ t +1 = c λ α t and δ t = c δ α t in (31), we have

<!-- formula-not-decoded -->

where the first inequality is by c λ ≥ 10Υ L µg c δ Ψ and α t ≤ 1 / 4 L f ; the last inequality follows from Ψ ≥ 27 L µg Υ L f ℓ 2 g, 1 d 2 c δ and Ω ≥ 27 L µg Υ L f ℓ 2 g, 1 d 1 c δ .

Since β t = c β α t and δ t = c δ α t in (31), we get

<!-- formula-not-decoded -->

Bounding (227e) .

From (229), we have

From (236) and (235), we have

## Bounding

(227d) .

From (228), we have

<!-- formula-not-decoded -->

From (229), γ t +1 = c γ α t , β t = c β α t in (31), we have

<!-- formula-not-decoded -->

where the first equality is by Γ = 11 M 2 f L µg c β and ρ 2 v = c v α t ; the first inequality follows from c γ ≥ 26 M 2 f Φ L 2 µg ; the second inequality is by α t ≤ 1 / 4 L f ; the last inequality follows from c v ≥ 54 L 2 µg M 2 f ℓ 2 g, 1 ( d 2 Ψ + d 1 Ω ) c 2 β , Φ ≥ 12 d 2 ℓ 2 g, 1 L 2 µg c 2 β L f M 2 f ,and Ψ ≥ 36 ℓ 2 f, 1 d 2 L 2 µg c 2 β L f M 2 f , and Ω ≥ 36 ℓ 2 f, 1 d 1 L 2 µg c 2 β L f M 2 f . From (229), β t = c β α t , ρ 2 v = c v α t in (31) and (241), we have

<!-- formula-not-decoded -->

where the first inequality follows from α t ≤ 1 /c β ( µ g + ℓ g, 1 ) ; the second inequality is by α ≤ 1 / 4 L f ; the last inequality is by c v ≥ 216 Γ ℓ 2 g, 1 ( d 2 Ψ + d 1 Ω ) c β ( µ g + ℓ g, 1 ) and Φ ≥ 24 d 2 ℓ 2 g, 1 ( µ g + ℓ g, 1 ) L f c β Γ , and Ψ ≥ 144 d 2 ℓ 2 f, 1 ( µ g + ℓ g, 1 ) c β L f Γ , and Ω ≥ 144 d 1 ℓ 2 f, 1 ( µ g + ℓ g, 1 ) c β L f Γ . Thus, we get

<!-- formula-not-decoded -->

Bounding (227f) . From (228), we have

<!-- formula-not-decoded -->

Thus, from δ t = c δ α t in (31), we have

<!-- formula-not-decoded -->

Bounding (227g) . From ρ 2 v = c v α t in (31), we have

<!-- formula-not-decoded -->

Bounding (227h) . From γ t +1 = c γ α t , η t +1 = c η α t , λ t +1 = c λ α t and ρ 2 v = c v α t in (31), we have

<!-- formula-not-decoded -->

Bounding (227i) . From β t = c β α t , δ t = c δ α t in (31), we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Moreover, we have

<!-- formula-not-decoded -->

which, implies that

<!-- formula-not-decoded -->

From (248), (249) and ρ 2 v = c v α t in (31), we get

<!-- formula-not-decoded -->

Combining the outcomes (227i) . Combining inequalities (234), (237), (238), (240), (244), (245), (246), (247), and (250) leads to

<!-- formula-not-decoded -->

From the definition of Λ in (105), we have

<!-- formula-not-decoded -->

From (28), we have ˆ σ = ˆ σ g y + ˆ σ g yy + ˆ σ f y + ˆ σ g xy + ˆ σ f x . Thus, using (251), (31), and rearranging the terms, we get

<!-- formula-not-decoded -->

where second inequality holds because we have

<!-- formula-not-decoded -->

Then, note that, we have

<!-- formula-not-decoded -->

From non-expansiveness of the projection operator and Lemma D.4, we have

<!-- formula-not-decoded -->

This implies

<!-- formula-not-decoded -->

Applying the upper bound in (252) yields

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This completes the proof.

## E Hyperparameter Tuning Results

As detailed in Section 4, we carefully tuned all hyperparameters to ensure stable and fair comparisons. Our analysis indicates that while ZO-SOGD exhibits sensitivity to hyperparameter choices, it remains robust within reasonable ranges. Below, we provide extensive tuning results for ZO-SOGD.

The hyperparameter sensitivity analysis for the adversarial attack scenario reveals critical insights about the algorithm's attack effectiveness across different parameter configurations. For the inner and outer stepsizes, we observe that the algorithm achieves optimal attack performance with specific combinations that balance perturbation strength and imperceptibility.

Table 2: Hyperparameter tuning results for inner ( β ) and outer ( α ) stepsizes in adversarial attack scenario. Values represent test accuracy (mean ± std) over 5 runs. Lower values indicate better attack performance.

| β \ α       | α = 0 . 001     | α = 0 . 005     | α = 0 . 01      | α = 0 . 1       |
|-------------|-----------------|-----------------|-----------------|-----------------|
| β = 0 . 001 | 0 . 68 ± 0 . 05 | 0 . 59 ± 0 . 07 | 0 . 47 ± 0 . 06 | 0 . 53 ± 0 . 08 |
| β = 0 . 005 | 0 . 54 ± 0 . 06 | 0 . 41 ± 0 . 05 | 0 . 35 ± 0 . 04 | 0 . 42 ± 0 . 05 |
| β = 0 . 01  | 0 . 48 ± 0 . 04 | 0 . 34 ± 0 . 05 | 0 . 57 ± 0 . 07 | 0 . 39 ± 0 . 06 |
| β = 0 . 1   | 0 . 26 ± 0 . 03 | 0 . 43 ± 0 . 06 | 0 . 33 ± 0 . 04 | 0 . 45 ± 0 . 07 |

The stepsize analysis reveals that larger inner stepsizes combined with smaller outer stepsizes tend to produce more effective attacks. Specifically, the configuration with β = 0 . 1 and α = 0 . 001 achieves the lowest test accuracy of 0 . 26 ± 0 . 03 , indicating the most successful adversarial perturbations. This pattern suggests that aggressive updates to the perturbation parameters ( β ) while maintaining conservative hyperparameter updates ( α ) creates an effective balance for generating strong yet imperceptible adversarial examples.

Table 3: Performance comparison across different smoothing parameters ( ρ r = ρ s ) in adversarial attack scenario.

| ρ v \ ρ r = ρ s   | 0 . 001         | 0 . 005         | 0 . 01          | 0 . 05          |
|-------------------|-----------------|-----------------|-----------------|-----------------|
| ρ v = 0 . 001     | 0 . 61 ± 0 . 06 | 0 . 52 ± 0 . 05 | 0 . 48 ± 0 . 04 | 0 . 57 ± 0 . 06 |
| ρ v = 0 . 005     | 0 . 47 ± 0 . 05 | 0 . 39 ± 0 . 04 | 0 . 35 ± 0 . 04 | 0 . 45 ± 0 . 05 |
| ρ v = 0 . 01      | 0 . 41 ± 0 . 04 | 0 . 28 ± 0 . 03 | 0 . 31 ± 0 . 03 | 0 . 43 ± 0 . 05 |
| ρ v = 0 . 05      | 0 . 53 ± 0 . 06 | 0 . 44 ± 0 . 05 | 0 . 40 ± 0 . 04 | 0 . 52 ± 0 . 06 |

The smoothing parameter analysis provides additional insights into the algorithm's convergence behavior in the adversarial setting. The optimal configuration occurs with ρ v = 0 . 01 and ρ r = ρ s = 0 . 005 , achieving a test accuracy of 0 . 28 ± 0 . 03 . These moderate smoothing values appear to provide the right balance between exploration and exploitation in the adversarial perturbation space, allowing the algorithm to find effective attack directions without excessive oscillation or premature convergence.

Table 4: Performance comparison across different momentum parameters in adversarial attack scenario.

| γ t \ λ t = η t   | 0 . 9           | 0 . 99          | 0 . 999         |
|-------------------|-----------------|-----------------|-----------------|
| γ t = 0 . 9       | 0 . 35 ± 0 . 04 | 0 . 29 ± 0 . 03 | 0 . 38 ± 0 . 05 |
| γ t = 0 . 99      | 0 . 31 ± 0 . 03 | 0 . 24 ± 0 . 02 | 0 . 33 ± 0 . 04 |
| γ t = 0 . 999     | 0 . 37 ± 0 . 04 | 0 . 32 ± 0 . 08 | 0 . 40 ± 0 . 05 |

The momentum parameter investigation reveals that moderate momentum values consistently produce the most effective adversarial attacks. The optimal configuration with γ t = 0 . 99 and λ t = η t = 0 . 99 achieves the lowest test accuracy of 0 . 24 ± 0 . 02 , representing the most successful attack performance. This configuration suggests that maintaining momentum across both inner and outer optimization loops helps the algorithm navigate the complex adversarial landscape more effectively than either no momentum or excessive momentum settings.

The comprehensive analysis demonstrates that ZO-SOGD maintains robust attack performance across a broad range of hyperparameter configurations. The algorithm consistently achieves test accuracies below 0 . 5 across most reasonable parameter combinations, indicating reliable adversarial attack capability. The standard deviations remain low throughout the parameter space, suggesting stable and reproducible attack performance across multiple experimental runs.

The optimal hyperparameter configuration for adversarial attacks consists of inner stepsize β = 0 . 1 , outer stepsize α = 0 . 001 , smoothing parameters ρ v = 0 . 01 and ρ r = ρ s = 0 . 005 , and momentum parameters γ t = λ t = η t = 0 . 99 . This configuration enables ZO-SOGD to achieve superior attack performance while maintaining the imperceptibility constraints essential for practical adversarial examples.

## NeurIPS Paper Checklist

## A. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: We provide detailed proofs and implementations of the algorithms.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## B. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations are provided in the Conclusion section.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## C. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Please refer to Theorems 2.6 and 3.2.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## D. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: It is provided in Section 4.

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

## E. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: The data is publicly available. For experimental results, please refer to Section 4 and the supplement.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## F. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: For experimental results, please refer to Section 4 and the supplement.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## G. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Please refer to Section 4 and the supplement.

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

## H. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA] .

Justification: All experiments were conducted on the same system and are easily reproducible on a standard personal computer.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## I. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The authors reviewed the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## J. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] .

Justification: Theory Paper.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## K. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA] .

Justification: Theory Paper.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## L. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] .

Justification: The authors cite the original paper that produced the code package or dataset.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## M. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA] .

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## N. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA] .

Justification: Theory paper.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## O. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] .

Justification: Paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## P. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA] .

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.