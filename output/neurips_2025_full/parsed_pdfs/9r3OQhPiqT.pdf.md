## An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds

Xu Shi * 1

Rufeng Xiao * 1

Rujun Jiang ‡ 1, 2

1 School of Data Science, Fudan University 2 Shanghai Key Laboratory for Contemporary Applied Mathematics, Fudan University

{xshi22, rfxiao24}@m.fudan.edu.cn rjjiang@fudan.edu.cn

## Abstract

Existing methods for solving Riemannian bilevel optimization (RBO) problems require prior knowledge of the problem's first- and second-order information and curvature parameter of the Riemannian manifold to determine step sizes, which poses practical limitations when these parameters are unknown or computationally infeasible to obtain. In this paper, we introduce the Adaptive Riemannian Hypergradient Descent (AdaRHD) algorithm for solving RBO problems. To our knowledge, AdaRHD is the first method to incorporate a fully adaptive step size strategy that eliminates the need for problem-specific parameters in RBO. We prove that AdaRHD achieves an O (1 /ϵ ) iteration complexity for finding an ϵ -stationary point, thus matching the complexity of existing non-adaptive methods. Furthermore, we demonstrate that substituting exponential mappings with retraction mappings maintains the same complexity bound. Experiments demonstrate that AdaRHD achieves comparable performance to existing non-adaptive approaches while exhibiting greater robustness.

## 1 Introduction

Bilevel optimization has garnered significant attention owing to its diverse applications in fields such as reinforcement learning [49, 35], meta-learning [6, 67, 41], hyperparameter optimization [26, 73, 95], adversarial learning [8, 87, 88], and signal processing [50, 24]. This paper focuses on 'Riemannian bilevel optimization (RBO)' [33, 23, 54], a framework that arises in applications such as Riemannian meta-learning [80], neural architecture search [77], image segmentation [28], Riemannian min-max optimization [44, 39, 102, 30, 89, 38], and low-rank adaption [98]. The general formulation of RBO is as follows:

<!-- formula-not-decoded -->

where M x , M y are d x - and d y -dimensional complete Riemannian manifolds [54], the functions f, g : M x ×M y → R are smooth, the lower-level function g ( x, y ) is geodesic strongly convex w.r.t. y , and the upper-level function f is not required to be convex.

To address Problem (1), we extend the recent advances in Euclidean adaptive bilevel optimization [93] to propose an adaptive Riemannian hypergradient method. Although certain aspects of Euclidean

* Equal contributions.

‡ Corresponding author.

analysis extend naturally to Riemannian settings, the manifold's curvature parameters introduce many analytical challenges. We propose key technical difficulties in designing adaptive methods for solving RBO problems as follows:

- (i) RBOnecessitates that variables are constrained on Riemannian manifolds, making it intrinsically more complex than its Euclidean counterpart. Furthermore, critical parameters such as geodesic strong convexity, Lipschitz continuity, and curvature parameters over Riemannian manifolds are more computationally demanding to estimate than their Euclidean counterparts. Consequently, the geometric structure of the manifold creates distinct analytical challenges in the convergence analysis of adaptive algorithms.
- (ii) Compared to single-level Riemannian (adaptive) optimization, RBO inherently involves interdependent variable updates, resulting in complex step size selection challenges arising from coupled (hyper)gradient dynamics.
- (iii) Unlike non-adaptive RBO methods, adaptive step-size strategies can cause divergent behavior during initial iterations since the initial step size may be too large, which must be rigorously addressed to enable robust theoretical convergence analysis.

In this work, we present a comprehensive convergence analysis that demonstrates our approach to resolving these challenges.

## 1.1 Related works

Table 1 summarizes key studies on (Riemannian) bilevel optimization that are relevant to our methods. The table compares their applicable scenarios, whether they are adaptive, and their computational complexity of first- and second-order information. Note that constants, such as condition numbers, are omitted from the complexity analysis for simplicity. For a comprehensive review of the related works, please refer to Appendix A.

Table 1: Comparisons of first-order and second-order complexities for reaching an ϵ -stationary point. Here, 'Euc' and 'Rie' represent the feasible region of the algorithms are Euclidean space and Riemannian manifolds, respectively. The notations 'Det', 'F-S', and 'Sto' represent the applicable problems as deterministic, functionsum, and stochastic, respectively. Additionally, G f and G g are the gradient complexities of f and g , respectively. JV g and HV g are the complexities of computing the second-order cross derivative and Hessian-vector product of g . The notation ˜ O denotes the omission of logarithmic terms in contrast to the standard O notation. Furthermore, the notation 'NA' represents that the corresponding complexity is not applicable.

| Methods                                               | Space   | Adaptive   | Type    | G f                                     | G g                                             | JV g                                    | HV g                                   |
|-------------------------------------------------------|---------|------------|---------|-----------------------------------------|-------------------------------------------------|-----------------------------------------|----------------------------------------|
| D-TFBO [93] S-TFBO [93]                               | Euc     | ✓          | Det     | O (1 /ϵ ) ˜ O (1 /ϵ )                   | O (1 /ϵ 2 ) ˜ O (1 /ϵ )                         | O (1 /ϵ ) ˜ O (1 /ϵ )                   | O (1 /ϵ 2 ) ˜ O (1 /ϵ )                |
| RHGD-HINV [33] RHGD-CG [33] RHGD-NS [33] RHGD-AD [33] | Rie     | ✗          | Det     | O (1 /ϵ ) O (1 /ϵ ) O (1 /ϵ ) O (1 /ϵ ) | ˜ O (1 /ϵ ) ˜ O (1 /ϵ ) ˜ O (1 /ϵ ) ˜ O (1 /ϵ ) | O (1 /ϵ ) O (1 /ϵ ) O (1 /ϵ ) O (1 /ϵ ) | NA ˜ O (1 /ϵ ) ˜ O (1 /ϵ ) ˜ O (1 /ϵ ) |
| RSHGD-HINV [33]                                       |         |            | F-S     | O (1 /ϵ 2 )                             | ˜ O (1 /ϵ 2 )                                   | O (1 /ϵ 2 )                             | NA                                     |
| RieBO [54] RieSBO [54]                                | Rie     | ✗          | Det Sto | O (1 /ϵ ) O (1 /ϵ 2 )                   | ˜ O (1 /ϵ ) ˜ O (1 /ϵ 2 )                       | O (1 /ϵ ) O (1 /ϵ 2 )                   | ˜ O (1 /ϵ ) ˜ O (1 /ϵ 2 )              |
| RF 2 SA [23] RF 2 SA [23]                             | Rie     | ✗          | Det Sto | ˜ O (1 /ϵ 3 / 2 ) ˜ O (1 /ϵ 7 / 2 )     | ˜ O (1 /ϵ 3 / 2 ) ˜ O (1 /ϵ 7 / 2 )             | NA NA                                   | NA NA                                  |
| AdaRHD-GD (Ours) AdaRHD-CG (Ours)                     | Rie     | ✓          | Det     | O (1 /ϵ ) O (1 /ϵ )                     | O (1 /ϵ 2 ) O (1 /ϵ 2 )                         | O (1 /ϵ ) O (1 /ϵ )                     | O (1 /ϵ 2 ) ˜ O (1 /ϵ )                |

## 1.2 Contributions

This paper introduces the Adaptive Riemannian Hypergradient Descent (AdaRHD) algorithm, the first method to incorporate a fully adaptive step size strategy, eliminating the need for parameter-specific prior knowledge for RBO. The contributions of this work are summarized as follows:

- (i) We develop an adaptive algorithm for solving RBO problems, eliminating reliance on prior knowledge of strong convexity, Lipschitzness, or curvature parameters. Our method achieves a convergence rate matching those of non-adaptive parameter-dependent approaches. Furthermore, by replacing exponential mappings with computationally efficient retraction mappings, our algorithm maintains comparable convergence guarantees while reducing computational costs.
- (ii) We establish upper bounds for the total iterations of resolutions of the lower-level problem in Problem (1) and the corresponding linear system required for computing the Riemannian hypergradient. The derived bounds match the iteration complexity of the standard AdaGradNorm method for solving the strongly convex problems [92].
- (iii) We evaluate our algorithm against existing methods on various problems over Riemannian manifolds, including the Riemannian hyper-representation and robust optimization problems. The results demonstrate that our method performs comparably with non-adaptive methods while exhibiting significantly greater robustness.

## 2 Preliminaries

In this section, we review standard definitions and preliminary results on Riemannian optimization. All results presented here are available in the literature [53, 10, 33, 54], we restate them for conciseness.

We first establish essential properties and notations for Riemannian manifolds. The definition of a Riemannian manifold has been conducted in the literature [53, 10]. The Riemannian metric on a Riemannian manifold M at x ∈ M is denoted by ⟨· , ·⟩ x : T x M× T x M→ R , with the induced norm on the tangent space T x M defined as ∥ u ∥ x = √ ⟨ u, u ⟩ x for any u ∈ T x M . We then recall the definitions of exponential mapping and parallel transport. For an tangent vector u ∈ T x M and a geodesic c : [0 , 1] → M satisfying c (0) = x , c (1) = y , and c ′ (0) = u , the exponential mapping Exp x : T x M→M is defined as Exp x ( u ) = y . The inverse of the exponential mapping, Exp -1 x : M → T x M , is called logarithm mapping [10, Definition 10.20], and the Riemannian distance between two points x, y ∈ M is given by d ( x, y ) = ∥ Exp -1 x ( y ) ∥ x = ∥ Exp -1 y ( x ) ∥ y . Parallel transport P x 2 x 1 : T x 1 M → T x 2 M is a linear operator that preserves the inner product structure, satisfying ⟨ u, v ⟩ x 1 = ⟨P x 2 x 1 u, P x 2 x 1 v ⟩ x 2 for all u, v ∈ T x 1 M .

We now define key properties of functions on Riemannian manifolds. The Riemannian gradient G f ( x ) ∈ T x M of a differentiable function f : M→ R is the unique vector satisfying

<!-- formula-not-decoded -->

where D f ( x )[ u ] denotes the directional derivative of f at x along u [84, 10, 54]. For twicedifferentiable f , the Riemannian Hessian H f ( x ) is defined as the covariant derivative of G f ( x ) . A function f : M → R is 'geodesically (strongly) convex' if f ( c ( t )) is (strongly) convex w.r.t. t ∈ [0 , 1] for all geodesics c : [0 , 1] → Ω , where Ω ⊆ M is a geodesically convex set [54, Definition 4]. Particularly, if f is twice-differentiable, µ -geodesic strong convexity of f is equivalent to H f ( x ) ⪰ µ Id , where Id is the identity operator. Further, f is 'geodesically L f -Lipschitz smooth' if

<!-- formula-not-decoded -->

and for twice-differentiable f , this property is equivalent to H f ( x ) ⪯ L f Id .

Finally, for a bi-function f : M x × M y → R , the Riemannian gradients of f w.r.t. x and y are denoted by G x f ( x, y ) and G y f ( x, y ) , respectively, while the Riemannian Hessians are H x f ( x, y ) and H y f ( x, y ) . Moreover, the Riemannian cross-derivatives G 2 xy f ( x, y ): T y M y → T x M x and G 2 yx f ( x, y ): T x M x → T y M y , discussed in [33, 54], are linear operators. Furthermore, the operator norm of a linear operator G : T y M y → T x M x is defined as

<!-- formula-not-decoded -->

Building on the concepts of Lipschitz continuity (cf. Definition B.1) and geodesic strong convexity for functions defined on Riemannian manifolds, we present the following proposition.

Proposition 2.1 ([10, 33, 54]) . For a function f : M → R , if its Riemannian gradient G f is L -Lipschitz continuous, then for all x, y ∈ U ⊆ M , it holds that

<!-- formula-not-decoded -->

If f is µ -geodesic strongly convex, then for all x, y ∈ U , it holds that

<!-- formula-not-decoded -->

Additionally, for the Lipschitzness of the functions and operators, and an important result of the trigonometric distance bound over the Riemannian manifolds, we present them in Appendix B.

## 3 Adaptive Riemannian hypergradient descent algorithms

Standard Riemannian bilevel optimization (RBO) approaches [23, 33, 54] determine step sizes for updating the variables using problem-specific parameters such as strong convexity, Lipschitzness, and curvature constants. However, these parameters are frequently impractical to estimate or compute, posing challenges for step size determination. This limitation underscores the need for adaptive RBO algorithms that operate without prior parameter knowledge.

## 3.1 Approximate Riemannian hypergradient

Prior to presenting our primary algorithm, we first introduce the Riemannian hypergradient, as the core concept of our methodology relies on this construct. Following general bilevel optimization frameworks, we define the Riemannian hypergradient of F ( x ) in Problem (1) as follows:

Proposition 3.1. [33, 54] The Riemannian hypergradient of F ( x ) are given by

<!-- formula-not-decoded -->

Proposition 3.1 fundamentally hinges on the implicit function theorem for Riemannian manifolds [30], which necessitates the invertibility of the Hessian of the lower-level objective function g w.r.t. y . In this paper, this requirement is satisfied as g is geodesically strongly convex w.r.t. y . Consequently, y ∗ ( x ) is unique and differentiable.

However, the Riemannian hypergradient defined in (3) is computationally challenging to evaluate. First, the exact solution y ∗ ( x ) of the lower-level objective is not explicitly available, necessitating the use of an approximate solution ˆ y , we employ the adaptive Riemannian gradient descent method to compute this approximation. Second, calculating the Hessian-inverse-vector product H -1 y g ( x, y ∗ ( x ))[ G y f ( x, y ∗ ( x ))] in (3) incurs prohibitive computational costs. To address this, given the approximation ˆ y of y ∗ ( x ) , we can approximate the Hessian-inverse-vector product H -1 y g ( x, ˆ y )[ G y f ( x, ˆ y )] by solving the linear system H y g [ v ]( x, ˆ y ) = G y f ( x, ˆ y ) , which is originated from the following quadratic problem:

<!-- formula-not-decoded -->

Denote ˆ v as the approximate solution of Problem (4). In this paper, we use the adaptive gradient descent method (cf. Steps 11-15 of Algorithm 1) or the tangent space conjugate gradient method [81, 10] (cf. Step 16 of Algorithm 1) to obtain such a ˆ v . Specifically, the tangent space conjugate gradient algorithm is presented in Algorithm 2 of Appendix C.

Then, given the estimations ˆ y and ˆ v , the approximate Riemannian hypergradient [23, 33, 54] is defined as follows,

<!-- formula-not-decoded -->

## 3.2 Adaptive Riemannian hypergradient descent algorithm: AdaRHD

Motivated by the recent adaptive methods [93] for solving general bilevel optimization, we employ the 'inverse of cumulative (Riemannian) gradient norm' strategy to design the adaptive step sizes [92, 90, 93], i.e., adapting the step sizes based on accumulated Riemannian (hyper)gradient norms.

Given the total iterations T , set the accuracies for solving the lower-level problem of Problem (1) and linear system as ϵ y = 1 /T and ϵ v = 1 /T . Our Adaptive Riemannian Hypergradient Descent (AdaRHD) algorithm for solving Problem (1) is presented in Algorithm 1. For simplicity, when employing the gradient descent (Steps 11-15) and conjugate gradient (Step 16) to solve the linear system, respectively, we denote AdaRHD as AdaRHD-GD and AdaRHD-CG, respectively.

## Algorithm 1 Ada ptive R iemannian H ypergradient D escent (AdaRHD)

```
1: Initial points x 0 ∈ M x , y 0 ∈ M y , and v 0 ∈ T y 0 M , initial step sizes a 0 > 0 , b 0 > 0 , and c 0 > 0 , total iterations T , and error tolerances ϵ y = ϵ v = 1 T . 2: for t = 0 , 1 , 2 , ..., T -1 do 3: Set k = 0 and y 0 t = y K t -1 t -1 if t > 0 and y 0 otherwise. 4: while ∥G y g ( x t , y k t ) ∥ 2 y k t > ϵ y do 5: b 2 k +1 = b 2 k + ∥G y g ( x t , y k t ) ∥ 2 y k t , 6: y k +1 t = Exp y k t ( -1 b k +1 G y g ( x t , y k t )) , 7: k = k +1 . 8: end while 9: K t = k . 10: Set n = 0 and v 0 t = P y Kt t y Kt -1 t -1 v N t -1 t -1 if t > 0 and v 0 otherwise. 11: while ∥∇ v R ( x t , y K t t , v n t ) ∥ 2 y Kt t > ϵ v do 12: c 2 n +1 = c 2 n + ∥∇ v R ( x t , y K t t , v n t ) ∥ 2 y Kt t , 13: v n +1 t = v n t -1 c n +1 ∇ v R ( x t , y K t t , v n t ) , ▷ Gradient descent 14: n = n +1 . 15: end while 16: Or set v 0 t = 0 and invoke v n t = TSCG( H y g ( x t , y K t t ) , G y f ( x t , y K t t ) , v 0 t , ϵ v ) . ▷ Conjugate gradient 17: N t = n. 18: ̂ G F ( x t , y K t t , v N t t ) = G x f ( x t , y K t t ) -G 2 xy g ( x t , y K t t )[ v N t t ] , 19: a 2 t +1 = a 2 t + ∥ ̂ G F ( x t , y K t t , v N t t ) ∥ 2 x t , 20: x t +1 = Exp x t ( -1 a t +1 ̂ G F ( x t , y K t t , v N t t )) . 21: end for
```

## 3.3 Convergence analysis

This section establishes the errors between the approximate Riemannian hypergradient (5) and the exact Riemannian hypergradient (3), the convergence result of Algorithm 1, and the corresponding computation complexity.

## 3.3.1 Definitions and assumptions

Similar to the definition of an ϵ -stationary point in general bilevel optimization [28, 43, 42], the definition of an ϵ -stationary point in Riemannian bilevel optimization [23, 33, 54] is given as follows. Definition 3.1 ( ϵ -stationary point) . A point x ∈ M x is an ϵ -stationary point of Problem (1) if it satisfies ∥G F ( x ) ∥ 2 x ≤ ϵ .

We first concern the upper bound of curvature constant ζ ( τ, c ) defined in Lemma B.1.

Assumption 3.1. The manifolds M x and M y are complete Riemannian manifolds. Moreover, M y is a Hadamard manifold whose sectional curvature is lower bounded by τ &lt; 0 . Moreover, for all iterates y k t in the lower-level problem, the curvature constant ζ ( τ, d ( y k t , y ∗ ( x t ))) defined in Lemma B.1 is always upper bounded by a constant ζ for all t ≥ 0 and k ≥ 0 .

Assumption 3.1 is commonly used in the Riemannian optimization, which ensures that the lower-level objective g can be geodesically strongly convex [100, 54] and is essential for the convergence of the lower-level problem [33, 54].

We then adopt the following assumptions concerning the fundamental properties of the upper- and lower-level objectives of Problem (1).

Assumption 3.2. Function f ( x, y ) is continuously differentiable and g ( x, y ) is twice continuously differentiable for all ( x, y ) ∈ U = U x ×U y ⊆ M x ×M y , and g ( x, y ) is µ -geodesic strongly convex w.r.t. y ∈ U y for any x ∈ U x .

This assumption is prevalent in (Riemannian) bilevel optimization [28, 43, 42, 57, 35, 51, 23, 33, 54]. Under this assumption, the optimal solution y ∗ ( x ) of the lower-level problem is unique for all x ∈ M x and differentiable [33, 54], ensuring the existence of the Riemannian hypergradient (3.1).

Assumption 3.3. Function f ( x, y ) is l f -Lipschitz continuous in U ⊆ M x ×M y . The Riemannian gradients G x f ( x, y ) and G y f ( x, y ) are L f -Lipschitz continuous in U . The Riemannian gradients G x g ( x, y ) and G y g ( x, y ) are L g -Lipschitz continuous in U . Furthermore, the Riemannian Hessian H y g ( x, y ) , cross derivatives G 2 xy g ( x, y ) , G 2 yx g ( x, y ) are ρ -Lipschitz in U .

Assumption 3.3 is a standard condition in the existing literature on (Riemannian) bilevel optimization [28, 43, 42, 33, 54]. These conditions ensure the smoothness of the objective function F in Problem (1) and the Riemannian hypergradient G F defined in (3).

Assumption 3.4. The minimum of F over M x , denote as F ∗ , is lower-bounded.

Assumption 3.4 concerns the existence of the minimum of F , which is a common requirement in the literature of adaptive (bilevel) optimization problems [90, 92, 93].

## 3.3.2 Convergence results

Denote ˆ v ∗ t ( x, y ) := arg min v ∈ T y M y R ( x, y, v ) . We can then bound the estimation error of the proposed schemes of approximated hypergradient as follows.

Lemma 3.1 (Hypergradient approximation error bound) . Suppose that Assumptions 3.1, 3.2, 3.3, and 3.4 hold. The error for the approximated Riemannian hypergradient generated by Algorithm 1 satisfies,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 3.1 establishes that the error between the approximate and exact Riemannian hypergradients is bounded by the errors arising from the resolutions of the lower-level problem in Problem (1) and the linear system (4). Consequently, we can actually integrate the stopping criterion ∥ ̂ G F ( x t , y K t t , v N t t ) ∥ 2 x t ≤ 1 /T into Algorithm 1 to enable early termination. By Lemma 3.1, this criterion ensures that ∥G F ( x t ) ∥ 2 x t ≤ O (1 /T ) , which implies that x t is an O (1 /T ) -stationary point of Problem (1). Moreover, integrating this criterion does not increase the overall computational complexity of Algorithm 1, since the norm of ̂ G F ( x t , y K t t , v N t t ) is already computed at Step 20.

Similar to Proposition 2 in [93], we derive upper bounds on the total number of iterations required to solve both the lower-level problem in Problem (1) and the linear system (4). However, owing to the geometric structures inherent in Riemannian manifolds, these bounds cannot be directly inferred from [93, Proposition 2] and require a meticulous analysis.

Proposition 3.2. Suppose that Assumptions 3.1, 3.2, 3.3, and 3.4 hold. Then, for any 0 ≤ t ≤ T , the numbers of iterations K t and N t required in Algorithm 1 satisfy:

<!-- formula-not-decoded -->

AdaRHD-GD:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C b , C c , ¯ b , ¯ c , ˜ b , and ˜ c are constants defined in Appendix G.

AdaRHD-CG:

<!-- formula-not-decoded -->

Remark 3.1. Proposition 3.2 provides upper bounds for the total iterations of the resolutions of the lower-level problem in Problem (1) and the linear system (4) . Since 1 / log(1 + ϵ y ) is of the same order as 1 /ϵ y , it holds that K t = O (1 /ϵ y ) , which matches the standard AdaGrad-Norm for solving

strongly convex problems [92]. Additionally, we also have N t = O (1 /ϵ v ) and N t = O (log(1 /ϵ v )) when employing gradient descent and conjugate gradient methods, respectively. Moreover, similar to AdaGrad-Norm [92], the step size adaptation for the lower-level problem proceeds in two stages: Stage 1 requires at most O (1 / log(1 + ϵ y )) iterations, while Stage 2 requires at most O (log(1 /ϵ y )) iterations. Furthermore, due to geometric properties of Riemannian manifolds, these bounds cannot be directly derived from [93] and introduce additional technical challenges. Specifically, the constants C b , C c , ¯ b , ¯ c , ˜ b , and ˜ c are all related to curvature ζ , and increase as ζ increases.

Based on Proposition 3.2, we have the following convergence result of Algorithm 1.

Theorem 3.1. Suppose that Assumptions 3.1, 3.2, 3.3, and 3.4 hold. The sequence { x t } T t =0 generated by Algorithm 1 satisfies,

<!-- formula-not-decoded -->

where C is a constant defined in Appendix G.

Theorem 3.1 demonstrates that our proposed adaptive algorithm achieves a convergence rate matching that of standard non-adaptive (Riemannian) bilevel algorithms (e.g., [28, 43, 42, 33, 54]), thereby demonstrating its computational efficiency. Moreover, although Algorithm 1 shares a similar algorithmic structure with D-TFBO [93] and some Euclidean analyses extend naturally to Riemannian settings, its convergence analysis still presents significant theoretical challenges. For example, the geometric structure of Riemannian manifolds necessitates the use of the trigonometric distance bound (Lemma B.1) instead of its Euclidean counterpart and requires incorporating the curvature constant into the step size adaptation (cf. Proposition 3.2), which substantially increases the analytical difficulty. We subsequently derive the complexity bound for Algorithm 1.

Corollary 3.1. Suppose that Assumptions 3.1, 3.2, 3.3, and 3.4 hold. Algorithm 1 needs T = O (1 /ϵ ) iterations to achieve an ϵ -accurate stationary point of Problem (1) . The gradient complexities of f and g are G f = O (1 /ϵ ) and G g = O (1 /ϵ 2 ) , respectively. The complexities of computing the second-order cross derivative and Hessian-vector product of g are JV g = O (1 /ϵ ) , HV g = O (1 /ϵ 2 ) for AdaRHD-GD, and HV g = ˜ O (1 /ϵ ) for AdaRHD-CG, respectively.

Corollary 3.1 demonstrates that the gradient complexity ( G g ) and Hessian-vector product complexity ( HV g ) of AdaRHD-GD surpass those of non-adaptive Riemannian bilevel optimization methods [33, 54] by a factor of 1 /ϵ . This gap originates from the additional iterations needed to guarantee the solution of the lower-level problem in Problem (1) and linear system 4, which is necessitated by the lack of prior knowledge regarding the strong convexity, Lipschitzness, and curvature constants.

Designing algorithms to eliminate the 1 /ϵ -gap is an interesting future direction. Potential strategies to bridge this gap could be drawn from adaptive methods in Riemannian optimization, such as sharpness-aware minimization on Riemannian manifolds (Riemannian SAM) [96], Riemannian natural gradient descent (RNGD) [37], and the framework for Riemannian adaptive optimization [70]. Riemannian extensions of Euclidean approaches like adaptive accelerated gradient descent [61], adaptive proximal gradient method (adaPGM) [52], adaptive Barzilai-Borwein algorithm (AdaBB) [103], and adaptive Nesterov accelerated gradient (AdaNAG) [76] may also be viable alternatives to the inverse of cumulative gradient norms strategy [92] employed in Algorithm 1.

## 3.4 Extend to retraction mapping

Given that retraction mapping is often preferred in practice for its computational efficiency over exponential mapping, this section demonstrates that its incorporation into Algorithm 1 achieves comparable theoretical convergence guarantees. A modified version of Algorithm 1, incorporating retraction mapping, is presented in Algorithm 3 (cf. Appendix D). To formalize this extension, the second condition of Assumption 3.1 must be modified as follows.

Assumption 3.5. All the iterates of the lower-level problem generated by Algorithm 3 lie in a bounded set belonging to M y that contains the optimal solution, i.e., there exists a constant ¯ D &gt; 0 such that d ( y k t , y ∗ ( x t )) ≤ ¯ D holds for all t ≥ 0 and k ≥ 0 .

Under Assumption 3.5, the curvature constant ζ ( τ, d ( y k t , y ∗ ( x t ))) in Assumption 3.1 is bounded by its definition, thereby ensuring consistency with Assumption 3.1. Additionally, it is necessary to consider the error between the exponential and retraction mappings.

Assumption 3.6. Given z 1 ∈ U ⊆ M (here M can be M x or M y ) and u ∈ T z 1 M , let z 2 = Retr z 1 ( u ) . There exist constants c u ≥ 1 , c R ≥ 0 such that d 2 ( z 1 , z 2 ) ≤ c u ∥ u ∥ 2 z 1 and ∥ Exp -1 z 1 ( z 2 ) -u ∥ z 1 ≤ c R ∥ u ∥ 2 z 1 .

Assumption 3.6 is standard (e.g., [33, Assumption 2] and [46, Assumption 1]) in bounding the error between the exponential mapping and retraction mapping, given that retraction mapping is a first-order approximation to the exponential mapping [33].

Then, similar to Theorem 3.1, we have the following convergence result of Algorithm 3.

Theorem 3.2. Suppose that Assumptions 3.1, 3.2, 3.3, 3.4, 3.5, and 3.6 hold. The sequence { x t } T t =0 generated by Algorithm 3 satisfies,

<!-- formula-not-decoded -->

where C retr is a constant defined in Appendix H.

Algorithm 3 achieves a convergence rate nearly identical to that of Algorithm 1 while being more practical to implement. Furthermore, the complexity analysis of Algorithm 3 closely aligns with that of Algorithm 1 and is, therefore, omitted here.

## 4 Experimental results

In this section, adhering to the experimental framework established by [33], we conduct comprehensive experiments to evaluate our algorithm against the RHGD method. Since the RieBO algorithm proposed by [54] shares the same algorithmic framework as RHGD, we categorize them as a single method for comparison. We designate RHGD-50 and RHGD-20 as configurations with maximum iteration limits for solving the lower-level problem in RHGD set to 50 and 20, respectively. For consistency, we exclusively employ Algorithm 3 in this section due to its computational efficiency in practice. Detailed experimental settings and additional experiments are provided in Appendix I, and our codes are available at https://github.com/RufengXiao/AdaRHD .

## 4.1 Simple problem

In the first experiment, following [33], we consider a simple problem, which aims to determine the maximum similarity between two matrices X ∈ R n × d and Y ∈ R n × r , where n ≥ d ≥ r , formulated as:

<!-- formula-not-decoded -->

where St( d, r ) = { W ∈ R d × r : W ⊤ W = I r } and S d ++ = { M ∈ R d × d : M ≻ 0 } represent the Stiefel manifold and the symmetric positive definite (SPD) matrices, respectively, and λ &gt; 0 is the regular parameter. The matrix W ∈ St( d, r ) aligns X and Y in a shared dimensional space, while the lower-level problem learns an appropriate geometric metric M ∈ S d ++ [97]. Additionally, the geodesic strong convexity of the lower-level problem and the Hessian inverse expression can be found in Appendix H of [33]. In this experiment, we generate random data matrices X and Y with two sample sizes: n = 100 and n = 1000 , where d = 50 and r = 20 .

Figure 1: Performances of methods in n = 100 .

<!-- image -->

Figure 2: Performances of methods in n = 1000 .

<!-- image -->

Figures 1 and 2 show the evolution of the upper-level objective function (Upper Objective) over time and the associated hypergradient estimation errors (Hypergrad error) across outer iterations.

Figure 1 corresponds to n = 100 , while Figure 2 represents the n = 1000 case. These results demonstrate that our algorithm exhibits faster convergence and superior performance compared to RHGD while maintaining scalability with increasing sample dimensionality, confirming its efficiency and robustness. Moreover, although GD achieves quicker initial convergence, CG demonstrates greater robustness, as evidenced by its lower hypergradient estimation errors.

## 4.2 Robustness analysis

In the second experiment, also from [33, Section 4.2], we address the deep hyper-representation problem for classification, which is a subclass of hyper-representation problems [26, 73, 95, 59, 74]. In contrast to the 2-layer SPD network employed for ETH-80 dataset classification in [33], to demonstrate the efficacy of our algorithm, here we utilize a 3-layer SPD network [40] as the upperlevel architecture to optimize input embeddings over the larger AFEW dataset [20], comprising seven emotion classes. The training set contains 1,747 matrices with imbalanced class distribution (267, 235, 173, 292, 342, 288, and 150 samples per class, respectively). This optimization problem is formulated as follows:

<!-- formula-not-decoded -->

where SPDnet ( · ; A 1 , A 2 , A 3 ) denotes the 3-layer SPD network with layer parameters A 1 , A 2 , and A 3 [40], the term y i represents the one-hot encoded label, and D = { D i } n i =1 denote a set of SPD matrices where D i ∈ S d ++ . Each matrix D i has dimensions of 400 × 400 , and we set d 1 = 100 , d 2 = 20 , and r = 5 .

In this study, we perform a series of experiments with varying initial step sizes to evaluate the robustness and advantages of our proposed AdaRHD algorithm. To address computational constraints, we utilize a 5% subset of the AFEW dataset [20] rather than the full dataset, ensuring tractable training durations. For AdaRHD, we initialize hyperparameters a 0 , b 0 , and c 0 to equal values and test performance across the range { 0 . 2 , 1 , 2 , 10 , 20 } . For RHGD, we fix η x and η y to equal values and evaluate the set { 5 , 1 , 0 . 5 , 0 . 1 , 0 . 05 } . To mitigate sampling bias and validate robustness, each algorithm is executed five times with distinct random seeds, each iteration employing a unique 5% data subset. This methodology enables systematic assessment of optimization sensitivity to step sizes and establishes the generalizability of our algorithm under practical constraints.

Figure 3: Epoch vs. validation accuracy and ergodic performance min i ∈ [0 ,t ] ∥ ̂ G F ( x i , y K i i , v N i i ) ∥ 2 x i under different initial step sizes for each algorithm. In the figures for 'AdaRHD-X', the labels indicate the values of 1 /a 0 = 1 /b 0 = 1 /c 0 ; in the figures for 'RHGD-X', the labels represent the values of η x = η y .

<!-- image -->

The results are presented in Figure 3 and Table 2. It can be observed that the RHGD-50 and RHGD-20 fail when the initial step sizes are set to 5, 1, or 0.5, whereas the corresponding configurations of

Table 2: Time to reach a specific validation accuracy under different initial step sizes for each algorithm. The values outside the parentheses indicate the mean over five random trials, while the values inside the parentheses represent the standard deviation. 'Step Size (A/R)' denotes the initial step sizes used in each method: 1 /a 0 = 1 /b 0 = 1 /c 0 in the 'AdaRHD-X' algorithm; η x = η y in the 'RHGD-X' algorithm. 'X%' indicates the time required to reach the corresponding validation accuracy.

|           | AdaRHD-CG   | AdaRHD-CG   | AdaRHD-CG   | AdaRHD-GD   | AdaRHD-GD   | AdaRHD-GD     | RHGD-50   | RHGD-50   | RHGD-50       | RHGD-20   | RHGD-20     | RHGD-20       |
|-----------|-------------|-------------|-------------|-------------|-------------|---------------|-----------|-----------|---------------|-----------|-------------|---------------|
| Step Size | 50%         | 70%         | 85%         | 50%         | 70%         | 85%           | 50%       | 70%       | 85%           | 50%       | 70%         | 85%           |
| 5.0       | 287.20      | 464.79      | 1217.06     | 432.66      | 690.62      | 1488.26       | /         | /         | /             | /         | /           | /             |
| 5.0       | ( 93.61 )   | ( 124.76    | )( 635.57 ) | (157 .      | 76)(136 .   | 52)(426 . 67) | /         | /         | /             | /         | /           | /             |
| 1.0       | 131.84      | 191.51      | 245.63      | 200.26      | 317.31      | 421.99        | /         | /         | /             | /         | /           | /             |
| 1.0       | ( 18.76 )   | ( 34.15 )   | ( 72.42 )   | (31 . 21)   | (68 . 92)   | (88 . 31)     | /         | /         | /             | /         | /           | /             |
| 0.5       | 118.09      | 169.79      | 241.54      | 192.51      | 291.83      | 477.49        | /         | /         | /             | /         | /           | /             |
| 0.5       | ( 19.91 )   | ( 28.59 )   | ( 33.09 )   | (22 . 86)   | (54 . 21)   | (120 . 17)    | /         | /         | /             | /         | /           | /             |
| 0.1       | 217.32      | 312.58      | 427.84      | 297.90      | 447.60      | 546.18        | 171.97    | 370.69    | 808.48        | 36.01     | 168.47      | 540.46        |
| 0.1       | (33 . 96)   | ( 33.66 )   | ( 53.40 )   | (42 . 60)   | (53 . 50)   | (66 . 68)     | (78 . 36) | (68 . 61) | (146 . 57)    | ( 12.37   | )(120 . 86) | (88 . 54)     |
| 0.05      | 251.47      | 415.51      | 633.46      | 358.25      | 504.80      | 648.86        | 137.88    | 715.91    | 1510.09       | 41.36     | 179.11      | 1067.92       |
| 0.05      | (36 . 27)   | (49 . 75)   | (96 . 54)   | (32 . 89)   | ( 15.58 )   | ( 42.91 )     | (136 .    | 10)(104 . | 62)(363 . 41) | ( 6.88 )  | (157 .      | 76)(122 . 98) |

AdaRHD remain relatively stable. Notably, even when using a step size greater than 1 (i.e., a 0 = b 0 = c 0 = 0 . 2 ), although the performance is less stable compared to other settings, AdaRHD is still able to converge effectively. Table 2 also shows that AdaRHD-CG consistently achieves 85% validation accuracy in the shortest amount of time. On the other hand, step sizes of a 0 = b 0 = c 0 = 1 or 2 yield the best performance among all initializations for the AdaRHD variants, while the corresponding RHGD configurations fail to converge. These results further demonstrate the robustness of the proposed AdaRHD method, which significantly reduces the sensitivity to the choice of initial step size and greatly improves training efficiency. Moreover, although RHGD-20 requires the shortest time to reach 50% and 70% validation accuracies, it demands more time than our method to achieve 85% validation accuracy, demonstrating the efficiency of our approach. Meanwhile, the lower standard deviation exhibited by our method further highlights the robustness of our algorithm.

## 5 Conclusion

This paper proposes an adaptive algorithm for solving Riemannian bilevel optimization (RBO) problems, employing two strategies: gradient descent and conjugate gradient, to compute approximate Riemannian hypergradients. To our knowledge, this is the first fully adaptive RBO algorithm, incorporating a step size mechanism, eliminating prior knowledge requirements of problem parameters. We establish that the method achieves an O (1 /ϵ ) iteration complexity to reach an ϵ -stationary point, matching the complexity of the standard non-adaptive algorithms. Additionally, we show that substituting the exponential mapping with a computationally efficient retraction mapping maintains this complexity guarantee.

Notably, this work focuses exclusively on developing an adaptive double-loop algorithm for deterministic Riemannian bilevel optimization (RBO) problems when the lower-level objective is geodesically strongly convex. Future research directions include: (1) designing single-loop adaptive algorithms [93]; (2) extending the framework to stochastic settings [33, 23, 54] via adaptive step sizes such as inverse cumulative stochastic (hyper)gradient norms [92] or other strategies [22, 47, 86, 58, 64, 96, 70]; (3) addressing geodesically convex (non-strongly convex) lower-level objectives through regularization [2] or Polyak-Łojasiewicz (PŁ) conditions [15]; and (4) investigating alternative adaptive step size strategies [61, 96, 37, 70, 52, 76] to resolve the 1 /ϵ -gap in gradient ( G g ) and Hessian-vector product ( HV g ) complexities compared to non-adaptive Riemannian bilevel methods [33, 54].

## Acknowledgements

We sincerely appreciate the reviewers for their invaluable feedback and insightful suggestions. This work is partly supported by the National Key R&amp;D Program of China under grant 2023YFA1009300, National Natural Science Foundation of China under grants 12171100, and the Major Program of NFSC (72394360,72394364).

## References

- [1] P-A Absil, Robert Mahony, and Rodolphe Sepulchre. Optimization Algorithms on Matrix Manifolds . Princeton University Press, 2009.
- [2] Jan Harold Alcantara and Akiko Takeda. Theoretical smoothing frameworks for general nonsmooth bilevel problems. arXiv preprint arXiv:2401.17852 , 2024.
- [3] Gary Bécigneul and Octavian-Eugen Ganea. Riemannian adaptive optimization methods. In International Conference on Learning Representations , 2019.
- [4] Amir Beck. First-order methods in optimization , volume 25. SIAM, 2017.
- [5] Tamir Bendory, Yonina C Eldar, and Nicolas Boumal. Non-convex phase retrieval from stft measurements. IEEE Transactions on Information Theory , 64(1):467-484, 2017.
- [6] Luca Bertinetto, Joao F. Henriques, Philip Torr, and Andrea Vedaldi. Meta-learning with differentiable closed-form solvers. In International Conference on Learning Representations , 2019.
- [7] Rajendra Bhatia. Positive definite matrices . Princeton university press, 2009.
- [8] Nicholas Bishop, Long Tran-Thanh, and Enrico Gerding. Optimal learning from verified training data. In Advances in Neural Information Processing Systems , pages 9520-9529, 2020.
- [9] Silvere Bonnabel. Stochastic gradient descent on riemannian manifolds. IEEE Transactions on Automatic Control , 58(9):2217-2229, 2013.
- [10] Nicolas Boumal. An introduction to optimization on smooth manifolds . Cambridge University Press, 2023.
- [11] Nicolas Boumal and Pierre-antoine Absil. Rtrmc: A riemannian trust-region method for low-rank matrix completion. In Advances in Neural Information Processing Systems , pages 406-414, 2011.
- [12] Nicolas Boumal, Pierre-Antoine Absil, and Coralia Cartis. Global rates of convergence for nonconvex optimization on manifolds. IMA Journal of Numerical Analysis , 39(1):1-33, 2019.
- [13] Jerome Bracken and James T McGill. Mathematical programs with optimization problems in the constraints. Operations research , 21(1):37-44, 1973.
- [14] Michael Brückner and Tobias Scheffer. Stackelberg games for adversarial prediction problems. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining , pages 547-555, 2011.
- [15] Lesi Chen, Jing Xu, and Jingzhao Zhang. On finding small hyper-gradients in bilevel optimization: Hardness results and improved analysis. In Conference on Learning Theory , pages 947-980. PMLR, 2024.
- [16] Tianyi Chen, Yuejiao Sun, Quan Xiao, and Wotao Yin. A single-timescale method for stochastic bilevel optimization. In International Conference on Artificial Intelligence and Statistics , pages 2466-2488. PMLR, 2022.
- [17] Tianyi Chen, Yuejiao Sun, and Wotao Yin. Closing the gap: Tighter analysis of alternating stochastic gradient methods for bilevel problems. In Advances in Neural Information Processing Systems , pages 25294-25307, 2021.
- [18] Anoop Cherian and Suvrit Sra. Riemannian dictionary learning and sparse coding for positive definite matrices. IEEE Transactions on Neural Networks and Learning Systems , 28(12):28592871, 2016.
- [19] Mathieu Dagréou, Pierre Ablin, Samuel Vaiter, and Thomas Moreau. A framework for bilevel optimization that enables stochastic and global variance reduction algorithms. In Advances in Neural Information Processing Systems , pages 26698-26710, 2022.

- [20] Abhinav Dhall, Roland Goecke, Jyoti Joshi, Karan Sikka, and Tom Gedeon. Emotion recognition in the wild challenge 2014: Baseline, data and protocol. In Proceedings of the International Conference on Multimodal Interaction , pages 461-466, 2014.
- [21] Justin Domke. Generic methods for optimization-based modeling. In International Conference on Artificial Intelligence and Statistics , pages 318-326. PMLR, 2012.
- [22] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research , 12(7):2121-2159, 2011.
- [23] Sanchayan Dutta, Xiang Cheng, and Suvrit Sra. Riemannian bilevel optimization. arXiv preprint arXiv:2405.15816 , 2024.
- [24] Rémi Flamary, Alain Rakotomamonjy, and Gilles Gasso. Learning constrained task similarities in graphregularized multi-task learning. Regularization, Optimization, Kernels, and Support Vector Machines , 103:1, 2014.
- [25] Luca Franceschi, Michele Donini, Paolo Frasconi, and Massimiliano Pontil. Forward and reverse gradient-based hyperparameter optimization. In International Conference on Machine Learning , pages 1165-1173. PMLR, 2017.
- [26] Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In International Conference on Machine Learning , pages 1568-1577. PMLR, 2018.
- [27] Daniel Gabay. Minimizing a differentiable function over a differential manifold. Journal of Optimization Theory and Applications , 37:177-219, 1982.
- [28] Saeed Ghadimi and Mengdi Wang. Approximation methods for bilevel programming. arXiv preprint arXiv:1802.02246 , 2018.
- [29] Riccardo Grazzi, Luca Franceschi, Massimiliano Pontil, and Saverio Salzo. On the iteration complexity of hypergradient computation. In International Conference on Machine Learning , pages 3748-3758. PMLR, 2020.
- [30] Andi Han, Bamdev Mishra, Pratik Jawanpuria, and Junbin Gao. Nonconvex-nonconcave min-max optimization on riemannian manifolds. Transactions on Machine Learning Research , 2023.
- [31] Andi Han, Bamdev Mishra, Pratik Jawanpuria, Pawan Kumar, and Junbin Gao. Riemannian hamiltonian methods for min-max optimization on manifolds. SIAM Journal on Optimization , 33(3):1797-1827, 2023.
- [32] Andi Han, Bamdev Mishra, Pratik Kumar Jawanpuria, and Junbin Gao. On riemannian optimization over positive definite matrices with the bures-wasserstein geometry. Advances in Neural Information Processing Systems , 34:8940-8953, 2021.
- [33] Andi Han, Bamdev Mishra, Pratik Kumar Jawanpuria, and Akiko Takeda. A framework for bilevel optimization on riemannian manifolds. In Advances in Neural Information Processing Systems , volume 37, pages 103829-103872, 2024.
- [34] Mehrtash Harandi, Mathieu Salzmann, and Richard Hartley. Dimensionality reduction on spd manifolds: The emergence of geometry-aware methods. IEEE Transactions on Pattern Analysis and Machine Intelligence , 40(1):48-62, 2017.
- [35] Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A two-timescale stochastic algorithm framework for bilevel optimization: Complexity analysis and application to actorcritic. SIAM Journal on Optimization , 33(1):147-180, 2023.
- [36] Inbal Horev, Florian Yger, and Masashi Sugiyama. Geometry-aware principal component analysis for symmetric positive definite matrices. In Asian Conference on Machine Learning , pages 1-16. PMLR, 2016.
- [37] Jiang Hu, Ruicheng Ao, Anthony Man-Cho So, Minghan Yang, and Zaiwen Wen. Riemannian natural gradient methods. SIAM Journal on Scientific Computing , 46(1):A204-A231, 2024.

- [38] Zihao Hu, Guanghui Wang, Xi Wang, Andre Wibisono, Jacob D Abernethy, and Molei Tao. Extragradient type methods for riemannian variational inequality problems. In International Conference on Artificial Intelligence and Statistics , pages 2080-2088. PMLR, 2024.
- [39] Feihu Huang and Shangqian Gao. Gradient descent ascent for minimax problems on riemannian manifolds. IEEE Transactions on Pattern Analysis and Machine Intelligence , 45(7):8466-8476, 2023.
- [40] Zhiwu Huang and Luc Van Gool. A riemannian network for spd matrix learning. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 31, pages 2036-2042, 2017.
- [41] Kaiyi Ji, Jason D Lee, Yingbin Liang, and H Vincent Poor. Convergence of meta-learning with task-specific adaptation over partial parameters. In Advances in Neural Information Processing Systems , volume 33, pages 11490-11500, 2020.
- [42] Kaiyi Ji, Mingrui Liu, Yingbin Liang, and Lei Ying. Will bilevel optimizers benefit from loops. In Advances in Neural Information Processing Systems , volume 35, pages 3011-3023, 2022.
- [43] Kaiyi Ji, Junjie Yang, and Yingbin Liang. Bilevel optimization: Convergence analysis and enhanced design. In International Conference on Machine Learning , pages 4882-4892. PMLR, 2021.
- [44] Michael Jordan, Tianyi Lin, and Emmanouil-Vasileios Vlatakis-Gkaragkounis. First-order algorithms for min-max optimization in geodesic metric spaces. In Advances in Neural Information Processing Systems , volume 35, pages 6557-6574, 2022.
- [45] Hiroyuki Kasai, Pratik Jawanpuria, and Bamdev Mishra. Riemannian adaptive stochastic gradient algorithms on matrix manifolds. In International Conference on Machine Learning , pages 3262-3271. PMLR, 2019.
- [46] Hiroyuki Kasai, Hiroyuki Sato, and Bamdev Mishra. Riemannian stochastic recursive gradient algorithm. In International Conference on Machine Learning , pages 2516-2524. PMLR, 2018.
- [47] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations , 2015.
- [48] Max Kochurov, Rasul Karimov, and Serge Kozlukov. Geoopt: Riemannian optimization in pytorch. arXiv preprint arXiv:2005.02819 , 2020.
- [49] Vijay Konda and John Tsitsiklis. Actor-critic algorithms. In Advances in Neural Information Processing Systems , volume 12, pages 1008-1014, 1999.
- [50] Gautam Kunapuli, Kristin P Bennett, Jing Hu, and Jong-Shi Pang. Classification model selection via bilevel programming. Optimization Methods &amp; Software , 23(4):475-489, 2008.
- [51] Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert D Nowak. A fully first-order method for stochastic bilevel optimization. In International Conference on Machine Learning , pages 18083-18113. PMLR, 2023.
- [52] Puya Latafat, Andreas Themelis, Lorenzo Stella, and Panagiotis Patrinos. Adaptive proximal algorithms for convex optimization under local lipschitz continuity of the gradient. Mathematical Programming , 213:433-471, 2025.
- [53] John M Lee. Riemannian manifolds: an introduction to curvature , volume 176. Springer Science &amp; Business Media, 2006.
- [54] Jiaxiang Li and Shiqian Ma. Riemannian bilevel optimization. Journal of Machine Learning Research , 26(18):1-44, 2025.
- [55] Lizhen Lin, Drew Lazar, Bayan Saparbayeva, and David Dunson. Robust optimization and inference on manifolds. Statistica Sinica , 34:1299-1323, 2024.

- [56] Lizhen Lin, Brian St. Thomas, Hongtu Zhu, and David B Dunson. Extrinsic local regression on manifold-valued data. Journal of the American Statistical Association , 112(519):1261-1273, 2017.
- [57] Bo Liu, Mao Ye, Stephen Wright, Peter Stone, and Qiang Liu. Bome! bilevel optimization made easy: A simple first-order approach. In Advances in Neural Information Processing Systems , volume 35, pages 17248-17262, 2022.
- [58] Nicolas Loizou, Sharan Vaswani, Issam Hadj Laradji, and Simon Lacoste-Julien. Stochastic polyak step-size for sgd: An adaptive learning rate for fast convergence. In International Conference on Artificial Intelligence and Statistics , pages 1306-1314. PMLR, 2021.
- [59] Jonathan Lorraine, Paul Vicol, and David Duvenaud. Optimizing millions of hyperparameters by implicit differentiation. In International Conference on Artificial Intelligence and Statistics , pages 1540-1552. PMLR, 2020.
- [60] Dougal Maclaurin, David Duvenaud, and Ryan Adams. Gradient-based hyperparameter optimization through reversible learning. In International Conference on Machine Learning , pages 2113-2122. PMLR, 2015.
- [61] Yura Malitsky and Konstantin Mishchenko. Adaptive gradient descent without descent. In International Conference on Machine Learning , pages 6702-6712. PMLR, 2020.
- [62] David Martínez-Rubio, Christophe Roux, Christopher Criscitiello, and Sebastian Pokutta. Accelerated methods for riemannian min-max optimization ensuring bounded geometric penalties. In International Conference on Machine Learning , pages 280-288. PMLR, 2025.
- [63] Bamdev Mishra, Hiroyuki Kasai, Pratik Jawanpuria, and Atul Saroop. A riemannian gossip approach to subspace learning on grassmann manifold. Machine Learning , 108:1783-1803, 2019.
- [64] Antonio Orvieto, Simon Lacoste-Julien, and Nicolas Loizou. Dynamics of sgd with stochastic polyak stepsizes: Truly adaptive variants and convergence to exact solution. In Advances in Neural Information Processing Systems , volume 35, pages 26943-26954, 2022.
- [65] Fabian Pedregosa. Hyperparameter optimization with approximate gradient. In International Conference on Machine Learning , pages 737-746. PMLR, 2016.
- [66] Peter Petersen. Riemannian geometry , volume 171. Springer, 2006.
- [67] Aravind Rajeswaran, Chelsea Finn, Sham M Kakade, and Sergey Levine. Meta-learning with implicit gradients. In Advances in Neural Information Processing Systems , volume 32, pages 113-124, 2019.
- [68] Soumava Kumar Roy, Zakaria Mhammedi, and Mehrtash Harandi. Geometry aware constrained optimization techniques for deep learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 4460-4469, 2018.
- [69] Hiroyuki Sakai and Hideaki Iiduka. Riemannian adaptive optimization algorithm and its application to natural language processing. IEEE Transactions on Cybernetics , 52(8):73287339, 2021.
- [70] Hiroyuki Sakai and Hideaki Iiduka. A general framework of riemannian adaptive optimization methods with a convergence analysis. arXiv preprint arXiv:2409.00859 , 2024.
- [71] Hiroyuki Sato. A dai-yuan-type riemannian conjugate gradient method with the weak wolfe conditions. Computational Optimization and Applications , 64:101-118, 2016.
- [72] Hiroyuki Sato, Hiroyuki Kasai, and Bamdev Mishra. Riemannian stochastic variance reduced gradient algorithm with retraction and vector transport. SIAM Journal on Optimization , 29(2):1444-1472, 2019.
- [73] Amirreza Shaban, Ching-An Cheng, Nathan Hatch, and Byron Boots. Truncated backpropagation for bilevel optimization. In International Conference on Artificial Intelligence and Statistics , pages 1723-1732. PMLR, 2019.

- [74] Daouda Sow, Kaiyi Ji, and Yingbin Liang. On the convergence theory for hessian-free bilevel algorithms. In Advances in Neural Information Processing Systems , volume 35, pages 4136-4149, 2022.
- [75] Suvrit Sra and Reshad Hosseini. Conic geometric optimization on the manifold of positive definite matrices. SIAM Journal on Optimization , 25(1):713-739, 2015.
- [76] Jaewook J Suh and Shiqian Ma. An adaptive and parameter-free nesterov's accelerated gradient method for convex optimization. arXiv preprint arXiv:2505.11670 , 2025.
- [77] Rhea Sanjay Sukthanker, Zhiwu Huang, Suryansh Kumar, Erik Goron Endsjo, Yan Wu, and Luc Van Gool. Neural architecture search of spd manifold networks. In Proceedings of the International Joint Conference on Artificial Intelligence , pages 3002-3009. International Joint Conferences on Artificial Intelligence Organization, 2021.
- [78] Ju Sun, Qing Qu, and John Wright. Complete dictionary recovery over the sphere ii: Recovery by riemannian trust-region method. IEEE Transactions on Information Theory , 63(2):885-914, 2016.
- [79] Ju Sun, Qing Qu, and John Wright. A geometric analysis of phase retrieval. Foundations of Computational Mathematics , 18:1131-1198, 2018.
- [80] Hadi Tabealhojeh, Peyman Adibi, Hossein Karshenas, Soumava Kumar Roy, and Mehrtash Harandi. Rmaml: Riemannian meta-learning with orthogonality constraints. Pattern Recognition , 140:109563, 2023.
- [81] Lloyd N Trefethen and David Bau. Numerical linear algebra . SIAM, 2022.
- [82] Nilesh Tripuraneni, Nicolas Flammarion, Francis Bach, and Michael I Jordan. Averaging stochastic gradient descent on riemannian manifolds. In Conference On Learning Theory , pages 650-687. PMLR, 2018.
- [83] Ioannis Tsaknakis, Prashant Khanduri, and Mingyi Hong. An implicit gradient-type method for linearly constrained bilevel problems. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing , pages 5438-5442. IEEE, 2022.
- [84] Loring W Tu. An Introduction to Manifolds . Springer, 2011.
- [85] Bart Vandereycken. Low-rank matrix completion by riemannian optimization. SIAM Journal on Optimization , 23(2):1214-1236, 2013.
- [86] Sharan Vaswani, Aaron Mishkin, Issam Laradji, Mark Schmidt, Gauthier Gidel, and Simon Lacoste-Julien. Painless stochastic gradient: Interpolation, line-search, and convergence rates. In Advances in Neural Information Processing Systems , volume 32, pages 3732-3745, 2019.
- [87] Jiali Wang, He Chen, Rujun Jiang, Xudong Li, and Zihao Li. Fast algorithms for stackelberg prediction game with least squares loss. In International Conference on Machine Learning , pages 10708-10716. PMLR, 2021.
- [88] Jiali Wang, Wen Huang, Rujun Jiang, Xudong Li, and Alex L Wang. Solving stackelberg prediction game with least squares loss via spherically constrained least squares reformulation. In International Conference on Machine Learning , pages 22665-22679. PMLR, 2022.
- [89] Xi Wang, Deming Yuan, Yiguang Hong, Zihao Hu, Lei Wang, and Guodong Shi. Riemannian optimistic algorithms. arXiv preprint arXiv:2308.16004 , 2023.
- [90] Rachel Ward, Xiaoxia Wu, and Leon Bottou. Adagrad stepsizes: Sharp convergence over nonconvex landscapes. Journal of Machine Learning Research , 21(219):1-30, 2020.
- [91] Quan Xiao, Han Shen, Wotao Yin, and Tianyi Chen. Alternating projected sgd for equalityconstrained bilevel optimization. In International Conference on Artificial Intelligence and Statistics , pages 987-1023. PMLR, 2023.

- [92] Yuege Xie, Xiaoxia Wu, and Rachel Ward. Linear convergence of adaptive stochastic gradient descent. In International Conference on Artificial Intelligence and Statistics , pages 1475-1485. PMLR, 2020.
- [93] Yifan Yang, Hao Ban, Minhui Huang, Shiqian Ma, and Kaiyi Ji. Tuning-free bilevel optimization: New algorithms and convergence analysis. In International Conference on Learning Representations , 2025.
- [94] Wei Yao, Chengming Yu, Shangzhi Zeng, and Jin Zhang. Constrained bi-level optimization: Proximal lagrangian value function approach and hessian-free algorithm. In International Conference on Learning Representations , 2024.
- [95] Tong Yu and Hong Zhu. Hyper-parameter optimization: A review of algorithms and applications. arXiv preprint arXiv:2003.05689 , 2020.
- [96] Jihun Yun and Eunho Yang. Riemannian sam: Sharpness-aware minimization on riemannian manifolds. In Advances in Neural Information Processing Systems , volume 36, pages 6578465800, 2023.
- [97] Pourya Zadeh, Reshad Hosseini, and Suvrit Sra. Geometric mean metric learning. In International Conference on Machine Learning , pages 2464-2471. PMLR, 2016.
- [98] Emanuele Zangrando, Francesco Rinaldi, Francesco Tudisco, et al. debora: Efficient bilevel optimization-based low-rank adaptation. In International Conference on Learning Representations , 2025.
- [99] Hongyi Zhang, Sashank J Reddi, and Suvrit Sra. Riemannian svrg: Fast stochastic optimization on riemannian manifolds. In Advances in Neural Information Processing Systems , volume 29, pages 4599-4607, 2016.
- [100] Hongyi Zhang and Suvrit Sra. First-order methods for geodesically convex optimization. In Conference on Learning Theory , pages 1617-1638. PMLR, 2016.
- [101] Jingzhao Zhang, Hongyi Zhang, and Suvrit Sra. R-spider: A fast riemannian stochastic optimization algorithm with curvature independent rate. arXiv preprint arXiv:1811.04194 , 2018.
- [102] Peiyuan Zhang, Jingzhao Zhang, and Suvrit Sra. Sion's minimax theorem in geodesic metric spaces and a riemannian extragradient algorithm. SIAM Journal on Optimization , 33(4):28852908, 2023.
- [103] Danqing Zhou, Shiqian Ma, and Junfeng Yang. Adabb: Adaptive barzilai-borwein method for convex optimization. Mathematics of Operations Research , 2025.
- [104] Pan Zhou, Xiao-Tong Yuan, and Jiashi Feng. Faster first-order methods for stochastic nonconvex optimization on riemannian manifolds. In International Conference on Artificial Intelligence and Statistics , pages 138-147. PMLR, 2019.

## Appendix

## A Related works

Bilevel optimization: Bilevel optimization is first introduced by [13] and rooted in the Stackelberg game [14, 8, 87, 88]. When the lower-level objective is strongly convex, various methods have been proposed [28, 17, 43, 42, 19, 57, 35, 51]. Hypergradient-based approaches include approximate implicit differentiation (AID) [21, 65]; iterative differentiation (ITD) [60, 25, 73, 29]; Neumann series (NS) [28]; and conjugate gradient (CG) [43]. Recent studies address bilevel problems with constraints on either the upper- or lower-level objective. For example, [16, 35] focus on scenarios where constraints are imposed solely on the upper-level objective, whereas [83, 91, 94] develop methods for problems with lower-level constraints. For a comprehensive overview, we refer readers to [43, 42, 51] and the references therein.

Riemannian optimization: Riemannian optimization has attracted considerable attention due to its broad applications, including low-rank matrix completion [11, 85]; phase retrieval [5, 79]; dictionary learning [18, 78]; dimensionality reduction [34, 82, 63]; and manifold regression [56, 55]. Various methods have been developed, such as Riemannian (stochastic) gradient descent [27, 1, 9, 5, 12]; nonlinear conjugate gradients [71]; and variance-reduced stochastic gradients [99, 101, 46, 72, 104]. For further details, see [10] and references therein.

Riemannian adaptive optimization: Several adaptive algorithms have been developed to solve Riemannian optimization problems. Notable examples include the Riemannian adaptive stochastic gradient algorithm (RASA) [45], Riemannian AMSGrad (RAMSGrad) [3], modified RAMSGrad [69], constrained RMSProp (cRMSProp) [68], sharpness-aware minimization on Riemannian manifolds (Riemannian SAM) [96], Riemannian natural gradient descent (RNGD) [37], and a framework for Riemannian adaptive optimization [70]. For a comprehensive overview of their application scopes, we refer readers to [70].

Riemannian bilevel optimization: Research on Riemannian bilevel optimization remains limited to a few recent studies. [54] investigated hypergradient computation for such problems on Riemannian manifolds and developed deterministic and stochastic algorithms, namely the algorithm for Riemannian bilevel optimization (RieBO) and the algorithm for Riemannian stochastic bilevel optimization (RieSBO), for solving Riemannian bilevel and stochastic bilevel optimization, respectively. In parallel, [33] proposed a framework termed Riemannian hypergradient descent (RHGD), which provides multiple hypergradient estimation strategies, supported by convergence and complexity analyses. The authors also extended their framework to address Riemannian minimax and compositional optimization problems. By leveraging the value-function reformulation and the Lagrangian method, [23] introduced a fully stochastic first-order approach, the Riemannian first-order fast stochastic approximation (RF 2 SA), which is applicable to scenarios where both objectives are stochastic or deterministic.

Adaptive bilevel optimization: The closest related approach to our method is the D-TFBO proposed in [93], which tackles Euclidean bilevel optimization by employing a similar adaptive step size strategy. While some aspects of Euclidean analysis extend directly to Riemannian settings, geometric curvature introduces distortions that create unique analytical challenges. We derive convergence guarantees similar to those of D-TFBO and enhance our algorithm's efficiency by replacing exponential mapping with retraction mapping, thereby reducing computational overhead. To our knowledge, this work presents the first fully adaptive method with theoretical convergence guarantees for solving Riemannian bilevel optimization problems.

## B Additional preliminaries for Section 2

The Lipschitzness of the functions and operators in the Riemannian manifolds is defined as follows: Definition B.1. [33, Definition 1] For any x, x 1 , x 2 ∈ U x , y, y 1 , y 2 ∈ U y , where U x × U y ⊆ M x ×M y ,

- (i) a function f : M x → R is said to have L -Lipschitz Riemannian gradient in U x if ∥P x 2 x 1 G f ( x 1 ) -G f ( x 2 ) ∥ x 2 ≤ Ld ( x 1 , x 2 ) .

- (ii) a bi-function f : M x ×M y → R is said to have L -Lipschitz Riemannian gradient in U x ×U y if ∥P y 2 y 1 G y f ( x, y 1 ) -G y f ( x, y 2 ) ∥ y 2 ≤ Ld ( y 1 , y 2 ) , ∥G x f ( x, y 1 ) -G x f ( x, y 2 ) ∥ x ≤ Ld ( y 1 , y 2 ) , ∥P x 2 x 1 G x f ( x 1 , y ) - G x f ( x 2 , y ) ∥ x 2 ≤ Ld ( x 1 , x 2 ) and ∥G y f ( x 1 , y ) - G y f ( x 2 , y ) ∥ y ≤ Ld ( x 1 , x 2 ) .
- (iii) a linear operator G ( x, y ) : T y M y → T x M x (e.g. G xy g ( x, y ) ), is said to be ρ -Lipschitz in U x × U y if ∥P x 2 x 1 G ( x 1 , y ) - G ( x 2 , y ) ∥ x 2 ≤ ρd ( x 1 , x 2 ) and ∥G ( x, y 1 ) - G ( x, y 2 ) P y 2 y 1 ∥ x ≤ ρd ( y 1 , y 2 ) .
- (iv) a linear operator H ( x, y ) : T y M y → T y M y (e.g. H y g ( x, y ) ), is said to be ρ -Lipschitz in U x × U y if ∥P y 2 y 1 H ( x, y 1 ) P y 1 y 2 - H ( x, y 2 ) ∥ y 2 ≤ ρd ( y 1 , y 2 ) and ∥H ( x 1 , y ) - H ( x 2 , y ) ∥ y ≤ ρd ( x 1 , x 2 ) .

Due to the curvature of Riemannian manifolds, the notion of distance differs from that in Euclidean space. Accordingly, we have the following trigonometric distance bound on Riemannian manifolds.

Lemma B.1 (Trigonometric distance bound [100, 99, 32]) . Let x a , x b , x c ∈ U ⊆ M and denote a = d ( x b , x c ) , b = d ( x a , x c ) and c = d ( x a , x b ) as the geodesic side lengths. Then, it holds that

<!-- formula-not-decoded -->

where ζ ( τ, c ) = √ | τ | c tanh( √ | τ | c ) if τ &lt; 0 and ζ ( τ, c ) = 1 if τ ≥ 0 , and τ denotes the lower bound of the sectional curvature of U [66, Section 3.1.3].

## C Tangent space conjugate gradient algorithm

In this section, we introduce the tangent-space conjugate gradient method [81, 10], which is used in Section 3.1.

Algorithm 2 Tangent Space Conjugate Gradient v n = TSCG( H y g ( x, ˆ y ) , G y f ( x, ˆ y ) , v 0 , ϵ v )

- 10:
- 1: Let r 0 = G y f ( x, ˆ y ) ∈ T y M y , p 0 = r 0 . 2: while ∥H y g ( x, ˆ y )[ v n ] -G y f ( x, ˆ y ) ∥ 2 ˆ y &gt; ϵ v do 3: Compute H y g ( x, ˆ y )[ p n ] . 4: a n +1 = ∥ r n ∥ 2 ˆ y ⟨ p n , H y g ( x, ˆ y )[ p n ] ⟩ ˆ y , 5: v n +1 = v n + a n +1 p n , 6: r n +1 = r n -a n +1 H y g ( x, ˆ y )[ p n ] , 7: b n +1 = ∥ r n +1 ∥ 2 ˆ y ∥ r n ∥ 2 ˆ y , 8: p n +1 = r n +1 + b n +1 p n , 9: n = n +1 . end while

## D Additional results of Section 3.4

Since the retraction mapping only affects the total number of iterations required for the lowerlevel problem in Problem (1) to converge, without influencing the error bounds, the hypergradient approximation error bound (Lemma 3.1) remains consistent with that in Algorithm 1. Moreover, analogous to Proposition 3.2, we establish the following upper bounds on the total number of iterations required to solve the lower-level problem in Problem (1) and the linear system (4).

Proposition D.1. Suppose that Assumptions 3.1, 3.2, 3.3, 3.4, 3.5, and 3.6 hold. Then, for any 0 ≤ t ≤ T , the iterations K t and N t required in Algorithm 3 satisfy:

<!-- formula-not-decoded -->

## Algorithm 3 AdaRHD with Retraction (AdaRHD-R)

- 1: Initial points x 0 ∈ M x , y 0 ∈ M y , and v 0 ∈ T y 0 M , initial step sizes a 0 &gt; 0 , b 0 &gt; 0 , and c 0 &gt; 0 , total iterations T , and error tolerances ϵ y = ϵ v = 1 T .
- 2: for t = 0 , 1 , 2 , ..., T -1 do
- 3: Set k = 0 and y 0 t = y K t -1 t -1 if t &gt; 0 and y 0 otherwise.
- 4: while ∥G y g ( x t , y k t ) ∥ 2 y k t &gt; ϵ y do
- 5: b 2 k +1 = b 2 k + ∥G y g ( x t , y k t ) ∥ 2 y k t ,
- 6: y k +1 t = Retr y k t ( -1 b k +1 G y g ( x t , y k t )) ,
- 7: k = k +1 .
- 8: end while
- 9: K t = k .
- 10: Set n = 0 and v 0 t = P y Kt t y Kt -1 t -1 v N t -1 t -1 if t &gt; 0 and v 0 otherwise.
- 11: while ∥∇ v R ( x t , y K t t , v n t ) ∥ 2 y Kt t &gt; ϵ v do
- 12: c 2 n +1 = c 2 n + ∥∇ v R ( x t , y K t t , v n t ) ∥ 2 y Kt t ,
- 13: v n +1 t = v n t -1 c n +1 ∇ v R ( x t , y K t t , v n t ) , ▷ Gradient descent
- 14: n = n +1 .
- 15: end while
- 16: Or set v 0 t = 0 and invoke v n t = TSCG( H y g ( x t , y K t t ) , G y f ( x t , y K t t ) , v 0 t , ϵ v ) . ▷ Conjugate gradient
- 17: N t = n .

<!-- formula-not-decoded -->

- 19: a 2 t +1 = a 2 t + ∥ ̂ G F ( x t , y K t t , v N t t ) ∥ 2 x t ,
- 20: x t +1 = Retr x t ( -1 a t +1 ̂ G F ( x t , y K t t , v N t t )) .
- 21: end for

## AdaRHD-GD:

<!-- formula-not-decoded -->

where C b , C c , ¯ b , ¯ c , ˜ b , ˜ c , and ¯ ζ are constants defined in Appendix H.

AdaRHD-CG:

<!-- formula-not-decoded -->

## E Extension to Riemannian min-max Problems

Riemannian min-max problems [44, 39, 31, 102, 89, 30, 62, 38] have recently attracted increasing attention. The general form of such problems is

<!-- formula-not-decoded -->

which can be regarded as a special case of the Riemannian bilevel optimization problem (1) with g ( x, y ) = -f ( x, y ) .

If g is geodesically strongly convex w.r.t. y over M y , the Riemannian hypergradient (3) reduces to G F ( x ) = G x f ( x, y ∗ ( x )) , and the approximate Riemannian hypergradient becomes ̂ G F ( x, ˆ y ) = G x f ( x, ˆ y ) . The pseudocode of the adaptive method for solving Problem (6) is summarized in Algorithm 4.

Before establishing the convergence result of Algorithm 4, we present the required assumptions.

Assumption E.1. The following conditions hold:

## Algorithm 4 AdaRHD for Riemannian Min-Max Optimization

- 1: Initial points x 0 ∈ M x and y 0 ∈ M y , initial step sizes a 0 &gt; 0 , and b 0 &gt; 0 , total iterations T , and error tolerance ϵ y = 1 T .
- 2: for t = 0 , 1 , 2 , . . . , T -1 do
- 3: Set k = 0 and initialize y 0 t = y K t -1 t -1 if t &gt; 0 , otherwise y 0 t = y 0 .
- 4: while ∥G y g ( x t , y k t ) ∥ 2 &gt; ϵ y do
- 5: Update b 2 k +1 = b 2 k + ∥G y g ( x t , y k t ) ∥ 2 y k t .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 8: k ← k +1 .
- 9: end while
- 10: K t ← k .
- 11: Compute ̂ G F ( x t , y K t t ) = G x f ( x t , y K t t ) .
- 12: Update a 2 t +1 = a 2 t + ∥ ̂ G F ( x t , y K t t ) ∥ 2 x t .
- 13: x t +1 = Exp x t ( -1 a t +1 ̂ G F ( x t , y K t t ) ) ,
- 14: or x t +1 = Retr x t ( -1 a t +1 ̂ G F ( x t , y K t t ) ) .
- 15: end for
- (i) Assumptions 3.1 and 3.4 are satisfied (Assumptions 3.5 and 3.6 are required when retraction is used);
- (ii) f ( x, y ) is continuously differentiable, and -f ( x, y ) is geodesically strongly convex w.r.t. y ;
- (iii) f ( x, y ) is l f -Lipschitz continuous, and its Riemannian gradients G x f ( x, y ) and G y f ( x, y ) are L f -Lipschitz continuous.

We now state the convergence guarantee of Algorithm 4.

Theorem E.1. Suppose that Assumption E.1 holds. Then, the sequence { x t } T t =0 generated by Algorithm 4 satisfies

<!-- formula-not-decoded -->

where C mm is a constant depending on the curvature, Lipschitz, and strong convexity parameters. Moreover, the overall gradient complexity of Algorithm 4 is O (1 /ϵ 2 ) .

## F Preliminary results for proofs

Lemma F.1. [90, Lemma 3.2] For any non-negative α 1 , ..., α T , and α 1 ≥ 1 , we have

<!-- formula-not-decoded -->

Lemma F.2. Suppose that Assumptions 3.1, 3.2, and 3.3 hold. Then, the following statements hold,

- (1) For x 1 , x 2 ∈ M x , it holds that

<!-- formula-not-decoded -->

where L y := L g µ and y ∗ ( x ) is the optimal solution of the lower-level problem in Problem (1) ;

- (2) The Riemannian hypergradient G F ( x ) satsifies ∥G F ( x ) ∥ x ≤ l f (1 + L g /µ ) , and for x 1 , x 2 ∈ M x , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

▷ retraction step

Proof. The proofs of Lemma F.2 can be derived from [28, Lemma 2.2]; here we provide them for completeness.

(1): For x 1 , x 2 ∈ M x , there exist two geodesics c 1 : [0 , 1] →M y and c 2 : [0 , 1] →M x such that c 1 ( t ) = y ∗ ( c 2 ( t )) for t ∈ [0 , 1] , and c 2 (0) = x 1 , c 2 (1) = x 2 . Then, we have

<!-- formula-not-decoded -->

where the first inequality follows from Dy ∗ ( x ) := -H -1 y g ( x, y ∗ ( x )) G 2 yx g ( x, y ∗ ( x )) .

- (2): By the definition of G F ( x ) , from Assumptions 3.2 and 3.3, it holds that

<!-- formula-not-decoded -->

Furthermore, we have

<!-- formula-not-decoded -->

By Assumptions 3.2 and 3.3, for the first term of (7), we have

<!-- formula-not-decoded -->

We then consider the fourth term of (7), we have

<!-- formula-not-decoded -->

For the second term of (7), we have

<!-- formula-not-decoded -->

For the third term of (7), we have

<!-- formula-not-decoded -->

where the last inequality follows from the operator norm satisfying

<!-- formula-not-decoded -->

Combining the above inequalities, we have

<!-- formula-not-decoded -->

The proof is complete.

<!-- formula-not-decoded -->

Denote v ( x ) := arg min v ∈ T y ∗ ( x ) M y R ( x, y ( x ) , v ) and ˆ v ( x, y ) := arg min v ∈ T y M y R ( x, y, v ) . Lemma F.3. Suppose that Assumptions 3.1, 3.2, and 3.3 hold. Then, the following statements hold,

- (1) The objective R ( x, y, v ) is µ -strongly convex and L g -smooth w.r.t. v ;

<!-- formula-not-decoded -->

- (3) For x, x 1 , x 2 ∈ M x and y 1 , y 2 ∈ M y , it holds that

<!-- formula-not-decoded -->

Proof. (1): Since ∇ 2 v R ( x, y, v ) = H y g ( x, y ) , the strong convexity of R ( x, y, v ) follows from Assumption 3.2. Then, by Assumption 3.3, for v 1 , v 2 ∈ T y M y , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(2): For ˆ v ( x, y ) , we have

<!-- formula-not-decoded -->

which demonstrates that

<!-- formula-not-decoded -->

Since v ∗ ( x ) satisfies v ∗ ( x ) = ˆ v ∗ ( x, y ∗ ( x )) , the desired result follows.

(3): By the definition of v ∗ ( x ) , we have

<!-- formula-not-decoded -->

The remaining proof is similar to that of Lemma F.2 (2), thus, we omit it.

By the definition of ˆ v ∗ ( x, y ) , we have

<!-- formula-not-decoded -->

where the second inequality follows from (8).

Lemma F.4. Suppose that Assumptions 3.1, 3.2, and 3.3 hold. Then, for any t ≥ 0 , we have

<!-- formula-not-decoded -->

Proof. According to the termination conditions of Algorithm 1 for the lower-level problem and the linear system, we have

<!-- formula-not-decoded -->

By the strong convexity of g and R (cf. Lemma F.3), we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Lemma F.5. Suppose that Assumptions 3.1, 3.2, and 3.3 hold. Then, for any t ≥ 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By the definition of ̂ G F , we have

<!-- formula-not-decoded -->

where the last inequality follows from Assumptions 3.1, 3.2, 3.3, Lemmas F.3 and F.4.

## G Proofs of Section 3.3

## G.1 Proof of Lemma 3.1

Proof.

<!-- formula-not-decoded -->

where the third and fourth inequalities follow from Assumption 3.3 and Lemma F.3 (2), respectively.

For ∥ v N t t -P y Kt t y ∗ ( x t ) v ∗ t ( x t ) ∥ y Kt t , we have

<!-- formula-not-decoded -->

where the last inequality follows from Lemma F.3 (3) and the fact that v ∗ t ( x t ) = ˆ v ∗ t ( x t , y ∗ ( x t )) . Combining this result with (9), we obtain

<!-- formula-not-decoded -->

The proof is complete.

## G.2 Proof of Proposition 3.2

Inspired by Proposition 1 in [93], we first give a result that concerns the step sizes a t , b k , and c n .

Proposition G.1. Suppose that Assumptions 3.1, 3.2, and 3.3 hold. Denote { T, K, N } as the iterations of { x, y, v } . Given any constants C a ≥ a 0 , C b ≥ b 0 , C c ≥ c 0 , then, we have AdaRHD-GD:

- (1) either a t ≤ C a for any t ≤ T , or ∃ t 1 ≤ T such that a t 1 ≤ C a , a t 1 +1 &gt; C a ;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We then consider the case where we use the conjugate gradient method to solve the linear system. Therefore, we do not need to consider the step size c n . AdaRHD-CG:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## G.2.1 Improvement on objective function for one step update

We first define a threshold for a t :

<!-- formula-not-decoded -->

Lemma G.1. Suppose that Assumptions 3.1, 3.2, and 3.3 hold. Then, we have

<!-- formula-not-decoded -->

Furthermore, if t 1 in Proposition G.1 exists, then for any t ≥ t 1 , we have

<!-- formula-not-decoded -->

where ϵ y,v := ¯ L 2 µ 2 ( ϵ y + ϵ v ) and ¯ L := max { √ 2 L 1 , √ 2 L g } with L 1 is defined in Lemma 3.1.

Proof. Combining Lemma F.2 (2) and Proposition 2.1, we have

<!-- formula-not-decoded -->

By Lemma 3.1, it holds that

<!-- formula-not-decoded -->

Therefore, by Lemma F.4, we have

<!-- formula-not-decoded -->

This Combine (13) deduces the desired result of (11).

If t 1 in Proposition G.1 exists , then for t ≥ t 1 , we have a t +1 &gt; C a ≥ 2 L F . The desired result of (12) follows from (11). The proof is complete.

Then, similar to [93, Lemma 8], we have the following upper bound for the step size a t .

Lemma G.2. Suppose that Assumptions 3.1, 3.2, 3.3, and 3.4 hold. If t 1 in Proposition G.1 does not exist, we have a t ≤ C a for all t ≤ T .

If the t 1 in Proposition G.1 exists, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Proof. If t 1 in Proposition G.1 does not exist , then for any t ≤ T , it holds that a t ≤ C a .

If t 1 in Proposition G.1 exists , then for any t &lt; t 1 , it holds that a t +1 ≤ C a . By Lemma G.1, for any t ≥ t 1 , it holds that

<!-- formula-not-decoded -->

Removing the nonnegative term -1 2 a t +1 ∥G F ( x t ) ∥ 2 x t , we have

<!-- formula-not-decoded -->

Summing (15) from t 1 to t , we have

<!-- formula-not-decoded -->

For F ( x t 1 ) , by (11), we have

<!-- formula-not-decoded -->

This combines with (16), by F ( x t +1 ) ≥ F ∗ , we have

<!-- formula-not-decoded -->

Since a 2 t +1 = a 2 t + ∥ ̂ G F ( x t , y K t t , v N t t ) ∥ 2 x t , we have

<!-- formula-not-decoded -->

The proof is complete.

Similar to (10), we define the thresholds for the step sizes b k and c n as follows:

<!-- formula-not-decoded -->

Before proving Proposition 3.2, we need two technical lemmas.

Lemma G.3. Suppose that Assumptions 3.1, 3.2, and 3.3 hold. Then, we have

<!-- formula-not-decoded -->

Proof. By Lemma B.1, we have

<!-- formula-not-decoded -->

where the second inequality follows from the geodesic strong convexity of g , and the third inequality follows from

<!-- formula-not-decoded -->

The proof is complete.

Lemma G.4. Suppose that Assumptions 3.1, 3.2, and 3.3 hold. Then, we have

<!-- formula-not-decoded -->

Proof. By Assumption 3.3, given any y ∈ M y , it holds that

<!-- formula-not-decoded -->

Taking y = Exp y k t ( -1 L g G y g ( x t , y k t )) in (20) derives

<!-- formula-not-decoded -->

which demonstrates that

<!-- formula-not-decoded -->

By the geodesic strong convexity of g w.r.t. y , we have

<!-- formula-not-decoded -->

This combines (21) implies that

<!-- formula-not-decoded -->

## Proof of Proposition 3.2

Proof. For AdaRHD-GD, denote

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where ˜ b := max { L g ( ¯ b -C b ) 2 , 1 } , ˜ c := max { L g (¯ c -C c ) , 1 } , ¯ b := C b + 2 L g ( 2 ϵ y µ 2 + 2 L 2 g C 2 f µ 2 a 2 0 +2 ζ log ( C b b 0 ) + ζ ) with C f = ( 2 L 2 g ϵ v µ 2 + 4 L 2 g l 2 f µ 2 +4 l 2 f ) 1 2 is defined in Lemma F.5, and ¯ c := C c + L g ( 2 ϵ y µ 2 + 8 l 2 f µ 2 +2log ( C c c 0 ) +1 ) .

For AdaRHD-CG, denote

<!-- formula-not-decoded -->

We first show that K t ≤ ¯ K for all 0 ≤ t ≤ T .

If k 1 in Proposition G.1 does not exist , it holds that b K t ≤ C b . By [92, Lemma 2], we must have K t ≤ log( C 2 b /b 2 0 ) log(1+ ϵ y /C 2 b ) . If K t &gt; log( C 2 b /b 2 0 ) log(1+ ϵ y /C 2 b ) , since ∥G y g ( x t , y p t ) ∥ 2 y k t ≥ ϵ y and b k ≤ C b hold for all k &lt; K t , we have

<!-- formula-not-decoded -->

which contradicts b K t ≤ C b .

If k 1 in Proposition G.1 exists , then, we have b k 1 ≤ C b and b k 1 +1 &gt; C b . We first prove k 1 ≤ log( C 2 b /b 2 0 ) log(1+ ϵ y /C 2 b ) . If k 1 &gt; log( C 2 b /b 2 0 ) log(1+ ϵ y /C 2 b ) , similar to (24), we have

<!-- formula-not-decoded -->

which contradicts b k 1 ≤ C b .

By Lemma B.1 and the update mode of y k 1 t , we have

<!-- formula-not-decoded -->

where (a) follows from (2), (b) follows from the warm start of y 0 t , (c) follows from [10, Definition 10.1] and Lemma F.1, (d) uses Lemmas F.2 (1) and F.4, (e) follows from Lemma F.5 and b k 1 ≤ C b .

For any K &gt; k 1 , by Lemma G.3, we have

<!-- formula-not-decoded -->

where the second inequality follows from b K ≥ C b ≥ 2 ζL g L g µ and 1 -x ≤ e -x for 0 &lt; x &lt; 1 , specifically, when b K ≥ 2 ζL g L g µ , it holds that 0 &lt; µ b K -ζ L 2 g b 2 K &lt; 1 , and the third inequality follows from (25).

Similar to (17), we have

<!-- formula-not-decoded -->

Moreover, by the update mode of y t K and Lemma B.1, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second and third inequalities follow from Lemma G.4 and b K ≥ C b ≥ 2 ζL g L g µ , respectively.

By (25), we have

<!-- formula-not-decoded -->

Then, by (27), 1 1 -µ/ 2 L g ≤ 2 , and b k 1 ≤ C b , we have

<!-- formula-not-decoded -->

Denote ¯ b := C b +2 L g ( 2 ϵ y µ 2 + 2 L 2 g C 2 f µ 2 a 2 0 +2 ζ log ( C b b 0 ) + ζ ) . We need to show that µ ¯ b -ζ L 2 g ¯ b 2 &gt; 0 . Under (26), considering the monotonicity of µ b -ζ L 2 g b 2 w.r.t. b when b ≥ 2 ζL g L g µ , it holds that 0 &lt; µ ¯ b -ζ L 2 g ¯ b 2 ≤ µ b K -ζ L 2 g b 2 K as ¯ b ≥ b K ≥ 2 ζL g L g µ . Then, we have

<!-- formula-not-decoded -->

Since k 1 ≤ log( C 2 b /b 2 0 ) log(1+ ϵ y /C 2 b ) , by the definition of ¯ K in (22), it holds that ¯ K &gt; k 1 as ˜ b ≥ 1 &gt; ϵ y . If L g ( ¯ b -C b ) 2 ≤ 1 , it holds that ˜ b = 1 in (22). Then, by (29), we have

<!-- formula-not-decoded -->

where (a) follows from the definition of ¯ K , and (b) follows from L g ( ¯ b -C b ) 2 ≤ 1 .

If L g ( ¯ b -C b ) 2 &gt; 1 , it holds that ˜ b = L g ( ¯ b -C b ) 2 in (22), and we have

<!-- formula-not-decoded -->

Above all, we conclude that after at most ¯ K iterations, the condition ∥G y g ( x t , y ¯ t K ) ∥ 2 y ¯ t K ≤ ϵ y is satisfied, i.e., K t ≤ ¯ K holds for all t ≥ 0 . We complete the proof.

AdaRHD-GD: We show that N t ≤ ¯ N gd for all 0 ≤ t ≤ T .

If n 1 in Proposition G.1 does not exist , we have c N t ≤ C c . Similar to K t of the lower-level problem, we have N t ≤ log( C 2 c /c 2 0 ) log(1+ ϵ v /C 2 c ) . If N t &gt; log( C 2 c /c 2 0 ) log(1+ ϵ v /C 2 c ) , by ∥∇ v R ( x t , y K t t , v n t ) ∥ 2 y Kt t ≥ ϵ v and c n ≤ C c hold for all n &lt; N t , we have

<!-- formula-not-decoded -->

which contradicts c N t ≤ C c .

If n 1 in Proposition G.1 exists and N t ≥ n 1 . Then, it holds that c n 1 ≤ C c and c n 1 +1 &gt; C c . Similarly, we have n 1 ≤ log( C 2 c /c 2 0 ) log(1+ ϵ v /C 2 c ) .

By the update rule of v n 1 t and the definition of ˆ v ∗ ( x t , y K t t ) = arg min v ∈ T y K t t M y R ( x t , y K t t , v ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (a) follows from Baillon-Haddad Theorem [4, Theorem 5.8 (iv)], i.e., for y ∈ M y and v ∈ T y M y , it holds that

<!-- formula-not-decoded -->

(b) follows from the warm start of v 0 t , (c) uses Lemma F.1, and (d) follows from Lemmas F.3 (2) and F.4, and c n 1 ≤ C c .

Then, for all N &gt; n 1 , we have

<!-- formula-not-decoded -->

where (a) follows from (33), (b) follows from c n &gt; C c ≥ L g , (c) follows from ∇ v R ( x t , y K t t , ˆ v ∗ ( x t , y K t t )) = 0 and the µ -strong convexity of R , (d) follows from c n ≥ C c ≥ L g ≥ µ and 1 -x ≤ e -x for 0 &lt; x &lt; 1 , and (e) follows from (32).

Similar to (27), we have

<!-- formula-not-decoded -->

For the second term of (34), we note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the first inequality follows from (33), and the second inequality follows from c N ≥ C c ≥ L g . Then, by (32), it holds that

<!-- formula-not-decoded -->

Combining (34) derives

<!-- formula-not-decoded -->

Denote ¯ c := C c + L g ( 2 ϵ y µ 2 + 8 l 2 f µ 2 +2log ( C c c 0 ) +1 ) . Then, considering the monotonicity of µ c w.r.t. c , we have

<!-- formula-not-decoded -->

Since n 1 ≤ log( C 2 c /c 2 0 ) log(1+ ϵ v /C 2 c ) , by the definition of ¯ N gd in (23), it holds that ¯ N gd &gt; n 1 . Then, similar to (30) and (31), we have

<!-- formula-not-decoded -->

where the last inequality follows from the definition of ¯ N gd . This indicates that, after at most ¯ N gd iterations, the condition ∥∇ v R ( x t , y K t t , v ¯ N gd t ) ∥ 2 y Kt t ≤ ϵ v is satisfied, i.e, N t ≤ ¯ N gd holds for all t ≥ 0 .

AdaRHD-CG: We show that N t ≤ ¯ N cg for all 0 ≤ t ≤ T .

Denote κ g = L g µ . From [10, Equation (6.19)], given x t ∈ M x and y N t t ∈ M y , we have

<!-- formula-not-decoded -->

where the last inequality we use the setting of that ˆ v 0 t = 0 and Lemma F.3 (2). Then we have

<!-- formula-not-decoded -->

which implies that, after at most ¯ N cg iterations, it holds that ∥∇ v R ( x t , y K t t , v N t t ) ∥ 2 y Kt t ≤ ϵ v , where

<!-- formula-not-decoded -->

We complete the proof.

## G.3 Proof of Theorem 3.1

Proof. If t 1 in Proposition G.1 does not exist , we have a T ≤ C a . Then, by (11) in Lemma G.1, for t &lt; T , we have

<!-- formula-not-decoded -->

where ϵ y,v is defined in Lemma G.1. Summing it from t = 0 to T -1 , we have

<!-- formula-not-decoded -->

where the second inequality follows from ∑ T -1 t =0 ∥ ̂ G F ( x t , y K t t , v N t t ) ∥ 2 x t ≤ a 2 T ≤ C 2 a , and F 0 is defined in (14).

If t 1 in Proposition G.1 exists , for any t &lt; t 1 , by (11) in Lemma G.1, we have

<!-- formula-not-decoded -->

For any t ≥ t 1 , by (12) in Lemma G.1, we have

<!-- formula-not-decoded -->

Summing (36) and (37) from 0 to T -1 , we have

<!-- formula-not-decoded -->

where the last inequality follows from Assumption 3.4 and a t 1 ≤ C a , and F 0 is defined in Lemma G.2. This result is equivalent to (35).

Then, since a t +1 ≤ a T , by Lemma G.2, we have

<!-- formula-not-decoded -->

Since ϵ y = 1 /T and ϵ v = 1 /T , by the definition of ϵ y,v := ¯ L 2 µ 2 ( ϵ y + ϵ v ) in Lemma G.1, we have ϵ y,v = 2 ¯ L 2 Tµ 2 . Then, by the definition of F 0 , (38) is equivalent to

<!-- formula-not-decoded -->

## G.4 Proof of Corollary 3.1

Proof. By Theorem 3.1, we have

<!-- formula-not-decoded -->

For the resolution of the lower-level problem, we have

<!-- formula-not-decoded -->

Similarly, for the resolution of the linear system, we have

AdaRHD-GD:

<!-- formula-not-decoded -->

AdaRHD-CG:

<!-- formula-not-decoded -->

Then, it is evident that the gradient complexities of f and g are G f = O (1 /ϵ ) and G g = O (1 /ϵ 2 ) , respectively. The complexities of computing the second-order cross derivative and Hessian-vector product of g are JV g = O (1 /ϵ ) , HV g = O (1 /ϵ 2 ) for AdaRHD-GD, and HV g = ˜ O (1 /ϵ ) for AdaRHD-CG, respectively.

## H Proofs for Section 3.4

This section provides the proofs of the results from Section 3.4. For consistency, we retain the notations introduced in Section 3.3, such as C a , C b , C c , etc.

Firstly, similar to Proposition G.1, we first give a lemma that concerns the step sizes a 1 , b t , and c t .

Proposition H.1. Suppose that Assumptions 3.1, 3.2, 3.3, 3.5, and 3.6 hold. Denote { T, K, N } as the iterations of { x, y, v } . Given any constants C a ≥ a 0 , C b ≥ b 0 , C c ≥ c 0 , then, we have AdaRHD-GD:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We then consider the case where the conjugate gradient solves the linear system. Therefore, we do not need to consider the step size c t .

AdaRHD-CG:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## H.1 Proof of Proposition D.1

## H.1.1 Improvement on objective function for one step update

Similar to Lemma G.1, we have the following lemma.

Lemma H.1. Suppose that Assumptions 3.1, 3.2, 3.3, 3.5, and 3.6 hold. Then, we have

<!-- formula-not-decoded -->

Furthermore, if t 1 in Proposition H.1 exists, then for t ≥ t 1 , we have

<!-- formula-not-decoded -->

where ¯ L F = ( L F c u +2 c R ( l f + l f L g /µ ) , ϵ y,v = ¯ L 2 µ 2 ( ϵ y + ϵ v ) , ¯ L := max { √ 2( L g (1 + L f µ + l f ρ µ 2 ) + l f ρ µ ) , √ 2 L g } , and c u , c R are defined in Assumption 3.6.

Proof. By Lemma F.2, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality uses Lemma F.2 (2) and Assumption 3.6.

If t 1 in Proposition H.1 exists , then for t ≥ t 1 , we have a t +1 &gt; C a ≥ 2( L F c u +2 c R ( l f + l f L g /µ )) . The desired result of (39) follows from (40). The proof is complete.

Similar to (10), we define a threshold for parameters a t when giving an upper bound of the step size a t .

<!-- formula-not-decoded -->

where ¯ L F = ( L F c u +2 c R ( l f + l f L g /µ ) is defined in Lemmas H.1.

Similar to Lemma G.2, we have the following upper bound for the step size a t .

Lemma H.2. Suppose that Assumptions 3.1, 3.2, 3.3, 3.4, 3.5, and 3.6 hold. If t 1 in Proposition H.1 does not exist, we have a t ≤ C a for any t ≤ T .

If there exists t 1 ≤ T described in Proposition H.1, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we define

The proof of Lemma H.2 closely parallels that of Lemma G.2, requiring only the substitution of L F with ¯ L F . Here we omit it.

Before proving Proposition D.1, similar to Lemma G.3, we present the following technical lemma when substituting the exponential mapping with the retraction mapping.

Lemma H.3. Suppose that Assumptions 3.1, 3.2, 3.3, 3.5, and 3.6 hold. Then, we have

<!-- formula-not-decoded -->

where ¯ ζ := ζc u +2 ¯ Dc R .

Proof. From Lemma B.1, by the definition that Retr -1 y k t y k +1 t = -1 b k +1 G g ( x t , y k t ) we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality follows from Assumption 3.6 and

<!-- formula-not-decoded -->

the third inequality follows from Assumption 3.5 by employing [10, Proposition 10.22], i.e.,

<!-- formula-not-decoded -->

and the fourth inequality follows from Assumption 3.6.

Similar to (18), we define the following thresholds for the step sizes b k and c n .

<!-- formula-not-decoded -->

where ¯ ζ = ζc u +2 ¯ Dc R are defined in Lemma H.3.

## Proof of Proposition D.1

Proof. For AdaRHD-GD, denote and

<!-- formula-not-decoded -->

where ˜ b := max { L g ( ¯ b -C b ) 2 , 1 } , ˜ c := max { L g (¯ c -C c ) , 1 } , ¯ b := C b + 2 L g ( 2 ϵ y µ 2 + 2 L 2 g C 2 f µ 2 a 2 0 +2 ¯ ζ log ( C b b 0 ) + ¯ ζ ) with C f = ( 2 L 2 g ϵ v µ 2 + 4 L 2 g l 2 f µ 2 +4 l 2 f ) 1 2 is defined in Lemma F.5, and ¯ c := C c + L g ( 2 ϵ y µ 2 + 8 l 2 f µ 2 +2log ( C c c 0 ) +1 ) .

For AdaRHD-CG, denote

<!-- formula-not-decoded -->

First, we show that K t ≤ ¯ K for all 0 ≤ t ≤ T .

If k 1 in Proposition H.1 does not exist , we have b K t ≤ C b . By [92, Lemma 2], it holds that K t ≤ log( C 2 b /b 2 0 ) log(1+ ϵ y /C 2 b ) . If K t &gt; log( C 2 b /b 2 0 ) log(1+ ϵ y /C 2 b ) , by ∥G y g ( x t , y p t ) ∥ 2 x t ≥ ϵ y and b k ≤ C b holds for all k &lt; K t , we have

<!-- formula-not-decoded -->

This contradicts b K t ≤ C b .

If k 1 in Proposition H.1 exists , we have b k 1 ≤ C b and b k 1 +1 &gt; C b . Wefirst prove k 1 ≤ log( C 2 b /b 2 0 ) log(1+ ϵ y /C 2 b ) . If k 1 &gt; log( C 2 b /b 2 0 ) log(1+ ϵ y /C 2 ) , similar to (44), we have

<!-- formula-not-decoded -->

which contradicts the setting that b k 1 ≤ C b .

By Lemma H.3, we have

<!-- formula-not-decoded -->

where (a) follows from the second inequality of(42), (b) follows from Assumption 3.5, (c) follows from Assumption 3.6, (d) follows from the convexity of g , (e) follows from the warm start of y 0 t , (f) follows from Lemma F.1, (g) follows from Lemmas F.2 (1) and F.4, and (h) follows from Lemma F.5 and b k 1 ≤ C b .

For all K &gt; k 1 , by Lemma H.3, we have

<!-- formula-not-decoded -->

b

where the first inequality follows from b k ≥ C b ≥ 2 ¯ ζL g L g µ and 1 -m ≤ e -m for 0 &lt; m &lt; 1 , specifically, we have C b ≥ 2 ¯ ζL g L g µ and it holds that 0 &lt; µ b k -¯ ζ L 2 g b 2 k &lt; 1 , and the second inequality follows from (45).

Similar to (27), we have

<!-- formula-not-decoded -->

Similar to (28), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality follows from Lemma G.4 and b K ≥ C b ≥ 2 ¯ ζL g L g µ . By (45), we have

<!-- formula-not-decoded -->

Then, by (47) and 1 1 -µ/ 2 L g ≤ 2 , we have

<!-- formula-not-decoded -->

Denote ¯ b := C b + 2 L g ( 2 ϵ y µ 2 + 2 L 2 g C 2 f µ 2 a 2 0 +2 ¯ ζ log ( C b b 0 ) + ¯ ζ ) . Then, under (46), considering the monotonicity of µ b -¯ ζ L 2 g b 2 w.r.t. b when b ≥ 2 ¯ ζL g L g µ , it holds that µ ¯ b -¯ ζ L 2 g ¯ b 2 ≤ µ b K -¯ ζ L 2 g b 2 K as ¯ b ≥ b K ≥ 2 ¯ ζL g L g µ , we have

<!-- formula-not-decoded -->

Since k 1 ≤ log( C 2 b /b 2 0 ) log(1+ ϵ y /C 2 b ) , by the definition of ¯ K in (43), it is obvious that ¯ K ≥ k 1 . Therefore, similar to (30) and (31), by (48), we have

<!-- formula-not-decoded -->

which indicates that after at most ¯ K iterations, the condition ∥G y g ( x t , y ¯ t K ) ∥ 2 y ¯ t K ≤ ϵ y is satisfied, i.e., K t ≤ ¯ K holds for all t ≥ 0 .

Regarding the total number of iterations N t required to solve the linear system, since solving it does not involve retraction mapping, the total iterations match those of Algorithm 1. The proof is complete.

## H.2 Proof of Theorem 3.2

Proof. The proof of Theorem 3.2 follows a similar approach to Theorem 3.1, requiring only the substitution of L F with ¯ L F and the definition of the constant C retr as

<!-- formula-not-decoded -->

where F 0 is defined in (41) of Lemma H.2. Due to this structural similarity, the remaining proof is omitted.

## I Experimental details

This section expands on the experiments discussed in Section 4 and presents supplemental empirical evaluations. Subsection I.1 details the experimental configurations, while Subsection I.2 includes an additional implementation to further demonstrate the robustness of our proposed algorithm through additional empirical validation.

## I.1 Experimental Settings

All implementations are executed using the Geoopt framework [48], matching the reference implementation [33] to guarantee equitable comparison conditions. Furthermore, all the experiments are implemented based on Geoopt [48] and are implemented using Python 3.8 on a Linux server with 256GB RAM and 96-core AMD EPYC 7402 2.8GHz CPU.

## I.1.1 Simple problem

For n = 100 , the maximum number of outer iterations in RHGD [33] is set to 200 , whereas for n = 1000 , this number is increased to 400 . In Algorithm 3, the number of outer iterations is set to T = 1000 for AdaRHD-GD and T = 10000 for AdaRHD-CG. Following [33], we set λ = 0 . 01 and fix the step sizes in RHGD as η x = η y = 0 . 5 . To ensure consistency, the initial step sizes in Algorithm 3 are set to a 0 = b 0 = c 0 = 2 .

For solving the linear system, we adopt the conjugate gradient (CG) method as the default approach in RHGD. For our algorithm, the CG procedure is terminated when the residual norm falls below a tolerance of 10 -10 or the number of CG iterations reaches 50 . Note that the tolerance 10 -10 for CG procedure can be seen as O ( 1 T α ) with α &gt; 1 is a constant, this will not influence the total iterations and the complexity results since the convergence rate of the CG procedure is linear. Moreover, for the GD procedure, we employ the stopping criteria specified in Algorithm 3 or terminate when the number of iterations reaches min { 50 ⌊ t/ 5 ⌋ , 500 } , where ⌊·⌋ denotes the floor function, and t represents the iteration round t . This warm-start-inspired strategy aims to rapidly obtain a reasonable solution during early iterations, and progressively increase internal iterations to satisfy the gradient norm tolerance ϵ v in later stages.

For solving the lower-level problem, our algorithm follows the stopping criteria outlined in Algorithm 1 or terminates when the number of iterations reaches min { 50 ⌊ t/ 5 ⌋ , 500 } . For RHGD, the maximum number of iterations is set to either 50 or 20 .

Moreover, the number of outer iterations in our algorithm is set to T = 1000 for AdaRHD-CG and T = 10000 for AdaRHD-GD. Additionally, our algorithm terminates if the approximate hypergradient norm falls below 10 -4 . For RHGD, the maximum number of outer iterations is set to 200 .

## I.1.2 Robustness analysis

The experimental configurations in Subsection 4.2 are maintained consistently across all comparative analyses in Subsection 4.1, except that the number of outer iterations is set to be T = 300 for AdaRHD-CG and T = 500 for AdaRHD-GD.

## I.2 Additional experiments: hyper-representation problem

## I.2.1 Shallow hyper-representation for regression

In this subsection, we adopt a shallow regression framework from [33], incorporating a Stiefel manifold constraint [40, 34, 36] to preserve positive-definiteness in learned representations. The SPD representation is transformed into Euclidean space using the matrix logarithm, thereby establishing a bijection between SPD and symmetric matrices, followed by vectorizing the upper-triangular entries via the vec( · ) operation. We maintain the problem framework established in Subsection 4.2, replacing the cross-entropy loss with the least-squares loss. The revised problem is as follows:

<!-- formula-not-decoded -->

where λ &gt; 0 is the regular parameter. Following [33], we set d = 50 , r = 10 , and λ = 0 . 1 .

In Problem (49), the upper-level objective is optimized on the validation set, whereas the lowerlevel problem is solved on the training set. Following [33], each element y i is defined by y i = vec(log( A ⊤ D i A )) β + ϵ i , where A and D i are randomly generated matrices, β is a randomly generated vector, and ϵ i represents a Gaussian noise term. We generate n samples of D i , evenly divided between the validation and training sets. To align with the step size initialization in [33], we initialize parameters a 0 , b 0 , and c 0 in our algorithm to 20. Moreover, we configure the number of outer iterations as T = 1000 for AdaRHD-CG and T = 10000 for AdaRHD-GD. All other settings align with Section 4.1.

Figure 4: Shallow hyper-representation for regression (Left two column: n = 200 , Right two column: n = 1000 ).

<!-- image -->

In Figure 4, we illustrate the validation set loss (Upper Objective) versus time for n = 200 and n = 1000 , respectively. We observe that AdaRHD-CG converges more effectively and rapidly to a smaller objective loss compared to AdaRHD-GD and RHGD. Furthermore, note that although AdaRHD-GD initially converges more slowly than RHGD (potentially due to imprecise hypergradient estimation affecting the upper-level loss reduction), Figure 4 demonstrates that AdaRHD-GD achieves a lower objective loss than RHGD at specific time intervals. These results collectively highlight the efficiency and robustness of our algorithm, as demonstrated in Section 4.1.

## I.2.2 Deep hyper-representation for classification

In this subsection, we further evaluate the efficiency and robustness of our proposed algorithm by extending the deep hyper-representation classification framework introduced in Subsection 4.2 to dataset sampling ratios of 12 . 5% and 25% . For each configuration, five independent trials are executed using distinct random seeds. In each trial, a randomly sampled validation set is reserved for the upper-level problem, with an equally sized training partition allocated to the lower-level task. All remaining experimental parameters align with those defined in Subsection 4.2.

Figure 5 depicts the validation accuracy against outer epochs (iterations) with 12 . 5% and 25% dataset sampling ratios (due to computational constraints, a subset of the full dataset is utilized to ensure manageable training durations). Table 3 summarizes the computational time required to achieve specified validation accuracy thresholds across algorithms. For each random seed, an independent data subset is sampled. Our algorithm exhibits rapid validation accuracy improvements during initial outer iterations, followed by more gradual advancement in later stages. This behavior likely originates from the inner-loop strategy: early iterations necessitate more intensive inner-loop computations to attain high precision, whereas later stages leverage accumulated progress, thereby requiring fewer

Figure 5: Deep hyper-representation for classification (Left: sampling ratio 12.5%, Right: sampling ratio 25%).

<!-- image -->

inner-loop iterations. In contrast, RHGD-20 inner iterations demonstrate rapid initial accuracy gains but experience pronounced degradation subsequently, particularly at the 25% sampling ratio, where its accuracy drops below that of our method. This instability is inherent in RHGD when fewer inner iterations are employed.

Furthermore, while RHGD-20 exhibits faster time-to50% -accuracy, its elevated variance (evident in Table 3) confirms inherent instability. Notably, our method achieves target accuracy faster than RHGD despite requiring additional outer iterations. This advantage, maintained under identical initial step sizes, demonstrates both the efficacy and stability of our adaptive approach in addressing Riemannian bilevel optimization problems.

Table 3: Time required for each algorithm to first achieve a specified validation accuracy. 'Time(s) to X % ' is defined as the time elapsed until the validation accuracy first reaches X % . Values are presented as mean ± standard deviation across five independent trials using fixed random seeds for all algorithms. Statistically significant results are marked in bold. 'Data ratio' refers to the proportion of the full dataset.

|                      | AdaRHD-CG        | AdaRHD-GD            | RHGD-50              | RHGD-20              |
|----------------------|------------------|----------------------|----------------------|----------------------|
| data ratio = 12 . 5% |                  |                      |                      |                      |
| Time(s) to 50%       | 878 . 60 ± 90.44 | 1258 . 83 ± 138 . 54 | 1188 . 66 ± 148 . 46 | 530.03 ± 306.29      |
| Time(s) to 70%       | 1666.17 ± 96.74  | 1865 . 13 ± 426 . 80 | 3070 . 22 ± 169 . 30 | 2085 . 61 ± 236 . 33 |
| data ratio = 25%     | data ratio = 25% | data ratio = 25%     | data ratio = 25%     | data ratio = 25%     |
| Time(s) to 50%       | 2511.13 ± 233.16 | 3408 . 80 ± 508 . 51 | 3944 . 26 ± 589 . 55 | 2598 . 87 ± 645 . 08 |

## I.2.3 Robust optimization on manifolds

In this subsection, following [54, Section 7.1], we consider the robust optimization on manifolds, which have the following forms:

<!-- formula-not-decoded -->

where ∆ n := { p ∈ R n : ∑ n i =1 p i = 1 , p i &gt; 0 } and S d ++ := { y ∈ R d × d : y ≻ 0 } represent the multinomial manifold (or probability simplex) and positive definite matrix space, respectively. For robust Karcher mean (KM) problem, ℓ ( y ; ξ i ) in Problem (50) represents the geodesic distance of two positive definite matrices [7]:

<!-- formula-not-decoded -->

where ξ i 's are the symmetric positive definite data matrices. For robust maximum likelihood estimation (MLE) problem, ℓ ( y ; ξ i ) in Problem (50) represents the log likelihood of the Gaussian distribution [75]:

<!-- formula-not-decoded -->

where ξ i 's are sample vectors from the Gaussian distribution.

In this experiment, we set the step sizes as η x = η y = 1 /a 0 = 1 /b 0 = 1 /c 0 = 0 . 1 , with all other settings identical to those in Section 4.2. For the robust Karcher mean problem, we set d = 20 and n ∈ { 5 , 10 , 20 , 50 } , randomly generating the symmetric positive definite data matrices ξ i 's. For the

robust maximum likelihood estimation problem, we set d = 50 and n ∈ { 100 , 300 , 500 , 1000 } , and randomly generate the sample vectors from a Gaussian distribution with mean 0 and covariance matrix being a random positive semidefinite symmetric matrix. We use five different random seeds to conduct experiments for each group. The results are shown in Figure 6, where the x-axis represents the average time, and the y-axis represents the average value of the ergodic performance min i ∈ [0 ,t ] ∥ ̂ G F ( x i , y K i i , v N i i ) ∥ 2 x i . Note that for the MLE experiment with n = 100 , we do not report the results of RHGD-20 because it fails to converge in this case. As shown in Figure 6, compared to RHGD, AdaRHD achieves superior results more efficiently and robustly, which further demonstrates the superiority and robustness of our algorithm.

Figure 6: Robust optimization on manifolds (Top row: KM model with d = 20 and different n , Bottom row: MLE model with d = 50 and different n ).

<!-- image -->

## J Broader impacts

This paper proposes an algorithm for solving Riemannian bilevel optimization problems. No negative societal impacts are anticipated from this research that warrant disclosure.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: For the paper's contributions and scope, please refer to Sections 3 and 4, which match the main claims made in the abstract and introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitations and future directions of our work, please refer to Section 5.

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

## Answer: [Yes]

Justification: We provide the full set of assumptions and a complete (and correct) proof for each lemma, proposition, theorem, and corollary. The formal statements of these results are provided in Section 3, while detailed proofs are documented in Appendices F, G, and H.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [Yes]

Justification: We disclose all the information needed to reproduce the main experimental results of the paper, please refer to Section 4 and Appendix I, along with the supplemental material.

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

Justification: We provide open access to all data and code in the supplementary materials. Furthermore, we include detailed instructions to replicate the experimental results presented in Section 4 and Appendix I, along with the corresponding codes in the supplementary materials.

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

Justification: We specify all the training and test details in Section 4 and Appendix I.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The values in Table 2 and Figure 3 of Section 4.2, as well as Figure 5, Table 3, and Figure 6 of Appendix I.2.2, are expressed as mean ± standard deviation across five independent trials conducted with fixed random seeds for each algorithm.

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

Justification: We provide sufficient information on the computer resources for all experiments, please refer to Section I.1.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The paper conforms, in every respect, with the NeurIPS Code of Ethics, we also discuss the broader impacts and technical limitations of this paper in Section 5 and Appendix J.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the broader impacts in Appendix J.

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

Justification: We do not release any data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets).

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: In Section 4 and Appendix I, along with the supplementary materials, the creators and original owners of assets (e.g., code, data, models) are properly cited.

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

Justification: We do not introduce new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: We do not introduce any human subject.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: We do not introduce any human subject.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The LLMs are not important, original, or non-standard components of the core methods in this research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.