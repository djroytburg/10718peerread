## Escaping saddle points without Lipschitz smoothness: the power of nonlinear preconditioning

## Alexander Bodard

ESAT-STADIUS &amp; Leuven.AI KU Leuven

alexander.bodard@kuleuven.be

## Panagiotis Patrinos

ESAT-STADIUS &amp; Leuven.AI

KU Leuven

panos.patrinos@esat.kuleuven.be

## Abstract

Westudy generalized smoothness in nonconvex optimization, focusing on ( L 0 , L 1 ) -smoothness and anisotropic smoothness. The former was empirically derived from practical neural network training examples, while the latter arises naturally in the analysis of nonlinearly preconditioned gradient methods. We introduce a new sufficient condition that encompasses both notions, reveals their close connection, and holds in key applications such as phase retrieval and matrix factorization. Leveraging tools from dynamical systems theory, we then show that nonlinear preconditioning - including gradient clipping - preserves the saddle point avoidance property of classical gradient descent. Crucially, the assumptions required for this analysis are actually satisfied in these applications, unlike in classical results that rely on restrictive Lipschitz smoothness conditions. We further analyze a perturbed variant that efficiently attains second-order stationarity with only logarithmic dependence on dimension, matching similar guarantees of classical gradient methods.

## 1 Introduction

We consider the unconstrained optimization problem

<!-- formula-not-decoded -->

where f : R n → R is a twice continuously differentiable nonconvex function. This work studies the nonlinearly preconditioned gradient method , with iterates described by

<!-- formula-not-decoded -->

where ϕ : R n → R ∪ {∞} is referred to as the reference function , and ϕ ∗ , its convex conjugate, is called the dual reference function .

Nonlinear preconditioning provides a flexible framework for constructing and analyzing gradientbased optimization algorithms [36, 21, 31]. For instance, when ϕ ( x ) = 1 2 ∥ x ∥ 2 , the update (P-GD) reduces to classical gradient descent. More broadly, we focus on isotropic reference functions of the form ϕ ( x ) = h ( ∥ x ∥ ) for some scalar kernel function h : R → R + ∪ {∞} , though our results extend in part to more general settings, including separable reference functions ϕ ( x ) = ∑ n i =1 h ( x i ) . Some kernel functions of interest include:

<!-- formula-not-decoded -->

each of which upper bounds the quadratic function x 2 / 2 , as visualized in Fig. 1. These choices induce preconditioners that closely resemble common gradient clipping heuristics, as shown in Fig. 1.

The effectiveness of gradient clipping has been justified using the concept of ( L 0 , L 1 ) -smoothness, which is empirically motivated by practical neural network training scenarios [42]. However, it

<!-- image -->

|

|

Figure 1: Comparison of (a) kernel functions and (b) their corresponding nonlinear preconditioners.

remains unclear under what precise conditions this smoothness assumption holds in real-world applications. On the other hand, the preconditioned gradient method is naturally analyzed under anisotropic smoothness [36], another generalization of the classical Lipschitz smoothness condition. Rather than imposing a global quadratic upper bound, anisotropic smoothness permits more flexible upper bounds defined in terms of the reference function ϕ . This makes the preconditioned gradient method particularly attractive in settings where the standard Lipschitz condition is too restrictive. This leads us to our first central question:

Can we formally establish anisotropic smoothness and ( L 0 , L 1 ) -smoothness of practical problems where traditional assumptions fail?

Our second line of inquiry focuses on the behavior of the preconditioned gradient method when applied to nonconvex objectives. Classical gradient descent is known to avoid strict saddle points under the assumption of (global) Lipschitz smoothness [24], a phenomenon which helps explain its strong empirical performance in nonconvex settings. However, for many practical applications Lipschitz smoothness holds only locally or on compact sets around a minimizer, meaning that this assumption is not truly satisfied. This raises the following question:

Does nonlinear preconditioning preserve the saddle point avoidance properties of gradient descent under a possibly less stringent smoothness assumption?

Our results reveal novel connections between different generalizations of smoothness and provide strong theoretical support for nonlinear preconditioning, particularly in nonconvex settings where the classical Lipschitz smoothness assumption may fail.

Contributions Our contributions can be summarized as follows.

- We investigate the classes of problems for which ( L 0 , L 1 ) -smoothness and anisotropic smoothness - two generalizations of the classical Lipschitz smoothness condition - are applicable. To this end, we propose a novel sufficient condition (Assumption 2.8) that guarantees both anisotropic and ( L 0 , L 1 ) -smoothness, thereby revealing a structural link between these two frameworks. We further demonstrate in section 2.3 that this condition holds for several prominent nonconvex problems, including phase retrieval, low-rank matrix factorization, and Burer-Monteiro factorizations of MaxCut-type problems.
- We establish that nonlinear preconditioning preserves the saddle point avoidance behavior of gradient descent, and moreover extends results from the classical Lipschitz smoothness framework to the broader setting of anisotropic smoothness. Specifically, we prove asymptotic avoidance of strict saddle points by leveraging the stable-center manifold theorem. By invoking a recent nonsmooth generalization of this theorem, this analysis is then further extended to accommodate hard gradient clipping. Finally, we present a complexity analysis for a perturbed variant of the preconditioned gradient method, showing that it converges to a second-order stationary point with only logarithmic dependence on the problem dimension.

Notation Let S n × n be the set of symmetric n × n matrices. We denote the standard Euclidean inner product on R n by ⟨· , · , ⟩ , and the corresponding norm by ∥ · ∥ . For X,Y ∈ R m × n , ⟨ X,Y ⟩ = trace( X ⊤ Y ) is the standard inner product on R m × n and ∥ · ∥ denotes the spectral norm. The class of k times continuously differentiable functions on an open set O ⊆ R n is denoted by C k ( O ) . We write sgn( x ) = x / ∥ x ∥ for x ∈ R n \ { 0 } and 0 otherwise. A function f ∈ C 2 ( R n ) is L -Lipschitz smooth if for all x, y ∈ R n it holds that ∥∇ f ( x ) -∇ f ( y ) ∥ ≤ L ∥ x -y ∥ , with L ≥ 0 , and ( L 0 , L 1 ) -smooth if ∥∇ 2 f ( x ) ∥ ≤ L 0 + L 1 ∥∇ f ( x ) ∥ for all x ∈ R n with L 0 , L 1 ≥ 0 . Otherwise, we follow [37].

## 1.1 Related work

Generalized smoothness Gradient descent is traditionally analyzed under the assumption of Lipschitz smoothness [34], although many applications violate this condition. Bregman relative smoothness is a popular extension which allows the Hessian to grow unbounded, see e.g. [30] which assumes a certain polynomial growth. More recently, the ( L 0 , L 1 ) -smoothness condition was proposed by Zhang et al. [42], based on empirical observations in LSTMs, and used to analyze clipped gradient descent and a momentum variant [41]. The framework has since been applied to stochastic normalized gradient descent [43] and generalized SignSGD [12]. Notably, Crawshaw et al. [12] provided empirical evidence that ( L 0 , L 1 ) -smoothness holds for Transformers [40], albeit with layer-wise variation in constants. Further generalizations include α -symmetric smoothness [9] and ℓ -smoothness [28], and the latter was used to analyze the convergence of Adam [29]. Despite empirical support for these conditions in key applications, theoretical guarantees remain limited.

Nonlinear preconditioning The preconditioned gradient method with updates given by (P-GD) was introduced in the convex setting by Maddison et al. [31]. Then, Laude et al. [22, 21] studied L -anisotropic smoothness and, under this condition, showed convergence of (P-GD) for nonconvex problems. The method was later extended to measure spaces [4]. Oikonomidis et al. [36] proposed the ( L, ¯ L ) -anisotropic smoothness condition, connected it to ( L 0 , L 1 ) -smoothness, and analyzed convergence of (P-GD) in both convex and nonconvex settings. We also highlight the works [26, 35] that study the concept of Φ -convexity, which is closely related to anisotropic smoothness.

Saddle point avoidance To explain the success of gradient descent on nonconvex problems, much work has focused on its (strict) saddle point avoidance properties [25, 24]. It was shown that gradient descent may take exponential time to escape saddle points, even with random initialization [13]. The works [27, 33] showed that noise-injected normalized gradient descent escapes them more efficiently. Jin et al. [17, 18] demonstrated that perturbed gradient descent escapes saddle points in time polylogarithmic in the problem dimension. Recently, Cao et al. [8] studied saddle point avoidance under a second-order self-bounding regularity condition rather than under classical Lipschitz smoothness.

## 2 Anisotropic smoothness

## 2.1 Definition and basic properties

This section introduces ( L, ¯ L ) -anisotropic smoothness as proposed by [36]. The following assumption, which guarantees in particular that ϕ ∗ ∈ C 1 ( R n ) and ϕ ≥ 0 , is considered valid throughout.

Assumption 2.1. The function ϕ : R n → R is proper, lsc, strongly convex and even with ϕ (0) = 0 .

We usually also assume the following condition, which ensures in particular that ϕ ∗ ∈ C 2 ( R n ) .

Assumption 2.2. int dom ϕ = ∅ ; ϕ ∈ C 2 (int dom ϕ ) , and for any sequence { x k } k ∈ N that converges to some boundary point of int dom ϕ , it follows that ∥∇ ϕ ( x k ) ∥ → ∞ .

̸

We follow the definition of anisotropic smoothness by [36], which reduces to [21, Def. 3.1] with reference function ¯ Lϕ if dom ϕ = R n . If f ∈ C 1 , this concept corresponds to a global version of anisotropic prox-regularity of -f [20, Def. 2.13]. For a geometric intuition, we refer to [26, 35, 36].

Definition 2.3 ( ( L, ¯ L ) -anisotropic smoothness [36]) . A function f : R n → R is ( L, ¯ L ) -anisotropically smooth relative to a reference function ϕ with constants L, ¯ L &gt; 0 if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following proposition provides a sufficient condition for anisotropic smoothness. We consider the case ϕ ∗ ∈ C 2 for simplicity of exposition, but note that a variant for ϕ ∗ / ∈ C 2 can also be formulated.

<!-- formula-not-decoded -->

Proposition 2.4 (Second-order characterization of ( L, ¯ L ) -anisotropic smoothness) . Suppose that Assumption 2.2 holds, and let f ∈ C 2 be such that for all x ∈ R n and lim ∥ x ∥→∞ ∥ T L -1 , ¯ L -1 ( x ) ∥ = ∞ . Moreover, assume that either dom ϕ is bounded or that dom ϕ = R n , and that for all x ∈ R n we have f ( x ) ≤ ¯ Lr -1 ϕ ( rx ) -β for some r ∈ (0 , L ) , b ∈ R . Then, f is ( δL, ¯ L ) -anisotropically smooth relative to ϕ for any δ &gt; 1 .

We say that f satisfies the second-order characterization of anisotropic smoothness if (3) holds. Note that the growth condition on f is not restrictive when ϕ = dom R n , and that the coercivity assumption on the iteration map T L -1 , ¯ L -1 is very mild; we refer the reader to the arguments in [36]. Finally, we connect anisotropic smoothness to some popular smoothness notions.

Example 2.5 (Lipschitz-smoothness [36, Proposition 2.3]) . Suppose that f ∈ C 2 is L f -Lipschitz smooth. Denote by µ &gt; 0 the parameter of strong convexity of a reference function ϕ . Then f is ( L f / µ , 1) -anisotropically smooth relative to ϕ .

Example 2.6 ( ( L 0 , L 1 ) -smoothness) . Let f ∈ C 2 be ( L 0 , L 1 ) -smooth, let L = L 1 , ¯ L = L 0 / L 1 , and let ϕ ( x ) = -∥ x ∥ -ln(1 - ∥ x ∥ ) . Then f satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness relative to ϕ [36, Proposition 2.6 &amp; Corollary 2.7].

## 2.2 A novel sufficient condition for generalized smoothness

Although it is well-known that univariate polynomials are ( L 0 , L 1 ) -smooth [42, Lemma 2], this is not necessarily the case for multivariate polynomials, as illustrated by the following example.

Example 2.7. Consider the polynomial f ( x, y ) = 1 4 x 4 + 1 4 y 4 -1 2 x 2 y 2 with gradient and Hessian

Remark that ∇ f ( x, -x ) = 0 and ∇ 2 f ( x, -x ) = x 2 ( 2 -2 -2 2 ) . Clearly, f cannot be ( L 0 , L 1 ) -smooth since ∥∇ 2 f ( x, -x ) ∥ F = 4 ∥ x ∥ 2 grows unbounded, while ∥∇ f ( x, -x ) ∥ = 0 for all x ∈ R .

<!-- formula-not-decoded -->

For multivariate polynomials there may exist a path of ∥ x ∥ → ∞ along which the gradient norm grows slower than the Hessian norm, in which case ( L 0 , L 1 ) -smoothness cannot hold. More examples are included in appendix A.2. Based on this insight, we propose the following novel condition.

Assumption 2.8. There exists an R ∈ N such that for all x ∈ R n

Here p R ( α ) = ∑ R i =0 a i α i and q R +1 ( α ) = ∑ R +1 i =0 b i α i are polynomials of degree R and R + 1 , respectively, and in particular we assume that b R +1 &gt; 0 .

<!-- formula-not-decoded -->

Note that [30] constructs a Bregman distance inducing kernel function under a similar polynomial upper bound to the Hessian norm. Appendix A.1 verifies Assumption 2.8 for univariate polynomials. The following result states that Assumption 2.8 is a sufficient condition for ( L 0 , L 1 ) -smoothness. 1

Theorem 2.9. Suppose that Assumption 2.8 holds for f ∈ C 2 . Then, for any L 1 &gt; 0 there exists an L 0 &gt; 0 such that f is ( L 0 , L 1 ) -smooth.

Under mild conditions on the kernel function h , which appendix A.5 shows hold for all examples in (2), Assumption 2.8 also implies the second-order characterization of anisotropic smoothness. In fact, it implies the stronger condition that ∥∇ 2 ϕ ∗ ( ¯ L -1 ∇ f ( x )) ∇ 2 f ( x ) ∥ is uniformly bounded.

<!-- formula-not-decoded -->

Assumption 2.10. The reference function ϕ is isotropic, i.e., ϕ ( x ) = h ( ∥ x ∥ ) , and such that (i) h ∗′ ( y ) / y is a decreasing function on R + , (ii) lim y → + ∞ yh ∗′′ ( y ) = C 2 , for some C 2 ∈ R + , and (iii)

Theorem 2.11. Suppose that f satisfies Assumption 2.8. If ϕ satisfies Assumption 2.2 and Assumption 2.10, then for any ¯ L &gt; 0 there exists an L &gt; 0 such that f satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness relative to ϕ .

1 In fact, Theorem 2.9 still holds if Assumption 2.8 is relaxed to ∥∇ f ( x ) ∥ ≥ q R ( ∥ x ∥ ) .

## 2.3 Applications

We now establish for a number of key applications that Assumption 2.8 holds, thus proving that the objective is ( L 0 , L 1 ) -smooth and satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness. Remark that for all of these, the classical Lipschitz smoothness assumption is violated.

## 2.3.1 Phase retrieval

Consider the real-valued phase retrieval problem with objective and gradient

<!-- formula-not-decoded -->

Here, a i ∈ R n and y i ∈ R for i ∈ N [1 ,m ] are the measurement vectors and the corresponding measurements, respectively. A relaxed smoothness condition for the phase retrieval problem has been explored in [3] based on Bregman distances. The following theorem establishes that whenever the measurement vectors span R n , the objective f also satisfies our Assumption 2.8. Note that the measurement vectors can only span R n if m ≥ n . Moreover, the assumption on spanning R n is mild compared to well-studied conditions that guarantee signal recovery in the phase retrieval problem. These conditions either require randomly sampled measurement vectors with m on the order of n log n [7], or the so-called complement property [1]. The former ensures the spanning property with high probability, while the latter guarantees it deterministically.

Theorem 2.12. Consider the phase retrieval problem with objective (4) and suppose that the vectors { a i } i m =1 span R n .

- (i) For any L 1 &gt; 0 there exists L 0 &gt; 0 such that f is ( L 0 , L 1 ) -smooth.
- (ii) If ϕ satisfies Assumptions 2.2 and 2.10, then for any ¯ L &gt; 0 , there exists an L &gt; 0 such that f satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness.

## 2.3.2 Symmetric matrix factorization

Consider the symmetric matrix factorization problem with objective and gradient

<!-- formula-not-decoded -->

Here, U ∈ R n × r is the optimization variable, and Y ∈ S n × n is a given symmetric matrix. When r &lt; n , minimizing f yields a low-rank approximation of Y with rank at most r . Such low-rank matrix factorizations are fundamental in a variety of applications, most notably in principal component analysis (PCA) [19], where one seeks to capture the most significant directions of variation in the data. More broadly, symmetric matrix factorization plays a central role across various domains: in machine learning, it underlies techniques such as non-negative matrix factorization for parts-based representation learning [23]; in signal processing, it is employed in matrix completion and compressed sensing to reconstruct structured signals from incomplete or noisy measurements [6].

Theorem 2.13. Consider the symmetric matrix factorization problem with objective (5) . Then the following statements hold.

- (i) For any L 1 &gt; 0 there exists an L 0 &gt; 0 such that f is ( L 0 , L 1 ) -smooth.
- (ii) If ϕ satisfies Assumptions 2.2 and 2.10, then for any ¯ L &gt; 0 , there exists an L &gt; 0 such that f satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness.

## 2.3.3 Asymmetric matrix factorization

Consider the regularized asymmetric matrix factorization problem with objective

<!-- formula-not-decoded -->

where W ∈ R m × r and H ∈ R r × n are the optimization variables, Y ∈ R m × n is a given matrix, and κ ≥ 0 is a regularization parameter. When κ = 0 and r &lt; min { m,n } , this reduces to the classical

low-rank matrix factorization problem. Additionally, such objectives have been used to model the training of two-layer linear networks, such as in the case of two-layer autoencoders [15]. We note that the results below also hold for regularization terms of the form κ ∥ W ⊤ W -HH ⊤ ∥ 2 F as described in [11], and highlight the work of [32], which designed a Bregman proximal-gradient method for similar regularized matrix factorization problems.

Theorem 2.14. Consider the asymmetric matrix factorization problem with objective (6) and let κ &gt; 0 . Then the following statements hold.

- (i) For any L 1 &gt; 0 there exists an L 0 &gt; 0 such that f is ( L 0 , L 1 ) -smooth.
- (ii) If ϕ satisfies Assumptions 2.2 and 2.10, then for any ¯ L &gt; 0 , there exists an L &gt; 0 such that f satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness.

Note that Theorem 2.14 requires κ &gt; 0 . To understand why, observe that the gradient of f is given by ∇ W f ( W,H ) = ( WH -Y ) H ⊤ + κ ∥ W ∥ 2 F W, and ∇ H f ( W,H ) = W ⊤ ( WH -Y )+ κ ∥ H ∥ 2 F H.

Finally, we remark that the key step in proving Theorem 2.14 entails lower bounding ∥∇ f ( W,H ) ∥ in terms of the variable V := max( ∥ W ∥ F , ∥ H ∥ F ) , and exploiting that ∥ V ∥ → ∞ if and only if ∥ x ∥ → ∞ . It appears that this strategy can be generalized to the factorization of Y into more than two factors, which is relevant for training deep linear networks.

Let x denote the concatenation of the vectors vec( W ) and vec( H ) , such that ∥ x ∥ 2 = ∥ W ∥ 2 F + ∥ H ∥ 2 F . In contrast to symmetric matrix factorization, if κ = 0 , the gradient norm of f can approach zero as ∥ x ∥ → ∞ , whereas Assumption 2.8 requires an asymptotic growth proportional to ∥ x ∥ 3 . To see this, consider W ⋆ ∈ R m × r , H ⋆ ∈ R r × n such that W ⋆ H ⋆ = Y . In this case, the gradient norm is zero, and rescaling W ⋆ and H ⋆ with a nonsingular matrix D ∈ R r × r , i.e., ˜ W = W ⋆ D and ˜ H = D -1 H ⋆ , preserves the gradient norm. As a result, one can construct counterexamples where the gradient norm remains zero while ∥ D ∥ → ∞ , and consequently ∥ x ∥ → ∞ .

## 2.3.4 Burer-Monteiro factorizations of MaxCut-type semidefinite programs

Let us consider so-called MaxCut-type semidefinite programs (SDPs)

<!-- formula-not-decoded -->

where C ∈ S n × n is the cost matrix . The relaxation (7) provides a precise relaxation to the MaxCut problem, a fundamental combinatorial problem arising in graph optimization [16, 14]. In an effort to exploit the typical low-rank structure of the solution, a Burer-Monteiro factorization [5] decomposes X = V V ⊤ for V ∈ R n × r . This yields

<!-- formula-not-decoded -->

Choosing r much smaller than n significantly decreases the number of variables from n 2 to nr . However, the downside of this approach is that convexity is lost. Fortunately, under certain conditions every second-order stationary point of this nonconvex problem is a global minimizer [14]. Let us denote by x i ∈ R r the i 'th row of V , such that V ⊤ = [ x 1 , x 2 , . . . , x n ] . We also define the vectorized variable x := [ x ⊤ 1 , x ⊤ 2 , . . . , x ⊤ n ] ⊤ ∈ R d where d = nr . Then we denote by f ( x ) the objective of (8) in terms of x , and likewise by A ( x ) = 0 the constraint of (8) in terms of x .

As proposed in the seminal work [5], this nonconvex constrained problem can be solved with an augmented Lagrangian method (ALM). Each iteration consists of minimizing with respect to the primal variable x the (unconstrained) augmented Lagrangian with penalty parameter β &gt; 0 , i.e.,

<!-- formula-not-decoded -->

followed by an update of the multipliers y ∈ R n . A similar strategy was also used in [38] for Burer-Monteiro factorizations of clustering SDPs. The following theorem establishes generalized smoothness of the augmented Lagrangian with respect to the primal variable.

Theorem 2.15. Consider the Burer-Monteiro factorization (8) of the MaxCut-type SDP (7) and let L β denote the augmented Lagrangian with penalty parameter β &gt; 0 of this factorized problem. Then, with respect to the primal variable x ∈ R d and for some fixed multiplier y ∈ R n the following statements hold.

- (i) For any L 1 &gt; 0 there exists L 0 &gt; 0 such that L β ( · , y ) is ( L 0 , L 1 ) -smooth.
- (ii) If ϕ satisfies Assumptions 2.2 and 2.10, then for any ¯ L &gt; 0 , there exists an L &gt; 0 such that L β ( · , y ) satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness.

## 3 Saddle point avoidance of the preconditioned gradient method

The remarkable performance of simple gradient descent-like methods for minimizing nonconvex functions is often attributed to the fact that they avoid strict saddle points of Lipschitz smooth objectives. This section establishes that nonlinear preconditioning of the gradient preserves this desirable property, and in fact generalizes this result to anisotropically smooth functions.

## 3.1 Asymptotic results based on the stable-center manifold theorem

Denote by X ⋆ the set of strict saddle points of a function f ∈ C 2 , i.e.,

Classical results like [25, 24], which are based on the stable-center manifold theorem [39], exploit the fact that the eigenvalues of the Hessian ∇ 2 f are uniformly bounded. In a similar way, for the preconditioned gradient descent method we require that the second-order characterization of ( L, ¯ L ) -anisotropic smoothness holds. By exploiting the fact that ∇ 2 ϕ ∗ (0) = I , we then obtain the following theorem, which generalizes [25, Theorem 4].

<!-- formula-not-decoded -->

Theorem 3.1. Let f ∈ C 2 and suppose that Assumption 2.2 holds. Consider the iterates ( x k ) k ∈ N generated by the preconditioned gradient method, i.e., x k +1 = T γ, ¯ L -1 ( x k ) , where the initial iterate x 0 ∈ R n is chosen uniformly at random. If f satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness, and if γ &lt; 1 L , then

<!-- formula-not-decoded -->

Assumption 2.2 ensures ϕ ∗ ∈ C 2 , which in turn guarantees that T γ, ¯ L -1 ∈ C 1 , as needed for the stable-center manifold theorem [39, 24]. Unfortunately, this means that the reference function ϕ ( x ) = h ( ∥ x ∥ ) with h = 1 2 ∥ · ∥ 2 + δ [ -1 , 1] , which gives rise to a version of the gradient clipping method [36, Example 1.7], is not covered by Theorem 3.1. Indeed, in this case we have h ∗′ ( y ) = Π [ -1 , 1] ( y ) = max { min { y, 1 } , -1 } . Note however that this projection is a piecewise affine function, and therefore h ∗′ is continuously differentiable almost everywhere , i.e., except at the points y = ± 1 .

Based on a recent variant of the stable-center manifold theorem [10] we now establish that also the above clipped gradient variant with ϕ ∗ / ∈ C 2 avoids strict saddle points with probability one. In particular, [10, Proposition 2.5] only requires that the iteration map T γ,λ is continuously differentiable on a set of measure one which contains the set of strict saddle points X ⋆ . We thus have to show that (i) ∇ ϕ ∗ ( ¯ L -1 ∇ f ( · )) is differentiable almost everywhere; and that (ii) ∇ ϕ ∗ ( · ) is differentiable around the point ¯ L -1 ∇ f ( x ⋆ ) = 0 , with x ⋆ ∈ X ⋆ . Remark that the former requires an additional assumption for guaranteeing that ∇ f maps a set of measure one onto a set on which ∇ ϕ ∗ is differentiable.

Theorem 3.2. Let f ∈ C 2+ and ϕ ( x ) = h ( ∥ x ∥ ) with h = 1 2 ∥ · ∥ 2 + δ [ -1 , 1] . Consider the iterates ( x k ) k ∈ N generated by the preconditioned gradient method, i.e., x k +1 = T γ, ¯ L -1 ( x k ) = x k -γ min( 1 / ∥∇ f ( x k ) ∥ , ¯ L -1 ) ∇ f ( x k ) , where the initial iterate x 0 ∈ R n is chosen uniformly at random. Moreover, suppose that the set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is a set of measure one. If f satisfies the second-order sufficient condition for ( L, ¯ L ) -anisotropic smoothness, and if γ &lt; 1 L , then

## 3.2 Efficiently avoiding strict saddle points through perturbations

Despite avoiding strict saddle points asymptotically for almost any initialization, gradient descent may actually be significantly slowed down around saddle points. In fact, gradient descent can take exponential time to escape strict saddle points [13], in the sense that the number of iterations depends exponentially on the dimension n of the optimization variable. Yet, by adding small perturbations, this issue can be mitigated, and the complexity of obtaining a second-order stationary point then depends only polylogarithmically on the dimension n [17, 18]. This section establishes a similar result for a perturbed preconditioned gradient method.

Existing works analyzing the complexity of gradient descent for converging to a second-order stationary point require not only Lipschitz continuity of the gradients, but also of the Hessian. This is quite restrictive, since for example any (non-degenerate) polynomial of degree more than 2 violates this assumption. Instead, we require Lipschitz continuity of the mapping

<!-- formula-not-decoded -->

To ensure well-definedness of H λ , Assumption 2.2 is assumed in the remainder of this section.

Assumption 3.3. The mapping H λ ( x ) := ∇ 2 ϕ ∗ ( λ ∇ f ( x )) ∇ 2 f ( x ) is ρ -Lipschitz-continuous, i.e.,

<!-- formula-not-decoded -->

This new condition appears significantly less restrictive, as illustrated by the following example.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we obtain that H λ ( x ) = λ -1 d ( ∇ ϕ ∗ ( λf ′ ( x )) ) d x = (3 x 2 -1) √ 1+ λ 2 ( x 3 -x ) 2 . One easily verifies that

H λ ∈ C 1 with bounded derivative, which implies the required Lipschitz-continuity of H λ . In fact, this reasoning generalizes to any univariate polynomial, regardless of its degree.

Under anisotropic smoothness, it is natural to consider λ -1 ϕ ( ∇ ϕ ∗ ( λ ∇ f ( x ))) as a first-order stationarity measure, and λ min ( H λ ( x )) as a second-order stationarity measure. Therefore, we say that a point x ∈ R n is an ϵ -second-order stationary point of an ( L, ¯ L ) -anisotropically smooth function f if

<!-- formula-not-decoded -->

For ϕ = 1 2 ∥ · ∥ 2 we recover the classical notion of ϵ -second-order stationarity, with ρ the constant of Lipschitz continuity of ∇ 2 f .

Algorithm 1 describes a perturbed preconditioned gradient method that closely resembles perturbation schemes presented in [17, 18]. In particular, whenever the first-order stationarity is sufficiently small, then a perturbation is added followed by ⌈T ⌉ &gt; 0 unperturbed iterations.

## Algorithm 1 Perturbed preconditioned gradient descent

REQUIRE: x 0 ∈ R n , γ, λ &gt; 0 , perturbation radius r &gt; 0 , time interval T &gt; 0 , tolerance G &gt; 0

- 1: k perturb = 0
- 2: for k = 0 , 1 , . . . do
- 4: x k ← x k + γξ k , ξ k ∼ B 0 ( r ) uniformly, k perturb ← k
- 3: if λ -1 ϕ ( ∇ ϕ ∗ ( λ ∇ f ( x k ))) ≤ G 2 2 and k -k perturb &gt; T then
- 5: x k +1 = x k -γ ∇ ϕ ∗ ( λ ∇ f ( x k ))

We analyze the complexity of algorithm 1 under the following assumption.

Assumption 3.5. Suppose that Assumption 2.2 holds, such that ϕ ∗ ∈ C 2 , and let ϕ ( x ) = h ( ∥ x ∥ ) where in particular h ∈ C 2 . Moreover, let h ( x ) ≥ x 2 / 2 , and h ( x ) = x 2 / 2 + o ( x 2 ) as x → 0 .

This assumption holds for kernel functions from (2). Remark that there is no real loss of generality by fixing the scale of h around 0 , since a rescaled version of h can be obtained by modifying ¯ L .

In our analysis, we specify the parameters of algorithm 1 in terms of L, ¯ L, ϵ and some χ ≥ 1 ,

<!-- formula-not-decoded -->

and introduce two additional constants that are used only in the analysis, i.e.,

<!-- formula-not-decoded -->

We obtain the following complexity of algorithm 1 for converging to a second-order stationary point. Theorem 3.6 (Iteration complexity) . Let f be ( L, ¯ L ) -anisotropically relative to ϕ . Moreover, suppose that Assumptions 3.3 and 3.5 hold, and define constants ∆ f ≥ f ( x 0 ) -inf f and χ = log 2 ( L 2 √ n ∆ f c √ ρ ¯ Lϵ 5 / 2 δ ) for some c &gt; 0 . There exists a constant c max &gt; 0 such that if c ≤ c max , then for any ϵ &gt; 0 sufficiently small, and for any δ ∈ (0 , 1) , Algorithm 1 with parameters as in (12) and (13) , visits an ϵ -second-order stationary point in at least T / 2 iterations with probability at least 1 -δ , where

<!-- formula-not-decoded -->

The ˜ O notation hides a factor χ 4 which is polylogarithmic in the dimension n and in the tolerance ϵ .

Theorem 3.6 generalizes [18, Theorem 18], and relies on a similar high-level proof strategy, which goes as follows. If the current iterate x is not an ϵ -second-order stationary point, then either λ -1 ϕ ( ∇ ϕ ∗ ( λ ∇ f ( x ))) is large, or λ min ( H λ ) is sufficiently negative. In either case, we establish a significant decrease in function value after at most ⌈T ⌉ iterations of algorithm 1. Since f ( x 0 ) -inf f is bounded, the number of iterates which are not ϵ -second-order stationary can be bounded.

Nevertheless, the generalization of [18, Theorem 18] to the setting of Algorithm 1 is by no means straightforward. The original proofs rely heavily on Lipschitz smoothness, in a way that often does not generalize directly to the anisotropically smooth setting. Here, we highlight two such difficulties. First, consider a point x ∈ R n and the perturbed point ¯ x := x + γξ for some perturbation ξ ∈ B 0 ( r ) . Then, by anisotropic smoothness we can upper bound

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

While Lipschitz-smoothness with ϕ = 1 2 ∥ · ∥ 2 readily provides an upper bound in terms of ∥ ξ ∥ 2 ≤ r 2 and ∥∇ f ( x ) ∥ 2 , the reference functions are not typically such that an upper bound in terms of can be obtained. Second, unlike in the L -Lipschitz-smooth case where ∥∇ 2 f ∥ ≤ L , the norm ∥ H λ ∥ cannot be upper bounded uniformly, even under the second-order characterization of ( L, ¯ L ) -anisotropic smoothness. The latter only guarantees that λ max ( H λ ) ≤ L ¯ L , but it does not lower bound λ min ( H λ ) . And even if the eigenvalues of H λ were bounded in absolute value, this still would not guarantee boundedness of ∥ H λ ∥ , since H λ is not a normal matrix in general.

## 4 Numerical validation

Lastly, we illustrate some merits of nonlinear preconditioning, and validate the complexity result of Theorem 3.6 numerically. The source code is publicly available. 2

Nonlinear preconditioning for symmetric matrix factorization For the symmetric matrix factorization problem (5) with n = 2 and r = 1 , Fig. 2 presents a 2D visualization of the level curves of the objective, along with the iterates of both vanilla gradient descent (GD) and the preconditioned variant (P-GD) with ϕ ( x ) = cosh( ∥ x ∥ ) -1 . Unless GD is initialized close to a stationary point, the stepsize must be chosen very small to prevent the iterates from diverging - as expected, because the quartic objective is not Lipschitz smooth. In contrast, the (P-GD) iterations take the form (for ¯ L = 1 )

<!-- formula-not-decoded -->

In this case, large gradients are damped - recall the close resemblance to clipping methods, cf. Fig. 1 - resulting in the convergence of (P-GD) for stepsizes γ that are often orders of magnitudes larger than the maximum stepsize of GD. In turn, this causes (P-GD) to often require significantly fewer iterations, and overall outperform GD for fixed stepsize.

2 https://github.com/alexanderbodard/escaping\_saddles\_with\_preconditioning

Figure 2: Iterates of GD (red) and (P-GD) (blue) on a symmetric matrix factorization problem.

<!-- image -->

Figure 3: Performance of vanilla GD (blue), perturbed vanilla GD [13, Alg 1] (orange), and Algorithm 1 (green) on the 'octopus' function [13].

<!-- image -->

Fast avoidance of saddle points Fig 3 validates the fast escape of saddle points by Algorithm 1. We consider the 'octopus' objective [13] which was constructed such that GD takes exponential time to escape saddle points. We select all hyperparameters as in [13, §5], and set the only additional hyperparameter ¯ L = 1 . We compare against vanilla GD and perturbed vanilla GD [13, Alg 1], and vary the constant L ∈ { 1 , 1 . 5 , 2 , 3 } and dimension n ∈ { 5 , 10 } , thus creating counterparts to [11, Figs 3 and 4]. We observe that algorithm 1 performs similar to perturbed vanilla GD, and also scales in a similar way with respect to n and L . This validates the complexity result from Theorem 3.6.

## 5 Conclusion

This work introduced a novel sufficient condition unifying ( L 0 , L 1 ) -smoothness and anisotropic smoothness. We showed that this condition holds in key applications such as phase retrieval, matrix factorization, and Burer-Monteiro factorizations of MaxCut.

We further analyzed the nonlinearly preconditioned gradient method, which naturally aligns with anisotropic smoothness. Notably, we proved that it preserves the saddle point avoidance properties of gradient descent and extends them to anisotropically smooth settings. This contrasts with prior analyses requiring either global Lipschitz smoothness, or local smoothness combined with compactness, both of which are often unmet in practice.

To our knowledge, this is the first work to rigorously establish saddle point avoidance for problems like phase retrieval and matrix factorization under a smoothness condition that is both practical and verifiable. These results strengthen the theoretical foundations of first-order methods for nonconvex optimization and in particular encourage further study of nonlinear gradient preconditioning.

## Acknowledgments and Disclosure of Funding

The authors are supported by the Research Foundation Flanders (FWO) research projects G081222N, G033822N, G0A0920N and the Research Council KU Leuven C1 project with ID C14/24/103.

We thank Konstantinos Oikonomidis, Jan Quan and Emanuel Laude for the insightful discussions.

## References

- [1] A. S. Bandeira, J. Cahill, D. G. Mixon, and A. A. Nelson. 'Saving phase: Injectivity and stability for phase retrieval'. In: Applied and Computational Harmonic Analysis 37.1 (July 2014), pp. 106-125.
- [2] D. P. Bertsekas. Nonlinear Programming . 2nd ed. Athena Scientific, 1999.
- [3] J. Bolte, S. Sabach, M. Teboulle, and Y. Vaisbourd. 'First order methods beyond convexity and Lipschitz gradient continuity with applications to quadratic inverse problems'. In: SIAM Journal on Optimization 28.3 (2018), pp. 2131-2151.
- [4] C. Bonet, T. Uscidda, A. David, P.-C. Aubin-Frankowski, and A. Korba. 'Mirror and Preconditioned Gradient Descent in Wasserstein Space'. In: Advances in Neural Information Processing Systems 37 (Dec. 2024), pp. 25311-25374.
- [5] S. Burer and R. D. Monteiro. 'A nonlinear programming algorithm for solving semidefinite programs via low-rank factorization'. In: Mathematical Programming 95.2 (Feb. 2003), pp. 329-357.
- [6] E. J. Candès and B. Recht. 'Exact Matrix Completion via Convex Optimization'. In: Foundations of Computational Mathematics 9.6 (Dec. 2009), pp. 717-772.
- [7] E. J. Candès, T. Strohmer, and V. Voroninski. 'PhaseLift: Exact and Stable Signal Recovery from Magnitude Measurements via Convex Programming'. In: Communications on Pure and Applied Mathematics 66.8 (2013), pp. 1241-1274.
- [8] D. Y. Cao, A. Y. Chen, K. Sridharan, and B. Tang. Efficiently Escaping Saddle Points under Generalized Smoothness via Self-Bounding Regularity . Mar. 2025.
- [9] Z. Chen, Y. Zhou, Y. Liang, and Z. Lu. 'Generalized-Smooth Nonconvex Optimization is As Efficient As Smooth Nonconvex Optimization'. In: International Conference on Machine Learning (June 2023), pp. 5396-5427.
- [10] P. Cheridito, A. Jentzen, and F. Rossmannek. 'Gradient Descent Provably Escapes Saddle Points in the Training of Shallow ReLU Networks'. In: Journal of Optimization Theory and Applications (Sept. 2024).
- [11] Y. Chi, Y. M. Lu, and Y. Chen. 'Nonconvex Optimization Meets Low-Rank Matrix Factorization: An Overview'. In: IEEE Transactions on Signal Processing 67.20 (Oct. 2019), pp. 5239-5269.
- [12] M. Crawshaw, M. Liu, F. Orabona, W. Zhang, and Z. Zhuang. 'Robustness to Unbounded Smoothness of Generalized SignSGD'. In: Advances in Neural Information Processing Systems 35 (Dec. 2022), pp. 9955-9968.
- [13] S. S. Du, C. Jin, J. D. Lee, M. I. Jordan, A. Singh, and B. Poczos. 'Gradient Descent Can Take Exponential Time to Escape Saddle Points'. In: Advances in Neural Information Processing Systems . Vol. 30. 2017.
- [14] F. R. Endor and I. Waldspurger. Benign landscape for Burer-Monteiro factorizations of MaxCut-type semidefinite programs . arXiv:2411.03103 [math]. Mar. 2025.
- [15] I. Fatkhullin and N. He. 'Taming Nonconvex Stochastic Mirror Descent with General Bregman Divergence'. In: Proceedings of The 27th International Conference on Artificial Intelligence and Statistics . PMLR, Apr. 2024, pp. 3493-3501.
- [16] M. X. Goemans and D. P. Williamson. 'Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming'. In: J. ACM 42.6 (Nov. 1995), pp. 1115-1145.
- [17] C. Jin, R. Ge, P. Netrapalli, S. M. Kakade, and M. I. Jordan. 'How to Escape Saddle Points Efficiently'. In: Proceedings of the 34th International Conference on Machine Learning . PMLR, July 2017, pp. 1724-1732.
- [18] C. Jin, P. Netrapalli, R. Ge, S. M. Kakade, and M. I. Jordan. 'On Nonconvex Optimization for Machine Learning: Gradients, Stochasticity, and Saddle Points'. In: J. ACM 68.2 (Feb. 2021), 11:1-11:29.
- [19] I. T. Joliffe. Principal Component Analysis . Springer Series in Statistics. New York: SpringerVerlag, 2002.
- [20] E. Laude. 'Lower envelopes and lifting for structured nonconvex optimization'. PhD thesis. Technische Universität München, 2021.

- [21] E. Laude and P. Patrinos. 'Anisotropic proximal gradient'. In: Mathematical Programming (Apr. 2025).
- [22] E. Laude, A. Themelis, and P. Patrinos. 'Dualities for Non-Euclidean Smoothness and Strong Convexity under the Light of Generalized Conjugacy'. In: SIAM Journal on Optimization 33.4 (Dec. 2023), pp. 2721-2749.
- [23] D. D. Lee and H. S. Seung. 'Learning the parts of objects by non-negative matrix factorization'. In: Nature 401.6755 (Oct. 1999), pp. 788-791.
- [24] J. D. Lee, I. Panageas, G. Piliouras, M. Simchowitz, M. I. Jordan, and B. Recht. 'First-order methods almost always avoid strict saddle points'. In: Mathematical Programming 176.1 (July 2019), pp. 311-337.
- [25] J. D. Lee, M. Simchowitz, M. I. Jordan, and B. Recht. 'Gradient Descent Only Converges to Minimizers'. In: Conference on Learning Theory . PMLR, June 2016, pp. 1246-1257.
- [26] F. Léger and P.-C. Aubin-Frankowski. Gradient descent with a general cost . arXiv:2305.04917 [math]. June 2023.
- [27] K. Y. Levy. The Power of Normalization: Faster Evasion of Saddle Points . arXiv:1611.04831 [cs]. Nov. 2016.
- [28] H. Li, J. Qian, Y . Tian, A. Rakhlin, and A. Jadbabaie. 'Convex and Non-convex Optimization Under Generalized Smoothness'. In: Advances in Neural Information Processing Systems 36 (Dec. 2023), pp. 40238-40271.
- [29] H. Li, A. Rakhlin, and A. Jadbabaie. 'Convergence of Adam Under Relaxed Assumptions'. In: Advances in Neural Information Processing Systems 36 (Dec. 2023), pp. 52166-52196.
- [30] H. Lu, R. M. Freund, and Y. Nesterov. 'Relatively Smooth Convex Optimization by First-Order Methods, and Applications'. In: SIAM Journal on Optimization 28.1 (Jan. 2018), pp. 333-354.
- [31] C. J. Maddison, D. Paulin, Y. W. Teh, and A. Doucet. 'Dual Space Preconditioning for Gradient Descent'. In: SIAM Journal on Optimization 31.1 (Jan. 2021), pp. 991-1016.
- [32] M. C. Mukkamala and P. Ochs. 'Beyond Alternating Updates for Matrix Factorization with Inertial Bregman Proximal Gradient Algorithms'. In: Advances in Neural Information Processing Systems . Vol. 32. 2019.
- [33] R. Murray, B. Swenson, and S. Kar. 'Revisiting Normalized Gradient Descent: Fast Evasion of Saddle Points'. In: IEEE Transactions on Automatic Control 64.11 (Nov. 2019), pp. 48184824.
- [34] Y. Nesterov. Lectures on Convex Optimization . Vol. 137. Springer Optimization and Its Applications. Springer International Publishing, 2018.
- [35] K. Oikonomidis, E. Laude, and P. Patrinos. Forward-backward splitting under the light of generalized convexity . arXiv:2503.18098 [math]. Mar. 2025.
- [36] K. Oikonomidis, J. Quan, E. Laude, and P. Patrinos. Nonlinearly Preconditioned Gradient Methods under Generalized Smoothness . arXiv:2502.08532 [math]. Feb. 2025.
- [37] R. T. Rockafellar and R. J. B. Wets. Variational Analysis . Ed. by M. Berger et al. Vol. 317. Grundlehren der mathematischen Wissenschaften. Berlin, Heidelberg: Springer, 1998.
- [38] M. F. Sahin, A. Eftekhari, A. Alacaoglu, F. Latorre, and V. Cevher. 'An Inexact Augmented Lagrangian Framework for Nonconvex Optimization with Nonlinear Constraints'. In: Advances in Neural Information Processing Systems . Vol. 32. 2019.
- [39] M. Shub. Global Stability of Dynamical Systems . Springer New York, 1987.
- [40] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. u. Kaiser, and I. Polosukhin. 'Attention is All you Need'. In: Advances in Neural Information Processing Systems . Vol. 30. 2017.
- [41] B. Zhang, J. Jin, C. Fang, and L. Wang. 'Improved Analysis of Clipping Algorithms for Non-convex Optimization'. In: Advances in Neural Information Processing Systems . Vol. 33. 2020, pp. 15511-15521.
- [42] J. Zhang, T. He, S. Sra, and A. Jadbabaie. Why gradient clipping accelerates training: A theoretical justification for adaptivity . arXiv:1905.11881 [math]. Feb. 2020.
- [43] S.-Y. Zhao, Y.-P. Xie, and W.-J. Li. 'On the convergence and improvement of stochastic normalized gradient descent'. In: Science China Information Sciences 64.3 (Feb. 2021), p. 132103.

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

Justification: We clearly define our assumptions and their limitations.

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

Justification: A detailed proof is provided for every theoretical result.

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

Justification: The paper discloses all the information needed to reproduce the main experimental results.

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

Justification: The paper provides open access to the code at https://github.com/ alexanderbodard/escaping\_saddles\_with\_preconditioning .

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

Justification: The paper specifies all hyperparameters directly or indirectly (same as described in other papers).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The illustrative experiments in this paper do not involve statistical results.

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

Justification: The illustrative experiments in this paper required negligible compute.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: Our research conforms with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: We study generalized smoothness and optimization algorithms from a theoretical perspective. Any societal impact would be indirect and the result of further research.

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

## A Additional results

## A.1 Univariate polynomials satisfy Assumption 2.8

Theorem A.1. Let f ( x ) = ∑ d i =0 a i x i be a univariate polynomial of degree d in x with coefficients a i ∈ R . Then, f satisfies Assumption 2.8.

̸

Proof. Without loss of generality, assume that a d = 0 , since otherwise f would be a polynomial of lower degree. Then, by the triangle inequality we have

<!-- formula-not-decoded -->

which is a polynomial of degree d -2 in | x | . In a similar way, we obtain from the triangle inequality

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which is a polynomial of degree d -1 in | x | where the leading coefficient d | a d | is nonzero.

## A.2 Multivariate polynomials for which ( L 0 , L 1 ) -smoothness fails

This section extends example 2.7 and provides some simple multivariate polynomials which are not ( L 0 , L 1 ) -smooth. In particular, we illustrate that this may still happen if the gradient norm grows unbounded.

Consider the following functions

<!-- formula-not-decoded -->

By a similar reasoning as in example 2.7, we remark that along a path y = -x these functions have gradients

<!-- formula-not-decoded -->

and Hessians

<!-- formula-not-decoded -->

Clearly, these functions cannot be ( L 0 , L 1 ) -smooth, because for y = -x the Hessian norms grow proportionally to | x | 2 , whereas the gradient norms are zero (for f 1 ), constant (for f 2 ), or grow proportionally to | x | (for f 3 ). This is visualized in fig. 4.

Remark that f 3 illustrates that unboundedness of the gradient norm is not sufficient for ( L 0 , L 1 ) -smoothness. Instead, the gradient norm needs to grow 'sufficiently fast'; a sufficient condition is given by Assumption 2.8.

## A.3 Anisotropic smoothness is more general than ( L 0 , L 1 ) -smoothness

Example 2.6 already established that if a function f is ( L 0 , L 1 ) -smooth, then it also satisfies the second-order characterization of ( L 1 , L 0 / L 1 ) -anisotropic smoothness (Assumption 2.8) relative to the reference function ϕ ( x ) = -∥ x ∥-ln(1 -∥ x ∥ ) . We now show that the function f ( x ) = exp( ∥ x ∥ 2 ) is not ( L 0 , L 1 ) -smooth , but satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness with L = 2 , ¯ L = 1 , thus confirming that anisotropic smoothness generalizes ( L 0 , L 1 ) -smoothness.

Note that the function f ( x ) = exp( ∥ x ∥ 2 ) has gradient and Hessian

<!-- formula-not-decoded -->

<!-- image -->

-

Figure 4: Surface plot of some multivariate polynomials which are not ( L 0 , L 1 ) -smooth. The gradient norm is zero (left), constant (middle), or scales proportional to | x | (right) along the path y = -x (red), whereas the Hessian norm scales with | x | 2 .

where we defined r := ∥ x ∥ for ease of notation. Remark also that where the norm of the Hessian follows from the observation that xx ⊤ has eigenvalues 0 and r 2 .

<!-- formula-not-decoded -->

The following theorem establishes that f ( x ) = exp( ∥ x ∥ 2 ) is not ( L 0 , L 1 ) -smooth.

Theorem A.2. There do not exist constants L 0 , L 1 ≥ 0 for which the function f ( x ) = exp( ∥ x ∥ 2 ) is ( L 0 , L 1 ) -smooth.

Proof. Assume, by contradiction that there exist constants L 0 , L 1 ≥ 0 for which

This means that or equivalently,

<!-- formula-not-decoded -->

Clearly, this cannot hold, since the left hand side grows faster than the right hand side as r →∞ .

Theorem A.3. The function f ( x ) = exp( ∥ x ∥ 2 ) satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness relative to the reference function ϕ ( x ) = -∥ x ∥ -ln(1 -∥ x ∥ ) for ¯ L = 1 and L ≥ 2 .

Proof. By [36, Lemma 2.5], the second-order characterization of anisotropic smoothness for ¯ L = 1 is equivalent to

<!-- formula-not-decoded -->

For our particular reference function, this condition becomes

<!-- formula-not-decoded -->

Let us define α := 1 + ∥∇ f ( x ) ∥ = 1 + 2 r exp( r 2 ) and recall that ∇ f ( x ) = 2 exp( r 2 ) x . Then it remains to show that the matrix

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is positive semidefinite for L ≥ 2 , uniformly in x ∈ R n . Note that we can rewrite

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

as the weighted sum of two symmetric positive semidefinite matrices. We now show that both weights A and B are nonnegative for L ≥ 2 , for any r ≥ 0 , from which the claim follows.

<!-- formula-not-decoded -->

The weight A equals and has derivative

<!-- formula-not-decoded -->

For L ≥ 2 this yields

<!-- formula-not-decoded -->

meaning that A ( r ) is strictly increasing. Since A (0) = 1 -2 L , we conclude that A ( r ) is nonnegative for r ≥ 0 .

As for the weight B , remark that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Therefore, the second coefficient becomes

<!-- formula-not-decoded -->

which is nonnegative for L ≥ 2 since exp( r 2 ) ≥ 1 &gt; 1 L

<!-- formula-not-decoded -->

As a second example, we consider the function f ( x ) = exp( ∥ x ∥ 2 ) -2 ∥ x ∥ 2 , which has gradient and Hessian

<!-- formula-not-decoded -->

where we defined r := ∥ x ∥ for ease of notation. Remark also that

<!-- formula-not-decoded -->

where the norm of the Hessian follows from the observation that xx ⊤ has eigenvalues 0 and r 2 .

The following theorem establishes that f is not ( L 0 , L 1 ) -smooth.

Theorem A.4. There do not exist constants L 0 , L 1 ≥ 0 for which the function f ( x ) = exp( ∥ x ∥ 2 ) -2 ∥ x ∥ 2 is ( L 0 , L 1 ) -smooth.

Proof. Assume, by contradiction that there exist constants L 0 , L 1 ≥ 0 for which

This means that

<!-- formula-not-decoded -->

or equivalently,

<!-- formula-not-decoded -->

Clearly, this cannot hold, since the left hand side grows faster than the right hand side as r →∞ .

Theorem A.5. The function f ( x ) = exp( ∥ x ∥ 2 ) -2 ∥ x ∥ 2 satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness relative to the reference function ϕ ( x ) = -∥ x ∥ -ln(1 -∥ x ∥ ) for ¯ L = 1 and L = 10 .

<!-- formula-not-decoded -->

Proof. By [36, Lemma 2.5], the second-order characterization of anisotropic smoothness for ¯ L = 1 is equivalent to

<!-- formula-not-decoded -->

For our particular reference function, this condition becomes

<!-- formula-not-decoded -->

Let us define α := 1+ ∥∇ f ( x ) ∥ = 1+2 r ∣ ∣ exp( r 2 ) -2 ∣ ∣ and recall that ∇ f ( x ) = 2 ( exp( r 2 ) -2 ) x . Then it remains to show that the matrix

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is positive semidefinite for L = 10 , uniformly in x ∈ R n . Note that we can rewrite

Our proof strategy goes as follows. First, we show that A ( r ) ≥ 0 . 8 for all r ≥ 0 . Since the eigenvalues of vv ⊤ ∥ v ∥ 2 are 0 and 1 , and since adding a multiple of the identity matrix shifts the eigenvalues by that multiple, it then suffices to show that B ( r ) ≥ -0 . 8 for r ≥ 0 . This then implies positive semidefiniteness of M L and proves the claim.

We start by lower bounding

<!-- formula-not-decoded -->

We distinguish three cases. If exp( r 2 ) &lt; 2 , then ∣ ∣ exp( r 2 ) -2 ∣ ∣ = 2 -exp( r 2 ) , and hence

<!-- formula-not-decoded -->

On the other hand, if exp( r 2 ) ≥ 2 , then ∣ ∣ exp( r 2 ) -2 ∣ ∣ = exp( r 2 ) -2 , and hence

<!-- formula-not-decoded -->

When additionally exp( r 2 ) ≤ 3 , then clearly A ( r ) ≥ 1 -2 L = 0 . 8 . Thus, it remains to verify the case exp( r 2 ) &gt; 3 , i.e., r &gt; √ ln(3) . We compute the derivative

<!-- formula-not-decoded -->

Since exp( r 2 ) ≥ 3 this yields

<!-- formula-not-decoded -->

meaning that A ( r ) is strictly increasing for r ≥ √ ln(3) . Since A ( √ ln(3)) = 1+2 √ ln(3) -2 L ≥ 0 . 8 , we conclude that A ( r ) ≥ 0 . 8 for r ≥ √ ln(3) . Putting everything together, we have shown that A ( r ) ≥ 0 . 8 for r ≥ 0 .

As for the weight B , remark that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Therefore, the second coefficient is lower bounded by

<!-- formula-not-decoded -->

Let us define z = r 2 ≥ 0 and w = exp( z ) ≥ 1 . Then we can express this lower bound as

The quadratic factor is negative only when 1 . 6 &lt; w &lt; 2 . 5 , and is minimized at w ⋆ = 2 . 05 where it attains the minimum value -0 . 81 . It remains to lower bound Q ( w,z ) for ln(1 . 6) &lt; z &lt; ln(2 . 5) . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We conclude that B ( r ) ≥ -0 . 8 for r ≥ 0 , which completes the proof.

## A.4 Connection to ( ρ, L 0 , L ρ ) -smoothness

This section establishes a connection between anisotropic smoothness and ( ρ, L 0 , L ρ ) -smoothness, which arguably describes the most important subset of ℓ -smooth functions [28].

Definition A.6 ([28, Definition 3]) . A twice continuously differentiable function f : R n → R is ( ρ, L 0 , L ρ ) -smooth for constants ρ, L 0 , L ρ ≥ 0 if ∥∇ 2 f ( x ) ∥ ≤ L 0 + L ρ ∥∇ f ( x ) ∥ ρ for all x ∈ R n .

Note that the original definition is slightly more general, as it encompasses functions without full domain and only requires the Hessian upper bound almost everywhere.

Theorem A.7. Suppose that a univariate function f : R → R is ( ρ, L 0 , L ρ ) -smooth for constants ρ, L 0 , L ρ ≥ 0 , with ρ ≤ 2 . Then, f satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness relative to the reference function ϕ ( x ) = -∥ x ∥ -ln(1 - ∥ x ∥ ) for ¯ L = 1 and L ≥ 2 max { L 0 , L ρ } .

Proof. By [36, Lemma 2.5], the second-order characterization of anisotropic smoothness for ¯ L = 1 is equivalent to

<!-- formula-not-decoded -->

For our particular reference function and because f is univariate, this condition becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now prove that this upper bound holds. By ( ρ, L 0 , L ρ ) -smoothness we have for all x ∈ R that

We distinguish two cases. If | f ′ ( x ) | ≤ 1 , then L -1 f ′′ ( x ) ≤ L 0 L + L ρ L and for L ≥ 2 max { L 0 , L ρ } we obtain L -1 f ′′ ( x ) ≤ 1 , which establishes the required upper bound. If | f ′ ( x ) | &gt; 1 , then it follows from ρ ≤ 2 and L ≥ max { L 0 , L ρ } that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies the required upper bound.

Remark that most examples of univariate ( ρ, L 0 , L ρ ) -smooth functions in [28] satisfy ρ ≤ 2 . Besides polynomials, this includes exponential functions a x with a &gt; 1 , and double exponentials a ( b x ) with a, b &gt; 0 .

## A.5 Verification of assumptions on the reference functions

Throughout this work, we have made a number of assumptions which only relate to the reference function ϕ , i.e., Assumptions 2.1, 2.2, 2.10 and 3.5. This section explicitly verifies these assumptions for isotropic reference functions ϕ = h ◦ ∥ · ∥ where the kernel function h is one of the following:

To this end, the following results from [36, Table 1] will prove useful:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumptions 2.1 and 2.2 These assumptions were proven for h 1 , h 2 , h 3 in [36].

Assumption 2.10 To verify (i), i.e., whether h ∗′ ( y ) y is decreasing on R + , we can check if for y &gt; 0

<!-- formula-not-decoded -->

or equivalently, yh ∗′′ ( y ) &lt; h ∗′ ( y ) . This holds for h 1 , h 2 , and h 3 . Part (ii) holds, since

<!-- formula-not-decoded -->

Also (iii) is satisfied, since h ∗ 1 ′ and h ∗ 2 ′ scale logarithmically and h ∗ 3 ′ is bounded. In particular,

<!-- formula-not-decoded -->

Here we used the fact that arcsinh( y ) = ln( y + √ y 2 +1) . Thus, Assumption 2.10 holds for h 1 , h 2 and h 3 .

Assumption 3.5 The kernel functions have the following Taylor expansions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We remark in particular that each summand is nonnegative, and that the term with the lowest degree equals x 2 / 2 for all three kernel functions. This immediately implies h i ( x ) ≥ x 2 / 2 and h i ( x ) = x 2 / 2 + o ( x 2 ) as x → 0 , for i ∈ { 1 , 2 , 3 } , and proves that the assumption holds for h 1 , h 2 and h 3 .

## A.6 Generalized smoothness of regularized neural networks with quadratic loss

The main goal of this section is to investigate the generalized smoothness of the quadratic loss of a deep neural network, and in particular, whether Assumption 2.8 holds. It turns out that, under sufficient regularization, this is indeed the case. To establish this result, we need to bound the gradient and Hessian norm of the loss by a polynomial of an appropriate degree.

Consider a deep N -layer neural network with quadratic loss. Each layer - of which we denote the index by t ∈ N [0 ,N -1] - consists of weights W t ∈ R n t +1 × n t , a bias term b t ∈ R n t +1 , and a componentwise activation function Σ t : R n t +1 → R n t +1 . The final mapping Σ N -1 is assumed to be the identity mapping, as common in regression problems. For a given data point ¯ x ∈ R n 0 and corresponding label ¯ y ∈ R n N , training this network entails minimizing the loss function

<!-- formula-not-decoded -->

Neural networks are usually trained on a large set of pairs { ¯ x i , ¯ y i } i ∈ N [1 ,I ] , in which case the total loss becomes a summation of the losses for each individual pair. To simplify the presentation, we proceed with I = 1 , i.e., with (14), but the results are easily extended to the case I &gt; 1 .

Let us denote intermediate variables x t +1 = f t ( x t , w t ) = Σ t ( W t x t + b t ) ∈ R n t +1 , where we use the convention 0 , and define the vectorized weights and bias at layer by t

<!-- formula-not-decoded -->

x = ¯ x t w = ( ⊤

The minimization of the loss (14) can then be interpreted as an optimal control problem (OCP) of horizon N with states x t ∈ R n t and inputs w t ∈ R ( n t +1) n t +1 , i.e., minimize

<!-- formula-not-decoded -->

where x 0 = ¯ x , ℓ t ≡ 0 for t ∈ N [0 ,N -1] and ℓ N ( x N ) = 1 2 ∥ x N -¯ y ∥ 2 . In the context of OCPs, the functions f t ( x t , w t ) = Σ t ( W t x t + b t ) are called the dynamics, and have gradients

<!-- formula-not-decoded -->

Let x := ( x 1 ⊤ , . . . , x N ⊤ ) ⊤ and w := ( w 0 ⊤ , . . . , w N -1 ⊤ ) ⊤ denote the vectors containing all states and inputs respectively. We aim to derive expressions for the gradient and Hessian of the loss function (14) with respect to w . To that end, we use a standard idea in optimal control and eliminate the dynamics. This approach is known as single shooting . Following Bertsekas [2, §1.9], we introduce mappings with F -1 ( w ) = ¯ x . Let F ( w ) = ( F 1 ( w ) ⊤ , . . . , F ⊤ N ) ⊤ and note that [2, Eq. (1.246) and below]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We highlight that the block columns correspond to the gradients of the individual mappings F t , i.e.,

<!-- formula-not-decoded -->

Therefore, the OCP, and equivalently, the neural network training problem, is compactly written as

<!-- formula-not-decoded -->

The gradient of J is then easily expressed by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Obtaining an expression for the Hessian is more involved. We follow [2, §1.9], which uses the Lagrangian function of the OCP with multipliers λ = ( λ 1 ⊤ , . . . , λ N ⊤ ) ⊤ ∈ R ∑ N -1 t =0 n t +1 , i.e.,

The central idea is to express ∇ J ( w ) in terms of ∇ x L ( x, w, λ ) and ∇ w L ( x, w, λ ) . The computation of ∇ 2 J ( w ) is simplified by selecting an appropriate multiplier, which is recursively defined by

This yields for example λ 1 = ∇ x 1 f 1 ∇ x 2 f 2 . . . ∇ x N -1 f N -1 ( F N -y ) . Bertsekas [2, Eq. (1.242)] establishes that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The following theorem establishes that, under sufficient regularization, the regularized loss function of neural network training satisfies Assumption 2.8, and consequently satisfies the generalized smoothness notions investigated in this work. We emphasize that the rather restrictive bound on the power P of the regularizer can be relaxed significantly by exploiting the structure of ∇ F ( w ) (cf. (16)) and further working out the Hessian ∇ 2 J ( w ) (cf. (19)) before upper bounding its norm. However, for simplicity, the proof below immediately uses submultiplicativity of the matrix norm (cf. (21)).

Theorem A.8. Consider the regularized neural network training problem with objective

<!-- formula-not-decoded -->

where J ( w ) is the quadratic loss of an N -layer network as defined in (17) . Suppose that the mappings Σ t are bounded and have bounded first and second derivatives for t ∈ N [0 ,N -2] , and that the mapping Σ N -1 is the identity map. If P ≥ 3 N +2 , then the following statements hold for any κ &gt; 0 .

- (i) For any L 1 &gt; 0 there exists an L 0 &gt; 0 such that ˜ J is ( L 0 , L 1 ) -smooth.
- (ii) If ϕ satisfies Assumptions 2.2 and 2.10, then for any ¯ L &gt; 0 , there exists an L &gt; 0 such that ˜ J satisfies the second-order characterization of ( L, ¯ L ) -anisotropic smoothness.

Proof. By boundedness of Σ t , also the states x t +1 are bounded for t ∈ N [0 ,N -2] , whereas we note that ∥ x N ∥ = ∥ F N ( w ) ∥ = O ( ∥ w ∥ ) . From the gradient expressions (15), we observe that ∥∇ w t f t ∥ = O (1) and ∥∇ x t f t ∥ = O ( ∥ w ∥ ) , and by (16) we obtain ∥∇ F ( w ) ∥ = O ( ∥ w ∥ N -1 ) . Therefore, the gradient ∇ J ( w ) as defined in (18) is upper bounded by a polynomial of degree N in ∥ w ∥ , i.e., ∥∇ J ( w ) ∥ = O ( ∥ w ∥ N ) . Consequently,

<!-- formula-not-decoded -->

As for the Hessian norm, we have by (19) that

<!-- formula-not-decoded -->

We compute the gradient of the Lagrangian with respect to x and w

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Clearly, ∇ 2 xx L ( F ( w ) , w, λ ) is block diagonal. The last block ∇ 2 x N x N L ( x, w, λ ) = I n N is straightforward to compute. For the other ones, we proceed by rewriting

<!-- formula-not-decoded -->

Here we used the fact that vec( z ) = z for any vector, and vec( IBA ) = ( A ⊤ ⊗ I ) vec( B ) for any matrices A,B and identity matrix I of compatible dimensions. This yields

<!-- formula-not-decoded -->

From the fact that ∥ λ t +1 ∥ = O ( ∥ w ∥ N -t ) for t = 0 , . . . , N -1 , and from boundedness of Σ ′′ t we obtain and it follows that ∥∇ 2 xx L ( x, w, λ ) ∥ = O ( ∥ w ∥ N +2 ) . Similar arguments can be used to show that also the other Hessian blocks of the Lagrangian satisfy a similar upper bound. In conclusion, we obtain that ∥∇ 2 J ( w ) ∥ = O ( ∥ w ∥ 2 N -2 ∥ w ∥ N +2 ) = O ( ∥ w ∥ 3 N ) . It follows immediately that ∥∇ 2 ˜ J ( w ) ∥ is upper bounded by a polynomial of degree 3 N in ∥ w ∥ . And since by (20) the gradient ∥∇ ˜ J ( w ) ∥ is lower bounded by a polynomial of degree 3 N +1 in ∥ w ∥ with strictly positive leading coefficient, we conclude that Assumption 2.8 holds. The claims then follow by Theorems 2.9 and 2.11.

<!-- formula-not-decoded -->

## B Auxiliary results

Lemma B.1 (Power Mean inequality) . Let p &gt; q &gt; 0 . Then,

<!-- formula-not-decoded -->

Proof. Note that the function φ ( α ) = α q / p is concave for α &gt; 0 since q &lt; p . By Jensen's inequality, this implies

<!-- formula-not-decoded -->

Raising both sides to the power 1 / q establishes the claim.

Lemma B.2. Let F : R n → R m be a mapping satisfying, for all x ∈ R n , where a &gt; 0 and b ≥ 0 . Then, for all x ∈ R n ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. In the proof we write r = ∥ x ∥ for simplicity. The given inequality becomes:

We distinguish two cases:

<!-- formula-not-decoded -->

Case 1: r 2 ≥ b a . In this case the given lower bound is nonnegative. Taking the square root yields

<!-- formula-not-decoded -->

We apply the inequality √ 1 -z ≥ 1 -z for 0 ≤ z ≤ 1 , by letting z = b ar 2 ∈ [0 , 1] . Then

<!-- formula-not-decoded -->

Multiplying both sides by r 2 , we obtain

<!-- formula-not-decoded -->

Case 2: r 2 &lt; b a . The claim follows immediately by nonnegativity of the norm and from the fact that the desired lower bound is negative in this case, i.e.,

<!-- formula-not-decoded -->

Lemma B.3. Suppose that Assumption 3.5 holds, and that a point x ∈ R n satisfies

<!-- formula-not-decoded -->

where G is defined as in (12) . Then,

<!-- formula-not-decoded -->

Proof. The bound ϕ ( x ) ≥ ∥ x ∥ 2 2 yields

<!-- formula-not-decoded -->

The claim then follows by ϕ = h ◦ ∥ · ∥ , nonnegativity of G and strict monotonicity of h .

## C Missing proofs of section 2

## C.1 Proof theorem 2.4

Proof. This proposition is a direct combination of [36, Propositions 2.6 &amp; 2.9].

## C.2 Proof of theorem 2.9

Proof. By Assumption 2.8 we have

<!-- formula-not-decoded -->

By nonnegativity of ∥∇ 2 f ( x ) ∥ F / ∥∇ f ( x ) ∥ we conclude that lim ∥ x ∥→∞ ∥∇ 2 f ( x ) ∥ F ∥∇ f ( x ) ∥ = 0 . Thus, for any L 1 &gt; 0 there exists δ &gt; 0 such that

<!-- formula-not-decoded -->

Moreover, by continuity of ∇ 2 f , we know that ∥∇ 2 f ( x ) ∥ F is bounded on the compact set Ω := { x | ∥ x ∥ ≤ δ } . We conclude that f is ( L 0 , L 1 ) -smooth with L 0 = max x ∈ Ω ∥∇ 2 f ( x ) ∥ F .

## C.3 Proof of theorem 2.11

Proof. Fix an arbitrary ¯ L &gt; 0 and, for ease of notation, define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If lim ∥ x ∥→∞ ∥ H λ ( x ) ∥ = 0 , then for any ϵ &gt; 0 there exists δ &gt; 0 such that

Therefore, the continuous function x →∥ H λ ( x ) ∥ is bounded on the compact set Ω := { x | ∥ x ∥ ≤ δ } , and for all x ∈ R n \ Ω we know that ∥ H λ ( x ) ∥ ≤ ϵ . We conclude that if lim ∥ x ∥→∞ ∥ H λ ( x ) ∥ = 0 , then ∥ H λ ( x ) ∥ is bounded on R n , and because λ max ( H λ ( x )) ≤ ∥ H λ ( x ) ∥ , this would prove the claim. By equivalence of norms, it suffices to show boundedness of any norm; we proceed with the Frobenius norm ∥ H λ ( x ) ∥ F . Because for isotropic reference functions it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking the limes superior and using Assumption 2.10 yields

<!-- formula-not-decoded -->

Here, the third step used the fact that h ∗′ ( x ) / x is decreasing on R + in combination with ∥ ¯ L -1 ∇ f ( x ) ∥ ≥ ¯ L -1 q R +1 ( ∥ x ∥ ) . Since ∥ · ∥ ≥ 0 , we conclude lim ∥ x ∥→∞ ∥ H λ ( x ) ∥ = 0 .

## C.4 Proof of theorem 2.12

Lemma C.1 (Gradient norm lower bound) . Consider the phase retrieval problem with objective (4) and suppose that that the vectors { a i } i m =1 span R n . Then, there exists a constant C &gt; 0 which depends on the measurement vectors { a i } i m =1 (but not on x ) such that

<!-- formula-not-decoded -->

Proof. Clearly, the gradient norm can be lower bounded by

<!-- formula-not-decoded -->

Let us further lower bound ∥ ∑ m i =1 ( a ⊤ i x ) 3 a i ∥ in terms of ∥ x ∥ . For ease of notation, we denote g ( x ) := ∥ G ( x ) ∥ where G ( x ) := ∑ m i =1 ( a ⊤ i x ) 3 a i . Observe that g is positively homogeneous of degree 3 [37, Definition 13.4], since for any λ &gt; 0

<!-- formula-not-decoded -->

For any x ∈ R n we have g ( x ) = ∥ x ∥ 3 g ( x / ∥ x ∥ ) , and hence it suffices to lower bound g on the unit sphere S n -1 := { x ∈ R n | ∥ x ∥ = 1 } . Remark that g achieves a minimum over S n -1 because S n -1 is compact and g is continuous. Denote this minimum by C := min u ∈ S n -1 g ( u ) , such that

<!-- formula-not-decoded -->

If C &gt; 0 , then the proof is done. For this reason, we show by contradiction that the case C = 0 is not possible. If C = 0 , there exists u ⋆ ∈ S n -1 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It follows that and since ( a ⊤ i u ⋆ ) 4 ≥ 0 we conclude that a ⊤ i u ⋆ = 0 for i = 1 , . . . m , i.e., u ⋆ is orthogonal to all measurement vectors. Therefore, u ⋆ ∈ (span { a 1 , . . . , a m } ) ⊤ = ( R n ) ⊤ = { 0 } . This results in a contradiction, since u ⋆ = 0 / ∈ S n -1 . We conclude that C &gt; 0 , which proves the claim.

Proof. The objective f is a fourth-order polynomial in the variables x . Hence, one easily verifies that

<!-- formula-not-decoded -->

where the upper bound is a polynomial of degree 2 in ∥ x ∥ . Moreover, by lemma C.1 the gradient norm ∥∇ f ( x ) ∥ can be lower bounded by a polynomial of degree 3 in ∥ x ∥ , with strictly positive leading coefficient. Therefore, Assumption 2.8 holds. Theorem 2.12(i) now follows directly from Theorem 2.9, and Theorem 2.12(ii) from Theorem 2.11.

## C.5 Proof of theorem 2.13

Lemma C.2 (Gradient norm lower bound) . Consider the matrix factorization problem with objective (5) . Then the gradient norm can be lower bounded by

<!-- formula-not-decoded -->

Proof. We have that

<!-- formula-not-decoded -->

Remark that U ⊤ U is symmetric and positive semi-definite with eigenvalues λ 1 , . . . , λ n ≥ 0 . Therefore, trace( U ⊤ U ) = ∑ n i =1 λ i = ∥ U ∥ 2 F , and

<!-- formula-not-decoded -->

The power mean inequality (lemma B.1) with p = 3 and q = 1 yields

<!-- formula-not-decoded -->

and hence

<!-- formula-not-decoded -->

Proof. We trace the steps from Theorem 2.12. Since f is a fourth-order polynomial in the variables U , the Hessian norm can be upper bounded by a second-order polynomial in ∥ U ∥ . By combining lemmas B.2 and C.2, we conclude that the gradient norm is lower bounded by a polynomial of degree 3 in ∥ U ∥ , with strictly positive leading coefficient. Therefore, Assumption 2.8 holds. The claim then follows from Theorems 2.9 and 2.11, respectively.

## C.6 Proof of theorem 2.14

Lemma C.3 (Gradient norm lower bound) . Consider the asymmetric matrix factorization problem with objective (6) . Denote V = max( ∥ W ∥ , ∥ H ∥ F ) . Then,

Proof. We have that

<!-- formula-not-decoded -->

The first term can be lower bounded by

<!-- formula-not-decoded -->

Here the second to last step used the cyclic property of the trace. In a similar way, we obtain

<!-- formula-not-decoded -->

Putting this all together, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let x be the concatenation of vec( W ) and vec( H ) , and let V := max( ∥ W ∥ F , ∥ H ∥ F ) . Then ∥ x ∥ = √ ∥ W ∥ 2 F + ∥ H ∥ 2 F ≥ V ≥ 1 2 √ ∥ W ∥ 2 F + ∥ H ∥ 2 F = 1 2 ∥ x ∥ , and we remark that ∥ x ∥ → ∞ if and only if V →∞ .

We now trace the steps from Theorem 2.12. Since f is a fourth-order polynomial in the variables x , the Hessian norm can be upper bounded by a second-order polynomial in ∥ x ∥ . By combining lemmas B.2 and C.3 with ∥ x ∥ ≥ V ≥ 1 2 ∥ x ∥ , we conclude that the gradient norm is lower bounded by a polynomial of degree 3 in ∥ x ∥ , with strictly positive leading coefficient. Therefore, Assumption 2.8 holds. The claim then follows from Theorems 2.9 and 2.11, respectively.

## C.7 Proof of theorem 2.15

We first establish the following lemma.

Lemma C.4. Consider the Burer-Monteiro factorization (8) of the MaxCut-type SDP (7) and let L β denote the augmented Lagrangian with penalty parameter β &gt; 0 of this factorized problem. Then, there exist constants C 1 , C 0 ≥ 0 such that

<!-- formula-not-decoded -->

Proof. The gradient of the augmented Lagrangian with respect to x is

<!-- formula-not-decoded -->

and since f and A are quadratic in x we can lower bound its norm by

<!-- formula-not-decoded -->

The constraint mapping A has the particular form

<!-- formula-not-decoded -->

and therefore the gradient of the augmenting term equals

<!-- formula-not-decoded -->

Since ∥ y ∥ 2 = ∑ n i =1 ∥∥ x i ∥ 2 x i ∥ 2 = ∑ n i =1 ∥ x i ∥ 6 , we can apply the power mean inequality lemma B.1 with p = 6 and q = 2 . This yields

<!-- formula-not-decoded -->

Thus ∥ y ∥ 1 / 3 ≥ 1 n 1 / 3 ∥ x ∥ or ∥ y ∥ ≥ 1 n ∥ x ∥ 3 . Putting everything together, we obtain

<!-- formula-not-decoded -->

The claim follows by redefining the constant C 1 .

We now present the proof of Theorem 2.15.

Proof. We again trace the steps from Theorem 2.12. Since for fixed multipliers y , the augmented Lagrangian L β is a fourth-order polynomial in the variables x , it follows that the Hessian norm can be upper bounded by a second-order polynomial in ∥ x ∥ . Moreover, from lemma C.4, we know that the gradient norm is lower bounded by a polynomial of degree 3 in ∥ x ∥ , with strictly positive leading coefficient. Therefore, Assumption 2.8 holds. The claim then follows from Theorems 2.9 and 2.11, respectively.

## D Missing proofs of section 3

## D.1 Proof of theorem 3.1

Proof. First, under Assumption 2.2 the iteration map T γ, ¯ L -1 is a C 1 mapping with Jacobian

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since λ max ( ∇ 2 ϕ ∗ ( ¯ L -1 ∇ f ( x )) ∇ 2 f ( x ) ) ≤ L ¯ L and γ &lt; 1 L it follows that

Hence, det JT γ, ¯ L -1 ( x ) = 0 for all x ∈ R n .

̸

Second, denote the set of unstable fixed points by

<!-- formula-not-decoded -->

For any stationary point x ⋆ , satisfying ∇ f ( x ⋆ ) = 0 , we know that x ⋆ = T γ, ¯ L -1 ( x ⋆ ) , and therefore

<!-- formula-not-decoded -->

Hence, if λ min ( ∇ 2 f ( x ⋆ )) &lt; 0 , then max i | λ i ( JT γ, ¯ L -1 ( x ⋆ )) | &gt; 1 . We conclude that X ⋆ ⊆ A ⋆ .

Thus, we have established all conditions of [24, Corollary 1], from which the claim follows.

## D.2 Proof of theorem 3.2

Proof. First, remark that U is an open set. For x ∈ U we have by [36, Lemma 1.3] that

We partition U into two sets

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If x ∈ U 1 , then ∇ ϕ ∗ ( ¯ L -1 ∇ f ( x )) = ¯ L -1 ∇ f ( x ) . Thus, T γ, ¯ L -1 is continuously differentiable with locally Lipschitz continuous Jacobian on U 1 . If x ∈ U 2 , then ∇ ϕ ∗ ( ¯ L -1 ∇ f ( x )) = ∇ f ( x ) / ∥∇ f ( x ) ∥ . Since the function ↦→ z / ∥ z ∥ is twice continuously differentiable on R n \ { 0 } and since f ∈ C 2+ , it follows that T γ, ¯ L -1 is continuously differentiable with locally Lipschitz continuous Jacobian on U 2 . Since U = U 1 ∪ U 2 , we conclude that JT γ, ¯ L -1 is locally Lipschitz continuous on U . By the same arguments as in the proof of Theorem 3.1 we conclude that

<!-- formula-not-decoded -->

and that for any x ⋆ ∈ X ⋆ the Jacobian JT γ, ¯ L -1 ( x ⋆ ) is symmetric and has an eigenvalue of absolute value strictly greater than 1 . The claim now follows immediately by applying [10, Proposition 2.5].

## D.3 Proof of theorem 3.6

Henceforth, we assume that f is ( L, ¯ L ) -anisotropically smooth relative to ϕ , without further mention. We first describe the following descent lemma.

Lemma D.1 (Descent lemma) . Suppose that Assumption 3.5 holds. Let γ = α / L , α ∈ (0 , 1) , and let ( x k ) k ∈ N denote a preconditioned gradient sequence, i.e., x k +1 = T γ,λ ( x k ) for k ∈ N . Then,

<!-- formula-not-decoded -->

Proof. By [36, §C.1] we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The claim then follows from ϕ ( x ) ≥ ∥ x ∥ 2 / 2 .

We now establish the two key lemmas for our analysis. First, we establish that if the objective does not decrease much, then the iterates will remain close to the initial point.

Lemma D.2 (Improve or localize) . Suppose that Assumption 3.5 holds. Let ( x k ) k ∈ N denote a preconditioned gradient sequence, i.e., x k +1 = T γ,λ ( x k ) for k ∈ N . Then, for any t ≥ τ &gt; 0

<!-- formula-not-decoded -->

Proof. By consecutively applying the triangle inequality, Cauchy-Schwarz, and lemma D.1 we obtain

<!-- formula-not-decoded -->

Second, we show that the region in which the iterates of algorithm 1 remain stuck for at least ⌈T ⌉ iterations (if initialized there) is small. We do this by showing that there exists a point, not to far away, which does yield sufficient decrease.

Lemma D.3 (Coupling sequence) . Suppose that Assumptions 3.3 and 3.5 holds. Let a point ˜ x ∈ R n satisfy λ min ( H λ (˜ x )) ≤ - √ ρϵ where ϵ ≤ L 2 ρ . Moreover, let ( x k ) k ∈ N , ( y k ) k ∈ N denote two preconditioned gradient sequences, i.e., x k +1 = T γ,λ ( x k ) and y k +1 = T γ,λ ( y k ) for k ∈ N , which additionally satisfy

<!-- formula-not-decoded -->

where e 1 is the minimum eigenvector of H λ (˜ x ) and r 0 &gt; ω := 2 2 -χ L Z . Then,

<!-- formula-not-decoded -->

Proof. By contradiction, assume that min { f ( x T ) -f ( x 0 ) , f ( y T ) -f ( y 0 ) } &gt; -F . Lemma D.2 states that for any t ≤ T

<!-- formula-not-decoded -->

Here the last step follows from ϵ ≤ L 2 ρ . Denote by z t := x t -y t the difference between the two sequences. Then, it follows by the mean value theorem that

<!-- formula-not-decoded -->

where H := H λ (˜ x ) and ∆ t := ∫ 1 0 [ H λ ( y t + θ ( x t -y t )) -H ] dθ . We show by induction that

<!-- formula-not-decoded -->

For the base case t = 0 , the claim holds trivially, since ∥ q (0) ∥ = 0 ≤ 1 2 ∥ p (0) ∥ . For the induction step, we assume that the claim holds for t and show that it also holds for t +1 . Since z 0 lies along the minimum eigenvector of H λ (˜ x ) , we have for any τ ≤ t

<!-- formula-not-decoded -->

where Γ := -λ min ( H λ (˜ x )) . By Lipschitz-continuity of H λ (cfr. Assumption 3.3) we have

<!-- formula-not-decoded -->

Combined with 2 γρ ZT = 1 / 2 , we obtain

<!-- formula-not-decoded -->

This completes the proof of (23). In turn, we conclude that

<!-- formula-not-decoded -->

Here we used (1 + x ) 1 / x ≥ 2 for x ∈ (0 , 1] . This contradicts (22) and completes the proof.

By combining lemma D.2 and lemma D.3, we can show that the iterates of algorithm 1 will escape from a strict saddle point with high probability.

Lemma D.4 (Escaping strict saddle points) . Suppose that a point ˜ x satisfies

<!-- formula-not-decoded -->

for ϵ &gt; 0 small enough. Let x 0 := ˜ x + γξ where ξ is sampled uniformly from a ball with radius r , and that x k +1 = x k -γ ∇ ϕ ∗ ( λ ∇ f ( x k )) for k ∈ N . Then,

<!-- formula-not-decoded -->

Proof. We define X stuck as in [18, Lemma 20], i.e.,

<!-- formula-not-decoded -->

By the same arguments as in [18, Lemma 20] we conclude that

<!-- formula-not-decoded -->

We now proceed by showing that f ( x T ) -f (˜ x ) ≤ -F 2 if x 0 / ∈ X stuck. By anisotropic smoothness, we have

<!-- formula-not-decoded -->

By monotonicity of h , the triangle inequality and lemma B.3, we have

Here, the last step uses the bound √ λ G ≤ r , which follows from G = min { 1 , 1 √ λ } r . For r sufficiently small - which holds for ϵ sufficiently small - we can further bound h (2 r ) ≤ 5 8 (2 r ) 2 = 5 2 r 2 . This yields, again using ϵ ≤ L 2 ρ , that f ( x 0 ) -f (˜ x ) ≤ 5 2 γ λ r 2 = F / 2 . We conclude that if x 0 / ∈ X stuck, then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 3.6 now follows by combining lemmas D.1 and D.4.

Proof. Let the total number of iterations of algorithm 1 be

<!-- formula-not-decoded -->

Using max { 50 χ 4 , 200 χ 3 } ≤ 2 8 · 50 χ 4 ≤ 2 8+7+13+ χ / 4 for all χ &gt; 1 / 4 and , we find

<!-- formula-not-decoded -->

and since for χ &gt; 1 4 we have χ 2 2 -χ ≤ 2 10 -χ / 2 it follows that

<!-- formula-not-decoded -->

Selecting c max = 2 -39 , i.e., χ ≥ 4 log 2 ( 2 39 L 2 √ n ∆ f √ ρ ¯ Lϵ 5 / 2 δ ) we obtain

<!-- formula-not-decoded -->

Observe that G ≤ ϵ , such that λ -1 ϕ ( ∇ ϕ ∗ ( λ ∇ f ( x k ))) ≤ G 2 2 implies λ -1 ϕ ( ∇ ϕ ∗ ( λ ∇ f ( x k ))) ≤ ϵ 2 . With probability at least 1 -δ , algorithm 1 adds a perturbation at most T / 4 T times to a point, because by lemma D.4 we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Excluding the iterations which follow within T steps after adding a perturbation, we have at most 3 T / 4 iterations left. They either satisfy λ -1 ϕ ( ∇ ϕ ∗ ( λ ∇ f ( x k ))) ≥ G 2 2 or are ϵ -second-order stationary points. Among these, at most T / 4 are not second-order stationary points, because by lemma D.1 we have

Therefore, we conclude that at least T / 2 iterations of algorithm 1 are ϵ -second-order stationary points.