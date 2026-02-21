## Solving and Learning Partial Differential Equations with Variational Q-Exponential Processes

Guangting Yu

Shiwei Lan ∗

School of Mathematical &amp; Statistical Sciences Arizona State University, Tempe, AZ 85287

## Abstract

Solving and learning partial differential equations (PDEs) lies at the core of physicsinformed machine learning. Traditional numerical methods, such as finite difference and finite element approaches, are rooted in domain-specific techniques and often lack scalability. Recent advances have introduced neural networks and Gaussian processes (GPs) as flexible tools for automating PDE solving and incorporating physical knowledge into learning frameworks. While GPs offer tractable predictive distributions and a principled probabilistic foundation, they may be suboptimal in capturing complex behaviors such as sharp transitions or non-smooth dynamics. To address this limitation, we propose the use of the q -exponential process (Q-EP), a recently developed generalization of GPs designed to better handle data with abrupt changes and to more accurately model derivative information. We advocate for Q-EP as a superior alternative to GPs in solving PDEs and associated inverse problems. Leveraging sparse variational inference, our method enables principled uncertainty quantification - a capability not naturally afforded by neural network-based approaches. Through a series of experiments, including the Eikonal equation, Burgers' equation, and an inverse Darcy flow problem, we demonstrate that the variational Q-EP method consistently yields more accurate solutions while providing meaningful uncertainty estimates.

Keywords: Probabilistic PDE Solvers, Bayesian Inverse Problems, Data Inhomogeneity, Modeling Derivatives, Uncertainty Quantification

## 1 Introduction

It is of fundamental importance in science and technology to solve mathematical models represented as a system of differential equations and to learn such a complex system by identifying crucial physical quantities with estimated uncertainty (a.k.a. inverse problems). Over centuries, theoretic foundations and computational methods have been developed for solving and learning partial differential equations (PDEs), which is facilitated by the development of modern computers. Traditional numerical algorithms such as the finite element method remain demanding for both domain knowledge and computing resources. Recently, there has been increasing interest and effort to efficiently automate this process using machine learning techniques.

The surge of physics-informed machine learning is driven by two main thrusts: neural network-based algorithms and Gaussian process (GP)-based probabilistic solvers. The former works are represented by physics-informed neural networks [PINN 37, 49], the deep Ritz method [9], the deep Galerkin method [44], and operator learning [20] methods including the Fourier neural operator [FNO 25], deep operator networks [DeepONet 26] and the neural inverse operator [NIO 28]. See [19] for a

∗ slan@asu.edu

review of recent advances. The basic idea is to parametrize the solution with a neural network and to minimize certain loss with respect to network parameters to obtain the solution. Despite empirical successes, these neural network-type approaches typically require large samples but lack convergence guarantee or uncertainty quantification (UQ). On the other hand, GP has been introduced to solve and learn ordinary differential equations (ODEs) [45, 39, 4, 43, 15] and PDEs [32, 33, 35, 38] with theoretic guarantee [34] and UQ [16]. More recent development on GP for solving and learning PDEs includes [6, 27, 14, 13, 29, 3]. Different from neural network approaches, these probabilistic solvers model the solution as a GP conditioned on PDE constraints and identify the solution as the maximum a posteriori (MAP).

Due to the tractability of conditional and predictive densities, GP has been widely adopted in machine learning and scientific computing [40]. However, as an L 2 regularization, GP tends to be over-smooth for modeling certain objects with abrupt changes or sharp contrast. For example, it is known in imaging analysis that GP may not detect or preserve edges very well in an image [23, 7]. On the other hand, researchers [42, 48] notice that total variation regularization penalizes the L 1 norm of derivatives and yields edge-preserving reconstructions. However, the total variation prior degenerates to GP prior with increasingly finer discretization mesh [23] and hence loses its edge-preserving feature. Therefore, [22] propose the Besov prior as an L q regularization and prove its discretizationinvariant property. [24] further develop the q -exponential process (Q-EP) as a probabilistic definition of the Besov process with tractable posterior prediction and demonstrate it as a superior generalization of GP (with q = 2 ) in modeling inhomogeneous data with sharp transitions.

In this paper, we discover that Q-EP (with q = 1 ) is better in modeling derivative information than GP, and hence presents as a preferable candidate for solving PDEs. Heuristically, this is attributed to Q-EP's enhanced ability to model inhomogeneous objects with sharp variations, resulting in better regularization of large derivatives. Theoretically, this can be explained by a faster posterior convergence rate in Bayesian modeling with Q-EP priors. Unlike optimization-based approaches [6, 27], we adopt sparse variational inference [46, 47] for Q-EP [30, 5] to solve and learn PDEs, allowing natural UQ. An emerging challenge is that in addition to mapping the Q-EP mean function by the nonlinear PDE dynamics, one also needs to propagate the whole variational distribution through, which no longer renders a Q-EP. We solve this difficulty by linearizing the complicated PDE mapping. We also extend the resulting variational Q-EP solver for inverse problems.

Connection to the literature Our work is motivated by [6] which optimizes the log-posterior for MAP as the PDE solution. Our proposed method replaces GP with a more general Q-EP and adopts variational Bayes for UQ. We investigate Q-EPs in solving various forward and inverse PDEs for a spectrum of q 's with q = 2 corresponding to GP. As a probabilistic solver, Q-EP may not be best compared with neural network-based approaches. However, we still include PINN [37] and Bayesian PINN [B-PINN 49] as baselines in our comparison. We emphasize that our algorithms rely only on limited data, e.g. boundary values or interior observations, while providing meaningful UQ. Our work is also related to the recently proposed physics-informed state-space GP [13], which however focuses on time-dependent PDEs. It adopts a variational spatiotemporal state-space GP and can be regarded as a related special case of ours for q = 2 . Our work on solving and learning PDEs makes multiple contributions to the field of physics-informed machine learning:

1. It is a novel probabilistic PDE solver based on Q-EP with superior capability of modeling data inhomogeneity and derivative information.
2. It theoretically justifies the preference of Q-EP over GP in solving and learning PDEs.
3. It provides efficient UQ for solving forward and inverse PDE problems.

The remainder of the paper is organized as follows. Section 2 reviews Q-EP as a flexible prior in Bayesian models for inhomogeneous data and introduces an extension to incorporate derivative information. Section 3 explains the details of applying Q-EP to solve the forward and inverse problems of PDEs. We follow [6] to model the solution as MAP of Q-EP but highlight the challenges of variational inference including distribution propagation and variational lower bound. In Section 4, we justify the preference of Q-EP for q = 1 over GP ( q = 2 ) in solving PDEs. In Section 5, we demonstrate the numerical advantages, particularly faster convergence, of Q-EP compared with alternatives using forward problems involving Eikonal equation and Burgers' equation and inverse problems of identifying permeability in the Darcy flow. Section 6 concludes with a discussion of limitations and future improvements.

## 2 Bayesian Modeling with Q-Exponential Process

## 2.1 Q -Exponential Process

The univariate q -exponential distribution [7] has density π q ( u ) ∝ exp( -1 2 | u | q ) , whose logarithm yields an L q regularization term. [24] generalize the univariate q -exponential random variable to a multivariate random vector, based on which a stochastic process can be defined. Suppose a function u ( x ) is observed at N locations, x 1 , · · · , x N ∈ Ω ⊂ R d . [24] define the multivariate q -exponential distribution for u := ( u ( x 1 ) , · · · , u ( x N )) , as a member of the family of elliptic distributions [18].

Definition 1. A multivariate random vector u ∈ R N follows the q -exponential distribution, denoted as u ∼ q -ED N ( µ , C ) , if it has the following density:

<!-- formula-not-decoded -->

Remark 1. The negative log density of q -ED in (1) yields a quantity dominated by some weighted L q norm of u -µ , i.e. 1 2 r q 2 = 1 2 ∥ u -µ ∥ q C . From the optimization perspective, q -ED , when used as a prior, imposes L q regularization in obtaining the maximum a posteriori (MAP).

Li et al. [24] prove that the above multivariate q -exponential distribution satisfies the conditions of Kolmogorov's extension theorem [31] and thus can be generalized to a stochastic process. For this purpose, we scale u ∼ q -ED N ( 0 , C ) by a factor N 1 2 -1 q so that the scaled q -exponential random variable u ∗ := N 1 2 -1 q u ∼ q -ED ∗ N ( 0 , C ) has covariance asymptotically finite [Proposition 3.1 of 24]. With a covariance (symmetric and positive-definite) kernel C : Ω × Ω → R , we define the following q -exponential process (Q-EP) based on the scaled q -exponential distribution.

Definition 2. A (centered) q -exponential process u ( x ) with a kernel C , q -EP (0 , C ) , is a collection of random variables such that any finite set, u = ( u ( x 1 ) , · · · u ( x N )) , follows a scaled multivariate q -exponential distribution q -ED ∗ ( 0 , C ) , where C = [ C ( x i , x j )] N × N .

Remark 2. When q = 2 , q -ED N ( µ , C ) reduces to N N ( µ , C ) and q -EP (0 , C ) becomes GP (0 , C ) . When q ∈ (0 , 2) , q -EP (0 , C ) lends flexibility to modeling functional data with more regularization than GP. In practice, q = 1 is often adopted for faster posterior convergence [1, 21] and the capability of preserving inhomogeneous features (rough functional data, edges in image, etc).

The covariance kernel C is associated with a Hilbert-Schmidt (HS) integral operator T C : L 2 (Ω) → L 2 (Ω) , u ( · ) ↦→ ∫ Ω C ( · , x ′ ) u ( x ′ ) µ ( d x ′ ) which has eigen-pairs { λ ℓ , ϕ ℓ ( · ) } ∞ ℓ =1 such that for ∀ ℓ ∈ N , T C ϕ ℓ ( x ) = ϕ ℓ ( x ) λ ℓ and ∥ ϕ ℓ ∥ 2 = 1 . Assume T C is trace-class, i.e. tr( T C ) := ∑ ∞ ℓ =1 λ ℓ &lt; ∞ . Theorem 3.4 of [24] presents a series representation of Q-EP similar to GP and the Besov process [8].

Theorem 2.1 (Karhunen-Loéve) . If u ( · ) ∼ q -EP (0 , C ) with a trace-class HS operator T C having eigen-pairs { λ ℓ , ϕ ℓ ( · ) } ∞ ℓ =1 , then we have the following series representation for u ( x ) :

<!-- formula-not-decoded -->

where E[ u ℓ ] = 0 and Cov( u ℓ , u ℓ ′ ) = λ ℓ δ ℓℓ ′ with Dirac function δ ℓℓ ′ = 1 if ℓ = ℓ ′ and 0 otherwise. Moreover, we have E[ ∥ u ( · ) ∥ 2 2 ] = ∑ ∞ ℓ =1 E[ u 2 ℓ ] = tr( T C ) &lt; ∞ .

## 2.2 Bayesian Regression with Q-EP Priors

Given X N × d = { x n } N n =1 and y N × 1 = { y n } N n =1 , we consider the Bayesian regression model:

<!-- formula-not-decoded -->

Li et al. [24, Theorem 3.5] show that the posterior (predictive) distribution is analytically tractable when both the prior and the likelihood are q -exponential.

Theorem 2.2. For the regression model (3) , the posterior distribution of u ( x ∗ ) at x ∗ is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 1: Contrasting Q-EP ( q = 1 . 0 , middle row) with GP ( q = 2 . 0 , bottom row) against the truth (top row) for modeling function values and derivatives of Rosenbrock (left) and Rastrigin (right).

<!-- image -->

## 2.3 Modeling with Derivative Information

Let u ∼ q -EP (0 , C ) . Denote the function and its derivatives by ˜ u = ( u, ∂ ∂ x u, · · · , ∂ k ∂ x k u ) up to order k . Because linear operation preserves elliptic distributions [18, 10], ˜ u ∼ q -EP (0 , ˜ C ) is also a Q-EP if C in Definition 2 is differentiable up to order k , where the augmented kernel, ˜ C , has a structure illustrated in Table A.1. For example, the (1 , 2) block of ˜ C is interpreted as Cov( u ( x ) , ∂ ∂ x ′ u ( x ′ )) = ∂ ∂ x ′ C ( x , x ′ ) . For solving the second-order PDEs in Section 5, we could adopt Matérn kernel ( matern52 for ν = 5 / 2 ), C ( x , x ′ ) = σ 2 (1 + √ 5 r + 5 3 r 2 ) exp( - √ 5 r ) , r = √ ∑ d i =1 ( x i -x ′ i ) 2 /ρ 2 i , for which the process is twice differentiable in the mean-square sense.

To prepare for solving PDEs, we define ∥ · ∥ s,q for u based on (2) with a smoothness parameter s &gt; 0 and an integrability parameter q ≥ 1 [22, 7]: ∥ u ( · ) ∥ s,q = (∑ ∞ ℓ =1 ℓ τ q ( s ) q | u ℓ | q ) 1 q , τ q ( s ) = s d + 1 2 -1 q . Consider the Banach space B s,q (Ω) := { u : Ω → R | ∥ u ( · ) ∥ s,q &lt; ∞} . If q = 2 and { ϕ ℓ } ∞ ℓ =1 form the Fourier basis, then B s, 2 (Ω) reduces to the Sobolev space H s (Ω) . For u to be regular enough, we make the assumption on ˜ C so that ˜ u ∈ L q (Ω) almost surely by the following proposition [21].

Assumption 1. Suppose λ = { λ ℓ } ∞ ℓ =1 are eigenvalues of HS operator T ˜ C for the kernel ˜ C . We assume λ ∈ ℓ q 2 , i.e. ∥ λ ∥ q 2 q 2 = ∑ ∞ ℓ =1 λ q 2 ℓ &lt; ∞ .

Proposition 2.1. If ˜ u ( · ) ∼ q -EP (0 , ˜ C ) with a trace-class HS operator T ˜ C satisfying Assumption 1, then ˜ u ( · ) ∈ L q P ( R ∞ , L q (Ω)) := { ˜ u : Ω × R ∞ → R | E ( ∥ ˜ u ∥ q q ) &lt; ∞} and E [ ∥ ˜ u ( · ) ∥ q q ] = ∥ λ ∥ q 2 q 2 &lt; ∞ .

<!-- formula-not-decoded -->

Let ˜ U := ˜ u ( X ) N × (1+ kd ) = [ u ( X ) , ∂ ∂ x u ( X ) , · · · , ∂ k ∂ x k u ( X )] . For brevity, we denote D = 1 + kd . Then Y N × D and E N × D are the corresponding observations and errors respectively in the model (3). We apply this model to Rosenbrock ( f ( x ) = ∑ d i =1 [100( x i +1 -x 2 i ) 2 +(1 -x i ) 2 ] ) and Rastrigin ( f ( x ) = 10 d + ∑ d i =1 [ x 2 i -10 cos(2 πx i )] ) test functions with function values and their derivatives observed on a 20 × 20 grid, i.e. N = 400 , d = 2 , k = 1 . Figure 1 contrasts Q-EP ( q = 1 . 0 ) and GP ( q = 2 . 0 ) in predicting function and derivative values on the 50 × 50 grid. Q-EP outperforms GP in yielding a more accurate recovery, especially in the more challenging example of Rastrigin function.

Heuristically, the superiority of Q-EP in modeling derivatives over GP comes from its improved ability to handle inhomogeneous data with sharp variation. From the perspective of the L q norm of the gradient, the L 1 norm imposes stronger regularization than the L 2 norm on large values in ∇ x f ,

resulting in a better model for prediction. In Section 4 we will give a more rigorous justification. In the following, we take advantage of Q-EP's capability of modeling derivative information and apply it to solve PDEs.

## 3 Solving Partial Differential Equations with Q-EP

## 3.1 Bayesian Solver

Consider the following general PDE defined on a bounded domain Ω ⊂ R d :

<!-- formula-not-decoded -->

where D : B s,q (Ω) → L q (Ω) is a differential operator and B : B s,q ( ∂ Ω) → L q ( ∂ Ω) is a boundary operator with data f ∈ L q (Ω) and g ∈ L q ( ∂ Ω) . Here we assume a sufficiently large smoothness index s &gt; 0 such that the PDE (4) is well-defined pointwise and has a unique strong solution [6]. Let Ω = Ω ∪ ∂ Ω . For convenience of exposition, we denote the joint operator as P = ( D , B ) : B s,q (Ω) → L q (Ω) , and the right-hand side function as h = ( f, g ) ∈ L q (Ω) .

A set of collocation points X = { x n } N n =1 consists of N d interior points X d = { x 1 , · · · , x N d ∈ Ω } and N b boundary points X b = { x N d +1 , · · · , x N ∈ ∂ Ω } , i.e. X = X d ∪ X b , and N = N d + N b . Regarding the evaluation of P ( u ) on X , we make the following assumption so that we can properly define the likelihood model.

Assumption 2. There exists a differentiable function P : R D → R such that P ( u )( x ) = P (˜ u ( x )) . And further there is a constant C &gt; 0 such that ∥∇ P ∥ ≤ C .

Then P ( u )( X ) becomes a nonlinear function of ˜ u ( X ) , denoted as P (˜ u ( X )) = P ( u )( X ) . Let h = h ( X ) . The probabilistic solver seeks to obtain ˜ U = ˜ u ( X ) based on observations ( P (˜ u ( X )) , h ) .

Even if we model ˜ u ( X ) ∼ q -ED(˜ u , S ) with ˜ u , S to be specified in (8) in Section 3.2, the nonlinear mapping P would not render P (˜ u ( X )) another q -ED random variable. Therefore, to properly define the likelihood model, we propose the following distribution propagation by linearizing P :

<!-- formula-not-decoded -->

where the Taylor expansion of P is about ˜ u 0 , which can be chosen as ˜ u n -1 from the previous training epoch or simply ˜ u , and δ &gt; 0 is a small nugget to ensure positive definiteness of Γ .

Let Y N × 1 := P (˜ u ( X )) . The Q-EP solver aims to minimize the discrepancy E = Y -h . The potential (negative log-likelihood) function, Φ : L q (Ω) × R N → R , can then be defined:

<!-- formula-not-decoded -->

where φ ( r ; Γ , N ) := -1 2 log | Γ | + N 2 ( q 2 -1 ) log r -1 2 r q 2 . Under Assumption 2, Φ is Lipschitz continuous in ˜ u , which is used in the convergence theorem in Section 4.

Proposition 3.1. Suppose that the PDE mapping P satisfies Assumption 2. Let q ∈ (0 , 2] . Then for every r &gt; 0 , there exists L = L ( r ) &gt; 0 such that for every Y ∈ R N and for all ˜ u 1 , ˜ u 2 ∈ L q (Ω) with max {∥ ˜ u 1 ∥ q , ∥ ˜ u 2 ∥ q } &lt; r , | Φ(˜ u 1 ; Y ) -Φ(˜ u 2 ; Y ) | ≤ L ∥ ˜ u 1 -˜ u 2 ∥ q .

<!-- formula-not-decoded -->

Therefore, the Bayesian model for the solution u to (4) can be summarized as

<!-- formula-not-decoded -->

Our goal is to infer the posterior p ( ˜ U | Y ) ∝ p ( Y | ˜ U ) p ( ˜ U ) . Note that because the extended function ˜ u ( X ) enters the likelihood model in a nonlinear way, Theorem 2.2 does not apply. In the following, we solve the inference problem using variational Bayes.

## 3.2 Variational Inference

We approximate the posterior p ( ˜ U | Y ) with some variational distribution q ( ˜ U ) using the variational Bayes method, which aims to minimize the Kullback-Leibler divergence KL( q ( ˜ U ) ∥ p ( ˜ U | Y )) . Because log p ( Y ) = KL( q ( ˜ U ) ∥ p ( ˜ U | Y )) + L ( q ( ˜ U )) , it reduces to maximizing the lower bound L ( q ( ˜ U )) . The sparse variational approximation [46, 47] is adopted by introducing inducing points ˜ X ∈ R M × d with their function values ˜ V = ˜ u ( ˜ X ) ∈ R M × D .

With the variational distribution for inducing values q ( ˜ V ) ∼ q -ED MD ( µ , Σ ) , the marginal variational distribution q ( ˜ U ) can be obtained as [17, 30]

<!-- formula-not-decoded -->

The final evidence lower bound (ELBO) L ∗ ( q ( ˜ U )) is (Refer to Section A.1 for more details.)

<!-- formula-not-decoded -->

The variational solution q ( ˜ U ) can be obtained by maximizing the ELBO (9) with respect to the variational parameters ( µ , Σ , ˜ X ) and hyper-parameters in the kernel C . By introducing the M inducing points, the overall computational complexity is reduced from O ( N 3 ) to O ( NM 2 ) [47].

## 3.3 Bayesian Inverse Problems

The above Bayesian framework can be readily extended to solve inverse problems. The following adaptation enables us to obtain both forward and inverse PDE solutions simultaneously.

Suppose that the PDE (4) contains a quantity of interest, a ( x ) , which could appear in the differential equation D or as part of the boundary condition B . The task of Bayesian inverse problems is to find a true solution, a † , with proper UQ based on observations. Suppose a is differentiable enough and we denote ˜ a = ( a, ∂ ∂ x a, · · · , ∂ k ′ ∂ x k ′ a ) to the order k ′ ≤ k . Now the joint operator P applies to both u and a , which produces a nonlinear function of ˜ u ( X ) , ˜ a ( X ) when evaluated on X , denoted as P (˜ u ( X ) , ˜ a ( X )) = P ( u, a )( X ) .

In addition, there is an observation operator O such that observations, O ( u )( X ) = u ( X o ) , are obtained on some set of N o observation points, X o ⊂ Ω with | X o | = N o . This can be achieved by solving (4) with true a † in simulations or simply modeling measurement data as noisy realization of (4) in real-world applications. Let ˜ N = N + N o . We supplement the joint equation operator P with the observation operator O to form an augmented operator ˜ P = ( P , O ) . Therefore, we have ˜ Y ˜ N × 1 = ˜ P ( u, a )( X ) = ˜ P (˜ u ( X ) , ˜ a ( X )) = [ P (˜ u ( X ) , ˜ a ( X )) T , O (˜ u ( X )) T ] T . Similarly, we augment the right-hand side data h with observed u ( X o ) to make ˜ h ˜ N × 1 = [ h T , u ( X o ) T ] T .

If we model ˜ a ( X ) using q (˜ a ( X )) ∼ q -ED(˜ a , S a ) with variational mean and covariance ˜ a , S a respectively, then q (˜ u ( X )) q (˜ a ( X )) propagates through the PDE dynamics similarly as in (5). Finally, we summarize the Bayesian inverse model for a ( x ) as follows:

<!-- formula-not-decoded -->

where ˜ a 0 can be similarly chosen as ˜ a n -1 from the previous training epoch or simply ˜ a . The variational Bayes procedure in Section 3.2 can be modified accordingly to obtain the variational solution of ˜ a ( X ) | ˜ Y . Meanwhile, we obtain the variational solution of ˜ u ( X ) | ˜ Y as a byproduct.

## 4 Convergence Theorem

In this section, we study the posterior contraction of the Bayesian model (7) in the infinite data limit. Similar theory can be developed for the model (10). We focus on q ∈ [1 , 2] and leave the technically more challenging case q ∈ (0 , 1) to future study. For brevity, we denote u = ˜ u , C = ˜ C , and n = N .

Consider the separable Banach space X = ( L q (Ω) , ∥ · ∥ q ) ⊃ ( B s,q (Ω) , ∥ · ∥ s,q ) for s &gt; d ( 2 q -1 2 ) , and Y = R n . Define the concentration function of Q-EP measure Π at u = u † as

<!-- formula-not-decoded -->

Let P ( n ) u be the measure of the observations Y ( n ) on ( Y , B , µ 0 ) having density p u and corresponding potential function Φ( u ; · ) with respect to the Lebesgue measure µ 0 , i.e. d P ( n ) u dµ 0 ( Y ) = p u ( Y ) ∝ exp( -Φ( u ; Y )) . Define the Hellinger distance as d 2 n,H ( u, u ′ ) = ∫ ( √ p u - √ p u ′ ) 2 dµ 0 . We have the following posterior contraction theorem.

Theorem 4.1 (Posterior Contraction) . Let u ∼ q -EP (0 , C ) with C satisfying Assumption 1 in Θ := L q (Ω) and P ( n ) u is the measure of Y ( n ) parameterized by u with PDE (4) satisfying Assumption 2. If the true value u † ∈ Θ is in the support of u , and ε n satisfies the rate equation φ u † ( ε n ) ≤ nε 2 n with ε n ≥ n -1 2 , then there exists Θ n ⊂ Θ such that Π n ( u ∈ Θ n : d n,H ( u, u † ) ≥ M n ε n | Y ( n ) ) → 0 in P ( n ) u † -probability for every M n →∞ .

<!-- formula-not-decoded -->

Denote a ∧ b = min { a, b } , a ∨ b = max { a, b } , and x + = x ∨ 0 . By solving the inequality φ u † ( ε n ) ≤ nε 2 n for the minimal ε n , we obtain the posterior contraction rate as follows.

Theorem 4.2 (Contraction Rate) . Let u ∼ q -EP (0 , C ) with C satisfying Assumption 1 in Θ := L q (Ω) . The rest of the settings are the same as in Theorem 4.1. If the true value u † ∈ B s † ,q † (Ω) with s † &gt; s ′ + ( d q † -d q ) + , s ′ = d q -d 2 , and q † , q ∈ [1 , 2] , then we have the rate of the posterior contraction as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark 3. If we set the smoothness parameter s = s † + d q -( d q † -d q ) + , and allow the integrability parameter q ≤ q † , then the contraction rate ε n is maximized as ε † n = n -1 2+ d s † -s ′ &gt; n -1 2 . Note that this optimal rate is achieved regardless of the value of modeling regularization parameter q as long as q ≤ q † . This implies that when modeling inhomogeneous data and derivative information, under-smoothing (with smaller regularization parameter q ) is preferred to over-smoothing. When the true integrability q † is at least L 1 , setting q = 1 guarantees the fastest convergence of the posterior. In the following section, we will demonstrate such optimal choice with various numerical examples.

## 5 Numerical Experiments

In this section, we test variational Q-EP for solving and learning PDEs. Using the Eikonal equation,

Burgers' equation, a nonlinear elliptic equation (Section C.3), and an inverse problem involving the Darcy flow, we investigate the proposed method for a variety of different q s. We include PINN [37] and B-PINN [49] as baselines and the variational GP as a special case with q = 2 and mainly compare them using the relative error of the estimated solution ˆ u with reference to the true solution u † : RLEp = ∥ u † -ˆ u ∥ p ∥ u † ∥ p for p = 1 , 2 , ∞ . Q-EP with q = 1 outperforms PINN and variational GP in all cases and attains the best results in most comparisons. The numerical evidence also supports that Q-EP converges the fastest with the optimal choice of q = 1 .

Figure 2: Solving Eikonal equation (12) using high-resolution finite difference method (upper left), PINN (lower left), Q-EPs (middle two in upper row: q = 0 . 8 ; right two in upper row: q = 1 . 0 ; middle two in lower row: q = 1 . 5 ), and GP (right two in lower row: q = 2 . 0 ) respectively. Blue crosses are learned inducing points.

<!-- image -->

In most experiments, we choose the interior collocation points on a 24 × 24 mesh grid and the corresponding 100 boundary collocation points unless otherwise stated. For the sparse variational inference, M = 256 inducing points are randomly initialized and learned by optimizing the ELBO (9). The kernel C of Q-EP/GP is chosen to be matern52 with the hyperparameters, e.g. the correlation strength, automatically tuned in the python package GPyTorch [11] implemented based on PyTorch . PINN is configured to have neural network parameters (weights and biases) of similar size to the variational Q-EP model. The computer codes are publicly available at https://github.com/ lanzithinking/Diff\_QEP .

## 5.1 Eikonal Equation

First, we consider the following regularized Eikonal equation on Ω = [0 , 1] 2 also considered in [6]:

<!-- formula-not-decoded -->

where f ≡ 1 and ε = 0 . 1 . Based on the set-up in Section 3.1, we define the nonlinear function D ( u, d 1 u, d 2 u, d 2 1 u, d 2 2 u ) = ( d 1 u ) 2 + ( d 2 u ) 2 -ε ( d 2 1 u + d 2 2 u ) . Then the observations are Y =

<!-- formula-not-decoded -->

obtain N d = 24 2 interior and N b = 100 boundary collocation points. Then we apply the variational Bayes in Section 3.2 to solve (12) with M = 256 inducing points. Though not required for convergence, we train each algorithm for 5000 iterations for fair comparison (PINN and B-PINN need much more training epochs than Q-EP solvers to converge).

Figure 2 compares solutions and uncertainty estimates generated by a variety of Q-EP solvers for q = 0 . 8 , 1 . 0 , 1 . 5 and 2 . 0 (GP) respectively. We solve the equation using a highly-resolved finite difference method with the Cole-Hopf transformation [6] and use it as the true solution for comparison (upper left). We notice that the solution by PINN (lower left) is much worse and UQ is not available. All Q-EP solvers ( q &lt; 2 ) produce better solutions than GP ( q = 2 ) which does not precisely characterize the pyramid feature of the true solution. Only two solvers with q = 0 . 8 and 1 . 0 yield solutions that correctly match the range of true solution. GP also manifests higher uncertainty in its generated solution. Table C.1 further verifies the superior accuracy of Q-EP solvers compared to GP and PINN (B-PINN) in terms of multiple error metrics including MAE, MSE and RLE by repeating the experiments for 10 times with different random seeds. In this example, Q-EP solver with q = 1 . 0 attains the result comparable to the most accurate solution. Note that with similar size of collocation points, our best results (4.43e-2 in L 2 error and 1.03e-2 in L ∞ error) are comparable to those reported (1.64e-2 in L 2 error and 7.76e-2 in L ∞ error) in [6] from which UQ is absent.

## 5.2 Burgers' Equation

Next, we test our Q-EP solvers on the Burgers' equation [6] with ν = 0 . 1 :

Figure 3: Comparing convergence of Q-EP (left three in top row: q = 1 . 0 ), GP (left three in middle row: q = 2 . 0 ) and PINN (left three in bottom row) in solving Burgers' equation (13), with the right column illustrating the error reducing in L 1 norm (top), L 2 norm (middle), and L ∞ norm (bottom) respectively. Blue crosses are learned inducing points. Shaded regions are standard errors based on 10 repeated experiments.

<!-- image -->

<!-- formula-not-decoded -->

We use the same experiment setup as above. Figure C.1 compares the solutions by a highly-resolved finite difference method (upper left, treated as true solution for comparison purpose), PINN (lower left), Q-EP (right three in upper row: q = 1 . 0 ), and GP (right three in lower row: q = 2 . 0 ). Q-EP is about one order of magnitude more accurate than PINN and GP, which can be verified from the pointwise error plots and Table 1. The plots of posterior standard deviation on the right column also indicate meaningful uncertainty in the middle area around x = 0 where the shock is difficult to resolve. Q-EP still achieves one order of magnitude lower uncertainty compared with GP.

Table 1: Comparing accuracy of various solvers for Burgers' equation (13) in terms of mean absolute error (MAE), mean squared error (MSE), and relative errors in L 1 norm (RLE-1), L 2 norm (RLE-2), and L ∞ norm (RLE-∞ ) respectively. Result in each cell are averaged over 10 experiments with different random seeds; values after ± are standard deviations of these repeated experiments.

| Model ( q )                         | MAE               | MSE                                                                                                               | RLE-1                                                                            | RLE-2                                                                                                                           | RL- ∞                                                                                           |
|-------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| PINN B-PINN 0.5 0.8 1.0 1.2 1.5 2.5 | 5.81e-2 ± 1.03e-4 | 7.12e-3 ± 3.96e-4 1.67e-3 ± 1.58e-3 3.50e-2 ± 1.59e-3 ± 1.77e-4 ± 2.16e-4 5.92e-4 ± 1.31e-3 ± 1.23e-3 1.08e-2 ± ± | 0.1508 ± 0.0017 0.0785 ± 0.2018 ± 0.0405 ± 0.0242 ± 0.0485 ± 0.0684 ± 0.1848 ± ± | 0.1896 ± 0.0052 0.0833 ± 0.0409 0.2056 ± 0.3764 0.0434 ± 0.0804 0.0266 ± 0.0140 0.0522 ± 0.0168 0.0754 ± 0.0316 0.2068 ± 0.1118 | 0.2842 ± 0.0071 0.1327 ± 0.0446 0.2177 ± 0.3838 0.0529 ± 0.0924 0.0324 ± 0.0145 0.0560 ± 0.0181 |
|                                     | 2.94e-2 ± 1.57e-2 |                                                                                                                   | 0.0420                                                                           |                                                                                                                                 |                                                                                                 |
|                                     | 7.77e-2 ± 7.09e-2 | 7.15e-2                                                                                                           | 0.3751                                                                           |                                                                                                                                 |                                                                                                 |
|                                     | 1.56e-2 ± 4.96e-3 | 5.25e-3                                                                                                           | 0.0780                                                                           |                                                                                                                                 |                                                                                                 |
|                                     | 9.33e-3 ± 1.76e-4 |                                                                                                                   | 0.0128                                                                           |                                                                                                                                 |                                                                                                 |
|                                     | 1.87e-2 ± 3.23e-4 | 4.06e-4                                                                                                           | 0.0151                                                                           |                                                                                                                                 |                                                                                                 |
|                                     | 2.64e-2 ± 8.58e-4 |                                                                                                                   | 0.0262                                                                           |                                                                                                                                 | 0.0854 ± 0.0430                                                                                 |
| 2.0(Gaussian)                       | 7.13e-2 ± 8.68e-3 | 1.28e-2                                                                                                           | 0.0908                                                                           |                                                                                                                                 | 0.2376 ± 0.1456                                                                                 |
|                                     | 2.26e-1 ± 3.20e-1 | 2.49e-1 6.04e-1                                                                                                   | 0.5879 0.8296                                                                    | 0.6669 ± 0.9576                                                                                                                 | 0.7558 ± 1.0521                                                                                 |

Table 1 compares the accuracy of solutions in terms of MAE, MSE and RLE, for which Q-EP with q = 1 . 0 attains the most accurate one, verifying its best performance. The error for Q-EP models does not decrease monotonically with q &gt; 0 , but reaches the lowest at q = 1 . 0 . To further investigate the convergence, we plot solutions by Q-EP (top row: q = 1 . 0 ), GP (middle row: q = 2 . 0 ), and PINN (bottom row) in Figure 3 at 10 (first column), 100 (second column), and 1000 (third column) iterations respectively. We can tell that Q-EP with q = 1 . 0 converges the fastest to the best estimate, which is confirmed by the error-reducing plots on the right column. Note that in this example, our algorithm does not need random perturbation of mesh grid as required by [6].

<!-- image -->

0.0

0.2

0.4

0.6

8'0

1.0

0.2

0.4

9°0

0.8

0.2

0.4

0.6

0.8

0.2

0.4

0.6

0.8

0.2

0.4

90

0.8

Figure 4: Solving inverse Darcy flow (14) using PINN (second column), Q-EP (right three in upper row: q = 1 . 0 ), and GP (right three in lower row: q = 2 . 0 ) respectively. Upper left: true inverse solution a † ; lower left: fine-resolution finite element solution u † to (14) with a † . Blue crosses are learned inducing points, and red dots indicate locations of observations.

## 5.3 Inverse Darcy Flow

Now we consider an inverse problem that involves the following Darcy flow:

<!-- formula-not-decoded -->

where the true coefficient such that exp( a † ( x )) = exp(sin(2 πx 1 )+sin(2 πx 2 ))+exp( -sin(2 πx 1 ) -sin(2 πx 2 )) [6] is plotted in the upper left panel of Figure 4. We generate data by solving (14) with a † on a mesh 80 × 80 to obtain u † using the finite element method, illustrated in the lower left panel in Figure 4. On a (coarser) mesh N d = 20 × 20 used for inference, we randomly select N o = 100 points X o in Ω and obtain observations as u † ( X o ) + ε with noise ε ∼ N (0 , γ 2 I N o ) for γ = 10 -3 .

We solve the inverse problem of finding a in (14) given these observations. With N d = 400 interior and N b = 84 boundary collocation points and M = 256 inducing points, we train Q-EP solvers for 2000 iterations. Figure 4 illustrates the forward PDE solution u (third column), the inverse solution a (forth column), and the uncertainty of a (rightmost column). Compared with the true a † , Q-EP ( q = 1 . 0 ) recovers a more faithfully than GP ( q = 2 . 0 ) and PINN (upper in the second column). Meanwhile, Q-EP also generates the solution u much closer to that by the finite element method, as shown in the lower left panel. Higher uncertainty is observed by Q-EP ( q = 1 . 0 ) around the corners with less data (Figure C.3), reflecting the configuration of observations. If run for longer (e.g. 5000) iterations, PINN may improve its forward solution, but still gets a poor inverse solution (Figure C.4).

Table C.5 compares the relative errors (RLEs) of forward and inverse solutions. Q-EP with q = 1 . 0 also achieves the best or comparable solutions. Here we emphasize that all the results are based on only N o = 100 observations from one solution, as opposed to the thousands of PDE solutions typically used in operator learning algorithms. Figure C.5 compares the inverse solutions of Q-EP with those of GP in increasingly finer meshes. If we view training in finer mesh as a process to see more data, Q-EP ( q = 1 . 0 ) has already converged fast to better estimates in coarser mesh, leaving less room to improve compared to GP ( q = 2 . 0 ).

## 6 Conclusion

In this paper, we propose variational Q-EP to solve and learn PDEs. We advocate Q-EP with q = 1 . 0 over GP ( q = 2 . 0 ) for modeling derivative information and hence a better probabilistic solver for PDEs. The fastest convergence at q = 1 . 0 is theoretically justified and empirically verified using two nonlinear forward PDE problems and an inverse Darcy flow problem.

One of the limitations might be the variational inference adopted in this paper. The highly nonlinear nature of some PDEs imposes challenges on the quality of variational approximation to the resulting posterior, which in turn may undermine both the solution accuracy and the associated UQ. A possible remedy could be more flexible inference methods, such as normalizing flow [41, 36].

## References

- [1] Sergios Agapiou, Masoumeh Dashti, and Tapio Helin. Rates of contraction of posterior distributions based on p-exponential priors. Bernoulli , 27(3):1616 - 1642, 2021.
- [2] Frank Aurzada. On the lower tail probabilities of some random sequences in lp. Journal of Theoretical Probability , 20(4):843-858, Dec 2007.
- [3] Ricardo Baptista, Edoardo Calvello, Matthieu Darcy, Houman Owhadi, Andrew M. Stuart, and Xianjin Yang. Solving roughly forced nonlinear pdes via misspecified kernel methods and neural networks. arXiv preprint arXiv:2501.17110 , 01 2025.
- [4] Ben Calderhead, Mark Girolami, and Neil Lawrence. Accelerating bayesian inference over nonlinear differential equations with gaussian processes. In D. Koller, D. Schuurmans, Y . Bengio, and L. Bottou, editors, Advances in Neural Information Processing Systems , volume 21. Curran Associates, Inc., 2008.
- [5] Zhi Chang, Chukwudi Obite, Shuang Zhou, and Shiwei Lan. Deep q-exponential processes. In Proceedings of the 7th Symposium on Advances in Approximate Bayesian Inference . AABI, 10 2025.
- [6] Yifan Chen, Bamdad Hosseini, Houman Owhadi, and Andrew M. Stuart. Solving and learning nonlinear pdes with gaussian processes. Journal of Computational Physics , 447:110668, 2021.
- [7] Masoumeh Dashti, Stephen Harris, and Andrew Stuart. Besov priors for bayesian inverse problems. Inverse Problems and Imaging , 6(2):183-200, may 2012.
- [8] Masoumeh Dashti and Andrew M. Stuart. The Bayesian Approach to Inverse Problems. In Roger Ghanem, David Higdon, and Houman Owhadi, editors, Handbook of Uncertainty Quantification , pages 311-428. Springer International Publishing, Cham, 2017.
- [9] Weinan E and Bing Yu. The deep ritz method: A deep learning-based numerical algorithm for solving variational problems. Communications in Mathematics and Statistics , 6(1):1-12, Mar 2018.
- [10] K. Fang and Y.T. Zhang. Generalized Multivariate Analysis . Science Press, 1990.
- [11] Jacob Gardner, Geoff Pleiss, Kilian Q Weinberger, David Bindel, and Andrew G Wilson. Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 31. Curran Associates, Inc., 2018.
- [12] S. Ghosal and A.W. van der Vaart. Convergence rates of posterior distributions for non-i.i.d. observations. Annals of Statistics , 35(1):192-223, 2007. MR2332274.
- [13] Oliver Hamelijnck, Arno Solin, and Theodoros Damoulas. Physics-informed variational statespace gaussian processes. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [14] Marc Harkonen, Markus Lange-Hegermann, and Bogdan Raita. Gaussian process priors for systems of linear partial differential equations with constant coefficients. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 12587-12615. PMLR, 23-29 Jul 2023.
- [15] Markus Heinonen, Cagatay Yildiz, Henrik Mannerström, Jukka Intosalmi, and Harri Lähdesmäki. Learning unknown ODE models with Gaussian processes. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pages 1959-1968. PMLR, 10-15 Jul 2018.
- [16] Philipp Hennig, Michael A. Osborne, and Mark Girolami. Probabilistic numerics and uncertainty in computations. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences , 471(2179):20150142, July 2015.

- [17] James Hensman, Alexander Matthews, and Zoubin Ghahramani. Scalable Variational Gaussian Process Classification. In Guy Lebanon and S. V. N. Vishwanathan, editors, Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics , volume 38 of Proceedings of Machine Learning Research , pages 351-360, San Diego, California, USA, 09-12 May 2015. PMLR.
- [18] Mark E. Johnson. Multivariate Statistical Simulation , chapter 6 Elliptically Contoured Distributions, pages 106-124. Probability and Statistics. John Wiley &amp; Sons, Ltd, 1987.
- [19] George Em Karniadakis, Ioannis G. Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics , 3(6):422-440, May 2021.
- [20] Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to pdes. Journal of Machine Learning Research , 24(89):1-97, 2023.
- [21] Shiwei Lan, Mirjeta Pasha, Shuyi Li, and Weining Shen. Spatiotemporal besov priors for bayesian inverse problems. arXiv preprint arXiv:2306.16378 , 06 2023.
- [22] Matti Lassas, Eero Saksman, and Samuli Siltanen. Discretization-invariant bayesian inversion and besov space priors. Inverse Problems and Imaging , 3(1):87-122, 2009.
- [23] Matti Lassas and Samuli Siltanen. Can one use total variation prior for edge-preserving Bayesian inversion? Inverse Problems , 20(5):1537, 2004.
- [24] Shuyi Li, Michael O' Connor, and Shiwei Lan. Bayesian learning via q-exponential process. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Proceedings of the 37th Conference on Neural Information Processing Systems , volume 36, pages 72867-72887. Curran Associates, Inc., 2023.
- [25] Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations , 2021.
- [26] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature Machine Intelligence , 3(3):218-229, March 2021.
- [27] Rui Meng and Xianjin Yang. Sparse gaussian processes for solving nonlinear pdes. Journal of Computational Physics , 490:112340, 2023.
- [28] Roberto Molinaro, Yunan Yang, Björn Engquist, and Siddhartha Mishra. Neural inverse operators for solving PDE inverse problems. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 25105-25139. PMLR, 23-29 Jul 2023.
- [29] Carlos Mora, Amin Yousefpour, Shirin Hosseinmardi, and Ramin Bostanabad. A gaussian process framework for solving forward and inverse problems involving nonlinear partial differential equations. Computational Mechanics , 75(4):1213-1239, 2025.
- [30] Chukwudi Paul Obite, Zhi Chang, Keyan Wu, and Shiwei Lan. Bayesian regularization of latent representation. In The Thirteenth International Conference on Learning Representations , 2025.
- [31] Bernt Øksendal. Stochastic Differential Equations . Springer Berlin Heidelberg, 2003.
- [32] Houman Owhadi. Bayesian numerical homogenization. Multiscale Modeling and Simulation , 13(3):812-828, January 2015.
- [33] Houman Owhadi. Multigrid with rough coefficients and multiresolution operator decomposition from hierarchical information games. SIAM Review , 59(1):99-149, January 2017.

- [34] Houman Owhadi and Clint Scovel. Operator-Adapted Wavelets, Fast Solvers, and Numerical Homogenization: From a Game Theoretic Approach to Numerical Approximation and Algorithm Design . Cambridge Monographs on Applied and Computational Mathematics. Cambridge University Press, 2019.
- [35] Houman Owhadi and Lei Zhang. Gamblets for opening the complexity-bottleneck of implicit schemes for hyperbolic and parabolic odes/pdes with rough coefficients. Journal of Computational Physics , 347:99-128, 2017.
- [36] George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research , 22(57):1-64, 2021.
- [37] M. Raissi, P. Perdikaris, and G.E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics , 378:686-707, 2019.
- [38] Maziar Raissi, Paris Perdikaris, and George Em Karniadakis. Numerical gaussian processes for time-dependent and nonlinear partial differential equations. SIAM Journal on Scientific Computing , 40(1):A172-A198, January 2018.
- [39] J. O. Ramsay, G. Hooker, D. Campbell, and J. Cao. Parameter estimation for differential equations: a generalized smoothing approach. Journal of the Royal Statistical Society: Series B (Statistical Methodology) , 69(5):741-796, 2007.
- [40] Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning . The MIT Press, 2005.
- [41] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning , volume 37 of Proceedings of Machine Learning Research , pages 1530-1538, Lille, France, 07-09 Jul 2015. PMLR.
- [42] Leonid I. Rudin, Stanley Osher, and Emad Fatemi. Nonlinear total variation based noise removal algorithms. Physica D: Nonlinear Phenomena , 60(1):259-268, 1992.
- [43] Michael Schober, David Duvenaud, and Philipp Hennig. Probabilistic ode solvers with rungekutta means. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems , volume 27. Curran Associates, Inc., 2014.
- [44] Justin Sirignano and Konstantinos Spiliopoulos. Dgm: A deep learning algorithm for solving partial differential equations. Journal of Computational Physics , 375:1339-1364, 2018.
- [45] John Skilling. Bayesian solution of ordinary differential equations. In C. Ray Smith, Gary J. Erickson, and Paul O. Neudorfer, editors, Maximum Entropy and Bayesian Methods: Seattle, 1991 , pages 23-37. Springer Netherlands, Dordrecht, 1992.
- [46] Michalis Titsias. Variational learning of inducing variables in sparse gaussian processes. In David van Dyk and Max Welling, editors, Proceedings of the Twelth International Conference on Artificial Intelligence and Statistics , volume 5 of Proceedings of Machine Learning Research , pages 567-574, Hilton Clearwater Beach Resort, Clearwater Beach, Florida USA, 16-18 Apr 2009. PMLR.
- [47] Michalis Titsias and Neil D. Lawrence. Bayesian gaussian process latent variable model. In Yee Whye Teh and Mike Titterington, editors, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics , volume 9 of Proceedings of Machine Learning Research , pages 844-851, Chia Laguna Resort, Sardinia, Italy, 13-15 May 2010. PMLR.
- [48] C.R. Vogel and M.E. Oman. Fast, robust total variation-based reconstruction of noisy, blurred images. IEEE Transactions on Image Processing , 7(6):813-824, 1998.
- [49] Liu Yang, Xuhui Meng, and George Em Karniadakis. B-pinns: Bayesian physics-informed neural networks for forward and inverse pde problems with noisy data. Journal of Computational Physics , 425:109913, 2021.

## Technical Appendices for 'Solving and Learning Partial Differential Equations with Variational Q-Exponential Process"

## A Calculations

Table A.1: The structure of kernel ˜ C with derivatives.

| Cov( · , · )                            | u ( x ′ )                                                 | ∂ ∂ x ′ u ( x ′ )                                                               | ∂ 2 ∂ ( x ′ ) 2 u ( x ′ )                                                                           |
|-----------------------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| u ( x ) ∂ ∂ x u ( x ) ∂ 2 ∂ x 2 u ( x ) | C ( x , x ′ ) ∂ ∂ x C ( x , x ′ ) ∂ 2 ∂ x 2 C ( x , x ′ ) | ∂ ∂ x ′ C ( x , x ′ ) ∂ 2 ∂ x ∂ x ′ C ( x , x ′ ) ∂ 3 ∂ x 2 ∂ x ′ C ( x , x ′ ) | ∂ 2 ∂ ( x ′ ) 2 C ( x , x ′ ) ∂ 3 ∂ x ∂ ( x ′ ) 2 C ( x , x ′ ) ∂ 4 ∂ x 2 ∂ ( x ′ ) 2 C ( x , x ′ ) |

## A.1 Variational Lower Bound

Because log p ( Y ) = KL( q ( ˜ U ) ∥ p ( ˜ U | Y )) + L ( q ( ˜ U )) , it reduces to maximizing the lower bound L ( q ( ˜ U )) . The sparse variational approximation [46, 47] is adopted by introducing inducing points ˜ X ∈ R M × d with their function values ˜ V = ˜ u ( ˜ X ) ∈ R M × D . Hence the joint distribution of Y and ˜ U can be augmented by including ˜ V :

<!-- formula-not-decoded -->

where we have vec( ˜ V ) | ˜ X ∼ q -ED MD ( 0 , ˜ C MM ) and the conditional distribution

<!-- formula-not-decoded -->

Now we approximate the joint posterior p ( ˜ U , ˜ V | Y ) with the following variational distribution

<!-- formula-not-decoded -->

where the covariance Σ is of size MD × MD and can be chosen as a (block)-diagonal matrix for convenience. A standard variational Bayes procedure yields the variational bound:

<!-- formula-not-decoded -->

where the marginal variational distribution q ( ˜ U ) can be obtained as [17, 30]

<!-- formula-not-decoded -->

Denote by φ ( r ; C , N ) := -1 2 log | C | + N 2 ( q 2 -1 ) log r -1 2 r q 2 which is convex for q ∈ (0 , 2] . Let φ 0 ( r ) = φ ( r ; Γ , N ) and r ( Y ) = ( Y -h ) T Γ -1 ( Y -h ) be a quadratic form of random variable Y . Then log p ( Y | ˜ U ) = φ 0 ( r ( Y )) . Therefore, by Jensen's inequality, we can bound from below as

<!-- formula-not-decoded -->

Now we compute the K-L divergence KL ˜ V := KL( q ( ˜ V ) ∥ p ( ˜ V )) :

<!-- formula-not-decoded -->

Denote by r ( ˜ V ) = vec( ˜ V -µ ) T Σ -1 vec( ˜ V -µ ) . Then log q ( ˜ V ) = φ ( r ( ˜ V ); Σ , MD ) . From [Proposition A.1. of 24] we know that r q 2 ∼ χ 2 ( MD ) . Therefore

<!-- formula-not-decoded -->

Let φ 1 ( r ) := φ ( r ; ˜ C MM , MD ) . Then by Jensen's inequality

<!-- formula-not-decoded -->

Therefore, the final evidence lower bound (ELBO) L ∗ ( q ( ˜ U )) is

<!-- formula-not-decoded -->

## B Proofs

Notations : ≲ means 'less than or approximately equal to"; a n ≲ b n implies a n ≤ Cb n for some constant C &gt; 0 . ≍ means 'asymptotically equal to"; a n ≍ b n implies lim n →∞ a n b n = c for some constant c .

Proof of Proposition 2.1. Note r (˜ u ℓ ) q 2 = λ -q 2 ℓ | ˜ u ℓ | q ∼ χ 2 (1) for all ℓ ∈ N by Proposition A.1. of [24]. Denote χ 2 ℓ := λ -q 2 ℓ | ˜ u ℓ | q iid ∼ χ 2 (1) . Hence ∥ ˜ u ∥ q q = ∑ ∞ ℓ =1 λ q 2 ℓ χ 2 ℓ becomes an infinite mixture of chi-squared random variables whose density is analytically intractable. Yet we have

<!-- formula-not-decoded -->

if Assumption 1 holds. Thus it completes the proof.

Proof of Proposition 3.1. Based on (5) and (6), the potential Φ(˜ u ; Y ) = -φ ( r ; Γ , N ) where φ ( r ; Γ , N ) := -1 2 log | Γ | + N 2 ( q 2 -1 ) log r -1 2 r q 2 is convex in r if q ∈ (0 , 2] , and r = r (˜ u ) with

<!-- formula-not-decoded -->

Since convex function is Lipschitz continuous over compact domain, it suffices to prove that r (˜ u ) is bounded (both bounds achievable) and Lipschitz.

Note that r (˜ u ) ≤ δ -1 ∥ A ˜ u -b ∥ 2 2 ≤ δ -1 ( ∥ A ∥∥ ˜ u ∥ + ∥ b ∥ ) 2 where ˜ u represents a PDE solution in the compact domain Ω ⊂ R d and is therefore bounded by some M &gt; 0 . By Assumption 2, ∥ A ∥ ≤ C . Therefore, 0 ≤ r (˜ u ) ≤ δ -1 ( CM + ∥ b ∥ ) 2 .

On the other hand, we have the gradient of the quadratic form r (˜ u ) bounded as

<!-- formula-not-decoded -->

Hence r (˜ u ) is Lipschitz.

Lastly, there exists L 1 , L 2 &gt; 0 such that

<!-- formula-not-decoded -->

assuming the collocation points are uniformly sampled. Therefore, it completes the proof.

<!-- formula-not-decoded -->

According to Theorem 3.1 and Lemma 5.14 of [1], the following general contraction conditions hold for Q-EP and will be used in the proof of posterior contraction Theorem 4.1.

Theorem B.1. Let µ be a q -EP (0 , C ) measure satisfying Assumption 1 in the separable Banach space ( L q (Ω) , ∥ · ∥ q ) , where q ∈ [1 , 2] . Let u ∼ µ and the true parameter u † ∈ L q (Ω) . Assume ε n &gt; 0 such that φ u † ( ε n ) ≤ nε 2 n , where nε 2 n ≳ 1 . Then for any C &gt; 1 , there exists a measurable set B n ⊂ L q (Ω) and a constant R &gt; 0 depending on C and q , such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where N (4 ε n , B n , ∥ · ∥ q ) is the minimal number of ∥ · ∥ q -balls of radius 4 ε n to cover B n .

We need the following lemma to bound the Hellinger distance, Kullback-Leibler (K-L) divergence, and K-L variation to complete the proof of Theorem 4.1 [21].

Lemma B.1. Suppose the potential function Φ (6) satisfies Lipschitz continuity in u as in Proposition 3.1. Then we have

- d H ( p u , p u ′ ) ≲ ∥ u -u ′ ∥ q .
- K ( p u , p u ′ ) ≲ ∥ u -u ′ ∥ q .
- V ( p u , p u ′ ) ≲ ∥ u -u ′ ∥ 2 q .

Proof. First, we consider K-L divergence:

<!-- formula-not-decoded -->

by Proposition 3.1. Similarly, we have for K-L variation:

<!-- formula-not-decoded -->

Lastly, we bound the Hellinger distance:

<!-- formula-not-decoded -->

where the inequality holds for ∥ u -u ′ ∥ 2 q small enough.

Proof of Theorem 4.1. Based on [Theorem 1 of 12], it suffices to verify the following two conditions (the entropy condition (2.4), and the prior mass condition (2.5)) for some universal constants η, K &gt; 0 and sufficiently large k ∈ N ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the left side of (18) is logarithm of the minimal number of d n,H -balls of radius ξε/ 2 needed to cover a ball of radius ε around the true value u † ; B n ( u † , ε n ) = { u ∈ Θ : 1 n K ( u † , u ) ≤ ε 2 , 1 n V ( u † , u ) ≤ ε 2 } with K ( u † , u ) = K ( p u † , p u ) and V ( u † , u ) = V ( p u † , p u ) .

Since u ( · ) ∈ L q (Ω) satisfy conditions for Theorem B.1, there exists B n ⊂ L q (Ω) such that (15)(17) holds. Now we set Θ n = B n . For ∀ u, u ′ ∈ Θ n such that ∥ u ( · ) -u ′ ( · ) ∥ q ≤ ε n , we have d n,H ( u, u ′ ) ≲ ∥ u -u ′ ∥ q ≤ ε n by Lemma B.1. Therefore by (15) we have the following global entropy bound holds

<!-- formula-not-decoded -->

which is stronger than the local entropy condition (18).

Now by Lemma B.1 and (17), we have

<!-- formula-not-decoded -->

Then the prior mass condition (19) is satisfied because the numerator is bounded by 1. The proof is hence completed.

The following lemma studies the small ball probability in the concentration function (11) [21].

Lemma B.2 (Small ball probability) . Let Π be a q -EP (0 , C ) prior on B s ′ ,q (Ω) with s ′ &lt; s -d q . Then as ε → 0 , we have

<!-- formula-not-decoded -->

Proof. We can compute

<!-- formula-not-decoded -->

where P is the probability measure on the infinite product space ( L q (Ω)) ∞ . From the proof of Proposition 2.1 we know ∥ u ∥ q q = ∑ ∞ ℓ =1 λ q 2 ℓ χ 2 ℓ is an infinite mixture of χ 2 (1) random variables, so the condition of [Theorem 4.2 of 2] is trivially met and we have

<!-- formula-not-decoded -->

The second lemma gives an upper bound of the first term of the concentration function (11) [21].

Lemma B.3 (Decentering function) . Assume u † ∈ B s † ,q † (Ω) for some s † &gt; s ′ and q † ∈ [1 , 2] . Then as ε → 0 , we have the following bounds

(i) If q † ≥ q , we require s † &gt; s ′ :

<!-- formula-not-decoded -->

(ii) If q † &lt; q , we require s † &gt; s ′ -d q + d q † :

<!-- formula-not-decoded -->

Proof. We identify u † ∈ B s † ,q † with { u † ℓ } ∞ ℓ =1 ∈ ℓ q † ,τ q † ( s † ) . Then we follow [1] to approximate u † with h 1: L = { u † ℓ } ∞ ℓ =1 where u † ℓ ≡ 0 for all ℓ &gt; L . Note h 1: L ∈ ℓ q,τ q ( s ) for any finite L ∈ N . Identifying h 1: L with h ∈ B s,q (Ω) , we could get

<!-- formula-not-decoded -->

Therefore, to have ∥ h -u † ∥ s ′ ,q ≤ ε we let

<!-- formula-not-decoded -->

On the other hand, the infimum is less than ∥ h ∥ q s,q with above h , which can be bounded as follows. If q † = q ,

<!-- formula-not-decoded -->

If q † &gt; q , by similar argument using Hölder inequality,

<!-- formula-not-decoded -->

If q † &lt; q , by similar argument,

<!-- formula-not-decoded -->

Substituting L in (20) to the above equations yields the conclusion.

Proof of Theorem 4.2. By Lemmas B.2 and B.3, we have the following bounds for the concentration function (11) as ε → 0 , if q † ≥ q ,

<!-- formula-not-decoded -->

For s ≤ s † , the bound is dominated by ε -1 s -s ′ d -1 q . For the last case, we need to determine a balancing point of s for the two terms by setting their powers equal. The calculation shows that if s ≤ s † + d q , the bound is still dominated by ε -1 s -s ′ d -1 q , but otherwise is dominated by ε -s -s † s † -s ′ q . Therefore, we have

<!-- formula-not-decoded -->

We need to determine minimal ε n such that φ u † ( ε n ) ≤ nε 2 n . Hence for q † ≥ q ,

<!-- formula-not-decoded -->

Now if q † &lt; q , by similar argument we have the concentration function (11) as ε → 0

<!-- formula-not-decoded -->

Thus the contraction rate for q † &lt; q becomes

<!-- formula-not-decoded -->

Rewriting the equations into one yields the conclusion.

## C More Numerical Results

## C.1 Eikonal Equation

Table C.1: Comparing accuracy of various solvers for Eikonal equation (12) in terms of mean absolute error (MAE), mean squared error (MSE), and relative errors in L 1 norm (RLE-1), L 2 norm (RLE-2), and L ∞ norm (RLE-∞ ) respectively. Result in each cell are averaged over 10 experiments with different random seeds; values after ± are standard deviations of these repeated experiments.

| Model ( q )                                        | MAE                                                                                                                                                               | MSE                                                                                                                                                               | RLE-1                                                                                                                                           | RLE-2                                                                                                                                           | RL- ∞                                                                                                                                           |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| PINN B-PINN 0.5 0.8 1.0 1.2 1.5 2.0 (Gaussian) 2.5 | 1.77e-2 ± 9.03e-4 1.81e-2 ± 2.59e-2 1.96e-3 ± 5.09e-4 1.44e-3 ± 3.44e-4 1.68e-3 ± 5.41e-4 1.69e-3 ± 3.85e-4 1.68e-3 ± 3.31e-4 9.64e-3 ± 1.33e-3 7.71e-2 ± 3.18e-2 | 4.65e-4 ± 4.08e-5 1.24e-3 ± 3.52e-3 6.20e-6 ± 2.87e-6 3.50e-6 ± 1.28e-6 4.42e-6 ± 2.26e-6 4.11e-6 ± 1.52e-6 4.65e-6 ± 1.35e-6 1.30e-4 ± 3.61e-5 8.36e-3 ± 5.29e-3 | 0.1120 ± 0.0057 0.1253 ± 0.1792 0.0124 ± 0.0032 0.0091 ± 0.0022 0.0106 ± 0.0034 0.0107 ± 0.0024 0.0106 ± 0.0021 0.0610 ± 0.0084 0.4881 ± 0.2011 | 0.1167 ± 0.0051 0.1162 ± 0.1700 0.0132 ± 0.0029 0.0100 ± 0.0018 0.0110 ± 0.0030 0.0108 ± 0.0020 0.0116 ± 0.0018 0.0612 ± 0.0087 0.4668 ± 0.1743 | 0.1354 ± 0.0061 0.2143 ± 0.2193 0.0387 ± 0.0067 0.0339 ± 0.0074 0.0321 ± 0.0071 0.0284 ± 0.0057 0.0398 ± 0.0080 0.1009 ± 0.0124 0.4442 ± 0.1390 |

To verify the systematic superiority of Q-EP with q = 1 . 0 than GP ( q = 2 . 0 ), we extend our comparison to more kernels including the radius basis function (rbf), C ( x , x ′ ) = σ 2 exp( -0 . 5 r 2 ) , and rational quadratic (rq), C ( x , x ′ ) = σ 2 (1 + 1 2 α r 2 ) -α , α &gt; 0 , for r = √ ∑ d i =1 ( x i -x ′ i ) 2 /ρ 2 i in Table C.2. Within each type of kernel, Q-EP with q = 1 . 0 consistently outperforms GP ( q = 2 . 0 ). This indicates that the advantage of Q-EP over GP is independent of the choice of kernels (as long as they are compared with the same kernel).

Table C.2: Comparing accuracy of Q-EP ( q = 1 ) against GP ( q = 2 ) solvers with various kernels for Eikonal equation (12) in terms of mean absolute error (MAE), mean squared error (MSE), and relative errors in L 1 norm (RLE-1), L 2 norm (RLE-2), and L ∞ norm (RLE-∞ ) respectively. Result in each cell are averaged over 10 experiments with different random seeds; values after ± are standard deviations of these repeated experiments.

| Model ( q )    | kernel   | MAE               | MSE               | RLE-1           | RLE-2           | RL- ∞           |
|----------------|----------|-------------------|-------------------|-----------------|-----------------|-----------------|
| 1.0            | Matern   | 1.68e-3 ± 5.41e-4 | 4.42e-6 ± 2.26e-6 | 0.0106 ± 0.0034 | 0.0110 ± 0.0030 | 0.0321 ± 0.0071 |
| 2.0 (Gaussian) | Matern   | 9.64e-3 ± 1.33e-3 | 1.30e-4 ± 3.61e-5 | 0.0610 ± 0.0084 | 0.0612 ± 0.0087 | 0.1009 ± 0.0124 |
| 1.0            | rbf      | 4.39e-3 ± 1.05e-3 | 3.91e-5 ± 2.52e-5 | 0.0278 ± 0.0066 | 0.0327 ± 0.0107 | 0.0648 ± 0.0324 |
| 2.0 (Gaussian) | rbf      | 1.55e-2 ± 4.97e-4 | 3.07e-4 ± 2.87e-5 | 0.0982 ± 0.0031 | 0.0949 ± 0.0044 | 0.1236 ± 0.0048 |
| 1.0            | rq       | 1.86e-3 ± 5.40e-4 | 5.51e-6 ± 3.44e-6 | 0.0118 ± 0.0034 | 0.0122 ± 0.0038 | 0.0168 ± 0.0064 |
| 2.0 (Gaussian) | rq       | 3.07-3 ± 1.70e-3  | 1.89e-5 ± 2.83e-5 | 0.0194 ± 0.0107 | 0.0200 ± 0.0134 | 0.0325 ± 0.0242 |

## C.2 Burgers' Equation

Define the nonlinear function D ( u, d 1 u, d 2 u, d 2 1 u ) = d 2 u + ud 1 u -νd 2 1 u . The observations Y can then be expressed as P (˜ u ( X )) = [ D ( u ( X d ) , ∂ ∂x u ( X d ) , ∂ ∂t u ( X d ) , ∂ 2 ∂x 2 u ( X d )) u ( X b ) ] , and h = [ 0 N d g ] , where g is a vector of size N b whose elements are -sin( πx ) or 0 depending on the order of the corresponding elements in X b . We use the same experiment setup as above.

In Table C.3, we also observe a consistently better performance of Q-EP ( q = 1 . 0 ) compared to GP ( q = 2 . 0 ) in solving Burgers' equation (13) using various kernels.

One can possibly find a GP with fine-tuned kernel, e.g. rational quadratic (rq), to have better (Q-EP with rbf in Table C.2) or matching (Q-EP with Matern52 in Table C.3) results. However, as a

Figure C.1: Solving Burgers' equation (13) using high-resolution finite difference method (upper left), PINN (lower left), Q-EP (right three in upper row: q = 1 . 0 ), and GP (right three in lower row: q = 2 . 0 ) respectively. Blue crosses are learned inducing points.

<!-- image -->

Table C.3: Comparing accuracy of Q-EP ( q = 1 ) against GP ( q = 2 ) solvers with various kernels for Burgers' equation (13) in terms of mean absolute error (MAE), mean squared error (MSE), and relative errors in L 1 norm (RLE-1), L 2 norm (RLE-2), and L ∞ norm (RLE-∞ ) respectively. Result in each cell are averaged over 10 experiments with different random seeds; values after ± are standard deviations of these repeated experiments.

| Model ( q )    | kernel   | MAE               | MSE               | RLE-1           | RLE-2           | RL- ∞           |
|----------------|----------|-------------------|-------------------|-----------------|-----------------|-----------------|
| 1.0            | Matern   | 9.33e-3 ± 1.76e-4 | 1.77e-4 ± 2.16e-4 | 0.0242 ± 0.0128 | 0.0266 ± 0.0140 | 0.0324 ± 0.0145 |
| 2.0 (Gaussian) | Matern   | 7.13e-2 ± 8.68e-3 | 1.08e-2 ± 1.28e-2 | 0.1848 ± 0.0908 | 0.2068 ± 0.1118 | 0.2376 ± 0.1456 |
| 1.0            | rbf      | 2.51e-3 ± 1.30e-3 | 1.37e-5 ± 1.59e-5 | 0.0065 ± 0.0033 | 0.0074 ± 0.0040 | 0.0150 ± 0.0088 |
| 2.0 (Gaussian) | rbf      | 2.49e-2 ± 3.72e-3 | 1.06e-3 ± 2.63e-4 | 0.0646 ± 0.0097 | 0.0726 ± 0.0094 | 0.1144 ± 0.0073 |
| 1.0            | rq       | 2.57e-3 ± 5.52e-4 | 1.39e-5 ± 3.15e-6 | 0.0067 ± 0.0014 | 0.0083 ± 0.0010 | 0.0199 ± 0.0049 |
| 2.0 (Gaussian) | rq       | 1.09e-2 ± 1.97e-3 | 1.94e-4 ± 7.02e-5 | 0.0283 ± 0.0051 | 0.0309 ± 0.0057 | 0.0551 ± 0.0096 |

principled regularization over function spaces, Q-EP facilitates effective modeling of derivatives and differential equations, thereby alleviating the struggle of GP on meticulous kernel engineering.

## C.3 Nonlinear Elliptic Equation

Now we consider a nonlinear elliptic equation with Dirichlet boundary condition on Ω = [0 , 1] 2 :

<!-- formula-not-decoded -->

where we choose τ ( u ) = u 3 , g ( x ) ≡ 0 , and f such that the following true solution u † satisfies equation (21) [6]

<!-- formula-not-decoded -->

We follow the variational procedure in Section 3.1 and use the same experiment setup in Section 5. Figure C.2 compares the solutions by a highly-resolved finite difference method (upper left, treated as true solution), PINN (lower left), Q-EP (right three in upper row: q = 1 . 0 ), and GP (right three in lower row: q = 2 . 0 ). Compared to the true solution, Q-EP with q = 1 . 0 yields an estimate more accurate than PINN and GP. Based on Table C.4, the best solution is obtained by Q-EP with q = 0 . 5 . Since the true solution is highly fluctuating over the domain Ω , imposing more challenges on the boundary, where the largest pointwise errors occur. Adding more boundary points might help with Q-EP solvers.

## C.4 Inverse Darcy Flow

Finally, we consider a more realistic Darcy flow data used by Fourier Neural Operator (FNO) and Physics-Informed Neural Operator (PINO) available in NVIDIA PhysicsNeMo. This dataset contains thousands of permeability-solution pairs that reflect realistic porous media. Since our method is

Figure C.2: Solving nonlinear elliptic equation (21) using high-resolution finite difference method (upper left), PINN (lower left), Q-EP (right three in upper row: q = 1 . 0 ), and GP (right three in lower row: q = 2 . 0 ) respectively. Blue crosses are learned inducing points.

<!-- image -->

Table C.4: Comparing accuracy of various solvers for nonlinear elliptic equation (21) in terms of relative error in L 1 norm (RLE-1), L 2 norm (RLE-2), L ∞ norm (RLE-∞ ), and time per iteration. Result in each cell are averaged over 10 experiments with different random seeds; values after ± are standard deviations of these repeated experiments.

| Model ( q )   | RLE-1           | RLE-2           | RLE- ∞          | time/iteration   |
|---------------|-----------------|-----------------|-----------------|------------------|
| PINN          | 1.6428 ± 0.7133 | 1.5979 ± 0.6536 | 1.5595 ± 0.4462 | 0.0052 ± 0.0023  |
| B-PINN        | 0.7026 ± 0.2879 | 0.7868 ± 0.2872 | 1.2395 ± 0.3480 | 0.0108 ± 0.0013  |
| 0.5           | 0.0341 ± 0.0203 | 0.0514 ± 0.0322 | 0.1617 ± 0.0880 | 0.2530 ± 0.0008  |
| 0.8           | 0.3020 ± 0.0915 | 0.3159 ± 0.0749 | 0.5549 ± 0.1092 | 0.2604 ± 0.0016  |
| 1.0           | 0.2835 ± 0.0532 | 0.2958 ± 0.0434 | 0.5122 ± 0.0870 | 0.2614 ± 0.0005  |
| 1.2           | 0.3379 ± 0.0845 | 0.3342 ± 0.0689 | 0.5229 ± 0.1094 | 0.2714 ± 0.0004  |
| 1.5           | 0.3953 ± 0.1058 | 0.4429 ± 0.1591 | 0.8938 ± 0.5714 | 0.2645 ± 0.0003  |
| 2.0(Gaussian) | 0.9876 ± 0.2777 | 1.1028 ± 0.3254 | 2.0255 ± 1.1255 | 0.2562 ± 0.0004  |

not to train a surrogate model, we take only one pair and impose the data on 200 × 200 mesh to obtain 2000 randomly sampled observations. We then train Q-EP solvers on 60 × 60 mesh (with collocation points taken from the grid) and with 512 inducing points. This problem has much larger scale ( ∼ 20 times larger) than all the previous examples. As illustrated in Table C.6, Q-EP with q = 1 still achieves remarkable advantage compared with GP ( q = 2 ) and PINN. Note, Q-EP with q = 1 has comparable accuracy with q = 0 . 5 in forward solution. With only one training pair in the NIVIDA FNO-Darcy data, it is understandably more challenging to obtain the inverse solution, yet for which Q-EP with q = 1 attains the best accuracy. In both solutions, GP ( q = 2 ) is much worse by most metrics.

Table C.5: Comparing accuracy of various solvers for inverse Darcy flow (14) in terms of relative error in L 1 norm (RLE-1), L 2 norm (RLE-2), and L ∞ norm (RLE-∞ ) respectively. Result in each cell are averaged over 10 experiments with different random seeds; values after ± are standard deviations of these repeated experiments.

|               | Forward PDE Solution   | Forward PDE Solution   | Forward PDE Solution   | Inverse Solution   | Inverse Solution   | Inverse Solution   |
|---------------|------------------------|------------------------|------------------------|--------------------|--------------------|--------------------|
| Model ( q )   | RLE-1                  | RLE-2                  | RLE- ∞                 | RLE-1              | RLE-2              | RLE- ∞             |
| PINN          | 0.3145 ± 0.0163        | 0.3331 ± 0.0137        | 0.4225 ± 0.0250        | 6.6922 ± 0.5800    | 6.2856 ± 0.5432    | 4.1070 ± 0.3189    |
| 0.5           | 0.2397 ± 0.0975        | 0.2689 ± 0.1137        | 0.3445 ± 0.1490        | 0.4827 ± 0.2011    | 0.5217 ± 0.1916    | 0.8711 ± 0.1414    |
| 0.8           | 0.1669 ± 0.0242        | 0.1784 ± 0.0248        | 0.2155 ± 0.0321        | 0.3577 ± 0.0758    | 0.4154 ± 0.0675    | 0.8110 ± 0.1239    |
| 1.0           | 0.1354 ± 0.0286        | 0.1432 ± 0.0253        | 0.1810 ± 0.0193        | 0.4340 ± 0.0376    | 0.4942 ± 0.0376    | 0.8494 ± 0.2132    |
| 1.2           | 0.1523 ± 0.0967        | 0.1549 ± 0.0898        | 0.2066 ± 0.1035        | 0.5468 ± 0.1078    | 0.5921 ± 0.0890    | 0.7801 ± 0.1301    |
| 1.5           | 0.3270 ± 0.3257        | 0.4723 ± 0.5048        | 1.2103 ± 1.0981315     | 1.6317 ± 0.8162    | 1.5914 ± 0.7810    | 1.7561 ± 0.8064    |
| 2.0(Gaussian) | 0.6328 ± 0.5954        | 1.0658 ± 0.9186        | 3.1391 ± 2.2492        | 2.2096 ± 0.9100    | 2.1675 ± 0.8823    | 2.8022 ± 1.6839    |

<!-- image -->

0.2

0.2

0.2

Figure C.3: Solving inverse Darcy flow (14) with sparse data using Q-EP (right three in upper row: q = 1 . 0 ) and GP (right three in lower row: q = 2 . 0 ) respectively. Upper left: true inverse solution a † ; lower left: fine-resolution finite element solution u † to (14) with a † . Blue crosses are learned inducing points, and red dots indicate locations of observations.

Figure C.4: Solving inverse Darcy flow (14) with PINN running for longer (5000) iterations.

<!-- image -->

Table C.6: Comparing accuracy of various solvers for the inverse problem with FNO-Darcy data in terms of relative error in L 1 norm (RLE-1), L 2 norm (RLE-2), and L ∞ norm (RLE-∞ ) respectively. Result in each cell are averaged over 10 experiments with different random seeds; values after ± are standard deviations of these repeated experiments.

|               | Forward PDE Solution   | Forward PDE Solution   | Forward PDE Solution   | Inverse Solution   | Inverse Solution   | Inverse Solution   |
|---------------|------------------------|------------------------|------------------------|--------------------|--------------------|--------------------|
| Model ( q )   | RLE-1                  | RLE-2                  | RLE- ∞                 | RLE-1              | RLE-2              | RLE- ∞             |
| PINN          | 0.9168 ± 0.9019        | 0.8170 ± 0.7272        | 0.8190 ± 0.3859        | 1.6550 ± 0.1876    | 1.5170 ± 0.1547    | 1.4303 ± 0.1190    |
| 0.5           | 0.3316 ± 0.0677        | 0.3191 ± 0.0402        | 0.5978 ± 0.0535        | 0.7847 ± 0.1394    | 0.8548 ± 0.1683    | 1.2072 ± 0.2975    |
| 1.0           | 0.3239 ± 0.1297        | 0.3195 ± 0.0985        | 0.6041 ± 0.0830        | 0.7724 ± 0.1798    | 0.8349 ± 0.1841    | 1.2895 ± 0.2849    |
| 1.5           | 0.2338 ± 0.0197        | 0.2490 ± 0.0105        | 0.6149 ± 0.1056        | 0.8466 ± 0.1617    | 0.8744 ± 0.1213    | 1.2070 ± 0.3958    |
| 2.0(Gaussian) | 0.5341 ± 0.5296        | 0.4904 ± 0.4292        | 0.6881 ± 0.2356        | 0.9124 ± 0.1791    | 0.9251 ± 0.1502    | 1.1018 ± 0.1219    |

Figure C.5: Comparing solutions to inverse Darcy flow (14) at refined mesh sizes using Q-EP (left four in upper row: q = 1 . 0 ) and GP (left four in lower row: q = 2 . 0 ) respectively, with relative errors in L 1 norm (upper right) and L 2 norm (lower right). Blue crosses are learned inducing points. Error bars indicate standard errors based on 10 repeated experiments with different random seeds.

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims made in the abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Refer to the last section Conclusion where the limitations and future directions are discussed.

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

Justification: Please refer to Assumption s and the theorem statements for the assumptions and the Appendix for the full proof.

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

Justification: All the details of the experiments are disclosed in the section Numerical Experiments as well as the Technical Appendices . Sample codes are included in the supplementary materials and all the codes will be released upon acceptance.

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

Justification: Sample codes are included in the supplementary materials and all the codes will be released upon acceptance. Data are simulated and details on how to generate them are included in the section Numerical Experiments and the Technical Appendices .

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines () for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines () for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All the details of the experiments are disclosed in the section Numerical Experiments and the Technical Appendices .

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Please refer to relevant figures for error bars.

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

Justification: Details of experiments are disclosed at the beginning of the section Numerical Experiments . One can also refer to relevant tables for the information.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics ?

Answer: [Yes]

Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper discusses the scientific contributions at the end of the section Introduction .

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

Answer: [Yes]

Justification: The authors cite the original paper that produced the code package or dataset and respect the license and terms of use.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.

- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
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

## Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy () for what should or should not be described.