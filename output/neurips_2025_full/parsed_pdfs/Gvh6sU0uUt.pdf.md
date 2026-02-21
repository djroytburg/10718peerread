## Least squares variational inference

Yvann Le Fay 1 , Nicolas Chopin 1, ∗ , Simon Barthelmé 1 ENSAE, CREST, IP Paris 2 GIPSA-Lab, CNRS {yvann.lefay,nicolas.chopin}@ensae.fr simon.barthelme@gipsa-lab.fr

## Abstract

Variational inference seeks the best approximation of a target distribution within a chosen family, where "best" means minimising Kullback-Leibler divergence. When the approximation family is exponential, the optimal approximation satisfies a fixed-point equation. We introduce LSVI (Least Squares Variational Inference), a gradient-free, Monte Carlo-based scheme for the fixed-point recursion, where each iteration boils down to performing ordinary least squares regression on tempered log-target evaluations under the variational approximation. We show that LSVI is equivalent to biased stochastic natural gradient descent and use this to derive convergence rates with respect to the numbers of samples and iterations. When the approximation family is Gaussian, LSVI involves inverting the Fisher information matrix, whose size grows quadratically with dimension d . We exploit the regression formulation to eliminate the need for this inversion, yielding O ( d 3 ) complexity in the full-covariance case and O ( d ) in the mean-field case. Finally, we numerically demonstrate LSVI's performance on various tasks, including logistic regression, discrete variable selection, and Bayesian synthetic likelihood, showing results competitive with state-of-the-art methods, even when gradients are unavailable.

## 1 Introduction

This paper focuses on parametric variational inference (VI, [1-3]). Given an (unnormalised) target density π , we aim at finding the distribution that minimises the (reverse) Kullback-Leibler divergence:

<!-- formula-not-decoded -->

where Q is a user-chosen parametric family (e.g., Gaussians), and ¯ π = π/ ∫ π . This approach has become a de facto standard in probabilistic machine learning in recent years and is implemented in various software packages, such as STAN, NumPyro, PyMC3, and Blackjax [4-7]. The minimisation is typically carried out through gradient-based procedures using automatic differentiation, either stochastic gradient descent (SGD, [8])-often applied after reparameterising the target distribution [9, 10]-or its faster alternative natural gradient descent (NGD, [11-14]). This is convenient for users, as they only have to provide the function f := log π to the software.

These procedures use different gradient estimators; some require log π to be amenable to automatic differentiation, which is the case when using a reparameterisation, while others only require gradient estimators of expectations under the variational distribution via the log-derivative trick [15]. The gradient estimator for expectations usually suffers from high variance, and practical implementations

∗ Corresponding author

2

rely on the reparameterisation trick, which is not possible in several important cases, for instance when π is a discrete distribution, or when π is intractable or non-differentiable (as in likelihoodfree inference). Additionally, convergence of SGD is sometimes slow and/or tedious to assess, and requires careful step sizes tuning [16] while a naive implementation of NGD requires costly matrix inversions.

## 1.1 Outline and contributions

We introduce practical algorithms for VI within exponential families when gradients of log π are unavailable. These algorithms involve taking biased stochastic gradient descent steps, but we show both theoretical convergence and good performance in non-toy problems. In Section 2, we derive an exact, but intractable, iteration we call LSVI, that boils down to performing successive least squares (OLS) regression. We highlight connections to NGD and discuss its convergence properties. In Section 3, we introduce a stochastic variant that updates the OLS estimate using multiple draws from the current approximation. Importantly, under standard smoothness and relative convexity assumptions on the objective, and bounded-moment assumptions on the variational family, we establish convergence guarantees and rates with respect to the numbers of draws and iterations, conditioned on high-probability events. In addition, we provide an adaptive method to calibrate step sizes by controlling the linear regression residuals. Section 4 focuses on the Gaussian variational family; we propose a reparametrisation of the linear regression such that the OLS procedure requires no inversion of the Fisher information matrix (FIM). These schemes tailored to Gaussian distributions are cost-efficient: our methods scale linearly with d in the mean-field case, and in the full-covariance case, the cost matches the cost of computing d × d matrix products, i.e., O ( d 3 ) . In Section 5, we extensively illustrate the performance of our methods compared to other inference procedures, including gradient-based VI and exact Bayesian inference procedures. Limitations are discussed in Section 6. We provide a Python package supporting GPU parallelisation via JAX to replicate the experiments: https://github.com/ylefay/LSVI .

## 2 Exact LSVI

Let π : X → R be some unnormalised target density, with X ⊂ R d . It will be convenient to work with an exponential family Q of unnormalised densities:

where η ∈ V is the natural parameter associated to q η ∈ Q , Z ( q ) := ∫ X q denotes the partition function, and s : X → R m the extended statistic function defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In words, we include an intercept in s to make the family closed under multiplication by a positive scalar. For η = ( η (0) , ¯ η ⊤ ) ⊤ , where η (0) denotes the first component of η , let ¯ q ¯ η be the normalised version of q η (which therefore depends only on ¯ η ): ¯ q ¯ η = q η /Z η , using the short-hand Z η := Z ( q η ) . Notation E η [ · ] means a properly normalised expectation, i.e. E η [ h ] = ∫ X q η h/Z η . Likewise, we replace the standard Kullback-Leibler objective with a divergence for unnormalised densities [17], which is defined by

<!-- formula-not-decoded -->

for any density q absolutely continuous with respect to π . In addition, we assume the variational family Q is minimal and regular, which is a standard assumption in VI [12, 18-20], and is met by any standard exponential families (e.g., Gaussian, Beta, Poisson, Bernoulli, etc., [21, Table 3.1]). These assumptions ensure η ∈ V ↦→ q η is injective and the log-partition function is differentiable everywhere [21, Prop. 3.1].

Assumption 2.1 (Minimality and regularity of Q ) . The components of s are linearly independent (minimality), and the set of natural parameters V is open (regularity).

The next proposition shows that the critical points of the uKL divergence are also critical points of the KL divergence, and vice versa. In words, nothing is lost by considering the uKL instead of KL.

Proposition 2.2. Let η = ( η (0) , ¯ η ⊤ ) ⊤ ∈ V , if ∇ η uKL( q η | π ) = 0 then ∇ ¯ η KL(¯ q ¯ η | ¯ π ) = 0 , and the reciprocal holds: ∇ ¯ η KL(¯ q ¯ η | ¯ π ) = 0 and ∂ η (0) uKL( q η | π ) = 0 , then ∇ η uKL( q η | π ) = 0 .

The first-order condition of the uKL minimisation problem is given by the following proposition.

Proposition 2.3. Let f = log π be the (unnormalised) log target density. Let η = ( η (0) , ¯ η ⊤ ) ⊤ ∈ V , ∇ η uKL( q η | π ) = 0 if and only if { E η [ ss ⊤ ] } η = E η [ fs ] . Furthermore, if ∇ η uKL( q η | π ) = 0 , then η (0) = -KL(¯ q ¯ η | ¯ π ) + log ( Z ( π ) / ∫ X exp ( ¯ η ⊤ ¯ s )) .

## 2.1 The exact LSVI scheme

The first-order optimality condition is equivalent to the fixed point equation: η = ϕ ( η ) with

<!-- formula-not-decoded -->

and F η is the Fisher information matrix (FIM) associated to q η . Salimans and Knowles [22] remark that ϕ is the ordinary least squares regressor (OLS; [23]) of f ( X ) with respect to s ( X ) when X ∼ q η :

<!-- formula-not-decoded -->

A nice property of ϕ when π is in the variational family with π = q η ⋆ , is that for any η ∈ V , ϕ ( η ) = η ⋆ , i.e., ϕ exactly recovers π . However, in general ϕ ( η ) may not be in V , and naively performing a fixed-point scheme can lead to unstable variational approximations or, worse, result in non-normalisable densities (i.e., ϕ ( η ) / ∈ V ). To address this, we consider a relaxation of the fixed-point scheme obtained via a momentum fixed-point iteration [24]:

<!-- formula-not-decoded -->

where ε t &gt; 0 is such that η t +1 is in V . Such an ϵ t necessarily exists because V is open (Assumption 2.1). Since iteration (5) assumes that one has access to expectations under the variational family (which in general is not the case), we refer to (5) as the exact Least Squares Variational Inference (LSVI) iteration. This relaxation has a natural interpretation in this specific context: η t +1 in (5) is the solution of the least squares objective (4) when π = exp f is replaced by the tempered (annealed) density q 1 -ε t η t π ε t .

## 2.2 LSVI as natural gradient descent (NGD) and mirror descent (MD)

This subsection summarises a well-established connection between NGD and MD in the variational inference literature [12, 18, 20] but generalised to the unnormalised KL divergence.

Let us define the (unnormalised) moment parameter mapping ω : η ∈ V ↦→ ∇ η Z η = ∫ s ( x ) q η ( x ) , and let W = ω ( V ) be the set of moment parameters. We denote by η : W → V the inverse mapping of ω : W → V , whose existence is guaranteed under Assumption 2.1 [21, Ch. 3]. Define l as the unnormalised KL divergence (2). When expressed in natural parameters, we write l : η ∈ V ↦→ uKL( q η | π ) , when expressed in moment parameters, we write l : ω ∈ W ↦→ uKL( q ω | π ) , and similarly for expectations: E ω := E η ( ω ) .

The following proposition states that LSVI iteration (5) is a NGD iteration on the uKL divergence in the natural space of parameters, and equivalently a MD in the moment space [25, 26, Ch. 3].

Proposition 2.4 (LSVI is NGD which is equivalent to MD, [20, Lemma 1]) . Under Assumption 2.1 and provided the sequence ( η t ) defined by (5) is in V , ( η t ) satisfies the dynamic (NGD), or equivalently,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, let ω 0 ∈ W and define for t ≥ 0 (MD), where D Z ⋆ is the Bregman divergence [27] with respect to Z ∗ the Legendre transform of Z : Z ∗ ( ω ) = argmin η ∈V { η ⊤ ω -Z η } . Then, the sequence ( η t ) defined by (5) with η 0 = η ( ω 0 ) satisfies for all t ≥ 0 , η t = η ( ω t ) . In words, LSVI performs a natural gradient step in the space of natural parameters, which corresponds to a mirror descent step in the dual (moment) space.

Algorithm 1 Generic LSVI (any family Q )

<!-- formula-not-decoded -->

## end while

<!-- formula-not-decoded -->

Proposition 2.4 allows us to leverage known convergence results for MD under standard smoothness and convexity assumptions on the uKL objective [in the VI literature, see, 12, 28, 29].

Assumption 2.5. The uKL objective l : ω ∈ W ↦→ uKL( q ω | π ) is L -smooth, µ -strongly convex relative to D Z ∗ .

Under Assumption 2.5, MD is known to converge with rate O (1 /k ) for sufficiently small and linearly decreasing step sizes: ε t = ( L + αt ) -1 for 0 ≤ α &lt; µ , [see, e.g., 30, 31, Theorem 4.5 and Lemma 4.8, Theorem 4]. The non-strongly convex case ( µ = 0 ) exhibits O (1 / √ k ) convergence rate for a specific choice of step sizes [see, e.g., 30, Corollary 4.6]. In practice, it is not trivial to set the ε to obtain a O (1 /k ) rate, as the relative strong convexity parameter µ , if it exists, might be unknown and eventually very small.

Remark 2.6 . The strong-convexity/convexity assumptions are rarely verified in practice, however, such assumptions are standard for analysing convergence of optimisation algorithms (including NGD and MD), to ensure a unique minimiser and tractable rates [see, e.g., 32, Ch. 5]. While non-conjugate VI objectives may not be globally convex [20] (but it holds when the variational family contains the target), local convexity near optima often suffices for local convergence to hold.

See [33, 34] for provable smoothness guarantees on the KL objective.

## 3 Practical algorithms and their analysis

The exact LSVI mapping ϕ assumed that one has access to expectations under the variational family. In practice, exact computation of those expectations is intractable for a general target log-density f . In this section, we introduce a practical algorithm in which these expectations are estimated via Monte Carlo, and we study the impact of the Monte Carlo error on the convergence guarantees.

## 3.1 Generic LSVI

Our first algorithm comes down to replacing the two expectations in (3) with Monte Carlo estimates:

<!-- formula-not-decoded -->

where X 1 , . . . , X N i.i.d. ∼ q η . The counterpart to the exact iteration (5) is then

<!-- formula-not-decoded -->

with ˆ η 0 = η 0 . At any iteration t ≥ 1 , the step size ε t can, in all generality, depend on the current state of the algorithm via a function stepsize . We discuss one possible choice in Section 3.2. This leads naturally to generic LSVI Algorithm 1, whose one-iteration cost is O ( m 3 + m 2 N ) .

Iteration (10) replaces the exact computation of F -1 η with a Monte Carlo estimate ˆ F -1 η . This approximation introduces a bias in the estimation of the inverse FIM, and consequently, in the estimation of the natural gradient involved in (6). Further analysis of the statistical properties of the sequence

▷ ordinary least squares estimator (OLS)

(ˆ η t ) , in particular, its convergence toward a neighbourhood of the optimum, requires a careful control of the bias. When s admits uniformly bounded fourth-order moment, and the spectrum of F η is bounded away from zero, the bias conditioned to a high-probability event can be controlled.

Assumption 3.1. The sufficient statistic s admits uniformly bounded fourth-order moments:

<!-- formula-not-decoded -->

Assumption 3.2. The smallest spectral value r := inf ω ∈W ∥ F -1 ω ∥ -1 is strictly positive.

Both assumptions are verified if i) W is a compact set, and ii) s ( X ) admits fourth-order moments, for X ∼ q ω and for all ω ∈ W . While W is generally not a compact set, it should not be considered as a limiting assumption in practice, and can be lifted, see [35-37]. We further assume that f admits uniformly bounded second-order moment as this is required to control the norm of ˆ z ω .

<!-- formula-not-decoded -->

We derive the convergence in expectation to the minimum of the KL loss conditioned on the event that the estimated FIMs are well-conditioned.

Theorem 3.4 (Explicit convergence rates for LSVI) . Assume 2.1, 2.5, 3.1, 3.2, and 3.3. Let k ≥ 0 , and let ˆ η 0 , ˆ η 1 , . . . , ˆ η k be given by (10) , with ˆ ω t = ω (ˆ η t ) for 0 ≤ t ≤ k . Let A k = ∩ k t =0 A (ˆ ω t ) with A ( ω ) = [ ∥ F ω -ˆ F ω ∥ &lt; ∥ F -1 ω ∥ -1 ] . Further assume that at each iteration t ≥ 1 , the quantities ˆ F ˆ η t and ˆ z ˆ η t are computed using two independent sets of samples. Let c t = c t -1 ε -1 t -1 ( ε -1 t -µ ) -1 for t ≥ 1 , c 0 = 1 , C k = ∑ k t =1 c t -1 . Let ¯ ω k = 1 C k ∑ k t =1 c t -1 ˆ ω t be the weighted average of the iterates, and let ω ∗ be the minimiser of l .

1. Fix δ ∈ (0 , 1) , provided √ N ≥ C 0 r -2 ( k +1) δ -1 ( √ log( m ) µ 4 ν + µ 2 4 log( m )) for some constant C 0 &gt; 0 , A k happens with probability at least 1 -δ .
2. Conditioned on A k ,

<!-- formula-not-decoded -->

where the bigO terms are independent of k .

3. Let ε -1 t = L + αt for some α &gt; 0 . The RHS of (12) has asymptotic convergence rates that depend on α compared to the strong-convexity parameter µ . When α &gt; µ , the sequence ( c t ) is strictly decreasing, and the rate is O ( k -µ/α ) + O ( N -1 ) . When α = µ , the sequence ( c t ) is constant, and the rate is O ( k -1 )+ O ( log( k ) k -1 N -1 ) + O ( N -1 ) . When α &lt; µ , the sequence ( c t ) is strictly increasing, and the rate is O ( k -µ/α ) + O ( k -1 N -1 ) + O ( N -1 ) .

Remark 3.5 . Our proof follows a similar strategy to that of Hanzely and Richtárik [30], extending their mirror descent lemma to biased estimates. We control both the bias and the variance of the FIM estimate, conditionally on the event that the estimated FIM is well-conditioned ( A k ). We show this occurs with high probability when N is sufficiently large, using concentration inequalities for positive-definite matrices [38].

Remark 3.6 . The convergence guarantees can be decomposed in three terms. The first term is due to initialisation and vanishes as k → ∞ , the third term is the Monte Carlo bias and vanishes as N →∞ , and the second is a cross term and vanishes whenever k →∞ or N →∞ .

Remark 3.7 . The OLS estimate to the regression problem uses a single set of samples to compute both ˆ F ˆ η t and ˆ z ˆ η t , contrary to the estimate introduced in the previous theorem. Additionally, for many exponential families, closed-form expressions for F η are known. Since the OLS is optimal with respect to the variance, it exhibits lower variance compared to others estimates. Importantly, it is inefficient to use two distinct set of samples or to replace the estimated FIM with the exact FIM [22].

## 3.2 The choice of the ε t 's

Setting ε to a small enough and linearly decreasing sequence of step sizes ensures convergence of the sequence (10) to a neighbourhood of a local minimizer η ⋆ [30, 31], see Theorem 3.4. However, the smoothness and strong-convexity parameters ( L, µ ) of the KL objective, if they exist, are rarely known in practice [33, 34]. For these reasons, choosing the ε t can be a tedious task, as in any stochastic optimisation scheme [16]: step sizes that are too large lead to unstable behaviours while too small step sizes lead to slow convergence.

Let η ∈ V and η ⋆ = ϕ ( η ) be the OLS, consider the following linear regression objective,

<!-- formula-not-decoded -->

where v i is the residual of the regression. Then (13) implies that for any ε ∈ (0 , 1]

<!-- formula-not-decoded -->

The previous equation (14) shows that descending toward the direction of the OLS with step size ε multiplies the variance of the residuals v 1 , . . . , v N , v 2 by ε 2 . Let u 2 be some upper bound on the variance of the residuals, and let ε ≤ u/v , then the residuals have variance less than u 2 . This remark, combined with a backtracking procedure to ensure that the iterates remain in the set of natural parameters, yields an adaptive schedule for choosing the step sizes (Algorithm 4 in Appendix B), which we have found to be robust against noisy iterates and slow descents.

## 4 Gaussian families

The two most commonly-used families Q in variational inference are the full-covariance Gaussian family ( N d ( µ, Σ) with arbitrary µ and Σ ≻ 0 ), and the mean-field Gaussian family ( Σ is diagonal). A single iteration of LSVI requires inverting the Fisher information matrix (FIM) F , which is too expensive to be practical in high-dimension; i.e., O ( m 3 ) , with m = O ( d ) (resp. m = O ( d 2 ) ) in the mean-field (resp. full-covariance) case.

Attempts to lessen the computational complexity of inference procedures over Gaussian distributions either rely on access to cheap gradient estimates in the space of moments [13, 18, 20, 39], on single draw updates making the FIM estimate cheap to invert but noisy [13, 40], or on restrictive assumptions on the target density [41, 42]. We derive closed-form formulae for the natural gradient descent iteration whose cost, in the full-covariance case, essentially amounts to the cost of computing d × d matrix products, that is O ( d 3 ) . In the mean-field case, the cost is O ( d ) .

Full-covariance Gaussian family Let Q be the family of (unnormalised) Gaussian densities of dimension d . The sufficient statistic is s ( x ) := (1 , x ⊤ , (vec ( xx ⊤ )) ⊤ ) ⊤ ∈ R m with m = d ( d +1)+ 1 , where vec ( xx ⊤ ) denotes the vector obtained by vertically stacking the columns of xx ⊤ , and we denote by unvec the inverse operation. Consider a natural parameter η = ( η (0) , η (1) , ⊤ , η (2) , ⊤ ) ⊤ ∈ V with η (0) ∈ R , η (1) ∈ R d and η (2) ∈ R d 2 , then it defines a unique Gaussian distribution with mean and covariance matrix given by

<!-- formula-not-decoded -->

We reparameterise the linear regression of f ( X ) with respect to s ( X ) , where X ∼ N ( µ, Σ) , into a regression of f ( µ + CZ ) with respect to t ( Z ) , where Z ∼ N (0 , I d ) , and C = Chol (Σ) is the Cholesky of Σ , and

<!-- formula-not-decoded -->

and

In brief, t is a one-to-one transformation such that the output vector has un-correlated components: E [ t ( Z ) t ⊤ ( Z )] = I . That makes possible the estimation of γ without inverting the FIM. The explicit mapping from γ to η depending on ( µ, Σ) is given by the next theorem.

<!-- formula-not-decoded -->

Theorem 4.1 (LSVI mapping ϕ for full-covariance Gaussian distributions) . Let η ∈ V defines a Gaussian distribution X ∼ N ( µ, Σ) , and let C = Chol (Σ) be the Cholesky of Σ . Then, β := ϕ ( η ) is defined recursively from bottom to top by

<!-- formula-not-decoded -->

where γ = E [ t ( Z ) f ( µ + CZ )] has subcomponents γ = ( γ (0) , γ (1) , ⊤ , γ (2) , ⊤ ) ⊤ , γ (0) ∈ R , γ (1) ∈ R d , γ (2) ∈ R d ( d +1) / 2 , and where Γ is the symmetric matrix given componentwise by Γ i,i = γ (2) 1+1 / 2(2 d +2 -i )( i -1) / √ 2 , Γ i,i + k = γ (2) 1+1 / 2(2 d +2 -i )( i -1)+ k / 2 for 1 ≤ i ≤ d and 1 ≤ k ≤ d -i . In addition, if f has second-order derivatives such that ∥ E X [ ∇ f ] ∥ &lt; ∞ and 0 ≺ -E X [ ∇ 2 f ] , then ϕ ( η ) defines a Gaussian distribution with mean and covariance given by

<!-- formula-not-decoded -->

Theorem 4.1 gives the regressor with respect to s ( X ) of f ( X ) , as a function of ( µ, C ) and γ . Furthermore, all the involved operations have cost dominated by the computation of C , which is the same as computing products of d × d matrices, O ( d 3 ) .

Mean-field Gaussian family The family of mean-field Gaussian distributions is treated similarly to the previous one by removing the cross-terms z i z j in the sufficient statistic. The total cost of the OLS computation is O ( d ) . See Appendix D.3 for the explicit regression procedure.

## 4.1 Stochastic schemes tailored to Gaussian distributions

We now take advantage of the reparametrisation tricks previously introduced to derive tailored implementations of LSVI for Gaussian variational families, with optimal one-iteration cost in d .

An unbiased estimate of the OLS (17) is given by

<!-- formula-not-decoded -->

We define ˆ η as the estimate obtained by plugging ˆ γ into (18) of Theorem 4.1. The mean-field case is treated in a similar manner. See Algorithms 2 and 3.

## Algorithm 2 LSVI-MF (mean-field Gaussian family)

<!-- formula-not-decoded -->

Algorithm 3 LSVI-FC (full-covariance Gaussian family)

<!-- formula-not-decoded -->

## 5 Numerical experiments

We consider three examples: one where SGD may be used to minimise the KL objective, and two where it may not, because the reparameterisation trick is not possible: distributions q in Q are discrete, log π is not differentiable, or because the log-derivative trick yields noisy estimates [15].

In the first example (logistic regression), we compare all three LSVI 1 instances with other gradientbased KL minimisation procedures, including ADVI, NGD, and a gradient-free procedure for Gaussian mixtures. In the second and third examples (variable selection and Bayesian synthetic likelihood, BSL), since SGD is not available, we assess the approximation error of LSVI relative to the true posterior using exact Bayesian inference.

## 5.1 Logistic regression

Given data ( x i , y i ) ∈ R d ×{-1 , 1 } , i = 1 , . . . , n , the posterior distribution of a logistic regression model is: π ( β ) ∝ p ( β ) ∏ n i =1 F ( y i x ⊤ i β ) where F ( x ) = 1 / (1 + e -x ) and p ( β ) is a (typically Gaussian) prior over the parameter β . This type of posterior is often close to a Gaussian, and is a popular benchmark in Bayesian computation [43]. See Appendix C.2 for a summary of the considered datasets and the priors.

Whenever applicable, we compare LSVI (Algorithms 1, 2, 3) with NGD and ADVI. For NGD, the gradients are obtained via JAX autodifferentiation [44] and the FIM is estimated via Monte Carlo. For ADVI, we use the standard implementations given by pyMC3 [6] and Blackjax [7] with default step size schedules (that is, a modification of Adam and RMSProp for pyMC, and comparable fixed step sizes for Blackjax). In addition, we provide a comparison of LSVI (Algorithm 1) in low dimension with the gradient-free iteration for Gaussian mixtures (GMMVI, [45]) which is a fair comparison since GMMVI and LSVI Algorithm 1 have the same complexity in this case. In addition, we illustrate the compatibility of our proposed methods with subsampling procedures for large datasets [46, 47] to reduce the cost of the log-likelihood evaluations.

Figure 1 summarises this comparison for the Pima dataset (full-covariance case). One sees that LSVI (Algorithm 1) converges essentially in one step, LSVI-FC (Algorithm 3) converges in less than 100 steps for linearly decreasing step sizes. For such a low-dimensional dataset ( d = 9 ), LSVI remains competitive with LSVI-FC since it converges faster, and the matrices it needs to invert are small. LSVI performs comparably to NGD and GMMVI, but is less noisy (with or without an adaptive schedule 4). We consider larger and more challenging datasets as recommended by [43]. Figure 2 (left) does the same comparison for the MNIST dataset (mean-field covariance), In Appendix C, Figure 4 for the Sonar dataset (full-covariance) and Figure 7 for the Census-Income dataset (mean-field covariance with subsampling). This time, inverting the FIM is too costly (e.g., 2015 × 2015 for Sonar), so we only use the tailored schemes LSVI-MF and LSVI-FC. Section C.2 contains extra details and results for all datasets in Table 2, including runtimes and memory usage (Table 1), average cost time per iteration with respect to N (Figures 4 and 5), loss vs elapsed time and classification performance (Figure 6), details on the considered schedules for the step sizes (Table 3).

## 5.2 Variable selection

Given a dataset D = ( x i , y i ) i =1 ,...,n , x i ∈ R d , y i ∈ R , the variable selection task in Bayesian linear regression may be modelled as y i = x ⊤ i diag( γ ) β + σε i , ε i ∼ N (0 , 1) , where γ ∈ { 0 , 1 } d is a vector of inclusion variables, which is assigned a prior distribution that is a product of Bernoulli ( p ) ; e.g., p = 1 / 2 . If ( β, σ 2 ) is assigned a conjugate prior, the marginal posterior distribution π ( γ |D ) (with β , σ 2 integrated out) admits a closed-form expression, the support of which is { 0 , 1 } d . It is therefore natural to set Q to the family of Bernoulli products, i.e. q ( γ ) = ∏ d i =1 q γ i i (1 -q i ) 1 -γ i with q i ∈ [0 , 1] for i = 1 , . . . , d . This family is discrete, which precludes a reparametrisation trick, and the application of ADVI.

Figure 2 (right) compares the posterior inclusion probabilities, i.e. π ( γ i = 1 |D ) approximated either through LSVI (Algorithm 1), or the Sequential Monte Carlo (SMC) sampler of Schäfer and Chopin [48], for the concrete dataset ( d = 92 ). This dataset is challenging as it generates strong posterior correlations between the γ i . Despite this, LSVI gives a reasonable approximation of the

1 Python package: https://github.com/ylefay/LSVI

Figure 1: Logistic regression, Pima data, full-covariance approximation. KL divergence (up to an unknown constant) between the variational approximation and the posterior, as a function of the number of iterations. Left: truncated from iteration t ≥ 20 for better readability. Right: focus on GMMVI, LSVI and NGD. Mean over 100 repetitions and one standard deviation interval ( jax.numpy.std ).

<!-- image -->

Figure 2: Left: Logistic regression, MNIST, mean-field approximation. KL divergence between the variational approximation and the posterior, as a function of the elapsed time. Truncated from iteration t ≥ 10 . Mean over 100 repetitions and one standard deviation interval. Right: Variable selection example, posterior marginal probabilities π ( γ i = 1 |D ) : LSVI vs SMC. (LSVI: 100 repetitions, the min-max intervals are reported with arrows, SMC: 3 repetitions).

<!-- image -->

true posterior. To the best of our knowledge, this is the first time variational inference is implemented for variable selection using the Bernoulli product family. See Section C.3 for extra numerical results and more details on the prior, the data, and the implementation.

## 5.3 Bayesian synthetic likelihood

BSL is a popular way to perform likelihood-free inference, that is, inference on a parametric model which is described only through a simulator: one is able to sample Y ∼ P θ , but not to compute the likelihood p ( y | θ ) ; see Frazier et al. [49] for a review.

BSL requires to specify s ( y ) , a low-dimensional summary of the data and assumes that s ( y ) ∼ N ( b ( θ ) , Σ( θ )) , leading to posterior density π ( θ ) ∝ p ( θ ) N ( s ( y ); b ( θ ) , Σ( θ )) , where p ( θ ) is the prior. Since functions b and Σ are unknown, they are replaced by empirical moments ˆ b ( θ ) , ˆ Σ( θ ) , computed from simulated data. This makes BSL, and in particular its Markov Chain Monte Carlo (MCMC) implementations, particularly CPU-intensive, as the data simulator must be run many times. Furthermore, each evaluation of π is corrupted with noise, making it impossible to differentiate log π . Note that, in general, the data simulator is too complex to implement some form of reparametrisation trick, or the application of automatic differentiation procedures.

We consider the toad's displacement example from [50], which has been considered in various BSL papers [49, 51]. The model is parameterised by θ = ( α, γ, p 0 ) ∈ R + × R + × [0 , 1] . See Section C.4 for more details on the model. We implement both LSVI-MF and LSVI-FC. For the former, we use a family of truncated Gaussian distributions, while for the latter, we re-parametrise the model in terms of ξ = f ( θ ) , where f is one-to-one transform between Θ and R d . The top panel of Figure 3 shows that both LSVI algorithms converge quickly. The bottom panel shows that the full-covariance LSVI approximation matches the posterior obtained via MCMC, at a fraction of the CPU cost, see Table 1. Again, we refer to Section C.4 for more details on the implementation of either LSVI or MCMC.

8

Figure 3: Left: Variational approximations of each coordinate of θ with one standard deviation interval, normalised. Truncated Gaussian: solid line. Full covariance Gaussian: dashed line. Right: Full-covariance Gaussian variational approximation (blue), MCMC approximation (orange).

<!-- image -->

## 6 Limitations

The current approach is limited to exponential families; mixture of exponential families may be tackled by adapting the expectation-maximisation approach of Arenz et al. [45], or by building on existing applications of NGD VI methods to mixtures of exponential families [19, 22]. For Gaussian approximations, if the posterior contains directions that are strongly non-Gaussian, then conditionalGaussian strategies like integrated nested Laplace approximations may be applied [52]. In discrete exponential families, independence can be lifted by considering tree-structured dependencies, which are quite flexible, see, e.g., Wainwright and Jordan [21].

## Acknowledgments and Disclosure of Funding

The first author gratefully acknowledges partial support from the Magnus Ehrnrooth foundation. The authors thank Sam Power, Mohammad Emtiyaz Khan and anonymous reviewers for insightful remarks on a preliminary version.

## References

- [1] Michael I. Jordan, Zoubin Ghahramani, Tommi S. Jaakkola, and Lawrence K. Saul. An introduction to variational methods for graphical models. Machine Learning , 37(2):183-233, 1999. ISSN 1573-0565. doi: 10.1023/A:1007665907178. URL https://doi.org/10.1023/A: 1007665907178 .
- [2] David M. Blei, Alp Kucukelbir, and Jon D. McAuliffe. Variational inference: a review for statisticians. Journal of the American Statistical Association , 112(518):859-877, 2017. ISSN 0162-1459,1537-274X. doi: 10.1080/01621459.2017.1285773. URL https://doi.org/ 10.1080/01621459.2017.1285773 .
- [3] Cheng Zhang, Judith Bütepage, Hedvig Kjellström, and Stephan Mandt. Advances in variational inference. IEEE Transactions on Pattern Analysis and Machine Intelligence , 41(8): 2008-2026, 2019. doi: 10.1109/TPAMI.2018.2889774. URL https://doi.org/10.1109/ TPAMI.2018.2889774 .

- [4] Bob Carpenter, Andrew Gelman, Matthew D. Hoffman, Daniel Lee, Ben Goodrich, Michael Betancourt, Marcus Brubaker, Jiqiang Guo, Peter Li, and Allen Riddell. Stan: A probabilistic programming language. Journal of Statistical Software , 76(1):1-32, 2017. doi: 10.18637/jss.v076.i01. URL https://www.jstatsoft.org/index.php/jss/article/ view/v076i01 .
- [5] Du Phan, Neeraj Pradhan, and Martin Jankowiak. Composable effects for flexible and accelerated probabilistic programming in numpyro, 2019. URL https://arxiv.org/abs/1912. 11554 .
- [6] John Salvatier, Thomas V. Wiecki, and Christopher Fonnesbeck. Probabilistic programming in python using PyMC3. PeerJ Computer Science , 2:e55, 2016. doi: 10.7717/peerj-cs.55. URL https://doi.org/10.7717/peerj-cs.55 .
- [7] Alberto Cabezas, Adrien Corenflos, Junpeng Lao, Rémi Louf, Antoine Carnec, Kaustubh Chaudhari, Reuben Cohn-Gordon, Jeremie Coullon, Wei Deng, Sam Duffield, Gerardo DuránMartín, Marcin Elantkowski, Dan Foreman-Mackey, Michele Gregori, Carlos Iguaran, Ravin Kumar, Martin Lysy, Kevin Murphy, Juan Camilo Orduz, Karm Patel, Xi Wang, and Rob Zinkov. Blackjax: Composable bayesian inference in jax, 2024. URL https://arxiv.org/ abs/2402.10797 .
- [8] Rajesh Ranganath, Sean Gerrish, and David Blei. Black Box Variational Inference. In Samuel Kaski and Jukka Corander, editors, Proceedings of the Seventeenth International Conference on Artificial Intelligence and Statistics , volume 33 of Proceedings of Machine Learning Research , pages 814-822, Reykjavik, Iceland, 22-25 Apr 2014. PMLR. URL https://proceedings.mlr.press/v33/ranganath14.html .
- [9] Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M. Blei. Automatic differentiation variational inference. J. Mach. Learn. Res. , 18(1):430-474, 2017. ISSN 1532-4435. URL http://jmlr.org/papers/v18/16-107.html .
- [10] Michalis Titsias and Miguel Lázaro-Gredilla. Doubly stochastic variational bayes for nonconjugate inference. In Eric P. Xing and Tony Jebara, editors, Proceedings of the 31st International Conference on Machine Learning , volume 32 of Proceedings of Machine Learning Research , pages 1971-1979, Bejing, China, 22-24 Jun 2014. PMLR. URL https: //proceedings.mlr.press/v32/titsias14.html .
- [11] Shun-ichi Amari. Natural gradient works efficiently in learning. Neural Computation , 10(2): 251-276, 02 1998. ISSN 0899-7667. doi: 10.1162/089976698300017746. URL https: //doi.org/10.1162/089976698300017746 .
- [12] Mohammad Khan and Wu Lin. Conjugate-Computation Variational Inference : Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models. In Aarti Singh and Jerry Zhu, editors, Proceedings of the 20th International Conference on Artificial Intelligence and Statistics , volume 54 of Proceedings of Machine Learning Research , pages 878887. PMLR, 20-22 Apr 2017. URL https://proceedings.mlr.press/v54/khan17a. html .
- [13] Mohammad Emtiyaz Khan and Didrik Nielsen. Fast yet simple natural-gradient descent for variational inference in complex models. In 2018 International Symposium on Information Theory and Its Applications (ISITA) , page 31-35. IEEE Press, 2018. doi: 10.23919/ISITA. 2018.8664326. URL https://doi.org/10.23919/ISITA.2018.8664326 .
- [14] Mohammad Emtiyaz Khan and Håvard Rue. The bayesian learning rule. Journal of Machine Learning Research , 24(281):1-46, 2023. URL http://jmlr.org/papers/v24/22-0291. html .
- [15] Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning , 8(3):229-256, May 1992. ISSN 1573-0565. doi: 10.1007/BF00992696. URL https://doi.org/10.1007/BF00992696 .

- [16] Manushi Welandawe, Michael Riis Andersen, Aki Vehtari, and Jonathan H. Huggins. A framework for improving the reliability of black-box variational inference. Journal of Machine Learning Research , 25(219):1-71, 2024. URL http://jmlr.org/papers/v25/22-0327. html .
- [17] Thomas P. Minka. Divergence measures and message passing. In Divergence measures and message passing , 2005. URL https://miat.inrae.fr/AIGM/biblios/TR-2005-173. pdf .
- [18] Mohammad Khan, Didrik Nielsen, Voot Tangkaratt, Wu Lin, Yarin Gal, and Akash Srivastava. Fast and scalable Bayesian deep learning by weight-perturbation in Adam. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pages 2611-2620. PMLR, 10-15 Jul 2018. URL https://proceedings.mlr.press/v80/khan18a.html .
- [19] WuLin, Mohammad Emtiyaz Khan, and Mark Schmidt. Fast and simple natural-gradient variational inference with mixture of exponential-family approximations. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning , volume 97 of Proceedings of Machine Learning Research , pages 3992-4002. PMLR, 09-15 Jun 2019. URL https://proceedings.mlr.press/v97/lin19b.html .
- [20] Kaiwen Wu and Jacob R. Gardner. Understanding stochastic natural gradient variational inference. Journal of Machine Learning Research , 2024. URL https://dl.acm.org/doi/10. 5555/3692070.3694258 .
- [21] Martin J. Wainwright and Michael I. Jordan. Graphical models, exponential families, and variational inference. Foundations and Trends in Machine Learning , 1(1-2):1-305, jan 2008. ISSN 1935-8237.
- [22] Tim Salimans and David A. Knowles. Fixed-Form Variational Posterior Approximation through Stochastic Linear Regression. Bayesian Analysis , 8(4):837 - 882, 2013. doi: 10.1214/13-BA858. URL https://doi.org/10.1214/13-BA858 .
- [23] R. Penrose. On best approximate solutions of linear matrix equations. Mathematical Proceedings of the Cambridge Philosophical Society , 52(1):17-19, 1956. doi: 10.1017/ S0305004100030929. URL https://doi.org/10.1017/S0305004100030929 .
- [24] Heinz H. Bauschke and Patrick L. Combettes. Fejér Monotonicity and Fixed Point Iterations , pages 91-109. Springer International Publishing, Cham, 2017. ISBN 978-3-319-48311-5. doi: 10.1007/978-3-319-48311-5\_5. URL https://doi.org/10.1007/978-3-319-48311-5\_ 5 .
- [25] Arkadij Semenoviˇ c Nemirovskij and David Borisovich Yudin. Problem Complexity and Method Efficiency in Optimization . Wiley, New York, 1983. Originally published in Russian in 1979.
- [26] Haihao Lu, Robert M. Freund, and Yurii Nesterov. Relatively smooth convex optimization by first-order methods, and applications. SIAM Journal on Optimization , 28(1):333-354, 2018. doi: 10.1137/16M1099546. URL https://doi.org/10.1137/16M1099546 .
- [27] L.M. Bregman. The relaxation method of finding the common point of convex sets and its application to the solution of problems in convex programming. USSR Computational Mathematics and Mathematical Physics , 7(3):200-217, 1967. ISSN 0041-5553. doi: 10.1016/ 0041-5553(67)90040-7. URL https://doi.org/10.1016/0041-5553(67)90040-7 .
- [28] Mohammad Emtiyaz Khan, Reza Babanezhad, Wu Lin, Mark Schmidt, and Masashi Sugiyama. Faster stochastic variational inference using proximal-gradient methods with general divergence functions. In Proceedings of the Thirty-Second Conference on Uncertainty in Artificial Intelligence , UAI'16, page 319-328, Arlington, Virginia, USA, 2016. AUAI Press. ISBN 9780996643115. URL https://dl.acm.org/doi/10.5555/3020948.3020982 .

- [29] Mohammad Emtiyaz Khan, Pierre Baque, François Fleuret, and Pascal Fua. Kullback-leibler proximal variational inference. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 28. Curran Associates, Inc., 2015. URL https://papers.nips.cc/paper\_files/paper/2015/hash/ 3214a6d842cc69597f9edf26df552e43-Abstract.html .
- [30] Filip Hanzely and Peter Richtárik. Fastest rates for stochastic mirror descent methods. Computational Optimization and Applications. An International Journal , 79(3):717-766, 2021. ISSN 0926-6003,1573-2894. doi: 10.1007/s10589-021-00284-5. URL https://doi.org/10. 1007/s10589-021-00284-5 .
- [31] Pierre-Cyril Aubin-Frankowski, Anna Korba, and Flavien Léger. Mirror descent with relative smoothness in measure spaces, with application to sinkhorn and EM. In Proceedings of the 36th International Conference on Neural Information Processing Systems , NIPS '22, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 978-1-713-87108-8. URL https://dl.acm.org/doi/10.5555/3600270.3601525 .
- [32] Francis Bach. Learning Theory from First Principles . Adaptive Computation and Machine Learning. The MIT Press, Cambridge, Massachusetts ; London, England, 2024. ISBN 9780262049443. "The aim of this book is to provide the simplest formulations that can be derived 'from first principles' with simple arguments".
- [33] Justin Domke. Provable smoothness guarantees for black-box variational inference. In Hal Daumé III and Aarti Singh, editors, Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , pages 25872596. PMLR, 13-18 Jul 2020. URL https://proceedings.mlr.press/v119/domke20a. html .
- [34] Justin Domke, Robert Gower, and Guillaume Garrigos. Provable convergence guarantees for black-box variational inference. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems , volume 36, pages 66289-66327. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ d0bcff6425bbf850ec87d5327a965db9-Paper-Conference.pdf .
- [35] Kevin Scaman and Cedric Malherbe. Robustness analysis of non-convex stochastic gradient descent using biased expectations. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 16377-16387. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/ paper\_files/paper/2020/file/bd4d08cd70f4be1982372107b3b448ef-Paper.pdf .
- [36] Kevin Scaman, Cedric Malherbe, and Ludovic Dos Santos. Convergence rates of non-convex stochastic gradient descent under a generic lojasiewicz condition and local smoothness. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 19310-19327. PMLR, 17-23 Jul 2022. URL https://proceedings.mlr.press/v162/scaman22a.html .
- [37] Bastien Batardière, Julien Chiquet, Joon Kwon, and Julien Stoehr. Importance sampling-based gradient method for dimension reduction in poisson log-normal model, 2024. URL https: //arxiv.org/abs/2410.00476 .
- [38] Richard Y. Chen, Alex Gittens, and Joel A. Tropp. The masked sample covariance estimator: an analysis using matrix concentration inequalities. Information and Inference: A Journal of the IMA , 1(1):2-20, 05 2012. ISSN 2049-8764. doi: 10.1093/imaiai/ias001. URL https: //doi.org/10.1093/imaiai/ias001 .
- [39] Linda S L Tan. Analytic natural gradient updates for cholesky factor in gaussian variational approximation. Journal of the Royal Statistical Society Series B: Statistical Methodology , page qkaf001, 01 2025. ISSN 1369-7412. doi: 10.1093/jrsssb/qkaf001. URL https://doi.org/ 10.1093/jrsssb/qkaf001 .

- [40] D. Nguyen A. Godichon-Baggioni and M.-N. Tran. Natural gradient variational bayes without fisher matrix analytic calculation and its inversion. Journal of the American Statistical Association , 0(0):1-12, 2024. doi: 10.1080/01621459.2024.2392904. URL https: //doi.org/10.1080/01621459.2024.2392904 .
- [41] David Barber and Christopher M. Bishop. Ensemble learning for multi-layer networks. In Michael I. Jordan, Michael J. Kearns, and Sara A. Solla, editors, Advances in Neural Information Processing Systems 10, [NIPS Conference, Denver, Colorado, USA, 1997] , pages 395401. The MIT Press, 1997. URL https://papers.nips.cc/paper\_files/paper/1997 .
- [42] Manfred Opper and Cédric Archambeau. The variational Gaussian approximation revisited. Neural Computation , 21(3):786-792, 2009. ISSN 0899-7667,1530-888X. doi: 10.1162/neco. 2008.08-07-592. URL https://doi.org/10.1162/neco.2008.08-07-592 .
- [43] Nicolas Chopin and James Ridgway. Leave Pima Indians Alone: Binary Regression as a Benchmark for Bayesian Computation. Statistical Science , 32(1):64 - 87, 2017. doi: 10.1214/ 16-STS581. URL https://doi.org/10.1214/16-STS581 .
- [44] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http: //github.com/jax-ml/jax .
- [45] Oleg Arenz, Gerhard Neumann, and Mingjun Zhong. Efficient gradient-free variational inference using policy search. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pages 234-243. PMLR, 10-15 Jul 2018. URL https://proceedings. mlr.press/v80/arenz18a.html .
- [46] Matthew D. Hoffman, David M. Blei, and Francis Bach. Online learning for latent dirichlet allocation. In Proceedings of the 24th International Conference on Neural Information Processing Systems - Volume 1 , NIPS'10, page 856-864, Red Hook, NY, USA, 2010. Curran Associates Inc. URL https://dl.acm.org/doi/10.5555/2997189.2997285 .
- [47] Matthew D. Hoffman, David M. Blei, Chong Wang, and John Paisley. Stochastic variational inference. Journal of Machine Learning Research , 14(40):1303-1347, 2013. URL http: //jmlr.org/papers/v14/hoffman13a.html .
- [48] Christian Schäfer and Nicolas Chopin. Sequential Monte Carlo on large binary sampling spaces. Stat. Comput. , 23(2):163-184, 2013. ISSN 0960-3174,1573-1375. doi: 10.1007/ s11222-011-9299-z. URL https://doi.org/10.1007/s11222-011-9299-z .
- [49] David T. Frazier, David J. Nott, Christopher Drovandi, and Robert Kohn. Bayesian inference using synthetic likelihood: Asymptotics and adjustments. Journal of the American Statistical Association , 118(544):2821-2832, 2023. doi: 10.1080/01621459.2022.2086132. URL https://doi.org/10.1080/01621459.2022.2086132 .
- [50] Philippe Marchand, Morgan Boenke, and David M. Green. A stochastic movement model reproduces patterns of site fidelity and long-distance dispersal in a population of fowler's toads (anaxyrus fowleri). Ecological Modelling , 360:63-69, 2017. ISSN 0304-3800. doi: 10.1016/j. ecolmodel.2017.06.025. URL https://doi.org/10.1016/j.ecolmodel.2017.06.025 .
- [51] Ziwen An, David J. Nott, and Christopher Drovandi. Robust Bayesian synthetic likelihood via a semi-parametric approach. Statistics and Computing , 30(3):543-557, 2020. ISSN 09603174,1573-1375. doi: 10.1007/s11222-019-09904-x. URL https://doi.org/10.1007/ s11222-019-09904-x .
- [52] Håvard Rue, Sara Martino, and Nicolas Chopin. Approximate bayesian inference for latent gaussian models by using integrated nested laplace approximations. Journal of the Royal Statistical Society: Series B (Statistical Methodology) , 71(2):319-392, 2009. doi: 10.1111/j. 1467-9868.2008.00700.x.

- [53] Nicolas Chopin and Omiros Papaspiliopoulos. SMC Samplers . Springer International Publishing, Cham, 2020. ISBN 978-3-030-47845-2.
- [54] I.-C. Yeh. Modeling of strength of high-performance concrete using artificial neural networks. Cement and Concrete Research , 28(12):1797-1808, 1998. ISSN 0008-8846. doi: 10.1016/ S0008-8846(98)00165-3. URL https://doi.org/10.1016/S0008-8846(98)00165-3 .
- [55] Edward I. George and Robert E. McCulloch. Approaches for Bayesian variable selection. Statistica Sinica , 7(2):339-373, 1997. ISSN 1017-0405.
- [56] David I. Warton. Penalized normal likelihood and ridge regularization of correlation and covariance matrices. Journal of the American Statistical Association , 103(481):340-349, 2008. ISSN 01621459. URL http://www.jstor.org/stable/27640044 .
- [57] Tropp Joel Aaron. An introduction to matrix concentration inequalities. Foundations and Trends in Machine Learning , 8(1-2):1-230, 2015. ISSN 1935-8237. URL https://doi. org/10.1561/2200000048 .
- [58] Léon Bottou, Frank E. Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. SIAM Review , 60(2):223-311, 2018. doi: 10.1137/16M1080173. URL https://doi.org/10.1137/16M1080173 .

## Contents

| A          | Notations                                                                                | 16   |
|------------|------------------------------------------------------------------------------------------|------|
| B Adaptive | schedule algorithm 17                                                                    |      |
| C          | Extra details on numerical experiments                                                   | 17   |
| C.1        | Runtime analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   | 17   |
| C.2        | Logistic regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  | 17   |
| C.3        | Variable selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . | 19   |
| C.4        | BSL and toads displacement model . . . . . . . . . . . . . . . . . . . . . . . . . .     | 20   |
| D          | Proofs                                                                                   | 22   |
| D.1        | First order condition and critical points of the uKL objective . . . . . . . . . . . .   | 22   |
| D.2        | The exact LSVI is a natural gradient descent . . . . . . . . . . . . . . . . . . . . .   | 23   |
| D.3        | The exact LSVI mapping for mean-field Gaussian distributions . . . . . . . . . . .       | 23   |
| D.4        | The exact LSVI mapping for Gaussian distributions . . . . . . . . . . . . . . . . .      | 25   |
| D.5        | Concentration bounds for the Fisher matrix in the compact case . . . . . . . . . . .     | 27   |
| D.6        | Convergence analysis of the stochastic LSVI algorithm . . . . . . . . . . . . . . .      | 28   |

## A Notations

For any vector u ∈ R p , we denote by u -1 ∈ R p the componentwise inverse of u , and u ⊗ u ′ the componentwise product of u with u ′ . For any matrix U ∈ R p × q , we denote by vec( U ) the p × q vector obtained by vertically stacking the columns of U , and by unvec the inverse operation satisfying unvec(vec( U )) = U , while U ⊗ U ′ denotes the Kronecker product between matrices. For any square matrix U ∈ R p × p , let diag( U ) be the p vector composed of the diagonal components of U and let ∥ U ∥ be the spectral norm of U .

For any set A , U ( A ) denotes the uniform distribution over A . N ( µ, Σ) denotes the Gaussian distribution with mean µ and covariance matrix Σ , and N ( µ, σ 2 ) with σ 2 = ( σ 2 1 , . . . , σ 2 d ) ⊤ denotes the Gaussian distribution with mean µ and diagonal covariance matrix diag( σ 2 ) .

The O is the usual bigO notation, i.e., A n = O ( B n ) for some sequences A n , B n , let it be reals, vectors or matrices, if there exists a constant C &gt; 0 such that for N large enough and all n ≥ N , ∥ A n ∥ ≤ C ∥ B n ∥ . If ∥ A n ∥ ≥ C ∥ B n ∥ , we write A n = Ω( B n ) . We write A n = O P (1) for a sequence of random variables ( A n ) such that, for any ε &gt; 0 , there exists a constant B &gt; 0 such that P ( ∥ A n ∥ &gt; B ) ≤ ε for n large enough.

For any definite positive matrix Σ , we denote by C = Chol (Σ) the unique lower triangular matrix such that CC ⊤ = Σ .

## B Adaptive schedule algorithm

Algorithm 4 Variance control and backtracking strategy

Require: ε ′ &gt; 0 , η ∈ V , η ′ ∈ R m , N ≥ 1 , X 1 , . . . , X N i . i . d . ∼ q η , u &gt; 0

ε/

- 1: ε ← ε ′ 2: while εη ′ +(1 -ε ) η / ∈ V do

2

- 5: η ← εη ′ +(1 -ε ) η
- 3: ε ← 4: end while
- 6: ˆ m ← N -1 ∑ N i =1 f ( X i ) -η ⊤ s ( X i )
- 8: if ˆ v ≥ u then
- 7: ˆ v 2 ← N -1 ∑ N i =1 ( f ( X i ) -ˆ m ) 2
- 9: ε ← min ( ε, u/ ˆ v )
- 10: end if
- 11: return ε

## C Extra details on numerical experiments

## C.1 Runtime analysis

All the experiments were conducted using Python 3.13, jax 0.5 with GPU support, Cuda 12.5, and using float64. The hardware specifications are CPU AMD EPYC 7702 64-Core Processor and GPU NVIDIA A100-PCIE-40GB, except for SONAR, Census and MNIST datasets where EPYC 7713 and NVIDIA A100-PCIE-80GB were used. See Table 1.

Table 1: For all conducted experiments, runtimes and max memory usage, across 5 repetitions. T is the number of iterations and N the number of samples whenever applicable. LR = Logistic regression, BSL = Bayesian Synthetic likelihood, MF Gaussian = mean-field Gaussian, Gaussian = full-covariance Gaussian.

| Experiment                                                                           | Runtime (seconds)   | Runtime (seconds)   | Runtime (seconds)   | max resident set size (memory usage)   |
|--------------------------------------------------------------------------------------|---------------------|---------------------|---------------------|----------------------------------------|
|                                                                                      | mean (std)          | min                 | max                 | (gigabytes)                            |
| BSL Gaussian, Alg. 1, ( N,T ) = (100 , 50) (JAX)                                     | 72 . 9 ( ± 2 . 8 )  | 71 . 5              | 77 . 8              | 1 . 07                                 |
| BSL Truncated MF Gaussian, Alg. 1, (100 , 50) (JAX)                                  | 137 . 5 ( ± 0 . 6 ) | 137 . 3             | 138 . 7             | 1 . 05                                 |
| BSL MCMC, Blackjax (JAX)                                                             | 268 . 1 ( ± 3 . 4 ) | 266 . 5             | 274 . 3             | 1 . 16                                 |
| Variable Selection, Alg. 1 sch. 3 , (5 × 10 4 , 25)                                  | 60 . 8 ( ± 0 . 3 )  | 60 . 3              | 61 . 1              | 0 . 42                                 |
| Variable Selection, SMC                                                              | 290 . 7 ( ± 1 . 7 ) | 284 . 1             | 298                 | 0 . 45                                 |
| LR Gaussian, PIMA , Alg. 1 sch. 3 , (10 4 , 10) (JAX)                                | 1 . 6 ( ± 1 . 4 )   | 1 . 0               | 4 . 1               | 1 . 28                                 |
| // NGD, ( 10 4 , 10 ) (JAX)                                                          | 2 . 2 ( ± 1 . 5 )   | 1 . 51              | 4 . 9               | 3 . 29                                 |
| // Alg. 3 sch. 1 , (10 5 , 100) , (JAX)                                              | 4 . 3 ( ± 1 . 7 )   | 3 . 5               | 7 . 3               | 1 . 28                                 |
| // Alg. 3 sch. 2 , (10 5 , 100) , (JAX)                                              | 3 . 4 ( ± 0 . 1 )   | 3 . 3               | 3 . 6               | 1 . 28                                 |
| // PyMC ADVI, ( T = 10 4 )                                                           | 5 . 9 ( ± 0 . 3 )   | 5 . 5               | 6 . 4               | 0 . 84                                 |
| // GMMVI, ( 10 4 , 10 ) (TensorFlow)                                                 | 4 . 8 ( ± 3 . 8 )   | 3 . 0               | 11 . 6              | 4 . 57                                 |
| LR Gaussian, SONAR , Alg. 3 sch. 1 , (10 5 , 100) , (JAX)                            | 5 . 1 ( ± 0 . 3 )   | 5 . 0               | 5 . 6               | 3 . 11                                 |
| // Alg. 3 sch. 2 , (10 5 , 100) , (JAX)                                              | 5 . 1 ( ± 1 . 3 )   | 4 . 4               | 7 . 4               | 3 . 11                                 |
| // PyMC ADVI, ( 10 4 )                                                               | 9 . 6 ( ± 1 . 3 )   | 7 . 4               | 10 . 5              | 3 . 21                                 |
| LR MF Gaussian, MNIST , Alg. 2 sch. 1 , (10 4 , 500) , (JAX)                         | 19 . 1 ( ± 1 . 0 )  | 18 . 7              | 21 . 8              | 2 . 36                                 |
| // Alg. 2 sch. 5 , (10 4 , 500) , (JAX)                                              | 10 . 4 ( ± 0 . 1 )  | 10 . 3              | 10 . 6              | 2 . 36                                 |
| // Alg. 2 sch. 6 , (10 4 , 500) , (JAX)                                              | 18 . 9 ( ± 1 . 1 )  | 22 . 1              | 18 . 5              | 2 . 36                                 |
| // Blackjax ADVI, (10 4 , 500) , (JAX)                                               | 19 . 5 ( ± 1 . 0 )  | 18 . 9              | 22 . 2              | 3 . 34                                 |
| // NGD (MF), sch. 5 , (10 4 , 500) (JAX)                                             | 25 . 3 ( ± 2 . 2 )  | 29 . 2              | 24 . 1              | 4 . 38                                 |
| LR MF Gaussian, subsampling, Census , Alg 2, sch. 5 , (10 4 , 10 4 ,P = 10 3 ) (JAX) | 12 . 9 ( ± 0 . 2 )  | 12 . 8              | 13 . 3              | 3 . 30                                 |
| // Alg 2, sch. 6 , (10 4 , 10 4 , 10 3 ) (JAX)                                       | 13 . 0 ( ± 0 . 03 ) | 12 . 9              | 13 . 0              | 3 . 30                                 |
| // NGD (MF), sch. 5 , (10 4 , 10 4 , 10 3 ) (JAX)                                    | 70 . 0 ( 1 . 7 )    | 69 . 0              | 73 . 1              | 3 . 33                                 |

## C.2 Logistic regression

Data The Sonar (CC BY 4.0 License) and the Census Income (CC BY 4.0 License) datasets are available in the UCI repository while the Pima dataset (CC0: Public Domain License) is in the example datasets of Python package particles (License MIT v0.4, [53, Ch. 1]) and MNIST (CC BYSA 3.0 License) is available at https://github.com/pjreddie/mnist-csv-png . We use the following standard [e.g., 43] pre-processing strategy for Pima, Sonar and Census-Income datasets: we add an intercept, and we rescale the covariates so that non-binary predictors are centred with standard deviation 0 . 5 , and the binary predictors are centred 0 and range 1 . For the third dataset

±

(MNIST dataset), we restrict ourselves to the binary classification problem by selecting pictures labelled 0 or 8 . The gray-scale features which range between 0 and 255 are normalised to be between 0 and 1 . No intercept is added. For the Census Income dataset, the categorical variables are mapped using one-hot encoding.

Table 2: Logistic regression example: summary of datasets and approximation families, in parentheses the batch-size

| Dataset              | Gaussian family   |   d | n             |
|----------------------|-------------------|-----|---------------|
| Pima                 | full-covariance   |   9 | 768           |
| Sonar                | full-covariance   |  62 | 128           |
| Census (subsampling) | mean-field        |  48 | 49 000 (1000) |
| MNIST                | mean-field        | 784 | 11,774        |

Prior For all datasets except MNIST, the prior π ( β ) is a zero-mean Gaussian distribution with diagonal covariance matrix, and the covariances are set to 25 for all the other covariates, except for the intercept, for which it is set to 400 . For the MNIST dataset, the prior is a Gaussian distribution with zero-mean and covariance matrix 25 I n .

Initialisations, schedules and number of samples The initialisation distributions for all datasets except MNIST are standard normal distributions. The initialisation for the MNIST dataset is N (0 , e -2 I n ) . The learning schedules ( ε t ) are obtained via Algorithm 4 with specific inputs ( u 2 , ε t ) summarised in Table 3 along with the number of samples N .

Table 3: Logistic regression setup. Left: Inputs to Algorithm 4 by dataset. Right: Schedule index reference.

| Dataset   | Algorithm                         | Schedule input ( u 2 , ε t )               | Samples N   |
|-----------|-----------------------------------|--------------------------------------------|-------------|
| Pima      | Alg. 1                            | ( ∞ , 1)                                   | 10 4        |
| Pima      | NGD                               | ( ∞ , 1 / ( t +1))                         | 10 4        |
| Pima      | Alg. 3                            | (10 , 1) , ( ∞ , 1 / ( t +1))              | 10 5        |
| Sonar     | Alg. 3                            | (10 , 1) , ( ∞ , 1 / ( t +1))              | 10 5        |
| MNIST     | Alg. 2                            | (10 , 1) , ( ∞ , 10 - 3 ) , (10 , 10 - 3 ) | 10 4        |
| MNIST     | Blackjax (meanfield_vi), NGD (MF) | ( ∞ , 10 - 3 )                             | 10 4        |
| Census    | Alg. 2                            | (10 , 10 - 3 ) , ( ∞ , 10 - 3 )            | 10 4        |
| Census    | NGD (MF)                          | ( , 10 - 3 )                               | 10 4        |

∞

Figure 4: Logistic regression posterior, Sonar data, full-covariance approximation, LSVI-FC and ADVI implementations. Left: average cost per iteration in seconds as a function of the number of samples N , mean over 5 repetitions with 2 std interval. Right: KL divergence (up to an unknown constant) between current Gaussian variational approximation and the posterior, as a function of t , mean over 100 repetitions with one standard deviation interval.

|   # | Schedule input ( u 2 , ε t )   |
|-----|--------------------------------|
|   1 | (10 , 1)                       |
|   2 | ( ∞ , 1 / ( t +1))             |
|   3 | ( ∞ , 1)                       |
|   4 | (1 , 1)                        |
|   5 | ( ∞ , 10 - 3 )                 |
|   6 | (10 , 10 - 3 )                 |

<!-- image -->

MNIST The PyMC3 (License Apache 2.0 v. 5.22, [6]) implementation fails in this context, and we resort to the stochastic gradient descent (SGD) implementation in Blackjax (License Apache 2.0 v1.2.5, [7]) of the mean-field ADVI Algorithm. For SGD, we set the learning rate to 0 . 001 and the

number of samples for the Monte Carlo gradient estimates to 10 4 . See Figure 5 for the average cost per iteration in seconds, and the same plot as Figure 2 with respect to elapsed time.

<!-- image -->

Number of samples

N

Iteration

t

Figure 5: Logistic regression posterior, MNIST data, diagonal covariance approximation, LSVIMF, NGD (JAX) and Blackjax (meanfield\_vi) implementations. Left: average cost per iteration in seconds as a function of the number of samples N , mean over 5 repetitions with 2 std interval. Right: KLdivergence (up to an unknown constant) between current Gaussian variational approximation and the posterior, as a function of t , mean over 100 repetitions with one standard deviation interval.

In addition, we provide missclassification rate for the logistic regression model using the mean (of the Gaussian approximation) as the regression parameter, see Figure 6.

<!-- image -->

Iteration

Iteration

Figure 6: Logistic regression posterior, MNIST data, diagonal covariance approximation, LSVIMF and Blackjax (meanfield\_vi) implementations. Top: Misclassification rate as a function of the iterations, mean over 100 repetitions with 1 standard deviation. Bottom: same in log-log axis.

Subsampling (Census dataset) At each iteration t , a new batch is sampled uniformly with replacement from the dataset:

<!-- formula-not-decoded -->

where U 1 , . . . , U P ∼ U (1 , . . . , n ) . A new batch is drawn at each iteration. The batch size is P = 10 4 . We also use ˆ f for evaluating the KL loss. See Figure 7.

## C.3 Variable selection

Dataset The Concrete Compressive Strength dataset [54] is made of 1030 observations and 8 initial predictors denoted by C, W, CA, FA, BLAST, FASH, PLAST, and A. We enrich the dataset by adding predictors computed from the existing predictors. 5 new predictors, LG\_C, LG\_W, LG\_CA, LG\_FA, LG\_A, where LG\_X stands for the logarithm of the corresponding feature X. The crossproduct of the predictors is also added, resulting in 78 new predictors. Finally, we add an intercept. The total of possible predictors is d = 92 .

Figure 7: Logistic regression posterior. KL loss for the Census-Income dataset (mean-field, with subsampling), mean over 100 repetitions with 1 standard deviation.

<!-- image -->

Prior The hierarchical prior on β, σ 2 , γ is given by

<!-- formula-not-decoded -->

We follow the recommendations of [55] by setting the hyperparameters to w = 4 . 0 , λ = ˆ σ 2 1 and v 2 = 10 /λ , where ˆ σ 2 1 is the variance estimate of the residuals for the saturated linear model γ = (1 , . . . , 1) .

Close-form expression for π ( γ |D ) For a model γ ∈ { 0 , 1 } d , let Z γ = [ Z i ] i/γ i =1 be the selected covariates and let b γ = Z ⊤ γ y . Consider the Cholesky decomposition C γ,v C ⊤ γ,v = Z ⊤ γ Z γ + v -2 I ∥ γ ∥ 1 , and define the least squares estimate for the residuals based on the model given by γ , ˆ σ 2 γ,v = 1 d ( y ⊤ y -( C -1 γ,v b γ ) ⊤ ( C -1 γ,v b γ )) . Then, the log-posterior for γ up to the log-partition constant is given by

<!-- formula-not-decoded -->

SMC, extra numerical results As a benchmark, we compute the posterior marginal probabilities of inclusion using a waste-free variant of the tempering SMC algorithm of [48] with chain length P = 10 4 and N = 10 5 particles.

Given any probability vector p ∈ [0 , 1] d , we plot the histogram of the variable log( π ∗ ( γ ) /q ( γ | p )) with γ ∼ q ( · | p ) (Bernoulli product). The pendant for the SMC discrete measure is obtained by replacing q with the SMC empirical measure ˆ π ∗ . In Figure 8 we plot the histograms when γ is distributed according to the SMC empirical distribution ˆ π ∗ , and when γ is distributed according to three different mean-field Bernoulli distributions γ ∼ q ( · | p ) : i) p = ( 1 , 1 2 , . . . , 1 2 ) , i.e., the intercept is always included and the other coordinates has 0 . 5 probability to be included, ii) the LSVI estimates, and iii) the marginal posterior probabilities estimated via SMC.

## C.4 BSL and toads displacement model

Model The model assumes that M toads move along a one-dimensional axis during D days. For any day 1 ≤ t ≤ D , the toad labelled by 1 ≤ i ≤ M , has observed position y i,t . During the night of day t +1 , the toad moves according to an overnight displacement, δy i,t which is assumed to be a Lévy-alpha stable distribution with stability parameter α and scale parameter δ . With probability p 0 , the toad takes refuge at y i,t + δy i,t . With probability 1 -p 0 , the toad moves back to one the

Figure 8: Variable selection example: distribution of scores log π ∗ ( γ ) /q ( γ ) when γ ∼ q = ˆ π ∗ , when γ ∼ q = q ( · | p ( i ) ) with p ( i ) given either by i), ii) or iii)).

<!-- image -->

previously explored sites y i,t ′ with t ′ chosen uniformly in 1 , . . . , t . Finally, for any day 1 ≤ t &lt; D the observed position is

<!-- formula-not-decoded -->

with B i,t ∼ Ber ( p 0 ) , t ′ ∼ U{ 1 , . . . , t } and δy i,t ∼ Lévy-alpha ( α, δ ) , all variables being mutually independent. The initial position y (1) i is set to δy (0) i ∼ Lévy-alpha ( α, δ ) . The model is parametrised by θ = ( α, δ, p 0 ) ∈ [1 , 2] × [0 , 100] × [0 , 0 . 9] := Θ . Simulating from the previous model yields the observed data Y = ( y i,t ) 1 ≤ t ≤ D, 1 ≤ i ≤ M .

Summary statistic The summary statistic is the concatenation of 4 sets of statistics of size 12 resulting in a total statistic of dimension 48 . Each subset is computed from the displacement information of lag l for l ∈ { 1 , 2 , 4 , 8 } , denoted by Y l = ( | y i,t -y i,t +1 | ) 1 ≤ t ≤ D -1 , 1 ≤ i ≤ M . If the displacement from t to day t +1 of the toad i , Y ( i,d ) l = | y i,t -y i,t +1 | is less than 10 , it is assumed the toad has not moved. The first statistic is the number of pairs ( i, t ) such that Y ( i,t ) l ≤ 10 . We then compute the median displacement and the log difference between adjacent p -quantiles with p = 0 , 0 . 1 , . . . , 1 for all the displacements greater than 10 .

Truncated Gaussian distributions approximation The dataset Y is generated with ( M,D ) = (66 , 63) and underlying θ ∗ = (1 . 7 , 35 , 0 . 6) . The mean and covariance estimates are obtained with P = 100 samples for each evaluation of the synthetic likelihood. We follow the methodology of [49] and use [56] shrinkage covariance estimate given by ˆ Σ = ˆ D 1 / 2 ( γ ˆ C +(1 -γ ) I ) ˆ D 1 / 2 where ˆ D is the estimated correlation matrices and γ = 0 . 5 is the regularization parameter. The prior distribution is the uniform distribution over Θ . The variational family is the set of truncated Gaussian distributions over Θ with diagonal covariances. The initial distribution has mean µ = (1 . 5 , 50 , 0 . 5) and diagonal covariances σ 2 = (0 . 05 , 10 , 0 . 01) . We run Algorithm 1 with N = 100 samples and T = 50 iterations, the step sizes are obtained by Algorithm 4 with u = 1 and linearly decreasing step sizes.

Full-covariance Gaussian distributions on transformed parameters To constrain the parameters θ , we perform inference on the transformed parameters g ( θ ) = logit ( g i ( θ i )) with g i ( θ i ) = ( θ i -a i ) /b i , with a i , b i such that g i scales θ i to [0 , 1] . The prior distribution on the unconstrained parameters θ ′ is 1 Θ ◦ g -1 ( θ ′ ) × |∇ g -1 ( θ ′ ) | . The variational family is the set of full-covariance Gaussian distributions. The initial distribution for θ ′ has mean µ ′ = (0 , 0 , 0) and covariance matrix Σ ′ = diag(0 . 1 , 0 . 1 , 0 . 1) . The benchmark is obtained via MCMC with random walk step

N (0 , 0 . 1 I 3 ) , the acceptance rate over the chain of length 10 4 is roughly 31% , excluding the first 10 3 states.

## D Proofs

## D.1 First order condition and critical points of the uKL objective

Proof of Proposition 2.3. Injecting π = exp( f ) and q η = exp ( η ⊤ s ) into the objective function, we obtain

<!-- formula-not-decoded -->

Using (23), we have

<!-- formula-not-decoded -->

Writing the first-order optimality condition for the following minimisation problem

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and applying (24), then dividing by Z η &lt; ∞ , yield E η [ ss ⊤ ] η = E η [ fs ] . Let s = (1 , ¯ s ⊤ ) ⊤ be some fixed statistic with first component 1 . Assume that η = ( η (0) , ¯ η ⊤ ) ⊤ ∈ V is a critical point, i.e., ∇ η uKL( q η | π ) = 0 . We have

Injecting η ⊤ s = η (0) + ¯ η ⊤ ¯ s into (26), setting ∇ η (0) uKL( q η | π ) = 0 and normalising by Z η yields from the definition of the KL divergence, we deduce

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Proposition 2.2. We have

<!-- formula-not-decoded -->

Computing the gradient of the KL requires computing the gradients of Z η , log Z η , and Z -1 η . We have ∇ ¯ η Z ¯ η = ∫ ¯ se ¯ η ⊤ ¯ s , and ∇ ¯ η log( Z ¯ η ) = Z -1 ¯ η ∇ ¯ η Z ¯ η = E η [¯ s ] . Similarly, ∇ ¯ η Z -1 ¯ η = -Z -2 ¯ η ∇ ¯ η Z ¯ η = -Z -1 ¯ η E η [¯ s ] . Then, using the previous equalities, the gradient of the KL is

Now, let us compute the gradient of the uKL objective with respect to η = ( η (0) , ¯ η ⊤ ) ⊤ . Using

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

to expand (24) yields

Assume that ∇ η uKL( q η | π ) = 0 , then from (31), we obtain E η [¯ sη (0) + ¯ s ¯ s ⊤ ¯ η ] -E [¯ sf ] = 0 . Reinjecting the previous inequality into the gradient of the KL (29) yields

<!-- formula-not-decoded -->

Injecting the expression for η (0) (27) into (32) yields ∇ ¯ η KL(¯ q ¯ η | ¯ π ) = 0 . Conversely, the previous computations show that if ∇ ¯ η KL(¯ q ¯ η | ¯ π ) = 0 and ∂ η (0) uKL( q η | π ) = 0 , then ∇ η uKL( q η | π ) = 0 .

## D.2 The exact LSVI is a natural gradient descent

Proof of Proposition 2.4. Assumption 2.1 ensures that for any η ∈ V , F η is invertible (minimality assumption), and ensures the differentiability of all the involved functions (regularity). Let us denote by ∇ η l the Jacobian of η ↦→ l ( η ) . Let η ∈ V , using (24), we have

<!-- formula-not-decoded -->

where Z η = Z ( q η ) is the normalisation constant of q η . Let ( η t ) be the sequence obtained via natural gradient descent given by (6). Then, by (6) and (33), we have

<!-- formula-not-decoded -->

Thus, the LSVI iteration with learning schedule ( ε t ) given by (5) is the natural gradient descent ( η t ) with learning schedule ( ε t /Z η ( t ) ) given by (6). Let us now prove (7). We have

<!-- formula-not-decoded -->

By the chain rule and (35), the Jacobian of η ↦→ l ( η ) is

<!-- formula-not-decoded -->

Finally, injecting (36) into (6) yields

<!-- formula-not-decoded -->

which is (7). This shows the first equivalence. Let ω 0 ∈ W , and define ( ω t ) as given by (8). The first order condition on the minimisation problem (8) yields

<!-- formula-not-decoded -->

but ∇ Z ∗ ( ω t ) = η ( ω t ) = η t , thus (38) is exactly (7).

## D.3 The exact LSVI mapping for mean-field Gaussian distributions

Let s ( x ) := (1 , x, x 2 ) ⊤ where x = ( x 1 , . . . , x d ) and x 2 = ( x 2 1 , . . . , x 2 d ) . The set of admissible natural parameters is given by V = R × R d × ( R -\{ 0 } ) d × ⊂ R m , m = 2 d + 1 . Let η = ( η (0) , η (1) , ⊤ , η (2) , ⊤ ) ⊤ ∈ V . The natural mapping from η to ( µ, σ 2 ) is given by T ( η ) := ( -1 2 η (1) ⊗ η (2) , -1 , -1 2 η (2) , -1 ) where ⊗ is the componentwise product and η (2) , -1 is the componentwise inverse of η (2) .

Lemma D.1 (Reparametrisation of the regression in the mean-field case) . Let X ∼ N ( µ, σ 2 ) be a mean-field Gaussian distribution with µ, σ ∈ R d , and σ i &gt; 0 for all i ∈ { 1 , . . . , d } . Let η = ( η (0) , η (1) , ⊤ , η (2) , ⊤ ) ∈ V , η (0) ∈ R , η (1) ∈ R d , η (2) ∈ R d be the natural parameter associated with X for the statistic s : x ∈ R d ↦→ (1 , x, x 2 ) ⊤ ∈ R 1+2 d . Let t be given by (42) . For any z ∈ R d , let

x ( z ) = µ + σ ⊗ z , if Z ∼ N (0 , I ) , then x ( Z ) ∼ N ( µ, σ 2 ) . Let γ = ( γ (0) , γ (1) , ⊤ , γ (2) , ⊤ ) ⊤ ∈ R 2 d +1 , γ (0) ∈ R , γ (1) ∈ R d , γ (2) ∈ R d be defined componentwise by

Then, for any z ∈ R d

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let us identify γ such that (40) is satisfied. Suppose that for all z ∈ R d , we have (40), then

<!-- formula-not-decoded -->

By identifying the factors in front of 1 (group 1 ), the z j 's (group 2 ), and the z 2 j 's (group 3 ), we obtain (39). By injecting (39) into (40), the equality is satisfied.

Theorem D.2 (LSVI mapping ϕ for the mean-field Gaussian distributions) . Let X ∼ N ( µ, σ 2 ) , and η ∈ V be the corresponding natural parameter and let t be given by

<!-- formula-not-decoded -->

Then, the LSVI mapping β := ϕ ( η ) is defined recursively bottom to top by

<!-- formula-not-decoded -->

and γ := E [ t ( Z ) f ( µ + σ ⊗ Z )] , with subcomponents γ = ( γ (0) , γ (1) , ⊤ , γ (2) , ⊤ ) ⊤ , γ (0) ∈ R , γ (1) , γ (2) ∈ R d . In addition, if f admits second-order derivatives such that E X [ f ] &lt; ∞ , ∥ E X [ ∇ f ] ∥ &lt; ∞ , and 0 ≺ -E X [ Diag ( ∇ 2 f ) ] , then ϕ ( η ) defines a Gaussian distribution with mean and variance given by

<!-- formula-not-decoded -->

Proof. We know that ϕ ( η ) realises the minimum of the OLS objective (4), i.e.,

<!-- formula-not-decoded -->

Using Lemma D.1, we can rewrite the regression objective with covariates given by s into a regression with covariates given by t . Using the notations of Lemma D.1, we let γ be given such that γ ⊤ t ( z ) = β ⊤ s ( x ( z )) for all z ∈ R d , and where β = ϕ ( η ) is the unique minimizer of the OLS objective (45). Then,

<!-- formula-not-decoded -->

since E Z [ tt ⊤ ( Z ) ] = I m . Inverting the relation (39) given by Lemma D.1 between γ and β , which is possible since all the σ i 's are strictly positive, we obtain

<!-- formula-not-decoded -->

But β = ϕ ( η ) , this proves the first statement (43) of Theorem D.2. For the second statement, assume that f admits second-order derivatives. Using Stein's Lemma and (46), we obtain,

<!-- formula-not-decoded -->

Injecting (48) into (43), we obtain for

<!-- formula-not-decoded -->

Using the natural mapping T ( η ) = ( -1 2 η (1) η (2) , -1 , -1 2 η (2) , -1 ) , we obtain (44).

## D.4 The exact LSVI mapping for Gaussian distributions

Lemma D.3 (Reparametrisation of the regression in the full-covariance case) . Let X ∼ N ( µ, Σ) be a Gaussian distribution with µ ∈ R d , and Σ ≻ 0 . Let η ∈ V be the natural parameter associated with X for the statistic s : x ∈ R d ↦→ (1 , X, (vec XX ⊤ ) ⊤ ) ⊤ ∈ R 1+ d + d 2 . Let t be given by (16) . For any z ∈ R d , let x ( z ) = µ + Cz with C ∈ R d × d such that CC ⊤ = Σ . If Z ∼ N (0 , I d ) , then x ( Z ) ∼ N ( µ, Σ) , and for any z ∈ R d

<!-- formula-not-decoded -->

with γ = ( γ (0) , γ (1) , ⊤ , γ (2) , ⊤ ) ⊤ ∈ R 1+ d + d ( d +1) / 2 . Furthermore, the components of γ , γ (0) ∈ R , γ (1) ∈ R d , γ (2) ∈ R d ( d +1) / 2 are given by

<!-- formula-not-decoded -->

where and

<!-- formula-not-decoded -->

Proof. The proof is similar to the proof of Lemma D.1. Let us rewrite the regression with respect to Z . Let η = ( η (0) , η (1) , ⊤ , η (2) , ⊤ ) ⊤ ∈ R 1+ d + d 2 with η (0) ∈ R , η (1) ∈ R d , η (2) ∈ R d 2 . Let X = µ + CZ with C such that CC ⊤ = Σ . Rewriting the linear regression on s ( X ) with s ( Z ) , we have

<!-- formula-not-decoded -->

where ˆ γ = (ˆ γ (0) , ˆ γ (1) , ⊤ , ˆ γ (2) , ⊤ ) ⊤ ∈ R 1+ d + d 2 are left to be identified. By identifying the quadratic terms in (54), we have for ˆ γ (2)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used vec ABC = ( C ⊤ ⊗ A ) vec B . Thus,

<!-- formula-not-decoded -->

where we used ( A ⊗ B ) -1 = A -1 ⊗ B -1 . For ˆ γ (1) , expanding the linear term in (54), we have

<!-- formula-not-decoded -->

i.e.,

Regrouping all the constants in (54), we obtain for ˆ γ (0) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, we want to rewrite the regression on s ( Z ) in terms of t ( Z ) where

<!-- formula-not-decoded -->

which satisfies E Z [ tt ⊤ ] = I m ′ with m ′ = d + d ( d +1) / 2 + 1 . We do that in two steps, let

<!-- formula-not-decoded -->

Let ˜ γ = (˜ γ (0) , ˆ γ (1) , ⊤ , ˜ γ (2) , ⊤ ) ⊤ ∈ R 1+ d + d 2 be such that

<!-- formula-not-decoded -->

i.e., keeping only the constant terms and the terms quadratic in Z ,

<!-- formula-not-decoded -->

We set, for any k ≥ 0 , ˜ γ (2) 1+( d +1) k = ˆ γ (2) 1+( d +1) k √ 2 , and ˜ γ (0) = ˆ γ (0) + ∑ d -1 k =0 ˆ γ (2) (1+( d +1) k ) . Then, (62) and (63) are satisfied. To go from t 1 to t , we need to get rid of the coordinates t ( Z ) k = Z i Z j for some i &gt; j , i.e., k ∈ [ dp + 1 , ( d + 1) p ] for some integer p . Let Γ = unvec(ˆ γ (2) ) , and let γ (2) ∈ R d ( d +1) / 2 be defined by

̸

<!-- formula-not-decoded -->

Then γ = [˜ γ (0) , ˆ γ (1) , ⊤ , γ (2) , ⊤ ] ⊤ ∈ R m satisfies

<!-- formula-not-decoded -->

All the previous computations give the expression of γ as a function of η

We now turn to prove Theorem 4.1 using the previous Lemma.

Proof of Theorem 4.1. As in the proof of Theorem D.2, the least squares regression on s ( X ) can be rewritten in terms of t ( Z ) . Then, by applying Lemma D.3, we can map the regressor γ with respect to t , to the regressor with respect to s , given β = ϕ ( η ) . Since E Z [ tt ⊤ ] = I , the OLS simplifies to γ = E Z [ t ( Z ) f ( µ + CZ )] . By Lemma D.3, the mapping from γ to β is given by

<!-- formula-not-decoded -->

.

where or componentwise Γ i,i = γ (2) 1+1 / 2(2 d +2 -i )( i -1) / √ 2 , Γ i,i + k = γ (2) 1+1 / 2(2 d +2 -i )( i -1)+ k / 2 for 1 ≤ i ≤ d and 1 ≤ k ≤ d -i , and Γ i,j = Γ j,i for j &lt; i . Regarding the complexity, the computation of the Cholesky matrix C and its inverse requires O ( d 3 ) operations; consequently, the computation of γ can also be performed in O ( d 3 ) operations. Using (66) to map γ to η , involves computing vec ( C -1 Γ C -⊤ ) and C -⊤ γ (1) , both of which can be performed in O ( d 3 ) .

<!-- formula-not-decoded -->

## D.5 Concentration bounds for the Fisher matrix in the compact case

We now prove a Lemma to control the bias induced by inverting the estimated FIM, conditioned on the event that the estimated FIM is well-conditioned, which happens with high-probability given the number of samples N is large enough. We first prove a version of the Lemma when s is bounded (Lemma D.4), and then tackle the case where s is unbounded but with bounded-moments (Lemma D.5).

Lemma D.4 (Mean error bound for the inverse of ˆ F when s is uniformly bounded) . Let δ ∈ (0 , 1) , N ≥ B (4 / 3 r +2 B ) r -2 log(2 mδ -1 ) , ω ∈ W , and A ( ω ) = [ ∥ F ω -ˆ F ω ∥ &lt; ∥ F -1 ω ∥ -1 ] . Then, under Assumptions 2.1, 3.2, and ∥ s ∥ 2 2 ≤ B , A ( ω ) occurs with probability at least 1 -δ . Furthermore,

<!-- formula-not-decoded -->

where the constant in the bigO term can be chosen independently of ω .

Proof of Lemma D.4 (exponential tail bound). Fix ω ∈ W . For the sake of notation, we drop the subscript in ω but indicate the dependency in N . For any N ≥ 1 , let ˆ F N = N -1 ∑ N i =1 ss ⊤ ( X i ) with X 1 , . . . , X N i.i.d ∼ q . Conditionally on A ( N ) = [ ∥ ˆ F N -F ∥ &lt; ∥ F -1 ∥ -1 ] , ˆ F N = F ( I -( I -F -1 ˆ F N )) is invertible because F is invertible thanks to Assumption 2.1 and 0 &lt; 1 -∥ F -1 ∥∥ F -ˆ F N ∥ ≤ 1 -∥ I -F -1 ˆ F N ∥ = ∥ I -( I -F -1 ˆ F N ) ∥ . Using the Neumann series, we have

<!-- formula-not-decoded -->

Since I -F -1 ˆ F N is an average of N i.i.d random matrices with uniformly bounded second moments thanks to Assumptions 3.1 and 3.2, { √ N ( I -F -1 ˆ F N ) i,j } N ≥ 1 is uniformly integrable (both in N and ω ). By the strong law of large numbers, we have ˆ F N a.s. → F , thus 1 [ A ( N ) ] a.s. → 1 . Since { √ N ( I -F -1 ˆ F N ) i,j } is uniformly integrable, the central limit also holds for the sequence of conditional random matrix components √ N ( I -F -1 ˆ F N ) i,j | A ( N ) . Then for any 1 ≤ i, j ≤ m , √ N ( I -F -1 ˆ F ) i,j | A ( N ) converges in law with uniformly bounded variance. Thus, conditioned on A ( N ) , √ N ( I -F -1 ˆ F ) i,j = O P (1) , which implies that √ N ∥ I -F -1 ˆ F ∥ F = O P (1) . Thus N ∥ I -F -1 ˆ F N ∥ 2 | A ( N ) = O P (1) , and taking the expectation in (69) yields ∥ E [ ˆ F -1 -F -1 | A ( N ) ] ∥ = O ( N -1 ) , again using Assumption 3.2.

<!-- formula-not-decoded -->

By [57, Th. 1.62] with uniform bound ∥ ( ss ⊤ ( X i ) -F ) /N ∥ ≤ 2 B/N and variance ∥ ∑ N i =1 E [(( ss ⊤ ( X i ) -F ) /n ) 2 ] ∥ ≤ B ∥ F ∥ /N , and the definition of r (Assumption 3.2), we have

where to go from the second to the third line, we use ∥ F ∥ ≤ B . Setting N ≥ B (4 / 3 r + 2 B ) r -2 log(2 mδ -1 ) yields P ( ∥ ˆ F N -F ∥ ≥ ∥ F -1 ∥ -1 ) ≤ δ , i.e., P ( A ) ≥ 1 -δ . The bound is independent of ω , and true for any ω ∈ W , finally yielding the result.

Lemma D.5. Under Assumptions 2.1, 3.1 and 3.2, for √ N ≥ r -2 δ -1 ( √ 8 e log( m ) µ 4 ν + 8 eµ 2 4 log( m )) , P ( A ( ω )) ≥ 1 -δ and (68) holds.

Proof of Lemma D.5 (polynomial tail bound). Wefollow the proof of D.4. The CLT for ˆ F still holds thanks to Assumption 3.1, and by the same argument as in D.4, the conditional CLT is still valid. Thus, ∥ E [ I -F -1 F | A ] ∥ = O ( N -1 ) , and the constant inside the bigO notation is also independent on ω using the uniform bounds on the fourth-moment of s . By the definition of r , the Bienaymé-Tchebychev's inequality, and [38, Theorem 3.1], we have

<!-- formula-not-decoded -->

where last line is solely here to simplify the requirement on N . Setting √ N ≥ r -2 δ -1 ( √ 8 e log( m ) µ 4 ν +8 eµ 2 4 log( m )) yields P ( ∥ ˆ F N -F ∥ ≥ ∥ F -1 ∥ -1 ) ≤ δ , i.e., P ( A ) ≥ 1 -δ . The bound is independent of ω , and true for any ω ∈ W .

## D.6 Convergence analysis of the stochastic LSVI algorithm

The proof of Theorem 3.4 relies on Lemmas D.6, D.7, D.9, and D.10. Lemma D.6 states the equivalence between stochastic mirror descent and stochastic natural gradient descent, the proof is very similar to the non-stochastic case (Proposition 2.4). Lemma D.7 gives the general convergence rate for stochastic mirror descent with the presence of an additional bias under the assumption the bias has bounded variance. This is a generalisation of Hanzely and Richtárik [30, Th. 4.5]. Both Lemmas D.9, D.10 are required to handle the two first moments of the bias induced by inverting the FIM estimate. This analysis requires conditioning on the event that the estimated FIMs are well-conditioned, which happens with high-probability (Lemma D.5). Theorem 3.4 follows by successively applying Lemma D.6 and Lemma D.7, the latter requires Lemmas D.9 and D.10.

Lemma D.6 (Equivalence between stochastic mirror descent and stochastic natural gradient descent) . Define the stochastic gradient ˆ ∇ ω l by given that ˆ F ω is invertible. Then (10) is equivalent to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˆ ω t = ω (ˆ η t ) . Furthermore, the previous dynamic is equivalent to

<!-- formula-not-decoded -->

with ˆ η t +1 = η (ˆ ω t +1 ) .

Proof. The first equivalence follows from the same computations as in Proposition 2.4. Let us show that iteration (10) can be recovered as the dual in the natural parameter space of a stochastic mirror descent, i.e., that ˆ η t +1 = η (ˆ ω t +1 ) with (ˆ ω t ) given by (74) recovers (10). The first order condition on (74) gives

<!-- formula-not-decoded -->

However, since ∇ Z ∗ (ˆ ω t +1 ) = η (ˆ ω t +1 ) = ˆ η t +1 , the desired equivalence between the two dynamics follows.

Lemma D.7 (General convergence for biased stochastic mirror descent) . Let us define the bias B t of the stochastic gradient at iteration t by

<!-- formula-not-decoded -->

given that ˆ F ˆ ω t is invertible, and let us denote by m (ˆ ω t ) := ω t +1 , ∗ the exact mirror-descent iterate starting from ˆ ω t , i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assume there exists σ 2 &gt; 0 (to be specified later) such that for any t ≥ 0 ,

Let ε t ≤ 1 L ∧ 1 µ for all t ≥ 0 , let c t = c t -1 ε -1 t -1 ( ε -1 t -µ ) -1 for t ≥ 1 , and let c 0 = 1 . Let C k = ∑ k t =1 c t -1 for k ≥ 1 . Then, under Assumptions 2.1, 2.5, and the additional bounded-noise assumption (78) ,

<!-- formula-not-decoded -->

Proof. Assumption 2.1 allows us to define (77). Under Assumption 2.5 and the boundedness of the gradient estimate (78), we can derive a slightly modified version of the descent lemma [30, Lemma 4.3] which accounts for the presence of the bias. Next line follows from the calculations done in the proof of Hanzely and Richtárik [30, Lemma 4.3]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ω ∗ = argmin ω ∈W l ( ω ) . Since ε -1 t ≥ L , the fourth term is negative. Therefore, (80) becomes

Taking the expectation of (81) gives

<!-- formula-not-decoded -->

Let c t = c t -1 ε -1 t -1 ( ε -1 t -µ ) -1 for t ≥ 1 , and let c 0 = 1 . Let k ≥ 1 and define C k = ∑ k t =1 c t -1 . Since ε t ≤ 1 µ , we have c t ≥ 0 . Multiply by c t ≥ 0 (82) and sum for t ∈ [1 , k ] , then divide by C k ,

<!-- formula-not-decoded -->

We essentially recover Hanzely and Richtárik [30, Th. 4.5], but with the additional bias terms. Finally, (79) follows from (83) and D Z ∗ ( ω ∗ , ω 0 ) = uKL( q ω ∗ | q ω 0 ) .

Remark D.8 . The previous lemma requires a boundedness assumption on the gradient estimate given by (78). This assumption is typically required for proving such descent lemmas, see [20, 30, 36, 37, 58]. In particular, this assumption is implied, by Cauchy-Schwarz inequality, if the gradient estimate ˆ ∇ ω l mapping given by (72) has bounded bias and variance, or directly if the gradient estimate is unbiased. We will prove it is satisfied under Assumptions 3.1 and 3.3 in Lemma D.10.

Lemma D.9 (Controlling the bias terms in (79)) . Let k ≥ 0 , and let A k = ∩ k t =0 A (ˆ ω t ) with A ( ω ) = [ ∥ F ω -ˆ F ω ∥ &lt; ∥ F -1 ω ∥ -1 ] , then under Assumptions 2.1, 3.1, 3.2 and 3.3, for any t ∈ [0 , k ]

<!-- formula-not-decoded -->

Proof. By Cauchy Schwarz inequality, for any t ∈ [0 , k ] ,

Conditionally on A k , ˆ F ˆ ω t is invertible, and by Lemma D.5, there exists C &gt; 0 such that for N large enough, N ∥ E [ F -1 ˆ ω t -ˆ F -1 ˆ ω t | A k , ˆ ω t ] ∥ ≤ C . Consequently,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we used to go from the first to the second line, the independency of z ˆ ω t with ˆ F ˆ ω t conditioned on ˆ ω t and E [ˆ z ˆ ω t | ˆ ω t ] = z ˆ ω t , and to go from the third to the fourth line, we use Lemma D.5, requiring Assumptions 2.1, 3.1 and 3.2, and to go from the fourth to the fifth line, we use bounds on the moment of s 2 and f 2 given by Assumptions 3.1, 3.3:

<!-- formula-not-decoded -->

Taking the expectation of (86) conditioned on A k and plugging it into (85) yields (84).

LemmaD.10 (High-probability uniform bound for the variance of the gradient estimate) . Let ε &gt; 0 . For any ω ∈ W , let M ( ω ) = argmin ω ′ ∈W {∇ ⊤ ω l ( ω ) ω ′ + εD Z ∗ ( ω ′ , ω ) } be the exact mirror-descent iterate starting from ω with step size ε . Similarly, let ˆ M ( ω ) be the mirror-descent using gradient estimate ˆ ∇ ω l given by (72) . Let σ 2 ( ω ) be defined by where all the expectations are taken conditioned on A ( ω ) . Under Assumptions 2.1, 3.1, and 3.3, there exists some constant C &gt; 0 , such that for N large enough (see Lemma D.5), and any ω ∈ W ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some constant C &gt; 0 , and the littleo term is independent of ω, N .

Proof. To obtain the uniform bound on (88), we independently bound both terms in the scalar product. Let ω ∈ W , and let η be the corresponding natural parameter, η = η ( ω ) . Conditionally on A ( ω ) , by the computations done in the proof of Lemmas D.4 and D.5 there exists a constant C &gt; 0 such that for N large enough, N E [ ∥ ˆ F -1 ω -F -1 ω ∥ 2 ] ≤ C , i.e., the mean-square error is O (1 /N ) , and thus the variance E [ ∥ ˆ F -1 ω -E [ ˆ F -1 ω ] ∥ 2 ] is O (1 /N ) .

Using the definition of the stochastic gradient ˆ ∇ l given by (72), and the previous bound, the first term is bounded by:

<!-- formula-not-decoded -->

where we used ( a + b ) 2 ≤ 2 a 2 +2 b 2 from the first to the second line, from the second to the third line the sub-multiplicativity of ∥·∥ , from the third to the fourth ∥ z ω ∥ 2 ≤ mµ 2 4 m 2 2 and the O (1 /N ) bound on the variance of ˆ F -1 ω . Furthermore, we can bound the last term in (90) by

<!-- formula-not-decoded -->

where we used that r -2 = sup ω ∥ F -1 ω ∥ 2 , and E [ ∥ z ω -ˆ z ω ∥ 2 ] = Tr ( Cov ( sf ( X ))) /N with Assumptions 3.1 and 3.3 to state that there exists C ′ &gt; 0 independent on ω such that E [ ∥ z ω -ˆ z ω ∥ 2 ] ≤ C ′ /N . Gathering (90) and (91) yields for the first term of the scalar product:

<!-- formula-not-decoded -->

which in turn can be bounded by some constant C ′′ &gt; 0 independent of ω . Let us tackle the second term inside the scalar product. By Proposition 2.4, and similarly for ˆ M ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under Assumption 2.1, the mapping ω : η ∈ V ↦→ ω ( η ) is differentiable with ∇ η ω = Z η F η = ∫ ss ⊤ q η , see (35). Let us denote by H i = D 2 η ω ( i ) the Hessian of the i -th component application of ω for any 1 ≤ i ≤ m , which is a R m × m matrix given by D 2 η ω ( i ) = ∫ s i ss ⊤ q η , and let D 2 η ω = ( H 1 , H 2 , . . . , H m ) ⊤ be the collection of the Hessian matrices. For any h ∈ R m , let us denote by D 2 η ω [ h, h ] = D 2 η ω [ h ] 2 = ( h ⊤ H 1 h, . . . , h ⊤ H m h ) ⊤ ∈ R m . A Taylor expansion with Lagrange remainder yields,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let ˆ R be the ε 2 remainder term in (95). Then, ˆ R/Z ω is a R m vector whose norm can be uniformly bounded using the uniform bounds on the fourth-moment of s using similar techniques as for the bound on ∥ F ω ∥ (see below), we omit the details. This implies that ε 2 ˆ R = Z ω o ( ε ) with constant in the littleo terms independent on ω . Consequently, for some constant C (4) &gt; 0 , and where we used ∥ F ω ∥ 2 ≤ m 2 µ 4 4 :

using the definition of the Frobesnius norm and then Cauchy Schwarz inequality to bound componentwise F ω . By Cauchy Schwarz inequality, (92) and (96), for N large enough,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This concludes the proof.

With previous Lemmas D.6, D.7, D.9, and D.10 in hand, we can prove the main result.

Proof of Theorem 3.4. Define ¯ ω k as given in the theorem. By convexity of l ,

<!-- formula-not-decoded -->

Combining Lemma D.7 with Lemma D.9 to control the bias terms, we find that the expectation of the RHS in (99) is upper bounded by

<!-- formula-not-decoded -->

where S k,N := m 2 m 1 / 2 µ 4 C k ∑ k -1 t =0 c t ( E [ ∥ ω ∗ -ˆ ω t +1 ∥ 2 | A k ]) 1 / 2 , where the bigO term is independent of k since it is independent of ω 0 = ˆ ω 0 , ˆ ω 1 , . . . ˆ ω k , and σ 2 some upper bound of sup k ≥ 1 σ 2 ( k ) with σ 2 ( k ) satisfying the assumption of Lemma D.7, for all t ≤ k :

<!-- formula-not-decoded -->

By Lemma D.10, we can set

<!-- formula-not-decoded -->

for some constant C independent on N and the sequence ˆ ω 0 , . . . , ˆ ω k -1 .

Let us tackle the terms which depend both upon N and k via the sequence ˆ ω 0 , ˆ ω 1 , . . . , ˆ ω k . By the law of large numbers, as N →∞ , ˆ F ω 0 → F ω 0 and ˆ z ω 0 → z ω 0 almost surely. Then, by the continuous mapping theorem, ˆ ∇ ω l ( ω 0 ) → ∇ ω l ( ω 0 ) almost surely, and thus ω 1 → ω 1 , ∗ = ω ∗ 1 almost surely, where ω ∗ 1 is the first mirror-descent iterate. By induction, we obtain that for any k ≥ 1 , ˆ ω t → ω ∗ t a-s for all t ∈ [1 , k ] , i.e., the finite sequence { ω 0 , . . . , ˆ ω k } converges to the exact mirror-descent sequence { ω 0 , ω ∗ 1 , . . . , ω ∗ k } . We deduce that, almost surely, for all t ≥ 1 , ∥ ω ∗ -ˆ ω t ∥ 2 →∥ ω ∗ -ω ∗ t ∥ 2 since the countable intersection of almost sure events is an almost sure event. By Aubin-Frankowski et al. [Th. 4 31], we know that the Mirror-Descent sequence l ( ω ∗ t ) converges to l ( ω ∗ ) . Since l is strongly-convex, l ( ω ∗ t ) → l ( ω ∗ ) implies that ∥ ω ∗ -ω ∗ t ∥ → 0 as t goes to ∞ . Combining with the previous almost-sure convergence, we obtain that for any k ≥ 1 , the following equality holds almost surely,

<!-- formula-not-decoded -->

with sup k ≥ 1 D k &lt; ∞ . For k ≥ 1 , let U k ⊂ W be the closed-ball of center ω ∗ and of radius 2 × D k , let U 0 = { ω 0 } , and let U be the reunion of U 0 and the ball centered at ω ∗ with radius sup k ≥ 1 D k &lt; ∞ . For each fixed k ≥ 1 , almost surely, when N →∞ , for any 0 ≤ t ≤ k , ˆ ω t ∈ U k , and therefore

<!-- formula-not-decoded -->

Since ω ↦→ Z ω is continuous and U is compact, we have sup ω ∈ U Z ω &lt; ∞ . Then almost surely, as N →∞ , sup k ≥ 1 σ 2 ( k ) → N -1 C sup ω ∈ U Z ω := σ 2 , and therefore almost surely, as N →∞ ,

<!-- formula-not-decoded -->

Almost surely, when N → ∞ , for any 0 ≤ t ≤ k , ∥ ω ∗ -ˆ ω t ∥ 2 ≤ 2 D k , which implies that sup 0 ≤ t ≤ k -1 E [ ∥ ω ∗ -ˆ ω t +1 ∥ 2 | A k ] 1 / 2 &lt; (2 sup k ≥ 1 D k ) 1 / 2 &lt; ∞ . Using ∑ k -1 t =0 c t = C k , and bounding uniformly the summands of S k,N yields

<!-- formula-not-decoded -->

Finally, plugging (105) and (106) into (100) yields the uniform bound over k :

<!-- formula-not-decoded -->

All the constants in the bigO terms can be chosen independently on the sequence of ˆ ω .

Using Proposition D.5 with δ/ ( k +1) and a union bound, we have P ( ∩ k t =0 A (ˆ ω t )) ≥ 1 -δ for the chosen N .

Finally, let us prove the explicit convergence rates for linearly increasing stepsizes ε t = ( L + αt ) -1 , t ≥ 0 . Similarly to Hanzely and Richtárik [Lemma 4.8 30], we distinguish three cases depending on α compared to µ . If α &gt; µ , then C k = Ω( k µ/α ) and ∑ k -1 t =0 c t ε t = O (1) which yields O ( k -µ/α ) + O ( N -1 ) for the RHS of (12). If α = µ , then C k = Ω( k ) and ∑ k -1 t =0 c t ε t = O (log( k )) . If α &lt; µ , then C k = Ω( k µ/α ) and ∑ k -1 t =0 c t ε t = O ( k µ/α -1 ) .

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main contributions of the paper are as follows: (i) KL minimisation within exponential families can be performed via successive linear regressions under tempered variational approximations. This approach is equivalent to natural gradient descent (NGD) and mirror descent (MD) but avoids the need for explicit gradient-based procedures. This is detailed in Section 2 and Proposition 2.4, with relevant prior work cited. (ii) In the Gaussian variational family, exact LSVI can be tailored to eliminate the need to invert the Fisher information matrix. The resulting procedures have computational complexity O ( d 3 ) in the full-covariance case and O ( d ) in the mean-field case, as shown in Section 4, specifically Theorems 4.1 and D.2. (iii) Under standard optimization assumptions, LSVI converges at explicit rates, established in Theorem 3.4 (Section 3). (iv) Empirical results demonstrate that LSVI achieves performance comparable to state-of-the-art variational inference methods and remains effective for non-differentiable target densities, as shown in Section 5 and Appendix C.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: (i) The approach is currently restricted to exponential families. Extending LSVI to more general variational families, such as mixtures of exponential families, is a promising direction for future work. (ii) Convergence guarantees for the stochastic versions of LSVI, namely, LSVI-MF and LSVI-FC, are not yet established. (ii) A comprehensive theoretical analysis of LSVI's convergence when the target distribution π is replaced by an unbiased estimator ˆ π (e.g., via subsampling) remains an open problem. (iii) A comprehensive study of the constants involved in the convergence rates, in particular with respect to the smallest singular value of the FIM r , the latent dimension d and the dimension of the statistic m is left for future work. We believe most of the proofs can be adapted but the analysis would be more involved. See Section 6.

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

Justification: All assumptions are provided in the core manuscript, see Theorem 2.1, Theorem 2.5, Theorem 3.1, Theorem 3.2 and Theorem 3.3. Furthermore, the assumptions are discussed in the core of the paper along with references mentioning existing similar assumptions in the VI literature. The proofs are deferred to the supplementary materials and are divided in several comprehensive steps, including lemmas in order to make the proofreading procedure easier.

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

Justification: Section 5 and Appendix C along with the pseudo-code Algorithms given in Section 3 are sufficient to reproduce all the experimental results. In particular, all input parameters are provided in Table 3.

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

Justification: The paper provides a Python (JAX) package that includes all discussed Algorithms (LSVI Algorithm 1, MF-LSVI Algorithm 2, FC-LSVI Algorithm 3, Variance control for the stepsizes Algorithm 4, NGD with details provided in Section C) as well as scripts to reproduce all the listed experiments. The package is explicitly divided into two parts, variational contains the generic implementations while experiments contains three sub-folders for the three distinct variational problems (logistic regression, variable selection and Bayesian synthetic likelihood). Full-pipeline for the experiments is provided (download and pre-processing of the datasets, inference procedures and post-processing scripts).

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

Justification: The paper provides all the necessary details to reproduce the experiments, including the hyperparameters (the number of samples N , the number of iterations T , the initialisation distributions and the schedules) which are given in Section C. Different schedules have been considered to demonstrate robustness of the proposed methods while the number of samples is set to obtain reasonable numerical stability.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: All experiments were conducted using multiple trials as indicated in the figure labels and the Appendix C. For the logistic regression examples, one standard-deviation confidence intervals are provided over 100 independent realisations. For the variable selection problem, the means and the min-max intervals for the posterior marginal probabilities obtained via LSVI over 100 independent realisations. No statistical assumption is made for uncertainty measurement.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed-form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The full hardware and software specifications are provided in Appendix C, specifically in Table 1, along with Figures 4 and 5, which report experiments runtime and memory usage. All performance statistics are computed using independent realisations for improved robustness. In addition, scripts for measuring the runtime and memory usage of the algorithms can be found in the package ( /experiments/{...}/time.py ). All experiments were successfully performed and are reported in Section 5 and Appendix C. There is the exception of the applicability of ADVI (PyMC3, [6]) on the MNIST dataset, which is explicitly stated in Appendix C. Instead, ADVI as provided by Blackjax [7] was used as a replacement to PyMC3.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: We carefully read through the NeurIPS Code of Ethics, and we see no violation of any guideline.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper focuses on VI methods, emphasizing theoretical analysis and algorithmic implementability. As such, the work is foundational in nature and does not directly pertain to real-world applications or deployments. Given its abstract and theoretical scope, it does not present identifiable positive or negative societal impacts, including concerns related to fairness, privacy, or misuse.

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

Justification: We see no risk in the application of variational inference procedures.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All used Python packages are open-source, have permissive licenses, and are explicitly mentioned both in the manuscript and the code ( pyproject.toml with complete dependency specifications). Specifically, Blackjax, PyMC, and JAX are mentioned in Section 1 and Section 5. Details on the datasets used, licenses and download links, are provided in Appendix C.

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

Answer: [Yes]

Justification: The provided Python (JAX) package for LSVI is well documented and includes a README file with instructions for installation and usage. The license is also included in the package (Apache License 2.0). In addition, we provide usage examples and accompanying commentaries.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No experiment involving human subjects were conducted.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No experiment involving human subjects were conducted.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: There is no mention of LLMs in the manuscript, and no LLM was used.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.