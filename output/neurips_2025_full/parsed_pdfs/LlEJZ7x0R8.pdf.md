## Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free

Stephen Y. Zhang ˚

University of Melbourne

Michael P. H. Stumpf University of Melbourne

## Abstract

We consider the Schrödinger bridge problem which, given ensemble measurements of the initial and final configurations of a stochastic dynamical system and some prior knowledge on the dynamics, aims to reconstruct the 'most likely' evolution of the system compatible with the data. Most existing literature assume Brownian reference dynamics, and are implicitly limited to modelling systems driven by the gradient of a potential energy. We depart from this regime and consider reference processes described by a multivariate Ornstein-Uhlenbeck process with generic drift matrix A P R d ˆ d . When A is asymmetric, this corresponds to a non-equilibrium system in which non-gradient forces are at play: this is important for applications to biological systems, which naturally exist out-of-equilibrium. In the case of Gaussian marginals, we derive explicit expressions that characterise exactly the solution of both the static and dynamic Schrödinger bridge. For general marginals, we propose MVOU-OTFM, a simulation-free algorithm based on flow and score matching for learning an approximation to the Schrödinger bridge. In application to a range of problems based on synthetic and real single cell data, we demonstrate that MVOU-OTFM achieves higher accuracy compared to competing methods, whilst being significantly faster to train.

## 1 Introduction

We are interested in reconstruction of stochastic dynamics of individuals from static population snapshots. This is a central problem with applications arising across the natural and social sciences, whenever longitudinal tracking of individuals over time is either impossible or impractical [56, 35, 48, 43, 34, 1]. In simple terms, consider a system of indistinguishable particles x t P R d undergoing some unobserved temporal dynamics. The practitioner observes the system to have distribution x 0 ' ρ 0 at an initial time t ' 0 and later to be x 1 ' ρ 1 at a final time t ' 1 . The question is then: can we, under some suitable assumptions, reconstruct the continuous-time behaviour of the system for the unobserved time interval 0 ă t ă 1 ?

The Schrödinger bridge problem (SBP), by now a centrepiece of the theoretical literature on this topic, places this task on a theoretical footing in terms of a mathematical formulation of this problem as a large deviations principle on the path space [28] and a (stochastic) least action principle intimately linked to optimal transportation theory [12]. Given a reference process that encodes prior knowledge on the dynamics, the SBP can be understood as identifying the 'most likely' evolution of the system that is compatible with the snapshot observations.

The SBP and related topics have enjoyed a great deal of recent interest from both applied and theoretical perspectives. Many applications arise from biological modelling of cell dynamics [7, 42]. The majority of existing work assumes, either explicitly or implicitly, that the system of interest is potential driven , that is, driven by the gradient of a potential energy [26, 13, 52, 5, 51, 57, 6]. In fact,

˚ Correspondence to syz@syz.id.au

Figure 1: The Schrödinger bridge problem with a multivariate Ornstein-Uhlenbeck reference process (1) can be solved via a generalised entropic transport problem and characterisation of the Q -bridges. For non-Gaussian endpoints, score and flow matching provide a route to building neural approximations without simulation.

<!-- image -->

most studies do not consider any prior drift, corresponding to the setting where the reference process is purely an isotropic Brownian motion. A few studies [5, 45, 15] allow for scalar Ornstein-Uhlenbeck (OU) processes, motivated by applications in the generative modelling domain. Since scalar OU dynamics have unidimensional drift which can always be written as the gradient of a scalar function, all of these are also potential driven systems.

Changing the choice for the reference process provides a route to dealing with systems which are not potential driven, i.e. driven by a drift that is a non-conservative vector field. Abundant motivations for modelling these dynamics arises from biological systems and other kinds of active matter; these exist naturally far from equilibrium [60, 37, 19, 17], exhibiting irreversible dynamics at a non-equilibrium steady state [27]. While a few studies [53, 44, 25] allow for reference processes described by a SDE with a generic drift term, they rely on computationally expensive simulation procedures such as numerical integration and suffer from accuracy issues in high dimensions. Here we explore a complementary approach and concentrate our attention on a family of linear reference dynamics as a middle ground between physical relevance and analytical tractability. Specifically, we consider reference processes arising in a class of linear drift-diffusion dynamics described by the SDE

<!-- formula-not-decoded -->

These processes are also known as multivariate Ornstein-Uhlenbeck (mvOU) processes, often used as a model of non-equilibrium systems in the physics literature [27, 20, 18]. Indeed, for asymmetric drift matrix A the system (1) has a drift that is no longer the gradient of a potential. As an example in the case of isotropic diffusion σ ' σ I , when A is asymmetric and has all eigenvalues with negative real part, a non-equilibrium steady state exists where the dynamics are irreversible and exhibit nonzero probability currents while the population density is unchanged [18, 20]. While allowing a broader range of reference dynamics, the statistics of (1) remain analytically tractable - in particular it is a Markovian Gaussian process with explicit formulae for its mean, covariance, and transition kernel. This provides avenues to efficient solution of the SBP that sidesteps the need for numerical integration, and as we will see, even analytical expressions in the Gaussian case. In particular, the family of SBP problems that we model is strictly larger than the standard (Brownian) SBP, which we recover as a special case when A , m ' 0 .

The contributions of our paper are twofold. Leveraging results on mvOU processes, we characterise for Gaussian endpoints the Gaussian Schrödinger Bridge (GSB) with reference dynamics described by (1). To handle general marginals, we develop a simulation free training procedure based on score and flow matching [52] to solve the SBP. We conclude by demonstrating our results in a range of synthetic and real data examples. We find that our approach solves the generalised SBP faster and more accurately than comparable simulation-based algorithms, at the cost of assuming the form (1) of the reference process. Our findings highlight the tradeoff between (i) analytically tractable but more restrictive models and (ii) more expressive non-parametric models, which are much less tractable to learn, both in terms of computational cost and statistical complexity.

## 2 Background and related work

## 2.1 Schrödinger bridges

In what follows we provide a concise summary of the SBP. For details we refer readers to [28, 12] for in-depth discussion. We work in X Ď R d and denote by P p X q the space of probability distributions on X . Let C pr 0 , 1 s , X q denote the space of continuous paths ω t : r 0 , 1 s ÞÑ X valued in X , and informally we will refer to path measures as probability measures supported on C pr 0 , 1 s , X q . Viewing stochastic processes as random variables valued in C pr 0 , 1 s , X q , path measures prescribe their law. Let Q be a general path measure describing a reference Markov stochastic process. For prescribed initial and final marginals ρ 0 , ρ 1 P P p X q , the Schrödinger bridge problem can be written

<!-- formula-not-decoded -->

where the minimum is taken over all candidate processes P absolutely continuous with respect to Q that are compatible with the observed data at t ' 0 , 1 . This dynamic form of the SBP, while elegant, is unwieldy for practical purposes owing to the formulation on path space. A well known result [28] connects the dynamical SBP with its static counterpart:

<!-- formula-not-decoded -->

Furthermore, the static and dynamic SBP are interchangeable via the disintegration identity regarding the optimal law P ‹ :

<!-- formula-not-decoded -->

where Q xy denotes the law of the Q -bridge conditioned at p 0 , x 0 q and p 1 , x 1 q . In other words, solution of (SBP-dyn) amounts to solving (SBP-static) for P ‹ 01 followed by construction of P ‹ as per (2) by taking mixtures of Q -bridges.

The problem (SBP-static) can be reformulated as an entropy-regularised optimal transportation problem [12] where the effective cost matrix is the log-transition kernel under Q . This admits efficient solution via the Sinkhorn-Knopp algorithm [14] in the discrete case. This provides a practical roadmap to constructing dynamical Schrödinger bridges by first solving (SBP-static), then using construction of Q -bridges with (2) to build a solution of (SBP-dyn).

## 2.2 Probability flows and (score, flow)-matching

Probability flows Consider a generic drift-diffusion process in d dimensions with drift v t p x q and diffusivity σ t , described by an Itô diffusion whose marginal densities evolve following the corresponding Fokker-Planck equation (FPE):

<!-- formula-not-decoded -->

where D t ' 1 2 σ t σ J t is the diffusivity matrix. Equivalently, the FPE can be rewritten in the form of a continuity equation involving a probability flow field u t [30, 3, 2, 52, 31]:

<!-- formula-not-decoded -->

By recognising the form of the continuity equation in (4), it is apparent that the family of marginal distributions t p t p x qu t ě 0 generated by the dynamics specified in (3) are also generated by the probability flow (PF)-ODE :

<!-- formula-not-decoded -->

The central quantity that allows us to convert between the SDE (3) and the PF-ODE (4) is the gradient of the log-density ∇ x log p t p x q ' : s t p x q , also known as the score function . Under mild regularity conditions, knowledge of the score s p x q allows for sampling from p p x q via Langevin dynamics: the SDE d X t ' 1 2 s p x q d t ` d W t has stationary distribution p p x q [46]. While typically one needs to resort to approximations to learn s [55, 46, 47], in setting of the Brownian (and as we will see, the mvOU) bridge, one has closed form expressions for the score and the objective.

Conditional flow matching Stated in its original form, let t ÞÑ p t p x q be a family of marginals on t P r 0 , 1 s satisfying the continuity equation B t p t p x q ' ´ ∇ ¨ p p t p x q u t p x qq , where u t p x q is a time-dependent vector field. Suppose further that p t p x q admits a representation as a mixture

<!-- formula-not-decoded -->

where z ' q p z q is some latent variable and t ÞÑ p t | z p x q are called the conditional probability paths . We introduce conditional flow fields u t | z that generate the conditional probability paths, i.e. B t p t | z p x q ' ´ ∇ ¨ p p t | z p x q u t | z p x | z qq . Then, the insight presented in [30, Theorem 1] is that, in fact,

<!-- formula-not-decoded -->

as can be easily verified by checking the continuity equation. When p t p x q is not tractable but q p z q , p t | z p x q and u t | z p x q are, as is the case for the Schrödinger bridge (2), conditional flow matching is useful: Lipman et al. [30, Theorem 2] prove that minimising

<!-- formula-not-decoded -->

(6)

is equivalent to regression on the true (marginal) vector field u t .

Simulation-free Schrödinger bridges Previous work [52, 39] has exploited the connection (2) between the dynamic and static SBP problems to create solutions to (SBP-dyn) without the need to numerically simulate SDEs. In particular, [52] proposed to utilise score matching and flow matching simultaneously to learn the probability flow and score of the dynamical SBP with a Brownian reference process. Crucially, they exploit availability of closed form expressions for the Brownian bridge and minimise the objective

<!-- formula-not-decoded -->

where π ' P ‹ 01 is the optimal coupling solving (SBP-static). In the above u t |p x 0 ,x 1 q , s t |p x 0 ,x 1 q are, respectively, the flow and score of the Brownian bridge conditioned on p x 0 , x 1 q , for which there are readily accessible closed form expressions [52, Eq. 8]: u t p x q ' 1 ´ 2 t 2 t p 1 ´ t q p x ´ x t q ` p x 1 ´ x 0 q and s t p x q ' p σ 2 t p 1 ´ t qq ´ 1 p x t ´ x q with x t ' t x 1 `p 1 ´ t q x 0 . Motivating the use of the conditional objective (7) in practice, the authors prove [52, Theorem 3.2] that minimising (7) is equivalent to regressing against the SB flow and score. Since P ‹ 01 can be obtained by solving an static entropic optimal transport problem.

## 2.3 Related work

The generalisation of dynamical optimal transport and the Schrödinger bridge to linear reference dynamics has been studied [11, 9, 10] from the viewpoint of stochastic optimal control. However, as was pointed out by [5], these studies have primarily focused on theoretical aspects of the problem, such as existence and uniqueness. In particular, the result for the Gaussian case [9] is in terms of a system of coupled matrix differential equations and does not lend itself to straightforward computation. More generally, forward-backward SDEs corresponding to a continuous-time iterative proportional fitting scheme [53, 44] have been proposed for general reference processes.

For the Gaussian case, the availability of closed form solutions is by now classical [36, 49], and more recent work provides analytical characterisations of Gaussian entropy-regularised transport [5, 22, 32]. [5] provides explicit formulae that characterise the bridge marginals as well as the force (bridge control) for a Brownian reference as well as a class of scalar OU references. To the authors' knowledge, however, all explicit formulae for Gaussian Schrödinger bridges assume either Brownian or scalar OU dynamics. The application of flow matching [30, 3] as a simulation-free technique to learn Schrödinger bridges was proposed in [52], but the authors restrict consideration to a Brownian reference process. Recently [33] apply flow matching for stochastic linear control systems, however they are concerned with interpolating distributions and do not study the SBP.

Finally, concurrent work [38] proposes a related flow matching scheme for approximately solving the Q -SBP where the reference process Q is described by a general, potentially non-linear diffusion. This comes however at the cost of using learned neural approximations for bridges and log-transition densities of Q . Our work treats the complementary case of linear reference dynamics, in which we may avail ourselves of analytical formulae for these quantities.

## 3 Schrödinger bridges for non-equilibrium systems

## 3.1 Multivariate Ornstein-Uhlenbeck bridges

As is evident from (2), the dynamical formulation of the Q -SBP amounts to solution of the static Q -SBP (SBP-static) together with characterisation of the Q -bridges, i.e. the reference process Q conditioned on initial and terminal endpoints. For Q described by a linear SDE of the form (1), explicit formulae for the Q -bridges are available and we state these in the form of the following theorem.

Theorem 1 (SDE characterisation of mvOU bridge, adapted from results in [8]) . Consider the d -dimensional mvOU process (1) . Conditioning on p 0 , x 0 q and p T, x T q , the bridges of this process Y t ' X t |t X 0 ' x 0 , X T ' x T u , 0 ď t ď T are generated by the SDE

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We remark that this characterisation can be found implicitly in the results of [8]; however, their results were conceived for a more general setting of time-varying coefficients and hence is stated in terms of the state transition matrix, which does not in general admit an explicit formula. In our case, computation of the control term Λ t relies upon a unidimensional integral as a key quantity and crucially this is independent of the endpoints, meaning it incurs a one-off computational cost. Although the derivation of Theorem 1 follows the same procedure as [8], we provide details in the Appendix since the work of Chen and Georgiou [8] is written for a control audience. Using the SDE characterisation (8) of the Q -bridge and the fact that it remains a Gaussian process, we obtain explicit formulae for the score and flow fields as well as mean and covariance functions of the Q -bridge. One can furthermore check that the Brownian bridge formulae are recovered as a special case.

Theorem 2 (Score and flow for multivariate Ornstein-Uhlenbeck bridge) . For the mvOU bridge conditioned on p 0 , x 0 q , p T, x T q , denote by p t |p x 0 , x T q the density at time 0 ă t ă T . Then, score function and probability flow of the bridge are respectively

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c t |p x 0 , x T q is the bridge control from Theorem 1 and p µ t |p x 0 , x T q , Σ t |p x 0 , x T q q are the mean and covariance of p t |p x 0 , x T q :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the above, µ x t denotes the mean at time t of an unconditioned mvOU process started from p 0 , x q , and the function Φ t ' ş t 0 e p t ´ s q A σσ J e p t ´ s q A J d s is independent of the endpoints p x 0 , x T q . Consequently, Σ t |p x 0 , x T q depends only on t , so we write Ω t ' Σ t |p¨ , ¨q to make this explicit.

While all our theoretical results are stated in terms of a generic diffusion matrix σ , we remark that in practice one can always assume σ to be diagonal. This is formalised in the following informal lemma, for which we provide a formal statement and proof in Appendix A.1.

Lemma 1. Up to a orthogonal change of coordinates, any linear SDE with generic drift and diffusion matrix, is equal in its law to another linear SDE with a transformed drift and diagonal diffusion matrix.

## 3.2 The Gaussian case

We derive explicit formulae for Gaussian Schrödinger bridges with general linear reference dynamics. Our results generalise those in [5] to mvOU reference processes, and we verify in the appendix that we recover several results of [5, Theorem 3, Table 1] as special cases. Both derivations use the disintegration property (2) together with analytical expressions for the optimal coupling and bridges. Theorem 3 (Characterisation of mvOU-GSB) . Consider a mvOU reference process Q described by (1) and Gaussian initial and terminal marginals at times t ' 0 and t ' T respectively:

<!-- formula-not-decoded -->

In what follows, we write Σ t for the covariance of the unconditioned process (1) started at a point mass. Define the transformed means and covariances

<!-- formula-not-decoded -->

and let C be the cross-covariance term of the Gaussian entropic optimal transport plan [21, Theorem 1] between ρ 0 ' N p a , A q and ρ T ' N p b , B q with unit diffusivity, i.e.

<!-- formula-not-decoded -->

Setting Γ t ' Φ t e p T ´ t q A J Φ ´ 1 T for brevity, define

<!-- formula-not-decoded -->

Then the Q -GSB is a Markovian Gaussian process for 0 ď t ď T with mean and covariance

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore, the Q -GSB (SBP-dyn) is described by a SDE

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Importantly, we show in the Appendix that we recover from the results of Theorem 3 two prior results of [5] - specifically, we obtain results for a Brownian and scalar OU reference process as special cases. Furthermore, in [5] the authors remark that the quantity S J t Ξ ´ 1 t , determining the GSB drift, is symmetric although S J t is asymmetric in general. The interpretation of this result is that the GSB is driven by a time-varying potential and is thus of gradient type. By contrast, we observe empirically that the mvOU-GSB drift is asymmetric when the underlying reference process Q is not of gradient type.

## 3.3 The Non-Gaussian case - score and flow matching

When the endpoints p ρ 0 , ρ T q are non-Gaussian, direct analytical characterisation of the solution to (SBP-dyn) is out of reach. Here we show that combining the exact characterisation of the Q -bridges (Section 3.1) with score and flow matching (Section 2.2) yields a simulation-free estimator of the Schrödinger bridge when Q is a mvOU process (1). Specifically, we propose to first exploit the static problem (SBP-static) which can be efficiently solved via Sinkhorn-Knopp iterations [14] with an analytically tractable cost function. Next, we use the property (2) together with the characterisation of Q -bridges presented in Theorem 2. The approach of [52], which considered a Brownian reference, thus arises as a special case for A ' 0 , m ' 0 , σ ' σ I .

Proposition 1. When Q is a mvOU process (1) , (SBP-static) shares the same minimiser as

<!-- formula-not-decoded -->

We remind that x ÞÑ µ x 0 T is affine and computation of Σ T relies on a one-off time integration that does not depend on the endpoints (Section A.1). Equipped with π the solution to (18), we parameterise the unknown probability flow and score function of (SBP-dyn) by u θ t p x q and s φ t p x q where p θ, φ q are trainable parameters. We seek to minimise the loss

<!-- formula-not-decoded -->

In the practical case when p ρ 0 , ρ T q are discrete, sampling from t, p x 0 , x T q ' U r 0 , T s b π is trivial once (18) is solved. Sampling from p t |p x 0 , x T q and evaluating u t |p x 0 ,x T q , s t |p x 0 ,x T q amounts to invoking the results of Theorem 2. Again, computations of these quantities involve one-off solution of a 1D matrix-valued integral: this can be solved to a desired accuracy and then cached and queried so it poses negligible computational cost, and this is what we do in practice. As in the Brownian case, the following result establishes the connection of the conditional score and flow matching loss (19) to (SBP-dyn), and is a generalisation and restatement of results from [52, Theorem 3.2, Proposition 3.4].

```
Input: Samples t x i u N i ' 1 , t x 1 j u N 1 j from source and target distribution at times t ' 0 , T , mvOU reference parameters p A , m , D ' 1 2 σσ J q , batch size B . Initialise: Probability flow field u θ t p x q , score field s φ t p x q . ˆ ρ 0 Ð N ´ 1 ř N i ' 1 δ x i , ˆ ρ 1 Ð N 1 ´ 1 ř N 1 i ' 1 δ x 1 i Form empirical marginals π Ð sinkhorn p C , ˆ ρ 0 , ˆ ρ 1 , reg ' 1 . 0 q Solve (18) with log-kernel cost while not converged do tp x i , x 1 i qu B i ' 1 Ð sample p π q , t t i u B i ' 1 Ð sample p U r 0 , T sq for 1 ď j ď B do z Ð N p µ t j |p x j , x 1 j q , Σ t j |p x j , x 1 j q q Sample from mvOU-bridge (Thm. 2) ℓ j Ð} u θ t j p z q ´ u t j |p x j , x 1 j q } 2 ` λ t j } s φ t j p z q ´ s t j |p x j , x 1 j q } 2 (Flow, score)-matching loss end for L ' B ´ 1 ř B j ' 1 ℓ j p θ, φ q Ð Step p ∇ θ L, ∇ φ L q . end while
```

Algorithm 1 mvOU-OTFM: score and flow matching for mvOU-Schrödinger Bridges

Theorem 4 (mvOU-OTFM solves (SBP-dyn)) . If p t p x q ą 0 for all p t, x q , (19) shares the same gradients as the unconditional loss: L p θ, φ q ' E t P U r 0 ,T s E z ' p t r} u θ t p z q ´ u t p z q} 2 ` λ t } s φ t p z q ´ s t p z q} 2 s . Furthermore, if the coupling π solves (SBP-static) as given in Proposition 1, then for p u θ t , s φ t q achieving the global minimum of (19) , the solution to (SBP-dyn) is given by the SDE d X t ' p u θ t p X t q ` Ds φ t p X t qq d t ` σ d B t .

We stress here that the linearity assumption is only imposed on the reference process Q in (1), and not on the SBP solution P . In the general non-Gaussian case the mvOU-SBP dynamics will still be nonlinear, much in the same way as for the Brownian SBP.

Further, a non-linear reference dynamics can be connected to linear dynamics of the form (1) by linearisation about a fixed point of the drift. That is, for a smooth reference drift f p x q with x 0 a fixed point, f p x 0 q ' 0 , one has f p x q ' pB x f qp x 0 qp x ´ x 0 q ` O p} x ´ x 0 } 2 q . This naturally motivates the study of mvOU processes with A ' pB x f qp x 0 q and m ' x 0 .

## 4 Results

Gaussian marginals: benchmarking accuracy As a first visual illustration of Theorem 3, we show in Fig. 2 the family of marginals solving (SBP-dyn) for the same pair of Gaussian marginals in dimension d ' 10 , where the reference process Q is taken to be (a)(i) a mvOU process with high-dimensional rotational drift and (b)(i) a standard Brownian motion. While both provide valid interpolations of ρ 0 to ρ 1 , it is immediately clear that the dynamics are completely different. To provide some further insight, we show in (a)(ii) and (b)(ii) the time-dependent vector field generating each bridge. According to Theorem 3, both processes are time-dependent linear SDEs of the form (17). For the mvOU reference the asymmetric nature of the drift matrix is evident, while for the Brownian reference the symmetry of the drift matrix corresponds to potential driven dynamics [5].

In the Gaussian case, our explicit formulae for the OU-GSB allow us to empirically quantify the accuracy of our neural solver. Since existing computational methods for solving the Q -Schrödinger bridge for non-gradient Q are limited, we compare MVOU-OTFM against Iterative Proportional Maximum Likelihood (IPML) [53] and Neural Lagrangian Schrödinger Bridge (NLSB) [25]. The former is based on a forward-backward SDE characterisation of the Q -SB and alternates between simulation and drift estimation using Gaussian processes, while the latter uses a Neural SDE framework. For d ranging from 2 to 100, we solve (SBP-dyn) with each method and report in Table 1 the average marginal error, measured in the Bures-Wasserstein metric [49] and the average vector field error in L 2 . In all cases, we find that MVOU-OTFM achieves the highest accuracy among all solvers considered. Among the others, NLSB is generally the most accurate followed by IPML, while BM-OTFM performs the poorest since the reference process is misspecified.

In addition to being more accurate, MVOU-OTFM is extremely fast to train owing to being simulation free, taking approximately 1-2 minutes to train on CPU for d ' 50 , regressing the score and flow networks against (19). NLSB took 15+ minutes to train on GPU requiring backpropagation through

SDE solves, while IPFP requires iterative SDE simulation. On the other hand, our approach directly exploits the exact solution to Q -bridges rather than relying on numerical approximations.

<!-- image -->

Figure 2: Gaussian Schrödinger Bridges. (a) (i) Marginals of the Gaussian Schrödinger bridge ( d ' 10 ) with mvOU reference (MVOU-GSB) (ii) Time dependent vector field generating the bridge. (b) Same as (a) but for Brownian reference.

Table 1: Marginal error measured in Bures-Wasserstein metric and force error measured in L 2 norm for the Schrödinger bridge learned between two Gaussian measures in varying dimension d .

|     | Marginal error ( BW 2 2 )   | Marginal error ( BW 2 2 )   | Marginal error ( BW 2 2 )   | Marginal error ( BW 2 2 )   | Marginal error ( BW 2 2 )   | Force error ( L 2 )   | Force error ( L 2 )   | Force error ( L 2 )   | Force error ( L 2 )   |
|-----|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| d   | MVOU-OTFM                   | BM-OTFM                     | IPML ( Ð )                  | IPML ( Ñ )                  | NLSB                        | MVOU-OTFM             | BM-OTFM               | IPML                  | NLSB                  |
| 2   | 0.19 ˘ 0.17                 | 8.40 ˘ 0.77                 | 5.55 ˘ 1.53                 | 5.65 ˘ 1.41                 | 1.21 ˘ 0.18                 | 3.56 ˘ 0.25           | 12.23 ˘ 0.27          | 10.31 ˘ 0.45          | 7.59 ˘ 0.33           |
| 5   | 0.23 ˘ 0.16                 | 9.06 ˘ 0.66                 | 6.34 ˘ 5.64                 | 3.24 ˘ 0.98                 | 1.16 ˘ 0.26                 | 3.66 ˘ 0.19           | 12.27 ˘ 0.12          | 10.08 ˘ 0.75          | 7.72 ˘ 0.15           |
| 10  | 0.59 ˘ 0.36                 | 8.93 ˘ 0.55                 | 3.00 ˘ 0.73                 | 3.00 ˘ 0.63                 | 1.36 ˘ 0.13                 | 3.82 ˘ 0.13           | 12.33 ˘ 0.13          | 10.96 ˘ 1.11          | 8.15 ˘ 0.25           |
| 25  | 0.84 ˘ 0.22                 | 8.81 ˘ 1.19                 | 6.97 ˘ 2.01                 | 5.03 ˘ 0.60                 | 2.97 ˘ 1.24                 | 4.67 ˘ 0.16           | 12.54 ˘ 0.16          | 12.42 ˘ 0.80          | 11.02 ˘ 2.41          |
| 50  | 2.21 ˘ 0.36                 | 11.74 ˘ 0.37                | 9.03 ˘ 0.21                 | 8.32 ˘ 0.63                 | 6.39 ˘ 0.13                 | 6.25 ˘ 0.17           | 13.26 ˘ 0.15          | 14.14 ˘ 1.00          | 14.40 ˘ 0.02          |
| 100 | 6.84 ˘ 0.78                 | 15.14 ˘ 0.95                | 16.19 ˘ 1.87                | 14.38 ˘ 0.38                | 17.40 ˘ 0.13                | 10.45 ˘ 0.18          | 15.53 ˘ 0.14          | 15.30 ˘ 0.39          | 16.49 ˘ 0.07          |

Figure 3: Non-Gaussian marginals. (a)(i) Sampled trajectories from the mvOU-SB learned between non-Gaussian marginals in d ' 10 , shown in p t, x 0 q coordinates with score field shown in background. (a)(ii, iii) Sampled SDE (stochastic) and PF-ODE (deterministic) trajectories shown in p x 0 , x 1 q coordinates, and reference drift shown in background. (b) Same as (a) but for Brownian reference.

<!-- image -->

Non-Gaussian marginals We demonstrate an application of MVOU-OTFM for general nonGaussian marginals in Fig. 3, where the reference dynamics are the same as those in Fig. 2. Again, the contrast between the mvOU and Brownian reference dynamics is clear, and we show the distinction between sampling of trajectories via the SDE (3) and PF-ODE formulations (4).

Repressilator dynamics with iterated reference As an application to a system of biophysical importance, we consider the repressilator [16], a model of a synthetic gene circuit exhibiting oscillatory, and hence non-equilibrium, dynamics in cells. This system is composed of three genes in a loop, in which each gene represses the activity of the next (Fig. 4(a)). This model, implemented as a SDE (162), was also studied by [44], which introduced a method, 'Schrödinger Bridge with Iterative Reference Refinement' (SBIRR) building upon IPFP [53] to solve a series of Schrödinger bridge problems whilst simultaneously learning an improved reference Q within a parametric family. In particular, they propose to alternate between solving the Q -SBP, using IPFP as a subroutine, and fitting of a parametric global drift describing Q . The same approach can be used with mvOU-OTFM, which we present in Algorithm 2 in the Appendix: we propose to apply Algorithm 1 as a subroutine, and solve a regularised linear regression problem to update the mvOU reference parameters p A , m q at each outer step.

Figure 4: Repressilator dynamics. (a) Repressilator population snapshots and trajectories. (b) (i) Ground truth vector field. (ii, iii) Inferred multi-marginal SB vector field with (fitted mvOU, Brownian) references. (c) Ground truth linearisation of system and drift A learned by mvOU-OTFM. (d) Leave-one-out error by iteration. (e) Illustration of leave-one-out interpolation between two example timepoints p i ´ 1 (blue), p i ` 1 (green) with learned mvOU reference vs. Brownian reference.

<!-- image -->

Table 2: Repressilator leave-one-out interpolation error for mvOU-OTFM and SBIRR.

| Error metric    | Iterate 0   | Iterate 1   | Iterate 2   | Iterate 3   | Iterate 4   | SBIRR (mvOU)   | SBIRR (MLP)   |
|-----------------|-------------|-------------|-------------|-------------|-------------|----------------|---------------|
| EMD             | 3.38 ˘ 1.52 | 2.22 ˘ 1.12 | 1.59 ˘ 0.66 | 1.49 ˘ 0.64 | 1.40 ˘ 0.57 | 2.10 ˘ 0.74    | 1.67 ˘ 0.95   |
| Energy distance | 1.86 ˘ 1.06 | 1.29 ˘ 0.86 | 1.03 ˘ 0.65 | 0.95 ˘ 0.58 | 0.89 ˘ 0.55 | 1.39 ˘ 0.82    | 1.10 ˘ 0.86   |

We use a similar setup to [44] and sample snapshots for T ' 10 instants from the repressilator system. We show in Fig. 4 the sampled data along with example trajectories. We run Algorithm 2 for 5 iterations and we compare in Fig. 4(b) the ground truth vector field to the SB vector field, both the fitted mvOU reference (at the final iterate of Alg. 2) and for a Brownian reference (as the first iterate of Alg. 2 or equivalently the output of [52]). As was found by [44, 58], iterated fitting of a global autonomous vector field for the reference process leverages multi-timepoint information and allows reconstruction of dynamics better adapted to the underlying system.

We reason that the 'best' mvOU process to describe the dynamics should resemble the linearisation of the repressilator system about its fixed point. Comparing the Jacobian of the repressilator system to the fitted drift matrix A in Fig. 4(c), shows a clear resemblance - in particular the cyclic pattern of activation and inhibition is recovered. For a quantitative assessment of performance, we show in Fig. 4(d) and Table 2 the averaged leave-one-out error for a marginal interpolation task: for each 1 ă i ă T , Alg. 2 is run on the T ´ 1 remaining timepoints with timepoint t i held out. The trained model is evaluated on predicting t i from t i ´ 1 . For both the earth mover's distance (EMD) and energy distance [41] we find that prediction accuracy improves with each additional iteration of Alg. 2.

We compare to SBIRR [44] with two choices of reference process: (i) mvOU and (ii) a general reference family parameterised by a feedforward neural network. The former scenario is thus comparable to mvOU-OTFM, whilst the latter considers a wider class of reference processes. We find that both methods offer an improvement on the null reference (iteration 0 of Alg. 2), and as expected SBIRR with a neural reference family outperforms the mvOU reference family. However, mvOU-OTFM achieves a higher accuracy than either of these methods which suggests that, while SBIRR is able to handle general reference processes and is thus more flexible, mvOU-OTFM trains more accurately. Furthermore, mvOU-OTFM is significantly faster, training in 1-2 minutes on CPU. By comparison, training of SBIRR requires at least several minutes with GPU acceleration.

Dynamics-resolved single cell data We next apply our framework to a single-cell RNA sequencing (scRNA-seq) dataset [4] of the cell cycle, where partial experimental quantification of the vector field is possible via metabolic labelling, i.e. for each observed cell state x i we also have a velocity estimate ˆ v i [40]. We consider N ' 2793 cells in d ' 30 PCA dimensions (Fig. 5(a)(i)). For each cell

Figure 5: Cell cycle scRNA-seq. (a) Streamlines of (i) transcriptomic vector field calculated from metabolic labelling data (ii) Fitted mvOU reference process, and Schrödinger bridge drift v SB learned by OTFM with (iii) mvOU reference and (iv) Brownian reference. (b) Marginal interpolation error between first and last snapshot as a function of the reference velocity scale parameter, γ .

<!-- image -->

an experimental reading of its progression along the cell cycle is available via fluorescence, and cells are binned into T ' 5 snapshots using this coordinate. From the paired state-velocity data t x i , ˆ v i u we fit parameters p A , m q of a mvOU process (1) by ridge regression (Fig. 5(a)(ii)). A limitation of 'RNA velocity' is that estimation of vector field direction is often significantly more reliable than magnitude [59]. We introduce therefore an additional 'scale' parameter γ ą 0 by which to scale the mvOU drift γ A . We reason that this amounts to a choice of matching the disparate and a priori unknown timescales of the snapshots and velocity data.

In order to select an appropriate scale, for 0 ď γ ď 100 we use the mvOU-GSB to interpolate between the first and last snapshots (see Fig. 7 for an illustration) and calculate the Bures-Wasserstein distance between the interpolant and each intermediate marginal (Fig. 5(b)). From this it is clear that choosing γ too small or too large leads to a mvOU-GSB that does not fit well the data, while a broad range of approximately 30 ď γ ď 70 yields good behaviour. We fix γ ' 50 and use mvOU-OTFM to solve a multi-marginal SBP with reference parameters p γ A , m q . From Fig. 5(a)(iii) we see that the expected cyclic behaviour is recovered. On the other hand, using a Brownian reference (equivalently, setting γ ' 0 ) fails to do so.

Finally, to further demonstrate robustness of our approach to the dimension d , we carry out the interpolation analysis for d ' 50 , 100 and provide numerical results in Table 4. These settings of moderately high dimensionality are typical for single cell studies, where PCA projection to 20-50 dimensions is routinely used prior to any downstream analysis. In all cases, we are able to run mvOU-OTFM within minutes on CPU.

## 5 Discussion

Motivated by applications to non-equilibrium dynamics, we study the Schrödinger bridge problem where the reference process is taken to be in a family of multivariate Ornstein-Uhlenbeck (mvOU) processes. These processes are able to capture some key behaviours of non-equilibrium systems whilst still being analytically tractable. Leveraging the tractability, we derive an exact solution of the bridge between Gaussian measures for mvOU reference. We extend our approach to non-Gaussian measures using a score and flow matching approach that avoids the need for costly numerical simulation. We showcase the improvement in both accuracy and speed of our approach in a variety of settings, both low and high-dimensional scenarios.

Limitations A key limitation of our method is that our analytical expressions involve matrix inversions and powers of positive definite matrices which scale as O p d 3 q with the dimension d . The same limitation was identified in [5] as also applying to their approach. In practice, we find that application to problems with d ď 100 work well. Scaling our methods to even higher dimensional settings is left to future work, and we expect that tools from the Gaussian process literature will be helpful for this. Regarding flow matching, the link between Alg. 1 and (SBP-dyn) requires the solution to (SBP-static) to be computed exactly. In all our experiments, we achieve this by computing couplings using all available samples. In large-scale settings where the number of data points N is very large or possibly streaming, a common approach is to use minibatch couplings [50]. In such settings, the use of minibatches results in learned flows being biased. An in-depth discussion of this is provided in [24], which advocates for the infrequent computation of OT couplings on large batch sizes rather than the commonly used per-iteration minibatch OT.

## References

- [1] Guy J Abel and Nikola Sander. Quantifying global international migration flows. Science , 343(6178):1520-1522, 2014.
- [2] Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797 , 2023.
- [3] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571 , 2022.
- [4] Nico Battich, Joep Beumer, Buys De Barbanson, Lenno Krenning, Chloé S Baron, Marvin E Tanenbaum, Hans Clevers, and Alexander Van Oudenaarden. Sequencing metabolically labeled transcripts in single cells reveals mrna turnover strategies. Science , 367(6482):1151-1156, 2020.
- [5] Charlotte Bunne, Ya-Ping Hsieh, Marco Cuturi, and Andreas Krause. The schrödinger bridge between gaussian measures has a closed form. In International Conference on Artificial Intelligence and Statistics , pages 5802-5833. PMLR, 2023.
- [6] Charlotte Bunne, Laetitia Papaxanthos, Andreas Krause, and Marco Cuturi. Proximal optimal transport modeling of population dynamics. In International Conference on Artificial Intelligence and Statistics , pages 6511-6528. PMLR, 2022.
- [7] Charlotte Bunne, Geoffrey Schiebinger, Andreas Krause, Aviv Regev, and Marco Cuturi. Optimal transport for single-cell and spatial omics. Nature Reviews Methods Primers , 4(1):58, 2024.
- [8] Yongxin Chen and Tryphon Georgiou. Stochastic bridges of linear systems. IEEE Transactions on Automatic Control , 61(2):526-531, 2015.
- [9] Yongxin Chen, Tryphon T Georgiou, and Michele Pavon. Optimal steering of a linear stochastic system to a final probability distribution, part i. IEEE Transactions on Automatic Control , 61(5):1158-1169, 2015.
- [10] Yongxin Chen, Tryphon T. Georgiou, and Michele Pavon. Optimal steering of a linear stochastic system to a final probability distribution, part ii. IEEE Transactions on Automatic Control , 61(5):1170-1180, 2016.
- [11] Yongxin Chen, Tryphon T Georgiou, and Michele Pavon. Optimal transport over a linear dynamical system. IEEE Transactions on Automatic Control , 62(5):2137-2152, 2016.
- [12] Yongxin Chen, Tryphon T Georgiou, and Michele Pavon. Stochastic control liaisons: Richard sinkhorn meets gaspard monge on a schrodinger bridge. Siam Review , 63(2):249-313, 2021.
- [13] Lénaïc Chizat, Stephen Zhang, Matthieu Heitz, and Geoffrey Schiebinger. Trajectory inference via mean-field langevin in path space. Advances in Neural Information Processing Systems , 35:16731-16742, 2022.
- [14] Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems , 26, 2013.
- [15] Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrödinger bridge with applications to score-based generative modeling. Advances in Neural Information Processing Systems , 34:17695-17709, 2021.
- [16] Michael B Elowitz and Stanislas Leibler. A synthetic oscillatory network of transcriptional regulators. Nature , 403(6767):335-338, 2000.
- [17] Xiaona Fang, Karsten Kruse, Ting Lu, and Jin Wang. Nonequilibrium physics in biology. Reviews of Modern Physics , 91(4):045004, 2019.
- [18] Matthieu Gilson, Enzo Tagliazucchi, and Rodrigo Cofré. Entropy production of multivariate ornstein-uhlenbeck processes correlates with consciousness levels in the human brain. Physical Review E , 107(2):024121, 2023.

- [19] Federico S Gnesotto, Federica Mura, Jannes Gladrow, and Chase P Broedersz. Broken detailed balance and non-equilibrium dynamics in living systems: a review. Reports on Progress in Physics , 81(6):066601, 2018.
- [20] Claude Godrèche and Jean-Marc Luck. Characterising the nonequilibrium stationary states of ornstein-uhlenbeck processes. Journal of Physics A: Mathematical and Theoretical , 52(3):035002, 2018.
- [21] Hicham Janati. Advances in Optimal transport and applications to neuroscience . PhD thesis, Institut Polytechnique de Paris, 2021.
- [22] Hicham Janati, Boris Muzellec, Gabriel Peyré, and Marco Cuturi. Entropic optimal transport between unbalanced gaussian measures has a closed form. Advances in neural information processing systems , 33:10468-10479, 2020.
- [23] Donald E Kirk. Optimal control theory: an introduction . Courier Corporation, 2004.
- [24] Michal Klein, Alireza Mousavi-Hosseini, Stephen Zhang, and Marco Cuturi. On fitting flow models with large sinkhorn couplings. arXiv preprint arXiv:2506.05526 , 2025.
- [25] Takeshi Koshizuka and Issei Sato. Neural lagrangian schr z " odinger bridge: Diffusion modeling for population dynamics. arXiv preprint arXiv:2204.04853 , 2022.
- [26] Hugo Lavenant, Stephen Zhang, Young-Heon Kim, and Geoffrey Schiebinger. Towards a mathematical theory of trajectory inference. arXiv preprint arXiv:2102.09204 , 2021.
- [27] Melvin Lax. Fluctuations from the nonequilibrium steady state. Reviews of modern physics , 32(1):25, 1960.
- [28] Christian Léonard. A survey of the schrodinger problem and some of its connections with optimal transport. arXiv preprint arXiv:1308.0215 , 2013.
- [29] Anders Lindquist. On feedback control of linear stochastic systems. SIAM Journal on Control , 11(2):323-343, 1973.
- [30] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 , 2022.
- [31] Suryanarayana Maddu, Victor Chardès, Michael Shelley, et al. Inferring biological processes with intrinsic noise from cross-sectional data. arXiv preprint arXiv:2410.07501 , 2024.
- [32] Anton Mallasto, Augusto Gerolin, and Hà Quang Minh. Entropy-regularized 2-wasserstein distance between gaussian measures. Information Geometry , 5(1):289-323, 2022.
- [33] Yuhang Mei, Mohammad Al-Jarrah, Amirhossein Taghvaei, and Yongxin Chen. Flow matching for stochastic linear control systems. arXiv preprint arXiv:2412.00617 , 2024.
- [34] Christopher E Miles, Scott A McKinley, Fangyuan Ding, and Richard B Lehoucq. Inferring stochastic rates from heterogeneous snapshots of particle positions. Bulletin of mathematical biology , 86(6):74, 2024.
- [35] Boris Muzellec, Richard Nock, Giorgio Patrini, and Frank Nielsen. Tsallis regularized optimal transport and ecological inference. In Proceedings of the AAAI conference on artificial intelligence , volume 31, 2017.
- [36] Ingram Olkin and Friedrich Pukelsheim. The distance between two random vectors with given dispersion matrices. Linear Algebra and its Applications , 48:257-263, 1982.
- [37] Juan MR Parrondo, Christian Van den Broeck, and Ryoichi Kawai. Entropy production and the arrow of time. New Journal of Physics , 11(7):073008, 2009.
- [38] Katarina Petrovi´ c, Lazar Atanackovic, Kacper Kapusniak, Michael M Bronstein, Joey Bose, and Alexander Tong. Curly flow matching for learning non-gradient field dynamics. In ICLR 2025 Workshop on Machine Learning for Genomics Explorations , 2025.

- [39] Aram-Alexandre Pooladian and Jonathan Niles-Weed. Plug-in estimation of schr z " odinger bridges. arXiv preprint arXiv:2408.11686 , 2024.
- [40] Xiaojie Qiu, Yan Zhang, Jorge D Martin-Rufino, Chen Weng, Shayan Hosseinzadeh, Dian Yang, Angela N Pogson, Marco Y Hein, Kyung Hoi Joseph Min, Li Wang, et al. Mapping transcriptomic vector fields of single cells. Cell , 185(4):690-711, 2022.
- [41] Maria L Rizzo and Gábor J Székely. Energy distance. wiley interdisciplinary reviews: Computational statistics , 8(1):27-38, 2016.
- [42] Geoffrey Schiebinger. Reconstructing developmental landscapes and trajectories from singlecell data. Current Opinion in Systems Biology , 27:100351, 2021.
- [43] Geoffrey Schiebinger, Jian Shu, Marcin Tabaka, Brian Cleary, Vidya Subramanian, Aryeh Solomon, Joshua Gould, Siyan Liu, Stacie Lin, Peter Berube, et al. Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. Cell , 176(4):928-943, 2019.
- [44] Yunyi Shen, Renato Berlinghieri, and Tamara Broderick. Multi-marginal schr z " odinger bridges with iterative reference refinement. arXiv preprint arXiv:2408.06277 , 2024.
- [45] Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schr z " odinger bridge matching. arXiv preprint arXiv:2303.16852 , 2023.
- [46] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems , 32, 2019.
- [47] Yang Song, Sahaj Garg, Jiaxin Shi, and Stefano Ermon. Sliced score matching: A scalable approach to density and score estimation. In Uncertainty in artificial intelligence , pages 574-584. PMLR, 2020.
- [48] Andrew M Stuart and Marie-Therese Wolfram. Inverse optimal transport. SIAM Journal on Applied Mathematics , 80(1):599-619, 2020.
- [49] Asuka Takatsu. Wasserstein geometry of gaussian measures. 2011.
- [50] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid RectorBrooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482 , 2023.
- [51] Alexander Tong, Jessie Huang, Guy Wolf, David Van Dijk, and Smita Krishnaswamy. Trajectorynet: A dynamic optimal transport network for modeling cellular dynamics. In International conference on machine learning , pages 9526-9536. PMLR, 2020.
- [52] Alexander Tong, Nikolay Malkin, Kilian Fatras, Lazar Atanackovic, Yanlei Zhang, Guillaume Huguet, Guy Wolf, and Yoshua Bengio. Simulation-free schr z " odinger bridges via score and flow matching. arXiv preprint arXiv:2307.03672 , 2023.
- [53] Francisco Vargas, Pierre Thodoroff, Austen Lamacraft, and Neil Lawrence. Solving schrödinger bridges via maximum likelihood. Entropy , 23(9):1134, 2021.
- [54] Pat Vatiwutipong and Nattakorn Phewchean. Alternative way to derive the distribution of the multivariate ornstein-uhlenbeck process. Advances in Difference Equations , 2019:1-7, 2019.
- [55] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation , 23(7):1661-1674, 2011.
- [56] Caleb Weinreb, Samuel Wolock, Betsabeh K Tusi, Merav Socolovsky, and Allon M Klein. Fundamental limits on dynamic inference from single-cell snapshots. Proceedings of the National Academy of Sciences , 115(10):E2467-E2476, 2018.
- [57] Grace Hui Ting Yeo, Sachit D Saksena, and David K Gifford. Generative modeling of single-cell time series with prescient enables prediction of cell trajectories with interventions. Nature communications , 12(1):3222, 2021.

- [58] Stephen Y Zhang. Joint trajectory and network inference via reference fitting. In Machine Learning in Computational Biology , pages 72-85. PMLR, 2024.
- [59] Shijie C Zheng, Genevieve Stein-O'Brien, Leandros Boukas, Loyal A Goff, and Kasper D Hansen. Pumping the brakes on rna velocity by understanding and interpreting rna velocity estimates. Genome biology , 24(1):246, 2023.
- [60] Ligang Zhu, Songlin Yang, Kun Zhang, Hong Wang, Xiaona Fang, and Jin Wang. Uncovering underlying physical principles and driving forces of cell differentiation and reprogramming from single-cell transcriptomics. Proceedings of the National Academy of Sciences , 121(34):e2401540121, 2024.

## A Theoretical results

| Quantity                                                | Description                                                 | Introduced in   |
|---------------------------------------------------------|-------------------------------------------------------------|-----------------|
| A P R d ˆ d , m P R d                                   | Multivariate Ornstein-Uhlenbeck drift parameters            | Eq. (1)         |
| D ' 1 2 σσ J                                            | Multivariate Ornstein-Uhlenbeck diffusivity                 | Eq. (1)         |
| P                                                       | Schrödinger bridge path measure                             | Section 2.1     |
| Q                                                       | Reference process path measure                              | Section 2.1     |
| µ x 0 t , Σ t                                           | Unconditional mean &cov. of mvOU process started at x 0     | Section 3.1     |
| p t &#124;p x 0 , x T q                                 | Conditional density of mvOU bridge                          | Section 3.1     |
| c t &#124;p x 0 , x T q                                 | mvOU bridge control between p x 0 , x T q                   | Section 3.1     |
| µ t &#124;p x 0 , x T q , Σ t &#124;p x 0 , x T q ' Ω t | Conditional mean and covariance of mvOU bridge              | Section 3.1     |
| Ω st                                                    | Conditional covariance process of mvOU bridge               | Section 3.1     |
| u t &#124;p x 0 , x T q , s t &#124;p x 0 , x T q       | Conditional probability flow and score field of mvOU bridge | Section 3.1     |
| N p a , A q , N p b , B q                               | Initial and terminal mvOU-GSB marginals                     | Section 3.2     |
| N p a , A q , N p b , B q                               | Transformed mvOU-GSB marginals                              | Section 3.2     |
| C                                                       | Cross-covariance of entropic transport plan                 | Section 3.2     |
| A t , B t , c t                                         | Key quantities for the mvOU-GSB                             | Section 3.2     |
| ν t                                                     | mvOU-GSB mean process                                       | Section 3.2     |
| Ξ t , Ξ st                                              | mvOU-GSB variance and covariance process                    | Section 3.2     |
| S J t Ξ ´ 1 t                                           | SDE drift matrix of mvOU-GSB                                | Section 3.2.    |

Table 3: Glossary of some key notations and quantities used in the statements of our theoretical results.

## A.1 Some calculations on multivariate Ornstein-Uhlenbeck processes

For convenience, we first collect some results about multivariate Ornstein-Uhlenbeck processes of the form (1), a detailed discussion of mvOU processes can be found in e.g. [54]. For a time-invariant process with drift matrix A and diffusion σ , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

If X 0 ' N p µ 0 , Σ 0 q and writing X 0 t ' X 0 ' X t , some tedious but straightforward applications of conditional expectations and the tower property reveal that

<!-- formula-not-decoded -->

More generally, for a process d X t ' p AX t ` b q d t ` σ d B t , one has:

<!-- formula-not-decoded -->

The expressions for the covariance of the mvOU process can be obtained from the fact that the covariance evolves following a Lyapunov equation . For the case of constant coefficients, we state the following result.

Lyapunov equation solution It is easy to verify that 9 G t ' AG t ` G t A J ` Q t has solution

<!-- formula-not-decoded -->

## A.2 Bridges of multivariate Ornstein-Uhlenbeck processes

This problem has been studied by Chen et al. in [8], however the material therein is geared towards a control audience and considers a more general case where all coefficients are time-dependent. We will re-derive the results that we will need for processes of the form (1). Additionally, while in practice we typically use σ ' σ I , we state some results in the general setting of non-isotropic noise.

Derivation of the dynamical OU-bridge (Theorem 1) Consider a mvOU process

<!-- formula-not-decoded -->

Now form the controlled version pinned at p 0 , x 0 q and p T, x T q , where u t p X t q is an additional force arising from the conditioning of the process at an endpoint:

<!-- formula-not-decoded -->

The main result that we need is that u t p x q is itself a linear time-dependent field whose coefficients are independent of p x 0 , x T q . In the above we consider the simplest case of classical optimal control, where the control is directly added to the system drift and there is no unobserved system.

Consider the system started at p 0 , x 0 q and pinned to p T, x T q . For now, we make all statements for time-dependent coefficients and the diffusion is non-isotropic. We then form the Lagrangian:

<!-- formula-not-decoded -->

+

Fixing some λ ą 0 , the resulting problem is known as a linear-quadratic-Gaussian problem [29] in the control literature: #

<!-- formula-not-decoded -->

for which we can write a HJB equation:

<!-- formula-not-decoded -->

subject to the terminal boundary condition V T p x q ' λ 2 } x ´ x T } 2 , and the operators r A , A are defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

these are the generators of the controlled (24) and uncontrolled (23) SDEs respectively. Substituting all these in, we find that the HJBE is

<!-- formula-not-decoded -->

It follows from the first order condition on the 'inner' problem that u t ' ´ ∇ x V t , so

<!-- formula-not-decoded -->

Use an ansatz that the value function is quadratic:

<!-- formula-not-decoded -->

Then:

<!-- formula-not-decoded -->

Now note that the quadratic form only depends on the symmetric part of the matrix, i.e.

<!-- formula-not-decoded -->

The HJBE is therefore

<!-- formula-not-decoded -->

So

So

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with the boundary condition M T ' λ I and c t ' ´ λ x T .

Let Λ t ' M ´ 1 t , so that 9 Λ t ' ´ M ´ 1 t 9 M t M ´ 1 t ' ´ Λ t 9 M t Λ t . Substituting, we find that

<!-- formula-not-decoded -->

Actually we can rewrite the value function in a different way, which is more convenient for us:

<!-- formula-not-decoded -->

in which case c t ' ´ M t k t , and so we have k T ' x T . The corresponding ODE for k t is

<!-- formula-not-decoded -->

Let us now specialise to the case of time-invariant coefficients, which allows us to write down explicit expressions for the solutions:

<!-- formula-not-decoded -->

Rewriting the SDE drift to have the form Ax ` b ' A p x ´ m q we have that b ' ´ Am ùñ A ´ 1 b ' ´ m , when A is nonsingular. Then:

<!-- formula-not-decoded -->

Now we deal with the quadratic term Λ t . Let G τ ' Λ T ´ τ , so that B τ G τ ' ´ 9 Λ T ´ τ . Then G τ satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting back, we find that

<!-- formula-not-decoded -->

The bridge control is therefore and covariance

<!-- formula-not-decoded -->

For the general case of non-constant coefficients, we express them as solutions of ODEs with terminal boundary conditions.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In practice, both of these equations can be approximately solved by numerical integration in time. Finally we remark here that the diffusion component does not play a role in the control, which is classical.

'Static' Ornstein-Uhlenbeck bridge statistics (Theorem 2) As was studied by [8] for a more general scenario of time-varying processes, the mvOU process and its conditioned versions are Gaussian processes. Under these assumptions, p X s , X t q has joint mean

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Φ t is the covariance of the process started from a point mass at time t :

<!-- formula-not-decoded -->

By taking the Schur complement, we have that X s | X t is distributed with variance

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and mean

More generally, we can consider the three-point correlation p X r , X s , X t q . Then the mean is µ r ' µ s ' µ t and the covariance is

<!-- formula-not-decoded -->

Using the Schur complement again, we have that p X r , X s q| X t has covariance

<!-- formula-not-decoded -->

For the case of time-varying coefficients, we need to introduce the state transition matrix Ψ ts [29] which describes the deterministic aspects of evolution between s ă t under p A t q t . That is, for a dynamics 9 x t ' A t x t one has x t ' Ψ t,s x s and Ψ p t, s q ' Ψ p t, r q Ψ p r, s q for t ě r ě s . In this case there is not an explicit expression for Φ t . Instead, let p Φ t q t be the unconditional variance evolutions for such a process started at t ' 0 , obtained by solving

<!-- formula-not-decoded -->

Then the more general result from [8] for the covariance of p X r , X s , X r q is

<!-- formula-not-decoded -->

From this result, the same Schur complement computation yields expressions for the conditional covariances and mean:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Formal statement and proof of Lemma 1

Lemma (Formal statement of Lemma 1) . Let d X t ' AX t d t ` σ d B t be a linear SDE with generic drift A P R d ˆ d and diffusion σ P R d ˆ d . There exists an orthogonal matrix U P R d ˆ d such that Y t ' U J X t obeys a linear SDE with transformed drift matrix U J AU and a diagonal diffusion matrix Λ ' diag p λ q , where λ ' p λ 1 , . . . , λ d q ě 0 .

Proof. Let D ' 1 2 σσ J . This is positive semidefinite and therefore can its spectral decomposition can be written as D ' 1 2 U Λ 2 U J where U is orthogonal and Λ ' diag p λ q ľ 0 . Consider the SDEs

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let f p x q be a twice differentiable test function. SDE (a.) has generator p L f qp x q ' xB x f p x q , Ax y ` 1 2 tr p σ J B xx f p x q σ q . Computing the generator for SDE (b.), p ˜ L f qp x q ' xB x f p x q , Ax y ` 1 2 tr p U Λ U J B xx f p x q U Λ U J q . Note that tr p U Λ U J B xx f p x q U Λ U J q ' tr pB xx f p x q U Λ 2 U J q ' tr p σ J B xx f p x q σ q . Hence both SDEs share the same generator, and so are equal in law.

Since orthogonal transformations of Brownian increments are also Brownian increments, we have that U J d B t ' d ˜ B t . Substituting into (b.), we have d X t ' AX t d t ` U Λ d ˜ B t . Then,

<!-- formula-not-decoded -->

and

## A.3 Simulation-free Schrödinger bridges with linear reference dynamics

Having characterised the solution of the SBP for general reference processes and now that we have derived the score and flow for the Ornstein-Uhlenbeck bridge, the path is clear towards a simulationfree scheme for learning Schrödinger bridges where the reference dynamics are given by a linear SDE.

For solution of the static SBP problem (18) (equivalently, (SBP-static)), the transition kernel of the reference dynamics is given by

<!-- formula-not-decoded -->

where µ x 0 t denotes the mean at time t conditional on p 0 , x 0 q . Notably, the last two terms depend only on x 0 . It is a classical result (and very easy to show) that these kinds of terms do not affect the minimiser of (18) and so they are immaterial. The cost function to use is thus effectively

<!-- formula-not-decoded -->

Once a solution π of the static SBP is on hand, we want to utilise the stochastic regression objective on the conditional score and flow. Sampling p x 0 , x t q from π , recall that P p x 0 ,x 1 q ' Q p x 0 ,x 1 q so we want to sample from the Q -bridge using (11), (12):

<!-- formula-not-decoded -->

Equations (9) and (12) give us the score and flow at x t .

## A.4 Characterisation of the Q -GSB (Theorem 3)

We will construct the Q -GSB utilising its characterisation (2). Our approach is similar to that of [5] - we first obtain explicit formulae for the static Q -GSB (SBP-static) and then build towards the dynamical Q -GSB (SBP-dyn) by using the characterisation of Q -bridges.

Static Q -GSB The standard Gaussian EOT problem has a well known solution [32, 21, 5, 22]. Here we use the notations of [21] and define

<!-- formula-not-decoded -->

where σ ą 0 is the regularisation level and α ' N p a , A q and β ' N p b , B q . The solution to (61) in the Gaussian case is given by [21, Theorem 1]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We are, of course, interested in a slightly different problem, namely, for ρ 0 ' N p a , A q and ρ T ' N p b , B q , we seek to solve min π P Π p ρ 0 ,ρ T q KL p π | Q 0 ,T q . Expanding the definition of KL and

simplifying, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where by » we denote equality of the objective up to additive constants which do not affect the minimiser π . We note that the last line is exactly (61) with α, β ' ρ 0 , ρ T and σ ' 1 and a modified cost. Further, recognise d Q T | 0 p x T | x 0 q{ d x T as the Q -transition density, and Q T | 0 ' N p µ x 0 T , Σ T q . Thus,

<!-- formula-not-decoded -->

where we remind that µ x 0 t is the mean at time t conditional on starting at p 0 , x 0 q , and Σ t is the covariance at time t , started at a point mass, i.e. Σ 0 ' 0 .

In particular, we have Σ t ' Φ t , using the notation of Theorem 2. Recall that in the case of constant coefficients, we have the following explicit expressions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define the following changes of variable p x 0 , x T q ÞÑ p x 0 , x T q , injective whenever Σ T is positive definite:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let ρ 0 ' p T 0 q # ρ 0 and ρ T ' p T T q # ρ T , and π ' p T 0 ˆ T T q # π and note that the relative entropy is invariant to this change of coordinates. So the problem (69) is equivalent to

<!-- formula-not-decoded -->

This is exactly OT b σ 2 ' 1 p ρ 0 , ρ T q as per (61) with ρ 0 ' N p a , A q and ρ T ' N p b , B q . The transformed means and covariances are

<!-- formula-not-decoded -->

We remark that for time-dependent coefficients our approach still applies, however computation of the transformation coefficients will be in terms of the general state transition matrix instead of matrix exponentials. In what follows, we focus on results for constant coefficients for simplicity and their practical relevance.

The solution to (72) is therefore

<!-- formula-not-decoded -->

where C is the cross-covariance the transport plan, given by (64).

Dynamic Q -GSB : marginals and covariance Let X t |p x 0 , x T q denote the Q -bridge pinned at p x 0 , x T q . Then by Theorem 2,

<!-- formula-not-decoded -->

Invoking the inverse mappings T ´ 1 0 , T ´ 1 T and substituting the expression for µ t |p x 0 , x T q from 2, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Γ t ' Φ t e p T ´ t q A J Φ ´ 1 T . For clarity, we state first a general formula:

ˆ'

N

Y

X

µ

0

µ

1

0

,

X

ȷ

1

'

,

'

Σ

00

Σ

10

N

p

Σ

Σ

AX

01

11

0

`

ȷ˙

. Let

BX

1

`

c

,

Ω

q

.

Then, E r Y s ' Aµ 0 ` Bµ 1 ` c and V r Y s ' Ω ` A Σ 00 A J ` B Σ 11 B J ` A Σ 01 B J ` B Σ J 01 A J .

Application to (74) gives us the following formula for the variance of the bridge at time t :

<!-- formula-not-decoded -->

Substituting and simplifying, we get the following expressions for each of the terms:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, the mean can be computed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To calculate the covariance of the SB process, we use again the disintegration property (2). We have from (52) for the Q -bridges and 0 ă s ă t ă T :

<!-- formula-not-decoded -->

Now, and

<!-- formula-not-decoded -->

in which the first (variance) term doesn't actually depend on p X 0 , X T q . So,

<!-- formula-not-decoded -->

In fact let's switch to the 'mapped' coordinates X 0 , X T for the endpoints. We abuse notation and omit the inverse map in what follows. Then, from previously,

<!-- formula-not-decoded -->

Expanding, collecting and cancelling terms, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Putting everything together, we get

<!-- formula-not-decoded -->

Lemma 2.

Let

p

X

0

,

X

1

q '

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Dynamic Q -GSB : SDE representation We proceed via the generator route also used by [5]. Expanding the covariance (91), we find that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have set the key quantity

<!-- formula-not-decoded -->

Now, p X t ` h , X t q are jointly Gaussian with mean and covariance

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let x ÞÑ u p x q be a twice differentiable test function. Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Subtracting u p x q and taking the limit h Ñ 0 , it is clear from the definition of the generator that the SDE drift is

<!-- formula-not-decoded -->

This takes the same form as the equation found in [5], however our formulae for S t allow us to apply it to any linear reference SDE, not necessarily ones with scalar drift. Additionally, we empirically verify that for asymmetric A the matrix S t is asymmetric. This contrasts with the symmetric nature of the drift for the gradient-type setting.

Now we want to work out what each of these terms are in practice. Effectively we need to compute 9 A t , 9 B t and pB t 1 Ω t,t 1 qp t q :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

So

Remark on the case of time-varying coefficients In this case we can still assume without loss of generality that we still work on the time interval r 0 , T s , by shifting the time coordinate if necessary. Let Ψ p t, s q be the state transition matrix associated with A t and Φ t be the solution at time t to (53). Then we still have Σ T ' Φ T for the transition kernel covariance in the cost, i.e. the covariance started from a point mass. For the mean of the reference process started from x 0 , we have the generalised formula

<!-- formula-not-decoded -->

and we remark that its inverse µ ´ 1 t always exists since Ψ is never singular:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall that the general expression for the Q -bridge conditioned on p 0 , x 0 q , p T, x T q is

<!-- formula-not-decoded -->

Letting x 0 ' µ ´ 1 T p Σ 1 { 2 T x 0 q and x T ' Σ 1 { 2 T x T and substituting the expression for µ ´ 1 t we have

<!-- formula-not-decoded -->

Set Γ t ' Φ t Ψ p T, t q J Φ ´ 1 T , then

Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Case: Brownian motion Let's check this against the results of Bunne and Hsieh [5], who consider a class of reference processes

<!-- formula-not-decoded -->

This corresponds to a special case of ours, where the drift is scalar-valued. Setting g t ' ω, α t ' c t ' 0 , their results give us the marginal parameters of the GSB connecting N p µ 0 , Σ 0 q and N p µ 1 , Σ 1 q . In what follows, using their results and notations of [5, Table 1], we have that r t ' t, r t ' 1 ´ t, ζ ' 0 , κ p t, t 1 q ' ω 2 t, ρ t ' t . Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us compute the same quantities using our formulae. We set A ' 0 , m ' 0 and T ' 1 . Then, Φ t ' tω 2 I , Γ t ' t I , Σ 1 { 2 T ' ω I . Then,

<!-- formula-not-decoded -->

Also from the underbraced expressions, we have

<!-- formula-not-decoded -->

Assembling our results, we get

<!-- formula-not-decoded -->

Now C is the transport plan obtained from the scaled input Gaussians, of variance A ' A { ω 2 , B ' B { ω 2 (in the case of Brownian reference process, there is no linear mapping via the flow map). For the scaled measures, EOT is calculated with ε ' σ 2 ' 1 . It stands to reason that once we scale everything back, we should get C ' C { ω 2 . To verify this, recall that for the untransformed problem

<!-- formula-not-decoded -->

Then for the RHS problem, we have that

<!-- formula-not-decoded -->

Also it is easy to verify that Ω t ' Σ t |p x 0 ,x 1 q ' ω 2 t p 1 ´ t q I . So,

<!-- formula-not-decoded -->

where C is the EOT plan between the unscaled measures N p a , A q , N p b , B q with ε ' ω 2 . This is exactly the same result as the one derived in [5].

Case: Centered scalar OU process A ' ´ λ I , m ' 0 From [5, Table 1], we have that r t ' sinh p λt q{ sinh p λ q , r t ' sinh p λt q coth p λt q ´ sinh p λt q coth p λ q , ζ ' 0 . The mean is then

<!-- formula-not-decoded -->

ρ

With

κ

p

t, t

1

q '

ω

2

e

´

λt sinh

p

λt q{

λ

and

t

'

e

´

λ

p

1

´

t

q

sinh

p

λt q{

sinh

p

λ

q

, the variance is

<!-- formula-not-decoded -->

where σ ‹ 2 ' ω 2 sinh p λ q{ λ . For convenience, we will check things term by term and using Mathematica.

Check first the mean. We have:

<!-- formula-not-decoded -->

Then, plugging into our formula,

<!-- formula-not-decoded -->

Now the variance. We have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From here it is straightforward to verify that

<!-- formula-not-decoded -->

Now note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C 1 { ? αβ denotes the EOT covariance for entropic regularisation level 1 { ? αβ . Then:

<!-- formula-not-decoded -->

It is also easy to check that

Finally,

<!-- formula-not-decoded -->

We have verified that all the terms in the expression for the variance agree.

## A.5 Proof of Theorem 4

Proof. We follow the arguments of [52] with application to the generalised Schrödinger bridge - not much changes for Q as the reference and we reproduce in detail the arguments as follows. For the equality of gradients, it suffices to show this for the flow matching component of the loss, since the score matching component can be handled using the exact same arguments [52]. Let u t |p x 0 , x T q be the conditional flow between p x 0 , x T q and u t be the SB flow defined by

<!-- formula-not-decoded -->

Let u θ t be the neural flow approximation with parameter θ . Assuming that p t p x q ą 0 for all p t, x q we then have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the last line terms not depending on θ are zero and we use the fact that E p x 0 , x T q E x |p x 0 , x T q f p x q ' E x f p x q . Now, using the relation between u t and u t |p x 0 , x T q ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

So we conclude that the two gradients are equal. Clearly, the unconditional loss can be rewritten over candidate flow and score fields p ˆ u , ˆ s q as

<!-- formula-not-decoded -->

where we denote by } h t p x q} 2 L 2 p d p t p x q d t q ' ş 1 0 d t ş d p t p x q} h t p x q} 2 L 2 p R d q for a test function h t : r 0 , 1 s ˆ R d Ñ R d . This loss is zero iff ˆ u ' u and ˆ s ' s p d p t ˆ d t q -almost everywhere.

Let P denote the law of the Schrödinger bridge as per (SBP-dyn) and write p t p x q ' P t to mean its marginal at time t . Then in (2) and for Q a mvOU process (1), identifying:

- d P ‹ 0 T ' π where π is prescribed in Proposition 1, and
- Q x 0 ,x T t ' p t |p x 0 , x T q p x q where p t |p x 0 , x T q is defined as in Theorem 2,

substituting all these into (2) one has

<!-- formula-not-decoded -->

Consequently, there exists u t p x q such that B t p t p x q ' ´ ∇ ¨ p p t p x q u t p x qq . Writing s t p x q ' ∇ x log p t p x q to be the score, recognising terms from the probability flow ODE, it follows that that the SDE

<!-- formula-not-decoded -->

generates the marginals p t p x q of the Schrödinger bridge. Since P is characterised as a mixture of Q -bridges, it follows that X t defined by the SDE (150) generates the Markovianisation of P (see Appendix B of [52]). Moreover, P is the unique process that is both Markov and a mixture of Q -bridges [28, 45].

## A.6 Additional background

Hamilton-Jacobi-Bellman equation Consider the deterministic control problem on r 0 , T s :

<!-- formula-not-decoded -->

subject to a dynamics 9 x t ' F p x t , u t q . V t p x q is the value function for the problem starting at p t, x q up to the final time T . The function x T ÞÑ D p x T q specifies a cost on the final value. The corresponding Hamilton-Jacobi-Bellman (HJB) equation [23, Section 3.11] is

<!-- formula-not-decoded -->

subject to the final boundary condition V T p x q ' D p x q . A heuristic derivation is as follows. Considering a time interval p t, t ` δt q and a path x t , it's clear that

<!-- formula-not-decoded -->

Using Taylor expansion and the constraint 9 x t ' F p x t , u t q we have to leading order that

<!-- formula-not-decoded -->

Substituting back and also approximating the integral, we find that

<!-- formula-not-decoded -->

Cancelling terms, rearranging and taking a limit δt Ó 0 , we get the desired result.

Stochastic Hamilton-Jacobi-Bellman Now we consider the stochastic variant of the HJB, in which case X t is driven by an SDE of the form

<!-- formula-not-decoded -->

The value function is thus in expectation:

<!-- formula-not-decoded -->

Carrying out the same expansion as before, we have

<!-- formula-not-decoded -->

Note that in the above, p X t ` δt | X t ' x t q is a random variable and hence so is V t ` δt p X t ` δt q which is why it appears in the expectation. Using Itô's formula to expand this, we get

<!-- formula-not-decoded -->

`x

∇

x

V

t

p

X

t

q

, σ

t

d

B

t

y

.

Plugging this in, cancelling terms, and noting that the final term has zero expectation, we find that

<!-- formula-not-decoded -->

where A is the generator of the SDE governing X t , i.e.

<!-- formula-not-decoded -->

The HJB equation for stochastic control problem is therefore

<!-- formula-not-decoded -->

i.e. this is the same as the HJB for the deterministic case, except with a diffusive term arising from the stochasticity.

## B Experiment details

For (mvOU, BM)-OTFM and IPFP, all computations were carried out by CPU (8x Intel Xeon Gold 6254). NLSB and SBIRR computations were accelerated using a single NVIDIA L40S GPU. Code is available at https://github.com/zsteve/mvOU\_SBP .

## B.1 Gaussian benchmarking

We consider settings of dimension d P t 2 , 5 , 10 , 25 , 50 u with data consisting of an initial and terminal Gaussian distribution and a mvOU reference process constructed as follows. We sample a d ˆ 2 submatrix U from a random d ˆ d orthogonal matrix. Then for the reference process, A is constructed as the d ˆ d matrix

<!-- formula-not-decoded -->

Similarly, we let m ' U r 1 ´ 1 s J . For the marginals, we take N p µ 0 , Σ 0 q and N p µ 1 , Σ 1 q where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For each d , the initial and final marginals are approximated by N ' 128 samples t x i 0 u N i ' 1 ' N p µ 0 , Σ 0 q , t x i 1 u N i ' 1 ' N p µ 1 , Σ 1 q . We fix σ ' I and compute the exact marginal parameters p µ t , Σ t q t Pr 0 , 1 s using the formulas of Theorem 3 and reference parameters p A , m q . Additionally, we compute the SDE drift v SB p t, x q of the mvOU-GSB using (17).

(mvOU, BM)-OTFM We apply mvOU-OTFM as per Algorithm 1 to learn a conditional flow matching approximation to the Schrödinger bridge. We choose to parameterise the probability flow and score fields using two feed-forward neural networks u θ p t, x q ' NN θ p d ` 1 , d qp t, x q and s φ p t, x q ' NN φ p d ` 1 , d qp t, x q , each with r 64 , 64 , 64 s hidden dimensions and ReLU activations. We use a batch size of 64 and learning rate 10 ´ 2 for 2,500 iterations using the AdamW optimiser. For BM-OTFM, the same training procedure was used except the reference process was taken to be Brownian motion with unit diffusivity.

IPFP Weuse the IPFP implementation provided by the authors of [53], specifically using the variant of their algorithm based on Gaussian process approximations to the Schrödinger bridge drift [53, Algorithm 2]. We provide the mvOU reference process as the prior drift function, i.e. x ÞÑ A p x ´ m q and employ an exponential kernel for the Gaussian process vector field approximation. We run the IPFP algorithm for 10 iterations, which is double that used in the original publication. Since IPFP explicitly constructs forward and reverse processes, at each time 0 ď t ď 1 , IPFP outputs two estimates of the Schrödinger bridge marginal, one from each process. We consider both outputs and distinguish between them using the pÑ , Ðq symbols in Table 1. For the vector field estimate we employ only the forward drift.

NLSB We use the NLSB implementation provided by the authors of [25]. Following [12, Section 4.6], for a mvOU reference process d X t ' f t d t ` σ d B t , the generalised SB problem (SBP-dyn) can be rewritten as a stochastic control problem

<!-- formula-not-decoded -->

On the other hand, the Lagrangian SB problem [25, Definition 3.1] is

<!-- formula-not-decoded -->

This shows that the appropriate Lagrangian to use is L p t, x , v t p x qq ' 1 2 } v t p x q ´ f t p x q} 2 2 . We apply this in the mvOU setting by taking f t p x q ' A p x ´ m q . The NLSB approach backpropagate through solution of the SDE to directly learn a neural approximation of the SB drift v t . We train NLSB the same hyperparameters as used in the original publication, using the Adam optimiser with learning rate 10 ´ 3 for a total of 2,500 epochs.

Metrics We measure both the approximation error of Gaussian Schrödinger bridge marginals as well as the error in vector field estimation. For the marginal reconstruction, for each method and for each 0 ď t ď 1 we sample points integrated forward in time and compute the mean ˆ µ t and variance ˆ Σ t from samples. We then compare to the ground truth marginals p µ t , Σ t q computed from exact formulas using the Bures-Wasserstein metric (62). For the vector field, for each method and for each time 0 ď t ď 1 we sample M ' 1024 points from the ground truth SB marginal N p µ t , Σ t q and empirically estimate } ˆ v SB ´ v SB } N p µ t , Σ t q from samples. All experiments were repeated over 5 independent runs and summary statistics for metrics are shown in Table 1.

## B.2 Gaussian mixture example

We consider initial and terminal marginals sampled from Gaussian mixtures. At time t ' 0 , we sample from

<!-- formula-not-decoded -->

in a 1:1 ratio, and at time t ' 1 we sample in the same fashion from

<!-- formula-not-decoded -->

Here, the matrix U and reference process parameters p A , m q are the same as used for the previous Gaussian example for d ' 10 . All other training details are the same as for the Gaussian example.

## B.3 Repressilator example

Based on the system studied in [44], we simulate stochastic trajectories from the system

<!-- formula-not-decoded -->

with initial condition N pr 1 , 1 , 2 s , 0 . 01 I q and simulated for the time interval t P r 0 , 10 s using the Euler-Maruyama discretisation. Snapshots were sampled at T ' 10 time points evenly spaced on r 0 , 10 s , each comprising of 100 samples.

We employ Algorithm 2 to this data, running for 5 iterations starting from an initial Brownian reference process A ' 0 , m ' 0 . For the inner loop running mvOU-OTFM (Algorithm 1), we parameterise the probability flow and score as in the Gaussian example, with feed-forward networks of hidden dimension r 64 , 64 , 64 s and ReLU activations. We run mvOU-OTFM for 1,000 iterations using the AdamW optimiser, a batch size of 64 and a learning rate of 10 ´ 2 . At each step of the outer loop, we employ ridge regression to fit the updated mvOU reference parameters p A , m q . We do this using the standard RidgeCV method implemented in the scikit-learn package, which automatically selects the regularisation parameter.

We also carry out hold-one-out runs where for 2 ď i ď 9 (i.e. all time-points except for the very first and last), Algorithm 2 is applied to T ´ 1 snapshots with the snapshot at t i held out. Once the mvOU reference parameters are learned, forward integration of the learned mvOU-SB is used to predict the marginal at t i . We report the reconstruction error in terms of the earth-mover distance (EMD) and energy distance [41]. Table 2 shows results averaged over held-out timepoints, and full results (split by timepoint) are shown in Table 5.

Since the system marginals are unimodal we reason that they can be reasonably well approximated by Gaussians. For each 2 ď i ď 9 we fit multivariate Gaussians to the snapshots at t i ´ 1 , t i , t i ` 1 and use the results of Theorem 3 with the fitted mvOU reference output by Algorithm 2 to solve the mvOU-GSB between p t i ´ 1 , p t i ` 1 . This is illustrated in Figure 4 for i ' 4 , and we show the full results for all timepoints in Figure 6 in comparison with the standard Brownian GSB.

SBIRR We apply SBIRR [44] using the implementation provided with the original publication, which notably includes an improved implementation of IPFP [53] that utilises GPU acceleration. We provide the reference vector field x ÞÑ A p x ´ m q and seek to learn a reference process in one of two families: (1) mvOU processes, i.e. we consider the family of reference drifts ˆ A p x ´ ˆ m q where ˆ A , ˆ m are to be fit, or (2) general drifts, i.e. we parameterise the drift using a feed-forward neural network with hidden dimensions r 64 , 64 , 64 s . For each choice of reference family, we run Algorithm 1 of [44] for 5 outer iterations and 10 inner IPFP iterations, as was also done in the original paper.

## B.4 Cell cycle scRNA-seq

The metabolic labelled cell cycle dataset of [4] is obtained and preprocessed following the tutorial available with the Dynamo [40] package. This gives a dataset of N ' 2 , 793 cells, embedded in 30 PCA dimensions. In addition to transcriptional state t x i u N i ' 1 , Dynamo uses metabolic labelling data to predict the transcriptional velocity t ˆ v i u N i ' 1 for each cell.

To fit the reference process parameters p A , m q , we use again ridge regression via the RidgeCV method in scikit-learn . We train mvOU-OTFM using Algorithm 1 with σ ' 0 . 3 , parameterising the probability flow and score as previously using feed-forward networks of hidden dimensions r 64 , 64 , 64 s and train with a batch size of 64, learning rate of 10 ´ 2 for a total of 1,000 iterations.

Figure 6: Repressilator mvOU-GSB interpolation. Using the learned mvOU reference process, we interpolate between p i ´ 1 (blue) and p i ` 1 (green). Middle timepoint p i is shown in red.

<!-- image -->

Figure 7: Cell cycle mvOU-GSB interpolation. Using the learned mvOU reference process and scale factor γ ' 50 , we interpolate between the first snapshot p 1 (blue) and last snapshot p T (green). All computations are done in d ' 30 and shown in leading 2 PCs.

<!-- image -->

Table 4: Single cell interpolation results for d ' 50 , 100 .

|         |      |   Scale factor γ |   Scale factor γ |   Scale factor γ |   Scale factor γ |   Scale factor γ |   Scale factor γ |   Scale factor γ |   Scale factor γ |   Scale factor γ |   Scale factor γ |   Scale factor γ |
|---------|------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|
|         | t    |             0    |            10    |            20    |            30    |            40    |            50    |            60    |            70    |            80    |            90    |           100    |
| d ' 50  | 0.25 |            40.18 |            30.8  |            22.55 |            16.34 |            12.77 |            11.07 |            10.26 |            10.11 |            10.69 |            11.93 |            13.65 |
| d ' 50  | 0.5  |            28.59 |            18.47 |            11.53 |             9.62 |            12.36 |            15.32 |            15    |            12.52 |            10.64 |            11.02 |            13.96 |
| d ' 50  | 0.75 |            15.31 |            10.31 |             7.47 |             7.06 |             8.18 |             8.9  |             8.5  |             7.9  |             8.09 |             9.42 |            11.92 |
|         | 0.25 |            44.49 |            36.82 |            29.97 |            24.19 |            19.99 |            17.48 |            16.13 |            15.29 |            14.78 |            14.7  |            15.18 |
| d ' 100 | 0.5  |            32.48 |            23.75 |            17.12 |            13.55 |            13.79 |            16.84 |            19.78 |            20.21 |            18.25 |            15.7  |            14.22 |
| d ' 100 | 0.75 |            19.33 |            14.95 |            12    |            10.74 |            11.19 |            12.68 |            13.98 |            14.36 |            14.03 |            13.69 |            13.82 |

Table 5: Full results for repressilator example

<!-- image -->

| Algorithm 2 Iterated reference fitting with mvOU-OTFM                                                                                                                         | Algorithm 2 Iterated reference fitting with mvOU-OTFM                                                                                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Input: Samples t x t j i u N j i ' 1 from multiple snapshots at times t t j u T j ' 1 , initial mvOU reference parameters p A , m q , diffusivity D ' 1 2 σσ J                | Input: Samples t x t j i u N j i ' 1 from multiple snapshots at times t t j u T j ' 1 , initial mvOU reference parameters p A , m q , diffusivity D ' 1 2 σσ J                |
| Initialise: Probability flow field u θ t p x q , score field s φ t p x q . Define: FitReference p X , V q : ' argmin A , m } V ´ A p X ´ m q} 2 2 ` λ } A } 2 F ` γ } m } 2 N | Initialise: Probability flow field u θ t p x q , score field s φ t p x q . Define: FitReference p X , V q : ' argmin A , m } V ´ A p X ´ m q} 2 2 ` λ } A } 2 F ` γ } m } 2 N |
| p u θ t , s θ t qÐ fitOTFM p A , m , D q p ˆ ρ 1 , . . . , ˆ ρ T q v t j i Ðp u θ t j ` Ds φ t j qp x t j i q , 1 ď i ď N j , 1 ď t , tp x t j q                              | Fit flow and score with reference parameters Get SDE                                                                                                                          |
| j ď T                                                                                                                                                                         | drift                                                                                                                                                                         |
| A , m Ð FitReference ptp v j i q N i i ' 1 u T j ' 1 i N i i ' 1 u T j ' end while                                                                                            | Update reference parameters                                                                                                                                                   |

|              |    | Leave-one-out   | Leave-one-out   | Leave-one-out   | Leave-one-out   | Leave-one-out   | Leave-one-out   | Leave-one-out   |
|--------------|----|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Error metric | t  | Iterate 0       | Iterate 1       | Iterate 2       | Iterate 3       | Iterate 4       | SBIRR (mvOU)    | SBIRR (MLP)     |
|              | 1  | 3.59 ˘ 0.17     | 3.14 ˘ 0.12     | 2.28 ˘ 0.15     | 2.08 ˘ 0.11     | 2.02 ˘ 0.11     | 2.24 ˘ 0.28     | 2.71 ˘ 0.41     |
|              | 2  | 5.20 ˘ 0.47     | 2.59 ˘ 0.29     | 1.62 ˘ 0.28     | 1.27 ˘ 0.25     | 1.13 ˘ 0.16     | 3.13 ˘ 0.62     | 2.29 ˘ 0.92     |
|              | 3  | 3.23 ˘ 0.24     | 1.42 ˘ 0.18     | 1.10 ˘ 0.14     | 0.86 ˘ 0.08     | 0.83 ˘ 0.10     | 2.67 ˘ 0.85     | 1.39 ˘ 0.55     |
| EMD          | 4  | 1.48 ˘ 0.20     | 0.52 ˘ 0.05     | 0.47 ˘ 0.05     | 0.47 ˘ 0.06     | 0.48 ˘ 0.06     | 1.38 ˘ 0.46     | 0.94 ˘ 0.28     |
|              | 5  | 2.50 ˘ 0.40     | 1.43 ˘ 0.65     | 1.12 ˘ 0.32     | 1.29 ˘ 0.11     | 1.21 ˘ 0.13     | 1.63 ˘ 0.17     | 1.40 ˘ 0.71     |
|              | 6  | 6.18 ˘ 0.41     | 3.42 ˘ 0.50     | 2.18 ˘ 0.28     | 1.91 ˘ 0.38     | 1.75 ˘ 0.17     | 2.40 ˘ 0.32     | 1.96 ˘ 1.41     |
|              | 7  | 2.56 ˘ 0.25     | 3.09 ˘ 1.53     | 2.13 ˘ 0.46     | 2.23 ˘ 0.55     | 1.93 ˘ 0.50     | 1.45 ˘ 0.20     | 0.51 ˘ 0.13     |
|              | 8  | 2.29 ˘ 0.26     | 2.12 ˘ 0.09     | 1.82 ˘ 0.33     | 1.81 ˘ 0.31     | 1.81 ˘ 0.16     | 1.93 ˘ 0.61     | 2.14 ˘ 0.40     |
|              | 1  | 3.00 ˘ 0.09     | 2.75 ˘ 0.07     | 2.24 ˘ 0.10     | 2.10 ˘ 0.07     | 2.05 ˘ 0.08     | 2.20 ˘ 0.18     | 2.56 ˘ 0.26     |
|              | 2  | 3.53 ˘ 0.21     | 2.27 ˘ 0.19     | 1.66 ˘ 0.20     | 1.38 ˘ 0.21     | 1.26 ˘ 0.13     | 2.61 ˘ 0.33     | 2.05 ˘ 0.61     |
|              | 3  | 2.29 ˘ 0.17     | 1.31 ˘ 0.16     | 1.00 ˘ 0.10     | 0.80 ˘ 0.06     | 0.78 ˘ 0.07     | 2.10 ˘ 0.52     | 1.25 ˘ 0.43     |
|              | 4  | 0.93 ˘ 0.14     | 0.25 ˘ 0.06     | 0.17 ˘ 0.04     | 0.15 ˘ 0.02     | 0.14 ˘ 0.01     | 0.99 ˘ 0.38     | 0.84 ˘ 0.25     |
| Energy       | 5  | 1.10 ˘ 0.18     | 0.53 ˘ 0.27     | 0.45 ˘ 0.13     | 0.55 ˘ 0.04     | 0.51 ˘ 0.07     | 0.64 ˘ 0.10     | 0.66 ˘ 0.37     |
|              | 6  | 2.49 ˘ 0.12     | 1.22 ˘ 0.10     | 1.13 ˘ 0.15     | 1.01 ˘ 0.22     | 0.91 ˘ 0.11     | 1.43 ˘ 0.17     | 0.90 ˘ 0.64     |
|              | 7  | 0.97 ˘ 0.13     | 1.39 ˘ 0.65     | 1.02 ˘ 0.21     | 1.05 ˘ 0.25     | 0.91 ˘ 0.25     | 0.65 ˘ 0.09     | 0.17 ˘ 0.08     |
|              | 8  | 0.54 ˘ 0.11     | 0.57 ˘ 0.03     | 0.56 ˘ 0.11     | 0.55 ˘ 0.10     | 0.55 ˘ 0.04     | 0.48 ˘ 0.08     | 0.36 ˘ 0.06     |

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The claims made in the abstract and introduction accurately reflect this paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes] .

Justification: We discuss the limitations of our work, specifically on the linearity of the reference process and the computational cost of our algorithm.

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

Justification: We provide all assumptions and complete proofs for all theoretical results stated.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulae, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: In our paper we provide all information needed to reproduce our main experimental results.

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

Justification: We provide data and code with instructions to reproduce our experimental results in our supplemental material.

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

Justification: These details are all available in our supplemental material.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We provide error bars for the main experimental results of our paper.

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

Justification: These details are all available in our supplemental material.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: All research in this paper conforms with the NeurIPS code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is foundational and we do not foresee any direct path to any societal implications arising from our work.

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

Justification: Our paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All assets used in our research are appropriately credited.

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

Justification: We provide the full code and data in our supplementary materials.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Not applicable.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Not applicable.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: No LLM usage towards this paper.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.