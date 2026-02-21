## Riemannian Proximal Sampler for High-accuracy Sampling on Manifolds

## Yunrui Guan

Department of Computational Applied Mathematics and Operations Research

Rice University Houston, TX 77005

yg83@rice.edu

## Krishnakumar Balasubramanian

Department of Statistics University of California, Davis Davis, CA 95616

kbala@ucdavis.edu

## Shiqian Ma

Department of Computational Applied Mathematics and Operations Research Rice University

Houston, TX 77005

sqma@rice.edu

## Abstract

We introduce the Riemannian Proximal Sampler , a method for sampling from densities defined on Riemannian manifolds. The performance of this sampler critically depends on two key oracles: the Manifold Brownian Increments (MBI) oracle and the Riemannian Heat-kernel (RHK) oracle. We establish high-accuracy sampling guarantees for the Riemannian Proximal Sampler, showing that generating samples with ε -accuracy requires O (log(1 /ε )) iterations in Kullback-Leibler divergence assuming access to exact oracles and O (log 2 (1 /ε )) iterations in the total variation metric assuming access to sufficiently accurate inexact oracles. Furthermore, we present practical implementations of these oracles by leveraging heat-kernel truncation and Varadhan's asymptotics. In the latter case, we interpret the Riemannian Proximal Sampler as a discretization of the entropy-regularized Riemannian Proximal Point Method on the associated Wasserstein space. We provide preliminary numerical results that illustrate the effectiveness of the proposed methodology.

## 1 Introduction

Westudy the problem of sampling from a density π X ∝ e -f on a Riemannian manifold ( M,g ) , where g is the metric and π X is defined with respect to the volume measure dV g . The normalization constant ∫ M e -f dV g is unknown. Riemannian sampling arises in Bayesian inference (e.g., hierarchical models and Bayesian deep learning) [Girolami and Calderhead, 2011, Byrne and Girolami, 2013, Patterson and Teh, 2013, Liu and Zhu, 2018, Arnaudon et al., 2019, Liu et al., 2016, Piggott and Solo, 2016, Muniz et al., 2022, Lie et al., 2023], statistical physics (e.g., constrained molecular dynamics) [Leimkuhler and Matthews, 2016], and manifold optimization problems such as eigenvalue computation, low-rank approximation, and diffusion models [Goyal and Shetty, 2019, Li and Erdogdu, 2023, Yu et al., 2023, Bonet et al., 2023, De Bortoli et al., 2022, Huang et al., 2022].

Table 1: Comparison of iteration complexity with Li and Erdogdu [2023] and Cheng et al. [2022]. Here, ε is the target accuracy, K 2 is the gradient Lipschitz constant, and d is the manifold dimension; for a product of n spheres of dimension m , d = mn . All works assume a uniform lower bound on Ricci curvature and include the hypersphere as a common example. Cheng et al. [2022] does not provide explicit dependence on d or other problem parameters.

| Assumptions                                                                | Source                | Complexity                            | Metric   |
|----------------------------------------------------------------------------|-----------------------|---------------------------------------|----------|
| LSI, M = S m ×S m × ... ×S m f is K 2 -smooth                              | Li and Erdogdu [2023] | O ( dK 2 2 α 2 ε log H πX ( ρ 0 ) ε ) | KL       |
| Distant-dissipativity, f is K 2 -smooth, M has bounded sectional curvature | Cheng et al. [2022]   | O ( 1 ε 2 Poly ( d,K 2 ) )            | W 1      |
| LSI, M = S d , f is L 1 -Lipschitz                                         | Corollary 10          | ˜ O ( L 2 1 d α log 2 1 ε )           | TV       |

√

On a Riemannian manifold, Langevin dynamics takes the form dX t = -grad f ( X t ) dt + 2 dB t , where grad is the Riemannian gradient and B t is Brownian motion on the manifold. This formulation extends Euclidean Langevin dynamics by incorporating geometric information through the Riemannian metric, but discretizing manifold Brownian motion is generally intractable. Li and Erdogdu [2023] considered product manifolds S m ×··· × S m , showing convergence for a scheme that discretizes only the drift while assuming exact Brownian motion-feasible on spheres. Gatmiry and Vempala [2022] extended this to general Hessian manifolds, however, requiring exact Brownian motion. Both assume a log-Sobolev inequality and achieve poly (1 /ε ) iteration complexity in KL divergence, but the need for exact Brownian motion in Gatmiry and Vempala [2022] limits practical applicability.

Cheng et al. [2022] analyzed a practical discretization of Riemannian Langevin dynamics, where both drift and noise are discretized. They established ˜ O (1 /ε 2 ) iteration complexity in the 1-Wasserstein distance under general assumptions, and in the 2-Wasserstein distance under a stronger, log-concavitylike condition. A key challenge is proving Wasserstein contractivity without convexity (e.g., on compact manifolds), addressed via a second-order expansion of the Jacobi equation [Cheng et al., 2022, Lemma 29]. Kong and Tao [2024] proposed a Lie-group MCMC sampler for densities on Lie groups, achieving polynomial ˜ O ( poly (1 /ε )) complexity in 2-Wasserstein distance.

In comparison to the above works for Riemannian sampling, for the Euclidean case, high-accuracy algorithms, i.e., algorithms with iteration complexity of ˜ O ( polylog (1 /ε )) are available under various assumptions (that are essentially based on (strong) log-concavity or isoperimetry); see for example Lee et al. [2021], Chen et al. [2022], Fan et al. [2023], He et al. [2024] for such results for the Euclidean proximal sampler and Dwivedi et al. [2019], Chen et al. [2020], Chewi et al. [2021], Lee et al. [2020], Wu et al. [2022], Chen and Gatmiry [2023], Andrieu et al. [2024], Altschuler and Chewi [2024] for various Metropolized algorithms including Metropolis Random Walk (MRW), Metropolis Adjusted Langevin Algorithm (MALA) and Metropolis Hamiltonian Monte Carlo (MHMC).

High-accuracy samplers for constrained Euclidean sampling-i.e., from densities supported on convex sets K ⊆ R d -have been developed using Hit-and-Run and Ball Walk under various conditions [Lovász, 1999, Kannan et al., 2006, 1997]; see Kook and Zhang [2025, Section 1.3] for a survey. Kook et al. [2022] introduced Constrained Riemannian HMC (CRHMC) with an Implicit Midpoint integrator and proved high-accuracy guarantees. Noble et al. [2023] proposed Barrier HMC (BHMC) and its discretizations with asymptotic guarantees. Kook et al. [2024] developed the "In-and-Out" algorithm for uniform sampling on convex bodies. Kook and Vempala [2024] achieved state-of-the-art accuracy for log-concave sampling via a proximal method. Srinivasan et al. [2024a,b] showed that Metropolized Mirror and preconditioned Langevin samplers also achieve high-accuracy under suitable assumptions.

Given the above, the following natural question arises:

Can one develop high-accuracy algorithms for sampling on Riemannian manifolds?

To the best of our knowledge, no prior work exists on providing an affirmative answer to this question. In this work, we develop the Riemannian Proximal Sampler which generalizes the Euclidean Proximal Sampler from Lee et al. [2021]. In contrast to the Euclidean case, the algorithm is based on the availability of two oracles: the Manifold Brownian Increment (MBI) oracle and the Riemannian Heat Kernel (RHK) oracle. We show in Theorem 6 and Theorem 8 that the algorithm achieves high-accuracy guarantees under functional inequality assumptions when exact oracles are available, and under Assumption 1 when inexact oracles are available, respectively. We further develop practical implementations of the aforementioned oracles that satisfy the conditions in Assumption 1 (Section 5), and that are connected to entropy-regularized proximal point method on Wasserstein spaces (Appendix A). A comparison with the existing results for the case of sampling on spheres is provided in Table 1. We also demonstrate the numerical performance of the algorithms via simulations in Appendix B.

## 2 Preliminaries

Throughout the paper, unless otherwise specified, we use ˜ O , to suppress dependency on other parameters except for ε , and only keep leading factor. For example, 1 α log( 1 ε )(log log 1 ε ) = ˜ O (log 1 ε ) . We first recall certain preliminaries on Riemannian manifolds; additional preliminaries are provided in Appendix C. We refer the readers to Lee [2018] for more details.

Let M be a Riemannian manifold of dimension d equipped with metric g . The manifold M is assumed to be complete, connected Riemannian manifold without boundary. For a point x ∈ M , T x M denotes the tangent space at x . For any v, w ∈ T x M , we can write the metric as g x ( v, w ) = ⟨ v, w ⟩ g . For x ∈ M and v ∈ T x M , exp x ( v ) denotes the exponential map. We use grad and dV g to represent the Riemannian gradient and the Riemannian volume form respectively.

For x ∈ M , Cut( x ) denotes the cut locus of x . For x, y ∈ M , we use d ( x, y ) to denote the geodesic distance between x and y . Let div denotes the Riemannian divergence, and Laplace-Beltrami operator ∆ : C ∞ ( M ) → C ∞ ( M ) is defined as the Riemannian divergence of Riemannian gradient: ∆ u = div (grad u ) . We use ν ( t, x, y ) to denote the density of manifold Brownian motion with time t , starting at x , evaluated at y .

Let ( M, F ) be a measurable space. Note that the Riemannian volume form dV g is a measure. A probability measure ρ and its corresponding probability density function p are related through dρ = pdV g . Given a measurable set A ∈ F , P ρ ( A ) denotes the probability assigned to the set A by ρ . We have P ρ ( A ) = ∫ A p ( x ) dV g ( x ) = ∫ A dρ ( x ) .

Definition 1 (TV distance) . Let ρ 1 , ρ 2 be probability measures defined on the measurable space ( M, F ) . The total variation distance between ρ 1 and ρ 2 is defined as ∥ ρ 1 -ρ 2 ∥ TV := sup A ∈F | ρ 1 ( A ) -ρ 2 ( A ) | .

Definition 2 (KL divergence and χ 2 divergence) . Let ρ 1 , ρ 2 be probability measures on the measurable space ( M, F ) , with full support. The Kullback-Leibler (KL) divergence and χ 2 divergence of ρ 1 with respect to ρ 2 are defined as (respectively)

<!-- formula-not-decoded -->

where dρ 1 dρ 2 is the Radon-Nikodym derivative.

It is known that H ρ 2 ( ρ 1 ) ≥ 0 with equality if and only if ρ 1 = ρ 2 . Although the KL divergence is not symmetric, it serve as a 'distance' function between two probability measures. For instance, the well known Pinsker inequality states that ∥ ρ 2 -ρ 1 ∥ 2 TV ≤ 1 2 H ρ 2 ( ρ 1 ) .

Definition 3 (Log-Sobolev Inequality (LSI)) . A probability measure ρ 2 satisfies Log-Sobolev Inequality with parameter α &gt; 0 ( α -LSI ) if H ρ 2 ( ρ 1 ) ≤ 1 2 α J ρ 2 ( ρ 1 ) , ∀ ρ 1 , where J ρ 2 ( ρ 1 ) := ∫ M ∥ grad log ρ 1 ρ 2 ∥ 2 dρ 1 is the relative Fisher information.

We also recall the definition of Poincaré inequality which is a generalization of LSI.

Definition 4 (Poincaré Inequality (PI)) . A probability measure ρ satisfies Poincaré Inequality with parameter α &gt; 0 ( α -PI ) if E ρ ( g 2 ) -E ρ [ g ] 2 ≤ 1 α E ρ [ ∥ grad g ∥ 2 ] , ∀ g ∈ C ∞ ( M )

For more technical details on LSI and PI , see Appendix H.2 and H.3. In Euclidean space, conditions like LSI and PI can be viewed as a relaxation of strong convexity assumption on f , and is used to establish convergence of sampling algorithms in KL divergence. See, for example, Vempala and Wibisono [2019] (for the Langevin Monte Carlo Algorithm) and Chen et al. [2022] (for the Euclidean proximal sampler). For a Riemannian manifold, the Bakry-Émery condition can be used to establish LSI . Informally speaking, when the potential f satisfies certain convexity, the corresponding probability measure satisfies LSI . For more details see for example Bakry et al. [2014] and [Li and Erdogdu, 2023, Appendix B]. When the manifold is compact, it is well known that the only convex function is the constant function, and therefore the Bakry-Émery condition does not yield useful information; but for non-compact manifolds, such a condition may serve as a useful tool to establish LSI . Moreover, recent works translated LSI / PI conditions on π to the Polyak-Lojasiewicz (PL) condition on f . For example, considering e -f ( x ) /t in the low-temperature regime, i.e., t → 0 limit, Chewi and Stromme [2024] related LSI constant and the PL constant. Similarly Gong et al. [2024] related the PI constant and a local PL constant. Chen and Sridharan [2024] considered an 'optimizability" condition and analyzed LSI / PI constant for (informally) t ≤ O (1 /d ) . It is interesting future work to establish similar relationships in the manifold setting.

## 2.1 Curvature

We also need notions of curvature on manifolds to present our main results. Let X ( M ) denote the set of all smooth vector fields on M . Define a map called Riemann curvature endomorphism by R : X ( M ) × X ( M ) × X ( M ) → X ( M ) by R ( X,Y ) Z = ∇ X ∇ Y Z -∇ Y ∇ X Z -∇ [ X,Y ] Z. While such definition is very abstract, we provide an intuitive explanation of what curvature is. Intuitively, on a manifold of positive curvature (say, a 2 -dimensional sphere), geodesics tend to 'contract". More precisely, given x, y ∈ M and v ∈ T x M , we can parallel transport v to u = P y x v ∈ T y M . It is a well-known result that (ignore higher order terms) d (exp x tv, exp y tu ) ≤ (1 -t 2 2 K ) d ( x, y ) for some K (which is actually the sectional curvature). From this, we see that for positive curvature, which means K &gt; 0 , the distance between geodesics would decrease.

Formally, given v, w ∈ T p M being linearly independent, the sectional curvature of the plane spanned by v and w can be computed through K ( v, w ) = ⟨ R ( v,w ) w,v ⟩ | v | 2 | w | 2 -⟨ v,w ⟩ 2 ; see Lee [2018, Proposition 8.29]. On the other hand, Ricci curvature can be viewed as the average of sectional curvatures. The Ricci curvature at x ∈ M along direction v is denoted as Ric x ( v ) , which is equal to the sum of the sectional curvatures of the 2-planes spanned by ( v, b i ) d i =2 where v, b 2 , ..., b d is an orthonormal basis for T x M ; see Lee [2018, Proposition 8.32].

We remark that the Ricci curvature is actually a symmetric 2-tensor field defined as the trace of the curvature endomorphism on its first and last indices [Lee, 2018], which sometimes is written as Ric x ( u, v ) for u, v ∈ T x M . The previous notation is a shorthand of Ric x ( v ) = Ric x ( v, v ) . When we say Ricci curvature is lower bounded by κ , we mean Ric ( v, v ) ≥ κ, ∀ v ∈ T x M, ∥ v ∥ = 1 . We end this subsection through some concrete examples.

1. The hypersphere S d has constant sectional curvature equal to 1 , and constant Ricci curvature Ric = ( d -1) g, ∀ x ∈ M (so that Ric x ( v ) = d -1 for all unit tangent vector v ∈ T x M ).
2. For P m ⊆ R m × m , the manifold of positive definite matrices (with affine-invariant metric), its sectional curvatures are in the interval [ -1 2 , 0] ; see, for example, Criscitiello and Boumal [2023]. Hence its Ricci curvature is lower bounded by -m ( m +1) -1 4 .

## 2.2 Brownian motion on manifolds

Now we briefly discuss Brownian motion on a Riemannian manifold. Recall that in Euclidean space, Brownian motion is described by the Wiener process. Given x ∈ R d and t &gt; 0 , the Brownian motion starting at x with time t has (a Gaussian) density function ν ( t, x, y ) = 1 (2 πt ) d/ 2 e -∥ x -y ∥ 2 2 t . It solves the heat equation ∂ ∂t ν ( t, x, y ) = 1 2 ∆ y ν ( t, x, y ) with initial condition ν (0 , x, y ) = δ x ( y ) .

On a Riemannian manifold, we can describe the density of Brownian motion (heat kernel) through heat equation. Let B x,t be a random variable denoting manifold Brownian motion starting at x with time t and let ν ( t, x, y ) be the density of B x,t . The Brownian motion density ν ( t, x, y ) is then defined

Algorithm 1 Riemannian proximal sampler

```
for k = 0 , 1 , 2 , ... do Step 1 (MBI): From x k , sample y k ∼ π Y | X η ( · , x k ) which is a manifold Brownian increment. Step 2 (RHK): From y k , sample x k +1 ∼ π X | Y η ( · , y k ) ∝ e -f ( x ) ν ( η, x, y k ) . end for
```

as the minimal solution of the following heat equation:

<!-- formula-not-decoded -->

More details can be found in Hsu [2002, Chapter 4]. Unlike the Euclidean case, on Riemannian manifold, the heat kernel does not have a closed-form solution in general. However, some properties of the Euclidean heat kernel is preserved on a Riemannian manifold. One such property is the following: Consider M = R d we have t log ν ( t, x, y ) = t log 1 (2 πt ) d/ 2 -∥ x -y ∥ 2 2 . As t → 0 , we get lim t log ν ( t, x, y ) = -∥ x -y ∥ 2 . On a Riemannian manifold, we have the following result.

t → 0 2

Fact 5 (Varadhan's asymptotic relation [Hsu, 2002]) . For all x, y ∈ M with y / ∈ Cut( x ) , we have

<!-- formula-not-decoded -->

When evaluation of the heat kernel is required for practical applications, the Varadhan asymptotics aforementioned is used [De Bortoli et al., 2022].

Yet another numerical method for evaluating the heat kernel on manifold is truncation method; see, for example, Corstanje et al. [2024, Section 5.1] and De Bortoli et al. [2022]. In many cases, the heat-kernel has an infinite series expansion. For example, a power series expansion of heat kernel on hypersphere is given in Zhao and Song [2018, Theorem 1], and more examples can be found in Eltzner et al. [2021, Example 1-5]. Similar results are also available for more general manifolds; see, for example, Azangulov et al. [2022] for compact Lie groups and their homogeneous space, and Azangulov et al. [2024] for non-compact symmetric spaces. Hence, a natural approach is to truncate this infinite series at an appropriate level. For example, on S 2 ⊆ R 3 , the heat kernel and its truncation up to the l -th term (denoted as ν l ) can be written respectively as

<!-- formula-not-decoded -->

where P 0 i are Legendre polynomials.

## 3 The Riemannian proximal sampler

We now describe the Riemannian Proximal Sampler, introduced in Algorithm 1. Similar to the Euclidean proximal sampler [Lee et al., 2021], the algorithm has two steps. The first step is sampling from the Manifold Brownian Increment (MBI) oracle. The second step is called the Riemannian HeatKernel (RHK) Oracle. Recall that ν ( η, x, y ) denotes the density of manifold Brownian motion with time η . Define a joint distribution π η ( x, y ) ∝ e -f ( x ) ν ( η, x, y ) . Then, step 2 consists of sampling from the aforementioned distribution. When there is no ambiguity, we omit the step size η and simply write π ( x, y ) ∝ e -f ( x ) ν ( η, x, y ) . Algorithm 1 is an idealized algorithm, in the sense that we assume exact access to MBI and RHK oracles. Following Chen et al. [2022], next we provide an intuitive explanation for the algorithm from a diffusion process perspective.

Step 1: For fixed x , we see that π Y | X η ( · , x ) ∝ ν ( η, x, · ) which is the density of Brownian motion starting from x for time η . From this we see that the first step of the algorithm is running forward manifold heat flow: dZ t = dB t .

Step 2: We will illustrate that the second step of the algorithm is running the time-reversed process of the forward process. Consider a stochastic process Z t : t ≥ 0 . When we have observations

of x η ∼ Z η , we can compute the conditional probability of Z 0 conditioned on endpoint Z η . We denote µ ( x 0 | x η ) as the posterior. Bayes Theorem says µ ( x 0 | x η ) ∝ µ ( x 0 ) L ( x η | x 0 ) , where µ ( x 0 ) is the prior guess and the likelihood L depends on the model. We consider the following model (forward heat flow): dZ t = dB t with Z 0 ∼ π X ∝ e -f ( x ) . Then µ ( x 0 ) = π X ( x 0 ) and L ( x η | x 0 ) = ν ( η, x 0 , x η ) . Thus we get µ ( x 0 | x η ) ∝ e -f ( x 0 ) ν ( η, x 0 , x η ) , and we observe that µ ( x 0 | x η ) is exactly π X | Y = x η ( x 0 | x η ) . For the forward heat flow dZ t = dB t with initialization Z 0 ∼ π X ∝ e -f ( x ) , there is a well-defined time reversed process ˆ Z -t , which satisfies ( Z 0 , Z η ) d = ( ˆ Z -η , ˆ Z -0 ) . See Appendix D.2 for more details. Based on this, for the time-reversed process ˆ Z -t , the law of ˆ Z -η conditioned on ˆ Z -0 = z is the same as the posterior µ ( x | z ) discussed previously, i.e., π X | Y = z ( x ) ∝ e -f ( x ) ν ( η, x, z ) . Thus we see that the RHK oracle is, from a diffusion perspective, running the time-reversed process.

Implementing Step 1 and Step 2 is non-trivial on Riemannian manifolds. In Section 5 and Appendix A respectively, we discuss two approaches based on heat-kernel truncation and Varadhan's asymptotics. Furthermore, geodesic random walk [Mangoubi and Smith, 2018, Schwarz et al., 2023] is a popular approach to simulate Manifold Brownian Increments (see Appendix B.1), however to the best of our knowledge (in various metrics of interest) is known only under strong assumptions [Cheng et al., 2022, Mangoubi and Smith, 2018].

## 4 High-accuracy convergence rates

In this section, we provide the convergence rates for the Riemannian Proximal Sampler (Algorithm 1) assuming that the target density satisfies the LSI assumption. Firstly, note that in [Lee et al., 2021] the analysis of Euclidean Proximal Sampler is done assuming the potential function is strongly convex. However, it is known that on a compact manifold, if a function is geodesically convex, then it has to be a constant. Hence assuming the potential f being geodesically convex is not much meaningful. Recently, Cheng et al. [2022] discussed an analog of log-concave distribution on manifolds. Although their setting works for compact manifolds, it requires the Riemannian Hessian of the potential f to be lower bounded by some curvature-related value, which is still restrictive. Hence, we adopt the setting as in Chen et al. [2022], assuming that the target distribution satisfies the LSI.

In Section 4.1, we consider the case where both steps of Algorithm 1 are implemented exactly, and in Section 4.2, we consider the case when MBI and RHK oracles are inexact. Regarding notation, we let ρ X k ( x ) , ρ Y k ( y ) denote the law of x and y generated by Algorithm 1 at k -th iteration, assuming exact MBI and exact RHK oracles. When the oracles are inexact, we let ˜ ρ X k ( x ) , ˜ ρ Y k ( y ) to denote the law of x and y generated by Algorithm 1 at k -th iteration.

## 4.1 Rates with exact oracles

Our first result is as follows, with the proof provided in Appendix D.

Theorem 6. Let M be a Riemannian manifold without boundary, i.e., ∂M = ∅ . Denote the distribution for the k -th iteration of Algorithm 1 as x k ∼ ρ X k . Let κ denote the lower bound of Ricci curvature. For any initial distribution ρ X 0 , we have

1. Assume π X satisfies α -LSI .
- For non-negative curvature we have H π X ( ρ X k ) ≤ H π X ( ρ X 0 ) / (1 + ηα ) 2 k , ∀ η &gt; 0
- For negative curvature, we have H π X ( ρ X k ) ≤ H π X ( ρ X 0 ) / (1 + η α 2 ) 2 k , ∀ 0 &lt; η ≤ 1 / | κ | .
2. Assume π X satisfies α -PI .
- For non-negative curvature we have χ 2 π X ( ρ X k ) ≤ χ 2 π X ( ρ X 0 ) / (1 + ηα ) 2 k , ∀ η &gt; 0
- For negative curvature, we have χ 2 π X ( ρ X k ) ≤ χ 2 π X ( ρ X 0 ) / (1 + η α 2 ) 2 k , ∀ 0 &lt; η ≤ 1 / | κ | .

Note that the resulting contraction rate depends on the curvature. If the curvature is non-negative, then we can recover the rate in Euclidean space. But in the case of negative curvature, the rate becomes more complicated, and in order to get the contraction rate as in Euclidean space, we need the step size to be bounded above by some curvature-dependent constant.

The above result provides a high-accuracy guarantee for the Riemannian Proximal Sampler in KLdivergence and χ 2 divergence. To see that, consider for example the case when the Ricci curvature is non-negative. Note that to achieve ε accuracy in KL divergence, we need H π X ( ρ X 0 ) (1+ ηα ) 2 k = ε . Taking log on both sides, we get k = O ( log( H π X ( ρ X 0 ) /ε ) log(1+ ηα ) ) . For small step size η , we have 1 log(1+ ηα ) = O ( 1 ηα ) . Hence k = O ( 1 ηα log H π X ( ρ X 0 ) ε ) = ˜ O ( 1 η log 1 ε ) . As η does not depend on ε , we see that we need ˜ O (log 1 ε ) number of iterations.

There are several challenges in obtaining the aforementioned result for the Riemannian Proximal Sampler. In Euclidean space, when a probability distribution π X satisfies α -LSI , its propagation along heat flow π X ∗ N (0 , tI d ) satisfies α t -LSI , with α t = α 1+ αt . This fact is very important and leveraged in Chen et al. [2022] for proving their convergence rates. A quantitative generalization of such a fact for Riemannian manifolds is not immediate and we establish the required results in Appendix H.2, following Collet and Malrieu [2008], under the required Ricci curvature assumptions.

## 4.2 Rates with inexact oracles

Recall that Algorithm 1 is an idealized algorithm, where we assumed the availability of the MBI and RHK oracles. Note that given x ∈ M , exact MBI oracle generate samples y ∼ π Y | X η ( ·| x ) . And given y ∈ M , exact RHK generate samples x ∼ π X | Y η ( ·| y ) . In practice, exactly implementing these oracles could be computationally expensive or even impossible. For the Euclidean case, we emphasize that, as the heat kernel has an explicit closed form density (which is the Gaussian), prior works, for example, Fan et al. [2023], only consider inexact Restricted Gaussian Oracles and control the propagated error along iterations.

In this section, we derive rates of convergence in the setting where both the MBI and RHK oracles are implemented inexactly. Specifically, we assume we are able to approximately implement the MBI oracle by generating y ∼ ˆ π Y | X η ( ·| x ) , and approximately implement the RKH oracle by generating x ∼ ˆ π X | Y η ( ·| y ) , see Assumption 1 below.

Assumption 1. Denote the output of exact RHK oracle as π X | Y η ( ·| y ) and inexact RHK oracle as ˆ π X | Y η ( ·| y ) . Similarly, denote the output of exact MBI oracle as π Y | X η ( ·| x ) and inexact MBI oracle as ˆ π Y | X η ( ·| x ) . Let ζ RHK and ζ MBI be the desired accuracy. We assume that, for inverse step size η -1 = ˜ O (log 1 ζ ) , the RHK and MBI oracle implementations can achieve respectively ∥ ˆ π X | Y η ( ·| y ) -π X | Y η ( ·| y ) ∥ TV ≤ ζ RHK , ∀ y , and ∥ ˆ π Y | X η ( ·| x ) -π Y | X η ( ·| x ) ∥ TV ≤ ζ MBI , ∀ x . We then let ζ := max { ζ RHK , ζ MBI } .

The need for assuming the step size satisfies η -1 = ˜ O (log 1 ζ ) for the approximation quality is as follows. Recall from the discussion below Theorem 6 that the complexity of Riemannian Proximal Sampler depends on the step size as O ( 1 η ) . Thus if η became too small, for example η -1 = O ( 1 ε ) , then the overall complexity would be Poly ( 1 ε ) , which is not a high-accuracy guarantee.

We also briefly explain the intuition in assuming total variation distance error bound in oracle quality, and postpone the detailed discussion to Section 5. To guarantee a high quality oracle, we need a high quality approximation of heat kernel. As mentioned previously, a popular method is through truncation of infinite series. Theoretically, the L 2 truncation error can be bounded for compact manifold [Azangulov et al., 2022], which says that the difference between the heat kernel and the approximation of heat kernel are close. This naturally imply an error bound in total variation distance, which motivates us to consider the propagated error in total variation distance.

We first start with a result quantifying the error propagated along iterations, under the availability of inexact oracles. The proof of the following result is provided in Appendix E.

Lemma 7. Let ρ X k denote the law of X through exact oracle implementation of Algorithm 1, and ˜ ρ X k denote the law of x through inexact oracle implementation of Algorithm 1. Under Assumption 1, we have ∥ ρ X k ( x ) -˜ ρ X k ( x ) ∥ TV ≤ k ( ζ RHK + ζ MBI ) .

Based on this result, we next obtain the following result analogues to Theorem 6; the proof is provided in Appendix E.

Theorem 8. Similar to Theorem 6, let M be a Riemannian manifold without boundary. Assume Assumption 1 holds. For any initial distribution ρ X 0 , to reach ˜ O ( ε ) total variation distance with oracle accuracy ζ = ζ RHK = ζ MBI = ε log 2 1 ε and step size 1 η = ˜ O (log 1 ε ) (for negative curvature, we additionally require η ≤ 1 / | κ | ),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remarks By Villani [2008, Thm. 6.15], W 1 ( µ, ν ) ≤ D ∥ µ -ν ∥ TV , where D is the manifold's diameter. For compact manifolds, when D is constant, our TV bound directly implies a W 1 bound.

## 5 Implementation of inexact oracles via heat kernel trucation

Theorem 8 shows that as long we have sufficient accuracy of MBI and RHK oracles satisfying Assumption 1, we can have a high-accuracy Riemannian sampling algorithm. In this section, we introduce an approximate implementation, based on heat kernel truncation (as introduced in Section 2) and rejection sampling. Numerical simulations for this approach are provided in Appendix B.2.

First note that for rejection sampling method (in general) there are two key ingredients: a proposal distribution and an acceptance rate. Assume we want to generate samples from ρ through rejection sampling. We choose a suitable proposal distribution denoted as µ , and a suitable scaling constant K such that the acceptance rate K ρ ( x ) µ ( x ) ≤ 1 , ∀ x . We generate a random proposal x ∼ µ and u ∈ [0 , 1] being a uniform random number. Then we compute K ρ ( x ) µ ( x ) , and accept x if u ≤ K ρ ( x ) µ ( x ) .

We also introduce the following definition of Riemannian Gaussian distribution, as defined next, which will be used as the proposal distribution in rejection sampling. A Riemannian Gaussian distribution centered at x ∗ with variable t is µ ( t, x ∗ , x ) ∝ µ u ( t, x ∗ , x ) := exp ( -d ( x ∗ ,x ) 2 2 t ) , where µ u denote an unnormalized version of µ and d denotes the geodesic distance. We use this as our proposal distribution to implement rejection sampling, as exact sampling from such a distribution is well-studied for certain specific manifolds; see, for example, Said et al. [2017] for symmetric spaces and Chakraborty and Vemuri [2019] for Stiefel manifolds. Furthermore, this notion of a Riemannian Gaussian distribution is also used in the study of differential privacy on Riemannian manifolds due to their practical feasibility [Reimherr et al., 2021, Jiang et al., 2023]. In section I.2 we provide an explicit algorithm for sampling from the Riemannian Gaussian distribution on the sphere via rejection sampling.

## 5.1 Implementation of RHK

We first recall the rejection sampling implementation of Restricted Gaussian Oracle (RGO) in the Euclidean setting. Note that, we have log ν u ( η, x, y k ) = -1 2 η ∥ x -y k ∥ 2 , where ν u = exp( -1 2 η ∥ x -y k ∥ 2 ) is an unnormalized heat kernel (or the Gaussian density) in Euclidean space. Then we have π X | Y η ( · , y k ) ∝ e -f ( x ) -1 2 η ∥ x -y k ∥ 2 . Then, the RGO is implemented through rejection sampling. Specifically, we can first find the minimizer x ∗ ∈ arg min x f ( x ) + 1 2 η ∥ x -y k ∥ 2 . Note that the minimizer represents the mode of π X | Y η ( · , y k ) . We can then sample a Gaussian proposal x p ∼ N ( x ∗ , tI d ) for suitable t centered at the mode x ∗ and perform rejection sampling. For more details, see, for example, Chewi [2023].

On a Riemannian manifold with ν denoting the heat kernel, to sample from π X | Y η ( · , y k ) ∝ e -f ( x ) ν ( η, x, y k ) through rejection sampling, we need evaluations of f ( x ) -log ν ( η, x, y k ) . But in general, we cannot evaluate the heat kernel exactly, hence we seek for certain heat kernel approximations. Hence, we use the truncated heat kernel ν l to replace ν , and perform rejection sampling, see Algorithm 2. In the rejection sampling algorithm, as mentioned previously, we use a Riemannian Gaussian distribution as the proposal for rejection sampling. When the minimizer of g is available, we can set x ∗ to be the minimizer; otherwise, we can simply set x ∗ = y k . We choose suitable step size η

```
Set x ∗ = y k and denote g ( x ) := f ( x ) -log ν l ( η, x, y k ) . Set suitable t and constant C RHK s.t. V RHK ( x ) := exp( -g ( x )+ g ( x ∗ )+ C RHK ) exp( -1 2 t d ( x,x ∗ ) 2 ) ≤ 1 , ∀ x ∈ M for i = 0 , 1 , 2 , ... do Generate proposal x ∼ µ ( t, x ∗ , · ) . Generate u uniformly on [0 , 1] . Return x if u ≤ V RHK ( x ) end for
```

```
Set suitable t and C MBI so that V MBI ( y ) := exp(log ν l ( η,x,y ) -log ν l ( η,x,x )+ C MBI ) exp( -d ( x,y ) 2 2 t ) ≤ 1 , ∀ y ∈ M for i = 0 , 1 , 2 , ... do Generate proposal y ∼ µ ( t, x, · ) . Generate u uniformly on [0 , 1] . Return y if u ≤ V MBI ( y )
```

## Algorithm 2 RHK through rejection sampling Algorithm 3 MBI through rejection sampling end for

and t that depends on η s.t. g ( x ) -g ( x ∗ ) + C RHK ≥ 1 2 t d ( x, x ∗ ) 2 . Such an inequality can guarantee that the acceptance rate (with Riemannian Gaussian distribution µ ( t, x ∗ , x ) as proposal) would not exceed one, i.e., V RHK ( x ) ≤ 1 , ∀ x . Then we see that the output of rejection sampling would follow ˆ π X | Y η ( x | y k ) ∝ exp( f ( x ) -log ν l ( η, x, y k )) . Similarly, to implement the MBI oracle, we also use rejection sampling to get a high-accuracy approximation. Specifically, Algorithm 3 generates inexact Brownian motion starting from x with time η .

## 5.2 Verification of Assumption 1

We now show that Assumption 1 is satisfied for the aforementioned inexact implementation of the Riemannian Proximal Sampler. To do so, we specifically consider the case when the manifold M is compact and is a homogeneous space. Recall that ν l denote the truncated heat kernel with truncation level l . Roughly speaking, a homogeneous space is a manifold that has certain symmetry, including Stiefel manifold, Grassmann manifold, hypersphere, and manifold of positive definite matrices.

Proposition 9. Let M be a compact manifold. Assume further that M is a homogeneous space. With truncation implementation of inexact oracles, in order for Assumption 1 to be satisfied with ζ = ε log 2 1 ε , we need truncation level l to be of order polylog (1 /ε ) .

Sketch of proof: We briefly mention the idea of proof. Azangulov et al. [2022, Proposition 21] provided an L 2 bound on the truncation error, and by Jensen's inequality we get an L 1 bound as desired. With truncation level l to be of order Poly (log 1 ε ) , we can achieve ∫ M | ν ( η, x, y ) -ν l ( η, x, y ) | dV g ( x ) = ˜ O ( ζ ) . See Proposition 19 and Proposition 22 for a complete proof.

Remark. Proposition 9 concerns Algorithms 2 and 3, which is one way to implement the RHK and MBI oracles. Rejection sampling is a part of the implementation in Algorithm 2 and 3, which are both based on truncated heat kernels. The cost of rejection sampling comes in the number of steps of the for loop in both the algorithms. Intuitively, one can expect that if we have a highly-accurate evaluation of the heat kernel, the cost of rejection sampling should be the same as that of rejection sampling with the exact heat kernel.

Remark. On the Euclidean space, for which the exact heat kernel is known, the cost of rejection sampling can be proved to be O (1) Chen et al. [2022]. Hypothetically, even if we have the exact heat kernel on a Riemannian manifold, the cost for rejection sampling is actually unknown for general Riemannian manifolds. For the case of sphere, we provide an end-to-end result (including the cost of rejection sampling) in Corollary 10. In proving this result, we first showed that when the acceptance rate V in rejection sampling would possibly exceed 1 in some unimportant regions, Assumption 1 still holds, via explicit computations (see Appendix I.1). Then, we show that the cost of rejection sampling (even with the inexact heat-kernel based on truncation level as stated in Proposition 9), is O (1) similar to the Euclidean case.

When M is not a homogeneous space, to the best of our knowledge, it is unknown how to implement the truncation method. Exploring this direction to further extend the above result is an interesting direction for future work.

## 5.3 A concrete example on hyperspheres

We provide a more specific computational complexity result that consider the dimension dependency as well as cost for rejection sampling; the proof is provided in Appendix I.1.

Corollary 10. Let M = S d , and let π X satisfies α -LSI with the potential function f additionally being L 1 -Lipschitz on M . Assume without the loss of generality that L 1 ≥ √ d . Consider heat kernel truncation implementation (i.e., Algorithm 2 and 3), without minimization (i.e., start rejection sampling from y k directly), and with step size η = 1 L 2 1 d log L 2 1 d log 2 1 ε ε and truncation level l =

O ( d 2 Poly (log 1 ε )) . To get an ϵ -accurate sample in TV distance, the iteration complexity, is k = ˜ O ( L 2 1 d α log 2 1 ε ) , where we use ˜ O to keep only the leading factors.

## 6 Additional results

- In Setion A, we design another practical implementations of the oracles based on Varadhan's asymptotics. While showing that this implementation satisfies Assumption 1 is left as future work, we show their connection to entropy-regularized proximal point methods on Wasserstein spaces (see Theorem 12).
- We evaluate the empirical performance of both implementations through simulation studies presented in Appendix B.

## 7 Concluding remarks

We introduced the Riemannian Proximal Sampler for sampling from densities on Riemannian manifolds. By leveraging the Manifold Brownian Increments (MBI) and the Riemannian Heat-kernel (RHK) oracles, we established high-accuracy sampling guarantees, demonstrating a logarithmic dependence on the inverse accuracy parameter (i.e., polylog (1 /ε ) ) in the Kullback-Leibler divergence (for exact oracles) and total variation metric (for inexact oracles). Additionally, we proposed practical implementations of these oracles using heat-kernel truncation and Varadhan's asymptotics, providing a connection between our sampling method and the Riemannian Proximal Point Method.

Future works include: (i) characterizing the precise dependency on other problem parameters apart from ε , (ii) improving oracle approximations for enhanced computational efficiency and (iii) extending these techniques to broader classes of manifolds (and other metric-measure spaces).

## Acknowledgments and Disclosure of Funding

Krishnakumar Balasubramanian was supported in part by NSF grant DMS-2413426. Shiqian Ma was supported in part by ONR grant N00014-24-1-2705, NSF grants CCF-2311275 and ECCS-2326591.

## References

- Jason M Altschuler and Sinho Chewi. Faster high-accuracy log-concave sampling via algorithmic warm starts. Journal of the ACM , 71(3):1-55, 2024.
- Christophe Andrieu, Anthony Lee, Sam Power, and Andi Q Wang. Explicit convergence bounds for Metropolis Markov chains: Isoperimetry, spectral gaps and profiles. The Annals of Applied Probability , 34(4):4022-4071, 2024.
- Alexis Arnaudon, Alessandro Barp, and So Takao. Irreversible Langevin MCMC on lie groups. In Geometric Science of Information: 4th International Conference, GSI 2019, Toulouse, France, August 27-29, 2019, Proceedings 4 , pages 171-179. Springer, 2019.
- Iskander Azangulov, Andrei Smolensky, Alexander Terenin, and Viacheslav Borovitskiy. Stationary Kernels and Gaussian Processes on Lie Groups and their Homogeneous Spaces I: the compact case. arXiv e-prints , pages arXiv-2208, 2022.
- Iskander Azangulov, Andrei Smolensky, Alexander Terenin, and Viacheslav Borovitskiy. Stationary Kernels and Gaussian Processes on Lie Groups and their Homogeneous Spaces II: non-compact symmetric spaces. Journal of Machine Learning Research , 25(281):1-51, 2024.
- Dominique Bakry, Ivan Gentil, and Michel Ledoux. Analysis and geometry of Markov diffusion operators , volume 103. Springer, 2014.
- Karthik Bharath, Alexander Lewis, Akash Sharma, and Michael V Tretyakov. Sampling and estimation on manifolds using the Langevin diffusion. arXiv preprint arXiv:2312.14882 , 2023.
- Clément Bonet, Paul Berg, Nicolas Courty, François Septier, Lucas Drumetz, and Minh Tan Pham. Spherical Sliced-Wasserstein. In The Eleventh International Conference on Learning Representations , 2023.
- Simon Byrne and Mark Girolami. Geodesic Monte Carlo on embedded manifolds. Scandinavian Journal of Statistics , 40(4):825-845, 2013.
- Djalil Chafaï. Entropies, convexity, and functional inequalities: On phi-entropies and phi-sobolev inequalities. Journal of Mathematics of Kyoto University , 44(2):325-363, 2004.
- Rudrasis Chakraborty and Baba C Vemuri. Statistics on the Stiefel manifold: Theory and applications. The Annals of Statistics , 47, 2019.
- August Y Chen and Karthik Sridharan. Optimization, isoperimetric inequalities, and sampling via lyapunov potentials. arXiv preprint arXiv:2410.02979 , 2024.
- Yongxin Chen, Sinho Chewi, Adil Salim, and Andre Wibisono. Improved analysis for a proximal algorithm for sampling. In Conference on Learning Theory , pages 2984-3014. PMLR, 2022.
- Yuansi Chen and Khashayar Gatmiry. When does Metropolized Hamiltonian Monte Carlo provably outperform Metropolis-adjusted Langevin algorithm? arXiv preprint arXiv:2304.04724 , 2023.
- Yuansi Chen, Raaz Dwivedi, Martin J Wainwright, and Bin Yu. Fast mixing of Metropolized Hamiltonian Monte Carlo: Benefits of multi-step gradients. Journal of Machine Learning Research , 21(92):1-72, 2020.
- Xiang Cheng, Jingzhao Zhang, and Suvrit Sra. Efficient sampling on Riemannian manifolds via Langevin MCMC. Advances in Neural Information Processing Systems , 35:5995-6006, 2022.
- Sinho Chewi. Log-concave sampling. Book draft available at https://chewisinho.github.io , 2023.
- Sinho Chewi and Austin J Stromme. The ballistic limit of the log-sobolev constant equals the Polyakłojasiewicz constant. arXiv preprint arXiv:2411.11415 , 2024.
- Sinho Chewi, Chen Lu, Kwangjun Ahn, Xiang Cheng, Thibaut Le Gouic, and Philippe Rigollet. Optimal dimension dependence of the Metropolis-adjusted Langevin algorithm. In Conference on Learning Theory , pages 1260-1300. PMLR, 2021.

- Jean-François Collet and Florent Malrieu. Logarithmic Sobolev inequalities for inhomogeneous Markov semigroups. ESAIM: Probability and Statistics , 12:492-504, 2008.
- Marc Corstanje, Frank van der Meulen, Moritz Schauer, and Stefan Sommer. Simulating conditioned diffusions on manifolds. arXiv preprint arXiv:2403.05409 , 2024.
- Christopher Criscitiello and Nicolas Boumal. An accelerated first-order method for non-convex optimization on manifolds. Foundations of Computational Mathematics , 23(4):1433-1509, 2023.
- Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, and Arnaud Doucet. Riemannian score-based generative modelling. Advances in Neural Information Processing Systems , 35:2406-2422, 2022.
- Paromita Dubey and Hans-Georg Müller. Fréchet analysis of variance for random objects. Biometrika , 106(4):803-821, 2019.
- Raaz Dwivedi, Yuansi Chen, Martin J Wainwright, and Bin Yu. Log-concave sampling: MetropolisHastings algorithms are fast. Journal of Machine Learning Research , 20(183):1-42, 2019.
- Benjamin Eltzner, Pernille Hansen, Stephan F Huckemann, and Stefan Sommer. Diffusion means in geometric spaces. arXiv preprint arXiv:2105.12061 , 2021.
- Jiaojiao Fan, Bo Yuan, and Yongxin Chen. Improved dimension dependence of a proximal algorithm for sampling. In The Thirty Sixth Annual Conference on Learning Theory , pages 1473-1521. PMLR, 2023.
- Maurice Fréchet. Les éléments aléatoires de nature quelconque dans un espace distancié. In Annales de l'institut Henri Poincaré , volume 10, pages 215-310, 1948.
- Khashayar Gatmiry and Santosh S Vempala. Convergence of the Riemannian Langevin Algorithm. arXiv preprint arXiv:2204.10818 , 2022.
- Mark Girolami and Ben Calderhead. Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society Series B: Statistical Methodology , 73(2):123-214, 2011.
- Yun Gong, Niao He, and Zebang Shen. Poincare inequality for local log-polyak-\ l ojasiewicz measures: Non-asymptotic analysis in low-temperature regime. arXiv preprint arXiv:2501.00429 , 2024.
- Navin Goyal and Abhishek Shetty. Sampling and optimization on convex sets in Riemannian manifolds of non-negative curvature. In Conference on Learning Theory , pages 1519-1561. PMLR, 2019.
- Ye He, Alireza Mousavi-Hosseini, Krishnakumar Balasubramanian, and Murat A Erdogdu. A Separation in Heavy-Tailed Sampling: Gaussian vs. Stable Oracles for Proximal Samplers. arXiv preprint arXiv:2405.16736 , 2024.
- Elton P Hsu. Logarithmic Sobolev inequalities on path spaces over Riemannian manifolds. Communications in mathematical physics , 189(1):9-16, 1997.
- Elton P Hsu. Stochastic analysis on manifolds . Number 38 in Graduate Studies in Mathematics,. American Mathematical Soc., 2002.
- Chin-Wei Huang, Milad Aghajohari, Joey Bose, Prakash Panangaden, and Aaron C Courville. Riemannian diffusion models. Advances in Neural Information Processing Systems , 35:2750-2761, 2022.
- Yangdi Jiang, Xiaotian Chang, Yi Liu, Lei Ding, Linglong Kong, and Bei Jiang. Gaussian differential privacy on Riemannian manifolds. Advances in Neural Information Processing Systems , 36: 14665-14684, 2023.
- Richard Jordan, David Kinderlehrer, and Felix Otto. The variational formulation of the Fokker-Planck equation. SIAM journal on mathematical analysis , 29(1):1-17, 1998.

- Ravi Kannan, László Lovász, and Miklós Simonovits. Random walks and an o*(n5) volume algorithm for convex bodies. Random Structures &amp; Algorithms , 11(1):1-50, 1997.
- Ravi Kannan, László Lovász, and Ravi Montenegro. Blocking conductance and mixing in random walks. Combinatorics, Probability and Computing , 15(4):541-570, 2006.
- Lingkai Kong and Molei Tao. Convergence of kinetic Langevin Monte Carlo on lie groups. arXiv preprint arXiv:2403.12012 , 2024.
- Yunbum Kook and Santosh S Vempala. Sampling and integration of logconcave functions by algorithmic diffusion. arXiv preprint arXiv:2411.13462 , 2024.
- Yunbum Kook and Matthew S Zhang. Rényi-infinity constrained sampling with d 3 membership queries. In Proceedings of the 2025 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA) , pages 5278-5306. SIAM, 2025.
- Yunbum Kook, Yin-Tat Lee, Ruoqi Shen, and Santosh Vempala. Sampling with Riemannian Hamiltonian Monte Carlo in a constrained space. Advances in Neural Information Processing Systems , 35:31684-31696, 2022.
- Yunbum Kook, Santosh S Vempala, and Matthew S Zhang. In-and-Out: Algorithmic Diffusion for Sampling Convex Bodies. arXiv preprint arXiv:2405.01425 , 2024.
- John M Lee. Introduction to Riemannian manifolds , volume 2. Springer, 2018.
- Yin Tat Lee, Ruoqi Shen, and Kevin Tian. Logsmooth gradient concentration and tighter runtimes for Metropolized Hamiltonian Monte Carlo. In Conference on learning theory , pages 2565-2597. PMLR, 2020.
- Yin Tat Lee, Ruoqi Shen, and Kevin Tian. Structured logconcave sampling with a restricted Gaussian oracle. In Conference on Learning Theory , pages 2993-3050. PMLR, 2021.
- Benedict Leimkuhler and Charles Matthews. Efficient molecular dynamics using geodesic integration and solvent-solute splitting. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences , 472(2189):20160138, 2016.
- Mufan Li and Murat A Erdogdu. Riemannian Langevin algorithm for solving semidefinite programs. Bernoulli , 29(4):3093-3113, 2023.
- Han Cheng Lie, Daniel Rudolf, Björn Sprungk, and Timothy J Sullivan. Dimension-independent Markov chain Monte Carlo on the sphere. Scandinavian Journal of Statistics , 50(4):1818-1858, 2023.
- Chang Liu and Jun Zhu. Riemannian Stein variational gradient descent for Bayesian inference. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 32, 2018.
- Chang Liu, Jun Zhu, and Yang Song. Stochastic gradient geodesic MCMC methods. Advances in neural information processing systems , 29, 2016.
- László Lovász. Hit-and-run mixes fast. Mathematical programming , 86:443-461, 1999.
- Oren Mangoubi and Aaron Smith. Rapid mixing of geodesic walks on manifolds with positive curvature. The Annals of Applied Probability , 28(4):2501-2543, 2018.
- Michelle Muniz, Matthias Ehrhardt, Michael Günther, and Renate Winkler. Higher strong order methods for linear Itô SDEs on matrix Lie groups. BIT Numerical Mathematics , 62(4):1095-1119, 2022.
- Tomohiro Nishiyama and Igal Sason. On relations between the relative entropy and χ 2-divergence, generalizations and applications. Entropy , 22(5):563, 2020.
- Maxence Noble, Valentin De Bortoli, and Alain Durmus. Unbiased constrained sampling with self-concordant barrier Hamiltonian Monte Carlo. Advances in Neural Information Processing Systems , 36:32672-32719, 2023.

Adam Nowak. Personal Communication, 2025.

- Adam Nowak, Peter Sjögren, and Tomasz Z Szarek. Sharp estimates of the spherical heat kernel. Journal de Mathématiques Pures et Appliquées , 129:23-33, 2019.
- Sam Patterson and Yee Whye Teh. Stochastic gradient Riemannian Langevin dynamics on the probability simplex. Advances in neural information processing systems , 26, 2013.
- Gabriel Peyré. Entropic approximation of Wasserstein gradient flows. SIAM Journal on Imaging Sciences , 8(4):2323-2351, 2015.
- Marc J Piggott and Victor Solo. Geometric Euler-Maruyama Schemes for Stochastic Differential Equations in SO (n) and SE (n). SIAM Journal on Numerical Analysis , 54(4):2490-2516, 2016.
- Matthew Reimherr, Karthik Bharath, and Carlos Soto. Differential privacy over Riemannian manifolds. Advances in Neural Information Processing Systems , 34:12292-12303, 2021.
- Salem Said, Hatem Hajri, Lionel Bombrun, and Baba C Vemuri. Gaussian distributions on Riemannian symmetric spaces: statistical learning with structured covariance matrices. IEEE Transactions on Information Theory , 64(2):752-772, 2017.
- Simon Schwarz, Michael Herrmann, Anja Sturm, and Max Wardetzky. Efficient random walks on Riemannian manifolds. Foundations of Computational Mathematics , pages 1-17, 2023.
- Vishwak Srinivasan, Andre Wibisono, and Ashia Wilson. Fast sampling from constrained spaces using the Metropolis-adjusted Mirror Langevin algorithm. In The Thirty Seventh Annual Conference on Learning Theory , pages 4593-4635. PMLR, 2024a.
- Vishwak Srinivasan, Andre Wibisono, and Ashia Wilson. High-accuracy sampling from constrained spaces with the Metropolis-adjusted Preconditioned Langevin Algorithm. arXiv preprint arXiv:2412.18701 , 2024b.
- Santosh Vempala and Andre Wibisono. Rapid convergence of the unadjusted langevin algorithm: Isoperimetry suffices. Advances in neural information processing systems , 32, 2019.
- Cédric Villani. Topics in optimal transportation , volume 58. American Mathematical Soc., 2003.
- Cédric Villani. Optimal transport: old and new , volume 338. Springer, 2008.
- Andre Wibisono. Sampling as optimization in the space of measures: The Langevin dynamics as a composite optimization problem. In Conference on Learning Theory , pages 2093-3027. PMLR, 2018.
- Keru Wu, Scott Schmidler, and Yuansi Chen. Minimax mixing time of the Metropolis-adjusted Langevin algorithm for log-concave sampling. Journal of Machine Learning Research , 23(270): 1-63, 2022.
- Xiangjin Xu. Heat kernel gaussian bounds on manifolds i: manifolds with non-negative ricci curvature. arXiv preprint arXiv:1912.12758 , 2019.
- Tianmin Yu, Shixin Zheng, Jianfeng Lu, Govind Menon, and Xiangxiong Zhang. Riemannian Langevin Monte Carlo schemes for sampling PSD matrices with fixed rank. arXiv preprint arXiv:2309.04072 , 2023.
- Chenchao Zhao and Jun S Song. Exact heat kernel on a hypersphere and its applications in kernel SVM. Frontiers in Applied Mathematics and Statistics , 4:1, 2018.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Yes, see the end of Section 1 for a summary of our main contributions, as well as references on the theorem number.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See section 7 for a discussion on future works.

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

Justification: We did state the assumptions in each theorem, and the proof can be found in appendix.

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

Justification: The information needed to reproduce the toy experiments are provided.

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

Justification: Codes were provided in supplementary material.

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

Justification: The details that are necessary to understand the results are provided in this paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Our work is theoretical and includes some toy examples. The plots are average over 1000 number of trails.

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

Justification: The toy examples are run on a personal laptop using Matlab, and only CPU (AMD Ryzen 7 PRO 5850U).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The authors have read the NeurIPS Code of Ethics and followed it in the paper.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our work is theoretical.

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

Justification: Our work is theoretical and poses no such risks.

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

Justification: Our work is theoretical.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our work is theoretical and does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our work is theoretical, and does not involve crowdsourcing nor research with human subjects.

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

## A Implementation via Varadhan's asymptotics and connection to entropy-regularized JKO scheme

In this section, we consider yet another approximation scheme for implementing Algorithm 1, motivated by its connection with the proximal point method in optimization, where the latter is in the sense of optimization over Wasserstein space 1 [Jordan et al., 1998, Wibisono, 2018, Chen et al., 2022]. Note that the proximal point method is usually called as the JKO scheme after the authors of Jordan et al. [1998].

Specifically, we consider approximating the heat kernel through Varadhan's asymptotics. Let ˆ ν ( η, x, y ) ∝ y exp( -d ( x,y ) 2 2 η ) =: ˆ ν u ( η, x, y ) be an inexact evaluation of heat kernel. According to Varadhan's asymptotics, lim η → 0 ˆ ν ( η, x, y ) = ν ( η, x, y ) . Hence when η is small, ˆ ν is a good approximation of the heat kernel. Note that ˆ ν ( η, x, · ) in Varadhan's asymptotic is exactly the Riemannian Gaussian distribution µ ( η, x, · ) . Denote ˜ π ( x, y ) = exp( -f ( x ) -d ( x,y ) 2 2 η ) . With inexact MBI implemented through Riemannian Gaussian distribution and inexact RHK implemented through rejection sampling (Algorithm 2) to generate ˜ π X | Y ( x | y ) ∝ exp( -f ( x ) -d ( x,y ) 2 2 η ) , we obtain Algorithm 4.

For the case when M = S d , we prove in Appendix I that to sample from ˜ π X | Y ( x | y ) through rejection sampling, with suitable parameters, the cost is O (1) in both dimension d and step size η . Obtaining similar results for more general manifolds seems non-trivial. Numerical simulations for this approach are provided in Appendix B.3. Verifying Assumption 1 for this implementation is open.

Algorithm 4 Inexact manifold proximal sampler with Varadhan's asymptotics for k = 0 , 1 , 2 , ... do From x k , sample y k ∼ ˜ π Y | X ( · , x k ) which is a Riemannian Gaussian distribution. From y k , sample x k +1 ∼ ˜ π X | Y ( · , y k ) ∝ e -f ( x ) -d ( x,y 2 k ) 2 η using Algorithm 2. end for

## A.1 RHKas a proximal operator on Wasserstein space

We first show that the inexact RHK output in Algorithm 4 can be viewed as a proximal operator on Wasserstein space, generalizing the Euclidean result in Chen et al. [2020] to the Riemannian setting. Recall that with a function f and d being a distance function, prox ηf ( y ) = arg min x f ( x ) +

1 2 η d ( x, y ) 2 . The (approximated) joint distribution is ˜ π ( x, y ) = exp( -f ( x ) -d ( x,y ) 2 2 η ) . By direct computation we have the following Lemma (proved in Appendix G).

Lemma 11. We have that

<!-- formula-not-decoded -->

which shows that the ineact RHK implementation is a proximal operator, i.e., ˜ π X | Y = y = prox ηH ˜ π X ( δ y ) .

## A.2 Connection to entropy-regularized JKO scheme

Observe that in Algorithm 4, the Riemannian Gaussian involves distance square, which naturally relates to Wasserstein distance. Now, recall that for a function F in the Wasserstein space, its Wasserstein gradient flow can be approximated through the following discrete time JKO scheme [Jordan et al., 1998]:

<!-- formula-not-decoded -->

1 If M is a smooth compact Riemannian manifold then the Wasserstein space P 2 ( M ) is the space of Borel probability measures on M , equipped with the Wasserstein metric W 2 . We refer the reader to Villani [2003] for background on Wasserstein spaces.

It was proved that as η → 0 , the discrete time sequence { ρ k } converge to the Wasserstein gradient flow of F . Later, Peyré [2015] proposed an approximation scheme through entropic smoothing of Wasserstein distance:

<!-- formula-not-decoded -->

where W 2 ,ε is the entropy-regularized 2-Wasserstein distance defined by (here H is the negative entropy)

<!-- formula-not-decoded -->

In Euclidean space, Chen et al. [2022] showed that the proximal sampler can be viewed as an entropyregularized JKO scheme. We extend such an interpretation to Riemannian manifolds. Specifically, we show that Algorithm 4 which is an approximation of the exact proximal sampler (Algorithm 1), can be viewed as an entropy-regularized JKO as stated in Theorem 12 (proved in Appendix G). Note that on a Riemannian manifold the negative entropy is H ( γ ) := ∫ M × M γ log( γ ) dV g ( x ) dV g ( y ) .

Theorem 12. Recall that π X ∝ e -f . Let x k , y k , x k +1 be generated by Algorithm 4. Let ˜ ρ X k , ˜ ρ Y k and ˜ ρ X k +1 be the distribution of x k , y k , x k +1 , respectively. Then

<!-- formula-not-decoded -->

## B Simulation results

## B.1 Brownian motion approximation via geodesic random walk

In our experiments, to compare against the Riemannian Langevin Monte Carlo Algorithm, we used the geodesic random walk algorithm to simulate the MBI oracle following Cheng et al. [2022], De Bortoli et al. [2022], Schwarz et al. [2023]; see Algorithm 5. More efficient implementation is a topic of great interest in the literature; see, for example, [Schwarz et al., 2023].

## Algorithm 5 Approximation of manifold Brownian motion using geodesic random walk

Input x ∈ M,t &gt; 0 .

Sample ξ being a Euclidean Brownian increment with time t in the tangent space T M .

x

x Output y = exp ( ξ ) .

While it is well-known that geodesic random walks converge asymptotically to the Brownian motion on the manifold, non-asymptotic rates of convergence in various metrics of interest is largely unknown. A basic non-asymptotic error bound for geodesic random walk is available in Wasserstein distance (see Cheng et al. [2022, Lemma 7]). Mixing time results are provided in Mangoubi and Smith [2018]. However, such a result is not immediately applicable to establish high-accuracy guarantees for the Riemannian proximal sampler, when the MBI oracle is implemented via geodesic random walk. An important and interesting future work is establishing rates of convergence for geodesic random walk in various metrics of interest so that those results could be leveraged to obtain high-accuracy guarantees for the Riemannian proximal sampler.

## B.2 Numerical experiments for Algorithms 2 and 3: von Mises-Fisher distribution on hyperspheres

In this experiment, we test the performance of Algorithms 2 and 3 for sampling from the von MisesFisher distribution on hyperspheres and compare it with the Riemannian LMC method. In this case, we have f ( x ) = -κµ T x . Note that this f ( x ) has a unique minimizer on S d . This implies that LSI is satisfied, see [Li and Erdogdu, 2023, Theorem 3.4]. We demonstrate the performance of our Algorithm on S 2 ⊆ R 3 with µ = (10 , 0 . 1 , 2) T and κ = 10 , and on S 5 with µ = (5 , 0 . 1 , 2 , 1 , 1 , 1) T and κ = 10 . For the purpose of numerical demonstration, we sample the Riemannian Gaussian distribution through rejection sampling.

Figure 1: Frechét variance (i.e., E [ d ( x, x ∗ ) 2 ] versus number of iterations. Left and Middle figure correspond to the implementation via Algorithm 2 and 3. Right figure corresponds to implementation via Algorithm 4.

<!-- image -->

To evaluate the performance, we estimate E [ d ( x, x ∗ ) 2 ] , where x ∗ is the minimizer of f , representing the mode of the distribution, and plot it as a function of iterations. Note that the quantity E [ d ( x, x ∗ ) 2 ] is referred to as Fréchet variance Fréchet [1948], Dubey and Müller [2019]. For this, we generate 1000 samples (by generating samples independently via different runs) and compute 1 1000 ∑ 1000 i =1 d ( x i , x ∗ ) 2 . We use rejection sampling to generate unbiased samples and get an estimation of the true value. Due to the biased nature of the Riemannian LMC method, to achieve a high accuracy we need a small step size. Contrary to the Riemanian LMC method, the proximal sampler is unbiased, and it can achieve an accuracy while using a large step size and a smaller number of iterations; see Figures 1-(a) and (b). For both algorithms, we use uniform distribution on the hypersphere as initialization.

## B.3 Numerical experiments for Algorithm 4: manifold of positive definite matrices

In this subsection we illustrate the performance of Algorithm 4 for sampling on the manifold of positive definite matrices. Let P m = { X ∈ GL ( m ) : X T = X and y T Xy &gt; 0 , ∀ y ∈ R m } be the set of symmetric positive definite matrices. According to Bharath et al. [2023, Section 6.2], we can choose g ( U, V ) = tr ( X -1 UX -1 V ) and make ( P m , g ) a Riemannian manifold. It is a non-compact manifold with non-positive sectional curvature, geodesically complete and is a homogeneous space of general linear group GL ( m ) . Additional details are provided in Appendix C.3.

We test the performance of Algorithm 4 when the potential function f ( X ) = 1 2 σ 2 d ( X,I m ) 4 , m = 3 , σ = 0 . 03 , following Bharath et al. [2023]. Note that f is not gradient Lipschitz. In the Figure 1-c, we estimate E [ d ( x, x ∗ ) 2 ] and plot it as function of iterations, where x ∗ = I 3 is the minimizer of f , representing the mode of the distribution. For a baseline comparison, we run Riemannian Langevin Monte Carlo for 200 iterations with decreasing step size to get a reference value of E [ d ( x, x ∗ ) 2 ] , which serves as the true E [ d ( x, x ∗ ) 2 ] . Similar to the previous experiment, we generate 1000 samples from independent run, and compute 1 1000 ∑ 1000 i =1 d ( x i , x ∗ ) 2 for each method. For the Riemannian Langevin Monte Carlo method, we find that if we set step size to 0 . 001 instead of 0 . 0001 , after a few iterations the algorithm diverges (potentially due to lack of gradient Lipschitz condition). But for the proximal sampler (which is an unbiased algorithm), even with a large step size as illustrated in the plots, the approximation scheme still works well and can achieve a higher accuracy than the Riemannian LMC algorithm. For both algorithms we initialize the algorithm at X 0 = 2 I 3 . Note that if we use a random initialization, the Riemannian LMC algorithm (prior work) might diverge (potentially due to lack of gradient Lipschitz condition in our potential).

## B.4 Numerical experiments for Algorithm 4: cost of rejection sampling for higer dimensional case

We also demonstrate that the cost for rejection sampling is not exploded by dimension. Following the same setup as Appendix B.2, we consider sampling from a von Mises-Fisher distribution on S 100 with µ = (10 , 0 . 1 , 2 , 1 , 1 , ..., 1) and κ = 10 . We test the performance of Algorithm 4 in terms of rejection sampling cost. We choose η = 0 . 0001 , and compute the average number of iterations executed by rejection sampling. With a total number of rejection sampling oracle being 1 × 10 6 , the average rejection sampling cost is found to be 2 . 661 .

## C Additional preliminaries

## C.1 Divergence

We will briefly discuss divergence for the manifold setting. More details can be found in Lee [2018]. Recall that in Euclidean space, for a vector field F = ( F 1 , ..., F n ) in R n , divergence of F is defined as ∇· F = ∑ n i =1 ∂F i ∂x i . It has a natural generalization to the manifold setting using interior multiplication and exterior derivative.

The Riemannian divergence is defined as the function such that d ( i X ( dV g )) = (div X ) dV g , where X is any smooth vector field on M , i denotes interior multiplication and d denotes exterior derivative. See for example Lee [2018, Appendix B] for more details. On a Riemannian manifold, recall the volumn form is dV g = √ det( g ij ) dx 1 ∧ ... ∧ dx n . Let Y = ∑ n i =1 Y i ∂ ∂x i . We can compute the interior multiplication as

<!-- formula-not-decoded -->

We can then compute its exterior derivative as

<!-- formula-not-decoded -->

Hence we get div ( Y ) = 1 √ det( g ij ) ∑ n j =1 ∂ ( Y j √ det( g ij )) ∂x j . In Euclidean space, this reduces to div ( Y ) = ∑ n j =1 ∂Y j ∂x j .

For u ∈ C ∞ ( M ) and X ∈ X ( M ) , the divergence operator satisfies the following product rule

<!-- formula-not-decoded -->

Furthermore, we have the 'integration by parts' formula (with ˜ g denote the induced Riemannian metric on ∂M )

<!-- formula-not-decoded -->

When M does not have a boundary, ∂M = ∅ . So we have

<!-- formula-not-decoded -->

## C.2 Normal coordinates

Riemannian normal coordinates. Let x ∈ M . There exist a neighborhood V of the origin in T x M and a neighborhood U of x in M such that the exponential map exp x : V → U is a diffeomorphism. The set U is called a normal neighborhood of x . Given an orthonormal basis ( z i ) of T x M , there is a basis isomorphism from T x M to R d . The exponential map can be combined with the basis isomorphism to get a smooth coordinate map φ : U → R d . Such coordinates are called normal coordinates at x . Under normal coordinates, the coordinates of x is 0 ∈ R d . For more details see for example Lee [2018, Chapter 5]

Cut locus and injectivity radius. Consider v ∈ T x M and let γ v be the maximal geodesic starting at x with initial velocity v . Denote t cut ( x, v ) = sup { t &gt; 0 : the restriction of γ v to [0 , t ] is minimizing } The cut point of x along γ v is γ v ( t cut ( x, v )) provided t cut ( x, v ) &lt; ∞ . The cut locus of x is denoted as Cut( x ) = { q ∈ M : q is the cut point of x along some geodesic. } . The injectivity radius at x is the distance from x to its cut locus if the cut locus is nonempty, and infinite otherwise [Lee, 2018, Proposition 10.36]. When M is compact, the injectivity radius is positive [Lee, 2018, Lemma 6.16].

Theorem 13. [Lee, 2018, Theorem 10.34] Let M be a complete, connected Riemannian manifold and x ∈ M . Then

1. The cut locus of x is a closed subset of M of measure zero.
2. The restriction of exp x to ID( x ) is surjective.
3. The restriction of exp x to ID( x ) is a diffeomorphism onto M \ Cut( x ) .

Here ID( x ) = { v ∈ T x M : | v | &lt; t cut ( x, v | v | ) } is the injectivity domain of x .

Then for any p ∈ M , under normal coordinates, for all well-behaved f , we have

<!-- formula-not-decoded -->

## C.3 Additional details for manifold of positive definite matrices

We briefly mention some properties of P m . The inverse of the exponential map is globally defined and the cut locus of every point is empty. For symmetric matrix S ∈ R m × m ,

<!-- formula-not-decoded -->

We have the following fact.

Lemma 14. Let ϕ ( x ) = d ( x, y ) 2 with y ∈ M being fixed. We have grad ϕ ( x ) = -2 exp -1 x ( y ) .

## D Proof of main Theorems

For a given ϕ , define the ϕ -divergence to be Φ π ( ρ ) = E π [ ϕ ( ρ π )] . Define the following dissipation functional

<!-- formula-not-decoded -->

We can now compute the time derivative of the ϕ -divergence along certain flow.

Let µ X t be the law of the continuous-time Langevin diffusion with target distribution π X ∝ e -f ( x ) . That is, we have the following SDE, dX t = -grad f ( X t ) dt + √ 2 dB t . Then, µ X t satisfies the following Fokker-Planck equation (see Lemma 25 for a proof).

<!-- formula-not-decoded -->

We now show that D π X ( µ X t ) = -∂ t Φ π X ( µ X t ) .

Lemma 15. We have that

<!-- formula-not-decoded -->

Proof. [Proof of Lemma 15] By using the fact that ∂ ∂t µ X t = div ( µ X t grad log µ X t π X ) , we have

<!-- formula-not-decoded -->

where in the last equality we used integration by parts.

To get more intuition on the notion of ϕ -divergence and dissipation functional, consider ϕ ( x ) = x log( x ) . We get KL divergence and fisher information:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Our proof is now based on generalizing the proof in Chen et al. [2022] to the Riemannian setting. In Section D.1 we analyze the first step of proximal sampler by viewing it as simultaneous (forward) heat flow. In Section D.2 we analyze the second step of proximal sampler by viewing it as simultaneous backward flow. Combining the two steps together, we prove convergence of proximal sampler under LSI in Section D.3.

## D.1 Forward step: simultaneous heat flow

We can first compute the time derivative of the ϕ -divergence along simultaneous heat flow.

Lemma 16. Define Q t to describe the forward heat flow. Let ρ X Q t and π X Q t evolve according to the simultaneous heat flow, satisfying

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. [Proof of Lemma 16] Denote ρ X t := ρ X Q t and π X t := π X Q t . Then, we have

<!-- formula-not-decoded -->

Recall that by construction,

<!-- formula-not-decoded -->

and ∂ t π X t = 1 2 ∆ π X t = 1 2 div ( π X t grad log π X t ) . Hence, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, notice that

<!-- formula-not-decoded -->

So we get

<!-- formula-not-decoded -->

## D.2 Backward step: simultaneous backward flow

We leverage the following result.

Theorem 17 (Theorem 3.1 in De Bortoli et al. [2022]) . For a SDE dX t = b ( X t ) dt + dB t , let p t denote the distribution of X t . Denote Y t = X T -t , t ∈ [0 , T ] to be the time-reversed diffusion. We have that dY t = ( -b ( Y t ) + grad log p T -t ( Y t )) dt + dB t .

Note that the time reversal can be understood as ( Y T , Y 0 ) has the same distribution as ( X 0 , X T ) .

Recall that ν ( t, x, y ) is the density of manifold Brownian motion starting from x with time t and evaluated at y , and that π ( x, y ) = π X ( x ) ν ( η, x, y ) . We denote π Y = π X Q η to be the Y -marginal of π ( x, y ) . Let π t := π X Q t . Consider the forward process dX t = dB t with X 0 ∼ π X . We know that the time-reversed process satisfies dY t = grad log π η -t ( Y t ) dt + dB t .

Define Q -t as follows. Given ρ Y , set ρ Y Q -t to be the law at time t , of the solution of the timereversed SDE (with T = η ). Thus if Y 0 ∼ ρ Y , we get X T ∼ ρ Y . By Bayes theorem X 0 ∼ ∫ M π X | Y ( x | y ) dρ Y ( y ) , hence Y T ∼ ∫ M π X | Y ( x | y ) dρ Y ( y ) . For the channel Q -t , we have

1. Q -0 is the identity channel.
2. Given input ρ Y , the output at time η is ρ Y Q -η ( x ) = ∫ M π X | Y ( x | y ) dρ Y ( y ) .

<!-- formula-not-decoded -->

Thus we see that the RHK step of proximal sampler can be viewed as going along the time reversed process. We now have the following result.

Lemma 18. For the time-reversed process, we have

<!-- formula-not-decoded -->

Proof. [Proof of Lemma 18] Denote π -t = π Y Q -t and ρ -t = ρ Y Q -t . The Fokker-Planck equation is

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

where we used the same steps as as in the proof Lemma 16 to obtain

<!-- formula-not-decoded -->

and used integration by parts, to obtain

<!-- formula-not-decoded -->

## D.3 Convergence under LSI

Now we prove the main theorem.

Proof. [Proof of Theorem 6 item 1] We first prove the theorem assuming curvature is non-negative. For the general case, we only need to replace the LSI constant α t , α -t .

1. The forward step. We know π X Q t satisfies LSI with α t := 1 t + 1 α . Using Lemma 16, we have

<!-- formula-not-decoded -->

This implies H π X Q t ( ρ X 0 Q t ) ≤ e -A t H π X ( ρ X 0 ) where, A t = ∫ t 0 α s ds = log(1 + tα ) . We also have e -A t = (1 + tα ) -1 . As a result,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2. The backward step. Using Lemma 18, we have

<!-- formula-not-decoded -->

Since π Y Q -t = π X Q η -t , we know the LSI constant for π Y Q -t is α -t := 1 ( η -t )+ 1 α . Same as in step 1, we get A -η = ∫ η 0 α -t dt = log(1 + αη ) . As a result,

<!-- formula-not-decoded -->

3. Putting together. We have π Y = π X Q η , ρ Y 0 = ρ X 0 Q η . Denote ρ X 1 = ρ Y 0 Q -η , we get

<!-- formula-not-decoded -->

4. Negative curvature. For negative curvature, we use α t as in Proposition 31 (the value to be integrated is κ 1 -e -κt + κd 0 e -κt where 1 α := d 0 ). We compute the integral

<!-- formula-not-decoded -->

Hence we have H π X ( ρ X k ) ≤ H π X ( ρ X 0 )( κ α ( e κη -1)+ κ ) 2 k .

Observe that in general, for x ∈ [0 , 1] we have that 1 -x 2 ≥ e -x . Thus for η &lt; 1 | κ | , we have | κ | η &lt; 1 , hence 1 -| κ | 2 η ≥ e -| κ | η . This implies 1 -e -| κ | η | κ | ≥ 1 2 η . On the other hand, we have 1 -x ≤ e -x , which implies 1 -e -| κ | η | κ | ≤ η . So we have α ( e κη -1)+ κ κ = 1 + α 1 -e -| κ | η | κ | = Θ(1 + αη ) .

To summarize, we have that when κ &lt; 0 ,

<!-- formula-not-decoded -->

which implies H π X ( ρ X k ) ≤ H π X ( ρ X 0 )( 1 1+ α 2 η ) 2 k

## D.4 Convergence under Poincaré inequality

In this subsection, we extend Theorem 6 item 1 from LSI to PI, under χ 2 divergence. We follow exactly the same strategy, but we first discuss how to modify the proof from the LSI setting to PI setting. Recall that if π satisfies Poincaré inequality with parameter α , we have

<!-- formula-not-decoded -->

Note that χ 2 divergence is a ϕ -divergence with ϕ ( x ) = ( x -1) 2 , which allows us have Var π ( dµ dπ ) = E π [( dµ dπ ) 2 ] -E π [ dµ dπ ] 2 = χ 2 π ( µ ) . Also, recall that we defined in Appendix D the dissipation of ϕ -divergence. By definition, the dissipation of χ 2 divergence is D π ( µ ) = 2 E ρ [ ⟨ grad ( µ π -1) , grad log µ π ⟩ ] = 2 E π [ ∥ grad dµ dπ ∥ 2 ] . Hence we can interpret the Poincaré inequality as

<!-- formula-not-decoded -->

From this perspective, to follow the proof for LSI setting, we need to know whether the Poincaré inequality constant is also preserved along heat propagation, in the same way as LSI constant. The answer is yes, see Appendix H.3 Proposition 32.

With this, we can prove Theorem 6 item 2 following exactly the same procedure as Theorem 6 item 1.

Proof. [Proof of Theorem 6 item 2] We first assume curvature is non-negative, then discuss the situation that curvature is negative.

1. The forward step. We know π X Q t satisfies PI with α t := 1 t + 1 α . Using Lemma 16 and Proposition 32, we have

<!-- formula-not-decoded -->

This implies χ 2 π X Q t ( ρ X 0 Q t ) ≤ e -A t H π X ( ρ X 0 ) where, A t = ∫ t 0 α s ds = log(1 + tα ) . We also have e -A t = (1 + tα ) -1 . As a result,

<!-- formula-not-decoded -->

2. The backward step. Using Lemma 18 and Proposition 32, we have

<!-- formula-not-decoded -->

Since π Y Q -t = π X Q η -t , we know the LSI constant for π Y Q -t is α -t := 1 ( η -t )+ 1 α . Same as in step 1, we get A -η = ∫ η 0 α -t dt = log(1 + αη ) . As a result,

<!-- formula-not-decoded -->

3. Putting together. We have π Y = π X Q η , ρ Y 0 = ρ X 0 Q η . Denote ρ X 1 = ρ Y 0 Q -η , we get

<!-- formula-not-decoded -->

4. Negative curvature. For negative curvature, we use α t as in Proposition 31 (the value to be integrated is κ 1 -e -κt + κd 0 e -κt where 1 α := d 0 ). We compute the integral

<!-- formula-not-decoded -->

Hence we have H π X ( ρ X k ) ≤ H π X ( ρ X 0 )( κ α ( e κη -1)+ κ ) 2 k .

Observe that in general, for x ∈ [0 , 1] we have that 1 -x 2 ≥ e -x . Thus for η &lt; 1 | κ | , we have | κ | η &lt; 1 , hence 1 -| κ | 2 η ≥ e -| κ | η . This implies 1 -e -| κ | η | κ | ≥ 1 2 η . On the other hand, we have 1 -x ≤ e -x , which implies 1 -e -| κ | η | κ | ≤ η . So we have α ( e κη -1)+ κ κ = 1 + α 1 -e -| κ | η | κ | = Θ(1 + αη ) .

To summarize, we have that when κ &lt; 0 ,

<!-- formula-not-decoded -->

which implies χ 2 π X ( ρ X k ) ≤ χ 2 π X ( ρ X 0 )( 1 1+ α 2 η ) 2 k

## E Proof of Theorem 8

Recall that ρ X k ( x ) , ρ Y k ( y ) denote the distribution generated by Algorithm 1, assuming exact Brownian motion and exact RHK. This notation is applied for all k . For practical implementation, using inexact RHK and inexact Brownian motion through all the iterations, we denote the corresponding distribution by ˜ ρ X k ( x ) , ˜ ρ Y k ( y ) respectively.

Note that at iteration k -1 , we are at distribution ˜ ρ X k -1 ( x ) . Denote ˆ ρ Y k -1 ( y ) to be the distribution obtained from ˜ ρ X k -1 ( x ) using exact Brownian motion. (Note that ˜ ρ Y k -1 ( y ) denote the distribution obtained from ˜ ρ X k -1 ( x ) using inexact Brownian motion).

We now prove Lemma 7.

Proof. [Proof of Lemma 7] Using triangle inequality, we have

<!-- formula-not-decoded -->

The first part can be bounded by ζ RHK :

<!-- formula-not-decoded -->

For the second part, we have

<!-- formula-not-decoded -->

Here, the last inequality follows from Lemma 37. Together, we have

<!-- formula-not-decoded -->

Iteratively applying this inequality and noting that ∥ ˜ ρ X 0 ( x ) -ρ X 0 ( x ) ∥ TV = 0 , we obtain ∥ ρ X k ( x ) -˜ ρ X k ( x ) ∥ TV ≤ k ( ζ RHK + ζ MBI ) .

Recall that Pinsker's inequality states ∥ µ -ν ∥ TV ≤ √ 1 2 H ν ( µ ) .

## Proof. [Proof of Theorem 8 item 1]

For simplicity, we assume non-negative curvature. The negative curvature case follows from the same proof strategy. Using Pinsker's inequality, we have

<!-- formula-not-decoded -->

We want to bound ∥ ρ X k -π X ∥ TV ≤ 1 2 ε . It suffices to have H π X ( ρ X 0 ) (1+ ηα ) 2 k ≤ 1 2 ε 2 . Hence we need log( 2 H π X ( ρ X 0 ) ε 2 ) ≤ 2 k log(1 + ηα ) , i.e., k = O ( log H π X ( ρ X 0 ) ε 2 log(1+ ηα ) ) .

For small step size η , we have 1 log(1+ ηα ) = O ( 1 ηα ) . Hence k = O ( 1 ηα log H π X ( ρ X 0 ) ε 2 ) = ˜ O ( 1 η log 1 ε ) . Recall that by assumption, 1 η = ˜ O (log 1 ζ ) . We pick ζ = ε log 2 1 ε and consequently 1 η = ˜ O (log log 2 1 ε ε ) = ˜ O (log 1 ε +2loglog 1 ε ) = ˜ O (log 1 ε ) . It follows that

<!-- formula-not-decoded -->

The result then follows from triangle inequality:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. [Proof of Theorem 8 item 2] For simplicity, we assume non-negative curvature. The negative curvature case follows from the same proof strategy. We know that for two probability measures π, ρ we have log(1 + χ 2 π ( ρ )) ≥ H π ( ρ ) ≥ 2 ∥ π -ρ ∥ 2 TV where the first inequality is [Nishiyama and Sason, 2020, equation 16] and the second inequality is Pinsker's inequality.

Hence, we have

<!-- formula-not-decoded -->

We want to bound ∥ ρ X k -π X ∥ TV ≤ 1 2 ε . It suffices to have log ( 1 + χ 2 π X ( ρ X 0 ) (1+ η α 2 ) 2 k ) ≤ 1 2 ε 2 . We need χ 2 π X ( ρ X 0 ) ≤ (exp( 1 2 ε 2 ) -1)(1 + η α 2 ) 2 k , that is,

<!-- formula-not-decoded -->

For small step size η , we have 1 log(1+ ηα/ 2) = O ( 1 ηα ) . Hence k = O ( 1 ηα log χ 2 π X ( ρ X 0 ) ε 2 ) . Recall that by assumption, 1 η = ˜ O (log 1 ζ ) . We pick ζ = ε log 2 1 ε and consequently 1 η = ˜ O (log log 2 1 ε ε ) = ˜ O (log 1 ε +2loglog 1 ε ) = ˜ O (log 1 ε ) . It follows that

<!-- formula-not-decoded -->

The result then follows from triangle inequality:

<!-- formula-not-decoded -->

where kζ = ˜ O ( ε ) .

## F Verification of Assumption 1

In this section, we consider implementing inexact oracles through the truncation method. Recall that we assume M is a compact manifold, which is a homogeneous space.

We use ˆ π Y | X , ˆ π X | Y to denote the output of MBI oracle and RHK when rejection sampling is exact. More precisely, since we use the truncated series to approximate heat kernel, we have ˆ π Y | X ∝ ν l ( η, x, y ) and ˆ π X | Y ∝ e -f ( x ) ν l ( η, x, y ) . When rejection sampling is not exact, i.e., there exists z ∈ M s.t. V ( z ) &gt; 1 , we denote the output to be π Y | X , π X | Y for inexact Brownian motion and inexact RHK, respectively.

<!-- formula-not-decoded -->

In subsection I.1, we consider a more general setting, where the acceptance rate is allowed to exceed 1 at some unimportant region. We show that on S d , for certain choices of parameters, ∥ ˆ π X | Y -π X | Y ∥ TV = ˜ O ( ζ ) and ∥ ˆ π Y | X -π Y | X ∥ TV = ˜ O ( ζ ) . This means that allowing the acceptance rate to exceed 1 in unimportant regions would not cause a significant bias for rejection sampling. It then follows from triangle inequality that π Y | X and π X | Y satisfy Assumption 1.

## F.1 Exact rejection sampling

We prove Proposition 9, i.e., verify that Assumption 1 is satisfied with ζ = ε log 2 1 ε as required in Theorem 8.

## F.1.1 Analysis in total variation distance

The first step is to bound the total variation distance, under the assumption that heat kernel evaluation is of high accuracy. We consider the following characterization of total variation distance (see Lemma 36):

<!-- formula-not-decoded -->

Proposition 19. Let M be a compact manifold. Let ζ be the desired accuracy. Assume for all y ∈ M we have ∫ M | ν ( η, x, y ) -ν l ( η, x, y ) | dV g ( x ) = ˜ O ( ζ ) and for all x ∈ M we have ∫ M | ν ( η, x, y ) -ν l ( η, x, y ) | dV g ( y ) = ˜ O ( ζ ) . Then ∥ ˆ π X | Y -π X | Y ∥ TV = ˜ O ( ζ ) and ∥ ˆ π Y | X -π Y | X ∥ = ˜ O ( ζ ) .

Proof. [Proof of Proposition 19]

Step 1. Note that A 1 := sup x ∈ M e -f ( x ) , A 2 := inf x ∈ M e -f ( x ) are positive constants independent of t . Denote Z 1 = ∫ M e -f ( x ) ν l ( η, x, y ) dV g ( x ) and Z 2 = ∫ M e -f ( x ) ν ( η, x, y ) dV g ( x ) . We know

<!-- formula-not-decoded -->

Hence, we have

<!-- formula-not-decoded -->

where by Lemma 21, we obtain min { Z 1 ,Z 2 } 2 Z 1 Z 2 = ˜ O (1) and

<!-- formula-not-decoded -->

Step 2. Denote Z l = ∫ M ν l ( η, x, y ) dV g ( y ) to be the normalizaing constant for ν l . Since ν is the heat kernel, we simply have ∫ M ν ( η, x, y ) dV g ( y ) = 1 . It holds that

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

Theorem 20 (Theorem 5.3.4 in Hsu [2002]) . Let M be a compact Riemannian manifold. There exist positive constants C 1 , C 2 such that for all ( t, x, y ) ∈ (0 , 1) × M × M ,

<!-- formula-not-decoded -->

Lemma 21. We have 1 / ∫ M e -f ( x ) ν ( η, x, y ) dV g ( x ) = ˜ O (1) .

Proof. [Proof of Lemma 21] Using lower bound of heat kernel from Theorem 20, we have

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

## F.1.2 Analysis of truncation error

Now we discuss the truncation level needed to guarantee a high accuracy evaluation of heat kernel as required in Proposition 19.

Proposition 22. Let M be a compact manifold, and assume M is a homogeneous space. With 1 η = ˜ O (log 1 ε ) and ζ = ε log 2 1 ε , to reach ∥ ν ( η, x, y ) -ν l ( η, x, y ) ∥ 2 L 2 = ˜ O ( ζ ) we need l = Poly (log 1 ε ) . Consequently, to achieve

<!-- formula-not-decoded -->

we need l = Poly (log 1 ε ) .

Proof. [Proof of Proposition 22] Following Azangulov et al. [2022, Proof of Proposition 21] we have

<!-- formula-not-decoded -->

Take 1 η = log 1 ε . Recall that in Theorem 8 we require ζ = ε log 2 1 ε . Requiring C ′ l 1 η 2 e -η 2 l 2 /d C = ˜ O ( ζ ) is equivalent to

<!-- formula-not-decoded -->

Take log on both sides, we get -1 log 2 1 ε l 2 /d C ≤ log ε C ′ l log 4 1 ε . This further implies

<!-- formula-not-decoded -->

It suffices to take l = Poly (log 1 ε ) . We verify that l = Poly (log 1 ε ) can guarantee the bound:

<!-- formula-not-decoded -->

On a homogeneous space, both ν and ν l are stationary [Azangulov et al., 2022]. Hence ∫ M | ν ( η, x, y ) -ν l ( η, x, y ) | dV g ( x ) does not depend on y , and ∫ M | ν ( η, x, y ) -ν l ( η, x, y ) | dV g ( y ) does not depend on x . Therefore using Jensen's inequality,

<!-- formula-not-decoded -->

Note that the same holds for ∫ M | ν ( η, x, y ) -ν l ( η, x, y ) | dV g ( y ) . Hence we get the desired bound, i.e., ∫ M | ν ( η, x, y ) -ν l ( η, x, y ) | dV g ( x ) = ˜ O ( ζ ) and ∫ M | ν ( η, x, y ) -ν l ( η, x, y ) | dV g ( y ) = ˜ O ( ζ ) .

## G Proofs for entropy-regularized JKO scheme

Proof. [Proof of Lemma 11] Note that, we have

<!-- formula-not-decoded -->

where C = 1 ∫ M e -f ( x ) dV g ( x ) , C ′ ( y ) and C ( y ) are some constants that only depends on y . The above computation implies

<!-- formula-not-decoded -->

Lemma 23. The minimization problem

<!-- formula-not-decoded -->

where the constraint means ∫ M γ ( x, y ) dV g ( y ) = ρ X ( x ) , has solution of the form

<!-- formula-not-decoded -->

Proof. [Proof of Lemma 23] Since ∫ M γ ( x, y ) dV g ( y ) = ρ X ( x ) , we have

<!-- formula-not-decoded -->

we can construct the following Lagrangian

<!-- formula-not-decoded -->

Recall that H ( γ ) = ∫ M × M γ log( γ ) dV g ( x ) dV g ( y ) . We have,

<!-- formula-not-decoded -->

For any function f , denote I f ( γ ) = ∫ M × M γ ( x, y ) f ( x, y ) dV g ( x ) dV g ( y ) . We then have

<!-- formula-not-decoded -->

Thus the variation of Lagrangian is given by

<!-- formula-not-decoded -->

We want the above to be zero for all φ . Thus we need 1 2 η d ( x, y ) 2 +log( γ ) + 1 -β ( x ) = 0 which is equivalent to

<!-- formula-not-decoded -->

This implies γ ( x, y ) ∝ e β ( x ) -1 2 η d ( x,y ) 2 Integrating with respect to the y variable, we get

<!-- formula-not-decoded -->

It then follows that

<!-- formula-not-decoded -->

## Lemma 24. The minimization problem

<!-- formula-not-decoded -->

where the constraint means ∫ M γ ( x, y ) dV g ( x ) = ρ Y ( y ) , has solution of the form

<!-- formula-not-decoded -->

Proof. [Proof of Lemma 24] The proof follows similarly to that of Lemma 23. Since ∫ M γ ( x, y ) dV g ( x ) = ρ Y ( y ) , we have

<!-- formula-not-decoded -->

We first constructing the following Lagrangian:

<!-- formula-not-decoded -->

Recall that H ( γ ) = ∫ M × M γ log( γ ) dV g ( x ) dV g ( y ) . Then, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For any function f , denote I f ( γ ) = ∫ M × M γ ( x, y ) f ( x, y ) dV g ( x ) dV g ( y ) . We have

<!-- formula-not-decoded -->

Thus the variation of Lagrangian is

<!-- formula-not-decoded -->

We want the above to be zero for all φ . Thus we need f ( x ) + 1 2 η d ( x, y ) 2 +log( γ ) + 1 -β ( y ) = 0 which is equivalent to

<!-- formula-not-decoded -->

This implies γ ( x, y ) ∝ e β ( y ) -f ( x ) -1 2 η d ( x,y ) 2 . Hence we can integrate with respect to the x variable and get

<!-- formula-not-decoded -->

Therefore, we obtain

<!-- formula-not-decoded -->

Proof. [Proof of Theorem 12] By definition we have

<!-- formula-not-decoded -->

By Lemma 24, we know the solution of

<!-- formula-not-decoded -->

is γ ( x, y ) ∝ ˜ ρ X k ( x ) e -1 2 η d ( x,y ) 2 . Hence the Y -marginal of inexact proximal sampler satisfies

<!-- formula-not-decoded -->

Similarly,

<!-- formula-not-decoded -->

and its solution is χ ( x ) = ∫ M ˜ ρ Y ( y )˜ π X | Y ( x | y ) dV g ( y ) = ˜ ρ X k +1 ( x ) .

## H Auxiliary results

## H.1 Diffusion process on manifold

It is well known that the law of the following SDE dX t = -b ( X t ) dt + dB t is related to the FokkerPlanck equation ∂ t ρ t = div ( ρ t b ( X t ) + 1 2 grad ρ t ) . Here we provide a proof for completeness.

Lemma 25. Let B t denote Brownian motion on a Riemannian manifold M . For SDE dX t = -b ( X t ) dt + dB t , the corresponding Fokker-Planck equation is

<!-- formula-not-decoded -->

Proof. The infinitesimal generator of the SDE is Lf = -⟨ grad f, b ⟩ + 1 2 ∆ f Cheng et al. [2022]. We compute the adjoint of L which is defined by ∫ M fL ∗ hdV g = ∫ M hLfdV g . By divergence theorem, we have

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

Thus we obtained L ∗ h = div ( bh ) + 1 2 ∆ h . By Kolmogorov forward equation [Bakry et al., 2014, Equation 1.5.2], we get

<!-- formula-not-decoded -->

We briefly mention some properties of Markov semigroup. The following results are from Bakry et al. [2014, Section 1.2].

- Definition 26. 1. Given a markov process, the assoicated markov semigroup ( P t ) t ≥ 0 is defined as (for suitable f )

<!-- formula-not-decoded -->

2. Let ρ be the law of X 0 , then P ∗ t ρ is the law of X t . We have

<!-- formula-not-decoded -->

3. Markov operators ( P t ) t ≥ 0 can be represented by kernels corresponding to the transition probabilities of the associated Markov process:

<!-- formula-not-decoded -->

Thus by definition, we have

<!-- formula-not-decoded -->

## H.2 Log-Sobolev inequality and heat flow

In the sampling literature, the log-Sobolev inequality is usually written in the following form:

<!-- formula-not-decoded -->

In the Euclidean setting, we know if µ 1 , µ 2 satisfy α 1 , α 2 -LSI respectively, then their convolution µ 1 ∗ µ 2 satisfies LSI with constant 1 1 α 1 + 1 α 2 , see Chewi [2023, Proposition 2.3.7]. In particular, if we take one of µ to be ν ( t, x, y ) (which is a Gaussian in the Euclidean setting), since the Gaussian density satisfies LSI, we have the following result.

Fact 27. Consider Euclidean space. Let µ be a probability measure that satisfies α -LSI . Then its propogation along heat flow, denoted by µ t = µ ∗ ν t , also satisfies LSI with constant 1 1 α + t = α 1+ tα . Here ν t denote the probability measure corresponding to heat flow for time t .

On a Riemannian manifold, the density for Brownian motion satisfies LSI.

Theorem 28. [Hsu, 1997, Theorem 3.1] Suppose M is a complete, connected manifold with Ric M ≥ -c . Here c ≥ 0 . Then for any smooth function on M , we have

<!-- formula-not-decoded -->

With κ = -c , we know the Brownian motion density for time t satisfies LSI with constant α = κ 1 -e -κt .

As a special case M = R d , we have c = 0 . Hence, the LSI constant became lim c → 0 e ct -1 c = t . That is, (with ν representing the measure for Brownian motion with time t ) H ν ( ρ ) ≤ t 2 I ν ( ρ ) , ∀ ρ . So the LSI constant for Brownian motion is α ν = 1 t .

In the following, we prove that on a Riemannian manifold, such a fact is still true. We follow the idea by Collet and Malrieu [2008, Theorem 4.1]. For notations, we denote Γ( f ) = Γ( f, f ) = ∥ grad f ∥ 2 g . We also require the following intermediate result.

Lemma 29 (Theorem 5.5.2 in Bakry et al. [2014]) . For Markov triple with semigroup ( P t ) t ≥ 0 , the followings are equivalent:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Corollary 30. With P t denote manifold Brownian motion, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. [Proof of Corollary 30] For the second item, we can replace f by g 2 for some g .

<!-- formula-not-decoded -->

Now we already know the manifold Brownian motion density ν t satisfies κ 1 -e -κt -LSI , i.e.,

<!-- formula-not-decoded -->

So we know, with P t representing manifold Brownian motion, P t ( f log f ) -P t f log( P t f ) ≤ c ( t ) P t ( Γ( f ) f ) , where

<!-- formula-not-decoded -->

Hence we know β can be taken as κ , s corresponds to 1 2 t . So we get √ Γ( P t f ) ≤ e -κ 2 t P t √ Γ( f ) .

Proposition 31. Let M be a Riemannian manifold with Ricci curvature bounded below by κ . Let ρ 0 be any initial distribution. Assume ρ 0 satisfies LSI with constant 1 d 0 :

<!-- formula-not-decoded -->

Then the propagation of ρ 0 along heat flow, denoted as ρ t , satisfies LSI with constant

<!-- formula-not-decoded -->

where c ( t ) = 1 -e -κt 2 κ . If κ ≥ 0 , we have c ( t ) ≤ 1 2 t .

<!-- formula-not-decoded -->

Proof. [Proof of Proposition 31] Since ρ 0 satisfies LSI with constant 1 d 0 , equivalently with f replace g 2 , we get

<!-- formula-not-decoded -->

For g &gt; 0 , using Corollary 30, we know the manifold Brownian motion (here represented by P t ) satisfies

<!-- formula-not-decoded -->

where c ( t ) = 1 -e -κt 2 κ . Using property of markov semigroup, we have

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

where the third inequality is due to Corollary 30, and in the last inequality we used Cauchy-Schwarz inequality:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence we know ρ t satisfies LSI with constant

<!-- formula-not-decoded -->

Note that we have lim κ → 0 1 2 c ( t )+ d 0 e -κt = 1 t + d 0 . This means that, we can recover the result for Euclidean space in the limit.

## H.3 Extension of Proposition 31 to Phi-Sobolev inequality

In the last subsection, we showed in Proposition 31 that on a Riemannian manifold, LSI is preserved along the propagation of heat flow. In this subsection, we extend the result to the setting of PhiSobolev inequality. We will prove the following proposition. This result will be useful for proving convergence of the Riemannian proximal sampler under χ 2 divergence and Poincaré inequality (Theorem 6 item 2, the proof is in Appendix D.4 ).

Proposition 32. Let M be a Riemannian manifold with Ricci curvature bounded below by κ . Assume ϕ : I → R is a C 4 function and convex, where I is an interval of R . Further assume 1 ϕ ′′ is concave on I . Let ρ 0 be any initial distribution. Assume ρ 0 satisfies ϕ -entropy inequality [Bakry et al., 2014, Section 7.6.1] with constant 1 d 0 :

<!-- formula-not-decoded -->

Then the propagation of ρ 0 along heat flow, denoted as ρ t , satisfies ϕ -entropy inequality with constant

<!-- formula-not-decoded -->

where c ( t ) = 1 -e -κt 2 κ .

We first present some useful lemmas.

Lemma 33. Let ϕ be such that -1 ϕ ′′ ( x ) is convex. Then

<!-- formula-not-decoded -->

In particular, when ϕ ( x ) = ( x -1) 2 , we know -1 ϕ ′′ ( x ) = -1 2 is a constant, and therefore P t (Γ( g )) ≥ ( P t ( √ Γ( g ))) 2 .

Proof. Using Jensen's inequality, since 1 ϕ ′′ ( x ) is concave, we have 1 ϕ ′′ ( E [ g ]) ≥ E [ 1 ϕ ′′ ( g ) ] .

<!-- formula-not-decoded -->

where in the last inequality we used Cauchy-Schwarz.

Lemma 34. For diffusion Markov triple with semigroup ( P t ) t ≥ 0 , the followings are equivalent:

1. The curvature dimension CD ( β, ∞ ) holds for some β ∈ R .
2. √ Γ( P s f ) ≤ e -βs P s √ Γ( f ) .
3. P s ( f log f ) -P s f log( P s f ) ≤ c ( s ) P s ( Γ( f ) f ) .
4. P s ( f 2 ) -( P s ( f )) 2 ≤ 2 c ( s ) P s (Γ( f )) .

<!-- formula-not-decoded -->

Proof. See Theorem 4.7.2 and Theorem 5.5.2 in Bakry et al. [2014], and Theorem 2.1 in Chafaï [2004].

Corollary 35. With P t denote manifold Brownian motion, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where c ( t ) = 1 -e -κt 2 κ .

Proof. [Proof of Corollary 35] For the second item, we can replace f by g 2 for some g .

<!-- formula-not-decoded -->

Now we already know the manifold Brownian motion density ν t satisfies κ 1 -e -κt -LSI , i.e.,

<!-- formula-not-decoded -->

So we know, with P t representing manifold Brownian motion, P t ( f log f ) -P t f log( P t f ) ≤ c ( t ) P t ( Γ( f ) f ) , where

<!-- formula-not-decoded -->

Hence we know β can be taken as κ , s corresponds to 1 2 t . So we get item 1, 3 and 4.

Now we are ready to prove Proposition 32

Proof. [Proof of Proposition 32]

For g &gt; 0 , using Corollary 35, we know the manifold Brownian motion (here represented by P t ) satisfies

<!-- formula-not-decoded -->

where c ( t ) = 1 -e -κt 2 κ . Using property of markov semigroup, we have

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the third inequality is due to Corollary 30, and in the last inequality we used Lemma 33. Hence we know ρ t satisfies ϕ -entropy inequality with constant

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We remark that the case ϕ ( x ) = ( x -1) 2 recovers Poincaré inequality, and ϕ ( x ) = x log x recovers log-Sobolev inequality.

## H.4 Total variation distance

Lemma 36. For TV distance, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Proof. [Proof of Lemma 36] Denote the set at which supremum is achieved to be A ∗ = { x ∈ M : ρ (2) ( x ) ≥ ρ (1) ( x ) } . Denote ρ (2) , ρ (1) to be the measure, or corresponding probability density function with respect to the Riemannian volumn form, when appropriate.

<!-- formula-not-decoded -->

Now we prove the second equation.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 37. Let ρ (1) , ρ (2) be probability measures. Let ρ (1) t , ρ (2) t denote propagation of ρ (1) , ρ (2) along heat flow on M , with ρ (1) 0 = ρ (1) , ρ (2) 0 = ρ (2) . We have

<!-- formula-not-decoded -->

Proof. [Proof of Lemma 37] By definition we have that for all f ,

<!-- formula-not-decoded -->

Assuming X 0 ∼ ρ (1) , we get

<!-- formula-not-decoded -->

Where we denote ∫ M f ( y ) p t ( x, y ) dV g ( y ) = g ( x ) . Note that

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

## I Concrete example: hypersphere

## I.1 Truncation method on hypersphere

Let M be a hypersphere. Previously, we discussed some existing results which provided a bound on the L 2 norm of ν l -ν . For hypersphere, we can derive a bound in L ∞ norm, see subsection I.1.2. We also consider the situation that the acceptance rate in rejection sampling might exceed 1 , and show that for such a situation, rejection sampling can still produce a high-accuracy sample.

Let V MBI ( y ) , V RHK ( x ) denote the acceptance rate in rejection sampling. Recall that ˆ π Y | X ∝ ν l ( η, x, y ) and ˆ π X | Y ∝ e -f ( x ) ν l ( η, x, y ) . In the actual rejection sampling implementation, if for example in Brownian motion implementation, it happens that there exists y ∈ M , s.t. V ( y ) &gt; 1 , then the output for rejection sampling will not be equal to ˆ π Y | X . For such situations, denote V MBI ( y ) = min { 1 , V MBI ( y ) } and V RHK ( x ) = min { 1 , V RHK ( x ) } . Note that V MBI ( y ) and V RHK ( x ) are the actual acceptance rate in rejection sampling. we denote the corresponding rejection sampling output by π Y | X , π X | Y , respectively.

Intuitively, the region B x ( r ) near x carries most of the probability for both Riemannian Gaussian distribution µ ( t, x, y ) as well as Brownian motion ν ( t, x, y ) , when the variable t is suitably small. Thus instead of choosing parameter to guarantee V RHK ( x ) , V MBI ( y ) ≤ 1 , ∀ x, y ∈ M , it suffices to guarantee V RHK ( x ) ≤ 1 , ∀ x ∈ B y ( r ) and V MBI ( y ) ≤ 1 , ∀ y ∈ B x ( r ) for some r .

Let L 1 be the Lipschitz constant of f . In the rejection sampling algorithm, we will use 1 η = L 2 1 d log 1 ζ (i.e., η = 1 L 2 1 d log 1 ζ ), exp( -d ( x,y ) 2 2( sη ) ) = exp( -d ( x,y ) 2 2( 1 L 2 1 ( d -2) log 1 ζ ) ) as proposal Riemannian Gaussian distribution.

Define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 38 (Nowak et al. [2019], Nowak [2025]) . Let d denote the dimension, and ν denote the heat kernel (that corresponds to ∂ ∂t ν ( t, x, y ) = 1 2 ∆ ν ( t, x, y ) ). For φ represent the size of neighborhood, we have

<!-- formula-not-decoded -->

Fix t = O ( 1 d ) . When φ = O ( 1 d ) , it holds that C ( φ,d ) c ( φ,d ) = O (1) . In this case, we denote C = C ( φ, d ) and c = c ( φ, d ) .

Proposition 39. Let M = S d be a hypersphere. We set 1 η = L 2 1 d log 1 ζ as the step size of proximal sampler, 1 t = L 2 1 ( d -2) log 1 ζ as the parameter for proposal (Riemannian Gaussian) distribution of rejection sampling, and truncation level l = Poly ( d 2 Poly log 1 ζ ) . There exists parameters

<!-- formula-not-decoded -->

s.t. π X | Y , π Y | X satisfy Assumption 1.

Proof. See Proposition 42 and Proposition 43.

Lemma 40. [Xu, 2019, Equation 17] Let ( M d , g ) be a complete manifold with Ricci curvature being non-negative. Then we have

<!-- formula-not-decoded -->

## I.1.1 Proof of Proposition 39

It's important to establish the order of a constant in algorithm first.

Lemma 41. There exists some C MBI = -d 2 log(2 π ) -log( C +1) s.t.

<!-- formula-not-decoded -->

Consequently, for r = √ 4 -2 s C η , we have

<!-- formula-not-decoded -->

Proof. Recall that we require 1 η = ˜ O (log 1 ζ ) .

Write 1 η = C η log 1 ζ where C η = L 2 1 d . Then we can write e -1 ηCη = ζ .

1. Step 1 Consider neighborhood B x ( r 0 ) with r 2 0 = 2 C η . We have

<!-- formula-not-decoded -->

with δ ( x, η ) := e -1 ηCη C η d 2 exp( -d ( x,y ) 2 2 η ) . For this δ , we can see that

<!-- formula-not-decoded -->

Hence C (1 + δ ( x, η )) ≤ C +1 . which further implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now consider a slightly larger neighborhood where s &gt; 1 will be set later: 1 C η ≤ d ( x,y ) 2 2 ≤ 2 -1 s C η , we have

<!-- formula-not-decoded -->

so that when ζ is small, we have

<!-- formula-not-decoded -->

which further implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Together, we conclude that for all y ∈ B x ( r ) ,

<!-- formula-not-decoded -->

## 2. Step 2

We have ν ( η, x, x ) ≥ 1 (2 πη ) d 2 and consequently

<!-- formula-not-decoded -->

Thus for all y ∈ B x ( r ) , for some constant C we have

<!-- formula-not-decoded -->

Therefore there exists some C MBI = -d 2 log(2 π ) -log( C +1) s.t.

<!-- formula-not-decoded -->

Proposition 42. Let M be hypersphere S d so that the truncation error bound can be proved in L ∞ . Consider Algorithm 3 with t = ηs where s = d d -2 &gt; 1 is a constant that does not depend on η, ζ . For small ζ , the error for inexact rejection sampling with ν l is of order ζ , i.e., ∥ ˆ π Y | X -π Y | X ∥ TV = ˜ O ( ζ ) . Hence by triangle inequality, ∥ π Y | X -π Y | X ∥ TV = ˜ O ( ζ ) .

## Proof. [Proof of Proposition 42]

Now we have for all y ∈ B x ( r ) ,

<!-- formula-not-decoded -->

This implies,

Hence we have

<!-- formula-not-decoded -->

Recall that µ denote the density for Riemannian Gaussian distribution. We compute

<!-- formula-not-decoded -->

Thus the desired rejection sampling output can be written as

<!-- formula-not-decoded -->

On the other hand we denote V MBI ( y ) = min { 1 , V MBI ( y ) } , and the actual rejection sampling output is π Y | X = µ ( sη, x, y ) V MBI ( y ) E µ ( sη,x,y ) V MBI ( y ) . Following Fan et al. [2023, Proof of Theorem 6], we get

<!-- formula-not-decoded -->

We aim to derive an upper bound for 2 E [ | V MBI -V MBI | ] | E [ V MBI ] | . Note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and similarly

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence we only need to bound

<!-- formula-not-decoded -->

We used ν l ( η, x, y ) ≤ ζ + ν ( η, x, y ) In the last equality, note that when d ( x,y ) 2 2 &gt; 2 -1 s C η , it holds that

<!-- formula-not-decoded -->

1 1

and then we know ζ 2 -s Poly ( 1 η ) = O ( ζ ) because for small ζ , the term ζ 1 -s Poly ( 1 η ) = O (1) . We used lower bound ∫ B x ( r ) ν l ( η, x, y ) dV g ( y ) :

<!-- formula-not-decoded -->

and for the choice of r , we know this is lower bounded by a constant.

Proposition 43. Let M be hypersphere S d . Consider Algorithm 2 with 1 η = L 2 1 d log 1 ζ and 1 t = L 2 1 ( d -2) log 1 ζ . For small ζ , the error for inexact rejection sampling with ν l is of order ζ , i.e., ∥ ˆ π X | Y -π X | Y ∥ TV = ˜ O ( ζ ) .

Proof. [Proof of Proposition 43]

1. Step 1 Set s = d d -1 in Lemma 41. Let 1 η = L 2 1 d log 1 ζ , C η = L 2 1 d . Note that we have r 2 / 2 = 2 -d -1 d L 2 1 d = d +1 L 2 1 d 2 and 1 t = L 2 1 ( d -2) log 1 ζ . Note that t = d d -2 η . We know, for all x ∈ B r ( y ) , we have

<!-- formula-not-decoded -->

We want to find C RHK so that the previously defined t = d d -2 η can be the variable for proposal distribution, i.e., we need f ( x ) -log ν l ( η, x, y ) -f ( y ) + log ν l ( η, y, y ) ≥ 1 2 t d ( x, y ) 2 + C RHK to hold for all x ∈ B r ( y ) . Hence we require

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

where in the last inequality we take d ( x, y ) = 1 L 1 log 1 ζ . Also note that when ζ is small, | -1 2 log 1 ζ | is small. We can just take C RHK = -1 2 log 1 ζ + C MBI ≈ C MBI .

Hence there exists constant C RHK = -1 2 log 1 ζ -d 2 log(2 π ) -log( C +1) s.t. for all x ∈ B y ( r ) ,

<!-- formula-not-decoded -->

## 2. Step 2

Denote

<!-- formula-not-decoded -->

and V RHK ( x ) = min { 1 , V RHK ( x ) } . Recall that the desired rejection sampling output can be written as ˆ π X | Y = µ ( t, x, y ) V RHK ( x ) E µ ( t,x,y ) V RHK ( x ) . On the other hand the actual rejection sampling output is π X | Y = µ ( t, x, y ) V RHK ( x ) E µ ( t,x,y ) V RHK ( x ) . Following Fan et al. [2023, Proof of Theorem 6], we get

<!-- formula-not-decoded -->

we only need to bound

<!-- formula-not-decoded -->

So it suffices to upper bound

<!-- formula-not-decoded -->

We need a sharper bound for distant points. With 1 T = L 2 1 ( d -0 . 5) log 1 ζ , we have

<!-- formula-not-decoded -->

where in the last inequality we set d ( x, y ) = 2 L 1 log 1 ζ Hence

<!-- formula-not-decoded -->

Here we used the fact that d 2 +0 . 5 d -0 . 5 d 2 &gt; 1 .

Proposition 44. Let R = √ 2 L √ d . The cost for rejection sampling as in Proposition 39 is O (1) number of rejections, in ζ and dimension.

## Proof. [Proof of Proposition 44]

Step 1: We first show that in a local neighborhood we have

<!-- formula-not-decoded -->

We have that

<!-- formula-not-decoded -->

where C η = L 2 1 d so that 1 η = C η log 1 ζ .

We have that for d ( x, y ) 2 ≤ 2 C η ,

<!-- formula-not-decoded -->

Thus we can assume W.L.O.G. that δ ( x, η ) &lt; C δ &lt; 1 is small enough s.t. log(1 -δ ( x, η )) is of constant order, for all d ( x, y ) 2 ≤ 2 C η . Then we have

<!-- formula-not-decoded -->

On the other hand, when x = y , we have ν ( η, x, x ) ≤ C η d 2 .

<!-- formula-not-decoded -->

observing that e -1 ηCη = ζ ≤ C η log 1 ζ = 1 η . Thus we have

<!-- formula-not-decoded -->

Hence for C, c we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 2: This step is similar to Proposition 48. With the first step, we see that

<!-- formula-not-decoded -->

With η = 1 L 2 1 d log 1 ζ and T = 1 L 2 1 ( d +1)log 1 ζ , viewing left hand side as a quadratic function of d ( x, y ) we get

<!-- formula-not-decoded -->

Thus we have

<!-- formula-not-decoded -->

The neighborhood for which the bound is valid is d ( x, y ) 2 ≤ 2 C η , i.e., a ball with radius R = √ 2 L √ d . We have that ( R sin R ) d -1 = O (1) . This allows us to do as exactly in Proposition 49 (with t from Proposition 39) to show the rejection sampling procedure finishes with O (1) number of rejections.

Proof. [Proof of Corollary 10] Using Pinsker's inequality, we have

<!-- formula-not-decoded -->

We want to bound ∥ ρ X k -π X ∥ TV ≤ 1 2 ε . It suffices to have H π X ( ρ X 0 ) (1+ ηα ) 2 k ≤ 1 2 ε 2 . Hence we need log( 2 H π X ( ρ X 0 ) ε 2 ) ≤ 2 k log(1 + ηα ) , i.e., k = O ( log H π X ( ρ X 0 ) ε 2 log(1+ ηα ) ) .

For small step size η , we have 1 log(1+ ηα ) = O ( 1 ηα ) . Hence k = O ( 1 ηα log H π X ( ρ X 0 ) ε 2 ) = ˜ O ( 1 αη log 1 ε ) .

Using Proposition 44, by setting 1 η = L 2 1 d log 1 ζ , the expected number of rejections in rejection sampling is O (1) . We pick ζ = αε L 2 1 d log 2 1 ε and consequently 1 η = L 2 1 d log L 2 1 d log 2 1 ε αε = ˜ O ( L 2 1 d log 1 ε ) . It follows that

<!-- formula-not-decoded -->

The result then follows from triangle inequality:

∥ ˜ ρ X k -π X ∥ TV ≤ ∥ ˜ ρ X k -ρ X k ∥ TV + ∥ ρ X k -π X ∥ TV ≤ k ( ζ RHK + ζ MBI ) + 1 2 ε = ˜ O ( ε ) where from Proposition 39, we can set the ε to be ζ k , so that kζ = ˜ O ( L 2 d α (log 2 1 ε ) αε L 2 d log 2 1 ε ) = ˜ O ( ε ) .

## I.1.2 Heat kernel truncation: hypersphere

In this subsection, we show that on hyperspheres S d , the truncation error bound ∥ ν -ν L ∥ ∞ = ˜ O ( ζ ) can be achieved with truncation level L = ˜ O (Poly (log 1 ε )) . As proved in Zhao and Song [2018], the heat kernel on S d can be written as the following uniformly convergent series (with φ := ⟨ x, y ⟩ R d +1 )

<!-- formula-not-decoded -->

where C α l are the Gegenbauer polynomials. Define

<!-- formula-not-decoded -->

Such M l is constructed to be an upper bound for Gegenbauer polynomials; see Zhao and Song [2018, Proof of Theorem 1]. The following proposition is directly implied by Zhao and Song [2018, Theorem 1], and we provide a proof for completeness.

Lemma 45. For l = Θ( d 2 ) , we have and consequently M l +1 M l = O (1) .

Proof. For l = Θ( d 2 ) , we have (by definition of M and Gamma function) M l = ( l + d -2)! ( d -2)! l ! and M l +1 = ( l +1+ d -2)! ( d -2)!( l +1)! . Hence M l = ( l + d -2)! ( d -2)! l ! ≥ ( l +1) d -2 ( d -2)! and M l +1 = ( l +1+ d -2)! ( d -2)!( l +1)! ≤ ( l + d -1) d -2 ( d -2)! Then we have (note that l = Θ( d 2 ) )

<!-- formula-not-decoded -->

Proposition 46. Let M = S d be a hypersphere. For truncation level L = Θ( d 2 Poly (log 1 ε )) , we can achieve | ν ( η, x, y ) -ν L ( η, x, y ) | = ˜ O ( ζ ) , ∀ x, y ∈ S d .

Proof. [Proof of Proposition 46] Throughout the proof, we denote φ = ⟨ x, y ⟩ R d +1 . The parameters M l satisfies | C d -2 2 l ( x ) | ≤ M l according to Zhao and Song [2018, Proof of Theorem 1]. Hence, we have

<!-- formula-not-decoded -->

Observe that for all l ≥ L +1 , since for large L (that depends on dimension) we have M l +1 M l = O (1) ; see, also, Zhao and Song [2018, Proof of Theorem 1]. Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Algorithm 6 Riemannian Gaussian on hypersphere through rejection sampling

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Generate proposal v ∝ e -1 2 t ∥ v ∥ 2 (in Euclidean space), repeat until ∥ v ∥ ≤ π Generate u uniformly on [0 , 1] .

<!-- formula-not-decoded -->

## end for

Generate E to be an orthonormal basis for T x S d , set v ← v ◦ E ∈ T x S d . Output sample y = exp ∗ ( v )

x

For the last line, note that with L = Poly (log 1 ζ ) and η = 1 C log 1 ε , we have that Lη = Poly (log 1 ζ ) . This implies exp( -Lη ) = O (exp(log ζ )) = O ( ζ ) . Now we compute the truncation error.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## I.2 Sampling from the Riemannian Gaussian distribution on hypersphere

We show that Riemannian Gaussian distribution on hypersphere can be generated efficiently through rejection sampling (Algorithm 6).

Note that Algorithm 6 first generate a Euclidean Gaussian v in the tangent space, which is guaranteed to satisfy ∥ v ∥ ≤ π . Its density can be computed exactly under the normal coordinates. Then we perform rejection sampling to get samples v s.t. exp x ∗ ( v ) follow the Riemannian Gaussian distribution µ ( t, x ∗ , · ) exactly.

Proposition 47. On hypersphere S d , when t = O ( 1 d 2 ) , Algorithm 6 output a sample following Riemannian Gaussian distribution µ ( t, x ∗ , y ) ∝ exp( -1 2 t d ( x ∗ , y ) 2 ) , with iteration complexity O (1) .

Proof. First note that when we generate a tangent space Gaussian restricted to ∥ x ∥ ≤ π , under normal coordinates the corresponding density would be ∝ e -1 2 t | x | 2 , x ∈ B 0 ( π ) .

Recall that the Riemannian metric g of S d under normal coordinates satisfies √ det g = ( sin | y | | y | ) d -1 . For Riemannian Gaussian distribution, we change it from Riemannian volume measure to the measure in local coordinates, and under the local coordinates, it has density

<!-- formula-not-decoded -->

Therefore we can compute the number of expected rejections as

<!-- formula-not-decoded -->

.

Recall that with R = √ 6 1+ d , we know ( R sin R ) d -1 = O (1) . Then we have

<!-- formula-not-decoded -->

Then we get

<!-- formula-not-decoded -->

As long as R 2 t -d ≥ 1 , i.e, t ≤ 6 (1+ d ) 2 , the last equality holds.

## I.3 Varadhan's asymptotics

We consider the approximation scheme introduced in Section A.2 using Varadhan's asymptotics. Let φ ( x ) = 1 2 η d ( x, y ) 2 . Intuitively, we want to see how the function φ can improve the convexity of f + φ .

On a manifold with positive curvature, we consider the situation that we cannot compute the minimizer of g ( x ) = f ( x ) + 1 2 η d ( x, y ) 2 , and instead use y as the approximation of it. Notice that when η is small, since f ( x ) is uniformly bounded, the function g ( x ) is dominated by 1 2 η d ( x, y ) 2 , thus the minimizer of g will be close to y . Therefore it is reasonable to use y as an approximation of the mode of e -g ( x ) . Then in rejection sampling, we use µ ( t, y, x ) as the proposal.

Let L 1 be the Lipschitz constant of f . In the next proposition, we show that for some constant C ε , with certain choices of η and t , it holds that

<!-- formula-not-decoded -->

Consequently, the acceptance rate defined by

<!-- formula-not-decoded -->

is guaranteed to be bounded by 1 . Then, in Proposition 49 we show that the expected number of rejections is O (1) in dimension d and step size η .

Proposition 48. Let f be L 1 -Lipschitz and C ε be some constant. Take η = C ε L 2 1 d . With T = C ε L 2 1 ( d +1) and t = C ε L 2 1 ( d -1) , it holds that

<!-- formula-not-decoded -->

Consequently, the acceptance rate is bounded by 1 , i.e., V ( x ) ≤ 1 , ∀ x ∈ M .

Proof. [Proof of Proposition 48] Since f is L 1 -Lipschitz, we have ∥ grad f ( x ) ∥ ≤ L 1 . Then we have L 1 d ( x, y ) ≥ f ( x ) -f ( y ) ≥ -L 1 d ( x, y ) .

1. The lower bound: The goal is to find some t &gt; 0 and constant C such that

<!-- formula-not-decoded -->

It suffices to find t, C such that

<!-- formula-not-decoded -->

The left hand side can be viewed as a quadratic function of d ( x, y ) . When d ( x, y ) = L 1 1 η -1 t , the left hand side is minimized, and the mimimum is -1 2 L 2 1 1 η -1 t + C . Hence we can take C = 1 2 L 2 1 1 η -1 t . Take η = C ε L 2 1 d and t = C ε L 2 1 ( d -1) . Then we have C = 1 2 L 2 1 1 η -1 t = C ε 2 .

2. The upper bound: For an upper bound, we want some T ≤ η for which we want to show that

<!-- formula-not-decoded -->

Similar as before, it suffices to show

<!-- formula-not-decoded -->

The left hand side is maximized at d ( x, y ) = L 1 1 T -1 η , with maximum 1 2 L 2 1 1 T -1 η -C ε 2 . Take T = C ε L 2 1 ( d +1) . We can then verify that

<!-- formula-not-decoded -->

3. Combining the two steps: From the above two steps, we get

<!-- formula-not-decoded -->

In the following proposition, we show that on a hypersphere (where the Riemannian metric in normal coordinates is well studied), the expected number of rejections which equals to

<!-- formula-not-decoded -->

which is independent of dimension and accuracy.

Proposition 49. Let M be hypersphere. Set C ε = 1 log 1 ε . Assume without loss of generality that L 1 ≥ max { 1 , d +1 √ 6 } . Then with η = C ε L 2 1 d and t = C ε L 2 1 ( d -1) , for small ε , the expected number of rejections is O (1) in both dimension and ε .

Proof. Let T = C ε L 2 1 ( d +1) . We try to bound the expected number of rejections. We compute it as follows:

<!-- formula-not-decoded -->

Using Li and Erdogdu [2023, Lemma 8.2] and Li and Erdogdu [2023, Lemma C.5], when β ≥ d R 2 , using Riemannian normal coordinates we have the following lower bound on the integral:

<!-- formula-not-decoded -->

where B y ( R ) denote the geodesic ball centered at y with radius R .

On the other hand, we have

<!-- formula-not-decoded -->

We next find a suitably small R which only depends on dimension, for which we have R sin R ≤ 1 + 1 d . Using Taylor series for sin( R ) , we have R sin R ≈ R R -R 3 6 . Hence for R 2 ≤ 6 1+ d , we have (approximately) R sin R ≤ 1 + 1 d . Consequently we set R = √ 6 1+ d , and we know ( R sin R ) d -1 = O (1) . Combining the bounds discussed previously, we have

<!-- formula-not-decoded -->

For small ε , we have C ε ≤ 1 . Since we assumed L 1 ≥ 1 and L 2 1 ≥ d +1 6 , we have 1 -exp( -1 2 ( L 2 1 ( d +1) C ε R 2 -d )) ≥ 1 -exp( -1 2 ( ( d +1) 2 6 6 d +1 -d )) ≥ 1 -exp( -1 2 ) . As a result, we see that the expect number of rejections is of order O (1) :

<!-- formula-not-decoded -->