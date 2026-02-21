## Sampling from multi-modal distributions with polynomial query complexity in fixed dimension via reverse diffusion

Adrien Vacher CREST, ENSAE Institut Polytechnique de Paris adrien.vacher@ensae.fr

Omar Chehab CREST, ENSAE Institut Polytechnique de Paris omar.chehab@ensae.fr

Anna Korba CREST, ENSAE Institut Polytechnique de Paris anna.korba@ensae.fr

## Abstract

Even in low dimensions, sampling from multi-modal distributions is challenging. Weprovide the first sampling algorithm for a broad class of distributions - including all Gaussian mixtures - with a query complexity that is polynomial in the parameters governing multi-modality, assuming fixed dimension. Our sampling algorithm simulates a time-reversed diffusion process, using a self-normalized Monte Carlo estimator of the intermediate score functions. Unlike previous works, it avoids metastability, requires no prior knowledge of the mode locations, and relaxes the well-known log-smoothness assumption which excluded general Gaussian mixtures so far.

## 1 Introduction

Sampling from a distribution whose density is only known up to a normalization constant is a fundamental problem in statistics. Formally, given some potential V : R d → R such that ∫ e -V ( x ) d x &lt; ∞ , the sampling problem consists in obtaining a sample from some distribution p such that p is ϵ -close to the target µ ∝ e -V with respect to some divergence while maintaining the complexity, i.e., the number of queries to V and possibly to its derivatives, as low as possible. Depending on the shape of the distribution, the typical complexity of existing sampling algorithms can significantly differ.

Log-concave and "PL-like" distributions As in Euclidean optimization, a common assumption in the sampling literature is to assume that µ is log-concave and log-smooth or equivalently, that V is convex and smooth. Specifically, when the potential V is assumed to be α -strongly convex and to have an L -Lipschitz gradient, the popular Unajusted Langevin Algorithm (ULA) is known to achieve fast convergence [Durmus and Moulines, 2017, Dalalyan and Karagulyan, 2019]. Because stronglog concavity implies uni-modality, thus excluding many distributions of interest, this assumption was further relaxed to µ verifying an α -1 -log-Sobolev inequality, later interpreted as a PolyakŁojasiewicz type condition on KL ( ·| µ ) for the Wasserstein geometry [Blanchet and Bolte, 2018]. Under these conditions, ULA was shown to achieve ϵ -error in Kullback-Leibler ( KL ) divergence in ˜ O ( L 2 α -2 dϵ -1 ) queries to ∇ V [Vempala and Wibisono, 2019]. While these polynomial guarantees

do go beyond the uni-modal setting, we show in the next paragraph that most existing algorithms still fail to sample from truly multi-modal distributions.

Multi-modal distributions Designing sampling algorithms for multi-modal distributions is an active area of research. However, most existing sampling algorithms are limited in at least two of the following ways:

1. The query complexity is exponential in the parameters of the problem. For instance, when the global α -strong convexity of V is relaxed with α -strong convexity outside a ball of radius R , the log-Sobolev constant of µ degrades to O ( e 8 LR 2 α ) [Ma et al., 2019, Prop. 2]. In practice, this does not simply translate to poor worst case bounds: practitioners are well aware that when dealing with multi-modal distributions, ULA-based algorithms suffer from metastability , where they get stuck in local modes, leading to slow convergence [Deng et al., 2020].
2. Guarantees are obtained under a log-smoothness assumption; we show in Sec. 3.2 that even for Gaussian Mixtures, which is arguably the most basic multi-modal model, this assumption may not be verified.
3. Explicit a priori knowledge on the target distribution is required for the algorithm to converge. For instance, importance sampling requires a proposal distribution whose "effective" support must cover sufficiently well the one of the target; however, since V can only be evaluated point-wise, the access to such a support is unclear. In fact, this limitation goes beyond the multi-modal setting in some cases: no matter the log-Sobolev constant of the target, similarly to euclidean gradient descent, ULA still requires an a priori bound on the inverse smoothness constant of V and sufficiently small step size in comparison in order to converge.

Generally speaking, finding a sampling algorithm that addresses even the first limitation is not possible. Recent results provide lower-bounds on the complexity of sampling from a multi-modal distribution: they are exponential in the dimension, as shown in Lee et al. [2018, Th K.1], Chak [2024, Th 3.3] and He and Zhang [2025, Th. 3]. However, when the dimensionality is fixed, the question of whether the three limitations can be addressed remains. In fixed dimension, can we sample from a broad class of multi-modal distributions with a polynomial number of queries in the parameters of the problem, (e.g. L, R, α ) and without prior knowledge on these parameters ?

Our contributions Our work answers this question positively. We provide a sampling algorithm that addresses the three limitations outlined above. First, we show that our algorithm has polynomial complexity in all the problem parameters but the dimension, thus enabling to efficiently sample from highly multi-modal distributions in fixed dimension. Second, we show that unlike most existing results, our guarantees hold under relaxed regularity assumptions that cover general Gaussian mixtures. Third, our algorithm yield guarantees without prior knowledge on the parameters of the distribution.

The workhorse of our algorithm is made of two key ingredients: the reverse diffusion scheme, that transfers the sampling problem into a score estimation problem with different levels of noise, and self-normalized importance sampling, to estimate these noisy scores using only query access to V and additional Gaussian samples. Both reverse diffusion [Huang et al., 2024a,b, He et al., 2024] and self-normalized importance sampling [Huang et al., 2025, Jiao et al., 2021, Ruzayqat et al., 2023, Ding et al., 2023, Saremi et al., 2024] were already used separately to sample in a Bayesian setting, where only the log-density of the target is available. However, these works either offer no theoretical guarantees or fail to improve upon existing results. In this article, we combine both reverse diffusion and self-normalized importance sampling in a single algorithm, allowing us to recover the first polynomial time guarantees to sample from highly multi-modal distributions in fixed dimension.

This paper is structured as follows. First, we survey related work on multi-modal sampling in Section 2 and present our main result in Section 3. Then, we detail our sampling algorithm in Section 4 and provide the key ingredients for the proof in Section 5.

## 2 Related work

Before detailing our sampling algorithm in Section 4, we review the main alternatives to ULA when sampling from a multi-modal distribution and the guarantees they offer. We show that existing approaches suffer from at least two of three drawbacks outlined in the introduction: exponential query complexity, a restrictive smoothness assumption that excludes general Gaussian mixtures (as shown later in Sec. 3.2), and unavailable prior knowledge of the distribution.

Proposal-based algorithms At a high level, proposal-based algorithms use a proposal distribution that is easy to sample from and that can either be directly used as a proxy for the target, or, alternatively, whose samples will be rejected (respectively re-weighted) to obtain approximate samples from the target; the corresponding algorithm for the latter case is the well-known rejection sampling (respectively importance sampling) scheme. Guarantees for these methods typically assume that the target distribution is log-smooth. Furthermore, they must use carefully designed proposals that require prior knowledge of the target distribution that is often unavailable in practice.

For instance, when the target is L -log-smooth and with finite second moment m 2 , one can design and sample from a proxy distribution that is ϵ -close in TV to the target using ˜ O (( Lm 2 ϵ -1 ) O ( d ) ) queries to the potential function V and its gradient ∇ V [He and Zhang, 2025]. While this bound is indeed polynomial in fixed dimension, the design of the proxy requires an ϵ -approximation of the global minimum of V which can only be achieved if m 2 , therefore the location of the mass, is explicitly available; this is rarely the case in practice.

Similarly, if we further assume that the target is α -strongly log-concave outside of a ball of radius R , one can achieve an ϵ -precise approximation of the target distribution in TV with polynomial number of queries to ∇ V when d is fixed via an importance sampling scheme [Chak, 2024, Th. 2.3]. In this case, the proposal is such that it coincides with the target when the potential of the target is above some cutoff, and is flat elsewhere. While there exists a cutoff value ensuring the log-concavity of the proposal, thus allowing its efficient sampling, this value depends on the unknown constants L, R and α [Chak, 2024, Prop. 5.1].

Tempering-based algorithms Instead of directly sampling from µ ∝ e -V , these algorithms start by sampling from a flattened version of µ given by µ β ∝ e -βV for small β , and gradually increase β to 1 , a strategy sometimes also referred to as annealing. When µ is assumed to be a finite mixture of the same shifted α -strongly log-concave and L -log-smooth distribution, Lee et al. [2018] proved that for a well-chosen (stochastic) sequence of flattened distributions µ β i , sampling up to precision ϵ in KL can be achieved in poly ( L, α -1 , w -1 min , d, ϵ -1 , R ) queries to ∇ V , where R is the location of the furthest mode and w min the minimum weight in the mixture. However, this setting is quite restrictive: for Gaussian mixtures for instance, it only handles the case where the covariance matrices are identical for all components. Furthermore, the algorithm requires explicit knowledge of R which is unavailable.

Föllmer Flows Instead of directly sampling from the target distribution µ ∝ e -V , Föllmer Flows start from a simple (e.g. Gaussian or Dirac mass) distribution that is progressively interpolated to the target using a Schrödinger bridge. When the initial distribution is a Dirac mass at 0 , this bridge solves a closed-from SDE [Wang et al., 2021, Theorem 3] which can thus be discretized to generate samples from the target. In the context of Bayesian inference, where one only has access to the unnormalized density, Vargas et al. [2022] estimated the drift of the resulting SDE via neural methods; in particular no guarantees on the sampling quality are provided. In Huang et al. [2025], Jiao et al. [2021], Ruzayqat et al. [2023], Ding et al. [2023], the drift is estimated via a Monte-Carlo method. While in appearance, these works provide strong polynomial guarantees, a closer look shows that these guarantees only hold if the function f ( x ) = e -V ( x )+ ∥ x ∥ 2 / 2 is Lipschitz, smooth and bounded from below; we show in Appendix B.1.2 that this assumption is quite restrictive.

Diffusion-based methods Over the past few years, diffusion-based algorithms, and especially reverse diffusion, that we shall review in details in Sec. 3, have emerged as solid candidates for multi-modal sampling. In essence, they allow to transfer the sampling problem into the problem of estimating the scores of intermediate distributions that are given by the convolution of the initial distribution with increasing levels of Gaussian noise. Under an ϵ -oracle of these intermediate scores,

it has been shown that diffusion-based methods could yield an ϵ -approximate sample of the target in poly ( ϵ -1 ) time under milder and milder assumptions [Chen et al., 2023b,a, Benton et al., 2024, Li et al., 2024, Conforti et al., 2025, Gentiloni-Silveri and Ocello, 2025, Cordero-Encinar et al., 2025]. This framework has been applied with tremendous empirical success in generative modeling, where numerous samples of the target are already available and one seeks to produce new samples from the target, for several years now [Song and Ermon, 2019, Ho et al., 2020, Bortoli et al., 2021, Chen et al., 2024]. However, it was only quite recently that this framework has been applied in a Bayesian context, where only an unnormalized density of the target, instead of samples, is available [Huang et al., 2024a,b, He et al., 2024, Grenioux et al., 2024, Akhound-Sadegh et al., 2024]. In a work closely related to ours, Huang et al. [2024b] showed that if the intermediate scores remain L -log-smooth for any noise level, which in particular implies that the target itself is L -log-smooth, their algorithm could reach a complexity of O ( e L 3 log 3 (( Ld + m 2 ) /ϵ ) ) with m 2 the second order moment of the target. In He et al. [2024], the smoothness assumption is relaxed with a sub-quadratic growth assumption V ( x ) -V ( x ∗ ) ≤ L ‖ x -x ∗ ‖ 2 , where x ∗ is any global minimizer of V , yet at the price of an oracle access to V ( x ∗ ) ; under these assumptions, the authors manage to obtain a complexity that is at best O ( L d/ 2 ϵ -d e L ∥ x ∗ ∥ 2 + ∥ x N ∥ 2 ) where x N is the final output sample. Under the reasonable (and desirable) assumption that E [ ‖ x N || 2 ] ≈ m 2 , Jensen's inequality yields an overall O ( L d/ 2 ϵ -d e L ∥ x ∗ ∥ 2 + m 2 ) complexity. In particular, both these works suffer at least two of the limitations mentioned in the introduction, making them ill-suited for multi-modal sampling.

## 3 Presentation of the main result and application to Gaussian Mixtures

## 3.1 Main result

As in the works mentioned above, we rely on the recent advances on reverse diffusion and focus on the task of estimating the intermediate scores. Using an estimator that is described in Sec. 4, we recover polynomial sampling guarantees for densities verifying the assumptions described hereafter.

Assumption 1 (Semi-log-convexity) We assume that µ ∝ e -V is such that log( µ ) is C 2 and verifies ∇ 2 log( µ ) glyph[followsequal] βI d or equivalently ∇ 2 V glyph[precedesequal] βI d for some β ≥ 0 .

This assumption shall be referred to as semi-log-convexity , by analogy with the functional analysis literature [Mikulincer and Shenfeld, 2023, Theorem 3]. Note that is has also been referred to as 1-sided Lipschitzness for ∇ V [Gentiloni-Silveri and Ocello, 2025]. It relaxes the classical logsmoothness assumption which implies the additional lower bound ∇ 2 V glyph[followsequal] βI d . In particular, unlike the latter, a mixture of semi-log-convex densities remains semi-log-convex [Marshall et al., 1979, Chap. 16.B]; we provide a quantitative version of this statement in Sec. 3.2 in the case of Gaussian mixtures.

Assumption 2 (Dissipativity) We assume that µ ∝ e -V is such that there exists a &gt; 0 , b ≥ 0 for which its potential satisfies 〈∇ V ( x ) , x 〉 ≥ a ‖ x ‖ 2 -b for all x ∈ R d .

This assumption is referred to as dissipativity as common in the sampling and optimization literature [Raginsky et al., 2017, Zhang et al., 2017, Erdogdu and Hosseinzadeh, 2021]. Note that this assumption relaxes strong convexity outside of a ball which can be equivalently re-written as 〈∇ V ( x ) -∇ V ( y ) , x -y 〉 ≥ a ‖ x -y ‖ 2 -b for all pairs ( x, y ) ∈ R d , also referred to as strong dissipativity [Eberle, 2013, Erdogdu et al., 2022]. We show in Sec. 3.2 that unlike strong-dissipativity, a mixture of dissipative distributions remain dissipative.

Theorem 1 [Main result, informal] Suppose that Assumption 1 and 2 hold. Then, for all ϵ &gt; 0 , there exists a stochastic algorithm whose parameters only depend on ϵ (and not on the parameters of the problem), that outputs a sample X ∼ ˆ p such that E [KL( µ, ˆ p )] ≲ ϵβ d +3 ( b + d ) /a 2 in O (poly( ϵ -d )) queries to V , where ≲ hides a universal constant as well as log quantities in d, ϵ -1 , β, a, b .

In particular, when d is fixed, this algorithm can output a sample from a distribution that is ϵ -close to µ in expected KL in poly(( b +1) /a, β, ϵ -1 ) running time.

Our algorithm addresses the three limitations outlined in the sections above. When the dimension d is fixed, we obtain a polynomial query complexity. This guarantee does not assume log-smoothness and applies to general Gaussian mixtures, as will be shown in the next subsection. Moreover, this guarantee does not require running the algorithm with any explicit knowledge of the target distribution's constants a, b, β .

| Algorithm                                 | Assumptions                                                                                | Oracle          | Complexity (Total Variation)                      |
|-------------------------------------------|--------------------------------------------------------------------------------------------|-----------------|---------------------------------------------------|
| ULA [Ma et al., 2019]                     | (( α + L ) 1 ∥ x ∥≥ R - L ) I d glyph[precedesequal] ∇ 2 V ( x ) glyph[precedesequal] LI d | ∇ V , L         | O ( e 16 LR 2 ( L/α ) 2 dϵ - 1 )                  |
| Proposal-based [He and Zhang, 2025]       | ‖∇ 2 V ‖ ≤ L , m 2 ≤ M                                                                     | V , ∇ V , M , L | O (( LMϵ - 1 ) O ( d ) )                          |
| RD + ULA [Huang et al., 2024b]            | ‖∇ 2 log( p t ) ‖ ≤ L , m 2 < ∞                                                            | ∇ V             | e O ( L 3 log 3 (( Ld + m 2 /ϵ )))                |
| RD + Rejection Sampling [He et al., 2024] | V ( x ) - V ( x ∗ ) ≤ L ‖ x - x ∗ ‖ 2 , m 2 < ∞                                            | V , V ∗         | O ( L d/ 4 ϵ - d/ 2 e ( L ∥ x ∗ ∥ 2 + m 2 ) / 2 ) |
| RD + Self-normalized IS (ours)            | 2 V βI d , V ( x ) ,x a x 2 b                                                              | V               | O ( √ dβ ( d +3) / 2 √ b + d/aϵ - ( d +2) )       |

∇

glyph[precedesequal]

〈∇

〉 ≥

‖

‖

-

Table 1: Complexity of sampling algorithms. We denote by x ∗ a global minimizer of V , by V ∗ = V ( x ∗ ) the global minimum of V , by m 2 the second moment of the target, by p t the density of the forward process (see Sec. 4), and ‖ · ‖ the operator norm for matrices. In RD + ULA, RD refers to a Reverse Diffusion algorithm and ULA to how the intermediate scores are estimated. Even though originally stated in KL for our work and the one of He et al. [2024], all the complexities are w.r.t. the Total Variation distance (obtained via the Pinsker inequality).

Overall comparison We summarize in Table 1 how our algorithm compares to previous approaches in terms of assumptions, oracles required to run the algorithm and resulting complexity. Along with the work of He and Zhang [2025], our algorithm is the only one that is polynomial in the parameters of the distribution when d is fixed. Furthermore, while our dissipativity assumption is stronger than finite second moment, we relax the log-smoothness assumption by semi-log-convexity which notably covers general Gaussian mixtures. Finally, as mentioned above, because their algorithm requires an ϵ -approximation of the global minimum of V , they require an explicit upper-bound on the second order moment of µ which may not be available in practice.

Numerical illustration In Figure 2, we consider a standard task in the literature: sampling from a mixture of 16 equally weighted Gaussians with unit variance and centers uniformly distributed in [ -40 , 40] 2 [Midgley et al., 2023]. We compare our algorithm against Unadjusted Langevin Algorithm (ULA) and to the reverse diffusion algorithm of Huang et al. [2024a] (RDMC). We also implemented the zeroth-order method of He et al. [2024], but it failed to converge. ULA was initialized from N (0 , I 2 ) and run for 5 × 10 4 steps. All three methods used the same discretization step size h = 0 . 01 . Our reverse diffusion algorithm and RDMC were both run with 500 reverse diffusion steps and both used 100 samples to estimate the intermediate scores. As discussed in Sec. 4, while our samples are simply drawn from a Gaussian distribution, the samples used in RDMC are drawn from a auxiliary, multi-modal distribution generated via an inner ULA step. In this experiment, this inner ULA was initialzed with a standard Gaussian and was run for 100 steps; in particular, we emphasize that while ULA and our algorithm were roughly given the same computational budget, the one given to RDMC was a hundred times superior. As a result, ULA and our algorithm took approximately one minute to run on a computer locally and the RDMC method required over an hour. We observe that unlike the two others, our algorithm successfully recovers all the modes.

In Figure 1, we monitor the convergence of the same three algorithms as a function of the problem difficulty, measured by the distance between the modes. Here, the target distribution is a mixture of three Gaussians in two dimensions. It has equal weights 1 / 3 , equidistant modes located at a distance R from the origin, and different covariances ( I, I/ 2 , I/ 4) . The final error is measured in Wasserstein distance, because it can be easily approximated using samples: we use 500 samples generated by the sampling algorithm and 500 samples from the true target distribution. We allowed each of the three algorithms 10 6 queries to the potential V or to its gradient:

Figure 1: Error in Wasserstein distance as a function of the between-mode distance.

<!-- image -->

we performed 10 6 iterations for ULA, we used 100 steps of reverse diffusion for our algorithm and RDMC. We used 100 inner steps of ULA for RDMC that was initialized with a standard Gaussian. We used 100 particles for score estimation for RDMC and 10000 for our algorithm. As expected, as the between-mode distance grows with R , our algorithm yields the lowest error.

## 3.2 Application to Gaussian mixtures

We now apply our results to derive provable sampling guarantees for general Gaussian mixtures. Apart from the very recent exception of Lytras and Mertikopoulos [2025] that we discuss below, note that despite their wide popularity, no sampling guarantees were yet derived for general Gaussian mixtures. Indeed, unless they verify some specific assumptions such as identical covariance matrices among components [Cordero-Encinar et al., 2025, Lemma B.1], general Gaussian mixtures do not satisfy the log-smoothness assumption, thus they do not fit the framework of many previously discussed works. In fact, their gradient may not even be Hölder continuous: consider the simple counter-example of a two-dimensional mixture µ = 0 . 5 N (0 , Σ 1 ) + 0 . 5 N (0 , Σ 2 ) with covariances Σ 1 = diag (1 , 0 . 5) and Σ 2 = diag (0 . 5 , 1) . On the diagonal x = y , the score is ∇ log( µ )( x, x ) = -3 / 2( x, x ) , while near the diagonal, right above it for instance, the score behaves asymptotically as ∇ log( µ )( x, x ) ∼ x → + ∞ -(2 x, x ) ; we provide a rigorous analysis in Appendix B.2. Fortunately though, Gaussian mixtures do verify Assumptions 1-2, as we next show.

Proposition 2 Let µ = ∑ p i =1 w i N ( µ i , Σ i ) and denote λ min &gt; 0 (resp. λ max ) the minimum (resp. maximum) eigenvalue of the covariance matrices Σ i . It holds that -∇ 2 log( µ ) glyph[precedesequal] I d /λ min and that for all x ∈ R d , 〈-∇ log( µ )( x ) , x 〉 ≥ ‖ x ‖ 2 / (2 λ max ) -λ max max i ( ‖ µ i ‖ /λ min ) 2 .

The proof is deferred to Appendix B.3. Combined with Theorem 1, Proposition 2 shows that we can sample any Gaussian mixture with average precision ϵ in KL in O (poly( κ, R, d, λ -d min , ϵ -d )) queries to V where R = max i ‖ µ i ‖ and κ = λ max /λ min . There exists a relatively recent literature seeking to relax the log-smoothness assumption. For instance, Chatterji et al. [2019], Erdogdu and Hosseinzadeh [2021], Nguyen et al. [2021] work under weak smoothness, i.e. α -Hölder continuous gradient of the potential for α in [0 , 1] (recall that α =1 recovers the smooth case). However, as shown above, this relaxation is not sufficient yet to cover general Gaussian mixtures. The only reference we know of that does is the work of Lytras and Mertikopoulos [2025], who used a regularized version of Langevin to relax the global smoothness condition with local Lipschitz smoothness and polynomial growth of the Lipschitz constant. Yet crucially, their complexity bound scales as a polynomial of the Poincaré constant of µ . We prove in Appendix B.4 that for the mixture µ = 0 . 5 N ( Ru,λ max I d ) + 0 . 5 N ( -Ru,λ min I d ) with u any unit vector and 0 &lt; λ min ≤ λ max , this constant is at least ( R 2 e R 2 / (2 λ max ) ) / 2 . In particular, this method still degrades exponentially with the multi-modality parameter R .

## 4 Our sampling algorithm

In this section, we introduce the reverse diffusion framework and explain how it reduces the sampling problem to that of estimating the scores along the forward Ornstein-Uhlenbeck (OU) process.

## 4.1 Reverse diffusion: from sampling to score estimation

Reverse diffusion methods emerged as an alternative to Langevin-based samplers in order to overcome metastability and were first introduced to the ML community in Song et al. [2021]. They rely on the so-called forward process

<!-- image -->

<!-- formula-not-decoded -->

Figure 2: From left to right: our algorithm vs. ULA vs. Huang et al. [2024a]. The color scheme indicates the probability density value of the distribution we want to sample from (dark is low probability density, bright is high probability). The blue dots are the samples produced by the algorithm.

which corresponds to the standard OU process initialized at µ , that is a specific case of a Langevin diffusion targeting a standard Gaussian, that we will denote π . Note that since the target of this process, the standard Gaussian, is 1 -strongly log-concave, the resulting process converges exponentially fast to the equilibrium. In order to sample from µ , reverse diffusion algorithms rely on the semi-discretized backward process : given a horizon T that we discretize as 0 = t 0 ≤ t 1 ≤ · · · t N -1 ≤ t N = T , the latter writes with Y 0 ∼ p T and where p t is the distribution of the forward process Eq. 1 at time t . Note that this reverse process cannot be readily implemented for two reasons: first, it requires the knowledge of the intermediate scores ∇ log( p t k ) which are not available in closed form. Second, it requires sampling from the distribution p T . Nevertheless, if one can access a proxy s t k of the scores ∇ log( p t k ) , and considering T large enough so that p T ≈ π , we can implement instead

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with Y 0 ∼ π and where all iterations can be solved in closed form. Because the forward process Eq. 1 converges exponentially fast, we can expect the initialization error Y 0 ∼ π instead of Y 0 ∼ p T to be small after a short time T . Furthermore, if the proxies s t k are sufficiently accurate, one can expect that the process output by the approximate scheme Eq. 3 has a distribution that is close to the target µ . Over the past three years, several works provided quantitative bounds of the error induced by the discretization, the use of an approximate score and the initialization error with respect to different divergences and under various assumptions [Bortoli et al., 2021, Lee et al., 2022, Chen et al., 2023b,a, Conforti et al., 2025]. Yet, we shall rely exclusively on the following theorem as it is the most suited to our framework.

Theorem 3 (Conforti et al. [2025]) Assume that µ ∝ e -V has finite Fisher-information w.r.t. π the standard gaussian density in R d :

<!-- formula-not-decoded -->

Then, for the constant step-size discretization t k = kT/N , denoting p the distribution of the sample Y T output by Eq. 3, it holds that

<!-- formula-not-decoded -->

where m 2 is the second order moment of µ and where ≲ hides a universal constant.

The previous theorem shows that under mild assumptions that notably allow for multi-modality, the problem of sampling from µ can be transferred into a score approximation problem along the forward process. In the next subsection, we present an estimator for these intermediate scores that is tractable given the knowledge of the unnormalized density µ ∝ e -V .

## 4.2 Derivation of an estimator

The key observation to derive an estimator of the intermediate scores is that the forward process Eq. 1 is nearly available in closed form. However, this closed form may be written in different manners.

Different expressions of the scores Consider Eq. 1 integrates to X t = √ λ t X 0 + √ 1 -λ t Z , where λ t = e -2 t , Z is a standard d -dimensional Gaussian, and X 0 is a random variable simulating the target distribution. The corresponding density convolves the target density with a Gaussian, as where π is a standard Gaussian and µ ( · ) ∝ exp( -V ( · )) is the target distribution. From this expression, we can obtain different formulas for the scores. We retain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Derivations can be found in Appendix A. The second [Akhound-Sadegh et al., 2024], third [Huang et al., 2024a, He et al., 2024, Grenioux et al., 2024], and fourth [Saremi et al., 2024] identities have been used to build Monte Carlo estimators of the score. Hence, one natural way to group these identities is by how difficult it is to draw the samples y . The second and fourth identities sample from a Gaussian distribution, whereas the first and third identities sample from an auxiliary multi-modal distribution from generating y | x . As t gets closer to zero, this auxiliary distribution resembles the original target, so sampling from it progressively becomes as difficult as sampling from the original target density. This observation explains why, unsurprisingly, Huang et al. [2024a], He et al. [2024] do not improve upon existing results. Another way to group these identities is whether or not they require evaluating the score of the target distribution: identities one and two do and are referred to as TSI estimators, referring to the Target Score Identity (TSI) used to obtain them [Bortoli et al., 2024]; identities three and four do not and are referred to as DSI estimators, referring to a Denoising Score Identity (DSI) used to obtain them [He et al., 2025]. Combinations of TSI and DSI estimators have also been considered [Phillips et al., 2024, He et al., 2025].

A self-normalized estimator We now focus on the estimator of the scores that we use. It is a rewrite of the fourth identity and is a ratio of expectations under Gaussians

<!-- formula-not-decoded -->

with Y t ∼ N (0 , (1 -e -2 t ) I d ) which is easy to simulate. While conventional statistical wisdom may suggest using independent samples to estimate both the numerator and the denominator, we voluntarily choose to correlate them and implement instead

<!-- formula-not-decoded -->

where the y i are independent Gaussians such that y i ∼ N (0 , (1 -e 1 -2 t ) I d ) ; we refer to this estimator as self-normalized as common in the sampling literature [Agapiou et al., 2017]. The key property of self-normalized estimators is that they remain nearly bounded: in our case, it holds uniformly in z that

<!-- formula-not-decoded -->

This boundedness will allow us to derive a non-asymptotical control on the quadratic error that we present in the next section.

We now explain how this differs from previous work. While this estimator was already considered in [Saremi et al., 2024], the authors did not use it within the reverse diffusion pipeline and more importantly, we are the first work to derive quantitative guarantees for this estimator, which is one of our core contributions. Similarly, an estimator close to this one was considered in the context of Föllmer Flows [Jiao et al., 2021, Ruzayqat et al., 2023, Ding et al., 2023] in order to approximate the shift in the corresponding Schrödinger bridge. We show in Appendix B.1.1 that while this shift is itself close to the intermediate scores that we seek to approximate, the resulting self-normalized estimator is more degenerate. Namely, as discussed above, their guarantees are significantly weaker.

## 5 Sketch of proof of the main result

The proof is decomposed in three steps: (i) we derive a non-asymptotic bound on the quadratic error of the estimator presented in Eq. 5 (ii) we show that under Assumptions 1-2, the integrated error of

this estimator (that appears in the bound of Theorem 3) can be fully controlled by the zeroth and second order moments of the ratio Φ t = p 2 V t /p V t , where p 2 V t (respectively p V t ) is the density of the forward process defined in Eq. 1 initialized at µ 2 (respectively µ ) (iii) we provide a quantitative bound on these moments as well as other relevant quantities and we conclude with Theorem 3.

Proposition 4 (Non-asymptotic bound on the quadratic error) For all z ∈ R d , n and t &gt; 0 , denoting p 2 V t (respectively p V t ) the density of the forward process defined in Eq. 1 initialized at µ 2 ∝ e -2 V (respectively µ ∝ e -V ), Z 2 V (respectively Z V ) the normalizing constant of µ 2 (respectively µ ) and π the density of the standard Gaussian, it holds that

The complete proof is left to Appendix C yet we briefly sketch the main arguments. We split the expectation on the event A where the empirical denominator ˆ D of Eq. 5 (respectively the empirical numerator ˆ N ) is not too small (respectively not too large) with respect to its expectation D (respectively ‖ N ‖ ) and on the complementary ¯ A . Over A , we use a second-order Taylor expansion to make the variances of both the numerator and the denominator appear, and compute them explicitly. Conversely, the quadratic error of the estimator remains almost bounded on ¯ A . We use Chebyshev's inequality to upper-bound P ( ¯ A ) and make the variances of the numerator and of the denominator appear again, which concludes the proof.

<!-- formula-not-decoded -->

Remark 5 Generic bounds on self-normalized estimators were derived in Agapiou et al. [2017, Theorem 2.3]. Yet, we show in Appendix B.5 that if used in our context, they would involve at least an extra 1 / ( p V t ) factor which can cause the integrated error ∫ E [ ‖ ˆ s t,n ( z ) -∇ log ( p V t ) ( z ) ‖ 2 ] d p V t ( z ) to diverge.

Then, we need to control the Laplacian of the forward processes as well as their gradient appearing in θ t in the former proposition. As mentioned in the previous section, the intermediate scores can be re-written as with q t,z ( x ) ∝ e -V ( x ) e -∥ e -t x -z ∥ 2 2(1 -e -2 t ) . Now we note that if µ ∝ e -V is dissipative, so is q t,z ; in particular we can quantitatively bound its second order moment and a fortiori , upper-bound ‖∇ log ( p V t ) ( z ) ‖ 2 .

<!-- formula-not-decoded -->

Proposition 6 (Regularity bounds on the forward process) Suppose that Assumption 2 holds. Then, for all z ∈ R d and t &gt; 0 , it holds that

The complete proof is left in Appendix D.2. This result implies that the θ t term defined in Proposition 4 is at most of order of order θ t ( z ) ∼ (1 + ‖ z ‖ 2 ) w.r.t. z . In particular, the average integrated error E [ ‖ ˆ s t,n ( z ) -∇ log ( p V t ) ‖ 2 L 2 ( p V t ) ] can be upper-bounded with respect to the zeroth and the second order moments of the ratio Φ t = p 2 V t /p V t . In the next proposition, we show that these moments are bounded under the semi-log-convexity assumption. By a slight abuse of notation, we shall denote m i (Φ t ) = ∫ ‖ x ‖ i Φ t ( z )d z the i -th moment of Φ t .

<!-- formula-not-decoded -->

Lemma 7 (Bounds on the moments of the ratio) Assume that µ ∝ e -V has finite second moment m 2 and that Assumption 1 holds. Then,



<!-- formula-not-decoded -->

 where Z 2 V (respectively Z V ) is the normalization constant of e -2 V (respectively e -V ) and where π in the right-hand-side refers to the constant π ≈ 3 . 14 .



The proof of this lemma is deferred to Appendix E. It relies on a key result in Mikulincer and Shenfeld [2023, Lemma 5] where it is shown that for β -semi-log convex distributions, the following bound holds:

In particular, the right-hand-side remains bounded w.r.t. β whenever t &gt; 0 , which allows to avoid an exponential dependence in β in our final bounds. It suffices now to control I ( µ, π ) , m 2 and µ (0) in order to apply Theorem 3 and conclude.

<!-- formula-not-decoded -->

Lemma 8 Assume that µ ∝ e -V is such that Assumption 1 and 2 hold: ∇ 2 V glyph[precedesequal] βI d and 〈∇ V ( x ) , x 〉 ≥ a ‖ x ‖ 2 -b for some a &gt; 0 , b, β ≥ 0 . Then,

Here again, the π in the right-hand-side refers to the constant π ≈ 3 . 14 .

<!-- formula-not-decoded -->

The proof is left to Appendix D.1. We can now state our main result.

Theorem 9 Under Assumptions 1 and 2, if we run the algorithm of Eq. 3 with T = log(1 /ϵ ) , N = 1 /ϵ , t k = kT/N and with the stochastic score estimators ˆ s n k ,t k defined in Eq. 5 with n k = dϵ -(2 d +3) , then, denoting ˆ p the stochastic distribution of the output Y N , it holds that

<!-- formula-not-decoded -->

where ≲ hides a universal constant as well as log factors with respect to d, ϵ -1 , a, b, β . In particular, the error above is achieved in ∑ N k =1 n k = dϵ -2( d +2) queries to V . The proof is deferred to Appendix F and is mainly an application of the results collected above.

## 6 Conclusion

In this article, we successfully applied the reduction from sampling to intermediate score estimation, initiated over the past three years by Chen et al. [2023b,a], Conforti et al. [2025], Benton et al. [2024], to the problem of low dimensional multi-modal sampling. Using the self-normalized estimator of the scores, our results provide polynomial query complexity guarantees in fixed dimension, apply to general Gaussian mixtures, and do not require prior knowledge of the target distribution's constants. Interesting future directions include extending theoretical guarantees to more general multi-modal distributions with heavy tails for instance.

We note that our sampling algorithm is based on time-reversed diffusions, which have recently gained traction for sampling from unnormalized densities. Our method stands out by offering rigorous, non-asymptotic theoretical guarantees on query complexity, especially in the presence of score estimation error that we precisely quantify. Such theoretical guarantees are scarce and we believe our results are therefore a meaningful and timely contribution to the field.

## 7 Acknowledgements

This work was supported by the Agence Nationale de la Recherche (ANR) through the JCJC WOS project and the PEPR PDE-AI project (ANR-23-PEIA-0004). We thank Pierre Monmarché for his help in proving Lemma 13.

## References

Sergios Agapiou, Omiros Papaspiliopoulos, Daniel Sanz-Alonso, and Andrew M Stuart. Importance sampling: Intrinsic dimension and computational cost. Statistical Science , 2017.

Tara Akhound-Sadegh, Jarrid Rector-Brooks, Avishek Joey Bose, Sarthak Mittal, Pablo Lemos, Cheng-Hao Liu, Marcin Sendera, Siamak Ravanbakhsh, Gauthier Gidel, Yoshua Bengio, et al. Iterated denoising energy matching for sampling from Boltzmann densities. In ICML , 2024.

- Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly d-Linear convergence bounds for diffusion models via stochastic localization. In ICLR , 2024.
- Adrien Blanchet and Jérôme Bolte. A family of functional inequalities: Lojasiewicz inequalities and displacement convex functions. Journal of Functional Analysis , 2018.
- Valentin De Bortoli, James Thornton, Jeremy Heng, and A. Doucet. Diffusion Schrödinger Bridge with applications to score-based generative modeling. In NeurIPS , 2021.
- Valentin De Bortoli, Michael Hutchinson, Peter Wirnsberger, and Arnaud Doucet. Target score matching. arXiv preprint arXiv:2402.08667 , 2024.
- Martin Chak. On theoretical guarantees and a blessing of dimensionality for nonconvex sampling. arXiv preprint arXiv:2411.07776 , 2024.
- Niladri S. Chatterji, Jelena Diakonikolas, Michael I. Jordan, and Peter L. Bartlett. Langevin monte carlo without smoothness. In AISTATS , 2019.
- Hongrui Chen, Holden Lee, and Jianfeng Lu. Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. In ICML , 2023a.
- Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. In ICLR , 2023b.
- Sitan Chen, Vasilis Kontonis, and Kulin Shah. Learning general Gaussian mixtures with efficient score matching. In COLT , 2024.
- Giovanni Conforti, Alain Durmus, and Marta Gentiloni Silveri. Score diffusion models without early stopping: finite Fisher information is all you need. SIAM Journal on Mathematics of Data Science (SIMODS) , 2025.
- Paula Cordero-Encinar, O Deniz Akyildiz, and Andrew B Duncan. Non-asymptotic analysis of diffusion annealed Langevin Monte Carlo for generative modelling. arXiv preprint arXiv:2502.09306 , 2025.
- Arnak S Dalalyan and Avetik G Karagulyan. User-friendly guarantees for the Langevin Monte Carlo with inaccurate gradient. Stochastic Processes and their Applications , 2019.
- Wei Deng, Guang Lin, and Faming Liang. A contour stochastic gradient Langevin dynamics algorithm for simulations of multi-modal distributions. In NeurIPS , 2020.
- Zhao Ding, Yuling Jiao, Xiliang Lu, Zhijian Yang, and Cheng Yuan. Sampling via Föllmer Flow. arXiv preprint arXiv:2311.03660 , 2023.
- Alain Durmus and Eric Moulines. Nonasymptotic convergence analysis for the unadjusted Langevin algorithm. The Annals of Applied Probability , 2017.
- Andreas Eberle. Reflection couplings and contraction rates for diffusions. Probability Theory and Related Fields , 2013.
- Murat A Erdogdu and Rasa Hosseinzadeh. On the convergence of Langevin Monte Carlo: The interplay between tail growth and smoothness. In COLT , 2021.
- Murat A. Erdogdu, Rasa Hosseinzadeh, and Shunshi Zhang. Convergence of Langevin Monte Carlo in chi-squared and Rényi divergence. In AISTATS , 2022.
- Marta Gentiloni-Silveri and Antonio Ocello. Beyond log-concavity and score regularity: Improved convergence bounds for score-based generative models in W2-distance. In ICML , 2025.
- Louis Grenioux, Maxence Noble, Marylou Gabrié, and Alain Oliviero Durmus. Stochastic localization via iterative posterior sampling. In ICML , 2024.
- Jiajun He, Wenlin Chen, Mingtian Zhang, David Barber, and José Miguel Hernández-Lobato. Training neural samplers with reverse diffusive KL divergence. In AISTATS , 2025.

- Ye He, Kevin Rojas, and Molei Tao. Zeroth-order sampling methods for non-log-concave distributions: Alleviating metastability by denoising diffusion. In NeurIPS , 2024.
- Yuchen He and Chihao Zhang. On the query complexity of sampling from non-log-concave distributions. In COLT , 2025.
- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS , 2020.
- Jian Huang, Yuling Jiao, Lican Kang, Xu Liao, Jin Liu, and Yanyan Liu. Schrödinger-Föllmer sampler. IEEE Transactions on Information Theory , 2025.
- Xunpeng Huang, Hanze Dong, Hao Yifan, Yi-An Ma, and Tong Zhang. Reverse diffusion Monte Carlo. In ICLR , 2024a.
- Xunpeng Huang, Difan Zou, Hanze Dong, Yi-An Ma, and Tong Zhang. Faster sampling without isoperimetry via diffusion-based Monte Carlo. In COLT , 2024b.
- Yuling Jiao, Lican Kang, Yanyan Liu, and Youzhou Zhou. Convergence analysis of SchrödingerFöllmer sampler without convexity. arXiv preprint arXiv:2107.04766 , 2021.
- Holden Lee, Andrej Risteski, and Rong Ge. Beyond Log-concavity: Provable Guarantees for Sampling Multi-modal Distributions using Simulated Tempering Langevin Monte Carlo. In NeurIPS , 2018.
- Holden Lee, Jianfeng Lu, and Yixin Tan. Convergence for score-based generative modeling with polynomial complexity. In NeurIPS , 2022.
- Gen Li, Yuting Wei, Yuxin Chen, and Yuejie Chi. Towards non-asymptotic convergence for diffusion-based generative models. In ICLR , 2024.
- Iosif Lytras and Panayotis Mertikopoulos. Tamed Langevin sampling under weaker conditions. In AISTATS , 2025.
- Yi-An Ma, Yuansi Chen, Chi Jin, Nicolas Flammarion, and Michael I Jordan. Sampling can be faster than optimization. Proceedings of the National Academy of Sciences , 2019.
- Albert W Marshall, Ingram Olkin, and Barry C Arnold. Inequalities: theory of majorization and its applications . Springer, 1979.
- Laurence Illing Midgley, Vincent Stimper, Gregor N.C. Simm, Bernard Schölkopf, and José Miguel Hernández-Lobato. Flow annealed importance sampling bootstrap. In ICLR , 2023.
- Dan Mikulincer and Yair Shenfeld. On the Lipschitz Properties of Transportation Along Heat Flows . Springer International Publishing, 2023.
- Dao Nguyen, Xin Dang, and Yixin Chen. Unadjusted langevin algorithm for non-convex weakly smooth potentials. Communications in Mathematics and Statistics , 2021.
- Angus Phillips, Hai-Dang Dau, Michael John Hutchinson, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. Particle denoising diffusion sampler. In ICML , 2024.
- Maxim Raginsky, Alexander Rakhlin, and Matus Telgarsky. Non-convex learning via stochastic gradient Langevin dynamics: a nonasymptotic analysis. In COLT , 2017.
- Hamza Ruzayqat, Alexandros Beskos, Dan Crisan, Ajay Jasra, and Nikolas Kantas. Unbiased estimation using a class of diffusion processes. Journal of Computational Physics , 2023.
- Saeed Saremi, Ji Won Park, and F. Bach. Chain of log-concave Markov chains. In ICLR , 2024.
- Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS , 2019.
- Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR , 2021.

- Francisco Vargas, Andrius Ovsianas, David Fernandes, Mark Girolami, Neil D. Lawrence, and Nikolas Nüsken. Bayesian learning via neural Schrödinger-Föllmer flows. In AABI , 2022.
- Santosh Vempala and Andre Wibisono. Rapid convergence of the unadjusted Langevin algorithm: Isoperimetry suffices. In NeurIPS , 2019.
- Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, and Can Yang. Deep generative learning via sSchrödinger bridge. In ICLR , 2021.
- Yuchen Zhang, Percy Liang, and Moses Charikar. A hitting time analysis of stochastic gradient Langevin dynamics. In COLT , 2017.

## Appendix Table of contents

| A   | Expressions the intermediate scores   | Expressions the intermediate scores                |   15 |
|-----|---------------------------------------|----------------------------------------------------|------|
| B   | Additional discussions                | Additional discussions                             |   17 |
|     | B.1                                   | Föllmer Flows . . . . . . . . . . . . . . . . . .  |   17 |
|     | B.2                                   | Details on the non-smooth example . . . . . .      |   17 |
|     | B.3                                   | Proof of Proposition 2 . . . . . . . . . . . . . . |   18 |
|     | B.4                                   | Discussion on Lytras and Mertikopoulos [2025]      |   19 |
|     | B.5                                   | Discussion on Agapiou et al. [2017] . . . . . .    |   20 |
| C   | Proof of Proposition 4                | Proof of Proposition 4                             |   22 |
| D   | Proof of Proposition 6 and Lemma 8    | Proof of Proposition 6 and Lemma 8                 |   26 |
|     | D.1                                   | Proof of Lemma 8 . . . . . . . . . . . . . . . .   |   27 |
|     | D.2                                   | Proof of Proposition 6 . . . . . . . . . . . . .   |   27 |
| E   | Proof of Lemma 7                      | Proof of Lemma 7                                   |   28 |
| F   | Proof of Theorem 9                    | Proof of Theorem 9                                 |   34 |

## A Expressions the intermediate scores

We next provide derivations of the Monte Carlo estimators of the intermediate scores discussed in section 4.2.

The probability law along the reverse diffusion path has density where π is a standard Gaussian and µ ( · ) ∝ exp( -V ( · )) is the target distribution

Case 1: write the convolution as an integral against the proposal distribution We have

<!-- formula-not-decoded -->

∇ log p V t ( x ) = ∇ p V t ( x ) p V t ( x ) = ∫ ∇ f ( x, y ) dy ∫ f ( x, y ) dy = 1 √ λ t ∫ ∇ log µ Å x -y √ λ t ã f ( x, y ) ∫ f ( x, y ) dy dy . (8) From this, we can either define a Monte Carlo estimator as ∇ log p V t ( x ) = 1 √ λ t E y | x ï ∇ log µ Å x -y √ λ t ãò , y | x ∼∝ f ( x, y ) (9) where f ( x, y ) is a smoothened version of the target distribution distribution. Replacing π with a standard Gaussian. Or else, by unpacking f ( x, y ) and using the proposal as the sampling distribution as the integration variable y appears in it,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

From this, we can either define a Monte Carlo estimator as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Ä ä which is not very useful given that sampling the target is hard in the first place. However, we can use the proposal distribution as the sampling distribution as the integration variable y appears in it as well. To see this, recall that

∇ log p V t ( x ) = 1 √ 1 -λ t E y | x ï ∇ log π Å x -y √ 1 -λ t ãò , y | x ∼∝ f ( x, y ) (14) where f ( x, y ) is a smoothened version of the target distribution distribution. Or else, by upacking f ( x, y ) and using the target as the sampling distribution as the integration variable y appears in it,

<!-- formula-not-decoded -->

This leads to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using a change of variables, we obtain with Y t ∼ N (0 , (1 -λ t ) I d ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B Additional discussions

## B.1 Föllmer Flows

## B.1.1 Comparison with our work

Instead of implementing a reverse diffusion process, Föllmer Flows Huang et al. [2025], Jiao et al. [2021], Ruzayqat et al. [2023] seek to implement the following Schrödinger bridge:

with b ( x, t ) = log [ f ( x + W )] and where f is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with µ the target density; for such a process, we have that X 1 ∼ µ . We refer the reader to the notations of Proposition 14 to observe that the shift b ( x, t ) is given by

<!-- formula-not-decoded -->

with π is the density of the standard Gaussian. Hence, up to the Gaussian term and a time-rescaling, the shift is exactly given by the intermediate scores of the reverse diffusion. However, because of the Gaussian correction, an extra exponential term appears in the associated self-normalized estimator of the shift: in Huang et al. [2025], Jiao et al. [2021], Ruzayqat et al. [2023], the authors implement with z i n -i.i.d. samples from the standard Gaussian distribution. We suspect that because of the extra exponential terms e ∥ x + √ 1 -tz i ∥ 2 / 2 , as discussed below, their estimator provide appealing guarantees only under very stringent assumptions.

<!-- formula-not-decoded -->

## B.1.2 Theoretical guarantees

The theoretical complexity bounds derived in the works of Huang et al. [2025], Jiao et al. [2021], Ruzayqat et al. [2023] quantitatively rely on the assumptions that f = d µ d N (0 ,I d ) is Lipschitz, has Lipschitz gradient and is bounded from below. In particular, assume for instance that µ is a standard Gaussian centered at some point c ∈ R d . In this case, the ratio f reads

<!-- formula-not-decoded -->

glyph[negationslash]

which verifies none of the assumptions above if c = 0 . More broadly, even if f verifies these assumptions, the resulting quantities degrade exponentially with the mismatch between x ↦→ -log( µ )( x ) and x ↦→ ‖ x ‖ 2 / 2 . As noted in Vargas et al. [2022], this limitation is not only a theoretical artifact; in practice, these methods are very unstable and fail to convergence on simple examples.

## B.2 Details on the non-smooth example

Consider the mixture µ = 0 . 5 N (0 , Σ 1 ) + 0 . 5 N (0 , Σ 2 ) with Σ 1 = diag (1 , 0 . 5) and Σ 1 = diag (0 . 5 , 1) . For ( x, y ) ∈ R 2 , it holds that

<!-- formula-not-decoded -->

Hence, when x = y , we have -∇ log( µ )( x, y ) = 3 / 2( x, x ) . Now for y = x + η , the score reads

In particular, -∇ log( µ ) is not Hölder.

<!-- formula-not-decoded -->

## B.3 Proof of Proposition 2

Recall we denote λ I d n := I d n i λ I d n (Σ i ) , λ max := max i λ max (Σ i ) . We write:

with ˜ w i = w i (2 π ) -d/ 2 det(Σ i ) -1 / 2 .

Bound on the Hessian. The Hessian of µ writes:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since

We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denoting γ i ( x ) := ˜ w i ϕ i ( x ) µ ( x ) and s i ( x ) = -Σ -1 i ( x -µ i ) we get

<!-- formula-not-decoded -->

where the first term is a covariance matrix of the vectors s i ( x ) under the weights γ i ( x ) , hence which is a positive semi-definite matrix. Therefore:

<!-- formula-not-decoded -->

Bound on the drift. For the (negative) score of the mixture we have:

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To justify our claims at the end of Section 3.2, we first need some preliminary background on Poincaré constants. Let q ∈ P ac ( R d ) . We say that q satisfies the Poincaré inequality with constant C P ≥ 0 if for all f ∈ W 1 , 2 0 ( q ) (functions with zero-mean with respect to q and whose gradient is squared integrable with respect to q ):

and let C P ( q ) be the best constant in Eq. 21, or + ∞ if it does not exist. We show that in the case where µ = 1 / 2 N ( c, λ max ) + 1 / 2 N ( -c, λ min ) the Poincaré constant is at least 2 e ∥ c ∥ 2 2 λmax / ‖ c ‖ 2 therefore the resulting sampling complexity in Lytras and Mertikopoulos [2025, Theorem 3] is at least O ( poly ( ‖ c ‖ 2 e ∥ c ∥ 2 2 λmax , ϵ -1 )) .

<!-- formula-not-decoded -->

Proposition 10 Let µ = 1 / 2 N ( c, λ max I d ) + 1 / 2 N ( -c, λ min I d ) with λ max ≥ λ min &gt; 0 and ‖ c ‖ &gt; 0 . It holds that

Proof. Denoting u = c/ ‖ c ‖ and defining g : R → R as consider the test function f ( x ) = g ( x ⊤ u ) . Denoting µ 1 = N ( c, λ max I d ) , µ 2 = N ( -c, λ min I d ) , we start to upper-bound the Dirichlet energy I = ∫ ‖∇ f ‖ 2 d µ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now remark that for X ∼ µ i , the random variable X ⊤ u is also a one dimensional gaussian: if X ∼ µ 1 , it holds that X ⊤ u ∼ N ( ‖ c ‖ , λ max ) and if X ∼ µ 2 , it holds that X ⊤ u ∼ N ( -‖ c ‖ , λ min ) . In particular, the Dirichlet energy re-writes as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Φ is the cumulative density function of the standard Gaussian. In particular, I is upperbounded as

Now there remains to lower-bound the variance term. The variance reads

<!-- formula-not-decoded -->

The first term can be lower-bounded as

<!-- formula-not-decoded -->

Conversely, it holds that E µ 2 [ f 2 ( X )] ≥ 1 / 2 and a fortiori Var µ ( f ) ≥ 1 . Hence we recover that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which concludes the proof.

## B.5 Discussion on Agapiou et al. [2017]

In Agapiou et al. [2017], for ν dominated by π , denoting g the unnormalised density ratio

<!-- formula-not-decoded -->

the authors derived error bounds for self-normalised estimators: for any test function ϕ : R d → R , the expectation of ϕ under ν , denoted ν ( ϕ ) , is estimated via

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with ( x i ) n -iid samples drawn from π . In Theorem 2.3, they provide the following quadratic error bound:

with m t ( h ) = π ( | h ( · ) -π ( h ) | t ) and d, e ∈ (1 , + ∞ [ such that 1 /d + 1 /e = 1 . Now recall that, denoting Y t ∼ N (0 , (1 -e -2 t ) I d ) the score along the forward is given by

<!-- formula-not-decoded -->

where ν ( x ) ∝ e -V ( e t ( z -x )) e -∥ x ∥ 2 2(1 -e -2 t ) . In particular, denoting π ( x ) ∝ e -∥ x ∥ 2 2(1 -e -2 t ) , the quadratic error of the self-normalized estimator reads

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We show in the next paragraph that, when integrated against p V t , this upper-bound is vacuous. By Jensen's inequality, it holds that

Hence, the previous right-hand side of is lower-bounded as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now compute explicitly each of the quantities in the right-hand-side above using Proposition 11. First we have

<!-- formula-not-decoded -->

Then, the variance term reads

<!-- formula-not-decoded -->

Finally, the second order moment term reads

<!-- formula-not-decoded -->

where we denoted

<!-- formula-not-decoded -->

Hence, once integrated against p V t , our lower bound reads

<!-- formula-not-decoded -->

Combining Proposition 6 and Lemma 7, we have that the third term is always finite. For the specific case V ( x ) = ∥ x ∥ 2 2 , we show that the second term also remains finite while the first diverges thus making the overall lower bound diverge. First, we recall that for this choice of potential, we have from what we obtain

<!-- formula-not-decoded -->

In particular, we obtain that the second term is bounded and that the first term diverges to infinity making the overall bound of Agapiou et al. [2017] vacuous.

<!-- formula-not-decoded -->

## C Proof of Proposition 4

Before starting the proof, we recall the following identities.

Proposition 11 (Tweedie's formulas) Denoting p V t the density of the forward process X t initialized at µ ∝ e -V , and Y t ∼ N (0 , (1 -e -2 t ) I d ) , it holds for all z ∈ R d that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

that and that

Proof. Recall that p V t is the law of the variable

<!-- formula-not-decoded -->

with X 0 ∼ µ and B s the standard Brownian motion evaluated at time s . Hence, using Bayes formula, we have

<!-- formula-not-decoded -->

After taking the logarithm and differentiating with respect to z , we obtain

<!-- formula-not-decoded -->

To obtain the Hessian, we differentiate the formula above. The Jacobian of the numerator is given by

<!-- formula-not-decoded -->

from which we can deduce

<!-- formula-not-decoded -->

In order to rewrite the quantities above as expectations, we make the change of variable y = z -xe -t so that x = ( z -y ) e t and we obtain for the density p V t :

<!-- formula-not-decoded -->

where Y t ∼ N (0 , I d (1 -e -2 t )) . Conversely, the score rewrites as

<!-- formula-not-decoded -->

and the Hessian rewrites as

<!-- formula-not-decoded -->

For the rest of the proof we shall drop the dependence in z and write ˆ s t,n ( z ) = ˆ N ˆ D with the empirical numerator ˆ N = -∑ n i =1 y i e -V ( e t ( z -y i )) 1 -e -2 t and denominator ˆ D = ∑ n i =1 e -V ( e t ( z -y i )) where we recall y i ∼ N (0 , (1 -e -2 t ) I d ) . In the following proposition, we explicitly compute the variances of ˆ N and ˆ D using the formulas above. In what follows, we shall denote N = E [ N ] and D = E [ D ]

Proposition 12 (Variance of estimators) Let y 1 , . . . , y n i.i.d. distributed as N (0 , (1 -e -2 t ) I d ) . Denote by π a standard normal density, and by ˆ N ( z ) and ˆ D ( z ) the numerator and denominator of the estimator defined in Eq. 5. We have:

Proof. For the numerator, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

hence, since the y i , i = 1 , . . . , n are i.i.d. distributed as Y t ∼ N (0 , (1 -e -2 t ) I d ) ,

Similarly, we have

<!-- formula-not-decoded -->

hence we get using again Proposition 11,

<!-- formula-not-decoded -->

We can now prove Proposition 4.

<!-- formula-not-decoded -->

Proof. Define the event A = ( ˆ D ≥ ηD ) ∩ ( ‖ ˆ N ‖ ≤ κ ‖ N ‖ ) where η ≤ 1 , κ ≥ 1 are positive scalars to be chosen later. We start to decompose the quadratic error as:

<!-- formula-not-decoded -->

We now separately analyze the first and the second term. For the first term, define

The gradient and Hessian of θ are given by

<!-- formula-not-decoded -->

We thus make a Taylor expansion of order 2 of θ ( ˆ N, ˆ D ) around ( N,D ) : there exists (a random) ˆ t ∈ [0 , 1] such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we denoted ˆ N ˆ t = ˆ t ˆ N + (1 -ˆ t ) N and ˆ D ˆ t = ˆ t ˆ D + (1 -ˆ t ) D . The two first terms in the expansion are null and we are left with

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∥ ∥ ∥ ∥ ∥ ˆ N ˆ D -N D ∥ ∥ ∥ ∥ ∥ 2 ≤ 1 η 2 D 2 Ç ‖ ˆ N -N ‖ 2 + 6 κ η ∥ ∥ ∥ ∥ N D ∥ ∥ ∥ ∥ ‖ ˆ N -N ‖| ˆ D -D | +5 ∥ ∥ ∥ ∥ N D ∥ ∥ ∥ ∥ 2 Å κ η ã 2 ( ˆ D -D ) 2 å . Hence, after taking the expectation and applying Cauchy-Schwarz, we obtain for the second term:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

6 κ η 3 D 2 ∥ ∥ ∥ ∥ N D ∥ ∥ ∥ ∥ ‖ ˆ N -N ‖| ˆ D -D | ≤ 6 κp 2 V t Z 2 V e td η 3 ( p V t ) 2 ( Z V ) 2 n ‖∇ log ( p V t ) ‖ Å ∆log ( p 2 V t ) -∆log( π ) 1 -e -2 t + ‖∇ log ( p 2 V t ) ‖ 2 ã 1 / 2 and for the last term:

Hence we finally obtain

<!-- formula-not-decoded -->

Now, recall that X i = || y i || 2 (1 -e -2 t ) -1 are n independent variables such that for all i , X i ∼ χ 2 ( d ) . Using Hölder inequality for some p ≥ 1 , the second term can be upper-bounded as

<!-- formula-not-decoded -->

where we used in the penultimate inequality that the max is smaller than the sum, and in the last one that E [ X p 1 ] = ∏ p -1 i =0 ( d +2 i ) when X 1 ∼ χ 2 ( d ) combined with the fact that the geometric mean is lower than the arithmetic mean.

<!-- formula-not-decoded -->

We now upper bound the probability of the event ¯ A = ( ˆ D &lt; ηD ) ∪ ( ‖ ˆ N ‖ &gt; κ ‖ N ‖ ) . By Chebyshev's inequality, using η &lt; 1 , it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, recalling that || N || = D ||∇ log ( p V t ) || , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combining this with the bound Eq. 27 eventually yields

<!-- formula-not-decoded -->

In the case where ‖∇ log( p t ) ‖ 2 &lt; 1 , we instead pick η = 1 / 2 and κ = 1+ 1 2 ∥∇ log ( p V t ) ∥ . We obtain that

<!-- formula-not-decoded -->

and as previously, for p = log( n ) we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D Proof of Proposition 6 and Lemma 8

<!-- formula-not-decoded -->

Before starting the proofs, we recall the following usefull lemma that bounds the second order moment of dissipative distributions.

Lemma 13 Let V be such that 〈∇ V ( x ) , x 〉 ≥ a ‖ x ‖ 2 -b . Then for µ ∝ e -V , denoting m 2 the second moment of µ , it holds that m 2 ≤ b + d a .

Proof. Define the Laplacian of µ as L ( f ) = ∆ f -〈∇ V, ∇ f 〉 for f sufficiently smooth. By integration by parts, it holds that

In particular, for f ( x ) = ‖ x ‖ 2 , we recover ∫ 〈∇ V ( x ) , ∇ f ( x ) 〉 d µ ( x ) = 2 d . For V dissipative, it implies or equivalently m 2 ≤ ( b +2 d ) /a .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D.1 Proof of Lemma 8

Proof. The first inequality was shown in the Lemma above. For the Fisher information, it holds that

There remains to lower-bound µ (0) . Denote x ∗ a global minimizer of V . By dissipativity, it must hold that ‖ x ∗ ‖ 2 ≤ b/a . Now observe that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Combined with the fact that for all x ∈ R d , it holds that V ( x ) -V ( x ∗ ) ≥ 0 , be recover that V ( x ) -V (0) ≥ -β ‖ x ∗ ‖ 2 ≥ -βb/a . Furthermore, for 0 &lt; δ &lt; 1 , we have

In particular, for δ = 1 / √ 2 , we obtain

Hence, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since β/a ≥ 1 and log(2) / 2 ≤ 1 , we recover that log ( µ (0) -2 /d ) ≤ 4 βb/ad +2 π +log(2 /a ) .

<!-- formula-not-decoded -->

## D.2 Proof of Proposition 6

Recall that the intermediate scores read

<!-- formula-not-decoded -->

with q t,z ( x ) ∝ e -V ( x ) e -∥ e -t x -z ∥ 2 2(1 -e -2 t ) . In particular, if V is dissipative with constant a, b then it holds that

<!-- formula-not-decoded -->

Now recall that e -t ‖ x ‖ 2 -〈 z, x 〉 ≥ -e t ‖ z ‖ 2 / 4 which yields

<!-- formula-not-decoded -->

Hence, using Lemma 13, it holds that

<!-- formula-not-decoded -->

Hence we recover that

Similarly, recall that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E Proof of Lemma 7

Before starting the proof, we recall the result of Mikulincer and Shenfeld [2023].

<!-- formula-not-decoded -->

Proposition 14 Let µ be a β -semi-log-convex probability distribution. Then, denoting p V t the distribution of the forward process in Eq. 1 and π the density of the standard Gaussian, it holds for all z ∈ R d that

<!-- formula-not-decoded -->

Proof. Define the Ornstein-Uhlenbeck semi-group Q t as for all function g integrable w.r.t. the standard Gaussian measure. Taking g as f = d µ d π with π the standard Gaussian, we obtain that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, we remark that ∇ log( Q t ( f )) = ∇ log ( p V t ) - ∇ log( π ) . Now, the quantity ∇ log( Q t ( f )) was studied in Mikulincer and Shenfeld [2023] and they prove in Lemma 5 that for all z

which is equivalent to

<!-- formula-not-decoded -->

Before proving Lemma 7, we introduce this preliminary result on the evolution of Φ t .

Lemma 15 (Evolution of the ratio) Let t &gt; 0 , it holds that

Proof. Recall that the log-density log ( p V t ) evolves as ∂ t log ( p V t ) = ∆log ( p V t ) + ‖∇ log ( p V t ) ‖ 2 -〈∇ log ( p V t ) , ∇ log( π ) 〉 -∆log( π ) . Hence, we deduce that log(Φ t ) evolves as

<!-- formula-not-decoded -->

∂ t log(Φ t ) = ∆log(Φ t ) -〈∇ log(Φ t ) , ∇ log( π ) 〉 + ‖∇ log ( p 2 V t ) ‖ 2 -‖∇ log ( p V t ) ‖ 2 . The difference of quadratic terms can be expressed as which allows to recover

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now provide the proof of Lemma 7.

Proof. Until the rest of the proof, the dependence on z of the integrand shall be implied unless expressed explicitly. We start by differentiating m 0 (Φ t ) with respect to t :

where we used Lemma 15 to compute ∂ t Φ t . Using integration by parts, the first term reads

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

hence the first and the third terms cancel and we recover

Using integration by parts again, we recover

Using Proposition 14, since µ is β -semi-log-convex, the term (∆log( π ) -∆log ( p V t ) ) can be upperbounded uniformly by d ( β -1) e -2 t (1 -e -2 t )( β -1)+1 so we eventually get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence we can use Gronwall's lemma which yields

Denoting by Z V (resp. Z 2 V ) the normalizing constant of e -V (resp. e -2 V ), the term m 0 (Φ 0 ) reads

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, let us compute the integral above. Making the change of variable u = e -2 s ( β -1) we have d u = -2( β -1) e -2 s d s which yields

Hence we recover

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In order to recover a bound on the second moment of Φ t , we need several intermediate results. We first prove that the maximum of the ratio decreases through time.

Lemma 16 (Decrease of the maximum of Φ ) The maximum of the ratio Φ t decreases with t .

Proof. Let z t be a point where Φ t attains its maximum and denote M t = log(Φ t )( z t ) . By the implicit function theorem, z t is differentiable hence we can compute ∂ t M t as

<!-- formula-not-decoded -->

Since z t is a maximum, we have in particular ∆logΦ t ( z t ) ≤ 0 which implies that M t decreases.

We then derive an upper-bound on the maximum of a log-smooth distribution.

Proposition 17 (Upper-bound of the maximum) If µ is β -semi-log-convex then it holds that d µ d z ≤ Ä β 2 π ä d 2 . Proof. Recall that the density of µ can be re-written as where V ∗ the minimum of V attained for some z ∗ . By definition e -( V ( z ) -V ∗ ) ≤ 1 for all z . Furthermore, since V verifies ∇ 2 V glyph[precedesequal] βI d , we are ensured that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies in particular that

Using the previous result, we can derive an upper-bound on the integrated squared gradient at 0 .

Lemma 18 (Upper-bound integrated gradient) Let µ ∝ e -V be a β -semi-log-convex measure. Denoting µ (0) the density of µ with respect to the Lebesgue measure at 0 , it holds that

Proof. Denoting π the density of the standard d dimensional Gaussian, recall that the density p V t evolves as

<!-- formula-not-decoded -->

which can also be re-written as

<!-- formula-not-decoded -->

In particular, for z = 0 this yields

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the uniform upper-bound of Proposition 14, the second term can upper-bounded as

<!-- formula-not-decoded -->

Furthermore, Proposition 14 shows that -∇ 2 log ( p V t ) glyph[precedesequal] β β (1 -e -2 t )+ e -2 t . Thus, using Proposition 17, we recover that log ( p V t ) (0) ≤ d 2 log β β (1 -e -2 t )+ e -2 t -d 2 log(2 π ) . In particular, we recover

We can now bound the first order moment of Φ t

Lemma 19 Let µ be a β -semi-log-convex measure with finite second moment m 2 . It holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Integration by parts of the first term yields

<!-- formula-not-decoded -->

hence the squared gradients terms cancel and we recover

<!-- formula-not-decoded -->

Let us denote by A the first term above. Integration by parts yields:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the upper bound given in Proposition 14, we get 2 ∫ Φ t (∆log( π ) -∆log ( p V t ) ) ‖ z ‖ d z ≤ 2 d ( β -1) e -2 t ( β -1)(1 -e -2 t )+1 m 1 (Φ t ) . Similarly, we re-write the second term as

Let us now handle the term B = -∫ 〈∇ Φ t , z ∥ z ∥ 〉 d z . In one dimension, B = 2Φ t (0) ≤ max(Φ t ) and for d ≥ 2 , we have

<!-- formula-not-decoded -->

Using Lemma 16 and Proposition 17, we have that max(Φ t ) ≤ max(Φ 0 ) = ( Z V ) 2 Z 2 V max( d µ d x ) ≤ ( Z V ) 2 Z 2 V ( β 2 π ) d/ 2 . Hence, if we pick R = ( Z 2 V m 0 (Φ t )Γ( d/ 2) /Z 2 V ) 1 /d β -1 / 2 2 1 / 2 -1 /d , we get as an upper-bound for B :

In particular, we recover that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

hence using Gronwall lemma, we have that

Using Lemma 7, we have that m 0 (Φ s ) ≤ Z 2 V Z 2 V e sd ( β (1 -e -2 s ) + e -2 s ) d hence the first term of the integral is upper-bounded as:

<!-- formula-not-decoded -->

By Cauchy-Schwarz it holds that

<!-- formula-not-decoded -->

The integral term is given by

<!-- formula-not-decoded -->

Similarly,

<!-- formula-not-decoded -->

Hence, we obtain

<!-- formula-not-decoded -->

We can now derive our upper-bound on m 2 (Φ t ) .

Proof. We start by differentiating m 2 (Φ t ) :

<!-- formula-not-decoded -->

The first term ∫ (∆log( π ) -∆log ( p V t ) ) ‖ z ‖ 2 Φ t d z is upper-bounded by ( β -1) e -2 t (1 -e -2 t )( β -1)+1 dm 2 (Φ t ) and for the second term we have

Hence we recover

<!-- formula-not-decoded -->

∂ t m 2 (Φ t ) ≤ ( d +2) Å 1 + 2( β -1) e -2 t (1 -e -2 t )( β -1) + 1 ã m 2 (Φ t )+4 ‖∇ log ( p V t ) (0) ‖ m 1 (Φ t )+2 dm 0 (Φ t ) . We now use the Gronwall lemma to obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence we recover that

<!-- formula-not-decoded -->

m 2 (Φ t ) ≤ e t ( d +2) ( β (1 -e -2 t ) + e -2 t ) d +2 Ç m 2 (Φ 0 ) + 2 C -2 log ( µ (0)) + d log Å β 2 π ã + d å . Using the expression of C , we recover eventually that

## F Proof of Theorem 9

Proof. As Proposition 6 shows, the average error of the estimator can be upper-bounded as

<!-- formula-not-decoded -->

Hence, the average integrated error reads

<!-- formula-not-decoded -->

We thus set n as n = d max( ϵ -2( d +1)+1 , ϵ -5 ) = ϵ -2( d +1)+1 and we eventually get for ϵ ≤ t ≤ log(1 /ϵ ) that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, plugging again the bounds of Lemma 8 in Theorem 3, we recover the desired result.

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (12 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: the abstract and introduction list our contributions, i.e. novel complexity guarantees for a sampling algorithm that simulates a time-reversed diffusion with a specific estimator of the intermediate scores.

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: we discuss the limitations of our theoretical results in the Related Work section.

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

Justification: For each theoretical result, we worked with justified assumptions, making all the dependencies of the problem clear. We provide clear and detailed proofs in the appendix.

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

Justification: Our experiments are simple and illustrative of our theory. We provide details on how to reproduce them.

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

Justification: our experiments are illustrative, we do not use specific datasets.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://nips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so No is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //nips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: our experiments are simple and we detail their setup.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: our experiments illustrate a theoretical result on the convergence rate of an algorithm. The goal is not investigate the stochasticity of results across different runs.

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

Justification: Our experiments are simple. We detail how long they take to run on a local computer.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: In our opinion, this paper does not address societal impact directly, and considers the generic problem of sampling from a distribution.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: In our opinion the paper does not have direct positive or negative social impact.

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

Justification: Our paper does not present such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: we coded our our own experiments that are illustrative of our theoretical result.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/ datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: we document the setup of our experiments.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: our experiments do not involve crowdsourcing.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: our study does not involve risk for participants.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLMs are not a part of this research project.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/ LLM ) for what should or should not be described.