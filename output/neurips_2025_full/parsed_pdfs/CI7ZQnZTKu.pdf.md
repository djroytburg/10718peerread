## Assessing the quality of denoising diffusion models in Wasserstein distance: noisy score and optimal bounds

## Vahan Arsenyan ∗ Elen Vardanyan ∗ Arnak S. Dalalyan

CREST, ENSAE, Institut Polytechnique de Paris 5 avenue Henry Le Chatelier 91764 Palaiseau, France

## Abstract

Generative modeling aims to produce new random examples from an unknown target distribution, given access to a finite collection of examples. Among the leading approaches, denoising diffusion probabilistic models (DDPMs) construct such examples by mapping a Brownian motion via a diffusion process driven by an estimated score function. In this work, we first provide empirical evidence that DDPMs are robust to constant-variance noise in the score evaluations. We then establish finite-sample guarantees in Wasserstein-2 distance that exhibit two key features: (i) they characterize and quantify the robustness of DDPMs to noisy score estimates, and (ii) they achieve faster convergence rates than previously known results. Furthermore, we observe that the obtained rates match those known in the Gaussian case, implying their optimality.

## 1 Introduction

We study the problem of generative modeling, which aims to construct a mechanism capable of producing synthetic samples that mimic a target distribution P ∗ , given access to independent observations from P ∗ . This fundamental task lies at the core of numerous applications, including image, text, music, and molecule generation. Among the recent advances in this domain, Denoising Diffusion Probabilistic Models (DDPMs), introduced in [HJA20], have emerged as a remarkably effective class of generative models; see, e.g. , [CMFW24, YZS + 24, TZ25] for comprehensive overviews. In this work, we contribute to the growing theoretical understanding of DDPMs by analyzing several of their key properties and performance guarantees.

The central idea underlying DDPMs is to construct a transport map that transforms a simple source of randomness into a sample from the target distribution P ∗ . More precisely, for any distribution P ∗ , there exists a map defined via a stochastic differential equation (SDE) that takes as input a standard Gaussian vector ξ 0 and a standard Brownian motion W , and outputs a vector with distribution P ∗ . Importantly, only the drift term of the SDE depends on P ∗ , and this dependence occurs through the score function, that is, the gradient of the log-density of a Gaussian-smoothed version of P ∗ . This formulation reduces the generative modeling task to that of score estimation: one can estimate the score function from data and substitute this estimate into the SDE to approximately sample from P ∗ .

For many commonly used datasets, such as CIFAR-10 and CelebA-HQ considered in Section 6, accurate estimators of the score function are available. Generating a synthetic sample reduces to drawing a Gaussian vector together with the increments of a Brownian motion, and simulating the SDE defined by the pretrained score. This procedure requires multiple evaluations of the score estimator. The first question we address in this paper is: what happens if each evaluation returns a value corrupted by additive centered noise? Such a scenario may arise when the pretrained model is hosted on a remote server and communication introduces random perturbations, or when the score values are compressed using stochastic rounding. Anticipating our main findings, we emphasize that, perhaps counterintuitively, we observe that adding even a constant level of noise to each score evaluation has only a limited effect on the quality of the generated samples; see Figure 1 for an illustration.

∗ Equal Contribution

Figure 1: Generated images obtained by DDPM with a constant-level noise added to the estimated score. Left: CelebA-HQ. Right: CIFAR10. The result is visually as good as the noiseless one.

<!-- image -->

<!-- image -->

The second question we investigate concerns the accuracy of DDPMs when performance is measured in terms of the Wasserstein distance. A natural criterion in this setting is the number of score function queries K required to achieve a prescribed level of accuracy ε . For the Gaussian target distribution, elementary computations show that K = O ( √ D/ε ) , where D denotes the ambient dimension. Surprisingly, however, it remains unclear whether DDPMs maintain this level of accuracy for broader classes of distributions beyond the Gaussian case.

Contributions. The main contributions of this work can be summarized as follows:

- We provide empirical evidence, based on experiments with the CIFAR-10 and CelebA-HQ datasets, that DDPMs are remarkably robust to noise in the evaluation of the score function.
- We derive non-asymptotic upper bounds on the Wasserstein-2 distance between the target distribution and the distribution induced by the DDPM with noisy score evaluations, thus offering a theoretical explanation for the observed robustness. √
- Our bounds match-up to a multiplicative constant-the rate D/ε of the case of a Gaussian target. Moreover, our results extend to a significantly broader class of distributions, including compactly supported semi-log-concave measures supported on low-dimensional subspaces.

Related work [KFL22] highlighted the connection between DDPMs and the Wasserstein distance. The first quantitative bounds-polynomial in the dimension and valid for a broad class of P ∗ -were established in [CCL + 23], covering several metrics. Unlike their result in total variation (TV) distance, their bound in Wasserstein distance has the poor scaling D 5 /ε 12 . Subsequent work significantly improved this rate: [CLL23] achieved D 4 /ε 2 under minimal assumptions, while [BZL + 23, GNZ25, YY25, SOB + 25] reduced it further to D/ε 2 , assuming stronger conditions on P ∗ . [SO25] proved the √ D/ε 2 rate and our paper closes the loop by proving that the optimal rate √ D/ε is achieved by the standard DDPM procedure. A related result by [GZ24] establishes similar bounds for the probability flow ODE, but under more restrictive assumptions, such as strong log-concavity of P ∗ .

Over the past three years, substantial progress has also been made in establishing guarantees for DDPMs in total variation and Kullback-Leibler divergence under weak assumptions on P ∗ [CDS25, LJLS25, LY25, BBDD24, LHE + 24], including acceleration techniques such as parallel sampling, randomized midpoint, and Runge-Kutta methods [CRYR24, GCC24, WCW24]. In parallel, a growing body of work investigates the statistical optimality of score-based models [OAS23, WWY24, HST25], as well as their ability to adapt to low-dimensional structure [Bor22, TY24, LY24, HWC24, ADR24, PAD24]. Analogous results for flow matching have been established in [KT25].

Notation For D ∈ N , I D is the D × D identity matrix. We use notation A ≺ B , A ≼ B , A ≻ B , A ≽ B to design that the matrix A -B is, respectively, negative definite, negative semi-definite, positive definite and positive semi-definite. We denote by N D ( µ , Σ ) the D -dimensional Gaussian distribution with mean µ and covariance matrix Σ . Let γ D be the density function of N D (0 , I D ) . The norm of a vector is always understood as the Euclidean norm, whereas the norm of a matrix is the operator norm (the largest singular value). The independence of random vectors X and Y is denoted by X ⊥ ⊥ Y . The Wassersteinq distance between two distributions P and Q is defined by

<!-- formula-not-decoded -->

where q ⩾ 1 and Γ( P, Q ) is the set of all joint distributions with marginals P and Q . For any function g : [0 , T ] × R D → R , we will write ∇ g and ∇ 2 g for the gradient and the Hessian of g with respect to its second variable. If g : [0 , T ] × R D → R D , we write D g for the differential of g with respect to its second variable. For each random vector X , we write ∥ X ∥ L 2 = ( E [ ∥ X ∥ 2 2 ]) 1 / 2 .

## 2 Problem statement and conditions

The goal of this section is to set the framework of denoising diffusion probabilistic models with randomized score estimators and to state the conditions imposed on the unknown target distribution.

The setting of randomized score estimators Our setting is a bit more general than those previously studied in the literature. For an unknown distribution P ∗ on R D , and for t &gt; 0 , we define P ∗ t as the distribution of α t X + β t ξ , where ( X , ξ ) ∼ P ∗ ⊗ γ D , α t = e -t , and β t = √ 1 -α 2 t . The set ( P ∗ t ) t ⩾ 0 can be seen as a curve in the space of probability measures interpolating between P ∗ and γ D , since P ∗ 0 = P ∗ and P ∗ ∞ = γ D . For t &gt; 0 , P ∗ t is absolutely continuous with respect to the Lebesgue measure λ D on R D with an infinitely differentiable density. Therefore, we can define the score function s by

<!-- formula-not-decoded -->

Since P ∗ t is unknown, we cannot access s ( t, x ) . Instead, we have access to randomized and noisy evaluations of this function: for each query ( t, x ) ∈ [0 , ∞ ) × R D , we can observe a random vector ˜ s ( t, x ) such that ∥ ˜ s ( t, x ) -s ( t, x ) ∥ L 2 is small. Our goal is to combine independent Gaussian random vectors and queries to the approximate score ˜ s to build a random vector Z in R D having a distribution P Z close to P ∗ . To this end, we focus on the DDPM algorithm presented in Algorithm 1.

Algorithm 1 Generation of Z by the denoising diffusion probabilistic model

Require:

Sequence ( t 1 , . . . , t K +1 ) for some integer K ⩾ 1

Ensure:

- 1: Set t 0 = 0 , T = t K +1 , and Z 0 ∼ γ D

Vector Z = Z K +1

- 2: for k = 0 to K do
- 3: Set h k = t k +1 -t k
- 4: Generate ξ k +1 ∼ γ D , independent of all previous randomness
- 5: Query ˜ s at ( t k , Z k )
- 6: Set Z k +1 = (1 + h k ) Z k +2 h k ˜ s ( T -t k , Z k ) + √ 2 h k ξ k +1
- 7: end for
- 8: Output Z K +1

We postpone the discussion of the origin of this algorithm to Section 3. The main difference between our setting and prior work lies in the randomness of ˜ s , which goes beyond the randomness of the training sample. Let us provide concrete examples to illustrate our setting.

Example 1 (Noisy score estimator). Assume that an estimator ̂ s is available. Due to issues such as communication constraints or privacy concerns, we do not observe ̂ s ( t, x ) directly, but rather a noisy version ˜ s ( t, x ) = ̂ s ( t, x ) + ζ , where ζ is random, typically with zero mean and bounded variance.

Example 2 (Compressed score estimator). Assume again that an estimator ̂ s is available, but only one of its coordinates can be queried at a time. At each iteration, we randomly choose i ∈ { 1 , . . . , D } uniformly and set ˜ s ( t, x ) = D × ( ̂ s ( t, x ) T e i ) e i , where e i is the i -th canonical basis vector.

Example 3 (Randomized network weights). The conventional approach fits the weights θ of a neural net ϕ ( t, x ; θ ) to the unknown score s ( t, x ) by minimizing the (estimated) prediction error:

<!-- formula-not-decoded -->

One can instead minimize an estimator of the integrated error under a Gaussian prior by solving

<!-- formula-not-decoded -->

where σ &gt; 0 is a hyperparameter. This may lead to a more robust score estimator. In this setting, the randomized estimator of the score at each query point ( t, x ) is ϕ ( t, x , ̂ µ + σ ζ ) , with ζ ∼ γ p generated independently by the user.

Conditions on the target distribution The guarantees on the precision of the DDPM that we will state in the next section depend on the properties of the target P ∗ . We will express these properties in terms of a function φ .

Assumption 1. For a function φ : R &gt; 0 → R &gt; 0 , we say that P ∗ or X satisfies Assumption 1 with function φ if, for ( X , ξ ) ∼ P ∗ ⊗ γ D , it holds that Var ( X | X + σ ξ = y ) ≼ φ ( σ ) I D for all σ &gt; 0 .

Many distributions satisfy this assumption (see Appendix A for the proofs):

- (a) If X has compact support K with diam( K ) = 2 D X , Assumption 1 holds with φ ( σ ) ≡ D 2 X ;
- (b) Any m -strongly log-concave distribution P ∗ satisfies Assumption 1 with φ ( σ ) = σ 2 1+ mσ 2 ;
- (c) If X is semi-log-concave with constant 2 M ⩾ 0 and has compact support of diameter 2 D X , then X satisfies Assumption 1 with φ ( σ ) = D 2 X ∧ σ 2 (1 -Mσ 2 ) + ;
- (d) If X satisfies Assumption 1 with some function φ , U is a D × D orthonormal matrix and b ∈ R D , then U X + b satisfies Assumption 1 with the same φ ;
- (e) If X is obtained by concatenating two independent vectors X 1 and X 2 satisfying Assumption 1 with the same function φ , then X satisfies Assumption 1 with φ .
- (f) If ( W , ζ ) ∼ P 0 ⊗ γ D such that W satisfies Assumption 1 with the function φ 0 , then, X = W + τ ζ satisfies Assumption 1 with the function φ τ ( σ ) = τ 2 σ 2 τ 2 + σ 2 + σ 4 φ 0 ( √ τ 2 + σ 2 ) ( τ 2 + σ 2 ) 2 .
- (g) If W is supported by a compact set of diameter 2 D and ζ ⊥ ⊥ W is m -strongly log-concave with an M -Lipschitz score function, then X = W + ζ satisfies Assumption 1 with φ ( σ ) = σ 2 1+ mσ 2 + ( M D σ 2 ) 2 (1+ Mσ 2 ) 2 .

The main purpose of Assumption 1 is to ensure that the drift coefficient of the backward diffusion process is strongly convex when the noise level is large and semi-log-concave for all noise levels. Moreover, the drift coefficient is always gradient-Lipschitz, with a Lipschitz constant depending on the noise level. These properties are summarized in the following result 3 .

Proposition 1. Let X and ξ be random vectors in R D drawn from P ∗ ⊗ γ D . For any α, β &gt; 0 , the density π Y of Y = α X + β ξ is twice continuously differentiable and satisfies

<!-- formula-not-decoded -->

Thus, Assumption 1 is equivalent to ∇ 2 log π Y ( y ) ≼ ( α 2 φ ( β/α ) -β 2 ) β 4 I D , for all y ∈ R D , α, β &gt; 0 .

The last inequality above implies that if φ ( β/α ) ⩽ ( β/α ) 2 , the distribution of Y = α X + β ξ is log-concave, and it is strongly log-concave if the inequality is strict.

Conditions on the estimated score As mentioned in Section 2, we consider randomized estimators ˜ s of the true score function s . The mean squared error of such an estimator can be decomposed into a bias and a variance term:

<!-- formula-not-decoded -->

In what follows, we analyze separately the impact of the bias and the variance on the overall error. As we will see, the variance term has a much weaker influence on the final accuracy than the bias term. To reflect this difference, we introduce the following assumption.

Assumption 2. There are constants ε b score and ε v score such that for all t ∈ { t k : k ⩽ K } of Algorithm 1,

<!-- formula-not-decoded -->

Assumption 2 imposes uniformity over all x ∈ R D and t ∈ t k : k ⩽ K and, therefore, is a stronger condition than the one used in previous work [CLL23]. The latter considers L 2 -norm with respect to P ∗ t , rather than a supremum, and involves a weighted average over t . While it may be possible to relax the requirement involving the maximum over the time grid, the uniformity with respect to x appears to be more difficult to replace by the L 2 -norm wrt P ∗ t . It is important to note, however, that for our proof needs only an L 2 bound with respect to the distribution of the DDPM output at time t .

2 We recall that X is semi-log-concave [Cla83] with constant M ∈ R if X has a density π X wrt the Lebesgue measure and -log π X ( x ) + M 2 ∥ x ∥ 2 is convex; see [VCK25] for an application in sampling.

3 The formula relating the Hessian of the log-density to the conditional variance, stated in Proposition 1 is often referred to as the second-order Tweedie formula.

## 3 Score-Based Generative Modeling: preliminary considerations

The starting point of a DDPM is the forward process given as a solution to a stochastic differential equation (SDE). The simplest and the most widespread choice is the Ornstein-Uhlenbeck process

<!-- formula-not-decoded -->

where ( B t ) t ⩾ 0 is a standard Brownian motion in R D . The Ornstein-Uhlenbeck process is a timehomogeneous Markov process which is also a Gaussian process, with stationary distribution equal to the standard Gaussian distribution γ D on R D . The forward process has the interpretation of transforming samples from the data generating distribution P ∗ into the latent distribution. From the classical theory of Markov diffusions, it is known that P ∗ t := law ( X t ) converges to γ D exponentially fast in various divergences and metrics such as the 2-Wasserstein metric W 2 : W 2 ( P ∗ t ; γ D ) ⩽ e -t W 2 ( P 0 ; γ D ) , see for instance [Vil08].

## 3.1 Reverse Process: continuous-time and time-discretized versions

If we reverse the forward process in time, we obtain a process that transforms the latent distribution into the target distribution P ∗ , which is the aim of generative modeling. Fix some large time horizon T &gt; 0 and set Y t := X T -t , then law ( Y 0 ) = law ( X T ) is close to the Gaussian distribution γ D . Notably, the dynamics of the reverse process can also be described by a stochastic differential equation, as stated in the next result.

Theorem 1 ([And82]) . If ( X t ) t ⩾ 0 is a solution to (2) and Y t = X T -t , then there exists a Brownian Motion ( ˜ B t ) t ⩾ 0 ⊥ ⊥ Y 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that π ( t, x ) in this theorem coincides with the one defined in (1) and ∇ log π ( T -t, Y t ) is the score function s evaluated at scale T -t and state Y t .

The forward process transforms a data point X 0 drawn from P ∗ into a point which is very close to being drawn from the latent distribution. The reverse process aims to transform a point Y 0 drawn from the latent distribution into a point drawn from P ∗ . To this end, we replace the unknown score function by its estimate ˜ s based on a training sample X 1 , . . . , X n ∼ P ∗ . The resulting process is defined as the solution to the SDE

<!-- formula-not-decoded -->

Both ˜ Y and Y are processes on the space C ([0 , T ] , R D ) , differing in their initial conditions and drift terms. We wish to assess the distance between the distributions of their states at time T .

To efficiently sample the final state of the reverse process, we have to discretize SDE (4). To this end, we introduce a sequence ( h k ) k ∈ N of positive numbers and set 4 t k = h 0 + . . . + h k -1 . We then define

<!-- formula-not-decoded -->

where ( ξ k ) k ∈ N is a sequence of independent standard Gaussian random variables. The rationale behind this definition is that Z k has approximately the same law as ˜ Y t k , for every k .

Definition 1. The denoising diffusion probabilistic model is the distribution P DDPM of the random vector Z K +1 defined by (5). It requires the choice of K ∈ N , the sequence ( t 1 , . . . , t K +1 ) and the score estimators ( ˜ s ( T -t k , · ) ) k =0 ,...,K .

In this paper, we are interested in quantifying the accuracy of the denoising diffusion generative model when the error is measured in terms of the Wasserstein distance, that is to upper bound W 2 ( P ∗ , P DDPM ) . In the rest of this section, we motivate the choice of the Wasserstein distance and discuss the challenges related to it in the framework of denoising diffusions.

4 By convention, t 0 = 0 .

## 3.2 Relevance of the Wasserstein distance

Recent work on assessing denoising diffusion models mainly focuses on accuracy measured by the total variation distance and the Kullback-Leibler divergence. However, we believe that for statistical purposes, measuring the quality of a generative model in the Wasserstein distance is highly appealing.

To justify this point of view, remind that the closeness of two distributions in TV-distance or KLdivergence does not guarantee the closeness of their means or their covariance matrices. In sharp contrast, the Wasserstein-2 distance offers such a guarantee, since it holds that

<!-- formula-not-decoded -->

for any matrix A satisfying 0 ≼ A ≼ I . The fact that the TV-distance and the KL-divergence are not suitable for controlling the moments of distributions can be demonstrated by the following example. Let P be the exponential distribution with parameter 1 and, for every n ∈ N , set P n = (1 -δ n ) P + δ n Q n , where δ n = 1 / √ n and Q n is the uniform distribution on [ n, n +2] . One can easily check that P n is very close to P both in the TV-distance and in the KL-divergence:

<!-- formula-not-decoded -->

Therefore, one could expect that P n is an excellent generative model for the target P . However, the generated examples will have a mean and variance that explode as n →∞ , and will be infinitely far away from the mean and the variance of the target, since E P n [ X ] = 1 + nδ n ⩾ n 1 / 2 and E P n [ X 2 ] ⩾ 2(1 -δ n ) + δ n n 2 ⩾ n 3 / 2 .

## 3.3 Challenges inherent to Wasserstein distance

When the distance W q is employed to assess the quality of a DDPM, a mathematical challenge arises in quantifying the error due to the absence of the data-processing inequality for W q -distance. Let us clarify this point. Consider a forward mechanism M → that transforms the target P ∗ into a distribution P ∗ 1 which is close to an easy-to-sample-from latent distribution Q 0 : P ∗ 1 := M → ( P ∗ ) ≈ Q 0 . Furthermore, assume we have knowledge of the 'inverse' forward mechanism, termed backward mechanism, which maps P ∗ 1 back to P ∗ : M ← ( P ∗ 1 ) = P ∗ . The forward-backward methods of generative modeling then define the generative model as Q 1 = ¯ M ← ( Q 0 ) , where ¯ M ← represents a suitably regularized estimator of M ← . In DDPM, M ← and ¯ M ← are specified by Markov kernels.

In this context, denoting d F as the F -divergence for some F , the following relationship holds:

<!-- formula-not-decoded -->

where the final equality derives from the data-processing inequality. Thus, the error of the generative distribution is dominated by how well the forward mechanism approximates the latent distribution, provided that the error of M ← approximation is suitably small. These arguments were central in prior work 5 establishing bounds on the error of denoising diffusion models measured in TV-distance and KL-divergence. However, this approach breaks down for the Wasserstein distance W q , for which no suitable equivalent of the data processing inequality exists.

In the case of denoising diffusion models, the qualitative difference between the Wasserstein distance and F -divergences (such as TV-distances and KL-divergence) can be formally demonstrated even when the backward kernel is known. This is illustrated in the following lemma.

Lemma 1. For any T &gt; 0 , let Q T, s 1 be the distribution of the backward process (4) at time T with ˜ s replaced by the true score s . Let N be the set of all the Gaussian distributions. It then holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This lemma reveals that when assessing accuracy through the rate of improvement in Wasserstein distance, the choice of parameter T must be carefully tailored to the target distribution P ∗ . This might be less important in the case of the TV-distance and the KL-divergence.

5 See [CCL + 23, BBDD24, HWC24, CDS25] and the references therein

## 4 Main results: bounds on the error in various settings

In this section, we upper bound the Wasserstein-2 distance between DDPM (see Algorithm 1) and the target P ∗ . Similar to [CLL23, BBDD24], we employ a discretization scheme composed of two regimes: an arithmetic grid in the first half and a geometric grid in the second half; see Algorithm 2.

## Algorithm 2 Definition of the discretization time steps

Require:

δ, a, T 1 &gt; 0 , and K 0 ∈ N &gt; 1

Ensure:

Sequence t 0 &lt; t 1 &lt; . . . &lt; t K +1

- 2: for k = 1 to K 0 do
- 1: Set t 0 = 0 , K = 2 K 0 , t K +1 = T 1 + 1 2 log(6 a )
- 3: Set t = ( T /K ) k

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 5: end for
- 6: Output ( t 0 , . . . , t K +1 )

## 4.1 Strongly log-concave distributions convolved with a distribution with compact support

In this section, we consider the case of a distribution P ∗ satisfying Assumption 1 with a function φ that has the following form: for some constants m,M,b ⩾ 0 ,

<!-- formula-not-decoded -->

If P ∗ is m -strongly log-concave, as discussed in Section 2, then (6) holds with b = 0 and any M &gt; 0 . Another class of distributions satisfying (6) consists of convolutions P ∗ = P slc ⋆ P cmpct , where P slc is m -strongly log-concave with an M -Lipschitz score, and P cmpct is supported on a compact set of diameter 2 D , for some M ⩾ m&gt; 0 and D ⩾ 0 . In this case, (6) holds with b = D 2 .

Finally, there are distributions satisfying Assumption 1 with φ given by (6) that are not absolutely continuous with respect to the Lebesgue measure on R D . For example, if P ∗ is supported on a linear subspace S of R D , and its restriction to S , viewed as a distribution on R d for some d ∈ 1 , . . . , D , satisfies Assumption 1 with φ given by (6), then P ∗ also satisfies the assumption with the same φ . This is a consequence of properties (d) and (e) presented in Section 2.

Theorem 2. Let the target distribution P ∗ satisfy E [ ∥ X ∥ 2 2 ] ⩽ D and Assumption 1 with function φ given by (6) for some m,M,b ⩾ 0 . Let us choose T 1 &gt; 0 ,

<!-- formula-not-decoded -->

and define the sequence ( t k ) 0 ⩽ k ⩽ K +1 by Algorithm 2. Let ˜ s be a randomized estimator of the score satisfying Assumption 2. Then, the distribution P DDPM of the output of Algorithm 1 based on 2 K 0 queries to the score estimator ˜ s satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

There are several notable features in the upper bound stated in Theorem 2, when we compare it to the previously known results.

Remark 1 (Optimality) . The dependence of the discretization error (the second term in (7)) on the step size h max is linear, whereas it was of order h 1 / 12 max in [CCL + 23, Cor. 6], h 1 / 4 max in [CLL23, Cor. 2.4], and h 1 / 2 max in [BZL + 23, Remark 12], [SOB + 25, Cor. 4.3], [SO25, GNZ25, YY25]. Moreover, [GNZ25] establishes that the lower bound on the Wasserstein-2 error, achieved by the Gaussian distribution, scales as √ Dh max , thereby implying the optimality of the bound in Theorem 2.

Remark 2 (Conditions) . Assumptions on P ∗ in Theorem 2 are less stringent than those in earlier works [BZL + 23, YY25, GNZ25]. In particular, for m -strongly log-concave P ∗ , we do not assume that the Hessian of the log-density is bounded from below. Furthermore, Theorem 2 covers the class

of distributions obtained as convolutions of a compactly supported distribution and a Gaussian, a framework not addressed in previous studies achieving a discretization error of h 1 / 2 max . However, our conditions may be regarded as stronger than those of [CLL23, Cor. 2.4] providing the discretization error of order h 1 / 4 max . These stronger assumptions are typically necessary for attaining faster rates of convergence. In conclusion, our conditions are weaker than those previously associated with the h 1 / 2 max rate, while enabling the faster convergence rate of h max .

Remark 3 (Impact of noise) . All previously known bounds are proportional to ∥ ( ˜ s -s )( τ, X ) ∥ L 2 , where the proportionality factor is often logarithmic in the number of queries, and the L 2 -norm can take different forms-the weakest being the case where τ ∼ Unif ([0 , T ]) and the law of X given τ = t is P ∗ t . If ˜ s ( t, x ) = ̂ s ( t, x ) + ζ , with ∥ ζ ∥ 2 L 2 = σ 2 ζ D as in Example 1 of Section 2, then ∥ ˜ s -s ∥ 2 L 2 ⩾ σ 2 ζ D . Thus, all known bounds include a term of constant order, independent of the number of queries. In contrast, the corresponding term in the bound of Theorem 2 is O ( √ Dh max ε v score ) , which scales as σ ζ √ DT 1 /K and thus vanishes as K , the number of queries, grows large.

Remark 4 (Informal statement) . To facilitate comparison with existing results, let us consider the strongly log-concave case b = 0 and denote by L := a the surrogate of the Lipschitz norm of the score of P ∗ . For T 1 = log( K 0 ) , our result implies that, after K queries to the score estimator,

In particular, W 2 ( P ∗ , P DDPM ) ≲ √ LDε b score , provided that the number of queries satisfies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As mentioned in Remark 3, this improves on [BZL + 23, YY25, GNZ25, SO25], which require K ≳ (log L ) / ( ε b score ) 2 and ε v score ≲ ε b score to achieve W 2 ( P ∗ , P DDPM ) ≲ √ LDε b score .

## 4.2 Semi log-concave distributions with compact support

In this section, we consider the case of a distribution P ∗ satisfying Assumption 1 with a function φ that has the following form: for some constants b, M ⩾ 0 ,

<!-- formula-not-decoded -->

The typical example of P ∗ satisfying this assumption is a distribution on a compact set K included in a linear subspace of R D , if in addition the log-density wrt to the Lebesgue measure on the subspace has a Hessian ≼ M I . It then follows from claims (c), (d), and (e) of Section 2 that P ∗ satisfies Assumption 1 with φ as in (8) with b = D 2 X .

Theorem 3. Let the target distribution P ∗ satisfy E [ ∥ X ∥ 2 2 ] ⩽ D and Assumption 1 with function φ given by (8) for some b, M ⩾ 0 . Let us choose T 1 &gt; 0 ,

<!-- formula-not-decoded -->

and define the sequence ( t k ) 0 ⩽ k ⩽ K +1 by Algorithm 2. Let ˜ s be a randomized estimator of the score satisfying Assumption 2. Then, the distribution P DDPM of the output of Algorithm 1 based on 2 K 0 queries to the score estimator ˜ s satisfies

<!-- formula-not-decoded -->

Since the conclusions of this theorem closely mirror those of Theorem 2, the remarks provided after the latter apply here as well and will not be repeated. We merely emphasize two points. First, P ∗ is not assumed to have a density wrt the Lebesgue measure on R D . Second, the number K of queries to the score estimator required to achieve W 2 error ε scales as 1 /ε , up to a factor that grows at most logarithmically in 1 /ε . The exponential terms in (7) and (9) depend on the parameters of the target distribution. The independent work [SO25] employs a different proof technique yet exhibits a similar exponential dependence, suggesting that this behavior is intrinsic to bounding the Wasserstein distance in DDPMs. For a log-concave distribution supported on a compact domain, we have ( M,b ) = (0 , D 2 X ) , so the exponential factor in the bound (9) becomes a universal constant. This complements the result obtained in the strongly log-concave setting from Theorem 2.

## 5 Relation to prior work: extended discussion

Given the wealth of work on Langevin algorithms and score-based generative models, it would be infeasible to provide an exhaustive account of existing results. Instead, this section offers a selective overview of prior work, to situate our contributions within the broader landscape.

Theoretical guarantees for DDPMs have been inspired by techniques from the sampling literature, particularly those used for Langevin Monte Carlo and its variants; see the overview in [Che24]. Prior work can be grouped into three categories based on the underlying proof strategies.

The first category, represented by [CCL + 23, CLL23, BBDD24, CDS25, LY25, LJLS25, LHE + 24], includes works that build on the approach initiated in [DT12, Dal17b], combining the Pinsker inequality with the Girsanov formula to derive bounds in TV. Its key strengths are:

- it requires only a bound on the mean integrated squared error (MISE) of the score estimator-one of the weakest conditions in this framework;
- it relies on mild assumptions on the data-generating distribution P ∗ .

As noted in [CCL + 23, CLL23], TV-distance bounds can be converted into Wasserstein bounds under additional assumptions, such as compact support or light-tailed P ∗ . If the support lies in a ball, one can project the generated sample onto this ball and use that W 2 2 is bounded by the radius of the ball times the TV distance. By the data-processing inequality, this projection does not increase the TV-error.

However, this versatility comes at a price. Let K TV ( ˜ ε ) be the number of steps required to achieve an error smaller than ˜ ε in TV-distance. Then, to achieve W 2 -error ε , one needs a TV-error ˜ ε = ε 2 /R 2 , leading to a number of steps at least K TV ( ε 2 /R 2 ) . As a result, the rates derived from this strategy are suboptimal: O ( D 4 /ε 2 ) in [CLL23], O ( D/ε 4 ) in [BBDD24, CDS25], and O ( D 3 /ε 2 ) in [LHE + 24], ignoring log-factors. Another limitation of this approach is that the resulting upper bound on the W 2 distance scales as the square root of the error of estimation of the score. Hence, to guarantee an error ε in W 2 , one needs the score estimation error ε score of order O ( ε 2 ) . Our results, as well as those of the third category below, typically require the weaker condition ε score = O ( ε ) .

The second category comprises results that exploit the interpretation of Langevin dynamics as a gradient flow in the space of probability measures . This perspective was initiated in [Wib18, Ber18] and further developed in [CB18, DMM19, VW19]. Interestingly, the first polynomial-in-dimension guarantees for DDPM fall within this framework, as shown in [LLT22, YW22]. These works evaluate the error in terms of f -divergences such as total variation, KL, or χ 2 divergence. However, when translated to bounds in the W 2 distance, they suffer from the same limitations as the TVbased approaches discussed above. Moreover, this line of work typically relies on strong structural assumptions on the target distribution P ∗ , notably the satisfaction of a log-Sobolev inequality. Another limitation, shared with our own analysis, is that the score estimation error is measured in the uniform norm. We believe, however, that this requirement could be relaxed, both in the gradient-flow framework and in the recursive method developed in our work.

The third category comprises works using the recursive approach to bound the error of iterative algorithms such as LMC or DDPM. This method, widely used in optimization theory, was shown to yield strong guarantees for sampling in [Dal17a, DM17, DM19, DK19]. For DDPM, it underlies the analyses in [BZL + 23, GNZ25, SOB + 25, YY25], which establish a W 2 -error rate of order D/ε 2 -an improvement over the bounds derived or derivable from the first two categories. However, despite having all the necessary ingredients, these works do not reach the faster rate √ D/ε . This is somewhat surprising, especially since their assumptions on P ∗ are often quite strong, such as strong log-concavity. We believe this gap arises from not fully exploiting the smoothness of the score of the distribution obtained from P ∗ by convolving with a Gaussian. Technically, their recursive bounds relate the error at iteration k to that at iteration k -1 via triangle inequalities, which can be loose when the two terms involved are weakly correlated. As we show, applying the recursive approach to the squared Wasserstein distance yields significantly tighter control and leads to optimal rates. We believe that this improvement can be further exploited to get even faster rates using the randomized midpoint discretization [SL19, HBE20, YKD24, YY25] or to get a faster algorithm exploiting parallelization [CRYR24, ACV24, GCC24, YD25].

## 6 Numerical experiments

We supplement our theoretical results with a small-scale empirical study on CIFAR-10 [KH09], CelebA-HQ [KALL18], and LSUN-Church [YZS + 15], evaluating the robustness of DDPMs to noise in the estimated score 6 .

Setup. We use pretrained DDPM models from the publicly available checkpoints google/ddpm-cifar10-32 , google/ddpm-celebahq-256 , and google/ddpm-church-256 , all licensed under Apache license 2.0 and hosted on HuggingFace. For each model, we follow the standard DDPM sampling procedure, and then repeat the generation process while injecting noise into the score network s θ at every denoising step. Specifically, we replace the score function with a perturbed version ˜ s θ ( t, x ) = s θ ( t, x ) + ζ , where ζ is a D -dimensional noise vector with independent and identically distributed components. We consider 4 noise distributions: centered Uniform , Gaussian , Laplace , and Student's-t with 3 degrees of freedom. For each noise type, we evaluate 6 values for the noise scale, σ ∈ { 0 . 25 , 0 . 5 , 1 , 2 , 3 , 4 } . All other elements of the generation pipeline-including the variance schedule, guidance scale, and number of sampling steps-are left unchanged. For each experimental setting, we generate 8192 CIFAR-10 images and 8192 CelebA-HQ images. Additional implementation details can be found in Appendix E.

Qualitative results. Figure 1 shows random generations for standard normal noise. We observe that injecting noise with constant variance into the score network has a negligible impact on the visual quality of the generated samples. As expected, the quality gradually degrades as the noise level increases. Additional qualitative results illustrating this phenomenon are provided in Appendix E.

FID sensitivity. The Fréchet Inception Distance (FID) is a widely used metric for assessing the quality of generative image models. In Figure 2, we plot the FID as a function of the noise scale σ . On CelebA-HQ, the FID increases only moderately up to σ ≈ 1 , while CIFAR-10 exhibits robustness up to σ ≈ 2 . In agreement with our theoretical findings, the shape of the noise distribution has negligible impact, only its scale matters. We also observe a sharp degradation in quality beyond a certain noise threshold, a phenomenon not accounted for by our theoretical analysis.

Figure 2: FID as a function of noise level for four distributions and different standard deviations.

<!-- image -->

## 7 Conclusion

In this paper, we provide a refined theoretical analysis of denoising diffusion probabilistic models (DDPMs), revealing two important features. First, we show that DDPMs exhibit robustness to noise in the estimated score function. Second, we establish that, when the true data-generating distribution belongs to a broad class-significantly larger than the class of log-concave distributions-DDPMs achieve fast convergence rates in the Wasserstein distance.

Our findings open several avenues for future research. One direction is the adaptation of our techniques to the analysis of kinetic Langevin diffusion-based DDPMs. It remains an open question whether such an extension would improve the dependence of the error bounds on the discretization step size. Additionally, the convergence rates we derive include terms that scale exponentially with certain parameters, such as the diameter of the support in the case of semi-log-concave targets. It is unclear whether this dependence is intrinsic to the problem or an artifact of our analysis. Finally, it would be of interest to assess the potential benefits of incorporating estimators of the Hessian of the log-density into the DDPM framework.

6 Code is available at https://github.com/VahanArsenian/DiffusionWasserstein

## Acknowledgements

This work was supported by Hi! PARIS and received government funding managed by the Agence Nationale de la Recherche under the France 2030 program, references (ANR-23-IACL-0005), (ANR23-PEIA-0004). This work was granted access to the HPC resources of IDRIS under the allocation 2025-AD011016491 made by GENCI. The work was partially supported by ERC grant SAGMOS (grant agreement No. 101201229).

## References

- [ACV24] Nima Anari, Sinho Chewi, and Thuy-Duong Vuong. Fast parallel sampling under isoperimetry. In Shipra Agrawal and Aaron Roth, editors, Proceedings of Thirty Seventh Conference on Learning Theory , volume 247 of Proceedings of Machine Learning Research , pages 161-185. PMLR, 30 Jun-03 Jul 2024.
- [ADR24] Iskander Azangulov, George Deligiannidis, and Judith Rousseau. Convergence of diffusion models under the manifold hypothesis in high-dimensions. CoRR , arXiv:2409.18804, 2024.
- [And82] B. D. O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications , 12(3):313-326, 1982.
- [BBDD24] Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly d-linear convergence bounds for diffusion models via stochastic localization. In The Twelfth International Conference on Learning Representations, ICLR 2024 , 2024.
- [Ber18] Espen Bernton. Langevin monte carlo and JKO splitting. In COLT , volume 75 of Proceedings of Machine Learning Research , pages 1777-1798. PMLR, 2018.
- [BL76] Herm Jan Brascamp and Elliott H. Lieb. Best constants in young's inequality, its converse, and its generalization to more than three functions. Advances in Mathematics , 20(2):151-173, 1976.
- [Bor22] Valentin De Bortoli. Convergence of denoising diffusion models under the manifold hypothesis. Transactions on Machine Learning Research , 2022.
- [BZL + 23] Stefano Bruno, Ying Zhang, Dong-Young Lim, Ömer Deniz Akyildiz, and Sotirios Sabanis. On diffusion-based generative models and their error bounds: The log-concave case with full convergence estimates. CoRR , arXiv:2311.13584, 2023.
- [CB18] Xiang Cheng and Peter L. Bartlett. Convergence of langevin MCMC in kl-divergence. In ALT , volume 83 of Proceedings of Machine Learning Research , pages 186-211. PMLR, 2018.
- [CCL + 23] Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. In The Eleventh International Conference on Learning Representations, ICLR 2023 , 2023.
- [CDS25] Giovanni Conforti, Alain Durmus, and Marta Gentiloni Silveri. KL convergence guarantees for score diffusion models under minimal data assumptions. SIAM J. Math. Data Sci. , 7(1):86-109, 2025.
- [Che24] Sinho Chewi. Log-Concave Sampling . Unpublished draft, 2024.
- [Cla83] Francis H. Clarke. Optimization and Nonsmooth Analysis . Classics in Applied Mathematics. Wiley-Interscience, New York, 1983.
- [CLL23] Hongrui Chen, Holden Lee, and Jianfeng Lu. Improved analysis of score-based generative modeling: User-friendly bounds under minimal smoothness assumptions. In ICML , volume 202 of Proceedings of Machine Learning Research , pages 4735-4763. PMLR, 2023.
- [CMFW24] Minshuo Chen, Song Mei, Jianqing Fan, and Mengdi Wang. An overview of diffusion models: Applications, guided generation, statistical rates and optimization. CoRR , arXiv:2404.07771, 2024.
- [CRYR24] Haoxuan Chen, Yinuo Ren, Lexing Ying, and Grant M. Rotskoff. Accelerating diffusion models with parallel sampling: Inference at sub-linear time complexity. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024 , 2024.

- [Dal17a] Arnak S. Dalalyan. Further and stronger analogy between sampling and optimization: Langevin monte carlo and gradient descent. In COLT , volume 65 of Proceedings of Machine Learning Research , pages 678-689. PMLR, 2017.
- [Dal17b] Arnak S. Dalalyan. Theoretical guarantees for approximate sampling from smooth and log-concave densities. J. R. Stat. Soc. Ser. B. Stat. Methodol. , 79(3):651-676, 2017.
- [DK19] Arnak S. Dalalyan and Avetik Karagulyan. User-friendly guarantees for the Langevin Monte Carlo with inaccurate gradient. Stochastic Process. Appl. , 129(12):5278-5311, 2019.
- [DM17] Alain Durmus and Éric Moulines. Nonasymptotic convergence analysis for the unadjusted Langevin algorithm. Ann. Appl. Probab. , 27(3):1551-1587, 2017.
- [DM19] Alain Durmus and Éric Moulines. High-dimensional Bayesian inference via the unadjusted Langevin algorithm. Bernoulli , 25(4A):2854-2882, 2019.
- [DMM19] Alain Durmus, Szymon Majewski, and Blazej Miasojedow. Analysis of langevin monte carlo via convex optimization. J. Mach. Learn. Res. , 20:73:1-73:46, 2019.
- [DT12] A. S. Dalalyan and A. B. Tsybakov. Sparse regression learning by aggregation and Langevin Monte-Carlo. J. Comput. System Sci. , 78(5):1423-1443, 2012.
- [Efr11] Bradley Efron. Tweedie's formula and selection bias. Journal of the American Statistical Association , 106:1602-1614, 12 2011.
- [EGZ19] Andreas Eberle, Arnaud Guillin, and Raphael Zimmer. Couplings and quantitative contraction rates for Langevin dynamics. The Annals of Probability , 47(4), 7 2019.
- [GCC24] Shivam Gupta, Linda Cai, and Sitan Chen. Faster diffusion-based sampling with randomized midpoints: Sequential and parallel. CoRR , arXiv:2406.00924, 2024.
- [GNZ25] Xuefeng Gao, Hoang M. Nguyen, and Lingjiong Zhu. Wasserstein convergence guarantees for a general class of score-based generative models. Journal of Machine Learning Research , 26(43):154, 2025.
- [GZ24] Xuefeng Gao and Lingjiong Zhu. Convergence analysis for general probability flow odes of diffusion models in wasserstein distances. arXiv:2401.17958 , 2024.
- [HBE20] Ye He, Krishnakumar Balasubramanian, and Murat A Erdogdu. On the ergodicity, bias and asymptotic normality of randomized midpoint sampling method. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 7366-7376. Curran Associates, Inc., 2020.
- [HJA20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS 2020, December 6-12, 2020, virtual , 2020.
- [HST25] Asbjørn Holk, Claudia Strauch, and Lukas Trottner. Statistical guarantees for denoising reflected diffusion models, 2025.
- [HWC24] Zhihan Huang, Yuting Wei, and Yuxin Chen. Denoising diffusion probabilistic models are optimally adaptive to unknown low dimensionality. CoRR , arXiv:2410.18784, 2024.
- [KALL18] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation, 2018.
- [KFL22] Dohyun Kwon, Ying Fan, and Kangwook Lee. Score-based generative modeling secretly minimizes the wasserstein distance. In Advances in Neural Information Processing Systems , 2022.
- [KH09] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical Report 0, University of Toronto, Toronto, Ontario, 2009.
- [KT25] Lea Kunkel and Mathias Trabs. On the minimax optimality of flow matching through the connection to kernel density estimation, 2025.
- [LHE + 24] Gen Li, Yu Huang, Timofey Efimov, Yuting Wei, Yuejie Chi, and Yuxin Chen. Accelerating convergence of score-based diffusion models, provably. In ICML . OpenReview.net, 2024.
- [LJLS25] Yuchen Liang, Peizhong Ju, Yingbin Liang, and Ness Shroff. Broadening target distributions for accelerated diffusion models via a novel analysis approach. In The Thirteenth International Conference on Learning Representations , 2025.

- [LLT22] Holden Lee, Jianfeng Lu, and Yixin Tan. Convergence for score-based generative modeling with polynomial complexity. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems , 2022.
- [LY24] Gen Li and Yuling Yan. Adapting to unknown low-dimensional structures in score-based diffusion models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024.
- [LY25] Gen Li and Yuling Yan. O(d/t) convergence theory for diffusion probabilistic models under minimal assumptions. In The Thirteenth International Conference on Learning Representations , 2025.
- [OAS23] Kazusato Oko, Shunta Akiyama, and Taiji Suzuki. Diffusion models are minimax optimal distribution estimators. In Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research , pages 26517-26582. PMLR, 23-29 Jul 2023.
- [PAD24] Peter Potaptchik, Iskander Azangulov, and George Deligiannidis. Linear convergence of diffusion models under the manifold hypothesis. CoRR , arXiv:2410.09046, 2024.
- [PW17] Yury Polyanskiy and Yihong Wu. Strong data-processing inequalities for channels and bayesian networks. In Convexity and Concentration , pages 211-249. Springer New York, 2017.
- [SGK10] Rajesh Sharma, Madhu Gupta, and G. Kapoor. Some better bounds on the variance with applications. Journal of Mathematical Inequalities , 4, 01 2010.
- [SL19] Ruoqi Shen and Yin Tat Lee. The randomized midpoint method for log-concave sampling. In Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019.
- [SO25] Marta Gentiloni Silveri and Antonio Ocello. Beyond log-concavity and score regularity: Improved convergence bounds for score-based generative models in w2-distance. In The Forty-Second International Conference on Machine Learning (ICML) , 2025.
- [SOB + 25] Stanislas Strasman, Antonio Ocello, Claire Boyer, Sylvain Le Corff, and Vincent Lemaire. An analysis of the noise schedule for score-based generative models. Transactions on Machine Learning Research , 2025.
- [SW14] Adrien Saumard and Jon A. Wellner. Log-concavity and strong log-concavity: a review, 2014.
- [TY24] Rong Tang and Yun Yang. Adaptivity of diffusion models to manifold structures. In Sanjoy Dasgupta, Stephan Mandt, and Yingzhen Li, editors, International Conference on Artificial Intelligence and Statistics, 2-4 May 2024, Palau de Congressos, Valencia, Spain , volume 238 of Proceedings of Machine Learning Research , pages 1648-1656. PMLR, 2024.
- [TZ25] Wenpin Tang and Hanyang Zhao. Score-based diffusion models via stochastic differential equations. Statistics Surveys , 19:28 - 64, 2025.
- [VCK25] Adrien Vacher, Omar Chehab, and Anna Korba. Polynomial time sampling from log-smooth distributions in fixed dimension under semi-log-concavity of the forward diffusion with application to strongly dissipative distributions. CoRR , arXiv:2501.00565, 2025.
- [Vil08] Cédric Villani. Optimal transport: old and new , volume 338. Springer Science &amp; Business Media, 2008.
- [VW19] Santosh S. Vempala and Andre Wibisono. Rapid convergence of the unadjusted langevin algorithm: Isoperimetry suffices. In NeurIPS , pages 8092-8104, 2019.
- [WCW24] Yuchen Wu, Yuxin Chen, and Yuting Wei. Stochastic runge-kutta methods: Provable acceleration of diffusion models. CoRR , arXiv:2410.04760, 2024.
- [Wib18] Andre Wibisono. Sampling as optimization in the space of measures: The langevin dynamics as a composite optimization problem. In Sébastien Bubeck, Vianney Perchet, and Philippe Rigollet, editors, Proceedings of the 31st Conference On Learning Theory , volume 75 of Proceedings of Machine Learning Research , pages 2093-3027. PMLR, 06-09 Jul 2018.
- [WWY24] Andre Wibisono, Yihong Wu, and Kaylee Yingxi Yang. Optimal score estimation via empirical bayes smoothing. In Shipra Agrawal and Aaron Roth, editors, The Thirty Seventh Annual Conference on Learning Theory, June 30 - July 3, 2023, Edmonton, Canada , volume 247 of Proceedings of Machine Learning Research , pages 4958-4991. PMLR, 2024.
- [YD25] Lu Yu and Arnak Dalalyan. Parallelized midpoint randomization for langevin monte carlo, 2025.

- [YKD24] Lu Yu, Avetik Karagulyan, and Arnak S. Dalalyan. Langevin monte carlo for strongly log-concave distributions: Randomized midpoint revisited. In The Twelfth International Conference on Learning Representations , 2024.
- [YW22] Kaylee Yingxi Yang and Andre Wibisono. Convergence in KL and rényi divergence of the unadjusted langevin algorithm using estimated score. In NeurIPS 2022 Workshop on Score-Based Methods , 2022.
- [YY25] Yifeng Yu and Lu Yu. Advancing wasserstein convergence analysis of score-based models: Insights from discretization and second-order acceleration. CoRR , arXiv:2502.04849, 2025.
- [YZS + 15] Fisher Yu, Yinda Zhang, Shuran Song, Ari Seff, and Jianxiong Xiao. Lsun: Construction of a largescale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365 , 2015.
- [YZS + 24] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Comput. Surv. , 56(4):105:1-105:39, 2024.

## Appendix

## Table of Contents

| A   | Classes of distributions satisfying Assumption 1 . . . . . . . . . . . . . . . . .   | . .                                                                                                                                                                  |   16 |
|-----|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
|     | A.1                                                                                  | Compactly supported distributions: property (a) . . . . . . . . . . . . . . . . .                                                                                    |   16 |
|     | A.2                                                                                  | Log-concave and semi-log-concave distributions: properties (b) and (c) . . . . .                                                                                     |   17 |
|     | A.3                                                                                  | Stability by orthogonal transform and concatenation: properties (d) and (e) . . .                                                                                    |   18 |
|     | A.4                                                                                  | Convolution with a spherical Gaussian: property (f) . . . . . . . . . . . . . . .                                                                                    |   19 |
|     | A.5                                                                                  | Convolution of a semi-log-concave and a compactly supported distribution: prop- erty (g) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   20 |
| B   | Proof of Lemma 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       | . . . .                                                                                                                                                              |   21 |
| C   | Proofs of the main results . . . . . . . . . . .                                     | . . . . . . . . . . . . . . . . . . . .                                                                                                                              |   22 |
|     | C.1                                                                                  | Main recursion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                   |   22 |
|     | C.2                                                                                  | Proof of Theorem 2: Strongly log-concave convolved with a compactly supported distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   |   25 |
|     | C.3                                                                                  | Proof of Theorem 3: Semi log-concave and compactly supported distribution on a subspace . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    |   28 |
| D   | Proofs of lemmas used in the proofs of main theorems . . . . . . . .                 | . . . . . . .                                                                                                                                                        |   29 |
|     | D.1                                                                                  | Proof of Lemma 10: the origin of the contraction/expansion . . . . . . . . . . .                                                                                     |   29 |
|     | D.2                                                                                  | Proof of Lemma 12: strength of the deflation in the contracting regime . . . . .                                                                                     |   30 |
|     | D.3                                                                                  | Proof of Lemma 13: assessing the increments of the drift . . . . . . . . . . . .                                                                                     |   31 |
| E   | Numerical Experiments .                                                              | . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                                        |   32 |
|     | E.1                                                                                  | Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                   |   33 |
|     | E.2                                                                                  | Additional Figures . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                   |   33 |
|     | E.3                                                                                  | Computational Resources . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                      |   34 |
|     | E.4                                                                                  | Dataset and Model Licensing . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                    |   34 |

## A Classes of distributions satisfying Assumption 1

Throughout the paper we make use of Tweedie's formula [Efr11, Eq. 1.4] which takes the following form using our notation: Let π Y be the probability density function of Y = α X + β ξ where ( X , ξ ) ∼ P ∗ ⊗ γ D , then

<!-- formula-not-decoded -->

This section shows that distributions mentioned in Section 2 satisfy Assumption 1.

## A.1 Compactly supported distributions: property (a)

Lemma 2. Let P X , Y be a probability measure defined on X ×Y , P X and P X | Y = y be the marginal and the conditional distributions of X . Then

<!-- formula-not-decoded -->

Proof. Let S X := supp( P X ) . Then by the definition of the marginal probability measure:

<!-- formula-not-decoded -->

On the other hand, by Bayes' theorem:

<!-- formula-not-decoded -->

where P Y is the marginal probability measure of Y . The proof is completed by noting that (11) yields P X | Y = y ( S X ) = 1 .

A simple consequence of Lemma 2 is that if diam(supp( P X )) ⩽ C then diam(supp( P X | Y = y )) ⩽ C . Using this result, we show that a random vector X with support diameter 2 D X satisfies Assumption 1 with φ ( σ ) = D 2 X .

Lemma 3 (Property (a) in Section 2) . Let X ∼ P such that diam(supp( P )) ⩽ 2 D X and let Y be any random variable defined on the same probability space. Then

<!-- formula-not-decoded -->

Proof. We need to prove that for any v ∈ R D :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By dividing both sides by ∥ v ∥ 2 , we can rewrite the target inequality with respect to a unit vector u ∈ R D :

<!-- formula-not-decoded -->

Denote Z = u T X . The supp( P Z ) is contained in the set { u T x | x ∈ supp( P X | Y = y ) } . By Lemma 2, the diam(supp( P X | Y = y )) ⩽ 2 D X . Let z 1 = u T x 1 and z 2 = u T x 2 for arbitrary x 1 , x 2 ∈ supp( P X | Y = y ) . The distance between them is:

<!-- formula-not-decoded -->

By the Cauchy-Schwarz inequality:

<!-- formula-not-decoded -->

Since ∥ u ∥ 2 = 1 , we write | z 1 -z 2 | ⩽ ∥ x 1 -x 2 ∥ 2 . The maximum possible value for ∥ x 1 -x 2 ∥ 2 is the diameter 2 D X . Therefore, | z 1 -z 2 | ⩽ 2 D X for all z 1 , z 2 in the support of Z . This implies that the support of Z is contained within an interval [ a, b ] such that the length of the interval b -a ⩽ 2 D X . We now apply Popoviciu's inequality on variances [SGK10], which yields that:

<!-- formula-not-decoded -->

which can be rewritten as:

## A.2 Log-concave and semi-log-concave distributions: properties (b) and (c)

Random vectors with m -strongly log-concave densities also satisfy Assumption 1, as shown in the lemma below.

Lemma 4 (Property (b) in Section 2) . Let ( X , ξ ) ∼ P ⊗ γ D , where the density of P , denoted as π ( x ) , is m -strongly log-concave. Then

<!-- formula-not-decoded -->

In addition, if x ↦→∇ log π ( x ) is M -Lipschitz for some M &gt; 0 , then

<!-- formula-not-decoded -->

Proof. By applying the preservation of strong log-concavity [SW14], we obtain that π X + σ ξ ( y ) is m 1+ mσ 2 -strongly log-concave. We then invoke Proposition 1 with parameters α = 1 and β = σ , which yields

<!-- formula-not-decoded -->

for Y = X + σ ξ , from which the first desired result follows.

For the second claim, set Y = X + σ ξ . The definition of semi-log-concavity yields

<!-- formula-not-decoded -->

The conditional density of X given Y satisfies

<!-- formula-not-decoded -->

with π Y | X = x ( y ) ∝ exp( -∥ y -x ∥ 2 2 σ 2 ) . Hence, the Hessian of π X | Y = y ( x ) is equal to:

<!-- formula-not-decoded -->

The Cramer-Rao inequality implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similar results hold for for semi-log-concave distributions with a compact support.

Lemma 5 (Property (c) in Section 2) . Let ( X , ξ ) ∼ P ⊗ γ D where P has a density w.r.t. Lebesgue measure denoted as π ( x ) and diam(supp( P )) ⩽ 2 D X . If π ( x ) is M -semi-log-concave for M ⩾ 0 , then:

<!-- formula-not-decoded -->

Proof. Denote Y = X + σ ξ . We obtain from the definition of semi-log-concavity that:

<!-- formula-not-decoded -->

The posterior of X given Y is proportional to the joint:

<!-- formula-not-decoded -->

with π ( y | x ) ∝ exp( -∥ y -x ∥ 2 2 σ 2 ) . Hence, the Hessian of log π ( x | y ) is equal to:

<!-- formula-not-decoded -->

where the last inequality follows from the semi-log-concavity of π ( x ) . By Brascamp-Lieb inequality [BL76], we have that:

<!-- formula-not-decoded -->

whenever Mσ 2 ⩽ 1 . The conditional variance of X can be bounded via Lemma 3, as P has a compact support:

<!-- formula-not-decoded -->

Combined with (12), we write:

<!-- formula-not-decoded -->

This completes the proof of the lemma.

## A.3 Stability by orthogonal transform and concatenation: properties (d) and (e)

Afterwards, we prove that if X satisfies Assumption 1 then its rotation also satisfies Assumption 1 with the same φ ( σ ) .

Lemma 6 (Property (d) in Section 2) . Let ( X , ξ ) ∼ P ⊗ γ D and

<!-- formula-not-decoded -->

Then for any orthonormal matrix U , we have that:

<!-- formula-not-decoded -->

for ξ ′ ∼ γ D and ξ ′ ⊥ ⊥ U X .

<!-- formula-not-decoded -->

Let y := U T y ′ and ξ := U T ξ ′ . By using the properties that U T U = I D as U is orthonormal and that ξ ∼ γ D independently from X as ξ ′ ∼ γ D and ξ ′ ⊥ ⊥ U X , we write:

<!-- formula-not-decoded -->

and the claim of the lemma follows.

We now show that the concatenation of two independent random vectors satisfying Assumption 1 also satisfies Assumption 1.

Lemma7 (Property (e) in Section 2) . Let ( X 1 , X 2 ) ∼ P 1 ⊗ P 2 , where P 1 and P 2 satisfy Assumption 1 for some φ . Then the concatenation of X 1 and X 2 , denoted as X 1 ⊕ X 2 also satisfies Assumption 1 for the same φ .

Proof. Let X 1 be d 1 -dimensional, X 2 be d 2 -dimensional, and D = d 1 + d 2 . Consider ξ ∼ γ D and independent of ( X 1 , X 2 ) . We may write

<!-- formula-not-decoded -->

We have that ( X 1 , X 2 , ξ 1 , ξ 2 ) are mutually independent as ( X 1 , X 2 , ξ ) are mutually independent and ξ 1 and ξ 2 are uncorrelated. From ( X 1 , ξ 1 ) ⊥ ⊥ ( X 2 , ξ 2 ) we get that ( X 1 , Y 1 ) ⊥ ⊥ ( X 2 , Y 2 ) . Applying the weak union property of the conditional independence twice we get:

<!-- formula-not-decoded -->

Hence the covariance of X 1 and X 2 given ( Y 1 , Y 2 ) is 0 . Finally,

<!-- formula-not-decoded -->

where the last inequality is due to P 1 and P 2 satisfying Assumption 1.

## A.4 Convolution with a spherical Gaussian: property (f)

Lemma 8 (Property (f) in Section 2) . Let ( W , ζ ) ∼ P 0 ⊗ γ D . If W satisfies Assumption 1 with the function φ 0 , then, for every τ &gt; 0 , X = W + τ ζ satisfies Assumption 1 with the function

<!-- formula-not-decoded -->

Proof. Let us define Y = X + σ ξ = W + τ ζ + σ ξ and η := τ ζ + σ ξ . Since ξ , ζ i.i.d. ∼ γ D are independent of W , we have η ∼ γ D with covariance ( τ 2 + σ 2 ) I D and Y = W + η . Equivalently,

<!-- formula-not-decoded -->

Using Assumption 1 with noise level √ τ 2 + σ 2 leads to

<!-- formula-not-decoded -->

To ease notation, we write E y and Var y to refer to the conditional expectation and conditional variance given Y = y , respectively. By the law of total variance, we have

<!-- formula-not-decoded -->

We know that τ ζ and η = τ ζ + σ ξ are linear transforms of two independent standard Gaussians. Hence, the standard covariance calculation gives us

<!-- formula-not-decoded -->

And since Var( X | Y = y , W ) = Var( τ ζ | η ) , we get the first part of (13) equal to

<!-- formula-not-decoded -->

For the second term, since ζ , ξ i.i.d. ∼ N ( 0 , I D ) , then the corresponding 2 D -dimensional vector

<!-- formula-not-decoded -->

So, the conditional expectation that we are interested in will be equal to

<!-- formula-not-decoded -->

Under the conditioning on both Y and W , the quantity η = Y -W is deterministic. Therefore,

<!-- formula-not-decoded -->

Given Y = y , the second term is deterministic, so

<!-- formula-not-decoded -->

Adding the two components gives us

<!-- formula-not-decoded -->

which proves the lemma with φ τ ( σ ) = τ 2 σ 2 2 2 + σ 4 φ 0 ( √ τ 2 + σ 2 ) 2 2 2 .

τ + σ ( τ + σ )

## A.5 Convolution of a semi-log-concave and a compactly supported distribution: property (g)

Lemma 9 (Property (g) in Section 2) . If P ∗ = P slc ⋆P cmpct , where P slc is an m -strongly log-concave distribution with an M -Lipschitz score function, and P cmpct is supported on a compact set with diameter 2 D , then P ∗ satisfies Assumption 1 with

<!-- formula-not-decoded -->

Proof. Let W ∼ P cmpct and ζ ∼ P slc be two independent random vectors so that X = W + ζ ∼ P ∗ . This means that for some compact set K with diameter 2 D , we have Var( W ) ⩽ 4 D 2 , and that the density π ζ is continuously differentiable with a score function s ζ satisfying

<!-- formula-not-decoded -->

For ξ ⊥ ⊥ ( W , ζ ) such that ξ ∼ γ D , and for Y = X + σ ξ , we have to prove that

<!-- formula-not-decoded -->

As before, to ease notation, we write E y and Var y to refer to the conditional expectation and conditional variance given Y = y , respectively. By the law of total variance, we have

<!-- formula-not-decoded -->

Since the random vector ζ is m -strongly log-concave, it follows from Lemma 4 that

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Hence, Var( X | Y = y , W ) ⩽ σ 2 1+ mσ 2 almost surely. This implies that

<!-- formula-not-decoded -->

We switch to assessing the second term in (14). It holds that

<!-- formula-not-decoded -->

where 1 is a consequence of X = W + ζ , 2 follows from the independence of ζ and W , 3 is obtained by the Tweedie formula recalled in (10). Let us set ψ ( w ) = ∇ log π ζ + σ ξ ( y -w ) . The second claim of Lemma 4 combined with Proposition 1 implies that ψ is Lipschitz-continuous with the constant M/ (1 + Mσ 2 ) . Therefore,

<!-- formula-not-decoded -->

where in the last step we used Lemma 3.

## B Proof of Lemma 1

We start by first proving that:

<!-- formula-not-decoded -->

The data processing inequality [PW17] states that:

<!-- formula-not-decoded -->

Combined with the concentration property of Ornstein-Uhlenbeck process [GZ24, EGZ19]:

<!-- formula-not-decoded -->

gives the desired result.

We now focus on a subset of N ′ ⊂ N that contains D dimensional Gaussian distributions with mean 0 and (1 + σ 2 ) I D covariance matrix with σ &gt; 0 . Clearly

<!-- formula-not-decoded -->

Let X t be defined by Equation (2), then the distribution of X t is N ( 0 , ( e -2 t σ 2 +1) I D ) . Hence, the true score function is

<!-- formula-not-decoded -->

where σ 2 ( t ) = e -2 t σ 2 +1 . Equation (4) obtains the following form under this score function:

<!-- formula-not-decoded -->

The integrating factor for the SDE is:

<!-- formula-not-decoded -->

From Itô's product rule applied to I ( t ) ˜ Y t , we get:

<!-- formula-not-decoded -->

where we have used the fact that d I ( t ) = -I ( t ) ( 1 -2 σ 2 ( T -t ) ) d t . Integrating both sides of (15) from 0 to t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

from which:

Note that ˜ Y 0 ∼ γ D . Combined with the fact that I ( t ) is a deterministic function, we infer from (16) that ˜ Y t is a zero mean Gaussian random variable. So the Wasserstein distance between γ D and the distribution of ˜ Y t depends only on the covariance matrices:

<!-- formula-not-decoded -->

where σ 2 ˜ Y t I D is the covariance of ˜ Y t .

Let Z t := √ 2 ∫ t 0 I ( u ) d ˜ B u . Hence, Z t ∼ N ( 0 , 2 ∫ t 0 I 2 ( u ) d u I D ) and it is independent of ˜ Y 0 . The variance of Z t is:

<!-- formula-not-decoded -->

The variance of ˜ Y T can be computed from Equation (16):

<!-- formula-not-decoded -->

Plugging in the value of σ ˜ Y T into (17) we get:

<!-- formula-not-decoded -->

We note that W 2 ( P ∗ ; γ D ) = | √ σ 2 +1 -1 | √ D σ →∞ ∼ σ √ D , so we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

When combined with the established contraction behavior of the backward diffusion-operating with the true score function-in the 2-Wasserstein metric for Gaussian distributions [EGZ19], we get:

<!-- formula-not-decoded -->

## C Proofs of the main results

We recall that P ∗ is the target distribution and P ∗ t = α t P ∗ + β t γ D is the distribution of the forward process at time t &gt; 0 , with α t = e -t = √ 1 -β 2 t . We also fix some T &gt; 0 and define Y t = X T -t and Q ∗ t = Law ( Y t ) ; Y t is the state of the backward process (3). We set ˜ P k to be the law of Z k defined by (5) so that P DDPM = ˜ P K +1 . Throughout this proof, we will repeatedly use the following notation:

<!-- formula-not-decoded -->

## C.1 Main recursion

We set T = t K +1 and consider a version of the continuous-time process ( Y t ) 0 ⩽ t ⩽ T and the discretetime process ( Z k ) 0 ⩽ k ⩽ K +1 defined on the same probability space and coupled by the relation ξ k +1 = ( ˜ B t k +1 -˜ B t k ) / √ h k . We then use the definition of the Wasserstein distance to infer that

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

In what follows, we use the notation ∆ k = Y t k -Z k and

<!-- formula-not-decoded -->

This allows us to rewrite (19) as follows

<!-- formula-not-decoded -->

In view of (18), we are interested in bounding the term

<!-- formula-not-decoded -->

We will proceed by establishing a recursive inequality upper bounding x k +1 by a simple expression involving x k , and then by unfolding this recursive inequality.

Let us introduce the filtration ( F k ) k ∈ N . The first element of this sequence is the σ -algebra generated by Y 0 and Z 0 . Then, each F k +1 is obtained by extending F k to the smallest σ -algebra for which both ζ k and the process ( ˜ B t -˜ B t k ) t ∈ [ t k ; t k +1 ] are measurable. Note that Z k is necessarily F k -measurable, but the same is not true for ζ k . Indeed, the estimator ˜ s ( T -t k , · ) may depend on some random variables that are not in F k .

It is clear that

<!-- formula-not-decoded -->

From (21), by the triangle inequality,

<!-- formula-not-decoded -->

∥ Furthermore,

<!-- formula-not-decoded -->

Combining displays (22), (23) and (24), we arrive at

<!-- formula-not-decoded -->

In what follows, it is convenient to use the following notation: for every k ∈ N , let α k = e -( T -t k ) and β 2 k = 1 -α 2 k .

Lemma 10. If P ∗ satisfies Assumption 1 with a function φ and

<!-- formula-not-decoded -->

then,

<!-- formula-not-decoded -->

Lemma 10 implies that

<!-- formula-not-decoded -->

The next lemma which can be easily deduced by induction applying the Minkowski inequality, will be used to derive a global bound on the error x K from recursive inequalities upper bounding the error x k +1 at the ( k +1) th step by the one of the k th step.

Lemma 11. Let ( A k ) k ∈ N , ( B k ) k ∈ N and ( C k ) k ∈ N be three sequences of real numbers such that B k ⩾ 0 and C k ⩾ 0 for every k . If ( x k ) k ∈ N satisfies the recursive inequality

<!-- formula-not-decoded -->

then, for ¯ A k = A 0 + . . . + A k ,

<!-- formula-not-decoded -->

For the subsequent steps of the proof, we leverage the properties of discretization. We begin with the portion employing constant step-sizes. This discretization is applied in the time interval where the inequality from (26) yields a near-contraction. This is equivalent to considering the values of k for which m k in (27) is positive and bounded away from zero.

Lemma 12. If T and a ⩾ 1 are real numbers such that T ⩾ 1 2 log(6 a ) . Let K 0 ∈ N be such that for every k ∈ { 0 , 1 , . . . , K 0 } ,

<!-- formula-not-decoded -->

Then, for α k = e -( T -t k ) , we have α 2 k ⩽ 1 / (6 a ) as well as

<!-- formula-not-decoded -->

for all k = 0 , . . . , K 0 .

Figure 3: Notations corresponding to the discretization schedule.

<!-- image -->

We set h k = h for k = 0 , . . . , K 0 . Then, (27), Lemma 11 and 1 -h k m k ⩽ 1 -h/ 3 ⩽ e -h/ 3 imply that

<!-- formula-not-decoded -->

where 1 comes from applying Lemma 11 with e A k = (1 -m k h ) ⩽ e -h/ 3 , form which we get e ¯ A j = e A 0 · e A 1 ...e A j = j ∏ l =0 (1 -m l h l ) ⩽ ( 1 -h 3 ) j +1 and e ¯ A K 0 -1 = e -K 0 h/ 3 , and 2 uses the fact that ∑ K 0 -1 k =0 (1 -h 3 ) K 0 -k -1 = 3 h [ 1 -( 1 -h 3 ) K 0 ] ⩽ 3 h and, similarly, ∑ K 0 -1 k =0 (1 -h 3 ) 2( K 0 -k -1) = 9 h (6 -h ) [ 1 -( 1 -h 3 ) 2 K 0 ] ⩽ 9 h (6 -h ) ⩽ 9 5 . 3 h ⩽ 1 . 7 h since h ⩽ 0 . 7 .

The next lemma provides an upper bound for the bias and the variance of the discretization error.

Lemma 13. Assume that for some a &gt; 0 and k ∈ { 0 , . . . , K } , P ∗ satisfies Assumption 1 with φ satisfying φ ( σ ) ⩽ a for every σ ∈ [ β k +1 /α k +1 ; β k /α k ] . Assume, in addition, that ¯ m 2 = ( E [ ∥ X ∥ 2 ] /D ) ∨ 1 &lt; ∞ . Then, it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If instead of φ ( σ ) ⩽ a , we have φ ( σ ) ⩽ ¯ aσ 2 for some ¯ a ⩾ 1 , then (30) can be strengthened as follows

<!-- formula-not-decoded -->

Finally, under the same condition, the error V K of the last iterate can be bounded by

<!-- formula-not-decoded -->

## C.2 Proof of Theorem 2: Strongly log-concave convolved with a compactly supported distribution

We know that

<!-- formula-not-decoded -->

Therefore, we can apply Lemma 10, Lemma 12 with a = 1 ∨ [(1 /m ) + b ] as well as inequalities (29) and (31) of Lemma 13 with ¯ a = 1 + 1 4 bM . In addition, to bound the last term in (31), we use the fact that

<!-- formula-not-decoded -->

Together with (28), this leads to

<!-- formula-not-decoded -->

On the time interval [ T -log(6 a ) 2 ; T ] , we use the discretization obtained by geometrically decreasing stepsizes as previously proposed in the literature:

<!-- formula-not-decoded -->

where c ⩽ 0 . 6 / log(6 a ) . This implies, in particular, that c ⩽ 0 . 6 / log 6 ⩽ 0 . 4 and that ¯ h := max k ∈ [ K 0 ,K ] h k ⩽ 0 . 3 . The constants c and K are chosen in such a way that t K = T -h K for some small h K ⩽ log(6 a ) 2 , and t K +1 = T . This means that

<!-- formula-not-decoded -->

This yields

<!-- formula-not-decoded -->

For k ⩾ K 0 +1 , we will apply Lemma 10. To check that its conditions are fulfilled, note that

<!-- formula-not-decoded -->

This expression, multiplied by h k , is less than 2 whenever h k ⩽ 0 . 3 . Indeed, on the one hand,

<!-- formula-not-decoded -->

On the other hand, for k &gt; K 0 ,

<!-- formula-not-decoded -->

This implies that

<!-- formula-not-decoded -->

since c ⩽ 0 . 6 . In addition, taking σ = β k /α k and using the substitution β 2 k = 1 -α 2 k , we have

<!-- formula-not-decoded -->

where 1 comes from the definition of m k from (25), 1 is true for any φ ( σ ) satisfying (6). Equality 3 comes from the fact that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Finally, noting that

<!-- formula-not-decoded -->

for any m,σ 2 ⩾ 0 , we arrive at

<!-- formula-not-decoded -->

Therefore, (27) yields

<!-- formula-not-decoded -->

From this recursion and Lemma 11, using the notation H ( k ) = -m K 0 h K 0 -. . . -m k h k , we infer that

<!-- formula-not-decoded -->

Inequality (35) yields

<!-- formula-not-decoded -->

Let us set y j = M ( e 2( T -t j ) -1) . On the one hand, we have

<!-- formula-not-decoded -->

On the other hand, since h j ⩽ 0 . 3 , we have e -2 h j -1 ⩽ -1 . 5 h j . Therefore,

<!-- formula-not-decoded -->

This implies that

<!-- formula-not-decoded -->

Using the standard inequalities

<!-- formula-not-decoded -->

we arrive at

<!-- formula-not-decoded -->

We apply then inequalities (29), (31) and (32) of Lemma 13 with ¯ a = 1 + 1 4 bM . This leads to

<!-- formula-not-decoded -->

The stepsizes h k of the geometric grid are much smaller than the noise levels β 2 k +1 , as attested by the following inequality 7

<!-- formula-not-decoded -->

7 We use the standard inequality 1 -e -x ⩾ (1 -e -1 )( x ∧ 1) for every x &gt; 0 .

It follows from (34) that T -t k +1 = c -1 h k +1 = c -1 (1 -c ) h k ⩾ 2 3 c -1 h k . Hence,

<!-- formula-not-decoded -->

Combining (37) and (38), we arrive at

<!-- formula-not-decoded -->

This inequality, in conjunction with (33), leads to

<!-- formula-not-decoded -->

where h max = max( h, ¯ h ) is the maximal step size of the entire discretization grid, comprising the parts defined through arithmetic and geometric progressions. These step sizes should satisfy the inequalities

<!-- formula-not-decoded -->

To bound x 0 , we note that

<!-- formula-not-decoded -->

as soon as T ⩾ log(6) . Thus, x 0 ⩽ 1 . 01 √ ¯ m 2 De -T . We set T = 1 2 log(6 a ) + T 1 and h K = δ = 0 . 5 e -2 T 1 and K = 2 K 0 . This leads to the claim of the theorem. Indeed, h ⩽ 0 . 7 translates into K 0 ⩾ (10 / 7) T 1 and ¯ h ⩽ 0 . 3 translates into

<!-- formula-not-decoded -->

which is satisfied when K 0 ⩾ 7 T 1 log(6 a ) + 4 log(6 a ) log log(6 a ) . Finally, notice that h ⩽ T 1 /K 0 and

<!-- formula-not-decoded -->

These inequalities yield the claimed upper bound on h max .

## C.3 Proof of Theorem 3: Semi log-concave and compactly supported distribution on a subspace

For P ∗ satisfying Assumption 1 with the function

<!-- formula-not-decoded -->

we can apply Lemma 12 with a = b ∨ 1 and Lemma 13 with ¯ a = bM +1 . Similarly to Appendix C.2, the application of Lemma 10 and Lemma 12 yields

<!-- formula-not-decoded -->

We again use the discretization with geometrically decreasing stepsize on the interval [ T -log(6 a ) 2 ; T ] :

<!-- formula-not-decoded -->

where c ⩽ 0 . 6 / log(6 a ) . Following the discussion in Appendix C.2, we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and, for k &gt; K 0

Combined with (39), we get

<!-- formula-not-decoded -->

Hence, 27 yields

<!-- formula-not-decoded -->

We denote H k = (2¯ a -1) ∑ k i = K 0 h k . We note that H K ⩽ 2¯ a -1 2 log(6 a ) . Lemma 11 states:

<!-- formula-not-decoded -->

As 2¯ a -1 = 2 bM +1 which is strictly positive, we may apply (36) which results in:

<!-- formula-not-decoded -->

We apply then inequalities (29), (31) and (32) of Lemma 13, which leads to

<!-- formula-not-decoded -->

The above inequality with (38) yields:

<!-- formula-not-decoded -->

Combining (41) with (40) and noting that (2¯ a -1) ⩾ 1 , we get:

<!-- formula-not-decoded -->

Following the discussion of Appendix C.2, we complete the proof by showing that:

<!-- formula-not-decoded -->

## D Proofs of lemmas used in the proofs of main theorems

We collect in this section the proofs of the building blocks of our main results.

## D.1 Proof of Lemma 10: the origin of the contraction/expansion

Since s is continuously differentiable, by the mean-value identity, we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The matrix M k is symmetric, and according to Proposition 1, all its eigenvalues satisfy

<!-- formula-not-decoded -->

Since U k = ∫ 1 0 M k ( θ ) ∆ k d θ , we get

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

We assume that h k is chosen so that

<!-- formula-not-decoded -->

This is equivalent to

<!-- formula-not-decoded -->

Regrouping the terms, we get

<!-- formula-not-decoded -->

This inequality can be checked to be the same as (25). Hence, (42) is indeed satisfied and, therefore,

<!-- formula-not-decoded -->

Therefore, by the triangle (Minkowski) inequality, we have

<!-- formula-not-decoded -->

This completes the proof of Lemma 10.

## D.2 Proof of Lemma 12: strength of the deflation in the contracting regime

First, notice that α k being an increasing function of t k , we have

<!-- formula-not-decoded -->

Second, since we assumed φ ( β k /α k ) ⩽ a , we have

<!-- formula-not-decoded -->

Since α 2 k ⩽ 1 / (6 a ) and we assumed a ⩾ 1 , we have α 2 k ⩽ a . Combining these inequalities with 0 ⩽ 1 -α 2 k ⩽ 1 , we arrive at

<!-- formula-not-decoded -->

For the second inequality of the lemma, it suffices to notice that φ ( σ ) ⩾ 0 and a ⩾ 1 imply that

<!-- formula-not-decoded -->

Combining with the condition h k ⩽ 0 . 7 , this yields h k ( 1+ α 2 k 1 -α 2 k + m k ) ⩽ 2 and completes the proof of the lemma.

## D.3 Proof of Lemma 13: assessing the increments of the drift

Let b t = Y t + 2 s ( T -t, Y t ) . To prove the first inequality, we recall that s ( T -t, y ) = ( α T -t E [ X 0 | Y t = y ] -y ) /β 2 T -t . Therefore,

<!-- formula-not-decoded -->

In addition, Y t = α X 0 + β ξ with ξ ⊥ ⊥ X 0 and ξ ∼ N D (0 , I D ) . It holds that

<!-- formula-not-decoded -->

since ξ is independent of X 0 and has zero mean.

Let us use the 'local notation' ¯ s ( t, y ) = s ( t, y ) + y as well as H ( t, y ) = D s ( t, y ) . According to [CDS25, Prop. 2], it holds that

<!-- formula-not-decoded -->

Since ˜ B t -˜ B t k is independent of the σ -algebra F k , we get

<!-- formula-not-decoded -->

and, therefore,

<!-- formula-not-decoded -->

The definition of V k given in (20) implies that V k = ∫ t k +1 t k ( b t -b t k ) d t . This leads to

<!-- formula-not-decoded -->

This yields the claim of (29).

We prove now (30). The definition of b t = Y t +2 s ( T -t, Y t ) leads to

<!-- formula-not-decoded -->

On the one hand, in view of the law of total variance, we have ∥ ∥ Y u -E [ Y u | F k ] ∥ ∥ L 2 ⩽ ∥ ∥ Y u ∥ ∥ L 2 . Therefore, using (43), we get

<!-- formula-not-decoded -->

On the other hand, the properties of the stochastic integral imply that

<!-- formula-not-decoded -->

Combining he definition of V k given in (20) with (44), (45) and (46), we get

<!-- formula-not-decoded -->

The integral in (46) can be bounded from above using Proposition 1 and various assumptions of the function φ from Assumption 1. Indeed, denoting σ T -u = β T -u /α T -u , we have H ( u ) ≼ β -2 T -u ( φ ( σ T -u ) σ -2 T -u -1) I D . Since, in addition H ( u ) ≽ -β -2 T -u I D , we get

<!-- formula-not-decoded -->

If we assume that φ ( σ T -u ) ⩽ a , we arrive at

<!-- formula-not-decoded -->

In view of (47), this yields

<!-- formula-not-decoded -->

This completes the proof of the second claim of the lemma.

If instead of the assumption φ ( σ ) ⩽ a , we use the assumption φ ( σ ) ⩽ ¯ aσ 2 with ¯ a ⩾ 1 , inequality (48), the fact that u ↦→ β T -u is decreasing, and inequality (47) imply that

<!-- formula-not-decoded -->

For the last claim, we use (47) and (48) as follows

<!-- formula-not-decoded -->

Thus, from (47), we infer that

<!-- formula-not-decoded -->

This completes the proof.

## E Numerical Experiments

Our experiments follow the standard DDPM sampling procedure as described in the original DDPM paper by [HJA20], specifically the pseudocode presented in their Algorithm 2.

## E.1 Implementation Details

For clarity, we re-state their algorithm below.

## Algorithm 3 DDPM Sampling [HJA20]

- 1: x T ∼ N ( 0 , I )
- 2: for t = T to 1 do
- 3: z ∼ N ( 0 , I ) if t &gt; 1 , else z = 0
- 4: µ θ ( x t , t ) = 1 √ α t ( x t -1 -α t √ 1 -¯ α t ϵ θ ( x t , t ) )
- 5: x t -1 = µ θ ( x t , t ) + σ t z
- 6: end for
- 7: return x 0

To better explain the correspondence between notation used in our paper and that of [HJA20], we provide the following table:

| Notation in [HJA20]                                 | Our notation                            |
|-----------------------------------------------------|-----------------------------------------|
| x T , . . . , x 0 z σ t α t ¯ α t ϵ θ ( x t , t ) √ | Z 0 , . . . , Z K +1                    |
|                                                     | ξ k +1                                  |
|                                                     | √ 2 h k                                 |
|                                                     | (1+ h k ) - 2 ≈ e - 2 h k ≈ 1 - 2 h k k |
|                                                     | ∏ j =0 (1+ h k ) - 2 ≈ e - 2 t K +1     |
|                                                     | - 2 ˜ s ( T - t k , Z k )               |
| 1 - ¯ α t                                           | - 2 ˜ s ( T - t k , Z k )               |

To evaluate the robustness of the generative process under perturbed score estimates, we had to isolate the score estimation component within the sampling loop. In the formulation of [HJA20], this corresponds to the rescaled neural network output -0 . 5 ϵ θ ( x t , t ) / √ 1 -¯ α t . In our experiments, we added various forms of noise (Gaussian, Uniform, Laplace, and Student'st ) directly to this term, simulating inaccurate or noisy score predictions. This modification allows us to assess the impact of score perturbations on the quality of generated samples, both visually and quantitatively.

We know that in our formulation of the problem, the conditional expectation of the next state given that the current state is x is given by µ θ ( x , t ) = (1+ h ) x +2 s ( t, x ) h . Therefore, adding ζ to s ( t, x

<!-- formula-not-decoded -->

## E.2 Additional Figures

Qualitative results. Figure 6, Figure 7 and Figure 8 extend the main-paper image grids. For each dataset (CIFAR-10, CelebA-HQ, and LSUN-Churches) we display samples generated with Gaussian, Laplace, and Student'st score noise at two strengths, σ = 0 . 5 or σ = 1 (moderate) and σ = 2 (severe). Rows share the same latent seed as the baseline to enable direct visual comparison.

Quantitative trends. Figure 4 tracks FID on the CIFAR-10 dataset as we truncate the 1 000-step DDPM schedule at {250, 500, 750, 1000} steps for the clean score and the i.i.d. N ( 0 , I D ) noise contaminated score. We observe that performance increases at a similar rate with the number of steps for both clean and noisy score estimates.

Additionally, Figure 5 illustrates the 'deterioration' of three distinct pictures for each of the different models (datasets) that we have - each starting with a fixed random noise, generating the corresponding image after 1000 diffusion steps with the noise contaminated score, as described before, parametrized by different σ . We observe that datasets with higher-resolution images and, respectively, deeper noise (alternatively, score) predicting neural networks exhibit higher deterioration than those with low-resolution images.

) .

Figure 4: FID as a function of time steps. Blue: standard DDPM inference. Orange: same sampler with i.i.d. N ( 0 , I D ) noise added to the score at each step.

<!-- image -->

Figure 5: A single example of CIFAR-10 (top), CelebA-HQ (middle) and LSUN-Churches (bottom) generated data, respectively, over different standard deviations.

<!-- image -->

## E.3 Computational Resources

This project was provided with computer and storage resources by GENCI at IDRIS thanks to the grant 2025-AD011016491 on the supercomputer Jean Zay's A100 partition.

Some of the experiments were run on two additional GPU nodes: one with AMD EPYC 7V12 64-Core Processor, 1TB of RAM, and with 8xA100 40GB VRAM version NVIDIA GPUs. The other one with AMD EPYC 9005 192-Core Processor, 0.5TB of RAM, and with 2xH100 NVIDIA GPUs.

Sampling 8192 CIFAR-10 images or 512 CelebA-HQ or 512 LSUN-Churches images takes 1.5 GPU-hours. FID evaluation for all the scale values of a single noise distribution takes 0.2 GPU-hours.

## E.4 Dataset and Model Licensing

- CIFAR-10: Licensed under the MIT License.

- CelebA-HQ: Licensed under CC BY-NC 4.0.

- LSUN-Churches: Licensed under CC BY-NC 4.0.

- google/ddpm-cifar10-32: Apache License, Version 2.0.

- google/ddpm-celebahq-256: Apache License, Version 2.0.

- google/ddpm-church-256: Apache License, Version 2.0.

Figure 6: Additional CIFAR-10 generations for 3 noise families (rows) and 2 noise levels (columns).

<!-- image -->

<!-- image -->

(a) No noise

<!-- image -->

(d) No noise

<!-- image -->

(g) No noise

<!-- image -->

(b) Gaussian noise, σ = 0 . 5

<!-- image -->

(e) Laplace, σ = 0 . 5

<!-- image -->

(h) Student's t , σ = 0 . 5

<!-- image -->

(c) Gaussian noise, σ = 1

(f) Laplace, σ = 1

<!-- image -->

(i) Student's t , σ = 1

<!-- image -->

Figure 7: Additional CelebA-HQ generations for 3 noise families (rows) and 2 noise levels (columns).

(g) No noise

<!-- image -->

(h) Student's t , σ = 1

(i) Student's t , σ = 2

Figure 8: Additional LSUN-Church generations for 3 noise families (rows) and 2 noise levels (columns).

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The summary of the contributions can be seen in Section 1 paragraph Contributions .

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations are summarized in the Section 7 Conclusion .

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

Justification: All of the proofs can be found in the Appendix (Supplementary Material).

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

Justification: See the details in Section 6 from the main paper and Appendix E from the supplementary materials.

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

Justification: The datasets and models used in our experiments are open-source. The code will be provided as a zip file.

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

Answer: [No]

Justification: We base our experiments on existing, already trained models. All of the details can be found in cited work.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: Due to the limited computational resources and the cost of the experiments of diffusion models, we do not report error bars.

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

Justification: The details can be found in Appendix E.3 from the supplementary materials. Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: No deviations from the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: [NA]

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

Justification: [NA]

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: See the details in Appendix E.4 Dataset and Model Licensing .

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

Justification: We release the code used in our experiments, including sampling with perturbed scores, under an open-source license. Anonymized code and documentation are included in the supplementary materials.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: [NA]

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.