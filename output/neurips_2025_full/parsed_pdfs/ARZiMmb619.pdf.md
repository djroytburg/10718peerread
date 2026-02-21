## Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics

Zhiyang Xun

UT Austin

zxun@cs.utexas.edu

## Shivam Gupta

UT Austin ∗

shivamgupta@utexas.edu

## Abstract

Given a noisy linear measurement y = Ax + ξ of a distribution p ( x ) , and a good approximation to the prior p ( x ) , when can we sample from the posterior p ( x | y ) ? Posterior sampling provides an accurate and fair framework for tasks such as inpainting, deblurring, and MRI reconstruction, and several heuristics attempt to approximate it. Unfortunately, approximate posterior sampling is computationally intractable in general.

To sidestep this hardness, we focus on (local or global) log-concave distributions p ( x ) . In this regime, Langevin dynamics yields posterior samples when the exact scores of p ( x ) are available, but it is brittle to score-estimation error, requiring an MGF bound (sub-exponential error). By contrast, in the unconditional setting, diffusion models succeed with only an L 2 bound on the score error. We prove that combining diffusion models with an annealed variant of Langevin dynamics achieves conditional sampling in polynomial time using merely an L 4 bound on the score error.

## 1 Introduction

Diffusion models are currently the leading approach to generative modeling of images. Diffusion models are based on learning the 'smoothed scores' s σ 2 ( x ) of the modeled distribution p ( x ) . Such scores can be approximated from samples of p ( x ) by optimizing the score matching objective [HD05]; and given good L 2 -approximations to the scores, p ( x ) can be efficiently sampled using an SDE [SE19, HJA20, SSDK + 21] or an ODE [SME20].

Much of the promise of generative modeling lies in the prospect of applying the modeled p ( x ) as a prior : combining it with some other information y to perform a search over the manifold of plausible images. Many applications, including MRI reconstruction, deblurring, and inpainting, can be formulated as linear measurements

<!-- formula-not-decoded -->

for some (known) matrix A ∈ R m × d . Posterior sampling , or sampling from p ( x | y ) , is a natural and useful goal. When aiming to reconstruct x accurately, it is 2-competitive with the optimal in any metric [JAD + 21] and satisfies fairness guarantees with respect to protected classes [JKH + 21].

Researchers have developed a number of heuristics to approximate posterior sampling using the smoothed scores, including DPS [CKM + 23], particle filtering methods [WTN + 23, DS24], DiffPIR [ZZL + 23], and second-order approximations [RCK + 24]. Unfortunately, unlike for unconditional sampling, these methods do not converge efficiently and robustly to the posterior distribution. In fact, a lower bound shows that no algorithm exists for efficient and robust posterior sampling in

∗ Now at Google DeepMind.

Eric Price

UT Austin &amp; Microsoft Research ecprice@cs.utexas.edu

general [GJP + 24]. But the lower bound uses an adversarial, bizarre distribution p ( x ) based on oneway functions; actual image manifolds are likely much better behaved. Can we find an algorithm for provably efficient, robust posterior sampling for relatively nice distributions p ? That is the goal of this paper: we describe conditions on p under which efficient, robust posterior sampling is possible.

A close relative to diffusion model sampling is Langevin dynamics , which is a different method for sampling that uses an SDE involving the unsmoothed score s 0 . Unlike diffusion, Langevin dynamics is in general slow and not robust to errors in approximating the score. To be efficient, Langevin dynamics needs stronger conditions, like that p ( x ) is log-concave and that the score estimation error satisfies an MGF bound (meaning that large errors are exponentially unlikely).

However, Langevin dynamics adapts very well to posterior sampling: it works for posterior sampling under exactly the same conditions as it does for unconditional sampling. The difference from diffusion models is that the unsmoothed conditional score s 0 ( x | y ) can be computed from the unconditional score s 0 ( x ) and the explicit measurement model p ( y | x ) , while the smoothed conditional score (which diffusion needs) cannot be easily computed.

So the current state is: diffusion models are efficient and robust for unconditional sampling, but essentially always inaccurate or inefficient for posterior sampling. No algorithm for posterior sampling is efficient and robust in general. Langevin dynamics is efficient for log-concave distributions, but still not robust. Can we make a robust algorithm for this case?

Can we do posterior sampling with log-concave p ( x ) and L p -accurate scores?

## 1.1 Our Results

Our first result answers this in the affirmative. Algorithm 1 uses a diffusion model for initialization, followed by an annealed version of Langevin dynamics, to do posterior sampling for log-concave p ( x ) with just L 4 -accurate scores. Annealing is necessary here; see Section F for why standard Langevin dynamics would not suffice in this setting.

Assumption 1 ( L 4 score accuracy) . The score estimates ̂ s σ 2 ( x ) of the smoothed distributions p σ 2 ( x ) = p ( x ) ∗ N (0 , σ 2 I d ) have finite L 4 error, i.e.,

<!-- formula-not-decoded -->

Theorem 1.1 (Posterior sampling with global log-concavity) . Let p ( x ) be an α -strongly logconcave distribution over R d with L -Lipschitz score. For any 0 &lt; ε &lt; 1 , there exist K 1 = poly( d, m, ∥ A ∥ η √ α , 1 ε ) and K 2 = poly( d, m, ∥ A ∥ η √ α , 1 ε , L α ) such that: if ε score ≤ √ α K 1 , then there exists an algorithm that takes K 2 iterations to sample from a distribution ̂ p ( x | y ) with

<!-- formula-not-decoded -->

For precise bounds on the polynomials, see Theorem E.6. To understand the parameters, ∥ A ∥ η √ α should be viewed as the signal-to-noise ratio of the measurement.

Local log-concavity. Global log-concavity, as required by Theorem 1.1, is simple to state but a fairly strong condition. In fact, Algorithm 1 only needs a local log-concavity condition.

As motivation, consider MRI reconstruction. Given the MRI measurement y of x , we would like to get as accurate an estimate ̂ x of x as possible. We expect the image distribution p ( x ) to concentrate around a low-dimensional manifold. We also know that existing compressed sensing methods (e.g., the LASSO [Tib96, CRT06]) can give a fairly accurate reconstruction x 0 ; not as accurate as we are hoping to achieve with the full power of our diffusion model for p ( x ) , but still pretty good. Then conditioned on x 0 , we know basically where x lies on the manifold; if the manifold is well behaved, we only really need to do posterior sampling on a single branch of the manifold. The posterior distribution on this branch can be log-concave even when the overall p ( x ) is not.

In the theorem below, we suppose we are given a Gaussian measurement x 0 = x + N (0 , σ 2 I d ) for some σ , and that the distribution p is nearly log-concave in a ball polynomially larger than σ . We can then converge to p ( x | x 0 , y ) .

(a) Density of p , the uniform distribution over the unit circle (white), convolved with N (0 , w 2 I 2 ) .

<!-- image -->

(b) λ max ( ∇ 2 log p ( x )) reaches Ω(1 /w 4 ) near the center, demonstrating strong non-log-concavity.

<!-- image -->

Figure 1: A 'locally nearly log-concave' distribution suitable for Theorem 1.2: uniform on the unit circle plus N (0 , w 2 I 2 ) . The Hessian's largest eigenvalue is much smaller near the bulk of the density than it is globally. Specifically, for ∥ A ∥ w/η = O (1) , a Gaussian measurement ˜ x with σ ≤ cw and ε score ≤ cw -1 for small enough c &gt; 0 enables sampling from p ( x | y, ˜ x ) .

Theorem 1.2 (Posterior sampling with local log-concavity) . For any ε, τ, R, L &gt; 0 , suppose p ( x ) is a distribution over R d such that

<!-- formula-not-decoded -->

Then, there exist K 1 , K 2 = poly( d, m, ∥ A ∥ σ η , 1 ε ) and K 3 = poly( d, m, ∥ A ∥ σ η , 1 ε , Lσ 2 ) such that: Given a Gaussian measurement x 0 = x + N (0 , σ 2 I d ) of x ∼ p with σ ≤ R/ ( K 1 + 2 τ ) . If ε score ≤ 1 K 2 σ , then there exists an algorithm that takes K 3 iterations to sample from a distribution ̂ p ( x | x 0 , y ) such that

<!-- formula-not-decoded -->

If p is globally log-concave, we can set σ = ∞ so x 0 is independent of x and recover Theorem 1.1; but if we have local information then this just needs local log-concavity. For precise bounds and a detailed discussion of the algorithm, see Section E.2.

The largest eigenvalue of ∇ 2 log p ( x ) quantifies the extent to which the distribution departs from log-concavity at a given point. In Figure 1, we show an instance of a locally nearly log-concave distribution: x is uniformly on the unit circle plus N (0 , w 2 I 2 ) . This distribution is very far from globally log-concave, but it is nearly log-concave within a w -width band of the unit circle. See Section E.4 for details.

Compressed Sensing. In compressed sensing, one would like to estimate x as accurately as possible from y . There are many algorithms under many different structural assumptions on x , most

Table 1: Summary of theorems and corresponding algorithms.

| Theorem       | Setting                                                     | Method                                                                           | Target                 |
|---------------|-------------------------------------------------------------|----------------------------------------------------------------------------------|------------------------|
| Theorem 1.1   | Global log-concavity                                        | Algorithm 1                                                                      | p ( x &#124; y )       |
| Theorem 1.2   | Local log-concavity with a Gaussian measurement x 0         | Run Algorithm 1 using p ( x &#124; x 0 ) as the prior (Algorithm 2)              | p ( x &#124; x 0 , y ) |
| Corollary 1.3 | Local log-concavity with an arbitrary noisy measurement x 0 | Run Algorithm 2 but replace x 0 with x ′ 0 = x 0 + N (0 ,σ 2 I d ) (Algorithm 3) | small ∥ x - x 0 ∥      |

Figure 2: Corollary 1.3 sampling process. Given the distribution p ( x ) and measurement y , we (1) start with a warm start estimate x 0 , which may not lie on the effective manifold containing p ( x ) ; (2) use the diffusion process to sample from p ( x ) in a ball around x 0 , getting x 1 on the manifold but not matching y ; and finally (3) use annealed Langevin dynamics to converge to p ( x | y ) . This works if p ( x ) is locally close to log-concave, even if it is globally complicated. See Section E.3 for a more detailed discussion.

<!-- image -->

notably the LASSO if x is known to be approximately sparse [Tib96, CRT06]. The LASSO does not use much information about the structure of p ( x ) , and one can hope for significant improvements when p ( x ) is known. Posterior sampling is known to be near-optimal for compressed sensing: if any algorithm achieves r error with probability 1 -δ , then posterior sampling achieves at most 2 r error with probability 1 -2 δ . But, as we discuss above, posterior sampling cannot be efficiently computed in general.

We can use Theorem 1.2 to construct a competitive compressed sensing algorithm under a 'local' log-concavity condition on p . Suppose we have a naive compressed sensing algorithm (e.g., the LASSO) that recovers the true x to within R error; and p is usually log-concave within an R · poly ball; then if any exponential time algorithm can get r error from y , our algorithm gets 2 r error in polynomial time.

Corollary 1.3 (Competitive compressed sensing) . Consider attempting to accurately reconstruct x from y = Ax + ξ . Suppose that:

- Information theoretically (but possibly requiring exponential time or using exact knowledge of p ( x ) ), it is possible to recover ̂ x from y satisfying ∥ ̂ x -x ∥ ≤ r with probability 1 -δ over x ∼ p and y .
- We have access to a 'naive' algorithm that recovers x 0 from y satisfying ∥ x 0 -x ∥ ≤ R with probability 1 -δ over x ∼ p and y .

<!-- formula-not-decoded -->

Then we give an algorithm that recovers ̂ x satisfying ∥ ̂ x -x ∥ ≤ 2 r with probability 1 -O ( δ ) , in poly( d, m, ∥ A ∥ R η , 1 δ ) time, under Assumption 1 with ε score &lt; 1 poly( d,m, ∥ A ∥ R η , 1 δ ,LR 2 ) R .

That is, we can go from a decent warm start to a near-optimal reconstruction, so long as the distribution is locally log-concave, with radius of locality depending on how accurate our warm start is. To our knowledge this is the first known guarantee of this kind. Per the lower bound [GJP + 24], such a guarantee would be impossible without any warm start or other assumption.

Figure 2 illustrates the sampling process of Corollary 1.3. The initial estimate x 0 may lie well outside the bulk of p ( x ) ; with just an L 4 error bound, the unsmoothed score at x 0 could be extremely bad. We add a bit of spherical Gaussian noise to x 0 , then treat this as a spherical Gaussian measurement of x , i.e., x + N (0 , RI ) ; for spherical Gaussian measurements, the posterior p ( x | x 0 ) can be sampled robustly and efficiently using the diffusion SDE. We take such a sample x 1 , which now won't be too far outside the distribution of p ( x ) , then use x 1 as initialization for annealed Langevin dynamics to sample from p ( x | y ) . The key part of our paper is that this process will never evaluate a score with respect to a distribution far from the distribution it was trained on, so the process is robust to error in the score estimates.

## Algorithm 1 Sampling from p ( x | Ax + N (0 , η 2 I ) = y )

<!-- formula-not-decoded -->

We summarize our results in Table 1.

## 2 Notation and Background

We consider x ∼ p ( x ) over R d . The 'score function' s ( x ) of p is ∇ log p ( x ) . The 'smoothed score function' s σ 2 ( x ) is the score of p σ 2 ( x ) = p ( x ) ∗ N (0 , σ 2 I d ) .

Unconditional sampling. There are several ways to sample from p using the scores. Langevin dynamics is a classical MCMC method that considers the following overdamped Langevin Stochastic Differential Equation (SDE):

<!-- formula-not-decoded -->

where B t is standard Brownian motion. The stationary distribution of this SDE is p , and discretized versions of it, such as the Unadjusted Langevin Algorithm (ULA), are known to converge rapidly to p ( x ) when p ( x ) is strongly log-concave [Dal17]. One can replace the true score s ( x ) with an approximation ̂ s , as long as it satisfies a (fairly strong) MGF condition

<!-- formula-not-decoded -->

In particular, [YW22] showed that Langevin dynamics needs an MGF bound for convergence, and an L p -accurate score estimator for any 1 ≤ p &lt; ∞ is insufficient.

An alternative approach, used by diffusion models, is to involve the smoothed scores. Starting from x 0 ∼ N (0 , I d ) , one can follow a different SDE [And82]:

<!-- formula-not-decoded -->

for a particular smoothing schedule σ t ; the result x T is exponentially close (in T ) to being drawn from p ( x ) . This also has efficient discretizations [CCL + 22, CCSW22, BBDD24], does not require log-concavity, and only requires an L 2 guarantee such as [CCL + 22]

<!-- formula-not-decoded -->

to accurately sample from p ( x ) . One can also run a similar ODE with similar guarantees but faster [CCL + 23].

Posterior sampling. Now, in this paper we are concerned with posterior sampling : we observe a noisy linear measurement y ∈ R m of x , given by

<!-- formula-not-decoded -->

and want to sample from p ( x | y ) . The unsmoothed score s y ( x ) := ∇ x log p ( x | y ) is easily computed by Bayes' rule:

<!-- formula-not-decoded -->

Thus we can run the Langevin SDE (3) with the same properties: if p ( x | y ) is strongly log-concave and the score estimate satisfies the MGF error bound (4), it will converge quickly and accurately.

Naturally, researchers have looked to diffusion processes for more general and robust posterior sampling methods. The main difficulty is that the smoothed score of the posterior involves ∇ x log p ( y | x σ 2 t ) rather than the tractable unsmoothed term ∇ x log p ( y | x ) . Because the smoothed score is hard to evaluate exactly, a range of approximation techniques has been proposed [BGP + 24, CKM + 23, MK25, RCK + 24, SVMK23, WYZ23]. One prominent example is the DPS algorithm [CKM + 23]. Other methods include Monte Carlo/MCMC-inspired approximations [CJeILCM24, DS24, WSC + 24, EKZL25], singular value decomposition and transport tilting [KVE21, KSEE22, WYZ23, BH24], and schemes that combine corrector steps with standard diffusion updates [CL23, CY22, CSRY22, KBBW23, LKA + 24, SKZ + 24, SSXE22, ZZL + 23, AVTT21, XC24, RLdB + 24, RRD + 23]. These approaches have shown strong empirical performance, and several provide guarantees under additional structure of the linear measurement; however, general guarantees for fast and robust posterior sampling remain limited beyond these restricted regimes.

Several recent studies [JAD + 21, ZCB + 25, KVE21] use various annealed versions of the Langevin SDE as a key component in their diffusion-based posterior sampling method and achieve strong empirical results. Still, these methods provide no theoretical guidance on two key aspects: how to design the annealing schedule and why annealing improves robustness. None of these approaches come with correctness guarantees for the overall sampling procedure.

Comparison with Computational Lower Bounds. Recent work of [GJP + 24] shows that it is actually impossible to achieve a general algorithm that is guaranteed fast and robust: there is an exponential computational gap between unconditional diffusion and posterior sampling. Under standard cryptographic assumptions, they construct a distribution p over R d such that

1. One can efficiently obtain an L p -accurate estimate of the smoothed score of p , so diffusion models can sample from p .
2. Any sub-exponential time algorithm that takes y = Ax + N (0 , η 2 I m ) as input and outputs a sample from the posterior p ( x | y ) fails on most y with high probability.

Our algorithm shows that, once an additional noisy observation ˜ x that is close to x is provided, then we can efficiently sample from p ( x | y, ˜ x ) , circumventing the impossibility result.

To illustrate why the extra observation helps, consider the following simplified version of the hardness instance:

<!-- formula-not-decoded -->

Here, f : { 0 , 1 } d/ 2 →{ 0 , 1 } d/ 2 is a one-way permutation - it takes exponential time to compute f -1 ( x ) for most x ∈ { 0 , 1 } d/ 2 . δ ( · ) is the Dirac delta function, and we choose σ ≪ d -1 / 2 . Thus, p ( x ) is a mixture of 2 d/ 2 well-separated Gaussians centered at the points ( s, f ( s )) .

Assume we observe

<!-- formula-not-decoded -->

and let rnd( y ) denote the vertex of { 0 , 1 } d closest to y . Then the posterior p ( x | y ) is approximately a Gaussian centered at ( f -1 (rnd( y )) , rnd( y )) with covariance σ 2 I d . Generating a single sample would therefore reveal f -1 (rnd( y )) , which requires exp(Ω( d )) time.

However, suppose we have a coarse estimate x 0 satisfying ∥ x 0 -x ∥ &lt; 1 / 3 (e.g., obtained by compressed sensing). Then, x 0 uniquely identifies the correct ( s, f ( s )) with f ( s ) = rnd( y ) , and the remaining task is just sampling from a Gaussian. Therefore, this hard instance becomes easy once we have localized the task and does not contradict our Theorem 1.2.

We are able to handle the hard instance above well because it is exactly the type of distribution our approach is designed for: despite its complex global structure, it exhibits well-behaved local properties. This gives an important conceptual takeaway from our work: the hardness of posterior sampling may only lie in localizing x within the exponentially large high-dimensional space.

Therefore, although posterior sampling is an intractable task in general, it is still possible to design a robust, provably correct posterior sampling algorithm - once we have localized the distribution. We view our work as a first step towards this goal.

## 3 Techniques

The algorithm we propose is clean and simple, but the proof is quite involved. Before we dive into the details, we provide a high-level overview of the intuitions behind the algorithm, concentrating on the illustrative case where the prior density p ( x ) is α -strongly log-concave. Under this assumption, every posterior density p ( x | y ) is also α -strongly log-concave. Therefore, posterior sampling could, in principle, be performed using classical Langevin dynamics.

The challenge arises because we lack access to the exact posterior score s y ( x ) . We only possess an estimator derived from an estimate ̂ s ( x ) of the prior score s ( x ) :

<!-- formula-not-decoded -->

Assumption 1 implies an L 4 accuracy of ̂ s y on average, but how do we use this to support Langevin dynamics, which demands exponentially decaying error tails?

## 3.1 Score Accuracy: Langevin Dynamics vs. Diffusion Models

Why can diffusion models succeed with merely L 2 -accurate scores, whereas Langevin dynamics require MGF accuracy?

Both diffusion models and Langevin dynamics utilize SDEs. The L 2 error in the score-dependent drift term relates directly to the KL divergence between the true process (using s ( x ) ) and the estimated process (using ̂ s ( x ) ). Consequently, bounding the L 2 score error with respect to the current distribution ̂ p t controls the KL divergence.

Diffusion models leverage this property effectively. The forward process transforms data into a Gaussian, and the reverse generative process starts exactly from this Gaussian. At any time t , suppose ̂ p t is close to p σ 2 t , then

<!-- formula-not-decoded -->

by the L 2 accuracy assumption. This keeps the process close to the ideal process, ensuring overall small error.

Langevin dynamics, by contrast, often starts from an arbitrary, not predefined initial distribution p initial. An L p score accuracy guarantee with respect to p target alone does not ensure accuracy for points x t that are not on the distributional manifold of p target (consider running Langevin starting from x 0 in Figure 2). Therefore, a stronger MGF error bound is needed to prevent this from happening.

## 3.2 Adapting Langevin Dynamics for Posterior Sampling

While we can only use Langevin-type dynamics for posterior sampling, we possess a source of effective starting points: we can sample x 0 ∼ p ( x ) efficiently using the unconditional diffusion model. Intuitively, x 0 already lies on the data manifold. The score estimator ̂ s y ( x ) initially satisfies:

<!-- formula-not-decoded -->

Figure 3: Let p = N (0 , 1) and y = x + N (0 , 0 . 01) . Starting from X 0 ∼ p , run the Langevin SDE d X t = s y ( X t ) d t + √ 2 d B t . Averaging over y , the marginal of X t remains Gaussian; its variance first contracts and then returns toward the prior. There is an intermediate time t ∗ where X t ∗ has a constant factor lower variance; in high dimensions, this means X t ∗ is concentrated on an exponentially small region of p , so an L p bound on score error under p does not effectively control the error under X t ∗ . See Section F for details.

<!-- image -->

As the dynamics evolves, the distribution p ( x t ) transitions from p ( x ) towards p ( x | y ) . If x t converges to p ( x | y ) , we again expect reasonable accuracy on average:

<!-- formula-not-decoded -->

Hence the estimator is accurate at the start and at convergence. The open question concerns the intermediate segment of the trajectory: does x t wander into regions where the prior score ̂ s ( x ) is unreliable? Ideally, the time-marginal of x t , averaged over y , remains close to p ( x ) throughout.

## 3.3 Annealing via Mixing Steps

In fact, even though x 0 and x ∞ both have marginal p ( x ) , so the score estimate ̂ s ( x ) is accurate on average at those times, this is not true at intermediate times. In Figure 3, we illustrate this with a simple Gaussian example: x 0 and x ∞ have distribution N (0 , I ) while x t has marginal N (0 , cI ) for a constant c &lt; 1 . An L p error bound under x ∼ N (0 , I ) does not give an L 2 error bound under x ∼ N (0 , cI ) , which means Langevin dynamics may not converge to the right distribution. A very strong accuracy guarantee like the MGF bound is needed here.

However, consider the case where the target posterior p ( x | y ) is very close to the initial prior p ( x ) , such as when the measurement noise η is very large (low signal-to-noise ratio). Langevin dynamics between close distributions typically converges rapidly. This suggests a key insight: if the required convergence time T is short, the process x t might not deviate substantially from its initial distribution p ( x 0 ) . In such short-time regimes, an L 2 score error bound relative to p ( x 0 ) could potentially suffice to control the dynamics. While p ( x ) itself is already a good approximation for p ( x | y ) when η is very large, this motivates a general strategy.

Instead of a single, potentially long Langevin run from p ( x ) to p ( x | y ) , we introduce an annealing scheme using multiple mixing steps . Given the measurement parameters ( A,η, y ) , we construct a decreasing noise schedule η 1 &gt; η 2 &gt; · · · &gt; η N = η . Correspondingly, we generate a sequence of auxiliary measurements y 1 , y 2 , . . . , y N = y such that each y i is distributed as Ax + N (0 , η 2 i I m ) and y i is appropriately coupled to y i +1 (specifically, y i ∼ N ( y i +1 , ( η 2 i -η 2 i +1 ) I m ) conditional on y i +1 ). This creates a sequence of intermediate posterior distributions p ( x | y i ) .

An admissible schedule (formally defined in Definition D.1) ensures that:

- η 1 is sufficiently large, making p ( x | y 1 ) close to the prior p ( x ) .
- Consecutive η i and η i +1 are sufficiently close, making p ( x | y i ) close to p ( x | y i +1 ) .

Our algorithm proceeds as follows:

Figure 4: For each of the three settings (inpainting, super-resolution, and Gaussian deblurring), we plot the L 2 distance between samples obtained by our annealed Langevin method and the ground truth samples in red. We plot the FID of the distribution obtained by running annealed Langevin in blue. We plot the baseline L 2 distance and FID for samples obtained by the DPS algorithm using red and blue dashed lines.

<!-- image -->

1. Start with a sample X 0 ∼ p ( x ) . Since η 1 is large, p ( x ) is close to p ( x | y 1 ) , so X 0 serves as an approximate sample X 1 ∼ ̂ p ( x | y 1 ) .
2. For i = 1 to N -1 : Run Langevin dynamics for a short time T i , starting from the previous sample X i ∼ ̂ p ( x | y i ) , targeting the next posterior p ( x | y i +1 ) using the score ̂ s y i +1 ( x ) . Let the result be X i +1 ∼ ̂ p ( x | y i +1 ) .
3. The final sample X N ∼ ̂ p ( x | y N ) approximates a draw from the target posterior p ( x | y ) .

The core idea behind this annealing scheme is to actively control the process distribution p ( x t ) , ensuring it remains on the manifold of the prior p ( x ) . By design, each mixing step i → i + 1 connects two statistically close intermediate posteriors, p ( x | y i ) and p ( x | y i +1 ) . This closeness guarantees that a short Langevin run T i can mix them, and this short duration prevents p ( x t ) from drifting significantly away from the step's starting distribution ̂ p ( x | y i ) , and we can then argue that

<!-- formula-not-decoded -->

This contrasts fundamentally with a single long Langevin run, where x t could venture far "offmanifold" into regions of poor score accuracy. By inserting frequent checkpoints that re-anchor the process, our annealing method substitutes such strong assumptions with structural control: the frequent 'checkpoints' p ( x | y i ) ensure the process is repeatedly localized to regions where the L 4 accuracy suffices. While error is incurred in each step, maintaining proximity to the manifold keeps this error small. The overall approach hinges on demonstrating that these small, per-step errors accumulate controllably across all N steps.

This strategy, however, requires rigorous analysis of three key technical challenges:

1. How to bound the required convergence time T i for the transition from p ( x | y i ) to p ( x | y i +1 ) ? In particular, what happens when p only has local strong log-concavity?
2. How to bound the error incurred during a single mixing step of duration T i , given the L 4 score error assumption on the prior score estimate?
3. How to ensure the total error accumulated across all N mixing steps remains small?

Addressing these questions forms the core of our proof.

Proof Organization. In Section A, we show that for globally strongly log-concave distributions p , Langevin dynamics converges rapidly from p ( x | y i ) to p ( x | y i +1 ) . We extend this convergence analysis to locally strongly log-concave distributions in Section B. In Section C, we provide bounds on the errors incurred by score errors and discretization in Langevin dynamics. In Section D, we show how to design the noise schedule to control the accumulated error of the full process. In Section E, we conclude the analysis for Algorithm 1, and apply it to establish the main theorems.

<!-- image -->

(a) Input

<!-- image -->

(a) Input

<!-- image -->

(b) DPS

(c) Ours

<!-- image -->

(d) Ground Truth

<!-- image -->

Figure 5: A set of samples for the inpainting task.

<!-- image -->

(b) DPS

(c) Ours

<!-- image -->

(d) Ground Truth

<!-- image -->

Figure 6: A set of samples for the super-resolution task.

## 4 Experiments

To validate our theoretical analysis and assess real-world performance, we study three inverse problems on FFHQ256 [KLA21]: inpainting, 4 × super-resolution, and Gaussian deblurring. Experiments use 1k validation images and the pre-trained diffusion model from [CKM + 23]. Forward operators are specified as in [CKM + 23]: inpainting masks 30% -70% of pixels uniformly at random; super-resolution downsamples by a factor of 4 ; deblurring convolves the ground-truth with a Gaussian kernel of size 61 × 61 (std. 3 . 0 ). We first obtain initial reconstructions x 0 via Diffusion Posterior Sampling (DPS) [DS24], then refine them with our annealed Langevin sampler to draw samples close to p ( x | x 0 , y ) . To control runtime, we sweep the step size while keeping the annealing schedule fixed.

For each step size, we report the per-image L 2 distance to the ground truth and the FID of the resulting sample distribution (Figure 4). Across all three tasks, increasing the time devoted to annealed Langevin decreases L 2 but increases FID; in the inpainting setting, when the step size is sufficiently small, our method surpasses DPS on both metrics. Qualitatively, our reconstructions better preserve ground-truth attributes compared to DPS (Figures 5 and 6). All experiments were run on a cluster with four NVIDIA A100 GPUs and required roughly two hours per task.

## Acknowledgments

This work is supported by the NSF AI Institute for Foundations of Machine Learning (IFML). ZX is supported by NSF Grant CCF-2312573 and a Simons Investigator Award (#409864, David Zuckerman).

## Bibliography

[And82] Brian DO Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications , 12(3):313-326, 1982.

[AVTT21] Marius Arvinte, Sriram Vishwanath, Ahmed H. Tewfik, and Jonathan I. Tamir. Deep j-sense: Accelerated mri reconstruction via unrolled alternating optimization. In Medical Image Computing and Computer-Assisted Intervention (MICCAI 2021),

Part VI , volume 12905 of Lecture Notes in Computer Science , pages 350-360. Springer, 2021.

- [BBDD24] Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly $d$-linear convergence bounds for diffusion models via stochastic localization. In The Twelfth International Conference on Learning Representations , 2024.
- [BGP + 24] Benjamin Boys, Mark Girolami, Jakiw Pidstrigach, Sebastian Reich, Alan Mosca, and Omer Deniz Akyildiz. Tweedie moment projected diffusions for inverse problems. Transactions on Machine Learning Research , 2024. TMLR (ICLR 2025 Journal Track).
- [BH24] Joan Bruna and Jiequn Han. Provable posterior sampling with denoising oracles via tilted transport. In Advances in Neural Information Processing Systems (NeurIPS) , 2024.
- [CCL + 22] Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru R Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. arXiv preprint arXiv:2209.11215 , 2022.
- [CCL + 23] Sitan Chen, Sinho Chewi, Holden Lee, Yuanzhi Li, Jianfeng Lu, and Adil Salim. The probability flow ODE is provably fast. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems , volume 36, pages 68552-68575. Curran Associates, Inc., 2023.
- [CCSW22] Yongxin Chen, Sinho Chewi, Adil Salim, and Andre Wibisono. Improved analysis for a proximal algorithm for sampling. In Conference on Learning Theory , pages 2984-3014. PMLR, 2022.
- [CJeILCM24] Gabriel Cardoso, Yazid Janati el Idrissi, Sylvain Le Corff, and Eric Moulines. Monte carlo guided denoising diffusion models for bayesian linear inverse problems. In International Conference on Learning Representations (ICLR) , 2024. Oral.
- [CKM + 23] Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In The Eleventh International Conference on Learning Representations , 2023.
- [CL23] Junqing Chen and Haibo Liu. An alternating direction method of multipliers for inverse lithography problem. Numerical Mathematics: Theory, Methods and Applications , 16(3):820-846, 2023.
- [CRT06] Emmanuel J Candes, Justin K Romberg, and Terence Tao. Stable signal recovery from incomplete and inaccurate measurements. Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences , 59(8):1207-1223, 2006.
- [CSRY22] Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. In Advances in Neural Information Processing Systems (NeurIPS) , 2022.
- [CY22] Hyungjin Chung and Jong Chul Ye. Score-based diffusion models for accelerated mri. Medical Image Analysis , 80:102479, 2022.
- [Dal17] Arnak S Dalalyan. Theoretical guarantees for approximate sampling from smooth and log-concave densities. Journal of the Royal Statistical Society Series B: Statistical Methodology , 79(3):651-676, 2017.
- [DS24] Zehao Dou and Yang Song. Diffusion posterior sampling for linear inverse problem solving: A filtering perspective. In The Twelfth International Conference on Learning Representations , 2024.
- [EKZL25] Filip Ekström Kelvinius, Zheng Zhao, and Fredrik Lindsten. Solving linear-gaussian bayesian inverse problems with decoupled diffusion sequential monte carlo. In Proceedings of the 42nd International Conference on Machine Learning (ICML) , volume 267 of Proceedings of Machine Learning Research , pages 15148-15181, 2025.

- [GJP + 24] Shivam Gupta, Ajil Jalal, Aditya Parulekar, Eric Price, and Zhiyang Xun. Diffusion posterior sampling is computationally intractable. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 1702017059. PMLR, 21-27 Jul 2024.
- [HD05] Aapo Hyvärinen and Peter Dayan. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research , 6(4), 2005.
- [HJA20] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems , 33:6840-6851, 2020.
- [JAD + 21] Ajil Jalal, Marius Arvinte, Giannis Daras, Eric Price, Alexandros G Dimakis, and Jon Tamir. Robust compressed sensing mri with deep generative priors. Advances in Neural Information Processing Systems , 34:14938-14954, 2021.
- [JCP24] Yiheng Jiang, Sinho Chewi, and Aram-Alexandre Pooladian. Algorithms for meanfield variational inference via polyhedral optimization in the Wasserstein space. In Shipra Agrawal and Aaron Roth, editors, Proceedings of Thirty Seventh Conference on Learning Theory , volume 247 of Proceedings of Machine Learning Research , pages 2720-2721. PMLR, 7 2024.
- [JKH + 21] Ajil Jalal, Sushrut Karmalkar, Jessica Hoffmann, Alex Dimakis, and Eric Price. Fairness for image generation with uncertain sensitive attributes. In International Conference on Machine Learning , pages 4721-4732. PMLR, 2021.
- [KBBW23] Ulugbek S. Kamilov, Charles A. Bouman, Gregery T. Buzzard, and Brendt Wohlberg. Plug-and-play methods for integrating physical and learned models in computational imaging: Theory, algorithms, and applications. IEEE Signal Processing Magazine , 40(1):85-97, 2023.
- [KLA21] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. IEEE Trans. Pattern Anal. Mach. Intell. , 43(12):4217-4228, December 2021.
- [KSEE22] Bahjat Kawar, Jiaming Song, Stefano Ermon, and Michael Elad. Denoising diffusion restoration models. In Advances in Neural Information Processing Systems (NeurIPS) , 2022.
- [KVE21] Bahjat Kawar, Gregory Vaksman, and Michael Elad. Snips: Solving noisy inverse problems stochastically. In Advances in Neural Information Processing Systems (NeurIPS) , volume 34, pages 21757-21769, 2021.
- [LKA + 24] Xiang Li, Soo Min Kwon, Ismail R. Alkhouri, Saiprasad Ravishankar, and Qing Qu. Decoupled data consistency with diffusion purification for image restoration. arXiv preprint arXiv:2403.06054 , 2024.
- [LM00] B. Laurent and Pascal Massart. Adaptive estimation of a quadratic functional by model selection. Annals of Statistics , 28, 10 2000.
- [MK25] Xiangming Meng and Yoshiyuki Kabashima. Diffusion model based posterior sampling for noisy linear inverse problems. In Proceedings of the 16th Asian Conference on Machine Learning (ACML) , volume 260 of Proceedings of Machine Learning Research , pages 623-638. PMLR, 2025.
- [RCK + 24] Litu Rout, Yujia Chen, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Beyond first-order tweedie: Solving inverse problems using latent diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 9472-9481, 2024.

- [RLdB + 24] Marien Renaud, Jiaming Liu, Valentin de Bortoli, Andrés Almansa, and Ulugbek S. Kamilov. Plug-and-play posterior sampling under mismatched measurement and prior models. In International Conference on Learning Representations (ICLR) , 2024.
- [RRD + 23] Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alexandros G. Dimakis, and Sanjay Shakkottai. Solving linear inverse problems provably via posterior sampling with latent diffusion models. arXiv preprint arXiv:2307.00619 , 2023.
- [SE19] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems (NeurIPS) , volume 32, 2019.
- [SKZ + 24] Bowen Song, Soo Min Kwon, Zecheng Zhang, Xinyu Hu, Qing Qu, and Liyue Shen. Solving inverse problems with latent diffusion models via hard data consistency. In International Conference on Learning Representations (ICLR) , 2024.
- [SME20] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 , 2020.
- [SSDK + 21] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR) , 2021.
- [SSXE22] Yang Song, Liyue Shen, Lei Xing, and Stefano Ermon. Solving inverse problems in medical imaging with score-based generative models. In International Conference on Learning Representations (ICLR) , 2022.
- [SVMK23] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverseguided diffusion models for inverse problems. In International Conference on Learning Representations (ICLR) , 2023.
- [Tib96] Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology , 58(1):267-288, 1996.
- [WSC + 24] Zihui Wu, Yu Sun, Yifan Chen, Bingliang Zhang, Yisong Yue, and Katherine Bouman. Principled probabilistic imaging using diffusion models as plug-and-play priors. In Advances in Neural Information Processing Systems (NeurIPS) , 2024.
- [WTN + 23] Luhuan Wu, Brian Trippe, Christian Naesseth, David Blei, and John P Cunningham. Practical and asymptotically exact conditional sampling in diffusion models. Advances in Neural Information Processing Systems , 36:31372-31403, 2023.
- [WYZ23] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. In International Conference on Learning Representations (ICLR) , 2023.
- [XC24] Xingyu Xu and Yuejie Chi. Provably robust score-based diffusion posterior sampling for plug-and-play image reconstruction. In Advances in Neural Information Processing Systems (NeurIPS) , 2024.
- [YW22] Kaylee Yingxi Yang and Andre Wibisono. Convergence in kl and rényi divergence of the unadjusted langevin algorithm using estimated score. In NeurIPS 2022 Workshop on Score-Based Methods , 2022.
- [ZCB + 25] Bingliang Zhang, Wenda Chu, Julius Berner, Chenlin Meng, Anima Anandkumar, and Yang Song. Improving diffusion inverse problem solving with decoupled noise annealing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , 2025.
- [ZZL + 23] Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 1219-1229, 2023.

## A Langevin Convergence Between Strongly Log-concave Distributions

In this section, we study the following problem. Let p be a probability distribution on R d , and let A ∈ R m × d be a matrix. For a sequence of parameters η i &gt; η i +1 satisfying

<!-- formula-not-decoded -->

consider two random variables y i and y i +1 defined as follows. First, draw x ∼ p . Then, generate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and further perturb it by

Define the score function

We analyze the following SDE:

<!-- formula-not-decoded -->

This is the ideal (no discretization, no score estimation error) version of the process (2) that we actually run. Our goal is to establish the following lemma.

Lemma A.1. Suppose the prior distribution p ( x ) is α -strongly log-concave. Then, running the process (6) for time

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ensures that

## A.1 χ 2 -divergence Between Distributions

In this section, our goal is to bound χ 2 ( p ( x | y i ) ∥ p ( x | y i +1 )) . Since the posterior distributions can be expressed as

<!-- formula-not-decoded -->

The χ 2 divergence is

<!-- formula-not-decoded -->

We bound the term E x ∼ p ( x | y i ) [ p ( y i | x ) p ( y i +1 | x ) ] first.

Lemma A.2. We have

<!-- formula-not-decoded -->

Proof. Let Z 1 = y i +1 -Ax , and let Z 2 = y i -Ax . Then we have

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

where f is the density function for N (0 , ( η 2 i -η 2 i +1 ) I m ) . Therefore,

<!-- formula-not-decoded -->

Since f is a density function, its integral over R m is 1 . This gives that

Hence,

<!-- formula-not-decoded -->

Corollary A.3. For any λ &gt; 1 , we have

<!-- formula-not-decoded -->

Proof. By Lemma A.2, we have

<!-- formula-not-decoded -->

Applying Markov's inequality gives the result.

Now we bound p ( y i +1 ) p ( y i ) . To make the lemma more self-contained, we abstract this a little bit.

Lemma A.4. Let η 1 &gt; η 2 be two positive numbers, and let X ∈ R d be an arbitrary random variable. Define Y 1 = X + Z 1 and Y 2 = Y 1 + Z 2 , where Z 1 ∼ N (0 , η 2 1 I d ) and Z 2 ∼ N (0 , η 2 2 I d ) . Then,

<!-- formula-not-decoded -->

where p ( Y 1 ) and p ( Y 2 ) are the densities of Y 1 and Y 2 , respectively.

Proof. First, we turn to bound

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

We have

<!-- formula-not-decoded -->

Write Y 1 -s = e 1 , and note that Y 2 -s = e 1 + Z 2 . Then define

<!-- formula-not-decoded -->

This gives that for any Y 1 , Y 2 , and t ,

<!-- formula-not-decoded -->

Bounding G ( e 1 ) To bound sup ∥ e 1 ∥≤ t G ( e 1 ) , we expand ϕ as the d -dimensional Gaussian probability density function:

<!-- formula-not-decoded -->

Using the quadratic expansion ∥ e 1 + Z 2 ∥ 2 = ∥ e 1 ∥ 2 +2 ⟨ e 1 , Z 2 ⟩ + ∥ Z 2 ∥ 2 , we rewrite:

<!-- formula-not-decoded -->

Since ∥ e 1 ∥≤ t and ⟨ e 1 , Z 2 ⟩ ≤ ∥ e 1 ∥∥ Z 2 ∥ , we bound

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Therefore, for any Y 1 , Y 2 , and t , we have

<!-- formula-not-decoded -->

This gives that

<!-- formula-not-decoded -->

Bounding expectation over Z 2 . We have

<!-- formula-not-decoded -->

We can apply results on the Gaussian moment generating functions to bound this. Using Lemma A.10 by setting α = η 2 2 2( η 2 1 + η 2 2 ) , β = tη 2 η 2 1 + η 2 2 , and γ = η 2 1 4( η 2 1 + η 2 2 ) , we have

<!-- formula-not-decoded -->

Finally, this gives

<!-- formula-not-decoded -->

One need to verify that

Also,

This gives the result.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma A.5. Let η 1 &gt; η 2 be two positive numbers, and let X ∈ R d be an arbitrary random variable. Define Y 1 = X + Z 1 and Y 2 = Y 1 + Z 2 , where Z 1 ∼ N (0 , η 2 1 I d ) and Z 2 ∼ N (0 , η 2 2 I d ) . There exists a constant C &gt; 0 such that for any λ &gt; 1 ,

<!-- formula-not-decoded -->

where p ( Y 1 ) and p ( Y 2 ) are the densities of Y 1 and Y 2 , respectively.

Proof. Let t = ( √ d + √ 2 ln(2 λ )) η 1 . By applying Laurent-Massart bounds (Lemma A.11), we have

<!-- formula-not-decoded -->

Taking these into Lemma A.4, we have

<!-- formula-not-decoded -->

By applying Markov's inequality, for a large enough constant C &gt; 0 , we have

<!-- formula-not-decoded -->

Cleaning up the bound a little bit, this implies that for a large enough constant C &gt; 0 ,

<!-- formula-not-decoded -->

Combining this with the probability that ∥ Z ∥ ≤ t , a union bound gives that

<!-- formula-not-decoded -->

The χ 2 divergence is

<!-- formula-not-decoded -->

Now we can bound the χ 2 -diversity.

Lemma A.6. There exists a constant C &gt; 0 such that for any λ &gt; 1 ,

<!-- formula-not-decoded -->

Proof. Note that

<!-- formula-not-decoded -->

By Corollary A.3, we have

<!-- formula-not-decoded -->

By Lemma A.5, there exists a constant C &gt; 0 such that

<!-- formula-not-decoded -->

A union bound over these two implies that with probability of 1 -1 /λ ,

<!-- formula-not-decoded -->

where C is a positive constant. This concludes the lemma.

## A.2 Convergence time of Langevin dynamics

We present the following result on the convergence of Langevin dynamics:

Lemma A.7 ([Dal17]) . Let p and q be probability distributions such that q is an α -strong logconcave distribution. Consider the Langevin dynamics initialized with p as the starting distribution. Then, for any t ≥ 0 , we have

<!-- formula-not-decoded -->

This implies that

Lemma A.8. Let p and q be probability distributions such that q is an α -strong log-concave distribution. Consider the Langevin dynamics initialized with p as the starting distribution. By running the diffusion for time

<!-- formula-not-decoded -->

we have TV( p T , q ) ≤ ε .

Now we show that the posterior distribution is even more strongly log-concave than prior distribution.

Lemma A.9. Suppose that p ( x ) is α -strongly log-concave. Then, the posterior density

<!-- formula-not-decoded -->

is α -strongly log-concave.

Proof. By Bayes' rule, the posterior density can be written (up to normalization) as

<!-- formula-not-decoded -->

Define the negative log-posterior

<!-- formula-not-decoded -->

Since p is α -strongly log-concave, its negative log-density satisfies

<!-- formula-not-decoded -->

Moreover, the Gaussian likelihood term has

<!-- formula-not-decoded -->

By the sum rule for Hessians,

<!-- formula-not-decoded -->

Hence φ is α -strongly convex, and the posterior density p ( x | Ax + N ( η 2 i I m ) = y i ) ∝ e -φ ( x ) is α -strongly log-concave.

Now we are ready to prove Lemma A.1:

Proof of Lemma A.1. By Lemma A.9, p ( x | y i +1 ) is alpha -strongly log-concave. This allows us to apply Lemma A.8. Therefore, to achieve ε TV error in convergence, we only need to run the process for

<!-- formula-not-decoded -->

Taking in the result in Lemma A.6, we have with 1 -1 λ probability over y i and y i +1 , we only need

<!-- formula-not-decoded -->

## A.3 Utility Lemmas.

Lemma A.10. Let Z ∼ N (0 , I d ) be a d -dimensional standard Gaussian random vector, and let α, β ∈ R . For any γ &gt; 0 satisfying α + γ &lt; 1 2 , we have

<!-- formula-not-decoded -->

Proof. For all r ≥ 0 and any γ &gt; 0 , it is easy to check that by AM-GM inequality,

<!-- formula-not-decoded -->

Taking r = ∥ Z ∥ and exponentiating both sides, we obtain

<!-- formula-not-decoded -->

Multiplying both sides by exp ( α ∥ Z ∥ 2 ) yields

<!-- formula-not-decoded -->

This gives that

<!-- formula-not-decoded -->

For Z ∼ N (0 , I d ) , when α + γ &lt; 1 2 we have

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Lemma A.11 (Laurent-Massart Bounds[LM00]) . Let v ∼ N (0 , I m ) . For any t &gt; 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B Convergence Between Locally Well-Conditioned Distributions

In the last section, we considered the convergence time between two posterior distributions of a globally strongly log-concave distribution. In this section, we will relax the assumption of global strong log-concavity and consider the convergence time between two distributions that are locally 'well-behaved'. We give the following formal definition:

Definition B.1. For δ ∈ [0 , 1) and R, ˜ L, α ∈ (0 , + ∞ ] , we say that a distribution p is ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned if there exists θ such that

- ∇ log p ( θ ) = 0 .
- Pr x ∼ p [ x ∈ B ( θ, r )] ≥ 1 -δ .
- For x, y ∈ B ( θ, R ) , we have that ∥ s ( x ) -s ( y ) ∥≤ ˜ Lα ∥ x -y ∥ .
- For x, y ∈ B ( θ, R ) , we have that ⟨ s ( y ) -s ( x ) , x -y ⟩ ≥ α ∥ x -y ∥ 2 .

Again, we consider the following process P , which is identical to process (6) we considered in the last section:

<!-- formula-not-decoded -->

Our goal is to prove the following lemma:

Lemma B.2. Suppose p is a ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned distribution. Let C &gt; 0 be a large enough constant. We consider the process P running for time

<!-- formula-not-decoded -->

Suppose that

<!-- formula-not-decoded -->

Then x T ∼ P T satisfies that

<!-- formula-not-decoded -->

In this section, we will assume that p is ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned. Without loss of generality, we assume that the mode of p is at 0, i.e., θ = 0 .

## B.1 High Probability Boundness of Langevin Dynamics

We consider the process P ′ defined as the process P conditioned on x t ∈ B (0 , R ) for t ∈ [0 , T ] . Our goal is to prove the following lemma:

Lemma B.3. Suppose the following holds:

<!-- formula-not-decoded -->

We have that

<!-- formula-not-decoded -->

We start by decomposing the total variation distance between P and P ′ as follows:

Lemma B.4. We have that

<!-- formula-not-decoded -->

Proof. Recall that the process P ′ is defined as the law of P conditioned on the event

<!-- formula-not-decoded -->

Thus, for any fixed y i we have

<!-- formula-not-decoded -->

where F c = {∃ t ∈ [0 , T ] : ∥ x t ∥≥ R } .

Let E := { x 0 ∈ B (0 , r ) } denote the event that the initial condition is 'good.' Then, by the law of total probability,

<!-- formula-not-decoded -->

Taking the expectation with respect to y i and y i +1 , we obtain

<!-- formula-not-decoded -->

Since

<!-- formula-not-decoded -->

and by the law of total probability, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

it follows that

<!-- formula-not-decoded -->

This completes the proof.

Now we focus on bounding E y i ,y i +1 [ Pr P [ ∃ t ∈ [0 , T ] : ∥ x t ∥≥ R ∣ ∣ ∣ x 0 ∈ B (0 , r ) ]] . We start by observing the following lemma for log-concave distributions.

Lemma B.5. Let p be a log-concave distribution such that p is continuously differentiable. Suppose the mode of p is at 0. Then, for all x ∈ R d ,

<!-- formula-not-decoded -->

Proof. Since log p is concave, for any x, θ ∈ R d the first-order condition for concavity yields

<!-- formula-not-decoded -->

Rearrange this inequality to obtain

<!-- formula-not-decoded -->

Because θ is a mode, log p ( θ ) ≥ log p ( x ) for every x ∈ R d ; hence,

<!-- formula-not-decoded -->

Lemma B.6. Let x t be the stochastic process

<!-- formula-not-decoded -->

where B t is a standard R d -valued Brownian motion and the functions f, g : R d → R d satisfy

<!-- formula-not-decoded -->

with a ≥ 0 . Then, for any time horizon T &gt; 0 and δ ∈ (0 , 1) ,

<!-- formula-not-decoded -->

̸

Proof. Define r ( t ) = ∥ x t ∥ . Although the Euclidean norm is not smooth at the origin, an application of Itô's formula yields that, for x t = 0 , one has

<!-- formula-not-decoded -->

where u ( t ) = x t / ∥ x t ∥ . Using the bound ∥ f ( x t ) ∥≤ a and the hypothesis ⟨ g ( x t ) , x t ⟩ ≤ 0 , it follows by the Cauchy-Schwarz inequality that

<!-- formula-not-decoded -->

Discarding the nonnegative Itô correction term d -1 ∥ x t ∥ dt (which can only increase the process), we deduce that √

<!-- formula-not-decoded -->

Introduce the one-dimensional process

<!-- formula-not-decoded -->

Since ∥ u ( s ) ∥ = 1 for all s , the process β ( t ) is a standard one-dimensional Brownian motion with quadratic variation ⟨ β ⟩ t = t . By a standard comparison theorem for one-dimensional stochastic differential equations, it follows that r ( t ) ≤ y ( t ) almost surely for all t ≥ 0 ; hence,

<!-- formula-not-decoded -->

A classical application of the reflection principle for one-dimensional Brownian motion shows that, for any ρ &gt; 0 ,

<!-- formula-not-decoded -->

To incorporate the d -dimensional nature of the noise, one may use a union bound over the d coordinate processes of B t , which yields that

<!-- formula-not-decoded -->

Combining the foregoing estimates, we deduce that

<!-- formula-not-decoded -->

which is the desired result.

Lemma B.7. For any δ ∈ (0 , 1) and T &gt; 0 , it holds that

<!-- formula-not-decoded -->

Proof. We first note that by Lemma B.5, for any x ∈ R d , we have

〈

A

T

η

2

i

+1

Ax

〉

, x

1

2

i

+1

s

(

x

)

By Lemma B.6, we have that

<!-- formula-not-decoded -->

This gives that

<!-- formula-not-decoded -->

Lemma B.8. For any δ ∈ (0 , 1) , suppose

<!-- formula-not-decoded -->

It holds that

<!-- formula-not-decoded -->

-

≤ ⟨

s

(

x

)

, x

⟩ -

η

∥

Ax

∥

2

≤

0

.

Proof. Recall that

With probability at least 1 -δ

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ∥ x ∥≤ r with probability 1 -δ . Thus, with probability at least 1 -2 δ , it follows that

<!-- formula-not-decoded -->

Hence, with the 1 -2 δ probability,

<!-- formula-not-decoded -->

Therefore, ensuring that

Noting that the function

<!-- formula-not-decoded -->

In this case, Lemma B.7 guarantees that

<!-- formula-not-decoded -->

Since the probability satisfying the condition is at least 1 -2 δ , we have

<!-- formula-not-decoded -->

Putting Lemma B.4 and Lemma B.8 together, we directly obtain Lemma B.3.

## B.2 Concentration of Strongly Log-Concave Distributions

Before moving futher, we first prove that a strongly log-concave distribution is highly concentrated. Lemma B.9 (Norm Bound for α -Strongly Logconcave Distributions) . Let X be a random vector in R d with density

<!-- formula-not-decoded -->

where the potential V : R d → R is α -strongly convex; that is,

<!-- formula-not-decoded -->

Denote by µ = E [ X ] the mean of X . Then, for any δ ∈ (0 , 1) , with probability at least 1 -δ we have

<!-- formula-not-decoded -->

Proof. Since V is α -strongly convex, the density π satisfies a logarithmic Sobolev inequality with constant 1 /α . Consequently, for any 1-Lipschitz function f : R d → R and any t &gt; 0 , one has the concentration inequality (via Herbst's argument)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is 1-Lipschitz (by the triangle inequality), it follows that

<!-- formula-not-decoded -->

A standard calculation using the fact that the covariance matrix of X satisfies Cov( X ) ⪯ 1 α I gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, setting we obtain

This completes the proof.

Lemma B.10 ([JCP24]) . Let µ and θ denote the mean and the mode of distribution p , respectively, where p is α -strongly log-concave and univariate. Then, | µ -θ | ≤ 1 √ α .

This immediately gives us the following corollary.

Corollary B.11. Let p be a α -strongly log-concave distribution on R d . Let θ be the mode of p . For every 0 &lt; δ &lt; 1 , we have

<!-- formula-not-decoded -->

This also implies that every α -strongly log-concave distribution is mode-centered locally wellconditioned.

Lemma B.12. Let p be an α -strongly log-concave distribution. Suppose the score function of p is L -Lipschitz. Then, for any 0 &lt; δ &lt; 1 , we have that p is ( δ, 2 √ d α + √ 2 log(1 /δ ) α , ∞ , L/α, α ) mode-centered locally well-conditioned.

## B.3 Convergence to Target Distribution

Since p is not globally strongly log-concave, we need to extend the distribution p to a globally strongly log-concave distribution. We will use the following lemma to extend the distribution.

Lemma B.13. Suppose g : B (0 , R ) → R is continuously differentiable with gradient s := ∇ g ∈ C ( B (0 , R ); R d ) and satisfies

<!-- formula-not-decoded -->

For every z ∈ B (0 , R ) define

<!-- formula-not-decoded -->

and set

<!-- formula-not-decoded -->

Then the density ˜ p ( x ) ∝ e ˜ g ( x ) is globally α -strongly log-concave.

Proof. For each fixed z ∈ B (0 , R ) the mapping φ z has Hessian -αI d , hence is α -strongly concave on the whole space. Because of (7) we have

<!-- formula-not-decoded -->

with equality when x = z . Consequently ˜ g defined in (8) agrees with g on B (0 , R ) .

Fix x ∈ R d and choose z x ∈ B (0 , R ) attaining the infimum in (8). Because φ z x touches ˜ g from above at x , the vector

<!-- formula-not-decoded -->

belongs to ∂ ˜ g ( x ) . By α -strong concavity of φ z x ,

<!-- formula-not-decoded -->

Taking the infimum over z on the left and using ˜ g ( x ) = φ z x ( x ) gives that

<!-- formula-not-decoded -->

hence ˜ g is globally α -strongly concave, and therefore p is α -strongly log-concave.

˜

LemmaB.14. Let p be a d -dimensional ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned probability distribution with 0 &lt; δ ≤ 1 / 2 and α &gt; 0 . Assume

<!-- formula-not-decoded -->

Then there exists an α -strongly log-concave distribution ˜ p on R d such that

<!-- formula-not-decoded -->

Proof. Let θ be the point in Definition B.1 and without loss of generality, we assume θ = 0 . Write B := B (0 , R ) and B c := R d \ B . By definition p ( B c ) ≤ δ .

Set g := log p , and let ˜ g be the function in Lemma B.13. Then, ρ ( x ) := e ˜ g ( x ) is α -strongly logconcave and ρ = p on B . Let Z := ∫ R d ρ and define ˜ p := ρ/Z .

Now we bound

<!-- formula-not-decoded -->

Corollary B.11 implies that ˜ p ( B c ) ≤ δ. Therefore,

<!-- formula-not-decoded -->

Note that ∫ B ρ = p ( B ) ≥ 1 -δ and ∫ B c ρ ≤ δZ (since ˜ p ( B c ) ≤ δ ). Thus,

<!-- formula-not-decoded -->

Since ˜ p = p/Z on B , we have

Therefore, I B ≤ 1 2 · 4 δ = 2 δ . Combining,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, we can consider process ˜ P defined as

<!-- formula-not-decoded -->

Then, we have the following lemma.

Lemma B.15. Suppose the following holds:

<!-- formula-not-decoded -->

We have that

<!-- formula-not-decoded -->

Proof. Let

<!-- formula-not-decoded -->

Because s ( x ) = ∇ log ˜ p ( x ) for every x ∈ B (0 , R ) , the drift coefficients of P and ˜ P coincide on the event E , and hence conditioning on E gives P ′ = ˜ P ′ .

Then, we have

<!-- formula-not-decoded -->

Taking expectation over ( y i , y i +1 ) gives

<!-- formula-not-decoded -->

Lemma B.3 implies that E [ P ( E c )] ≲ δ . Furthermore, the same argument also implies that E [ ˜ P ( E c )] ≲ δ . Therefore, we have

<!-- formula-not-decoded -->

Proof of Lemma B.2. We start by considering another process ˜ P s defined as

<!-- formula-not-decoded -->

We can see that

<!-- formula-not-decoded -->

Combining this with Lemma B.15, we have that

<!-- formula-not-decoded -->

By Markov's inequality, we have that

<!-- formula-not-decoded -->

Furthermore, by Lemma A.1 and our constraint on T , we have that

<!-- formula-not-decoded -->

Therefore, we have that

<!-- formula-not-decoded -->

Combining this with Pr[TV( ˜ p ( x | y i +1 ) , p ( x | y i +1 )) ≤ λδ ] ≥ 1 -O ( λ -1 ) , we conclude that for x T ∼ P T ,

<!-- formula-not-decoded -->

## C Control of Score Approximation and Discretization Errors

In this section, we consider these processes running for time T :

- Process P :

<!-- formula-not-decoded -->

- Process ̂ P : Let 0 = t 1 &lt; · · · &lt; t M = T be the M discretization steps with step size t j +1 -t j = h . For t ∈ [ t j , t j +1 ] ,

<!-- formula-not-decoded -->

Note that ̂ P is exactly the process (2) we run in Algorithm 1, except that we start from x 0 ∼ p ( x | y i ) .

We have shown that the process P will converge to the target distribution p ( x | y i +1 ) . We will show that the process ̂ P will also converge to p ( x | y i +1 ) with a small error

Lemma C.1. Let p be a ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned. Suppose the followings hold for a large enough constant C &gt; 0 :

<!-- formula-not-decoded -->

- ∥ A ∥ 4 ( T 2 m + TR 2 ) ≤ η 4 i Cγ 2 i .

<!-- formula-not-decoded -->

Then running ̂ P for time T guarantees that with probability at least 1 -1 /λ over y i and y i +1 , we have:

<!-- formula-not-decoded -->

In this section, we assume p is ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned. Without loss of generality, we assume that the mode of p is at 0, i.e., θ = 0 . Let L := ˜ Lα , i.e., the Lipschitz constant inside the ball B (0 , R ) .

We will also consider the following stochastic processes:

<!-- formula-not-decoded -->

- Process Q ′ is the process Q conditioned on x t ∈ B (0 , R ) for t ∈ [0 , T ] .
- Process P ′ is the process P conditioned on x t ∈ B (0 , R ) for t ∈ [0 , T ] .

We first note that following the same proof in Lemma B.3 that bounds TV( P, P ′ ) , we can also bound TV( Q,Q ′ ) .

Lemma C.2. Suppose the following holds:

<!-- formula-not-decoded -->

We have that

Lemma C.3. We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ∫ t t j √ 2 dB s ∼ N (0 , ( t -t j ) I d ) , we have that E ∥ ∫ t t j √ 2 dB s ∥ 4 ≲ d 2 ( t -t j ) 2 ≲ d 2 h 2 . This gives that

<!-- formula-not-decoded -->

Lemma C.4. Suppose ∥ A ∥ 4 ( T 2 m + TR 2 ) ≤ η 4 i η 4 i +1 C ( η 2 i -η 2 i +1 ) 2 for a large enough constant C .

<!-- formula-not-decoded -->

Proof. By Girsanov's theorem, for any trajectory x 0 ,...,t ,

<!-- formula-not-decoded -->

where the Girsanov exponent M t is given by

<!-- formula-not-decoded -->

for

<!-- formula-not-decoded -->

Since Q ′ is supported in B (0 , R ) ,

<!-- formula-not-decoded -->

Now, for ζ y := ∫ t 0 ∥ ∆ b y ( x u ) ∥ 2 du , we have that M t ∼ N ( -1 4 ζ y , 1 2 ζ y ) So,

<!-- formula-not-decoded -->

Note that ∥ η 2 i y i +1 -η 2 i +1 y i ∥ 2 has mean ∥ ( η 2 i -η 2 i +1 ) Ax ∥ 2 and is subgamma with variance m ( η 2 i +1 η 4 i -η 4 i +1 η 2 i ) 2 and scale η 2 i +1 η 4 i -η 4 i +1 η 2 i . Thus, for t ∥ A ∥ 2 ≤ η 2 i +1 η 2 i C ( η 2 i -η 2 i +1 ) we have

<!-- formula-not-decoded -->

Lemma C.5. Let E be the event on y i such that TV( Q,Q ′ ) ≤ 1 2 . Suppose

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

Proof. Note that the bound is trivial when Pr[ E ] &lt; 1 / 2 . Therefore, we can use the fact that E [ · | E ] ≲ E [ · ] throughout the proof. We have, for any t ∈ [ t j , t j +1 ] , .

<!-- formula-not-decoded -->

The first term can be bounded using Lemma C.4. Now we focus on the second term. Note that

<!-- formula-not-decoded -->

Since s is L -Lipschitz in B (0 , R ) , and using Lemma C.3, we have

<!-- formula-not-decoded -->

Since Q ′ is a conditional measure of Q , conditioned on E , we have d Q ′ d Q ≤ 1 1 -TV( Q ′ ,Q ) ≤ 2 . Therefore,

<!-- formula-not-decoded -->

This gives that

<!-- formula-not-decoded -->

Thus, by Girsanov's theorem,

<!-- formula-not-decoded -->

By Pinsker's inequality,

<!-- formula-not-decoded -->

Hence,

<!-- formula-not-decoded -->

Then have the following as a corollary:

## Corollary C.6. Suppose

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

Proof. We have that

<!-- formula-not-decoded -->

Furthermore,

<!-- formula-not-decoded -->

Proof of Lemma C.1. We note that by our definition of γ i ,

<!-- formula-not-decoded -->

Then, combining Corollary C.6 with Lemmas B.3 and C.2, we have

<!-- formula-not-decoded -->

The conditions in Lemmas B.3 and C.2 are satisfied by our assumptions, noting that η i +1 &lt; η i implies the bound on R holds for both processes.

Applying Markov's inequality and combining Lemma B.2 with the above, we conclude the proof.

## D Admissible Noise Schedule

Recall that we can define process ̂ P i that converges from p ( x | y i ) to p ( x | y i +1 ) : Let 0 = t 1 &lt; · · · &lt; t M = T be the M discretization steps with step size t j +1 -t j = h . For t ∈ [ t j , t j +1 ] ,

<!-- formula-not-decoded -->

We have already proven that we can converge the process from p ( x | y i ) to p ( x | y i +1 ) with good probability, as long as some conditions are satisfied. Those conditions actually depend on the choice of the schedule of η i and T i . In this section, we will specify the schedule of η i and T i .

Now we specify the schedule of η i and T i .

Definition D.1. We say a noise schedule η 1 &gt; · · · &gt; η N together with running times T 1 , · · · , T N -1 is admissible (for a set of parameters C,α,λ,A,d,ε,η,R ) if:

- η N = η ;

<!-- formula-not-decoded -->

- For all γ i = ( η i /η i +1 ) 2 -1 , we have γ i ≤ 1 and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The reason we need to satisfy the last inequality is to satisfy the conditions in Lemma C.1. We formalize this in the following lemma.

Lemma D.2. Let C &gt; 0 be a sufficiently large constant and p be a ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned distribution. For any δ, ε ∈ (0 , 1) and λ &gt; 1 , suppose

<!-- formula-not-decoded -->

For any admissible schedule ( η i ) i ∈ [ N ] and ( T i ) i ∈ [ N -1] , running the process ̂ P i for time T i guarantees that with probability at least 1 -1 /λ over y i and y i +1 :

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Furthermore,

Proof. It is straightforward to verify that an admissible schedule satisfies the first two conditions of Lemma C.1.

For the third condition regarding R , our assumption states:

<!-- formula-not-decoded -->

Given that T i ≲ m +log( λ/ε ) α , this choice of R is sufficient to satisfy the third condition in Lemma C.1. Therefore, applying Lemma C.1 at each step i , we obtain that with probability at least 1 -1 /λ over y i and y i +1 :

<!-- formula-not-decoded -->

We also want to prove the following two lemmas:

Lemma D.3. Let p be a d -dimensional ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned distribution. For any δ ∈ (0 , 1) , suppose

<!-- formula-not-decoded -->

Then, suppose η 1 ≥ λ ∥ A ∥ ε √ d α , with probability at least 1 -1 λ over y 1 ,

<!-- formula-not-decoded -->

Lemma D.4. There exists an admissible noise such that

<!-- formula-not-decoded -->

where ρ = ∥ A ∥ η √ α .

## D.1 The Closeness Between p ( x | y 1 ) and p ( x )

In this part, we prove Lemma D.3, showing that any admissible schedule has a large enough η 1 , enabling us to use p ( x ) to approximate p ( x | y 1 ) .

We have the following standard information-theoretic result.

Lemma D.5. Let X ∈ R m be a random variable, and Y = X + N (0 , η 2 I m ) . Then,

<!-- formula-not-decoded -->

Lemma D.6. For any distribution p with E x ∼ p [ ∥ x -E x ∥ 2 ] = m 2 2 , we have

<!-- formula-not-decoded -->

Proof. Note that E [KL( p ( x | y i ) ∥ p ( x ))] is exactly the mutual information between x and y i . In addition, we have

<!-- formula-not-decoded -->

By Pinsker's inequality, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma D.7. Let p be a d -dimensional ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned probability distribution. Assume

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then

Proof. Lemma B.14 provides an α -strongly log-concave density ˜ p satisfying

<!-- formula-not-decoded -->

For an α -strongly log-concave law the Brascamp-Lieb inequality yields Cov ˜ p ⪯ α -1 I d ; hence

<!-- formula-not-decoded -->

Applying Lemma D.6 to ˜ p gives

<!-- formula-not-decoded -->

Note that

TV( p ( x | y 1 ) , p ( x )) ≤ TV( p ( x | y 1 ) , ˜ p ( x | y 1 )) + TV( ˜ p ( x | y 1 ) , ˜ p ( x )) + TV( ˜ p ( x ) , p ( x )) . Integrating in y 1 and using the elementary fact

<!-- formula-not-decoded -->

together with the above calculaion, yields

<!-- formula-not-decoded -->

This proves the stated bound.

Now we prove Lemma D.3.

Proof of Lemma D.3. By Lemma D.7, we have

<!-- formula-not-decoded -->

Since all admissible noise schedules satisfy η 1 ≥ λ ∥ A ∥ ε √ d α . This implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Consequently,

By Markov's inequality, with probability at least 1 -1 λ over y 1 ,

<!-- formula-not-decoded -->

which proves the lemma.

## D.2 Bound for N Mixing Steps

In this part, we prove Lemma D.4.

Lemma D.8. Let a, x 0 &gt; 0 , and let c &gt; 0 . Consider the number sequence

<!-- formula-not-decoded -->

For every B &gt; 0 , let k ( B ) be the minimum integer i such that x i ≥ B . Then

<!-- formula-not-decoded -->

Proof. We show in two steps that the time to go from x 0 to 1 /a , then to B . Define

<!-- formula-not-decoded -->

Bound for k 1 . We first show that k 1 ≲ ( ax 0 ) -c . Consider the quantities

<!-- formula-not-decoded -->

and let j ∗ be the smallest j such that x N j ≥ 1 /a . If instead x 0 ≥ 1 /a already, then k 1 = 0 and there is nothing to prove.

Assume x 0 &lt; 1 /a . For each j &lt; j ∗ define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We claim that

Indeed, for each j &lt; j ∗ ,

<!-- formula-not-decoded -->

Since we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By monotonicity of the sequence ( x i ) , it follows that N j +1 ≤ N j + t j . Summing over j up to j ∗ -1 gives

<!-- formula-not-decoded -->

By definition, N j ∗ is the first index i such that x i ≥ 1 /a , so k 1 = N j ∗ ≲ ( ax 0 ) -c .

Bound to achieve B . If B ≤ 1 /a , the bound already holds. Now we analyze how many steps Note that for every i ≥ k 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, we have

This proves that

<!-- formula-not-decoded -->

Lemma D.9. Given parameters x 0 , a, b &gt; 0 , consider sequence inductively defined by x i +1 = (1 + γ i ) x i , where

<!-- formula-not-decoded -->

Given B , let k ( B ) be the minimum integer i such that x i ≥ B . Then,

<!-- formula-not-decoded -->

Proof. We do case analysis.

Case 1: x 0 ≥ b 2 /a . We always choose γ i = √ x i /a . We can verify that

<!-- formula-not-decoded -->

and this satisfies the requirement for γ i . By applying Lemma D.8, we have that

<!-- formula-not-decoded -->

Case 2: x 0 ≤ B ≤ b 2 /a . We always choose γ i = min( x i /b, 1) . We can verify that

<!-- formula-not-decoded -->

and this satisfies the requirement for γ i . By applying Lemma D.8, we have that

<!-- formula-not-decoded -->

Case 3: x 0 ≤ b 2 /a ≤ B . We combine the bound for the first two cases, where we first go from x 0 to b 2 /a , then go from b 2 /a to B . Then we have

<!-- formula-not-decoded -->

Proof of Lemma D.4. Now we describe how we construct an admissible noise schedule. Consider we start from η ′ 1 = η , and for each i , we iteratively choose γ ′ i to be the maximum γ ≤ 1 such that

<!-- formula-not-decoded -->

and then set η ′ i +1 = √ (1 + γ ′ i )( η ′ i ) 2 . We continue this process until we reach η ′ N ≥ λ ∥ A ∥ ε √ d α . It is easy to verify that ( η ′ N , η ′ N -1 , . . . , η ′ 1 ) is an admissible noise schedule. Now we bound the number of iterations N .

Since for all γ , we have ∥ A ∥ 4 ( f 2 T ( γ ) m + f T ( γ ) R 2 ) ≤ ∥ A ∥ 4 ( √ mf T ( γ ) + R 2 2 √ m ) 2 , a sufficient condition for ∥ A ∥ 4 ( f 2 T ( γ ) m + f T ( γ ) R 2 ) ≤ ( η ′ i ) 4 Cγ 2 is that

<!-- formula-not-decoded -->

Therefore, fixing η ′ i , we have that γ ′ i is at least

<!-- formula-not-decoded -->

Now we look at the inductive sequence starting from x 1 = η 2 , and x i +1 = (1 + ˜ γ i ) x i , where

<!-- formula-not-decoded -->

By Lemma D.9, we know that for any η goal &gt; 0 , we can achieve x N ≥ η 2 goal within

<!-- formula-not-decoded -->

Taking in η goal = λ ∥ A ∥ ε √ d α

, we conclude the lemma.

## E Theoretical Analysis of Algorithm 1

In this section, we analyze the algorithm presented in Algorithm 1. In Algorithm 1, the algorithm initializes by drawing a sample from the prior distribution p ( x ) via the diffusion SDE, which introduces sampling error. [CCL + 22] demonstrated that this diffusion sampling error is polynomially small, with the exact magnitude depending on the discretization scheme chosen for the diffusion SDE. Since the focus of this paper is on enabling an unconditional diffusion sampling model to perform posterior sampling, the choice of diffusion discretization and its associated error are not not the focus of our analysis. Consequently, we omit the diffusion sampling error in the error analysis presented in this section. This omission does not impact the rigor of the theorems in the main paper, as the error is polynomially small.

We start with the following lemma:

LemmaE.1. Let C &gt; 0 be a large enough constant. Let p be a ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned distribution. For every δ, ε ∈ (0 , 1) and λ &gt; 1 , suppose

<!-- formula-not-decoded -->

Then running Algorithm 1 will guarantee that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let ε step := C 0 ( ε + λδ + λ √ m +log( λ/ε ) α · ( ε dis + ε score ) ) , where C 0 is a constant large enough to absorb the implicit constants in Lemma D.3 and Lemma D.2.

We prove by induction that for each i ∈ [ N ] :

<!-- formula-not-decoded -->

For the base case ( i = 1 ), since X 1 ∼ p ( x ) , Lemma D.3 gives that TV( p ( x ) , p ( x | y 1 )) ≤ ε step with probability at least 1 -1 /λ over y 1 .

For the inductive step, assume the statement holds for some i &lt; N . Let E i be the event that TV( X i , p ( x | y i )) ≤ i · ε step , so Pr[ E c i ] ≤ i/λ .

Let X ∗ i ∼ p ( x | y i ) and let X ∗ i +1 be the result of evolving X ∗ i for time T i using the SDE in Equation (2). By Lemma D.2, the event F i +1 that TV( X ∗ i +1 , p ( x | y i +1 )) ≤ ε step has probability at least 1 -1 /λ over y i , y i +1 and the SDE path.

By the triangle inequality and data processing inequality:

<!-- formula-not-decoded -->

If both E i and F i +1 occur, then TV( X i +1 , p ( x | y i +1 )) ≤ ( i + 1) ε step . The probability that this bound fails is at most:

<!-- formula-not-decoded -->

Thus, the induction holds for i +1 , and the lemma follows for i = N .

Lemma E.2. Let S 1 and S 2 be two random variables such that

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

Proof. Let E ( y 1 , . . . , y N ) be the event such that TV(( S 1 | E ) , p (( S 2 | E )) ≤ ε . Then, we have that

<!-- formula-not-decoded -->

Since Pr[ E ] ≥ 1 -δ , we apply Markov's inequality, and have

<!-- formula-not-decoded -->

Hence, we have with probability 1 -δ ε over y ,

<!-- formula-not-decoded -->

Applying Lemma E.2 on Lemma E.1 gives the following corollary.

Corollary E.3. Let C &gt; 0 be a large enough constant. Let p be a ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned distribution. For every δ, ε ∈ (0 , 1) and λ &gt; 1 , suppose

<!-- formula-not-decoded -->

Define Then running Algorithm 1 will guarantee that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with where

<!-- formula-not-decoded -->

Lemma E.4 (Main Analysis Lemma for Algorithm 1) . Let ρ = ∥ A ∥ η √ α . For all 0 &lt; ε, δ &lt; 1 , there exists

<!-- formula-not-decoded -->

such that: suppose distribution p is a ( ε K 2 , ˜ r/ √ α, R, ˜ L, α ) mode-centered locally well-conditioned distribution with R ≥ √ K √ m/α ρ , and ε score ≤ √ α/m K 2 δ ; then Algorithm 1 samples from a distribution ̂ p ( x | y ) such that δ.

<!-- formula-not-decoded -->

Furthermore, the total iteration complexity can be bounded by

<!-- formula-not-decoded -->

Proof. To distinguish the ε and δ in the lemma and the one in Corollary E.3, we will use ε error and δ error to denote the ε and δ in our lemma statement. We need to set parameters in Corollary E.3. For any given 0 &lt; δ error , ε error , we set

<!-- formula-not-decoded -->

and we set λ to be the minimum λ that satisfies

<!-- formula-not-decoded -->

Now we verify the correctness. Taking in the bound for N in Lemma D.4, we have

<!-- formula-not-decoded -->

By the setting of our parameters, we have Nε ≲ ε error , λδ ≲ ε error , and N/λε error ≲ δ error . This guarantees that

<!-- formula-not-decoded -->

It is easy to verify our bound on R satisfies the condition in Corollary E.3. Note that if a distribution is ( δ, r, R, ˜ L, α ) mode-centered locally well-conditioned, then it is also ( δ, r, R ′ , ˜ L, α ) modecentered locally well-conditioned for any R ′ ≤ R . Therefore, we can set R to be the minimum R that satisfies the condition.

<!-- formula-not-decoded -->

Therefore, we only need λN √ m +log( λ/ε ) α ( ε dis + ε score ) ≲ ε error . This can be satisfied when

<!-- formula-not-decoded -->

Recall that

<!-- formula-not-decoded -->

Therefore, we need to set

<!-- formula-not-decoded -->

Note that the bound for the sum of N mixing times can be bounded by

<!-- formula-not-decoded -->

Therefore, the total iteration complexity is bounded by ˜ O ( Kmδ error ε error αh ) ,

<!-- formula-not-decoded -->

We can relax it and make the bound be

<!-- formula-not-decoded -->

Take in R , and we have

<!-- formula-not-decoded -->

## E.1 Application on Strongly Log-concave Distributions

By Lemma B.12, any α -strongly log-concave distribution that has L -Lipschitz score is locally well-conditioned distribution p is ( δ, 2 √ d α + √ 2 log(1 /δ ) α , ∞ , L/α, α ) mode-centered locally wellconditioned. Therefore, take this into Lemma E.4, we have the following result.

Lemma E.5. Let p ( x ) be an α -strongly log-concave distribution over R d with L -Lipschitz score. Let ρ = ∥ A ∥ η √ α . For all 0 &lt; ε, δ &lt; 1 , there exists

<!-- formula-not-decoded -->

such that: suppose ε score ≤ α/m K 2 δ , then Algorithm 1 samples from a distribution ̂ p ( x | y ) such that Pr y [TV( ̂ p ( x | y ) , p ( x | y )) ≤ ε ] ≥ 1 -δ.

Furthermore, the total iteration complexity can be bounded by

<!-- formula-not-decoded -->

To enhance clarity, we state our result in terms of expectation and established the following theorem: Theorem E.6 (Posterior sampling with global log-cancavity) . Let p ( x ) be an α -strongly logconcave distribution over R d with L -Lipschitz score. Let ρ = ∥ A ∥ η √ α . For all 0 &lt; ε &lt; 1 , there exists

<!-- formula-not-decoded -->

such that: suppose ε score ≤ α/m K 2 ε , then Algorithm 1 samples from a distribution ̂ p ( x | y ) such that E y [TV( ̂ p ( x | y ) , p ( x | y ))] ≤ ε.

Furthermore, the total iteration complexity can be bounded by

<!-- formula-not-decoded -->

This gives Theorem 1.1.

Theorem 1.1 (Posterior sampling with global log-concavity) . Let p ( x ) be an α -strongly logconcave distribution over R d with L -Lipschitz score. For any 0 &lt; ε &lt; 1 , there exist K 1 = poly( d, m, ∥ A ∥ η √ α , 1 ε ) and K 2 = poly( d, m, ∥ A ∥ η √ α , 1 ε , L α ) such that: if ε score ≤ √ α K 1 , then there exists an algorithm that takes K 2 iterations to sample from a distribution ̂ p ( x | y ) with

<!-- formula-not-decoded -->

Remark E.7. The analysis above is restricted to strongly log-concave distributions, where ∇ 2 log p ( x ) ≺ 0 . However, this directly implies that we can use our algorithm to perform posterior sampling on log-concave distributions, for which ∇ 2 log p ( x ) ⪯ 0 .

Specifically, for any log-concave distribution p , we can define a distribution q ( x ) ∝ p ( x ) · exp ( -ε 2 ∥ x -θ ∥ 2 2 m 2 2 ) , where θ is the mode of p and m 2 2 is the variance of p . It is straightforward to verify that TV( p, q ) ≲ ε , and q is ( ε 2 /m 2 2 ) -strongly log-concave. Therefore, by sampling from q ( x | y ) , we can approximate p ( x | y ) , incurring an additional expected TV error of ε .

## E.2 Gaussian Measurement

In this section, we prove Theorem 1.2. In Algorithm 2, we describe how to make Algorithm 1 work on the Gaussian case.

We first verify that suppose Assumption 1 holds, we can also have L 4 -accurate estimates for the smoothed scores of p x 0 , so this satisfies the requirement of running Algorithm 1. We need to use the following lemma, with proof deferred to Section E.5.

Lemma E.8. Let X , Y , and Z be random vectors in R d , where Y = X + N (0 , σ 2 1 I d ) and Z = X + N (0 , σ 2 2 I d ) . The conditional density of Z given Y , denoted p ( Z | Y ) , is a multivariate normal distribution with mean

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, the gradient of the log-likelihood log p ( Z | Y ) with respect to Y is given by

<!-- formula-not-decoded -->

Using this, we can calculate the smoothed conditional score given x 0 :

LemmaE.9. For any smoothing level t ≥ 0 , suppose we have score estimate ̂ s t 2 ( x ) of the smoothed distributions p t 2 ( x ) = p ( x ) ∗ N (0 , t 2 I d ) that satisfies

<!-- formula-not-decoded -->

Then we can calculate a score estimate ̂ s x 0 ,t 2 ( x ) of the distribution p x 0 ,t 2 ( x ) = p x 0 ( x ) ∗N (0 , t 2 I d ) such that

<!-- formula-not-decoded -->

Proof. Let x ( t ) ∼ p t 2 . Then, for any value of x ( t ) , we have

<!-- formula-not-decoded -->

Note that the second term is exactly in the form of Lemma E.8, so we can calculate this exaclty. For the first term, we use our score estimate ̂ s t 2 ( x ( t ) ) for it. In this way, we have that for any x ,

<!-- formula-not-decoded -->

and covariance matrix

Therefore,

<!-- formula-not-decoded -->

Applying Markov's inequality, we have:

Corollary E.10. Suppose Assumption 1 holds for our prior distribution p . Then with 1 -δ probability over x 0 : we have smoothed score estimates for p x 0 with L 4 error bounded by ε 4 score /δ ; in other words, Assumption 1 holds for p x 0 , where ε score is substituted with ε score /δ 1 / 4 .

To capture the behavior of a Gaussian measurement more accurately, we first define a relaxed version of mode-centered locally well-conditioned distribution.

Definition E.11. For δ ∈ [0 , 1) and R, ˜ L, α ∈ (0 , + ∞ ] , we say that a distribution p is ( δ, r, R, ˜ L, α ) locally well-conditioned if there exists θ such that

- Pr x ∼ p [ x ∈ B ( θ, r )] ≥ 1 -δ .
- For x, y ∈ B ( θ, R ) , we have that ∥ s ( x ) -s ( y ) ∥≤ ˜ Lα ∥ x -y ∥ .
- For x, y ∈ B ( θ, R ) , we have that ⟨ s ( y ) -s ( x ) , x -y ⟩ ≥ α ∥ x -y ∥ 2 .

Note that this definition can still imply that the distribution is mode-centered local well-conditioned, due to the following fact:

Lemma E.12. Let p be a probability density on R d . Fix 0 &lt; r &lt; R and θ ∈ R d such that

<!-- formula-not-decoded -->

If R &gt; 4 dr , then there exists θ ′ ∈ B ( θ, 4 dr ) with ∇ log p ( θ ′ ) = 0 .

We defer its proof to Section E.5. This implies the following lemma:

Lemma E.13. Let p be a ( δ, r, R, ˜ L, α ) locally well conditioned distribution with R &gt; 9 dr and δ &lt; 0 . 1 . Then p is ( δ, (4 d +1) r, R -4 dr, ˜ L, α ) mode-centered locally well conditioned.

This gives a version of Lemma E.4 for locally well-conditioned distributions as a corollary:

Lemma E.14. Let ρ = ∥ A ∥ η √ α . For all 0 &lt; ε, δ &lt; 1 , there exists

<!-- formula-not-decoded -->

such that: suppose distribution p is a ( ε K 2 , ˜ r/ √ α, R, ˜ L, α ) mode-centered locally well-conditioned distribution with R ≥ √ K √ m/α ρ , and ε score ≤ √ α/m K 2 δ . Then Algorithm 1 samples from a distribution ̂ p ( x | y ) such that

<!-- formula-not-decoded -->

Furthermore, the total iteration complexity can be bounded by

<!-- formula-not-decoded -->

The reason we want this relaxed notion of locally well-conditioned is that, this captures the behavior of a Gaussian measurement. First note that:

Lemma E.15. Let p be a distribution on R d . Let ˜ x = x true + N (0 , σ 2 I d ) be a Gaussian measurement of x true ∼ p . Let p ˜ x ( x ) be the posterior distribution of x given ˜ x . Then, for any δ ∈ (0 , 1) and δ ′ ∈ (0 , 1) , with probability at least 1 -δ ′ over ˜ x ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Algorithm 2 Sampling from p ( x | x 0 , y ) given an extra Gaussian measurement x 0

- 1: function GAUSSIANSAMPLER( p : R d → R , x 0 ∈ R d , y ∈ R m , A ∈ R m × d , η, σ ∈ R )
- 2: Let p x 0 ( x ) := p ( x | x + N (0 , σ 2 I d ) = x 0 ) .
- 3: Use Algorithm 1, return

## 4: end function

Again, we defer its proof to Section E.5. This implies the following lemma.

Lemma E.16. For δ ∈ (0 , 1) , suppose p is a distribution over R d such that

<!-- formula-not-decoded -->

Given a Gaussian measurement x 0 = x + N (0 , σ 2 I d ) of x ∼ p with

<!-- formula-not-decoded -->

Let x 0 = x + N (0 , σ 2 I d ) , where x ∼ p . Then, suppose R . with probability at least 1 -3 δ probability over x 0 , p x 0 is ( δ, σ ( √ d + √ 4 log 1 δ ) , R/ 2 , 2 Lσ 2 +2 , 1 2 σ 2 ) locally well-conditioned.

Proof. Let us check the locally well-conditioned conditions with θ = x 0 one by one. The concentration follows directly from Lemma E.15, incurring an error probability of δ .

By our choice of σ , we have that

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

By direct calculation, we have that

<!-- formula-not-decoded -->

By our choice of σ , we have that whenever -LI d ⪯ ∇ 2 log p ( x ) ⪯ ( τ 2 /R 2 ) I d ,

<!-- formula-not-decoded -->

This satisfies the Lipschitzness and the strong log-concavity condition by giving an additional error probability of 2 δ .

This gives us the main lemma for our local log-concavity case:

Lemma E.17. For any δ, ε, τ, σ, R, L &gt; 0 , suppose p ( x ) is a distribution over R d such that

<!-- formula-not-decoded -->

Let ρ = ∥ A ∥ σ η . There exists

<!-- formula-not-decoded -->

such that: suppose R 2 ≥ ( K √ m ρ 2 +4 τ ) σ 2 and ε score ≤ 1 K 2 √ mσ , then Algorithm 2 samples from a distribution ̂ p ( x | x 0 , y ) such that

<!-- formula-not-decoded -->

Furthermore, the total iteration complexity can be bounded by

<!-- formula-not-decoded -->

POSTERIORSAMPLER ( p x 0 , y, A, η ) .

## Algorithm 3 Competitive Compressed Sensing Algorithm Given a Rough Estimation

- 1: function COMPRESSEDSENSING( p : R d → R , x 0 ∈ R d , y ∈ R m , A ∈ R m × d , η, R ∈ R )
- 2: Let σ = R/δ .
- 3: Sample x ′ 0 = x 0 + N (0 , σ 2 I d ) .
- 4: Use Algorithm 2, return

GAUSSIANSAMPLER ( p, x ′ 0 , y, A, η, σ )

## 5: end function

Proof. Combining Corollary E.10 with Lemma E.16 enables us to apply Lemma E.14 and proves the lemma.

Expressing this in expectation, we have the following theorem.

Theorem E.18 (Posterior sampling with local log-concavity) . For any ε, τ, R, L &gt; 0 , suppose p ( x ) is a distribution over R d such that

<!-- formula-not-decoded -->

Let ρ = ∥ A ∥ σ η . There exists

<!-- formula-not-decoded -->

such that: given a Gaussian measurement x 0 = x + N (0 , σ 2 I d ) of x ∼ p with R 2 ≥ ( K √ m ρ 2 +4 τ ) σ 2 , and ε score ≤ 1 K 2 √ mσ ; then Algorithm 2 samples from a distribution ̂ p ( x | x 0 , y ) such that

<!-- formula-not-decoded -->

Furthermore, the total iteration complexity can be bounded by

<!-- formula-not-decoded -->

This gives us Theorem 1.2:

Theorem 1.2 (Posterior sampling with local log-concavity) . For any ε, τ, R, L &gt; 0 , suppose p ( x ) is a distribution over R d such that

<!-- formula-not-decoded -->

Then, there exist K 1 , K 2 = poly( d, m, ∥ A ∥ σ η , 1 ε ) and K 3 = poly( d, m, ∥ A ∥ σ η , 1 ε , Lσ 2 ) such that: Given a Gaussian measurement x 0 = x + N (0 , σ 2 I d ) of x ∼ p with σ ≤ R/ ( K 1 + 2 τ ) . If ε score ≤ 1 K 2 σ , then there exists an algorithm that takes K 3 iterations to sample from a distribution ̂ p ( x | x 0 , y ) such that

<!-- formula-not-decoded -->

## E.3 Compressed Sensing

In this section, we prove Corollary 1.3. We first describe the sampling procedure in Algorithm 3. Now we verify its correctness.

Lemma E.19. For any δ, τ, R, R ′ , L &gt; 0 , suppose p ( x ) is a distribution over R d such that

<!-- formula-not-decoded -->

Let ρ = ∥ A ∥ R η . There exists

<!-- formula-not-decoded -->

such that: suppose ( R ′ ) 2 ≥ ( K √ m ρ 2 +4 τ ) R 2 and ε score ≤ 1 K 2 √ mR , then conditioned on ∥ x 0 -x ∥≤ R , Algorithm 3 of Algorithm 3 samples from a distribution ̂ p (depending on x ′ 0 and y ) such that

<!-- formula-not-decoded -->

Furthermore, the total iteration complexity can be bounded by

<!-- formula-not-decoded -->

Proof. This is a direct application of Lemma E.17. The sole difference is that x ′ 0 follows x 0 + N (0 , σ 2 I d ) instead of x + N (0 , σ 2 I d ) . Because ∥ x 0 -x ∥≤ R , x ′ 0 remains sufficiently close to x for the local Hessian condition to hold, so the proof of Lemma E.17 carries over verbatim.

Now we explain why we want to sample from p ( x | x + N (0 , σ 2 I d ) = x ′ 0 , Ax + ξ = y ) . Essentially, the extra Gaussian measurement won't hurt the concentration of p ( x | y ) itself. We abstract it as the following lemma:

Lemma E.20. Let ( X,Y ) be jointly distributed random variables with X ∈ R d . Assume that for some r &gt; 0 and 0 &lt; δ &lt; 1

<!-- formula-not-decoded -->

Define Z = X + ε where ε ∼ N (0 , σ 2 I d ) is independent of ( X,Y ) . If

<!-- formula-not-decoded -->

then for ̂ X ∼ p ( X | Y, Z ) one has

<!-- formula-not-decoded -->

Proof. Fix Y and draw an auxiliary point ˜ X ∼ p ( X | Y ) . Let Z ′ = ˜ X + ε ′ with ε ′ ∼ N (0 , σ 2 I d ) independent of everything else. On the event

<!-- formula-not-decoded -->

Z and Z ′ are Gaussians with the same covariance σ 2 I d and means X and ˜ X . Pinsker's inequality combined with the KL divergence between the two Gaussians gives

<!-- formula-not-decoded -->

Hence

<!-- formula-not-decoded -->

because Pr[ E c ] ≤ δ by the hypothesis on p ( X | Y ) .

By construction, so

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the set A = { ( Y, Z, ̂ X ) : ∥ X -̂ X ∥ &gt; r } the total-variation bound gives

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This implies the following lemma:

whence

Lemma E.21. Consider the random variables in Algorithm 3. Suppose that

- Information theoretically, it is possible to recover ̂ x from y satisfying ∥ ̂ x -x ∥ ≤ r with probability 1 -δ over x ∼ p and y .
- Pr[ ∥ x 0 -x ∥ ≤ R ] ≥ 1 -δ .

Then drawing sample ̂ x ∼ p ( x | x + N (0 , σ 2 I d ) = x ′ 0 , Ax + ξ = y ) would give that

<!-- formula-not-decoded -->

Proof. By [JAD + 21], the first condition implies that,

<!-- formula-not-decoded -->

Then by Lemma E.20, suppose we have x ′ = x + N (0 , σ 2 I d ) , then

<!-- formula-not-decoded -->

Note that whenever ∥ x -x 0 ∥≤ r , we have

<!-- formula-not-decoded -->

This proves that

<!-- formula-not-decoded -->

Lemma E.22. Consider attempting to accurately reconstruct x from y = Ax + ξ . Suppose that:

- Information theoretically, it is possible to recover ̂ x from y satisfying ∥ ̂ x -x ∥ ≤ r with probability 1 -δ over x ∼ p and y .
- We have access to a 'naive' algorithm that recovers x 0 from y satisfying ∥ x 0 -x ∥ ≤ R with probability 1 -δ over x ∼ p and y .

Let ρ = ∥ A ∥ R ηδ . There exists

<!-- formula-not-decoded -->

such that: suppose for R ′ = ( R/δ ) · √ K √ m ρ 2 +4 τ ,

<!-- formula-not-decoded -->

Then we give an algorithm that recovers ̂ x satisfying ∥ ̂ x -x ∥ ≤ 2 r with probability 1 -O ( δ ) , in poly( d, m, ∥ A ∥ R η , 1 δ ) time, under Assumption 1 with ε score &lt; 1 K 2 √ m ( R/δ ) .

Proof. By our assumption and Lemma E.19, we have that we are sampling from p ( x | x + N (0 , σ 2 I d ) = x ′ 0 , Ax + ξ = y ) with δ TV error with 1 -O ( δ ) probability. By Lemma E.21, this would recover x within distance 2 r with 1 -O ( δ ) probaility. Combining the two gives the result.

Setting τ = 0 would give Corollary 1.3 as a corollary.

Corollary 1.3 (Competitive compressed sensing) . Consider attempting to accurately reconstruct x from y = Ax + ξ . Suppose that:

- Information theoretically (but possibly requiring exponential time or using exact knowledge of p ( x ) ), it is possible to recover ̂ x from y satisfying ∥ ̂ x -x ∥ ≤ r with probability 1 -δ over x ∼ p and y .

- We have access to a 'naive' algorithm that recovers x 0 from y satisfying ∥ x 0 -x ∥ ≤ R with probability 1 -δ over x ∼ p and y .
- For R ′ = R · poly( d, m, ∥ A ∥ R , 1 )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we give an algorithm that recovers ̂ x satisfying ∥ ̂ x -x ∥ ≤ 2 r with probability 1 -O ( δ ) , in poly( d, m, ∥ A ∥ R η , 1 δ ) time, under Assumption 1 with ε score &lt; 1 poly( d,m, ∥ A ∥ R η , 1 δ ,LR 2 ) R .

## E.4 Ring example

Let w ∈ (0 , 0 . 01) and let p 0 be the uniform probability measure on the unit circle S 1 = { x ∈ R 2 : ∥ x ∥ = 1 } . Define the circle-Gaussian mixture

<!-- formula-not-decoded -->

Lemma E.23. For any x ∈ R 2 with radius r = ∥ x ∥ &gt; 0 , the Hessian of the log-density satisfies

<!-- formula-not-decoded -->

Proof. Rotational invariance gives p ( x ) = p ( r ) with

<!-- formula-not-decoded -->

Write f ( r ) = log p ( r ) and set z = r/w 2 &gt; 0 . Using I ′ 0 ( z ) = I 1 ( z ) , we get the first and second derivatives:

<!-- formula-not-decoded -->

For r &gt; 0 , the eigenvalues of ∇ 2 log p are

<!-- formula-not-decoded -->

The Turán inequality I 1 ( z ) 2 -I 0 ( z ) I 2 ( z ) ≥ 0 implies λ r ( r ) ≤ -1 /w 2 ; thus, the largest eigenvalue is λ t ( r ) .

Since I 1 ( z ) /I 0 ( z ) ≤ 1 for all z &gt; 0 and I 1 ( z ) /I 0 ( z ) ≤ z/ 2 for 0 &lt; z ≤ 1 ,

<!-- formula-not-decoded -->

Lemma E.24. For every x ∈ R 2 , we have

<!-- formula-not-decoded -->

Proof. Write u = (cos θ, sin θ ) and

<!-- formula-not-decoded -->

Differentiating under the integral gives

<!-- formula-not-decoded -->

so

Differentiating once more,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A standard score-covariance identity shows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

hence

Since Cov u | x ( u ) ⪰ 0 , it follows that as claimed.

Lemma E.25. For any w ∈ (0 , 1 / 2) , we have that

<!-- formula-not-decoded -->

Proof. Note that

## E.5 Deferred Proof

Lemma E.8. Let X , Y , and Z be random vectors in R d , where Y = X + N (0 , σ 2 1 I d ) and Z = X + N (0 , σ 2 2 I d ) . The conditional density of Z given Y , denoted p ( Z | Y ) , is a multivariate normal distribution with mean

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, the gradient of the log-likelihood log p ( Z | Y ) with respect to Y is given by

<!-- formula-not-decoded -->

and covariance matrix

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The rest follows by combining Lemma E.23 and Lemma E.24.

Hence, we can apply Theorem 1.2 on our ring distribution p and get the following corollary:

Corollary E.26. Let A ∈ R C × 2 be a matrix for some constant C &gt; 0 . Consider x ∼ p with two measurements given by

<!-- formula-not-decoded -->

Suppose ∥ A ∥ w/η = O (1) . Then, if σ ≤ cw and ε score ≤ cw -1 for sufficiently small constant c &gt; 0 , Algorithm 2 takes a constant number of iterations to sample from a distribution ̂ p ( x | x 0 , y ) such that

<!-- formula-not-decoded -->

Proof. Since Z | Y ∼ N ( µ Z | Y , Σ Z | Y ) , the log-likelihood function is

<!-- formula-not-decoded -->

To compute the gradient with respect to Y , we focus on the term involving µ Z | Y :

<!-- formula-not-decoded -->

Differentiating with respect to Y gives:

<!-- formula-not-decoded -->

Since µ Z | Y = σ 2 2 ( σ 2 1 + σ 2 2 ) -1 Y , we have

<!-- formula-not-decoded -->

Thus, the gradient becomes

<!-- formula-not-decoded -->

Substituting the inverse of the covariance matrix Σ Z | Y , we get

<!-- formula-not-decoded -->

and the final expression for the gradient is

<!-- formula-not-decoded -->

Lemma E.12. Let p be a probability density on R d . Fix 0 &lt; r &lt; R and θ ∈ R d such that

<!-- formula-not-decoded -->

.

If R &gt; 4 dr , then there exists θ ′ ∈ B ( θ, 4 dr ) with ∇ log p ( θ ′ ) = 0 .

Proof. By Lemma B.13, there is a normalised density q satisfying ∇ log q = ∇ log p on B ( θ, R ) and such that log q is α -strongly concave on R d . The difference log p -log q is therefore constant on B ( θ, R ) ; hence

<!-- formula-not-decoded -->

for some C &gt; 0 .

Let µ = arg max q ; strong concavity gives ∇ log q ( µ ) = 0 and uniqueness of µ . Assume for contradiction that ∥ µ -θ ∥≥ 4 dr . Set λ = 2 r/ ∥ µ -θ ∥≤ 1 / (2 d ) and define

<!-- formula-not-decoded -->

Then det Dτ = (1 -λ ) d and τ ( B ( θ, r )) = B ( θ ′ , (1 -λ ) r ) with θ ′ = τ ( θ ) ⊂ B ( θ, R ) . Along any ray starting at µ the function t ↦→ log q ( µ + t ( x -µ )) is strictly decreasing for t ≥ 0 ; hence q ( τ ( x )) ≥ q ( x ) for every x .

A change of variables yields

<!-- formula-not-decoded -->

Because λ ≤ 1 / (2 d ) , (1 -λ ) d ≥ e -1 / 2 &gt; 0 . 6 . Multiplying by C and using p = Cq on B ( θ, R ) gives

<!-- formula-not-decoded -->

The two balls B ( θ, r ) and B ( θ ′ , (1 -λ ) r ) are disjoint, so 1 ≥ 0 . 9 + 0 . 54 , a contradiction. Thus ∥ µ -θ ∥ &lt; 4 dr .

Because 4 dr &lt; R we have µ ∈ B ( θ, R ) and here ∇ log p = ∇ log q ; consequently ∇ log p ( µ ) = 0 . Putting θ ′ = µ completes the proof.

Lemma E.15. Let p be a distribution on R d . Let ˜ x = x true + N (0 , σ 2 I d ) be a Gaussian measurement of x true ∼ p . Let p ˜ x ( x ) be the posterior distribution of x given ˜ x . Then, for any δ ∈ (0 , 1) and δ ′ ∈ (0 , 1) , with probability at least 1 -δ ′ over ˜ x ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let Q ( ˜ x ) = Pr x ∼ p ˜ x [ ∥ x -˜ x ∥ &gt; r ] . We want to show that with probability at least 1 -δ ′ over ˜ x , Q ( ˜ x ) ≤ δ . This is equivalent to showing that Pr ˜ x [ Q ( ˜ x ) &gt; δ ] ≤ δ ′ .

We use Markov's inequality. For any δ &gt; 0 :

<!-- formula-not-decoded -->

Thus, it suffices to show that E ˜ x [ Q ( ˜ x )] ≤ δδ ′ .

Let's compute E ˜ x [ Q ( ˜ x )] :

<!-- formula-not-decoded -->

Using p ( x 1 , ˜ x ) = p ( ˜ x | x 1 ) p ( x 1 ) , we can change the order of integration:

<!-- formula-not-decoded -->

Given x 1 , the distribution of ˜ x is N ( x 1 , σ 2 I d ) . Let Z = ˜ x -x 1 . Then Z ∼ N (0 , σ 2 I d ) . The inner integral is Pr Z ∼ N (0 ,σ 2 I d ) [ ∥ Z ∥ &gt; r ] . Let W = Z/σ . Then W ∼ N (0 , I d ) . The inner integral becomes P G ( r/σ ) = Pr W ∼ N (0 ,I d ) [ ∥ W ∥ &gt; r/σ ] . So, E ˜ x [ Q ( ˜ x )] = ∫ p ( x 1 ) P G ( r/σ ) dx 1 = P G ( r/σ ) .

We need to show P G ( r/σ ) ≤ δδ ′ . We use the standard Gaussian concentration inequality: for W ∼ N (0 , I d ) and t ≥ 0 ,

<!-- formula-not-decoded -->

We want P G ( r/σ ) ≤ δδ ′ . So we set e -t 2 / 2 = δδ ′ . This implies t 2 / 2 = log(1 / ( δδ ′ )) , so t = √ 2 log(1 / ( δδ ′ )) . This choice of t is real and non-negative since δ, δ ′ ∈ (0 , 1) implies δδ ′ ∈ (0 , 1) , so log(1 / ( δδ ′ )) ≥ 0 . We set r/σ = √ d + t = √ d + √ 2 log(1 / ( δδ ′ )) . Thus, for r = σ ( √ d + √ 2 log 1 δδ ′ ) , we have P G ( r/σ ) ≤ δδ ′ .

With this choice of r , we have E ˜ x [ Q ( ˜ x )] ≤ δδ ′ . By Markov's inequality,

<!-- formula-not-decoded -->

This means that Pr ˜ x [ Q ( ˜ x ) ≤ δ ] ≥ 1 -δ ′ , which is the desired statement:

<!-- formula-not-decoded -->

## F Why Standard Langevin Dynamics Fails

As discussed in Section 3, after we get an initial sample X 0 ∼ p on the manifold, a natural attempt to get a sample from p y is to simply run vanilla Langevin SDE starting from X 0 :

<!-- formula-not-decoded -->

where ̂ s ( x ) is an approximation to the true score ∇ log p ( x ) . We now show that under any L p score accuracy assumption, the score error could get exponentially large as the dynamics evolves.

Averaging over y does not preserve the prior law. We first consider the simplest one-dimensional Gaussian case of (13). Suppose p = N (0 , 1) , A = 1 , and noise ξ = N (0 , η 2 ) ; so y ∼ N (0 , 1 + η 2 ) . Then with the perfect score estimator ̂ s ( X t ) = ∇ log p ( X t ) = -X t , (13) reduces to

<!-- formula-not-decoded -->

Recall that the hope of guaranteeing the robustness using only an L p guarantee is that at any time t , averaging X t over y will preserve the original law p . We now show that this hope is unfounded even in this simplest case.

Lemma F.1. Let X t follow (14) . Averaging over y ∼ N (0 , 1 + η 2 ) , X t is Gaussian with mean 0 and variance

<!-- formula-not-decoded -->

where α := 1+ η 2 η 2 &gt; 1 . In particular, Var( X t ) = 1 -1 2(1+ η 2 ) at time t ⋆ := η 2 ln 2 1 + η 2 .

Proof. Write the mild solution of (14):

<!-- formula-not-decoded -->

Because X 0 , B are independent of y , conditional moments are

<!-- formula-not-decoded -->

Applying the law of total variance with Var( y ) = 1 + η 2 gives the stated formula.

Since X 0 and B are independent of y , conditioning on y gives

<!-- formula-not-decoded -->

By the law of total variance and Var( y ) = 1 + η 2 ,

<!-- formula-not-decoded -->

Using α = (1 + η 2 ) /η 2 and simple algebra, this simplifies to

<!-- formula-not-decoded -->

which is at most 1 and attains 1 -1 / [2(1 + η 2 )] when e -αt = 1 / 2 , that is at t ⋆ .

Thus Var( X t ) first shrinks below 1 (by a constant factor bounded away from 1 when η is small) before relaxing back to equilibrium. The phenomenon is harmless in one dimension but is catastrophic in high dimension.

High-dimensional amplification. Let p = N (0 , I d ) , take A = I d , and set η 2 = 0 . 1 . Then with the perfect score estimator, (13) reduces to

<!-- formula-not-decoded -->

By Lemma F.1 applied coordinatewise, at time t ⋆ := η 2 ln 2 1 + η 2 , averaging over y yields

<!-- formula-not-decoded -->

Hence X t ⋆ is exponentially more concentrated in high dimension. We next show that this concentration amplifies score-estimation errors exponentially with the dimension.

Lemma F.2. Let p = N (0 , I d ) and let p t ⋆ = N (0 , σ 2 I d ) with σ 2 = 6 11 . For any finite k &gt; 1 and 0 &lt; ε &lt; 1 , there exists a score estimate ̂ s : R d → R d such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some constant c &gt; 0 depending only on k .

Proof. Fix k &gt; 1 and 0 &lt; ε &lt; 1 . Let σ 2 = 6 11 ∈ (0 , 1) and choose ρ ∈ (0 , min { 1 / 2 , 1 /σ 2 -1 } ) . Define the shell

<!-- formula-not-decoded -->

Write m := Pr x ∼ p [ x ∈ S ρ ] and q := Pr x ∼ p t ⋆ [ x ∈ S ρ ] . Since ∥ X ∥ 2 /σ 2 ∼ χ 2 d under p t ⋆ , the chi-square concentration inequality Lemma A.11 gives

<!-- formula-not-decoded -->

Since (1 + ρ ) σ 2 &lt; 1 , the Chernoff left-tail bound for χ 2 d yields

<!-- formula-not-decoded -->

Choose any unit vector u and set

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

Moreover ∥ e ( x ) ∥≡ M on S ρ , hence

<!-- formula-not-decoded -->

Using m ≤ e -Id we have M = ε m -1 /k ≥ ε e ( I/k ) d . Setting

<!-- formula-not-decoded -->

which depends only on σ and k , gives

<!-- formula-not-decoded -->

This completes the proof.

yet

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Yes, the main claims are all reflected in the scope and contributions, as discussed in the paper.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss limitations of our algorithm and theorems in the introduction.

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

Justification: The main paper contains proof overviews, and the appendices contain the full rigorous versions of all proofs of all stated lemmas and theorems.

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

Justification: We provide details for reproducing the experiments in the paper.

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

Justification: We provide open access to the code. The datasets we use are open access. Guidelines:

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

Justification: We provide all details for the experiments in the paper

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report error bars where relevant.

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

Justification: We provide information about the compute needed in the paper.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The authors read through and reviewed the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: Our contribution is domain-agnostic foundational research, so a detailed societal-impact discussion would necessarily be high-level and speculative. We do not believe it would be a useful use of reviewer time.

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

Justification: We are not releasing data or models.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All the assets are properly credited and the license and terms of use are explicitly mentioned and properly respected.

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

Answer: [NA] .

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

Justification: There is no LLM usage in the core methods in this research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.