## Temperature is All You Need for Generalization in Langevin Dynamics and other Markov Processes

Itamar Harel ∗ Technion

Yonathan Wolanowsky Technion

Nathan Srebro

Toyota Technological Institute at Chicago

## Abstract

We analyze the generalization gap (gap between the training and test errors) when training a potentially over-parametrized model using a Markovian stochastic training algorithm, initialized from some distribution θ 0 ∼ p 0 . We focus on Langevin dynamics with a positive temperature β -1 , i.e. gradient descent on a training loss L with infinitesimal step size, perturbed with β -1 -variances Gaussian noise, and lightly regularized or bounded. There, we bound the generalization gap, at any time during training , by √ ( β E L ( θ 0 ) + ln(1 /δ )) /N with probability 1 -δ over the dataset, where N is the sample size, and E L ( θ 0 ) = O (1) with standard initialization scaling. In contrast to previous guarantees, we have no dependence on either training time or reliance on mixing, nor a dependence on dimensionality, gradient norms, or any other properties of the loss or model. This guarantee follows from a general analysis of any Markov process-based training that has a Gibbs-style stationary distribution. The proof is surprisingly simple, once we observe that the marginal distribution divergence from initialization remains bounded, as implied by a generalized second law of thermodynamics.

## 1 Introduction

One main goal of contemporary machine learning theory is to predict a model's behavior before training occurs. A commonly desired metric is the generalization of overparameterized models, such as neural networks (NN). For these models, such a predictive theory of generalization is still lacking, despite great empirical success [71, 23]. In particular, a significant line of work aimed to explain the role of optimization in generalization (e.g. [23, 64, 40, 66]), and specifically the effect of stochasticity (e.g. [59, 49, 10, 8]).

Data-dependent Markov processes are a common optimization approach. These include stochastic gradient descent (SGD), as well as other stochastic gradient methods either studied theoretically [30, 59], or used in practice such as SGD with momentum [52], ADAM [34], and many more. Of particular interest are continuous Langevin dynamics (CLD) and discrete analogues of it, which have been studied extensively as models for SGD (see Section 4.1).

In Section 2 we develop, for the first time, a generalization bound applicable to any data-dependent Markov process with a Gibbs-type stationary distribution (i.e. whose finite density exists and is nonzero w.r.t. some data-independent base measure). An important feature of our analysis is that it is entirely independent of the training time t , both in that we do not rely on training for only a

∗ Corresponding author: itamarharel01@gmail.com

Gal Vardi Weizmann Institute of Science

Daniel Soudry Technion

small number of steps, nor that we rely on mixing - the guarantees are valid at any time, with no dependence at all on t . Furthermore, it is also completely trajectory independent.

In Section 3 we apply these general results to the particular case where training is done with CLD with loss L and inverse temperature β , deriving a particularly simple generalization bound for CLD, which we compare to previous generalization bounds for CLD in Section 4, as well as discussing other related work. Finally, we address limitations and future work in Section 5.

To prove these results, we first show in Section 2 how, for the marginal distribution at time t , p t , its divergence (either KL or the Rényi infinity divergence) from initialization is bounded due to its monotonicity, i.e. a generalized second law of thermodynamics [11, 46]. This surprisingly simple derivation 2 leads to our key technical result (Corollary 2.5). Standard PAC-Bayes generalization bounds [43] then yield our generalization bounds (Theorem 2.7 and Corollary 3.1).

## 2 Generalization Bounds for General Markov Process

In this Section, we consider general data-dependent Markov processes over predictors and obtain a bound on their generalization gap. Importantly, although the bound only depends on the initialization distribution and a stationary distribution, it will apply to predictors at any time t ≥ 0 along the Markov process. Our main goal is to apply these bounds to stochastic training methods, such as Langevin dynamics, where the iterates form a data-dependent Markov process. But to emphasize the broad generality of the results, in this section we consider a generic stochastic optimization framework and general data-dependent Markov processes.

We obtain generalization guarantees by bounding the KL-divergence (or, for high probability bounds, the Rényi infinity divergence, see Definition 2.1) between the data-dependent marginal distribution p t of the predictors at time t , and some data-independent base measure ν (the PAC-Bayes 'prior'). The crux of the analysis is therefore bounding the divergence between p t and ν , based only on assumptions on the initial distribution p 0 (specifically, the divergence between p 0 and ν ) and a stationary distribution p ∞ (specifically, requiring that p ∞ can be expressed as a Gibbs distribution with bounded potential or expected potential, see Definition 2.2) - we do this in Section 2.1. Then, in Section 2.2 we plug these bounds on the divergence between p t and ν into standard PAC-Bayes bounds to obtain the desired generalization guarantees.

Detailed proofs of all the results in this section can be found in Appendix B.

## 2.1 Bounding the Divergence of a Markov Process

In this subsection, we consider a general time-invariant Markov process 3 h t ∈ H over a state space H . The Markov process can be either in discrete or continuous time, i.e. we can think of t as either an integer or a real index. We denote by p t the marginal distribution at time t , i.e. h t ∼ p t . We do not assume that the Markov process is ergodic, and all our results will rely on the existence of some stationary distribution p ∞ . The main goal of this subsection is to bound the divergence D ( p t ∥ ν ) between the marginal distribution at time t and some reference distribution ν . We can think of a bound on the divergence as ensuring high entropy relative to ν , or in other words that p t does not concentrate too much relative to ν , i.e. does not have too much probability mass in a small ν -region. We present all bounds for both the KL-divergence KL( p ∥ q ) and the Rényi infinity divergence D ∞ ( p ∥ q ) , defined below.

Divergences and Gibbs distributions. We recall the definitions of our two divergences, and also relate them to the Gibbs distribution. It will also be convenient for us to introduce 'relative' versions of divergences.

2 e.g. to bound the KL divergence of a Markov process having a stationary distribution with potential Ψ ∈ [0 , ∞ ) , i.e. d p ∞ / d p 0 ∝ e -Ψ (e.g., Ψ = βL for CLD), the second law implies the first inequality below

<!-- formula-not-decoded -->

3 Formally stated: we require that for any 0 ≤ t 1 &lt; t 2 &lt; t 3 we have that h t 3 is independent of h t 1 conditioned on h t 2 (Markov property) and that for any 0 ≤ t 1 , t 2 , ∆ we have that h t 1 +∆ | h t 1 has the same conditional distribution as h t 2 +∆ | h t 2 (time-invariance).

Definition 2.1 (Divergences 4 ) . For probability distributions p, q and µ :

1. The µ -weighted Kullback-Leibler (KL) divergence (a.k.a. relative cross-entropy) is 5 KL µ ( p ∥ q ) = ∫ d µ ln d p d q , and the KL-divergence is then KL( p ∥ q ) = KL p ( p ∥ q ) .
2. The Rényi infinity divergence is 6 D µ ∞ ( p ∥ q ) = ess sup µ ln d p d q , with D ∞ ( p ∥ q ) = D p ∞ ( p ∥ q )
3. .

Definition 2.2 (Gibbs distribution) . A distribution p is Gibbs w.r.t. a base distribution q with potential Ψ: H → R if Z = ∫ e -Ψ d q &lt; ∞ and

<!-- formula-not-decoded -->

Claim 2.3. If p, q, µ, ν are probability measures, and p is Gibbs w.r.t. q with potential Ψ &lt; ∞ , then

1. KL µ ( p ∥ q ) + KL ν ( q ∥ p ) = E ν Ψ -E µ Ψ ,
2. D µ ∞ ( p ∥ q ) + D ν ∞ ( q ∥ p ) = ess sup ν Ψ -ess inf µ Ψ .

<!-- formula-not-decoded -->

That is, the potential of a Gibbs distribution p allows us to bound the divergence in both directions between p and the base measure q . A generalized converse of Claim 2.3 also holds, and we have that bounding on the symmetrized divergences (but not just on one direction!) is also sufficient for p being Gibbs with a bounded potential. 7

Second Law of Thermodynamics. Central to our analysis is the following monotonicity property on the divergence between the marginal distribution of a Markov process and any stationary distribution.

Claim 2.4 (Cover's Second Law of Thermodynamics) . Let p t be the marginal distribution of a time-invariant Markov process, and p ∞ a stationary distribution for the transitions of the Markov process (the process need not be ergodic, and p t need not converge to p ∞ ). Then for any t ≥ 0

<!-- formula-not-decoded -->

When the stationary distribution is uniform (thus having maximal entropy), the KL-form of Claim 2.4 recovers the familiar second law of thermodynamics, i.e. that the entropy is monotonically nondecreasing. The more general form, as in Claim 2.4, is a direct consequence of the data processing inequality, as pointed out by Theorem 4 of Cover [11] (see also [12, 46] and the generalization to Rényi divergences in [65, Theorem 9 and Example 2] -for completeness we provide a proof in Appendix A.2).

In our case, the stationary distribution p ∞ will not be uniform, but rather will be very data-dependent (we are interested mostly in processes that aim to optimize some data-dependent quantity, such as Langevin dynamics). Nevertheless, we do want to use Claim 2.4 to control the entropy of p t relative to some benign data-independent base distribution ν (which we can informally think of as 'uniform'). To do so, we can use the chain rule and plug in Claim 2.4 to obtain that for any distribution ν and at any time t we have (see Lemma B.1 in Appendix B for the full derivation):

<!-- formula-not-decoded -->

and similarly,

<!-- formula-not-decoded -->

4 The term 'divergence' is a slight abuse of notation, as the following definitions are not strictly non-negative, without specifying µ .

5 For two measures p and q , d p/ d q is the Radon-Nikodym derivative (i.e. the density of p w.r.t. q ) when it exists (i.e. when p ≪ q , i.e. p is absolutely continuous w.r.t. q ), or ∞ otherwise.

6 The essential supremum of a function f w.r.t. a measure µ is ess sup µ f = inf { b ∈ R | µ ( f &gt; b ) = 0 } , i.e. the smallest (infimum) number that bounds f from above almost everywhere. The essential infimum is defined similarly.

7 More formally: KL( p ∥ q ) + KL( q ∥ p ) ≤ β iff there exists a potential Ψ such that p is Gibbs w.r.t. q with potential Ψ and E q Ψ -E p Ψ ≤ β , and similarly D ∞ ( p ∥ q ) + D ∞ ( q ∥ p ) ≤ β iff there exists a potential 0 ≤ Ψ ≤ β such that p is Gibbs w.r.t. q with potential Ψ . See Claim B.8 for a proof.

Bounding the last two terms in (1) and (2) using Claim 2.3 we obtain the main result of this subsection:

Corollary 2.5. For any distribution ν and any time-invariant Markov process, and any stationary distribution p ∞ that is Gibbs w.r.t. ν with potential Ψ ≥ 0 (the Markov chain need not be ergodic, and need not converge to p ∞ ), at any time t ≥ 0 :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The important feature of Corollary 2.5 is that it bounds the divergence at any time t , in terms of a right-hand side that depends only on the initial distribution p 0 and a stationary distribution p ∞ . Interpreting the divergence D ( p t ∥ ν ) as a measure of concentration, the Corollary ensures that at no point during its run, and regardless of mixing, does the Markov process concentrate too much, and it always maintains high entropy (relative to the base measure ν ).

Remark 2.6 . In order to bound the divergence D ( p t ∥ ν ) at finite time t , it is not enough to rely only on the divergences D ( p 0 ∥ ν ) and D ( p ∞ ∥ ν ) from the initial and stationary distributions, and it is necessary to rely also on the reverse divergence D ( ν ∥ p ∞ ) -see Appendix C.

## 2.2 From Divergences to Generalization

Corollary 2.5 can be directly used to obtain PAC-Bayes type generalization guarantees. Specifically, we consider a generic stochastic optimization setting specified by a bounded instantaneous objective f : H×Z → [0 , 1] over a class H , which we will refer to as the 'predictor' class, and instance domain Z . For example, in supervised learning Z = X × Y , H ⊆ Y X and f ( h, ( x, y )) = I { h ( x ) = y } measures the error of predicting h ( x ) when the correct label is y . For a source distribution D over Z and data S ∼ D N of size N we would like to relate the population and empirical objectives

̸

<!-- formula-not-decoded -->

In our case, we are interested in predictors generated by a data-dependent Markov process h t . That is, conditioned on the data S , { h t } t ≥ 0 is a time-invariant Markov process, specified by some (possibly data-dependent) initial distribution p 0 ( h 0 ; S ) , and a transition distribution that would also depend on the data S , and specifies a (randomized) rule for generating the next iterate h t +1 (if in discrete time) from the current iterate h t and the data S (as in, e.g., stochastic gradient descent or stochastic gradient Langevin dynamics; SGLD).

We present two types of generalization guarantees: guarantees that hold in expectation over a draw from the Markov process ((6) below) and guarantees that hold with high probability over a single draw from the Markov process (as in (7), e.g. a single run of CLD). In both cases, the guarantees hold with high probability over the training set.

Theorem 2.7. Consider any distribution D over Z , function f : H×Z → [0 , 1] , sample size N ≥ 8 , and any distribution ν over H . Let { h t ∈ H} t ≥ 0 be a discrete or continuous time process (i.e. t ∈ Z + or t ∈ R + ) that is time-invariant Markov conditioned on S , that starts from an initial distribution p 0 ( · ; S ) (that may depend on S ), and admits a stationary distribution conditioned on S , p ∞ ( · ; S ) . Let Ψ S ( h ) ≥ 0 be a non-negative potential function and assume that p ∞ ( · ; S ) is Gibbs w.r.t. ν with potential Ψ S . Then:

1. with probability 1 -δ over S ∼ D N ,

<!-- formula-not-decoded -->

2. with probability 1 -δ over S ∼ D N and over h t :

<!-- formula-not-decoded -->

Proof. The Theorem follows immediately by plugging the divergence bounds of Corollary 2.5 into standard PAC-Bayes guarantees, which we do in Appendix B.

Remark 2.8 . A simplified variant of Theorem 2.7 can be stated when the initial distribution p 0 is data-independent and always equal to ν . In this case the divergence between p 0 and ν vanishes, and (6) and (7) become

<!-- formula-not-decoded -->

̸

But allowing p 0 = ν is more general, as it both allows using a data-dependent initialization (recall that ν must be data independent) and it allows initializing to a distribution where D ( p ∞ ∥ p 0 ) is infinite - e.g., we can allow initializing to a degenerate initial distribution p 0 whose support is a strict subset of the support of p ∞ (in which case p ∞ will definitely not be Gibbs w.r.t. p 0 ), as long as the ν -mass of the support of p 0 is not too small.

Remark 2.9 . In Theorem 2.7, the Markov process need not be ergodic, and need not converge to p ∞ , or converge at all. If there are multiple stationary distributions, the theorem holds for all of them, and so we can take p ∞ to be any stationary distribution we want. And in any case, there is no mixing requirement, and the theorem holds at any time t .

Remark 2.10 . Our data-dependent Markov process of interest, and in particular CLD and SGD, might aim to minimize E S ( h t ) , and the potential Ψ might also be related to it (as in, e.g., CLD). This is allowed, but is in no way required in Theorem 2.7. Even for CLD, these might be related but not the same, as we might be minimizing a surrogate loss, such as a logistic loss, but are interested in bounding the generalization gap for a zero-one error. In stating Theorem 2.7 we intentionally refer to an arbitrary stochastic optimization problem and an arbitrary data-dependent Markov process, that are allowed to be related or dependent in arbitrary ways.

Remark 2.11 . In Appendix C we show that in order to ensure generalization at every intermediate t , it is not sufficient to only bound KL( p ∞ ∥ ν ) or D ∞ ( p ∞ ∥ ν ) , and we do need the stronger symmetric bound ensured by the Gibbs potential and Claim 2.3; and that it is also necessary to relate both p 0 and p ∞ to the same data independent distribution ν , as relating them to different data-independent distributions ensures generalization at the beginning and at the end, but not the middle of training.

Remark 2.12 . In Theorem 2.7 we plugged Corollary 2.5 into a simplified PAC-Bayes bound that allows for easy interpretation and comparison with other results. But once we have the divergence bounds of Corollary 2.5, we can just as easily plug them into tighter PAC-Bayes bounds - see Appendix B. For example, when E S ( h t ) ≈ 0 , these yield a rate of O (1 /N ) .

## 3 Special Case: Continuous Langevin Dynamics

Clearly, given Theorem 2.7 all we need to do in order to derive explicit generalization bounds for any Markovian training procedure, is to find a stationary distribution, and bound its potential (or its expectation at p 0 ). In this section, we will exemplify our results in a few special cases of continuous-time Langevin dynamics (CLD), a commonly studied approximation for NN training with 'infinitesimal learning rate' (e.g. [41], see Section 4.1 for additional references), which have a normalized stationary distribution that we can write analytically.

Additional notation. In the following, it will be convenient to consider a parametric model. Specifically, we assume that there exists some parameter space Θ ⊆ R d that parameterizes a hypothesis class H ⊆ Y X via a mapping Θ ∋ θ ↦→ h θ ∈ H , and assume Markovian dynamics in parameter space , instead of in the hypothesis space (note that Markov processes in parameter space may not be Markovian in hypothesis space, but the same generalization results apply ). We shall also use, with some abuse of notation, φ ( θ ) = φ ( h θ ) for any data-dependent or data-independent function φ over hypotheses, e.g. a training loss/objective L S w.r.t a training set S . Finally, we use C 2 to denote the space of twice continuously differentiable functions on Θ .

CLD in a bounded domain. Let Θ be a box in R d , and suppose that training is modeled with CLD in a bounded domain, i.e. that the parameters evolve according to the stochastic differential equation with reflection at the boundary (SDER)

<!-- formula-not-decoded -->

where L S ≥ 0 is twice continuously differentiable, w t is a standard Brownian motion, and r t is a reflection process that constrains θ t within Θ . Such weight clipping is quite common in practical scenarios such as NN training. For simplicity, we assume that r t has normal reflection, meaning that the reflection is perpendicular to the boundary. An established result in the analysis of SDERs states that under these assumptions (9) has a stationary distribution p ∞ ( θ ) ∝ e -βL S ( θ ) I Θ { θ } (see Appendix H.2). Thus, when p 0 = Uniform(Θ) , we have p 0 = ν .

Regularized CLD in R d . Suppose that the parameters evolve according to the stochastic differential equation (SDE) with weight decay (i.e. ℓ 2 regularization)

<!-- formula-not-decoded -->

where L S ≥ 0 is twice continuously differentiable, w t is a standard Brownian motion. Such weight decay is also quite common in practical scenarios such as NN training. Similar to the previous case, with the regularization and twice continuous differentiability of L S this process has a unique stationary distribution p ∞ ( θ ) ∝ e -βL S ( θ ) ϕ λ ( θ ) , where ϕ λ is the density of the multivariate Gaussian N ( 0 , λ -1 I d ) . Thus, when p 0 = N ( 0 , λ -1 I d ) , we also have p 0 = ν .

We can now formulate a generalization bound for both cases.

Corollary 3.1. Assume that the parameters evolve according to either (9) with p 0 = Uniform(Θ) , or (10) with p 0 = N ( 0 , λ -1 I d ) . Then for any time t ≥ 0 , and δ ∈ (0 , 1) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2. w.p. 1 -δ over S ∼ D N and θ t ∼ p t

<!-- formula-not-decoded -->

The proof is simple - by assumption, in both cases p 0 = ν so D ∞ ( p 0 ∥ ν ) = 0 . The rest is a direct substitution into Theorem 2.7, and in particular, using βL S as potential Ψ S .

## 3.1 Interpreting Corollary 3.1

Corollary 3.1 raises questions on the relevance of this setting, which we address below: (1) How large is E p 0 L S ( θ ) in practically relevant cases? (2) Can we attribute the generalization to the regularization (either with the ℓ 2 regularization term, or the bounded domain)? (3) Can models successfully train in the presence of noise with a variance large enough to make the bounds non-vacuous?

Magnitude of the initial loss. Commonly, the dependence on E p 0 L S ( θ ) with realistic p 0 and L S is relatively mild. For example, using standard initialization schemes, Gaussian process approximations [50, 42, 35, 25] imply that the output of an infinitely wide fully connected neural network converges to a Gaussian with mean 0 and O (1) variance at initialization. So in many cases E p 0 L S ( θ ) = O (1) , such as for the scalar square and logistic losses. In the multi-output case, E p 0 L S ( θ ) may also depend on the number of outputs (e.g., logarithmically so in softmax-cross-entropy). A more difficult question is concerned with the case that ess sup p 0 L S = ∞ , which is common when p 0 has infinite support. This can be mitigated by clipping the loss, which is standard in practice (e.g. in reinforcement learning [48, 62]) and in the theory of optimization [37, 33]. Moreover, this clipping can be done in a differentiable way (e.g. using either softmin, tanh (e.g. c · tanh( L/c ) ), etc) and at values only slightly higher than the typical loss at the initialization (since the loss is roughly monotonically decreasing in CLD with small noise, the optimization process would typically operate below the clipping and will not be affected by it).

Magnitude of regularization. In the above result we must use regularization (or a bounded domain) that matches the initialization p 0 (this can be somewhat relaxed, see Section 3.2). The same assumption, that the regularization matches the initialization, was also made in other theoretical works on CLD [49, 38, 19]. Note that, NN models regularized this way remain highly expressive, both empirically (Appendix F) and theoretically (Appendix G), and therefore we cannot use this regularization

alone, together with classical uniform convergence approaches to show generalization. Intuitively, this is because the regularization term can be tiny, for example, in (10) the regularization term is divided by β . Therefore, when β = O ( N ) (which is sufficient for a non-vacuous result), p 0 = ν , and we use a standard deep nets initialization distribution p 0 (e.g., [21, 28], where λ ∝ layer width ), the regularization coefficient is O ( layer width N ) that is rather small in realistic cases. Therefore, we found (empirically) that it does not seem to have a large effect at practical timescales. In addition, one can always increase the regularization by modifying the loss L S ← L S + c ∥ θ ∥ 2 in (10). Under standard initializations, this changes the loss in the bound by an O ( c ˜ d ) factor, where ˜ d is the depth of the neural network and so c ˜ d is small, for common values of c and ˜ d . Therefore, combining these observations, we do not see the magnitude of the regularization as a significant practical issue.

Magnitude of noise: theoretical perspective. In the above result we must use β = O ( N ) to obtain a non-vacuous bound. This requirement is standard in many theoretical works. For example, as we will discuss below in Section 4.1, all previous generalization bounds for CLD and SGLD also required, to generalize well, β = O ( N ) and potentially much worse (lower β ). In addition, other theoretical works on noisy training also typically had β = O ( N ) or worse. For example, when considering the ability of noisy gradient descent to escape saddle points, Jin et al. [30] uses noise sampled uniformly from a ball with a radius that depends on the dimensionality and smoothness of the problem, and thus cannot decay with N . Moreover, it is known that the Gibbs posterior 8 generalizes well with β = O ( √ N ) (e.g. see Theorem 2.8 in 1), which is significantly smaller than β = O ( N ) .

Lastly, in Appendix E we examine the impact of β in the simple model of linear regression with i.i.d. standard Gaussian input, labels produced by a constant-magnitude teacher label noise, trained using regularized CLD as in (10), with λ ∝ d to match standard initialization. We find there that whenever d ≪ β ≪ N , the added noise does not significantly affect the training or population losses, and our bound is useful, i.e., it implies a vanishing generalization gap (since β ≪ N and E p 0 L = O (1) ). Note that d ≪ N is not a major constraint, since d ≪ N is required to obtain low population loss in this setting, even if we did not add noise to the training process (i.e. β = ∞ ).

Magnitude of noise: empirical perspective. An inverse temperature of β = O ( N ) is also relevant in many practical settings. For example, in Bayesian settings, when we wish to (approximately) sample from the posterior, it is quite common to use variants of SGLD; then inverse temperatures of order β = O ( N ) are commonly used to achieve good generalization [69], which matches our results. In the standard practical training settings, the inverse temperature is a hyperparameter tuned to best fit a given problem. Empirically, in Appendix F we find that β = O ( N ) can be tuned to obtain nonvacuous generalization bounds for overparameterized NNs in a few small binary classification datasets (binary MNIST, Fashion MNIST, SVHN, and a parity problem), i.e. the sum of the generalization gap bound and the training error is smaller than 0 . 5 . Importantly, these non-vacuous bounds do not use any trajectory-dependent quantities as other non-vacuous bounds (e.g. [15, 39]), which can make them arguably more useful as they can be calculated before training. The bounds are still not very tight (at noise levels that allow for non-vacuous bounds), but we believe there is still much room for improvement in future work.

## 3.2 Extensions and Modifications

State dependent diffusion coefficient. Consider a state-dependent diffusion coefficient

<!-- formula-not-decoded -->

where σ 2 ∈ C 2 . For example, in Appendix D.1 we derive the explicit form of stationary distributions when σ 2 ( θ ) = ( L S ( θ ) + α ) k or σ 2 ( θ ) = e αL S ( θ ) , for some k ∈ N and α &gt; 0 . In both cases, the analytic form of the stationary potential Ψ can be used directly with Theorem 2.7 to derive generalization bounds.

Restricted initialization. In Appendix D.2 we present generalizations of Corollary 3.1 to cases where p 0 and ν are different. Specifically, for the bounded case we consider p 0 that is uniform in a subset Θ 0 ⊂ Θ of the domain, and for the regularized case we consider general diagonal Gaussian initialization and regularization. In particular, this means that some of the parameters can be more

8 Generalization bounds for the Gibbs posterior typically assume that it is 'trained' and 'tested' on the same function, while here the distribution is defined by the loss and 'tested' on the error.

Table 1: Comparison of generalization bounds for CLD. We compare the main bounds in settings similar to the CLD setting considered here. All the bounds here consider different functions for training and evaluation, as was done in this paper with L S and E S , E D , respectively. For simplicity, we assume that E S , E D are bounded in [0 , 1] , and are therefore 1 / 2 -subGaussian via Hoeffding's inequality. We use g t to denote trajectory-dependent statistics of the gradients, K for the Lipschitz constant, and C for a bound on the loss, or the expected loss at initialization, when they are required. For compactness, low-order terms are omitted, time-dependent quantities are simplified to an approximate asymptotic value, and trajectory dependent integrals are solved by considering the statistics g t constant w.r.t. the variable of integration. Finally, all bounds assume a Gaussian initialization N ( 0 , λ -1 I d ) and regularization term λ 2 β ∥ θ t ∥ 2 , both with the same λ .

| Paper                                                   | Trajectory dependent   | dimension dependence                          | Bound (big O )                                                    |
|---------------------------------------------------------|------------------------|-----------------------------------------------|-------------------------------------------------------------------|
| Mou et al. [49] Li et al. [38] Futami and Fujisawa [19] | ✓ ✗ ✓                  | through gradients through K through gradients | √ β N · √ 1 λ g 2 t e 4 βC √ β N · 2 K √ λ √ β N e 8 βC · √ 1 λ g |

loosely regularized/bounded at a cost proportional to their number. For example, in a deep NN, if only a single layer is loosely regularized/bounded, the KL-divergence cost will be proportional only to the number of parameters in that layer, not the entire d .

## 4 Related Work

Information theoretic guarantees and PAC-Bayes theory. A common type of generalization bounds consists of a measure of the dependence between the learned model and the dataset used to train it, such as the mutual information between the data and algorithm [58, 70, 61] or the KLdivergence between the predictor's distribution and any data-independent distribution [44, 9, 1]. In particular, recent works were able to estimate such dependence measures from trained models to derive non-vacuous generalization bounds, even for deep overparameterized models. For example, Dziugaite et al. [17] used held-out data to bound the KL-divergence in a PAC-Bayes bound with a datadependent prior. Other works used some property of the trained model to estimate the information content, adding valuable insight to the mechanisms facilitating the successful generalization, such as the size of the compressed model after training, due to noise stability [3], and data structure [39].

Generalization of the Gibbs posterior. One classic result in the PAC-Bayesian theory of generalization is that the Gibbs posterior with properly tuned temperature minimizes the PAC-Bayes bound of McAllester [44], i.e. the KL-regularized expected loss. Raginsky et al. [59] used uniform stability [7] to derive a different generalization bound for sampling from the Gibbs distribution. Due to these known generalization capabilities, some works relied on it to derive bounds for related algorithms.

## 4.1 Explicit Comparison for CLD

Many previous works [59, 49, 38, 18, 19, 14] derived generalization bounds specifically for CLD, under different assumptions. Our bound offers some improvements over previous ones:

- It is trajectory independent, and does not require gradient statistics [49, 19].
- It does not require very large time scales to make sure we have already converged near Gibbs [59], nor does it deteriorate with time, as is common for stability-based bounds [49, 14].
- It does not depend on the dimension of the parameters, neither explicitly through constants [18], nor implicitly, e.g. through the Lipschitz constant or the norms of the gradients [49, 38, 19]. In particular, as previously discussed, using standard initialization, our in-expectation bound in (11) is dimension independent. However, our high-probability bound (12) relies on the effective supremum at t = 0 , and may also depend on the dimension if the loss is not bounded.
- The dependence on the inverse temperature β and loss' (or expected loss) bound C is polynomial ( √ βC ) instead of exponential [38, 18, 19].

- The bounded expectation assumption in (11) is weaker than a uniform bound on the loss [38, 19].
- Theorem 2.7 and Corollary 3.1 demonstrate that our results hold for general initializationregularization pairs, beyond Gaussian initialization with matching ℓ 2 regularization.

In Table 1 we compare in more detail Corollary 3.1 to other bounds that remain bounded as t →∞ .

Finally, Dupuis et al. [14] recently derived bounds on the generalization gap for all intermediate times 0 ≤ s ≤ t simultaneously . Naturally, as avoiding parameters with large generalization gap is increasingly less likely as the process mixes, their bounds grow with time. Therefore, Dupuis et al. [14]'s bounds are qualitatively different, and higher than most other bounds, including ours.

## 4.2 Technical Novelty

As a representative example, we first focus on Raginsky et al. [59], which provided a bound for CLD (as an intermediate step for deriving a generalization bound for SGLD, a discretized version of CLD). Using spectral methods [e.g. 5], they bound the distance between the process' distribution to the Gibbs posterior, which, when combined with the generalization bound for the Gibbs distribution, results in generalization bounds for intermediate times. Our Corollary 2.5 and the preceding arguments are similar to the proof of Lemma 3.4 of Raginsky et al. [59] that bounds the divergence between the initialization and the Gibbs distribution, where their dissipativity coefficient m corresponds to our explicit ℓ 2 regularization coefficient λ . We use some significant observations that make the bound simpler, and time/dimension/Lipschitz/smoothness independent.

- Instead of a bound on the convergence of intermediate time distributions to Gibbs, which restricts the result to very large times and introduces exponential dependence on dimensionality through the spectral gap, we only require the monotonic convergence to it. As a result, we do not use a spectral gap, but a complexity term for the initial distribution. This also enables us to generalize the result to any Markov process, relying on E p 0 Ψ as a complexity term for the Gibbs posterior, which is also included in Lemma 3.4 of Raginsky et al. [59] along other quantities.
- By using a symmetric version of the divergence (e.g. by summing KL( p ∥ q ) and KL( q ∥ p ) ) we were able to completely remove the partition function from the analysis, avoiding the complications arising from it.
- By separating the regularization from the loss we were able to disentangle their effects.

This approach also sidesteps the main difficulties encountered by other works, e.g., using stabilitybased bounds [49, 38, 19] which either diverge with training time or have dimension dependence.

## 4.3 Generalization Guarantees Applicable for Neural Networks

Many additional lines of work established generalization guarantees applicable for NNs, but are less directly related to our work. These results have some limitations that do not exist in ours. For example, NTK analysis [29] can imply generalization guarantees in certain settings, but they do not allow for feature learning; Mean-field results [45] require non-standard initialization and specific architectures; Algorithmic stability analysis Bousquet and Elisseeff [7], Hardt et al. [26], Richards and Rabbat [60], Lei et al. [36], Wang et al. [67] only apply when the number of iterations is sufficiently small; Norm-based generalization bounds [6, 22] ignore optimization aspects and depend exponentially on the network's depth; And bounds for random interpolators [8] involve impractical training procedures.

Aclosely related setting to the one studied here is SGLD, i.e. a discretized version of CLD. There is an extensive line of work bounding the generalization gap of such models (see [59, 49, 55, 51, 18, 19, 13] for a partial list). These results typically have a significant dependence on hyperparameter stemming from the discretization such as the learning rate and batch size, or suffer from constraints similar to the ones discussed in Section 4.1, such as dependence on trajectory or dimensionality (e.g. via smoothness, parameter norms, log-Sobolev or spectral gap constants).

## 5 Discussion, Limitations, and Future Work

Summary. We derived a simple generalization bound for general parametric models trained using a Markov-process-based algorithm, where the dynamics have a stationary distribution with bounded

potential or expected potential. For CLD with regularization/boundedness constraint matching the initial distribution, we proved that the model generalizes well when the inverse temperature is of order β = O ( N ) . There are several interesting directions to extend this result.

Non-isotropic noise. We can consider a more general model for training, such as

<!-- formula-not-decoded -->

where Σ is a matrix-valued dispersion coefficient, and r t is some regularization process, such as ℓ 2 regularization or a reflection process in a bounded domain. In contrast, in this paper, to derive concrete generalization bounds, we focused on CLD with isotropic noise, i.e. such that Σ is a scalar multiple of the identity matrix. The reason for this was that our bound (Corollary 3.1) relies on explicit analytical expressions or bounds on stationary distributions, which are difficult to find in the general case. In addition, in typical overparameterized settings, the noise induced by the randomness of SGD may not only be non-isotropic, but also singular with low-rank. The analysis of such processes poses various challenges beyond the ability to derive an analytic form for their stationary distribution. For example, they may concentrate on low-dimensional manifolds, possibly making the KL-divergence term infinite, or making some of the assumptions unrealistic (e.g. the choice of initial distribution).

No regularization. In this work, we only considered processes that have stationary probability measures. For this reason, in the examples in Section 3 we used either a bounded domain or regularization. This seems essential for generalization at t →∞ , unless there are other architectural constraints. For example, consider training a model for the classification of randomly labeled data. Without regularization, a sufficiently expressive model is likely to arrive (at some point) at high training accuracy, yet it cannot generalize in this setting. Nonetheless, it might be possible to ensure generalization as a function of time, but here we focus on time-independent bounds.

Discrete time steps. The behavior of SGD with a large step size may be qualitatively different than that of the continuous process considered here. Specifically, Azizian et al. [4] showed that while the asymptotic distribution of SGD resembles the Gibbs posterior, it is influenced by the step size and geometry of the loss surface. While an extension of our analysis to this setting is straightforward given a stationary distribution, such stationary distributions are typically hard to find explicitly (except in simple cases, such as quadratic potentials), and the error terms coming from their approximations are typically detrimental to finding non-vacuous generalization bounds, as they may depend on the dimension of the parameters through the model's Lipschitz or smoothness coefficients, etc. (49, 38, 19, 14). Hence, a direct application of our approach to such algorithms requires additional considerations. An alternative approach is to incorporate a Metropolis-Hastings type rejection [47, 27], ensuring that the stationary distribution is indeed the Gibbs posterior.

Can noise be useful for generalization? There is a long line of work in the literature (e.g. see [20] and references therein), debating the effect of noise on generalization. Our work does not imply that higher noise improves the test error, only that it decreases the gap between training and testing. Since higher noise could hurt the training error, the overall effect depends on the specific situation. Even if introducing noise does not improve test performance, there could still be an advantage to introducing noise, based on our results, in that it reduces the gap and thus could increase the training error to match the test error in cases we cannot hope to learn (i.e. to get a small test error). This is a good thing since it prevents being mislead by overfitting, hopefully without hurting the test error when we can generalize well (i.e. in learnable regimes, both training and test errors are low, perhaps also without noise, but in non-learnable regimes, where the test error is necessarily high, noise forces the training error to be high as well, so that the gap is small). Indeed, in our small-scale experiments in Appendix F, we noticed that a small amount of noise can decrease the generalization gap, without significantly harming the test error (e.g. see the bottom half of Tables 2 to 4). Further analysis is necessary in order to establish general conditions under which test performance is not significantly hurt by noise, while ensuring a small gap. This, in particular, requires studying the effect of noise on the training loss, and what noise level still ensures obtaining a small training loss in learnable regimes.

## Acknowledgments and Disclosure of Funding

The research of DS was Funded by the European Union (ERC, A-B-C-Deep, 101039436). Views and opinions expressed are however those of the author only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency (ERCEA). Neither the

European Union nor the granting authority can be held responsible for them. DS also acknowledges the support of the Schmidt Career Advancement Chair in AI. GV is supported by the Israel Science Foundation (grant No. 2574/25), by a research grant from Mortimer Zuckerman (the Zuckerman STEM Leadership Program), and by research grants from the Center for New Scientists at the Weizmann Institute of Science, and the Shimon and Golde Picker - Weizmann Annual Grant. Part of this work was done as part of the NSF-Simons funded Collaboration on the Mathematics of Deep Learning. NS was partially supported by the NSF TRIPOD Institute on Data Economics Algorithms and Learning (IDEAL) and an NSF-IIS award.

## References

- [1] Pierre Alquier et al. User-friendly introduction to pac-bayes bounds. Foundations and Trends® in Machine Learning , 17(2):174-303, 2024.
- [2] Martin Anthony and Peter L Bartlett. Neural network learning: Theoretical foundations . cambridge university press, 2009.
- [3] Sanjeev Arora, Rong Ge, Behnam Neyshabur, and Yi Zhang. Stronger generalization bounds for deep nets via a compression approach. In International conference on machine learning , pages 254-263. PMLR, 2018.
- [4] Waïss Azizian, Franck Iutzeler, Jerome Malick, and Panayotis Mertikopoulos. What is the long-run distribution of stochastic gradient descent? a large deviations analysis. In Forty-first International Conference on Machine Learning , 2024. URL https://openreview.net/ forum?id=vsOF7qDNhl .
- [5] D. Bakry and M. Émery. Diffusions hypercontractives. In Jacques Azéma and Marc Yor, editors, Séminaire de Probabilités XIX 1983/84 , pages 177-206, Berlin, Heidelberg, 1985. Springer Berlin Heidelberg. ISBN 978-3-540-39397-9.
- [6] Peter L Bartlett, Dylan J Foster, and Matus J Telgarsky. Spectrally-normalized margin bounds for neural networks. Advances in neural information processing systems , 30, 2017.
- [7] Olivier Bousquet and André Elisseeff. Stability and generalization. J. Mach. Learn. Res. , 2: 499-526, March 2002. ISSN 1532-4435. doi: 10.1162/153244302760200704. URL https: //doi.org/10.1162/153244302760200704 .
- [8] Gon Buzaglo, Itamar Harel, Mor Shpigel Nacson, Alon Brutzkus, Nathan Srebro, and Daniel Soudry. How uniform random weights induce non-uniform bias: Typical interpolating neural networks generalize with narrow teachers. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning , volume 235 of Proceedings of Machine Learning Research , pages 5035-5081. PMLR, 21-27 Jul 2024. URL https://proceedings.mlr.press/v235/buzaglo24a.html .
- [9] Olivier Catoni. Pac-bayesian supervised classification: the thermodynamics of statistical learning. arXiv preprint arXiv:0712.0248 , 2007.
- [10] Ping-yeh Chiang, Renkun Ni, David Yu Miller, Arpit Bansal, Jonas Geiping, Micah Goldblum, and Tom Goldstein. Loss landscapes are all you need: Neural network generalization can be explained without the implicit bias of gradient descent. In The Eleventh International Conference on Learning Representations , 2022.
- [11] Thomas M. Cover. Which processes satisfy the second law? In J. J. Halliwell, J. Perez-Mercader, and W. H. Zurek, editors, Physical Origins of Time Asymmetry , pages 98-107. Cambridge University Press, New York, 1994.
- [12] Thomas M. Cover and Joy A. Thomas. Entropy, Relative Entropy and Mutual Information , chapter 2, pages 12-49. John Wiley &amp; Sons, Ltd, 2001. ISBN 9780471200611. doi: https: //doi.org/10.1002/0471200611.ch2. URL https://onlinelibrary.wiley.com/doi/abs/ 10.1002/0471200611.ch2 .

- [13] Leello Tadesse Dadi and Volkan Cevher. Generalization of noisy SGD in unbounded nonconvex settings. In Forty-second International Conference on Machine Learning , 2025. URL https://openreview.net/forum?id=Au9rfI6Fjd .
- [14] Benjamin Dupuis, Paul Viallard, George Deligiannidis, and Umut Simsekli. Uniform generalization bounds on data-dependent hypothesis sets via pac-bayesian theory on random sets. Journal of Machine Learning Research , 25(409):1-55, 2024.
- [15] Gintare Karolina Dziugaite and Daniel M. Roy. Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data. In Proceedings of the Conference on Uncertainty in Artificial Intelligence , 2017.
- [16] Gintare Karolina Dziugaite and Daniel M. Roy. The size of teachers as a measure of data complexity: Pac-bayes excess risk bounds and scaling laws. In Yingzhen Li, Stephan Mandt, Shipra Agrawal, and Emtiyaz Khan, editors, Proceedings of The 28th International Conference on Artificial Intelligence and Statistics , volume 258 of Proceedings of Machine Learning Research , pages 3979-3987. PMLR, 03-05 May 2025. URL https://proceedings.mlr. press/v258/dziugaite25a.html .
- [17] Gintare Karolina Dziugaite, Kyle Hsu, Waseem Gharbieh, Gabriel Arpino, and Daniel Roy. On the role of data in pac-bayes bounds. In International Conference on Artificial Intelligence and Statistics , pages 604-612. PMLR, 2021.
- [18] Tyler Farghly and Patrick Rebeschini. Time-independent generalization bounds for sgld in non-convex settings. Advances in Neural Information Processing Systems , 34:19836-19846, 2021.
- [19] Futoshi Futami and Masahiro Fujisawa. Time-independent information-theoretic generalization bounds for sgld. Advances in Neural Information Processing Systems , 36:8173-8185, 2023.
- [20] Jonas Geiping, Micah Goldblum, Phil Pope, Michael Moeller, and Tom Goldstein. Stochastic training is not necessary for generalization. In International Conference on Learning Representations , 2022. URL https://openreview.net/forum?id=ZBESeIUB5k .
- [21] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Yee Whye Teh and Mike Titterington, editors, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics , volume 9 of Proceedings of Machine Learning Research , pages 249-256, Chia Laguna Resort, Sardinia, Italy, 13-15 May 2010. PMLR. URL https://proceedings.mlr.press/v9/glorot10a.html .
- [22] Noah Golowich, Alexander Rakhlin, and Ohad Shamir. Size-independent sample complexity of neural networks. In Conference On Learning Theory , pages 297-299. PMLR, 2018.
- [23] Suriya Gunasekar, Blake E Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nati Srebro. Implicit regularization in matrix factorization. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 30. Curran Associates, Inc., 2017.
- [24] Arjun K. Gupta and Daya K. Nagar. Matrix Variate Distributions . Monographs and Surveys in Pure and Applied Mathematics. Chapman &amp; Hall/CRC, Boca Raton, FL, 1999. ISBN 9781584880462.
- [25] Boris Hanin. Random neural networks in the infinite width limit as gaussian processes. The Annals of Applied Probability , 33(6A):4798-4819, 2023.
- [26] Moritz Hardt, Ben Recht, and Yoram Singer. Train faster, generalize better: Stability of stochastic gradient descent. In International conference on machine learning , pages 1225-1234. PMLR, 2016.
- [27] W. K. Hastings. Monte carlo sampling methods using markov chains and their applications. Biometrika , 57(1):97-109, 1970. ISSN 00063444, 14643510. URL http://www.jstor.org/ stable/2334940 .

- [28] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision , pages 1026-1034, 2015.
- [29] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems , 31, 2018.
- [30] Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M Kakade, and Michael I Jordan. How to escape saddle points efficiently. In International conference on machine learning , pages 1724-1732. PMLR, 2017.
- [31] Weining Kang and Kavita Ramanan. Characterization of stationary distributions of reflected diffusions. The Annals of Applied Probability , 24(4):1329 - 1374, 2014. doi: 10.1214/ 13-AAP947. URL https://doi.org/10.1214/13-AAP947 .
- [32] Weining Kang and Kavita Ramanan. On the submartingale problem for reflected diffusions in domains with piecewise smooth boundaries. The Annals of Probability , 45(1):404 - 468, 2017. doi: 10.1214/16-AOP1153. URL https://doi.org/10.1214/16-AOP1153 .
- [33] Ali Kavis, Kfir Yehuda Levy, and Volkan Cevher. High probability bounds for a class of nonconvex algorithms with adagrad stepsize. arXiv preprint arXiv:2204.02833 , 2022.
- [34] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings , 2015. URL http://arxiv.org/abs/1412.6980 .
- [35] Jaehoon Lee, Yasaman Bahri, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, and Jascha Sohl-Dickstein. Deep neural networks as gaussian processes. In International Conference on Learning Representations (ICLR) , 2018. URL https://arxiv.org/abs/1711.00165 .
- [36] Yunwen Lei, Rong Jin, and Yiming Ying. Stability and generalization analysis of gradient methods for shallow neural networks. Advances in Neural Information Processing Systems , 35: 38557-38570, 2022.
- [37] Kfir Yehuda Levy, Ali Kavis, and Volkan Cevher. STORM+: Fully adaptive SGD with recursive momentum for nonconvex optimization. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems , 2021. URL https://openreview.net/forum?id=ytke6qKpxtr .
- [38] Jian Li, Xuanyuan Luo, and Mingda Qiao. On generalization error bounds of noisy gradient methods for non-convex learning. In International Conference on Learning Representations , 2020. URL https://openreview.net/forum?id=SkxxtgHKPS .
- [39] Sanae Lotfi, Marc Finzi, Sanyam Kapoor, Andres Potapczynski, Micah Goldblum, and Andrew G Wilson. Pac-bayes compression bounds so tight that they can explain generalization. Advances in Neural Information Processing Systems , 35:31459-31473, 2022.
- [40] Kaifeng Lyu and Jian Li. Gradient descent maximizes the margin of homogeneous neural networks. In International Conference on Learning Representations , 2020. URL https: //openreview.net/forum?id=SJeLIgBKPS .
- [41] Stephan Mandt, Matthew D Hoffman, and David M Blei. Stochastic gradient descent as approximate bayesian inference. Journal of Machine Learning Research , 18(134):1-35, 2017.
- [42] Alexander G de G Matthews, Jiri Hron, Mark Rowland, Richard E Turner, and Zoubin Ghahramani. Gaussian process behaviour in wide deep neural networks. In International Conference on Learning Representations , 2018.
- [43] Andreas Maurer. A note on the pac bayesian theorem. arXiv preprint cs/0411099 , 2004.
- [44] David A McAllester. Some pac-bayesian theorems. In Proceedings of the eleventh annual conference on Computational learning theory , pages 230-234, 1998.

- [45] Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. Proceedings of the National Academy of Sciences , 115(33): E7665-E7671, 2018.
- [46] Neri Merhav. Data processing theorems and the second law of thermodynamics. IEEE Transactions on Information Theory , 57(8):4926-4939, 2011. doi: 10.1109/TIT.2011.2159052.
- [47] Nicholas Metropolis, Arianna W. Rosenbluth, Marshall N. Rosenbluth, Augusta H. Teller, and Edward Teller. Equation of state calculations by fast computing machines. Technical report, Los Alamos Scientific Lab., Los Alamos, NM (United States); Univ. of Chicago, IL (United States), 03 1953. URL https://www.osti.gov/biblio/4390578 .
- [48] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature , 518(7540):529-533, 2015.
- [49] Wenlong Mou, Liwei Wang, Xiyu Zhai, and Kai Zheng. Generalization bounds of sgld for non-convex learning: Two theoretical viewpoints. In Conference on Learning Theory , pages 605-638. PMLR, 2018.
- [50] Radford M. Neal. Priors for Infinite Networks , pages 29-53. Springer New York, New York, NY, 1996. ISBN 978-1-4612-0745-0. doi: 10.1007/978-1-4612-0745-0\_2. URL https://doi.org/10.1007/978-1-4612-0745-0\_2 .
- [51] Jeffrey Negrea, Mahdi Haghifam, Gintare Karolina Dziugaite, Ashish Khisti, and Daniel M Roy. Information-theoretic generalization bounds for sgld via data-dependent estimates. Advances in Neural Information Processing Systems , 32, 2019.
- [52] Yurii Nesterov. A method for solving the convex programming problem with convergence rate o (1 /k 2 ) . Proceedings of the USSR Academy of Sciences , 269:543-547, 1983. URL https://api.semanticscholar.org/CorpusID:145918791 .
- [53] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y. Ng. Reading digits in natural images with unsupervised feature learning. In NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011 , 2011. URL http://ufldl.stanford. edu/housenumbers/nips2011\_housenumbers.pdf .
- [54] Bernt Øksendal. Stochastic Differential Equations , pages 65-84. Springer Berlin Heidelberg, Berlin, Heidelberg, 2003. ISBN 978-3-642-14394-6. doi: 10.1007/978-3-642-14394-6\_5. URL https://doi.org/10.1007/978-3-642-14394-6\_5 .
- [55] Ankit Pensia, Varun Jog, and Po-Ling Loh. Generalization error bounds for noisy, iterative algorithms. In 2018 IEEE International Symposium on Information Theory (ISIT) , pages 546-550. IEEE, 2018.
- [56] K. B. Petersen and M. S. Pedersen. The matrix cookbook, nov 2012. URL http://www2. compute.dtu.dk/pubdb/pubs/3274-full.html . Version 20121115.
- [57] Andrey Pilipenko. An introduction to stochastic differential equations with reflection, 09 2014.
- [58] Maxim Raginsky, Alexander Rakhlin, Matthew Tsao, Yihong Wu, and Aolin Xu. Informationtheoretic analysis of stability and bias of learning algorithms. In 2016 IEEE Information Theory Workshop (ITW) , pages 26-30, 2016. doi: 10.1109/ITW.2016.7606789.
- [59] Maxim Raginsky, Alexander Rakhlin, and Matus Telgarsky. Non-convex learning via stochastic gradient langevin dynamics: a nonasymptotic analysis. In Conference on Learning Theory , pages 1674-1703. PMLR, 2017.
- [60] Dominic Richards and Mike Rabbat. Learning with gradient descent and weakly convex losses. In International Conference on Artificial Intelligence and Statistics , pages 1990-1998. PMLR, 2021.
- [61] Daniel Russo and James Zou. How much does your data exploration overfit? controlling bias via information usage. IEEE Transactions on Information Theory , 66(1):302-323, 2020. doi: 10.1109/TIT.2019.2945779.

- [62] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- [63] Zeev Schuss. Euler's Scheme and Wiener's Measure , pages 35-88. Springer New York, New York, NY, 2013. ISBN 978-1-4614-7687-0. doi: 10.1007/978-1-4614-7687-0\_2. URL https://doi.org/10.1007/978-1-4614-7687-0\_2 .
- [64] Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. Journal of Machine Learning Research , 19 (70):1-57, 2018.
- [65] Tim Van Erven and Peter Harremos. Rényi divergence and kullback-leibler divergence. IEEE Transactions on Information Theory , 60(7):3797-3820, 2014.
- [66] Gal Vardi. On the implicit bias in deep-learning algorithms. Communications of the ACM , 66 (6):86-93, 2023.
- [67] Puyu Wang, Yunwen Lei, Di Wang, Yiming Ying, and Ding-Xuan Zhou. Generalization guarantees of gradient descent for shallow neural networks. Neural Computation , 37(2):344402, 2025.
- [68] Jonathan Wenger, Beau Coker, Juraj Marusic, and John P Cunningham. Variational deep learning via implicit regularization. arXiv preprint arXiv:2505.20235 , 2025.
- [69] Florian Wenzel, Kevin Roth, Bastiaan Veeling, Jakub Swiatkowski, Linh Tran, Stephan Mandt, Jasper Snoek, Tim Salimans, Rodolphe Jenatton, and Sebastian Nowozin. How good is the bayes posterior in deep neural networks really? In International Conference on Machine Learning , pages 10248-10259. PMLR, 2020.
- [70] Aolin Xu and Maxim Raginsky. Information-theoretic analysis of generalization capability of learning algorithms. Advances in neural information processing systems , 30, 2017.
- [71] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In International Conference on Learning Representations , 2017.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: See Sections 2 and 3.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: See Section 5.

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

Justification: See Sections 2 and 3 and appendices.

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

Justification: See Appendix F.

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

## Answer: [No]

Justification: The experiments conducted use standard models and datasets, and are described in a manner that allows for simple reproducibility.

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

Justification: The required details appear in Appendix F.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: See Appendix F.

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

Answer: [No]

Justification: The experiments are not computationally demanding and can be reproduced with basic resources.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: This work does not deviate from the code of ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The research conducted in this work is theoretical, and foundational in nature. Thus, there are no broader impacts we feel must be specifically highlighted.

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

Justification: The research conducted in this work is theoretical, and foundational in nature. Thus, there are no risks we feel must be specifically safeguarded.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We properly reference previous theoretical work throughout the paper, and the datasets used for experimentation in Appendix F.

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

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.

## Appendix structure:

- In Appendix A we recap and establish notation and conventions, and present some wellknown lemmas.
- In Appendix B we prove Theorem 2.7 and its related claims in Section 2.
- In Appendix C we discuss the tightness and necessity of the divergence conditions found in Appendix B.
- In Appendix D we prove a generalized version of Corollary 3.1.
- The bounds found in this paper only bound the generalization gap, and not the absolute error of a model. In Appendix E and Appendix F we study the applicability of our bound in realistic settings. Specifically, whether the regime in which the bound on the generalization gap is non-vacuous allows for meaningful learning, i.e. coincides with a regime in which the absolute training error is also small. In Appendix E we study linear regression trained with CLD, for which we can analytically characterize the training loss, and in Appendix F we experiment with NNs trained with SGLD (discretized version of CLD) on standard training sets.
- As Section 3 deals only with models trained with some form of regularization, it is natural to ask whether the regularization alone is sufficient for the use of uniform convergence to arrive the desired generalization bounds. In Appendix G we show that the regularization used in Section 3 is not sufficient for such bounds, and that the models can remain highly expressive.
- Finally, for completeness, in Appendix H we recall some definitions and properties related to SDEs used throughout the paper.

## A Preliminary and Auxiliary Results

## A.1 Preliminaries

We start by restating and introducing notation.

Notation. We use bold lowercase letters (e.g. x ∈ R d ) to denote vectors, bold capital letters to denote matrices (e.g. A ∈ R d × d ), and regular capital letters to denote random elements (e.g. S, X, Y ). We may deviate from these conventions when it does not create confusion. Unless stated otherwise, all vectors are assumed to be column vectors. Specifically, we use e i ∈ R d , i = 1 , . . . , d , to denote the standard basis vector with 1 in the i th entry, and 0 elsewhere. For a subset Ω ⊆ R d , we denote by Ω , ∂ Ω , and Ω ◦ , the closure, boundary, and interior of Ω , respectively. In addition, we denote the volume of B ⊂ Ω , when it is defined, by | B | . With some abuse of notation, when B is finite we denote its cardinality by | B | . We use ∥·∥ for the standard Euclidean norm on R d . Then, the open Euclidean ball centered at x ∈ R d with radius r &gt; 0 is B r ( x ) = { y ∈ R d | ∥ y -x ∥ &lt; r } . In addition, we use I {·} for the indicator function, and specifically for A ⊂ R d and x ∈ R d , I A { x } = I { x ∈ A } . We denote the set of all probability measures over Ω by ∆(Ω) . For some µ ∈ ∆(Ω) with density p , with some abuse of notation we denote p ∈ ∆(Ω) , and p ( B ) = µ ( B ) for measurable B ⊆ Ω . In addition, we use E X ∼ p or E p to denote the expectation w.r.t p , and omit the subscript when it can be inferred. For two distributions µ, ν with densities p, q we denote by KL( µ ∥ ν ) = KL( p ∥ q ) their KL-divergence (relative entropy). Furthermore, we use H ( δ ) = -δ ln ( δ ) -(1 -δ ) ln (1 -δ ) , δ ∈ [0 , 1] , for the binary entropy function (in nats). We denote the divergence of a vector field by ∇· , and the gradient and Laplacian of a scalar function by ∇ and ∆ = ∇· ∇ , respectively. Given a domain E ⊂ R k and k ∈ Z + ∪ {∞} , we denote by C k ( E ) the set of real valued functions that are continuous over ¯ E , and k -times continuously differentiable with continuous partial derivatives in E . In particular, C = C 0 is the set of continuous functions.

Conventions. Unless stated otherwise, we use Ω ⊂ R d to denote a non-empty, connected, and open domain. In addition, we follow the following naming conventions for probability distributions.

- For a discrete/continuous-time Markov process, we use p n or p t for its marginal distribution at time n ∈ N or t ∈ R + .

- We denote stationary distributions of Markov processes by p ∞ .
- In the context of PAC-Bayesian theory, we denote prior distributions by ρ , and data dependent posteriors by ˆ ρ = ˆ ρ S .
- In case some stationary distribution is also data-dependent, we use p ∞ .
- We also use p, q for generic distributions, or modify the pervious notation.

## A.2 General Lemmas: Data processing inequality and generalized second laws of thermodynamics

For completeness, we start by proving some well known results in probability and the theory of Markov processes.

Lemma A.1 (Data processing inequality) . Let p ( x, y ) and q ( x, y ) be the densities of two joint distributions over a product measure space X ×Y . Denote by p X ( x ) , q X ( x ) the marginal densities, e.g.

<!-- formula-not-decoded -->

and by p ( y | x ) , q ( y | x ) the conditional densities, so p ( x, y ) = p ( y | x ) p X ( x ) , and similarly for q . Then

<!-- formula-not-decoded -->

Proof. By definition of the KL divergence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The KL divergence is non-negative and therefore the expectation in the last line is non-negative as well, and we conclude that

<!-- formula-not-decoded -->

Let X n = { X n } ∞ n =0 be a discrete-time Markov chain on Ω ⊂ R d , with transition kernel P ( y | x ) such that for all n ∈ N 0 ,

<!-- formula-not-decoded -->

In addition, assume that the there exists an invariant distribution p ∞ such that

<!-- formula-not-decoded -->

We proceed to present a generalized form of the second law of thermodynamics, regarding the monotonicity of the (relative) entropy of Markov processes with possibly non-uniform stationary distributions [11, 12].

Lemma A.2 (Generalized second law of thermodynamics) . For all n ≥ 0 ,

<!-- formula-not-decoded -->

Proof. First, note that we can assume that KL( p n ∥ p ∞ ) &lt; ∞ , since otherwise the claim holds trivially. Let q ( x, y ) = p n ( x ) P ( y | x ) be the joint densities of ( X n , X n +1 ) where X n ∼ p n , and let r ( x, y ) = p ∞ ( x ) P ( y | x ) be the joint distribution under X n ∼ p ∞ . By definition of p n +1 ,

<!-- formula-not-decoded -->

and by definition of the stationary distribution,

<!-- formula-not-decoded -->

Therefore according to Lemma A.1,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In addition, and overall

<!-- formula-not-decoded -->

A similar result can be obtained form D ∞ ( · ∥ · ) .

Lemma A.3 (The Pointwise Second Law) . For all n &gt; 0 :

<!-- formula-not-decoded -->

Proof. Let p, q be some probability measures such that d p d q exists. By definition,

<!-- formula-not-decoded -->

Let C ∈ R and suppose that for all measurable A ⊂ X , p ( A ) ≤ e C q ( A ) . Assume by way of contradiction that D ∞ ( p ∥ q ) &gt; C , that is, that there exists c &gt; C such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote

then

<!-- formula-not-decoded -->

in contradiction to the assumption. Therefore, for all C such that p ( A ) ≤ e C q ( A ) for all measurable A , C ≥ D ∞ ( p ∥ q ) . We can now show the claim.

Let P (d y | x ) be the processes' transition kernel (in measure form). We can assume D ∞ ( p n ∥ p ∞ ) &lt; ∞ , since otherwise the claim holds trivially. Let A be measurable, then by definition,

<!-- formula-not-decoded -->

We can now state the relevant results for continuous-time processes.

Corollary A.4. Let X t be a Markov process with marginals p t and stationary distribution p ∞ . Then, for all t &gt; 0 :

<!-- formula-not-decoded -->

Proof. Let 0 &lt; t and let ∆ t &gt; 0 such that t ∈ ∆ t · N . Define Y n = X n ∆ t , then Y n is a discrete time Markov chain with marginals p n · ∆ t and stationary distribution p ∞ , so Lemma A.2 and Lemma A.3 imply the results.

## B Proof of Theorem 2.7 and its Related Claims in Section 2

In this section, we present the proof of Theorem 2.7, the claims leading to it, and some of its generalizations.

## B.1 Derivation of Corollary 2.5

Recall Claim 2.3. If p, q, µ, ν are probability measures, and p is Gibbs w.r.t q with potential Ψ &lt; ∞ , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, KL( p ∥ q ) + KL ( q ∥ p ) = E q Ψ -E p Ψ , and D ∞ ( p ∥ q ) + D ∞ ( q ∥ p ) = ess sup q Ψ -ess inf p Ψ .

Proof. By definition d p d q = Z -1 e -Ψ where Z &lt; ∞ is the appropriate partition function. Then we have

<!-- formula-not-decoded -->

Also,

<!-- formula-not-decoded -->

where in the last equality we used the fact that ess sup ( -Ψ) = -ess inf Ψ , and that Z is a constant.

Using the Chain Rule and Claim 2.4, we derive the bounds of (1) and (2), as re-stated and established in the following lemma.

Lemma B.1. If p t is the marginal distribution of a Markov process with initial distribution p 0 at time t , p ∞ is a stationary distribution, and ν is a probability measure, then

<!-- formula-not-decoded -->

and similarly,

<!-- formula-not-decoded -->

Proof. This is a simple application of the chain rule,

<!-- formula-not-decoded -->

where in the first inequality we used Claim 2.4. Similarly,

<!-- formula-not-decoded -->

Corollary 2.5 now follows from plugging in Claim 2.3 into Lemma B.1.

Given these bounds on the divergences, All that remains in order to prove Theorem 2.7 is plugging Corollary 2.5 into a PAC-Bayes bound.

## B.2 In-Expectation PAC-Bayes Bounds

Theorem B.2 (Theorem 5 from Maurer [43]) . For any δ ∈ (0 , 1) and any N ≥ 8 , for any dataindependent prior distribution ρ :

<!-- formula-not-decoded -->

where kl ( a ∥ b ) = a ln a b +(1 -a ) ln 1 -a 1 -b for 0 ≤ a, b ≤ 1 is the KL divergence for a Bernoulli random variable, and ˆ ρ denotes a posterior distribution.

## B.3 Single-Sample PAC-Bayes Bounds

Theorem B.2 can be viewed as a bound in expectation over the draw from the posterior, which corresponds to the traditional PAC-Bayes view of considering the expected error of a randomized predictor. But it is actually possible to get guarantees for a single draw from this predictor, which is more appropriate when we view the randomness as part of the training algorithm, that then outputs a single deterministic predictor (chosen at random). High probability guarantees for a single draw from the posterior were shown by Alquier et al. [1] based on Catoni [9] and also discussed by Dziugaite and Roy [16]. Here we present a tight version based on a simple modification to Maurer's proof [43].

Theorem B.3. For any δ ∈ (0 , 1) and N ≥ 8 , for any data independent prior ρ , and any learning rule specified by a conditional probability h | S ∼ ˆ ρ S such that ρ ≪ ˆ ρ S S -a.s.,

<!-- formula-not-decoded -->

and so, by the definition of D ∞ (ˆ ρ S ∥ ρ ) ,

<!-- formula-not-decoded -->

Proof. Following and modifying the proof of Theorem 5 of Maurer [43], we start with the inequality E S [ e N kl( E S ( h ) ∥ E D ( h )) ] ≤ 2 √ N [43, Theorem 1], which holds for any h , and so also in expectation over h w.r.t. ρ :

<!-- formula-not-decoded -->

with a change of measure from ρ to ˆ ρ S ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now applying Markov's inequality, we get:

<!-- formula-not-decoded -->

Rearranging terms, we get the desired bound.

## B.4 Arriving at Theorem 2.7

Theorem B.4. Consider any distribution D over Z , function f : H × Z → [0 , 1] , and sample size N ≥ 8 , any distribution ν over H , and any discrete or continuous time process { h t ∈ H} t ≥ 0 (i.e. t ∈ Z + or t ∈ R + ) that is time-invariant Markov conditioned on S . Denote p 0 ( · ; S ) the initial distribution of the Markov process (that may depend on S ). Let p ∞ ( · ; S ) be any stationary distribution of the process conditioned on S , and Ψ S ( h ) ≥ 0 a non-negative potential function that can depend arbitrarily on S , such that p ∞ ( · ; S ) is Gibbs w.r.t. ν with potential Ψ S . Then:

1. With probability 1 -δ over S ∼ D N

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2. With probability 1 -δ over S ∼ D N and over h t :

<!-- formula-not-decoded -->

and so, when E S ( h t ) &lt; E D ( h t )

<!-- formula-not-decoded -->

Lemma B.5. Let a, b ∈ [0 , 1] . Then

<!-- formula-not-decoded -->

Proof. The KL divergence is non-negative, so it suffices to consider the case that b ≥ a . Defining φ : [0 , 1 -a ] → R as

<!-- formula-not-decoded -->

it can be readily checked by differentiation that for all u ∈ [0 , 1 -a ] ,

<!-- formula-not-decoded -->

In particular, for u = b -a ∈ [0 , 1 -a ] ,

<!-- formula-not-decoded -->

Next, we consider the following inequality

<!-- formula-not-decoded -->

Solving for u , it turns out that the inequality holds when

<!-- formula-not-decoded -->

In addition, under the assumption that b ≥ a ,

<!-- formula-not-decoded -->

Combining (21), (23), and (24), u = √ kl ( a ∥ b )

solves (22) implying (20).

Proof. The inequalities (16) and (18) follow by plugging Corollary 2.5 into Theorems B.2 and B.3. For inequalities (17) and (19), we use (20). For (17), we use a = E D ( h t ) and b = E S ( h t ) , which yields:

<!-- formula-not-decoded -->

and similarly for (19).

Remark B.6 . Notice that when h t has a small training error E [ E S ( h t ) | S ] ≈ 0 , the effective generalization gap decays as O (1 /N ) instead of as O ( 1 / √ N ) .

Remark B.7 . In order to get the version in Theorem 2.7 we use the upper bound of Pinsker's inequality, i.e. that for all a, b ∈ (0 , 1)

<!-- formula-not-decoded -->

and simplify ln 2 √ N δ ≤ ln N δ as N ≥ 8 .

Finally, we prove the equivalence statement made in Footnote 7:

Claim B.8. KL( p ∥ q ) + KL ( q ∥ p ) ≤ β iff there exists a potential Ψ such that p is Gibbs w.r.t. q with potential Ψ and E q Ψ -E p Ψ ≤ β , and similarly D ∞ ( p ∥ q ) + D ∞ ( q ∥ p ) ≤ β iff there exists a potential 0 ≤ Ψ ≤ β such that p is Gibbs w.r.t. q with potential Ψ .

Proof. The first direction follows directly from Claim 2.3, so we only need to prove the converse. Assume that either KL( p ∥ q ) + KL( q ∥ p ) ≤ β , or D ∞ ( p ∥ q ) + D ∞ ( q ∥ p ) ≤ β . In these cases, both d p/ d q and d q/ d p exist, and for any measurable event B , p ( B ) = 0 ⇐⇒ q ( B ) = 0 , or equivalently, p ( B ) &gt; 0 ⇐⇒ q ( B ) &gt; 0 . Therefore, supp( p ) = supp ( q ) , and d p/ d q &gt; 0 on supp( p ) . Denote Ψ = -ln d p/ d q , then p is Gibbs w.r.t. q with potential Ψ . The same derivation as in the proof of Claim 2.3 results in the bounds E q Ψ -E p Ψ ≤ β and ess sup q Ψ -ess inf p Ψ ≤ β . In particular, if the latter holds then Ψ can be shifted such that essentially 0 ≤ Ψ ≤ β .

## C Tightness and Necessity of the Divergence Conditions

If we are only interested in ensuring generalization at time t →∞ , and when we converge to the stationary distribution p ∞ , then it is enough to bound the divergence D ( p ∞ ∥ ν ) . If we are interested in bounding D ( p t ∥ ν ) (and consequently, the generalization gap) at all times t , then we need also to limit p 0 's dependence on S , since p 0 (as well as p t for small t ) can be completely different from a stationary p ∞ , and just bounding D ( p ∞ ∥ ν ) does not say anything about it. Bounding D ( p 0 ∥ µ ) , for some data-independent distribution µ , ensures generalization at p 0 . This leaves the following questions regarding the proof of Theorem 2.7:

̸

- Why do we need to bound the divergences D ( p ∞ ∥ ν ) and D ( p 0 ∥ ν ) from the same distribution ν ? That is, we do we need to require µ = ν ? Bounding the divergences of p 0 and p ∞ to two different divergences µ = ν is sufficient to get generalization at the beginning (i.e. initialization) and end (i.e. after mixing)-is it sufficient for generalization in the middle (i.e. at any t )?
- Why do we need to also bound the reverse divergence D ( ν ∥ p ∞ ) ? I.e., why do we need to require p ∞ is Gibbs w.r.t. ν with a bounded potential, instead of just controlling the divergence D ( p ∞ ∥ ν ) , which is a weaker requirement and sufficient for generalization after mixing?

As we now show, both are necesairy, and without requiring both, i.e. if we drop either one of these, we cannot ensure generalization at intermediate times t ≥ 0 .

̸

Construction. Consider a supervised learning problem with Z = X × Y , X = [0 , 1] , Y = { 0 , 1 } , H = all measurable functions from X to Y , and the zero-one loss f ( h, ( x, y )) = I { h ( x ) = y } , with D being the uniform distribution over X , and y being Bernoulli( 1 2 ) independent of x . For all h , E D ( h ) = 0 . 5 . Let p 0 be the constant zero function with probability 1 2 and the constant one function with probability 1 2 . Consider the following deterministic S -dependent transition function over h : if h t is the constant zero function, then h t +1 = h S which memorizes S , i.e. h S ( x ) = y for ( x, y ) ∈ S , and h S ( x ) = 1 otherwise. If h t is not the constant zero function, then h t +1 is the constant ones function. We have that p ∞ is deterministic at the constant one function, and KL( p ∞ ∥ p 0 ) = ln 2 , and in fact p t = p ∞ for t &gt; 1 . But with probability half, h 1 = h S , for which for any sample size N &gt; 0 , E S ( h S ) = 0 while E D ( h S ) = 1 2 .

How does this show it is not enough to bound D ( p 0 ∥ ν ) and D ( p ∞ ∥ ν ) , but that we also need the reverse D ( ν ∥ p ∞ ) ? Since p 0 is data independent, we can take ν = p 0 , in which case KL( p 0 ∥ ν ) = D ∞ ( p 0 ∥ ν ) = 0 and KL( p ∞ ∥ ν ) = D ∞ ( p ∞ ∥ ν ) = ln 2 , but even as N →∞ , the gap for h 1 does not diminish. Indeed, D ( ν ∥ p ∞ ) = ∞ , and so p ∞ is not Gibbs w.r.t. ν and Theorem 2.7 does not apply.

̸

How does this show it is not enough to bound D ( p ∞ ∥ ν )+ D ( ν ∥ p ∞ ) and D ( p 0 ∥ µ ) for µ = ν ? Since in this example p ∞ is also data independent, we can take ν = p ∞ and µ = p 0 , in which case D ( p 0 ∥ µ ) = 0 and D ( p ∞ ∥ ν ) + D ( ν ∥ p ∞ ) = 0 . We are indeed ensured a small gap for h 0 and h ∞ , but not for h 1 .

## D Generalized Version of Corollary 3.1

̸

We start by characterizing the stationary distributions of SDERs in a box with different noise scales σ 2 . The stationary distributions for Gaussian initialization can be found similarly. Then, we extend Corollary 3.1 to scenarios where p 0 = ν , as an immediate consequence of Theorem 2.7.

## D.1 Stationary distributions of CLD

We first derive the stationary distribution of SDERs of the form

<!-- formula-not-decoded -->

with normal reflection in a box domain (for a full definition see (45)-(47) in Appendix H.2), where L ≥ 0 is some C 2 loss function, β &gt; 0 is an inverse temperature parameter, and σ 2 is a diffusion coefficient. First, we present a well known characterization of the stationary distribution of (25).

Lemma D.1. If L, σ 2 ∈ C 2 , σ 2 ( · ) &gt; 0 is uniformly bounded away from 0 in Ω ,

<!-- formula-not-decoded -->

the integrals exist, and the field ∇ L/σ 2 is conservative (curl-free), then

<!-- formula-not-decoded -->

is a stationary distribution of (25) .

For completeness, the proof is presented in Appendix H.2.1, following additional results and definitions in Appendix H. We can now calculate explicit stationary distributions for some choices of σ 2 . Specifically, we focus on cases where σ 2 ( x ) = g ( L ( x )) for some scalar function g , as it guarantees the curl-free condition, and is convenient to integrate.

Example D.2 (Uniform noise scale) . Assuming that σ 2 ( x ) ≡ 1 , the stationary distribution becomes the well-known Gibbs distribution

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Example D.3 (Linear noise scale) . Let α &gt; 0 , and suppose that σ 2 ( x ) = ( L ( x ) + α ) . Then

<!-- formula-not-decoded -->

so so the stationary distribution is

<!-- formula-not-decoded -->

which is integrable in a bounded domain. Recall that we want to represent p ∞ using a potential Ψ with inf Ψ ≥ 0 . In this case, we can start from ˜ Ψ( x ) = ( β +1)ln( L ( x ) + α ) . Since L ≥ 0 it clearly holds that ˜ Ψ ≥ ( β +1)ln( α ) , so we can use the shifted version

<!-- formula-not-decoded -->

Example D.4 (Polynomial noise scale) . Let α &gt; 0 , and k &gt; 1 . Suppose that σ 2 ( x ) = ( L ( x ) + α ) k . Then

<!-- formula-not-decoded -->

so

<!-- formula-not-decoded -->

As before, the potential is monotonically increasing with L ( x ) , so we can make a shift

<!-- formula-not-decoded -->

Example D.5 (Exponential noise scale) . Let α &gt; 0 and suppose that σ 2 ( x ) = e αL ( x ) . Then

<!-- formula-not-decoded -->

so

<!-- formula-not-decoded -->

Denote ψ ( τ ) = ατ -β α e -ατ , then ψ ′ ( τ ) = α + βe -ατ ≥ 0 . Therefore, min τ ≥ 0 ψ ( τ ) = ψ (0) = -β α , and we can take

<!-- formula-not-decoded -->

## D.2 Generalization bounds

Bounded domain with uniform initialization. Assume that training follows a CLD in a bounded domain as described in (25) with uniform initialization p 0 = Uniform(Θ 0 ) , where Θ 0 ⊆ Θ . For simplicity we take σ 2 ≡ 1 . In that case Theorem 2.7 implies the following.

Lemma D.6. Assume that the parameters evolve according to (25) with σ 2 ≡ 1 and uniform initialization p 0 = Uniform(Θ 0 ) , where Θ 0 ⊆ Θ . Then for any time t ≥ 0 , and δ ∈ (0 , 1) ,

1. w.p. 1 -δ over S ∼ D N ,

<!-- formula-not-decoded -->

2. w.p. 1 -δ over S ∼ D N and θ t ∼ p t

<!-- formula-not-decoded -->

Proof. This is a direct corollary of Theorem 2.7 with KL( p 0 ∥ ν ) = ln | Θ | / | Θ 0 | .

ℓ 2 regularization with Gaussian initialization. Let λ ∈ R d &gt; 0 be regularization terms, and consider the unconstrained SDE

<!-- formula-not-decoded -->

Notice that -β -1 diag ( λ ) θ t dt corresponds to an additive regularization of the form 1 2 β θ ⊤ t diag ( λ ) θ t , so each parameter can have a different regularization coefficient. We shall denote by ϕ λ a multivariate Gaussian distribution with mean 0 and covariance matrix diag ( λ -1 ) , where λ -1 = ( λ -1 1 , . . . , λ -1 d ) . For simplicity, we present the results with σ 2 ≡ 1 .

Lemma D.7. Let λ 0 , λ 1 &gt; 0 , and let θ t evolve according to (34) with σ 2 ≡ 1 and λ = λ 1 , and start from a Gaussian initialization p 0 = ϕ λ 0 . Then for any time t ≥ 0 , and δ ∈ (0 , 1) ,

1. w.p. 1 -δ over S ∼ D N ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. This is a direct corollary of Theorem 2.7 with the explicit expression for the KL divergence between two Gaussians.

Remark D.8 (Dependence on the parameters' dimension) . While the bound in Lemma D.7 depends on the dimension of the parameters d , this can be mitigated in practice. For example, by matching the regularization coefficient and initialization variance, the KL-divergence term vanishes and we lose the dependence on dimension. Furthermore, we can control each parameter separately by using parameter specific initialization variances and regularization coefficients. Then, the KL-divergence can have different dependencies, if any, on the dimension d .

9 For λ 0 = λ 0 I , λ 1 = λ 1 I , λ 0 , λ 1 &gt; 0 , this simplifies to KL( ϕ λ 0 ∥ ϕ λ 1 ) = d 2 ( ln λ 0 λ 1 -1 + λ 1 λ 0 )

## E Linear Regression with CLD

Theorem 2.7 and Corollary 3.1 only bound the gap between the population and training errors, yet this does not necessarily bound the population error itself. One way to do this is by separately bounding the training error and showing that in the regime in which the generalization gap is small, the training error can be small as well. In Appendix F we show empirically that deep NNs can reach low training error when trained with SGLD in the regime in which Corollary 3.1 is not vacuous. Here, we look at the particular case of the asymptotic behavior of ridge regression with CLD training with Gaussian i.i.d. data, for which we can analytically study the training and population losses .

Setup. Let θ ⋆ ∈ R d , y = x ⊤ θ ⋆ + ε with ∥ θ ⋆ ∥ = 1 and ε ∼ N ( 0 , σ 2 ) independent of x . We assume that x has i.i.d. entries with E x = 0 and covariance E [ xx ⊤ ] = I . Let X ∈ R N × d be the data (design) matrix, y ∈ R N the training targets, ε ∈ R N the pointwise perturbations, and θ ∈ R d the parameters in a linear regression problem. In what follows, we focus on the overdetermined case N &gt; d , where X has full column rank with probability 1, so the empirical covariance A = 1 N X ⊤ X ≻ 0 a.s. In addition, we denote θ LS = 1 N A -1 X ⊤ y , and ˜ θ = θ -θ LS . The training objective is then the minimization of the regularized empirical loss

<!-- formula-not-decoded -->

where C S = L S ( θ LS ) = 1 2 N ∥ y ∥ 2 -1 2 θ LS A θ LS = 1 2 N ∥ y ∥ 2 -1 2 N y ⊤ X ( X ⊤ X ) -1 X ⊤ y , is the empirical irreducible error.

CLD training. Assume that training is performed by CLD with inverse temperature β &gt; 0 , which, because L S is quadratic, takes the form

<!-- formula-not-decoded -->

Since A ≻ 0 and λ &gt; 0 , the Gibbs distribution

<!-- formula-not-decoded -->

is the unique stationary distribution, and furthermore, it is the asymptotic distribution of (37). We can simplify this to a Gaussian. Denote α = λ/β and

<!-- formula-not-decoded -->

then

<!-- formula-not-decoded -->

Since the last term is constant w.r.t. θ , we deduce that

<!-- formula-not-decoded -->

i.e. the stationary distribution is a Gaussian N ( ¯ θ , Σ ) . We can now calculate the expected training and population losses.

Goal. In the rest of this section, our final aim is to calculate the expected training and population losses in the setup described above, in the case when the data is sampled i.i.d. from standard Gaussian distribution, σ is a fixed constant, λ ∝ d (to match standard initialization), 10 N,β and d are large, but

10 Since this is a linear model d = layer width , and as we assume the regularization matches the standard initialization. This initialization is considered in many works as a Bayesian prior in various settings [35, 68].

β ≪ N , so our generalization bound is small (since E p 0 L is a fixed constant in this case). We will find (in Remark E.2 and Remark E.4) that if also d ≪ β then the training and expected population loss are not significantly degraded. This is not a major constraint, since we need d ≪ N to get good population loss anyway, even without noise (i.e. β → ∞ ). This shows that in this regime d ≪ β ≪ N , the randomness required by our generalization bound (the KL bounds in Corollary 3.1) does not significantly harm the training loss or the expected population loss.

Claim E.1. With some abuse of notation, denote L S ( θ ∞ ) = E θ ∼ p ∞ L S ( θ ) . Then

<!-- formula-not-decoded -->

Proof. From Petersen and Pedersen [56] (equation 318)

<!-- formula-not-decoded -->

For the second term, notice that

<!-- formula-not-decoded -->

A and Σ are simultaneously diagonalizable. To see this, let A = QΛQ ⊤ be a spectral decomposition of A , then A + α I = Q ( Λ + α I ) Q ⊤ , so Σ = β -1 Q ( Λ + α I ) -1 Q ⊤ . This means that A , Σ , and their inverses all multiplicatively commute. Therefore,

<!-- formula-not-decoded -->

Conditioned on X , standard results about the residuals in linear regression imply that,

<!-- formula-not-decoded -->

In addition, for any symmetric matrix M we have

<!-- formula-not-decoded -->

In particular,

<!-- formula-not-decoded -->

where we used the definition of A , the joint diagonalizability of A and Σ , and the cyclicality of the trace. In total, the expected training loss, conditioned on the data is

<!-- formula-not-decoded -->

Remark E.2 . We intuitively derive the asymptotic behavior of Claim E.1. Let λ be constant, and let β grow (so α shrinks). We can decompose ( A + α I ) -1 as

<!-- formula-not-decoded -->

This can be readily verified as

<!-- formula-not-decoded -->

where we used the multiplicative commutativity, as before. Notice that since A ≻ 0 , A + α I ≻ A , so ( A + α I ) -k ≺ A -k for any k ∈ N . Denote

<!-- formula-not-decoded -->

then ∥ R 2 ( α ) ∥ 2 ≤ α 2 λ min ( A ) 3 , where λ min ( A ) is the minimal eigenvalue of A . As the elements of X are i.i.d. with mean 0 and variance 1 , the limiting distribution of the spectrum of A as N,d → ∞ with d/N → γ ∈ (0 , 1) is the Marchenko-Pastur distribution, which is supported on [ ( 1 - √ γ ) 2 , ( 1 + √ γ ) 2 ] . In particular, as N,d →∞ , λ min ( A ) ≥ ( 1 -√ d/N ) 2 , so for ε &gt; 0 ,

<!-- formula-not-decoded -->

with high probability. Therefore, in the following we shall treat the remainder as R 2 ( α ) = O ( α 2 ) , even when taking the expectation over X .

Since α = λ/β and λ ∝ d , then for d ≤ β , we have α/β = O ( α 2 ) , and we conclude that

<!-- formula-not-decoded -->

Therefore, the added noise does not significantly hurt the training loss when 1 β ⪅ σ 2 ( 1 d -1 N ) , or equivalently, β ⪆ Nd ( N -d ) σ 2 . In particular, this holds when d ≪ β ≪ N , which is a regime where our generalization bound Corollary 3.1 also becomes small (since β ≪ N ). This shows that the randomness required by Corollary 3.1 can allow for successful optimization of the training loss.

Moving on to the population loss, we define L D in the usual way

<!-- formula-not-decoded -->

Due to the independence between x and ε ,

<!-- formula-not-decoded -->

Claim E.3. With some abuse of notation, denote L D ( θ ∞ ) = E θ ∼ p ∞ L D ( θ ) . Then

<!-- formula-not-decoded -->

Proof. Taking the expectation w.r.t θ ∼ N ( ¯ θ , Σ ) we get from Petersen and Pedersen [56]

<!-- formula-not-decoded -->

We can simplify some of the terms when taking the expectation conditioned on X .

<!-- formula-not-decoded -->

In addition,

<!-- formula-not-decoded -->

Combining these we get the desired result.

Remark E.4 . As we have done for the training loss in Remark E.2, we can estimate the expected population loss in some asymptotic regimes. Let λ be constant, and let β grow (so α shrinks). As in Remark E.2, we use the approximation ( A + α I ) -1 = A -1 -α A -2 + O ( α 2 I ) , which also implies ( A + α I ) -2 = A -2 -2 α A -3 + O ( α 2 I ) , and treat the remainders as O ( α 2 ) even when taking the expectation w.r.t. X . Then,

<!-- formula-not-decoded -->

Simplifying, we arrive at

<!-- formula-not-decoded -->

Assuming that x are i.i.d. N ( 0 , I ) , N · A ∼ W d ( N, I ) , i.e. has a Wishart distribution. According to Theorem 3.3.16 of [24], if N &gt; d +3 then

<!-- formula-not-decoded -->

Then, the expectation over X and if σ 2 N ⪅ α (which is true for λ ∝ d and β ≪ N like we assume here),

<!-- formula-not-decoded -->

This result is similar to the one in Remark E.2 - for the expected population loss not to be significantly hurt by the added noise, it must hold that β ⪆ Nd ( N -1) σ 2 . In particular, this holds when d ≪ β ≪ N , which is a regime where our generalization bound Corollary 3.1 also becomes small (since β ≪ N ). This shows that the randomness required by Corollary 3.1 does not harm the expected population loss.

## F Numerical Experiments

## F.1 Experimental results

The following are results of training with SGLD (a discretized version of the CLD in (10)) on a few benchmark datasets. Notice we use the regularized version where regularization coefficient is λ · β -1 and the λ hyperparameter is dictated by the initialization from the normal distribution p 0 = N ( 0 , λ -1 I d ) . We used a common initialization of N ( 0 , 1 d in ) , i.e. λ = d in .

We use several different values of β relative to N (the number of training samples). For simplicity, we focused on binary classification cases. In all datasets with more than 2 classes, we constructed a binary classification task by partitioning the original label set into 2 disjoint sets of the same size.

The results demonstrate that learning with SGLD is possible with various values of β . In fact, in several instances, the injected noise appears to improve the generalization gap , e.g, in SVHN [53], in all the tested β values between 0 . 4 · N and 2 · N the average test error remained almost the same while the training error decreased as β increased (i.e. the generalization gap increased). Notably, we also observe that for sufficiently large levels of noise, the generalization bounds are non-vacuous.

Table 2: MNIST (binary classification)

| β          | E S                   | E D                   |   E D - E S | Bound (11) w.p 0.99   |
|------------|-----------------------|-----------------------|-------------|-----------------------|
| 0 . 01 · N | 0 . 2279( ± 0 . 0021) | 0 . 1972( ± 0 . 0243) |     -0.0307 | 0.06124               |
| 0 . 03 · N | 0 . 1161( ± 0 . 0028) | 0 . 1074( ± 0 . 0035) |     -0.0087 | 0.10498               |
| 0 . 1 · N  | 0 . 0618( ± 0 . 001)  | 0 . 062( ± 0 . 0041)  |      0.0002 | 0.19096               |
| 0 . 15 · N | 0 . 0497( ± 0 . 0014) | 0 . 0494( ± 0 . 0031) |     -0.0003 | 0.23376               |
| 0 . 4 · N  | 0 . 0281( ± 0 . 0002) | 0 . 0358( ± 0 . 0029) |      0.0077 | 0.38147               |
| 0 . 7 · N  | 0 . 0202( ± 0 . 0006) | 0 . 0284( ± 0 . 0024) |      0.0082 | 0.50456               |
| N          | 0 . 0162( ± 0 . 0006) | 0 . 0278( ± 0 . 0023) |      0.0116 | 0.60302               |
| 2 · N      | 0 . 0092( ± 0 . 0004) | 0 . 0262( ± 0 . 0016) |      0.017  | 0.85273               |
| ∞          | 0 . 0001( ± 0)        | 0 . 0229( ± 0 . 0004) |      0.0228 | > 1                   |

Table 3: fashionMNIST (binary classification)

| β          | E S                   | E D                   |   E D - E S | Bound (11) w.p 0.99   |
|------------|-----------------------|-----------------------|-------------|-----------------------|
| 0 . 01 · N | 0 . 1215( ± 0 . 0027) | 0 . 1251( ± 0 . 0087) |      0.0036 | 0.06833               |
| 0 . 03 · N | 0 . 0999( ± 0 . 001)  | 0 . 1087( ± 0 . 0167) |      0.0088 | 0.11738               |
| 0 . 1 · N  | 0 . 0821( ± 0 . 0012) | 0 . 086( ± 0 . 001)   |      0.0039 | 0.21368               |
| 0 . 15 · N | 0 . 0765( ± 0 . 0009) | 0 . 0803( ± 0 . 0015) |      0.0038 | 0.26159               |
| 0 . 4 · N  | 0 . 0635( ± 0 . 0005) | 0 . 0722( ± 0 . 002)  |      0.0087 | 0.42695               |
| 0 . 7 · N  | 0 . 0567( ± 0 . 0006) | 0 . 0691( ± 0 . 0019) |      0.0124 | 0.56473               |
| N          | 0 . 0525( ± 0 . 0005) | 0 . 0675( ± 0 . 0013) |      0.015  | 0.67495               |
| 2 · N      | 0 . 043( ± 0 . 0007)  | 0 . 0672( ± 0 . 0023) |      0.0242 | 0.95446               |
| ∞          | 0 . 0248( ± 0 . 001)  | 0 . 0675( ± 0 . 0033) |      0.0427 | > 1                   |

Table 4: SVHN (binary classification)

| β          | E S                   | E D                   |   E D - E S |   Bound (11) w.p 0.99 |
|------------|-----------------------|-----------------------|-------------|-----------------------|
| 0 . 01 · N | 0 . 0746( ± 0 . 0012) | 0 . 1033( ± 0 . 0032) |      0.0287 |               0.05898 |
| 0 . 03 · N | 0 . 0441( ± 0 . 0004) | 0 . 067( ± 0 . 0026)  |      0.0229 |               0.10203 |
| 0 . 1 · N  | 0 . 0282( ± 0 . 0008) | 0 . 0476( ± 0 . 007)  |      0.0194 |               0.1862  |
| 0 . 15 · N | 0 . 0251( ± 0 . 0005) | 0 . 0445( ± 0 . 002)  |      0.0194 |               0.22803 |
| 0 . 4 · N  | 0 . 0182( ± 0 . 0005) | 0 . 0374( ± 0 . 0017) |      0.0192 |               0.37235 |
| 0 . 7 · N  | 0 . 0146( ± 0 . 0004) | 0 . 0363( ± 0 . 002)  |      0.0217 |               0.49256 |
| N          | 0 . 0124( ± 0 . 0002) | 0 . 0342( ± 0 . 0014) |      0.0218 |               0.58872 |
| 2 · N      | 0 . 0085( ± 0)        | 0 . 0371( ± 0 . 001)  |      0.0286 |               0.83256 |

Figure 1: Parity Results. Left: Training error. Right: test error and generalization bound.

<!-- image -->

## F.2 Training details

MNIST and fashionMNIST. We trained a fully connected network with 4 hidden layers of sizes [256 , 256 , 256 , 128] and ReLU activation, lr = 0 . 01 , for 60 epochs.

SVHN. The network was trained with a convolutional neural network with 5 convolutional layers, lr = 0 . 01 , for 80 epochs. The complete architecture:

- Two convolutional layers (3×3 kernel, padding 1) with 32 channels, followed by ReLU activations and a 2×2 max pooling.
- Two convolutional layers (3×3 kernel, padding 1) with 64 channels, followed by ReLU activations and a 2×2 max pooling.
- A 3×3 convolution with 128 channels, ReLU, and 2×2 max pooling.
- 2 A linear layer 2048 → 512 , followed by ReLU and another 512 → 1 linear layer

Parity. In this experiment, we consider a synthetic binary classification task where each input is a binary vector of length 70 and the target label is defined as the parity of 3 randomly selected input dimensions. We train a neural network using SGLD with varying values of the inverse temperature parameter β and different sample sizes.

The network was trained with a fully connected network with 4 hidden layers of sizes [512 , 1028 , 2064 , 512] and ReLU activation, lr = 0 . 05 , for 100 epochs.

The results show that injecting noise can improve the generalization gap: specifically, the case of β ≥ N 2 leads to overfitting, while smaller values of β (e.g., 1 . 5 · N to 12 · N ) yield better generalization. Moreover, as well as in the benchmark datasets, in this setting, our generalization bound is non-vacuous in several cases.

## F.3 Comparison with the bound of Mou et al. [49]

The bound proposed by Mou et al. [49] has demonstrated non-vacuous results. To further assess the effectiveness of our bound and evaluate its relative tightness, we conducted a series of numerical experiments on the MNIST binary classification task (see Tables 5-8).

It is worth emphasizing that our bound offers a distinct advantage: it can be evaluated directly at initialization, whereas the bound of Mou et al. [49] depends on gradients and therefore cannot be computed before training. When testing their bound we used the continuous version, i.e.

<!-- formula-not-decoded -->

For simplicity, we omitted the term involving M (which makes the bound more favorable). In addition, we set s = 0 . 5 since the zero-one loss (denoted here by f ( w ) , unlike [49]) is bounded within the interval [0 , 1] . We observed that the relative tightness of the two bounds varies across different values of β and at different points in time. Consequently, in some instances, the bound of Mou et al. [49] is tighter, while in others our bound performs better, and we could not draw any further conclusions.

Table 5: 20 training epochs

| β               | Train Error   | Test Error   | Generalization Gap   | Mou et al. [49]   | Our bound   |
|-----------------|---------------|--------------|----------------------|-------------------|-------------|
| 0 . 03 N = 1800 | 0 . 1224      | 0 . 137      | 0 . 0146             | 0 . 0539          | 0 . 1144    |
| 0 . 15 N = 9000 | 0 . 0515      | 0 . 0747     | 0 . 0232             | 0 . 1279          | 0 . 2548    |
| 0 . 4 N = 24000 | 0 . 0335      | 0 . 058      | 0 . 0245             | 0 . 2845          | 0 . 4157    |
| 0 . 7 N = 42000 | 0 . 0278      | 0 . 0498     | 0 . 0220             | 0 . 4930          | 0 . 5499    |
| N = 60000       | 0 . 0249      | 0 . 0428     | 0 . 0179             | 0 . 7032          | 0 . 6572    |
| 2 N = 120000    | 0 . 0209      | 0 . 0356     | 0 . 0147             | 1 . 4044          | 0 . 9294    |

Table 6: 50 training epochs

| β               | Train Error   | Test Error   | Generalization Gap   | Mou et al. [49]   | Our bound   |
|-----------------|---------------|--------------|----------------------|-------------------|-------------|
| 0 . 03 N = 1800 | 0 . 1156      | 0 . 1697     | 0 . 0541             | 0 . 0637          | 0 . 1144    |
| 0 . 15 N = 9000 | 0 . 0491      | 0 . 0615     | 0 . 0124             | 0 . 1324          | 0 . 2548    |
| 0 . 4 N = 24000 | 0 . 0295      | 0 . 0348     | 0 . 0053             | 0 . 2992          | 0 . 4157    |
| 0 . 7 N = 42000 | 0 . 0217      | 0 . 0283     | 0 . 0066             | 0 . 4903          | 0 . 5499    |
| N = 60000       | 0 . 0173      | 0 . 0277     | 0 . 0104             | 0 . 6827          | 0 . 6572    |
| 2 N = 120000    | 0 . 0108      | 0 . 0265     | 0 . 0157             | 1 . 3153          | 0 . 9294    |

Table 7: 250 training epochs

| β               | Train Error   | Test Error   | Generalization Gap   | Mou et al. [49]   | Our bound   |
|-----------------|---------------|--------------|----------------------|-------------------|-------------|
| 0 . 03 N = 1800 | 0 . 122       | 0 . 1049     | - 0 . 0171           | 0 . 1273          | 0 . 1144    |
| 0 . 15 N = 9000 | 0 . 0502      | 0 . 0476     | - 0 . 0026           | 0 . 1503          | 0 . 2548    |
| 0 . 4 N = 24000 | 0 . 0284      | 0 . 0296     | 0 . 0011             | 0 . 2853          | 0 . 4157    |
| 0 . 7 N = 42000 | 0 . 0178      | 0 . 0247     | 0 . 0069             | 0 . 4595          | 0 . 5499    |
| N = 60000       | 0 . 0127      | 0 . 0240     | 0 . 0113             | 0 . 6478          | 0 . 6572    |
| 2 N = 120000    | 0 . 0050      | 0 . 0234     | 0 . 0184             | 1 . 2158          | 0 . 9294    |

Table 8: 400 training epochs

| β               | Train Error   | Test Error   | Generalization Gap   | Mou et al. [49]   | Our bound   |
|-----------------|---------------|--------------|----------------------|-------------------|-------------|
| 0 . 03 N = 1800 | 0 . 1224      | 0 . 1105     | - 0 . 0119           | 0 . 1900          | 0 . 1144    |
| 0 . 15 N = 9000 | 0 . 0499      | 0 . 0556     | 0 . 0057             | 0 . 1774          | 0 . 2548    |
| 0 . 4 N = 24000 | 0 . 0261      | 0 . 0357     | 0 . 0096             | 0 . 3005          | 0 . 4157    |
| 0 . 7 N = 42000 | 0 . 0161      | 0 . 0271     | 0 . 0110             | 0 . 4548          | 0 . 5499    |
| N = 60000       | 0 . 0112      | 0 . 0255     | 0 . 0143             | 0 . 6247          | 0 . 6572    |
| 2 N = 120000    | 0 . 0038      | 0 . 0249     | 0 . 0211             | 1 . 1455          | 0 . 9294    |

## G Mild Overparametrization Prevents Uniform Convergence

In this section, we consider fully-connected ReLU networks, where the weights are bounded, such that for each layer j the absolute values of all weights are bounded by 1 √ d j -1 , where d j -1 is the width of layer j -1 . Moreover, we assume that the input x is such that each coordinate x i is bounded in [ -1 , 1] . We show that m training examples do not suffice for learning constant depth networks with O ( m ) parameters. Thus, even a mild overparameterization prevents uniform convergence in our setting.

Our result follows by bounding the fat-shattering dimension, defined as follows:

Definition G.1. Let F be a class of real-valued functions from an input domain X . We say that F shatters m points { x i } i m =1 ⊆ X with margin ϵ &gt; 0 if there are r 1 , . . . , r m ∈ R such that for all y 1 , . . . , y m ∈ { 0 , 1 } there exists f ∈ F such that

<!-- formula-not-decoded -->

The fat-shattering dimension of F with margin ϵ is the maximum cardinality m of a set of points in X for which the above holds.

The fat-shattering dimension of F with margin ϵ lower bounds the number of samples needed to learn F within accuracy ϵ in the distribution-free setting (see, e.g., [2, Part III]). Hence, to lower bound the sample complexity by some m it suffices to show that we can shatter a set of m points with a constant margin.

Theorem G.2. We can shatter m points { x i } i m =1 where ∥ x i ∥ ∞ ≤ 1 , with margin 1 , using ReLU networks of constant depth and O ( m ) parameters, such that for each layer j the absolute values of all weights are bounded by 1 √ d j -1 , where d j -1 is the width of layer j -1 .

Proof. Consider input dimension d 0 = 1 . For 1 ≤ i ≤ m , consider the points x i = i m , and let { y i } i m =1 ⊆ { 0 , 1 } . Consider the following one-hidden-layer ReLU network N , which satisfies N ( x i ) = y i m for all i . First, the network N includes a neuron with weight 0 and bias y 1 m , i.e., [0 · x + y 1 m ] + . Now, for each i such that y i = 0 and y i +1 = 1 we add two neurons: [ x -y i ] + -[ x -y i +1 ] + , and for i such that y i = 1 and y i +1 = 0 we add -[ x -y i ] + + [ x -y i +1 ] + . It is easy to verify that this construction has width at most 2 m -1 and allows us to shatter m points with margin 1 2 m . However, the output weights of the neurons are ± 1 , and thus it does not satisfy the theorem's requirement. Consider the network N ′ ( x ) = N ( x ) · 1 √ 2 m -1 obtained from N by modifying the output weights. The network N ′ satisfies the theorem's requirement on the weight magnitudes, and allows for shattering with margin 1 2 m √ 2 m -1 . We will now show how to increase this margin to 1 using a constant number of additional layers.

Let ˜ N be a network obtained from N ′ as follows. First, we add a ReLU activation to the output neuron of N ′ . Since for every x i we have N ′ ( x i ) ≥ 0 , it does not affect these outputs. Next, we add L = 8 additional layers (layers 3 , . . . , 3 + L -1 ) of width √ m and without bias terms, where the incoming weights to layer 3 are all 1 and the weights in layers 4 , . . . , 3 + L -1 are 1 m 1 / 4 . Finally, we add an output neuron (layer 3 + L ) with incoming weights 1 m 1 / 4 . The network ˜ N satisfies the theorem's requirements on the weight magnitudes, and it has depth 3 + L = 11 and O ( m ) parameters. Now, suppose that all neurons in a layer 3 ≤ j ≤ 3 + L -1 have values (i.e., activations) z ≥ 0 , then the values of all neurons in layer j + 1 are z · 1 m 1 / 4 · √ m = z · m 1 / 4 .

Hence, if the value of the neuron in layer 2 is 1 2 m √ 2 m -1 , then the output of the network ˜ N is 1 2 m √ 2 m -1 · ( m 1 / 4 ) L = m L/ 4 2 m √ 2 m -1 = m 2 2 m √ 2 m -1 ≥ 2 for large enough m . If the value of the neuron in layer 2 is 0 then the output of ˜ N is also 0 . Hence, this construction allow for shattering m points with margin at least 1 , using O ( m ) parameters and weights that satisfy the theorem's conditions.

## H Background on Stochastic Differential Equations with Reflection

We supply an introduction to the theory of stochastic differential equations with reflection (SDERs), then proceed to characterize the stationary distribution of a family of SDERs in a box. The background of standard (non-reflective) SDEs is similar and more common, and is therefore not included here. See for example [54] for more.

## H.1 SDEs with reflection

One of the main analytical tools of this work is the characterization of stationary distributions of SDER in bounded domains (see 57, 63, for an introduction).

The purpose of this section is to present more rigorously the setting of the paper, and supply the relevant definitions and results required to arrive at Lemma D.1. As Lemma D.1 is considered a well-known result, this section is mainly intended for completeness. Specifically, in the following we present some relevant definitions and results by Kang and Ramanan [31, 32], and specifically, ones that relate solutions to SDERs (Definition 2.4 in [32]), to solutions to sub-martingale problems (Definition 2.9 in [32]), and that characterize the stationary distributions of such solutions. For simplicity, we sometimes do not state the results in full generality.

Setting. Let Ω ⊂ R d be a domain (non-empty, connected, and open). Let the drift term b : R d → R d and dispersion coefficient Σ : R d → R d × d be measurable and locally bounded. We also denote the diffusion coefficient by A ( · ) = Σ ( · ) Σ ( · ) ⊤ = ( a ij ( · )) d i,j =1 , and denote its columns by a i ( · ) . We say that the diffusion coefficient is uniformly elliptic if there exists σ &gt; 0 such that

<!-- formula-not-decoded -->

Let η be a set valued mapping of allowed reflection directions defined on Ω such that η ( x ) = { 0 } for x ∈ Ω , and η ( x ) is a non-empty, closed and convex cone in R d such that { 0 } ⊆ η ( x ) for x ∈ ∂ Ω , and furthermore assume that the set { ( x , v ) : x ∈ Ω , v ∈ η ( x ) } is closed in R 2 d . In addition, for x ∈ ∂ Ω let ˆ n ( x ) be the set of inwards normals to Ω at x ,

<!-- formula-not-decoded -->

Then, denote the set of boundary points with inward pointing cones

<!-- formula-not-decoded -->

and let V ≜ ∂ Ω \ U . For example, if Ω is a convex polyhedron and η ( x ) is the cone defined by the positive span of ˆ n ( x ) we get that V = ∅ .

Throughout this section and the rest of the paper, the stochastic differential equation with reflection (SDER) in (Ω , η )

<!-- formula-not-decoded -->

where w t is a Wiener process, and r t is a reflection process with respect to some filtration, is understood as in Definition 2.4 of [32], and the submartingale problem associated with (Ω , η ) , V , b and Σ , refers to Definition 2.9 of [32]. In addition, we use the following definition.

Definition H.1 (Piecewise C 2 with continuous reflection; Definition 2.11 in [32]) . The pair (Ω , η ) is said to be piecewise C 2 with continuous reflection if it satisfies the following properties:

1. Ω is a non-empty domain in R d with representation

<!-- formula-not-decoded -->

where I is a finite set and for each i ∈ I , Ω i is a non-empty domain with C 2 boundary in the sense that for each x ∈ ∂ Ω , there exist a neighborhood N ( x ) of x , and functions φ i x ∈ C 2 ( R d ) , i ∈ I ( x ) = { i ∈ I | x ∈ ∂ Ω i } , such that

<!-- formula-not-decoded -->

̸

and ∇ φ i x = 0 on N ( x ) . For each x ∈ ∂ Ω i and i ∈ I ( x ) , let

<!-- formula-not-decoded -->

denote the unit inward normal vector to ∂ Ω i at x .

2. The (set-valued) direction 'vector field' η : Ω → R d is given by

<!-- formula-not-decoded -->

where for each i ∈ I , η i ( · ) is a continuous unit vector field defined on ∂ Ω i that satisfies for all x ∈ ∂ Ω i

<!-- formula-not-decoded -->

If η i ( · ) is constant for every i ∈ I , the the pair (Ω , η ) is said to be piecewise C 2 with constant reflection. If, in addition, n i ( · ) is constant for every i ∈ I , then the pair (Ω , η ) is said to be polyhedral with piecewise constant reflection.

In addition, let S denote the smooth parts of ∂ Ω .

Remark H.2 . It is clear from the definition that if Ω is polyhedral, i.e. if all Ω i 's are half-spaces, and η consists of inward normal reflections, then (Ω , η ) is polyhedral with piecewise constant reflection. Theorem H.3 (Theorem 3 in [31], simplified) . Suppose that the pair (Ω , η ) is piecewise C 2 with continuous reflection, for all i ∈ I and x ∈ ∂ Ω i , 〈 n i ( x ) , η i ( x ) 〉 = 1 , V = ∅ , b ( · ) ∈ C 1 ( Ω ) and A ∈ C 2 ( Ω ) (elementwise), and the submartingale problem associated with (Ω , η ) and V is well posed. Furthermore, suppose there exists a nonnegative function p ∈ C 2 ( Ω ) with Z p = ∫ Ω p ( x ) d x &lt; ∞ that solves the PDE defined by the following three relations:

1. For x ∈ Ω :

<!-- formula-not-decoded -->

2. For each i ∈ I and x ∈ ∂ Ω ∩ S ,

<!-- formula-not-decoded -->

where and

<!-- formula-not-decoded -->

̸

3. For each i, j ∈ I , i = j , and x ∈ ∂ Ω i ∩ ∂ Ω j ∩ ∂ Ω ,

<!-- formula-not-decoded -->

Then the probability measure on Ω defined by

<!-- formula-not-decoded -->

is a stationary distribution for the well-posed submartingale problem.

<!-- formula-not-decoded -->

Weare now ready to state a characterization of stationary distributions of (39). Note that for simplicity, we do not maintain full generality.

Corollary H.4 (Stationary distribution of weak solutions to SDERs) . Suppose that, Ω is convex and bounded, b ∈ C 1 ( Ω ) and A ∈ C 2 ( Ω ) , (Ω , η ) is piecewise C 2 with continuous reflection, A is uniformly elliptic (see (38) ), and V = ∅ . Then p ∈ C 2 satisfying the conditions in Theorem H.3 defines a stationary distribution for (39) .

Proof. Assumptions compactness of the domain, and continuous differentiability of the drift and dispersion coefficient imply that they are Lipschitz, hence Exercise 2.5.1 and Theorem 2.5.4 of [57] imply that there exists a unique strong solution to the SDER (39). Then, piecewise C 2 with continuous reflection, the uniform ellipticity assumption, Theorems 1 and 3 of [32], and Theorem H.3 imply that if there exists p ∈ C 2 satisfying (41)-(43), then (44) is a stationary distributions of (39).

In the next subsection we use this to derive explicit expressions for the stationary distribution in the setting of this paper.

## H.2 SDER with isotropic diffusion in a box

We proceed to assume that the diffusion term is a scalar matrix of the form A ( x ) = 2 σ 2 ( x ) I d , and that Ω is a bounded box in R d , i.e. there exist { m i &lt; M i } d i =1 such that

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and that the reflecting field is normal to the boundary, i.e. given by (40) with

<!-- formula-not-decoded -->

for i = 1 , . . . , d . In this setting, we can considerably simplify the conditions in Theorem H.3, as done in the following corollary.

Lemma H.5 (Stationarity condition for SDER in a box with normal reflection) . Let b ( · ) ∈ C 1 , and let σ ( · ) ∈ C 2 be uniformly bounded away from 0, i.e. there exists σ 2 &gt; 0 such that for all x ∈ Ω , σ 2 ( x ) &gt; σ 2 . If there exists p ∈ C 2 such that

<!-- formula-not-decoded -->

and ∫ Ω p ( x ) d x = 1 , then p is a stationary distribution of

<!-- formula-not-decoded -->

in Ω .

Remark H.6 . (48) is exactly the stationarity condition derived from the Fokker-Planck equation with Neumann boundary conditions ensuring conservation of mass.

Proof. Under the assumptions we see that the conditions of Corollary H.4 are satisfied, and we can use (41)-(43) to find stationary distributions of (49). First, notice that (41) simplifies to

<!-- formula-not-decoded -->

Next, we can considerably simplify the boundary conditions. First, notice that S consists of the interior of the domain's faces so for x ∈ ∂ Ω ∩S , the set of active boundary regions I ( x ) is a singleton I ( x ) = { ( i, s ) } , for some i = 1 , . . . , d and s ∈ { m,M } . We focus on the lower boundaries ( m ), as the conditions for the upper boundaries are symmetric.

For i = 1 , . . . , d and x ∈ ∂ Ω ∩ S , η i m ( x ) = n i m ( x ) = e i so

<!-- formula-not-decoded -->

so (43) is satisfied. In addition,

<!-- formula-not-decoded -->

so (42) becomes, for all i = 1 , . . . , d ,

<!-- formula-not-decoded -->

which is

<!-- formula-not-decoded -->

## H.2.1 Reflected Langevin dynamics in a box

In this section, we derive some useful properties of the SDER

<!-- formula-not-decoded -->

in a box domain as defined in (45)-(47), where L ≥ 0 is some (loss/potential) function, and β &gt; 0 is an inverse temperature parameter. First, we characterize the stationary distribution of this process.

Recall Lemma D.1. If L, σ 2 ∈ C 2 , σ 2 ( · ) &gt; 0 is uniformly bounded away from 0 in Ω ,

<!-- formula-not-decoded -->

the integrals exist, and the field ∇ L/σ 2 is conservative (curl-free), then

<!-- formula-not-decoded -->

is a stationary distribution of (50).

Proof. The drift term in this setting is b = -β ∇ L . Therefore, from Lemma H.5, we get that any distribution that satisfies

<!-- formula-not-decoded -->

on Ω , is a stationary distribution. We can solve this PDE as

<!-- formula-not-decoded -->

Then,

<!-- formula-not-decoded -->

where we used the assumption that the integral on the RHS exists, and is well defined. Hence

<!-- formula-not-decoded -->

When the integral in (51) is solvable, we can find an explicit expression for the stationary distribution, as was done in Appendix D.1.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->