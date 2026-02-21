## Variational Transdimensional Inference

Laurence Davies 1 ∗ Dan Mackinlay 2ˆ Rafael Oliveira 2ˆ Scott A. Sisson 1$ 1 University of New South Wales 2 CSIRO Data61

∗ laurence@latentlogic.com.au, $ scott.sisson@unsw.edu.au ˆ {dan.mackinlay,rafael.dossantosdeoliveira}@data61.csiro.au

## Abstract

The expressiveness of flow-based models combined with stochastic variational inference (SVI) has expanded the application of optimization-based Bayesian inference to highly complex problems. However, despite the importance of multimodel Bayesian inference for problems defined on a transdimensional joint model and parameter space, such as Bayesian structure learning, flow-based SVI has been limited to problems defined on a fixed-dimensional parameter space. We introduce CoSMIC normalizing flows (COntextually-Specified Masking for Identity-mapped Components), an extension to neural autoregressive conditional normalizing flow architectures that enables use of a single flow-based variational density for inference over a transdimensional (multi-model) conditional target distribution. We propose a combined stochastic variational transdimensional inference (VTI) approach to training CoSMIC flows using ideas from Bayesian optimization and Monte Carlo gradient estimation. Numerical experiments show the performance of VTI on challenging problems that scale to high-cardinality model spaces.

## 1 Introduction

Variational inference via stochastic optimization [23, 37] has surged in interest since the introduction of normalizing flows [43]. Flow-based densities can be used for a variety of downstream tasks, such as importance sampling [43], simulation-based inference [39, 56], adaptive Markov chain Monte Carlo (MCMC) [17], and generative modeling [27]. While many existing approaches only consider continuous supports, there is a growing interest in applications where the support is either discrete or discretely indexed [11]. One such application concerns a target transdimensional probability distribution π with support X = ⋃ m ∈ M ( { m } × Θ m ) , where M is a finite discrete index set, Θ m ⊆ R d m , and the dimension d m of Θ m may vary with m . Hence X is a transdimensional space [16, 21, 48]. Such spaces arise in Bayesian model inference, where Θ m correspond to model parameters , and m ∈ M is a model index . Discrete indices parameterize many practical inference problems, including variable selection [16], mixtures-of-regressions, learning directed acyclic graphs (DAGs) from data [52], phylogenetic tree topology search [15], mixture-component inference [8], geoscientific inversion [45], and change-point models [21]. This article is concerned with estimating the target distribution π with associated density function π ( m, θ m ) , θ m ∈ Θ m , whose dimension depends on m . For simplicity we refer to π ( m, θ m ) and related functions as density functions, even though they are not continuous. Typically, this density is only available in a conditional unnormalized form, η ( θ m | m ) = Z m π ( θ m | m ) , where Z m = ∫ Θ m η ( θ m | m ) d θ m . The factorization η ( m, θ m ) = η ( θ m | m ) π ( m ) implies there is a discrete target probability mass function over models, π ( m ) = Z m Z -1 , where Z = ∑ m ∈ M Z m . Estimation of η ( m, θ m ) then becomes estimation of both η ( θ m | m ) and π ( m ) .

In the presence of a likelihood function p ( D| m, θ m ) for data D , and priors p ( θ m | m ) and p ( m ) , the target distribution is defined by the D -conditional transdimensional posterior π ( m, θ m | D ) ∝ p ( D| m, θ m ) p ( θ m | m ) p ( m ) . In the context of variational Bayesian inference (VI; see [2, 25]) approximation of the transdimensional posterior π ( m, θ m | D ) has not been addressed in generality. Such a scheme would approximate some unnormalized target density η ( m, θ m | D ) = Z π ( m, θ m | D ) by choosing parameters ϕ ∈ R n ϕ , ψ ∈ R n ψ of a tractable variational density family q ψ,ϕ ( m, θ m ) = q ϕ ( θ m | m ) q ψ ( m ) to minimize

<!-- formula-not-decoded -->

where D KL is the Kullback-Leibler (KL) divergence. There are two impediments to constructing such a variational approximation: (i) defining and optimizing q ϕ as θ m may vary in dimension conditional on m , and (ii) the inference of q ψ for discrete latent variables m during the optimization of q ϕ , a non-stationary objective as ϕ → ϕ ∗ and ψ → ψ ∗ are interdependent.

Background: Flow-based models for stochastic variational inference: Rezende and Mohamed [43] showed that a normalizing flow for q ϕ (with fixed m ) is able to approximate many challenging fixed-dimensional distributions that are not well approximated by common parametric families. A normalizing flow is defined by a diffeomorphism T ϕ : R d → R d between two random vectors θ ∼ q and z ∼ ν d , such that their distributions q and ν d are absolutely continuous with respect to a d -dimensional Lebesgue measure, have well-defined densities q ( θ ) and ν d ( z ) respectively, and can be related by z = T ϕ ( θ ) so that q ( θ ; ϕ ) = ν d ( T ϕ ( θ )) | det ∇ T ϕ ( θ ) | , θ ∈ R d . As is typical of normalizing flow-based models, we refer to ν d as the reference distribution and assume it factorizes into a product of d identical marginal distributions ν d = ν ⊗ · · · ⊗ ν = ⊗ d ν . Construction of T ϕ is typically achieved by defining d bijective, univariate functions τ ρ i : R ↦→ R , z i = τ ρ i ( θ i ) for i ∈ { 1 , . . . , d } . The parameters ρ i = NN ϕ ( θ \ i ) for the i th transformation are determined by a neural network NN ϕ such that ρ i is not dependent on θ i , so that the inverse τ -1 ρ i ( · ) can be calculated without requiring inversion of NN ϕ . This independency remains if the neural network NN ϕ is autoregressive with respect to the inputs θ 1 , . . . , θ d [38]. Benefits of autoregressive flows are their higher-overall expressiveness and efficiency in the variational inference setting versus e.g. coupling flows [7]. For these reasons, this paper employs autoregressive NN ϕ . A conditional normalizing flow extends is a natural extension of a normalizing flow with a conditioning variate, ξ , passed as a contextual input to the NN ϕ , such that ρ i = NN ϕ ( θ \ i ; ξ ) . Applications include classification, where ξ is an index, or likelihood estimation [55] where ξ encodes the parameters of the likelihood function.

The MADE encoder [19] enables autoregressive neural flow architectures, which can be coupled with any τ such as affine [38] and spline [13] transformations. The cost of an autoregressive flow depends on the direction. In the forward (sampling) direction, it evaluates each dimension sequentially, for a time complexity of O ( d ) . In the the inverse (likelihood) direction, computation can be parallel. The inverse autoregressive flow (IAF) [28] reverses this dependence, setting θ = T ϕ ( z ) , yielding the variational density q ϕ ( θ m ) = ν d ( T -1 ϕ ( θ )) | det ∇ T -1 ϕ ( θ ) | = ν d ( z ) | det ∇ T ϕ ( z ) | -1 .

Contributions: We introduce CoSMIC ( COntextually-Specified Masking for Identity-mapped Components ) flows, a widely applicable and simple modification to conditional neural flow architectures (Section 2). CoSMIC flows fundamentally expand the use cases for normalizing flows to encompass variational inference applications, so that a single flow-based variational density can be used for variational inference over a transdimensional (multi-model) target distribution. In effect, this extends the reparameterization trick exploited by IAF-based VI to the transdimensional setting. In Section 3, we demonstrate the efficacy of CoSMIC transformations within a novel variational transdimensional inference (VTI) framework with two implementations. The first builds upon principles of Bayesian optimization [50], and the second uses Monte Carlo gradient estimation [34]. We also provide a theoretical analysis of VTI approximation error bounds under a Gaussian process surrogate, and convergence guarantees for the marginal model distribution under convergent optimization steps. Finally, we demonstrate the applicability of VTI to problems with model spaces that cannot be easily enumerated within the memory limitations of current computing architectures. In particular, Section 5 explores problems in Bayesian robust variable selection [35] and Bayesian causal discovery [22]. 1

1 PyTorch CUDA code for all experiments is available at https://github.com/daviesl/avti.

## 2 Formulating a transdimensional variational density

Rather than constructing a variational density separately for each model m ∈ M , it is preferable to construct a single density on the transdimensional support X . To account for the varying dimension of θ m , we adopt the dimension saturation approach of Brooks et al. [5], where the dimension of the parameter space conditional on each model is unified across all models. This is achieved by augmenting the space of model-conditional parameters with auxiliary variables u ∼ ν , as discussed below. We use \ m to identify auxiliary variables of dimension d max -d m , where d max := max m { d m } . We define the saturated support ( θ m , u \ m ) ∈ Θ m ×U m ⊆ R d max , with unnormalized, dimension-saturated, conditional target density

<!-- formula-not-decoded -->

Defined on the same augmented support is the family of saturated variational densities

<!-- formula-not-decoded -->

where, noting the availability of a transport ( θ m , u \ m ) = T ϕ ( z | m ) , z ∈ U d max , we define the IAF

<!-- formula-not-decoded -->

Our goal is to construct the IAF so that equation 4 factorizes into active and i.i.d. auxiliary parts, i.e.

<!-- formula-not-decoded -->

and to exploit this factorization in the construction of a transdimensional loss function. To achieve this factorization, we define the following notation. Let A i : M→{ 0 , 1 } flag whether latent coordinate i appears in model m , and let B i : { 0 , 1 } → { 0 , 1 } | ρ i | , B i ( b ) = ( b, . . . , b ) , broadcast this bit to the corresponding parameter block. Their composition C i := B i ◦ A i : M→{ 0 , 1 } | ρ i | therefore activates exactly the autoregressive parameters ρ i needed by τ ρ i ( z ( i ) ) under model m . Concatenating the blocks gives the global context-to-mask map (see Figure 1(b) for a visualization):

<!-- formula-not-decoded -->

Similarly, A and B denote the respective coordinate-concatenated maps similar in form to equation 6. After a fixed left-align permutation aligning latents with θ m , Proposition 2.2 proves this factorization is exact for any autoregressive network NN ϕ that parametrizes the transport T ϕ .

Recalling the univariate bijective maps of the inverse autoregressive flow as τ ρ i : R ↦→ Θ i for i = 1 , . . . , d max , we assume the existence of a static point ρ Id such that τ ρ Id ( z ) = z for all z ∈ R , i.e., the transform becomes the identity map at ρ Id . For example, a simple affine transformation (scale and location shift) is θ = τ ρ i ( z ) = ρ (0) + ρ (1) z , where ρ i = ( ρ (0) , ρ (1) ) . In this case, the static point is ρ Id = (0 , 1) as then θ = z . We can then construct a simple mechanism for 'choosing' between ρ i and ρ Id for each individual transform τ , i = 1 , . . . , d max , via the convex combination

<!-- formula-not-decoded -->

Each coordinate-wise transform then becomes θ ( i ) m = τ ρ C i ( z ( i ) ) , i ∈ { 1 , . . . , d max } . That is, the transformation parameters become a context-dependent composition of the elements of ρ i and the static point ρ Id (Figure 1(c)). A composition of transforms parametrized according to equation 7 is a Contextually-Specified Masking for Identity-mapped Components (CoSMIC) normalizing flow.

<!-- formula-not-decoded -->

Proposition 2.2. Fix m ∈ M . Let P m be the permutation matrix that places the coordinates indexed by I ( m ) (from the proof of Theorem 2.1) before those in I c ( m ) while preserving the original order inside each group. Define the left-align-permuted flow T ◁ ϕ := P -1 m ◦ T ϕ ◦ P m and the corresponding density ˜ q ◁ ϕ ( θ m , u \ m ) = ν d max ( z ) | det ∇ T ◁ ϕ ( z | m ) | -1 , z = T ◁, -1 ϕ ( θ m , u \ m ) . Redefine C := C ◁ = B ◦ P m ◦ A . Then (a) ˜ q ◁ ϕ ( θ m , u \ m ) factorizes as per equation 5 with the substitution ˜ q ϕ := ˜ q ◁ ϕ , and (b) the marginal q ϕ ( θ m | m ) is consistent.

Figure 1: (a) CoSMIC flow composition, (b) Context-to-mask map, (c) A single CoSMIC IAF step.

<!-- image -->

From here on, we use the notational convenience T ϕ := T ◁ ϕ and q ϕ := q ◁ ϕ to denote the composition of transforms and associated variational density that include the left-align permutation P m required by Proposition 2.2. We also write the partitioning z = ( z m , z \ m ) as explicitly obtained by [ z m z \ m ] ⊤ = P m z . By construction, ν d max = ν d m ⊗ ν d \ m , i.e. ν d max ( z ) = ν d m ( z m ) ν d \ m ( z \ m ) .

Corollary 2.3. Given Lemma 2.1 and Proposition 2.2, then

<!-- formula-not-decoded -->

and, substituting ℓ ( m ; ϕ ) := E z ∼ ν d max [log h ϕ ( z | m )] , the loss in equation 1 becomes

<!-- formula-not-decoded -->

Proposition 2.2 states that, conditional on model m , the CoSMIC IAF (Figure 1) achieves the factorization of the saturated-space variational approximation in equation 5. From Corollary 2.3, this means that when computing the loss function in equation 1, the ratio of the dimension-saturated variational density ˜ q and conditional target ˜ η (equation 8, LHS) which are both d max -dimensional and which involve the auxiliary variables, collapses down to a direct comparison only on the d m -dimensional model specific densities q ϕ ( θ m | m ) and η ( θ m | m ) (equation 8, RHS), and without the involvement of any auxiliary variables. That is, the CoSMIC flow enables the IAF to calculate on a fixed-dimensional space, while permitting the model-specific comparison within the loss function to operate on the natural d m -dimensional space.

The implementation of a CoSMIC inverse autoregressive flow step T i as part of a composition of transforms T L ◦ · · · ◦ T 1 is visualized in Figure 1(a). Individual architectures for affine and rational quadratic spline transforms [13] and compositions are described in Appendix A.2.

## 3 Formulating a model weights distribution

Formulating and estimating q ψ is not as straightforward as that of q ϕ because the discrete random variables m ∼ q ψ are not automatically linked to the density parameters ψ by automatic differentiation. This problem naturally lends itself to methods developed in black-box variational inference [41, 53, 54] and multi-armed bandits [6], as described below. The representation of m is any discrete random variable on a finite space M . Writing the true distribution of m as π m , a finite M implies the existence of a categorical distribution π ζ which is bijectively equivalent to π m . The random variables ζ ∼ π ζ exist on the finite support ζ ∈ C ⊂ N , thus |C| = |M| . This property is used by the surrogate-based approach described in Section 3.1.We formalize this concept via Theorem D.1.

We consider two approaches to model q ψ . Firstly, we derive a non-parametric surrogate-based approach which comes equipped with theoretical convergence guarantees and is applicable to model spaces M of low cardinality. We then present an approach based on parametric models that can scale to arbitrarily large spaces M that are trained using doubly stochastic gradient estimators.

## 3.1 Estimation via surrogate

The objective in Equation 9 can be rewritten as a single-variable objective with respect to ϕ :

<!-- formula-not-decoded -->

where P Ψ denotes the space of probability measures over M parameterized by ψ ∈ Ψ ⊆ R n ψ , and H denotes entropy. If we replace P Ψ by P ( M ) , i.e., the whole space of probability measures over M , the solution to the inner optimization problem admits a closed-form expression:

<!-- formula-not-decoded -->

Computing the expression above within an optimization loop over ϕ in practice would, however, require the evaluation of flow-based densities over the entire model space. We may, instead, follow a cheaper-to-evaluate density q u,ϕ which approximates q ∗ ℓ,ϕ for a given ϕ , by means of learning a surrogate model over ℓ within the same optimization loop 2 . In particular, we derive a Gaussian process (GP) upper confidence bound [49], which provides the following approximation to the optimal model probabilities:

<!-- formula-not-decoded -->

where u t ( m ) := µ t ( m,ϕ t ) + βσ t ( m,ϕ t ) , with µ t and σ 2 t representing the posterior mean and variance of a GP model conditioned on all mini-batches of data B t := { ϕ t -1 , m t,i , log h ϕ t -1 ( z t,i | m t,i ) } B i =1 available at iteration t of stochastic gradient descent, and ϕ t denotes the current flow parameters. In this form, u t provides an upper confidence bound (UCB) over -ℓ ( m ; ϕ t ) determined by the choice of confidence parameter β ≥ 0 . The GP posterior mean and variance can be derived in closed form if the observation noise is Gaussian with, e.g., variance σ 2 ϵ . We, however, show that a sub-Gaussian noise assumption is sufficient to use a conventional GP model. In addition, if ϕ t follows a convergent sequence (e.g., by ensuring diminishing step sizes during gradient-based optimization), we have the following guarantee.

Corollary 3.1. Let ℓ ∼ GP (0 , κ ) , where κ : ( M× Φ) 2 → R is a bounded, continuous positivesemidefinite kernel over M× Φ . Assume log h ϕ ( z | m ) -ℓ ( m ; ϕ ) is σ 2 ϵ -sub-Gaussian with respect to z ∼ ν . Then, if ϕ t follows a convergent sequence, the following also holds:

<!-- formula-not-decoded -->

where O P characterizes convergence in probability. 3

The result above tells us that the UCB-based models distribution approaches the optimal distribution at a rate of O P ( t -1 / 2 ) and ultimately converges to it as t → ∞ . Therefore, a stochastic gradient optimizer using samples from the surrogate density q u,t should asymptotically converge to the optimization path determined by the optimal q ∗ ℓ,ϕ t . That is, under appropriate settings for, e.g., its learning rate schedule, the optimization will converge to ϕ ∗ . Lastly, note that the result in Theorem 3.1 is independent of the choice of β , which can be set to β = 0 . Our analysis is mainly based on obtaining enough samples almost everywhere across the model space, which can be ensured by sampling according to the predictive mean µ t of the surrogate, as exp µ t &gt; 0 under mild assumptions. However, in practice, a non-zero value of β helps to accelerate convergence in finite time by encouraging exploration. Corollary 3.1 is a direct application of Theorem C.3, proved in Appendix C.5, where we also contrast it with existing results [36].

Due to the reliance on GP-based approximations, a naive implementation of this approach would incur a cost of O ( B 3 t 3 ) per stochastic gradient step, where B is the mini-batch size, due to matrix inversions [42]. However, for model spaces of moderate cardinality |M| = M , we can keep compute

2 We are here assuming that the prior p ( m ) is cheap to evaluate. If not, we can model -ℓ ( m ; ϕ ) + log p ( m ) , instead, with a surrogate, which leads to similar theoretical guarantees after minimal adjustments.

3 ξ t ∈ O P ( g t ) if lim C →∞ lim sup t →∞ P [ ξ t g -1 t &gt; C ] = 0 .

costs linear with the number of optimization steps by applying recursive equations to evaluate the GP posterior mean and covariance over the model space (see Eq. 21 and 22), leading to a cost of O ( B 3 + MB 2 + M 2 B ) = O ( M 2 B ) per step, as B ≪ M , totaling O ( TM 2 B ) over T steps. Sparse approximations to GPs can further reduce this cost [20, 42] to make it practical for larger model spaces. For our purposes, we implemented a diagonal Gaussian approximation, which makes the cost linear in the batch size and constant in t via a mean-field approximation.

## 3.2 Categorical and neural probability mass functions

By Theorem D.1, we may represent probability distributions over the model space M by arbitrarily parametrized categorical distributions. A drawback of the surrogate is the need to maintain and update estimates over the entire model space, which can be impractical for spaces of very large cardinality, such as DAG discovery. Hence, we introduce two parametric alternatives.

Categorical: Assume |M| = M ∈ N . Then, for ψ ∈ R M , the distribution over M is defined by q ψ ( m ) := ( ∑ M j =1 exp ψ j ) -1 ∑ M i =1 I [ m i = m ] exp ψ i . The logit weights vector ψ is unconstrained in R M and can be jointly optimized with ϕ by gradient methods. Density evaluations and the entropy can be computed with memory cost O ( |M| ) .

Autoregressive: If the model space is too large we may use a structured sample generation process which allows for the number of parameters to be smaller than cardinality of the model space i.e., dim( ψ ) &lt; M . For instance, Germain et al. [19] proposed an autoregressive parametrization for distributions over binary strings s ∈ { 0 , 1 } d s via the decomposition p ψ ( s ) = ∏ d s i =1 p ψ ( s i | s 1 , . . . , s i -1 ) . For each s , we assign a unique m ∈ M and define q ψ ( m ) := p ψ ( s ( m )) . The conditional densities and sampling can be implemented via MADE, allowing us to map the entire model space with fewer parameters when 2 d s ≥ |M| . The same reasoning can be applied to a DAG via decomposition of its adjacency matrix. Details of MADE are in Appendix G.8, and for DAGs in Appendix G.7.

## 3.3 Estimation via Monte Carlo gradients

When |M| is too large to use a surrogate-based approach, or to even parameterize an entire vector of categorical weights in physical memory, we can employ neural-based methods that use gradient descent and estimation of the gradients of ψ via Monte Carlo estimation of gradients (MCG) [34]. Using ∇ ψ q ψ ( m ) = q ψ ( m ) ∇ ψ log q ψ ( m ) , the gradient of the expectation in equation 9 with respect to ψ is

<!-- formula-not-decoded -->

In practice, the variance of this estimator can be very high. However, techniques exist to reduce this variance [34, 41 ? ] for general applications. We use a control variate ς in the form

<!-- formula-not-decoded -->

where g ( ϕ, ψ, ς ) = E z ∼ ν d max [log h ϕ ( z | m ) + log q ψ ( m ) -log p ( m ) -ς ] . We compute ς using the method described in Appendix E.1 (full description in Appendix E).

The benefit of using MCG for variational parameter estimation is the flexibility of choice for q ψ . We compare two: (1) MCG of the logits of a standard categorical distribution, and (2) MCG of multi-layer perceptron weights that parameterize a configuration of the MADE neural autoregressive density estimator of Germain et al. [19] (see Appendix G.7). When |M| is large, such implementations of q ψ permit an efficient approximate representation of the true model distribution.

## 3.4 Information-Limiting the optimization

The convergence of ψ → ψ ∗ is dependent on the convergence of ϕ → ϕ ∗ , and optimal sample efficiency for the inference of ϕ is achieved when ψ ≈ ψ ∗ . Intuitively, q ϕ should focus primarily

on the higher-probability models that contribute most to estimator variance, but discovering these models requires stable approximation of each q ϕ ( θ m | m ) to inform ∇ ψ . This circular dependence motivates practical regulation of the optimization of ψ when estimating ∇ ψ via Monte Carlo gradient estimates, addressing an instability similar to that discussed in the reinforcement learning literature [46], but without modifying the objective. Our approach is to reduce the variance of the estimates of q ψ by bounding the information gain in the transition q ψ t → q ψ t +1 , which determines the step size, thereby stabilizing the optimization (detailed in Appendix E.2).

## 4 Related work

Conditional normalizing flows [13, 55] have emerged as powerful tools for incorporating conditioning information. Existing methods use the context variable as a conditioning input, but fewer adapt the flow architecture itself. An exception is the transport-based reversible jump MCMC method [9], which learns proposals for transdimensional moves, but does not readily allow its use as an inverse autoregressive flow [28]. In contrast, we introduce an identity-parameterized CoSMIC transformation without identity-map training. We bypass path-wise approximations to discrete distributions [24, 33], instead comparing Monte Carlo gradient estimation [34] with Bayesian optimization [47]. We adopt an information-based approach to scale gradient steps using 'small steps,' inspired by reinforcement learning [46]. Bayesian methods for model selection and optimization have advanced with black-box variational inference [41, 53, 54] and flexible flows [13, 38, 43]. Recent work in amortized Bayesian mixture models [29] shows amortization over multiple mixture components using conditional normalizing flows, but not for variable dimensions. Conversely, Li et al. [31] introduces an architecture for learning imputation over transdimensional inputs, but lacks immediate application as a variational density. Our approach unifies transdimensional inference with flow-based variational methods, bypassing the need for tailored dimension jumps and broadening applications.

## 5 Experiments

We present experiments involving synthetic and real data on two representative applications: robust variable selection and directed acyclic graphs. To evaluate the quality of the approximation q ψ,ϕ ( m, θ m ) to the target distribution π ( m, θ m ) for a relatively small |M| &lt; 2 19 model space, we use the average negative log-likelihood (NLL) computed over a set of samples drawn from π via a baseline sampling method, in this case reversible jump MCMC [48]. Let { ( m i , θ i m ) } N i =1 denote N independent samples from π ( m, θ m ) . The average NLL corresponds to the cross-entropy H ( π, q ψ,ϕ ) between π and q ψ,ϕ , which quantifies the expected number of bits needed to encode samples from π using q ψ,ϕ , and is defined as NLL = 1 N ∑ N i =1 -log q ψ,ϕ ( m i , θ i m ) . Comparison of VTI DAG inference quality with baseline frequentist and Bayesian approaches use standard metrics [30].

## 5.1 Bayesian misspecified robust variable selection

We study a robust Bayesian variable selection problem where the response y ∈ R is related to predictors x ∈ R p (including an intercept) through a linear model. The innovation is a mixtureof-Gaussians noise specification, accommodating outliers via a heavy-tailed component. A subset indicator γ ∈ { 0 , 1 } p selects which predictors enter the model. If β ∈ R p are the coefficients, only the components where γ j = 1 contribute to the linear predictor. In particular, for data { ( x i , y i ) } n i =1 the prediction function is µ ( x ) = x ⊤ ( β ⊙ γ ) , the likelihood is

<!-- formula-not-decoded -->

and priors p ( γ ) = 2 -p and p ( β ) = N (0 , σ 2 β I ) . Here, α controls the fraction of outliers, and ( σ 2 1 , σ 2 2 ) encode the variances of in-distribution and outliers, respectively. To complicate the inference problem, two misspecified data-generating processes were used (medium- and high-misspecification) which encourages multi-modality in the approximating posterior π ( θ m | m ) .

0

Figure 2: Quality of VTI approximation for Bayesian misspecified robust variable selection. Outer columns denote medium (left) or high (right) likelihood misspecificaton, inner columns indicate different normalizing flow constructions, increasing flow expressivity from left to right. Flow types are described in Appendix A.2. Top row: Estimated model probabilities q ψ ( m ) vs true model probabilities π ( m ) on the log scale. Bottom row: Cross entropy between individual model estimates q ϕ ( θ m | m ) and true density π ( θ m | m ) versus true model probability. Colors indicate 10 replicated analyses, each with |M| = 2 7 models.

<!-- image -->

Figure 3: Left: A simulation study of the robust variable selection example showing the cross entropy (NLL) between RJMCMC samples and an flow-based variational transdimensional density using rational quadratic spline CoSMIC flows under a fixed number of iterations (30,000). Each cardinality was run with 10 independently sampled synthetic data sets. Right: Comparison of bivariate plots of variables θ (1) m , θ (5) m obtained by RJMCMC and VTI for a single |M| = 2 7 problem.

<!-- image -->

Table 2 in Appendix F summarizes the full experiment configuration. Figure 2 offers a holistic assessment of inference quality relative to a sampling baseline using RJMCMC, where cross-entropy reduces as flow expressivity increases. It shows two problem settings, mid and high misspecification, and for each setting shows how increasing complexity of the variational density (left-to-right panels) improves the quality of the approximations of both π ( θ m | m ) (bottom row) and estimated model probabilities (top row), and that the approximation quality of π ( θ m | m ) is higher for higher probability models.

Cardinality sweep: Using the focused prior setup on both the medium and high misspecification level targets, we sweep the cardinality of the model space |M| from 2 9 to 2 24 and compute the cross entropy H ( π, q ψ,ϕ ) , where samples ( m, θ m ) ∼ π are obtained via RJMCMC (see Appendix F.3). Figure 3 (left) compares the cross entropy between the three q ψ ( m ) types discussed in Sections 3.1 and 3.3 in simulated problems of increasing |M| . As expected, H ( π, q ψ,ϕ ) generally increases with |M| when the flow architecture is held fixed. The surrogate method (blue bars) performs comparably with the other methods for the smaller model spaces ( |M| = 2 9 ), whereas the neural density (orange bars) performs consistently as |M| increases. Figure 3 (right) shows two bivariate plots of selected variables ( θ (1) m , θ (5) m ) from the posterior inferred using RJMCMC and VTI. This qualitative visual comparison shows how well the CoSMIC flow is able to capture non-trivial model distributions versus the sampling approach (for the full multivariate comparison see Figure 11). Appendix F describes the experiments in further detail and demonstrates VTI robustness to diffuse priors.

10 node DAG MLP comparison using F1, SHD, Brier, and AUROC

Figure 4: Simulation study comparing VTI to DAGMA [1], DiBS/DiBS+ [32], and JSPGFlowNets [10] for discovery of a 10-node non-linear DAG visualized using standard metrics (Appendix G.2, left to right, where better is: higher, lower, lower, higher). Bars display mean and standard error over nine i.i.d. repetitions for each data set size.

<!-- image -->

## 5.2 Bayesian structure learning via non-linear directed acyclic graph discovery

We consider a dataset of real-valued observations, denoted by X ∈ R n × N d , where n is the number of data samples and N d is the number of nodes. Our goal is to perform Bayesian inference over a space of non-linear structural equation models (SEMs) which is isomorphic to a space of directed acyclic graphs (DAGs) and non-linear functions over the active edges. A DAG is represented by a directed adjacency matrix A ∈ { 0 , 1 } N d × N d , where A ij = 1 indicates a directed edge from node i to node j and A ij = 0 otherwise. The acyclicity constraint requires that the directed edges in A do not form any directed cycle. In a non-linear SEM, each node X j depends non-linearly on its parents in the form X = f ( X ) + ϵ , ϵ j ∼ N ( 0 , σ 2 ) , where f : R N d ↦→ R N d is a nonlinear function possessing an acyclic Jacobian matrix. We follow [1, 52] whereby f is a multi-layer-perceptron (MLP) structured as f ( X ) = ( f 1 ( X ) , . . . , f N d ( X )) ⊤ . We implement f using a single hidden layer, with rectified linear unit (ReLU) activation functions used to model non-linearity where the bias term can be optionally included (see Appendix G). By introducing a topological ordering of the N d nodes, we simultaneously enforce acyclicity and a consistent mapping of parameters to each graph. Let P be a permutation matrix that reorders nodes into a valid topological order and define U to be strictly upper-triangular. By construction, any acyclic adjacency matrix can be represented as A = P ⊤ UP . Each edge is guaranteed to point from lower-indexed nodes to higher-indexed nodes in the topological order [3]. Note that this parametrization does not conform to Theorem D.1, as the correspondence between ( P , U ) and A is many-to-one. However, this does not violate the consistent parameter mapping. We use a MADE-based discrete distribution [19] for q ψ for inference over a very high cardinality model space (see Appendices G.8 and G.7 for details). The simulation study in Figure 4 contrasts VTI with state-of-the-art Bayesian and non-Bayesian baselines (DiBS/DiBS+ [32], JSP-GFlowNets [10], and DAGMA [1]) with the aim of demonstrating that the performance of the generic VTI approach can be competitive with application-specific approaches, where one would expect the latter to have better performance. Evaluation of each method is depicted using the commonly accepted F1 score, structural Hamming distance (SHD), Brier score, and area under receiver operating curve (AUROC) (see Appendix G.2). A complete description of this study is in Appendix G.4.

Real data example in flow cytometry: Sachs et al. [44] use Bayesian networks to analyze multiparameter single-cell data for deriving causal influences in cellular signaling networks of human immune cells. Causal interactions are validated by comparing to a domain-agreed adjacency matrix representing causality within the data, establishing a baseline for causal prediction accuracy. We use VTI to discover the distribution of non-linear DAGs for these data, comprising n = 7466 entries over

Table 1: Comparison of DAG discovery on flow cytometry data [44]: VTI versus baselines

| Method             |   F1 |   SHD |   Brier |   AUROC |
|--------------------|------|-------|---------|---------|
| VTI non-linear DAG | 0.44 |  23   |      23 |    0.68 |
| DAGMA non-linear   | 0.32 |  25   |      25 |    0.6  |
| DiBS+ non-linear   | 0.22 |  28   |      17 |    0.54 |
| JSP-GFN non-linear | 0.23 |  54.5 |      44 |    0.51 |

N d = 11 nodes, and benchmark this against the agreed adjacency. Table 1 shows strong performance of VTI compared to state of the art methods. A complete description is in Appendix G.5.

## 6 Discussion

Wehave introduced CoSMIC normalizing flows as a means to implement variational transdimensional inference (VTI), the approximation of a target density over a transdimensional space with a single variational density. VTI is broadly applicable to a wide class of transdimensional inference problems. Although the specification of a CoSMIC flow requires augmenting all model dimensions to d max , VTI is not sensitive to these added dimensions during training and inference due to the construction of the auxiliary variable transforms. We presented two approaches for simultaneously optimizing the variational parameters ψ, ϕ . The Gaussian surrogate-based approach benefits from our derivation of the approximation error bounds and established convergence guarantees for the marginal models distribution under convergent optimization steps. The two approaches that use Monte Carlo gradient estimation for SGD optimization benefit from recent advances in neural architectures and neural approximation of very large model spaces. The choice of model sampler is dependent on both the cardinality of the model space and the structure of the problem. When |M| is small, the Gaussian process surrogate-based sampler or the categorical sampler using Monte Carlo gradients are both appropriate, although in practice it is usually safe to default to the latter approach. For high cardinality problems, we recommend a neural model sampler for approximate inference on the distribution of model weights.

The quality of the VTI approximation possesses two notable characteristics. Firstly, those models m ∈ M estimated to have large posterior model probabilities will contribute most significantly to the loss. Hence the CoSMIC flow will produce a relatively more accurate (in the KL sense) approximation of such models, compared to models with low probabilities. This effect is seen in Figure 2 (bottom row). While one might prefer greater accuracy on more dominant models, structured changes to L ( ψ, ϕ ) could give greater control over where the quality of the variational approximation should focus. The second characteristic is that when the normalizing flow is unable to approximate the conditional target π ( θ m | m ) well, a smaller loss can be achieved by shrinking the estimated model probability q ψ ( m ) to zero. This effect is seen in Figure 2 (top row), which lessens as flow expressivity increases. Here the question is how to design the normalizing flow, i.e. the flow context ξ and the mapping C ( m ) , to best allocate resources to produce good approximations of models likely to be of relatively high posterior model probability. In many transdimensional problems, two models could be considered adjacent by structural similarity (e.g. in variable selection where they differ by one included covariate) and so may have similar posterior model probabilities. This could be achieved by e.g. extending the architecture of the context encoder (Appendix A.2) to capture similarities between models that generalize over the model space, and learning structural similarity within the surrogate-based model sampler itself via a reward-based criterion.

The left-align permutation used in Proposition 2.2 raises the question of whether the alignment of variables across models in the normalizing flow is important for computational efficiency. More broadly: (a) is there shared information between models? And, if so, (b) would careful manual construction of the flow improve exploitation of this versus allowing the optimization to determine it agnostically? (a) is answered by the robust variable selection example in Figures 3 (right) and 11 where high probability mass regions for each model do not overlap and thus there there is no such shared information. To answer (b) would be an avenue for future research. Future work could also derive convergence rates, which will depend on the choice of optimization algorithm for the flow parameters, and extension of the CoSMIC architecture to coupling flows, in applications outside variational inference. In addition, our analysis for the surrogate-based approach is general enough to be extended to a variety of methods for approximating a distribution over models. Finally, in extending VI to the transdimensional setting we inherit the same strengths and weaknesses of singlemodel VI, including the challenges of mode-collapse. Users would need to take the same steps to manage it as in the standard setting. Mode collapse in the model distribution is mitigated by the exploration versus exploitation strategies discussed in Section 3.

## Acknowledgments

This research was made possible by funding support from the CSIRO Machine Learning and Artificial Intelligence Future Science Platform. The authors would also like to thank Ryan Thompson for critical advice in the construction of the non-linear DAG example, and Daniel M. Steinberg and Edwin Bonilla for advice on Monte Carlo gradient estimation methods. SAS is supported by the Australian Research Council.

## References

- [1] Kevin Bello, Bryon Aragam, and Pradeep Ravikumar. DAGMA: Learning DAGs via m-matrices and a log-determinant acyclicity characterization. In Advances in Neural Information Processing Systems , 2022.
- [2] David M. Blei, Alp Kucukelbir, and Jon D. McAuliffe. Variational inference: A review for statisticians. Journal of the American Statistical Association , 112(518):859-877, 2017. ISSN 0162-1459, 1537-274X. doi: 10.1080/01621459.2017.1285773. URL https://www.tandfonline.com/doi/full/10.1080/ 01621459.2017.1285773 .
- [3] Edwin V. Bonilla, Pantelis Elinas, He Zhao, Maurizio Filippone, Vassili Kitsios, and Terry O'Kane. Variational DAG estimation via state augmentation with stochastic permutations, 2024. URL http: //arxiv.org/abs/2402.02644 .
- [4] Stéphane Boucheron, Gábor Lugosi, and Pascal Massart. Concentration inequalities: A nonasymptotic theory of independence . Oxford University Press, 2013.
- [5] S. P. Brooks, P. Giudici, and G. O. Roberts. Efficient construction of reversible jump Markov chain Monte Carlo proposal distributions. Journal of the Royal Statistical Society Series B: Statistical Methodology , 65 (1):3-39, 2003. ISSN 1369-7412. doi: 10.1111/1467-9868.03711. URL https://doi.org/10.1111/ 1467-9868.03711 .
- [6] Sébastien Bubeck and Nicolò Cesa-Bianchi. Regret analysis of stochastic and nonstochastic multi-armed bandit problems , volume 5. Now, 2012. ISBN 978-1-60198-626-9. doi: 10.1561/2200000024. URL http://arxiv.org/abs/1204.5721 . OCLC: 931223688.
- [7] Andrea Coccaro, Marco Letizia, Humberto Reyes-González, and Riccardo Torre. Comparison of affine and rational quadratic spline coupling and autoregressive flows through robust statistical tests. Symmetry , 16(8):942, 2024. ISSN 2073-8994. doi: 10.3390/sym16080942. URL https://www.mdpi.com/20738994/16/8/942 . Publisher: MDPI AG.
- [8] Moumita Das and Sourabh Bhattacharya. Transdimensional transformation-based Markov chain Monte Carlo. arXiv preprint arXiv:1403.5207 , 2014.
- [9] Laurence Davies, Robert Salomone, Matthew Sutton, and Chris Drovandi. Transport reversible jump proposals. In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics , pages 6839-6852. PMLR, 2023. URL https://proceedings.mlr.press/v206/davies23a.html . ISSN: 2640-3498.
- [10] Tristan Deleu, Mizu Nishikawa-Toomey, Jithendaraa Subramanian, Nikolay Malkin, Laurent Charlin, and Yoshua Bengio. Joint Bayesian inference of graphical structure and parameters with a single generative flow network. Advances in Neural Information Processing Systems , 36: 31204-31231, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html .
- [11] Gian C. Diluvi, Benjamin Bloem-Reddy, and Trevor Campbell. Mixed variational flows for discrete variables. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics , pages 2431-2439. PMLR, 2024. URL https://proceedings.mlr.press/v238/diluvi24a.html . ISSN: 2640-3498.
- [12] Lester E. Dubins and David A. Freedman. A sharper form of the Borel-Cantelli lemma and the strong law. The Annals of Mathematical Statistics , 36(3):800-807, 1965. ISSN 00034851. URL http://www.jstor. org/stable/2238191 . Publisher: Institute of Mathematical Statistics.
- [13] Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows. In Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019.

- [14] Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Nflows: Normalizing flows in PyTorch. Zenodo, 2020.
- [15] Richard G. Everitt, Richard Culliford, Felipe Medina-Aguayo, and Daniel J. Wilson. Sequential Monte Carlo with transformations. Statistics and Computing , 30:663-676, 2020.
- [16] Yanan Fan, Scott A. Sisson, and Laurence Davies. Reversible jump Markov chain Monte Carlo and multi-model samplers. In Handbook of Markov Chain Monte Carlo . Chapman and Hall/CRC, 2026. URL https://arxiv.org/abs/1001.2055 .
- [17] Marylou Gabrié, Grant M. Rotskoff, and Eric Vanden-Eijnden. Adaptive Monte Carlo augmented with normalizing flows. Proceedings of the National Academy of Sciences , 119(10):e2109420119, 2022. doi: 10.1073/pnas.2109420119. URL https://www.pnas.org/doi/full/10.1073/pnas.2109420119 . Publisher: Proceedings of the National Academy of Sciences.
- [18] Andrew Gelman and Yuling Yao. Holes in Bayesian statistics. Journal of Physics G: Nuclear and Particle Physics , 48(1):014002, 2020. ISSN 0954-3899. doi: 10.1088/1361-6471/abc3a5. URL https: //dx.doi.org/10.1088/1361-6471/abc3a5 . Publisher: IOP Publishing.
- [19] Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE: Masked autoencoder for distribution estimation. In Proceedings of the 32nd International Conference on Machine Learning , pages 881-889. PMLR, 2015. URL https://proceedings.mlr.press/v37/germain15.html . ISSN: 1938-7228.
- [20] Arjan Gijsberts and Giorgio Metta. Real-time model learning using incremental sparse spectrum Gaussian process regression. Neural Networks , 41:59-69, 2013. ISSN 08936080. doi: 10.1016/j.neunet.2012.08.011. URL https://linkinghub.elsevier.com/retrieve/pii/S0893608012002249 .
- [21] Peter J. Green. Reversible jump Markov chain Monte Carlo computation and Bayesian model determination. Biometrika , 82(4):711-732, 1995. ISSN 0006-3444. doi: 10.1093/biomet/82.4.711. URL https: //doi.org/10.1093/biomet/82.4.711 .
- [22] David Heckerman, Christopher Meek, and Gregory Cooper. A Bayesian approach to causal discovery. In Dawn E. Holmes and Lakhmi C. Jain, editors, Innovations in Machine Learning: Theory and Applications , pages 1-28. Springer, 2006. ISBN 978-3-540-33486-6. URL https://doi.org/10.1007/3-54033486-6\_1 .
- [23] Matthew D. Hoffman, David M. Blei, Chong Wang, and John Paisley. Stochastic Variational Inference. Journal of Machine Learning Research , 14(40):1303-1347, 2013.
- [24] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with Gumbel-softmax. In International Conference on Learning Representations , 2017. URL https://openreview.net/forum? id=rkE3y85ee .
- [25] Michael I. Jordan, Zoubin Ghahramani, Tommi S. Jaakkola, and Lawrence K. Saul. An introduction to variational methods for graphical models. Machine Learning , 37(2):183-233, 1999. ISSN 1573-0565. doi: 10.1023/A:1007665907178. URL https://doi.org/10.1023/A:1007665907178 .
- [26] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. URL http: //arxiv.org/abs/1412.6980 .
- [27] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 , 2013. URL https://openreview.net/forum?id=33X9fd2-9FyZd .
- [28] Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. Advances in neural information processing systems , 29, 2016.
- [29] Simon Kucharsky and Paul Christian Burkner. Amortized Bayesian Mixture Models, 2025.
- [30] Erich Kummerfeld and Alexander Rix. Simulations evaluating resampling methods for causal discovery: ensemble performance and calibration. In 2019 IEEE international conference on bioinformatics and biomedicine (BIBM) , pages 2586-2593. IEEE, 2019.
- [31] Yang Li, Shoaib Akbar, and Junier Oliva. ACFlow: Flow models for arbitrary conditional likelihoods. In Proceedings of the 37th International Conference on Machine Learning , pages 5831-5841. PMLR, 2020. URL https://proceedings.mlr.press/v119/li20a.html . ISSN: 2640-3498.

- [32] Lars Lorch, Jonas Rothfuss, Bernhard Schölkopf, and Andreas Krause. DiBS: Differentiable bayesian structure learning. In Advances in Neural Information Processing Systems , volume 34, pages 24111-24123. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/ hash/ca6ab34959489659f8c3776aaf1f8efd-Abstract.html .
- [33] Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. In International Conference on Learning Representations , 2017. URL https://openreview.net/forum?id=S1jE5L5gl .
- [34] Shakir Mohamed, Mihaela Rosca, Michael Figurnov, and Andriy Mnih. Monte carlo gradient estimation in machine learning. Journal of Machine Learning Research , 21(132):1-62, 2020. ISSN 1533-7928. URL http://jmlr.org/papers/v21/19-346.html .
- [35] R. B. O'Hara and M. J. Sillanpää. A review of bayesian variable selection methods: what, how and which. Bayesian Analysis , 4(1):85-117, 2009. ISSN 1936-0975, 1931-6690. doi: 10.1214/09-BA403. URL https://projecteuclid.org/journals/bayesian-analysis/volume-4/issue-1/A-reviewof-Bayesian-variable-selection-methods--what-how/10.1214/09-BA403.full . Publisher: International Society for Bayesian Analysis.
- [36] Rafael Oliveira, Lionel Ott, and Fabio Ramos. No-regret approximate inference via bayesian optimisation. In Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence , pages 2082-2092. PMLR, 2021. URL https://proceedings.mlr.press/v161/oliveira21a.html . ISSN: 26403498.
- [37] John Paisley, David M. Blei, and Michael I. Jordan. Variational bayesian inference with stochastic search. In Proceedings of the 29th International Coference on International Conference on Machine Learning , ICML'12, pages 1363-1370. Omnipress, 2012. ISBN 978-1-4503-1285-1.
- [38] George Papamakarios, Theo Pavlakou, and Iain Murray. Masked autoregressive flow for density estimation. In Advances in Neural Information Processing Systems , volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 6c1da886822c67822bcf3679d04369fa-Abstract.html .
- [39] George Papamakarios, David Sterratt, and Iain Murray. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics , pages 837-848. PMLR, 2019. URL https://proceedings.mlr. press/v89/papamakarios19a.html . ISSN: 2640-3498.
- [40] Gilles Pisier. Subgaussian sequences in probability and fourier analysis. Graduate Journal of Mathematics , 1:59-78, 2016. URL http://arxiv.org/abs/1607.01053 . Publisher: Mediterranean Institute for the Mathematical Sciences tex.arxivid: 1607.01053.
- [41] Rajesh Ranganath, Sean Gerrish, and David M Blei. Black box variational inference. In Proceedings of the 17th International Conference on Artificial Intelligence and Statistics (AISTATS) , 2014.
- [42] Carl E. Rasmussen and Christopher K. I. Williams. Gaussian Processes for Machine Learning . The MIT Press, 2006. ISBN 0-262-18253-X.
- [43] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Proceedings of the 32nd International Conference on Machine Learning , pages 1530-1538. PMLR, 2015. URL https://proceedings.mlr.press/v37/rezende15.html . ISSN: 1938-7228.
- [44] Karen Sachs, Omar Perez, Dana Pe'er, Douglas A. Lauffenburger, and Garry P. Nolan. Causal ProteinSignaling Networks Derived from Multiparameter Single-Cell Data. Science , 308(5721):523-529, 2005. doi: 10.1126/science.1105809.
- [45] Malcolm Sambridge, Thomas Bodin, Kevan Gallagher, and Hrvoje Tkalcic. Transdimensional inference in the geosciences. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences , 371(1984):20110547, 2013.
- [46] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL http://arxiv.org/abs/1707.06347 .
- [47] Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P. Adams, and Nando de Freitas. Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE , 104(1):148-175, 2016. ISSN 1558-2256. doi: 10.1109/JPROC.2015.2494218. URL https://ieeexplore.ieee.org/abstract/ document/7352306 . Conference Name: Proceedings of the IEEE.

- [48] Scott A Sisson. Transdimensional markov chains: A decade of progress and future perspectives. Journal of the American Statistical Association , 100(471):1077-1089, 2005. ISSN 0162-1459, 1537274X. doi: 10.1198/016214505000000664. URL http://www.tandfonline.com/doi/abs/10.1198/ 016214505000000664 .
- [49] Niranjan Srinivas, Andreas Krause, Sham Kakade, and Matthias Seeger. Gaussian process optimization in the bandit setting: no regret and experimental design. In Proceedings of the 27th International Conference on International Conference on Machine Learning , ICML'10, pages 1015-1022. Omnipress, 2010. ISBN 978-1-60558-907-7. URL https://arxiv.org/abs/0912.3995 .
- [50] Niranjan Srinivas, Andreas Krause, Sham M. Kakade, and Matthias Seeger. Gaussian process optimization in the bandit setting: No regret and experimental design. IEEE Transactions on Information Theory , 58(5):3250-3265, 2012. ISSN 0018-9448, 1557-9654. doi: 10.1109/TIT.2011.2182033. URL http: //arxiv.org/abs/0912.3995 .
- [51] Daniel M. Steinberg, Rafael Oliveira, Cheng Soon Ong, and Edwin V. Bonilla. Variational search distributions. In NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty , 2024. URL https://openreview.net/forum?id=UYfx9b7Z1j .
- [52] Ryan Thompson, Edwin V. Bonilla, and Robert Kohn. ProDAG: Projection-Induced Variational Inference for Directed Acyclic Graphs, 2025.
- [53] Michalis Titsias and Miguel Lázaro-Gredilla. Doubly stochastic variational bayes for non-conjugate inference. In Proceedings of the 31st International Conference on Machine Learning , pages 1971-1979. PMLR, 2014. URL https://proceedings.mlr.press/v32/titsias14.html . ISSN: 1938-7228.
- [54] David Wingate and Theophane Weber. Automated variational inference in probabilistic programming, 2013. URL http://arxiv.org/abs/1301.1299 .
- [55] Christina Winkler, Daniel Worrall, Emiel Hoogeboom, and Max Welling. Learning likelihoods with conditional normalizing flows. CoRR , 2019. URL https://openreview.net/forum?id=rJg3zxBYwH .
- [56] Andrew Zammit-Mangion, Matthew Sainsbury-Dale, and Raphaël Huser. Neural methods for amortized inference. Annual Review of Statistics and Its Application , 2024. ISSN 2326-8298, 2326-831X. doi: 10.1146/annurev-statistics-112723-034123. URL https://www.annualreviews.org/content/ journals/10.1146/annurev-statistics-112723-034123 .

## A Implementation of CoSMIC normalizing flows

## A.1 Inverse autoregressive flow sampling procedure

Draw reference samples

<!-- formula-not-decoded -->

For a given m , define the permutation matrix P m that groups active coordinates first:

<!-- formula-not-decoded -->

Concatenate the coordinate-wise transforms τ ρ i into the map T ϕ and bookend with permutations P m to give the strict CoSMIC bijection

<!-- formula-not-decoded -->

## A.2 Experimental CoSMIC transform compositions

The experiments use the below compositions of transforms as inverse autoregressive flows T ϕ ( z | m ) where z are the inputs from the reference distribution and m is the context input. All compositions except for the diagonal Gaussian are assumed to have the strict left-align permutations discussed in Appendix A. The term 'block' is defined in Appendix A.3.

Context encoder: Experiments will sometimes use a context encoder that projects the context input to a higher dimensional space. Typically this will take the form of a multi-layered perceptron with hidden layers of increasing size (fixed to powers of 2) and terminating in an activation layer at the largest size, say 2 12 nodes.

Model-specific reverse-permutation: Flow compositions commonly include reverse permutations to ensure expressibility of an autoregressive-NN-based flow is (approximately) the same for all coordinates. Denoting the generic reverse permutation for all coordinates as P rev , we assume the strict left-right permutation P m (as per Appendix A) has been applied, and hence define the left-most d m -coordinate reverse permutation P rev &lt;d m .

Affine(5,5): The learned component is the affine masked autoregressive transform [38], denoted here as T Affine ϕ k for transforms k = 1 , . . . , 5 , each having 5 blocks. We set T ϕ := T Affine ϕ 5 ◦ P rev &lt;d m ◦· · · ◦ P rev &lt;d m ◦ T Affine ϕ 1 .

Spline(4,6): The learned component is the rational quadratic spline masked autoregressive flow architecture [13], denoted here as T RQ-Spline ϕ k . Each T RQ-Spline ϕ k has 6 blocks. Additionally, we define a fixed global affine transform T µ g ,σ g that is not dependent on inputs nor context and hence has only two learnable parameters: scale µ g and shift σ g . We set T ϕ := T µ g ,σ g ◦ T RQ-Spline ϕ 5 ◦ P rev &lt;d m ◦ · · · ◦ P rev &lt;d m ◦ T RQ-Spline ϕ 1 .

## A.3 Autoregressive flow definitions

We use the residual variant of the Masked Autoencoder for Distribution Estimation (MADE) [19], implemented in PyTorch by [14]. Each block maintains the autoregressive property by assigning degrees { 1 , . . . , d } to inputs and propagating them forward.

Given input x ∈ R d and optional context z , each residual block computes:

<!-- formula-not-decoded -->

Here, MaskedLinear i are masked linear layers respecting the autoregressive structure, BN i are optional batch norm layers, and δ ( z ) is an optional context projection. All layers preserve feature dimensionality and respect degree ordering to ensure autoregressive validity.

## B Analysis of a CoSMIC normalizing flow

Proof of Theorem 2.1. Let I ( m ) = { i ∈ { 1 , . . . , d max } : A i ( m ) = 1 } , and I c be the complement. The result holds from equation 7 as, for all coordinates i ∈ I c ( m ) , u ( i ) \ m = τ ρ C i ( z ( i ) \ m ) = τ ρ Id ( z ( i ) \ m ) = z ( i ) \ m .

Proof of Theorem 2.2. (a)-(b) Density factorization and marginal consistency.

We aim to prove

<!-- formula-not-decoded -->

Write T ◁, -1 ϕ = ( T -1 ϕ,m , Id) , where Id denotes the identity transform, and let the permuted reference vector be P m z = ( z m , z \ m ) ∈ R d m × R d \ m . Because the masking function C sets every transform τ ρ i with C i ( m ) = 0 to the identity, the inverse flow splits as

<!-- formula-not-decoded -->

where T -1 ϕ,m : Θ m → R d m is the active block and the dummy block is exactly the identity. Consequently, the Jacobian matrix of T ◁, -1 ϕ is block upper-triangular with det ∇ T ◁, -1 ϕ = det ∇ T -1 ϕ,m × 1 .

Apply change-of-variables with ν d max = ν d m ⊗ ν \ d m to obtain which proves the factorization (a).

<!-- formula-not-decoded -->

Integrating the right-hand side over u \ m recovers (b) q ϕ ( θ m | m ) , completing the proof.

Proof of Theorem 2.3. It is sufficient to show D KL (˜ q ψ,ϕ || ˜ η ) = D KL ( q ψ,ϕ || η ) := L ( ψ, ϕ ) . Note by equation 2, ˜ η ( m, θ m , u \ m ) = p ( m )˜ η ( θ m , u \ m | m ) = p ( m ) η ( θ m | m ) ν d \ m ( u \ m ) .

<!-- formula-not-decoded -->

Corollary B.1 (Computational complexity) .

- Sampling (forward IAF) : all coordinates can be updated in parallel ⇒ O (1) wall-time depth.
- Evaluation (inverse direction) : must populate z ( &lt;i ) sequentially ⇒ O ( d max ) arithmetic operations, identical to a standard IAF .

Proof. The forward IAF updates θ m via closed-form τ i that read previous outputs -all available after one pass through the network- which are thereby fully parallelizable. Conversely, evaluating T ( m ) , -1 ϕ at an arbitrary point in Θ m ×M must reconstruct z sequentially, exactly as for any IAF, giving O ( d max ) time.

## C Theoretical analysis of the model weights distribution

We consider the following bi-level stochastic optimization problem over a function f : M × Φ → R as:

<!-- formula-not-decoded -->

where P ( M ) denotes the space of probability measures over M , H is the entropy, and the optimal q f for a given ϕ can be shown to be:

<!-- formula-not-decoded -->

This formulation corresponds to a stochastic optimization problem over two variables ϕ and q f , where the optimum for q f has a closed-form expression q ∗ f,ϕ for every given ϕ ∈ Φ . To solve this problem, we will follow a sequential optimization process over ϕ (e.g., stochastic gradient descent). However, sampling from the optimal model distribution q ∗ f,ϕ (above) requires evaluating the summation in the normalization constant, which is expensive. Therefore, we will instead approximate each q ∗ f,ϕ t with a distribution q u,t composed of a cheaper-to-evaluate surrogate u t based on noisy observations y t -1 ,i = ˜ f ( z i , m i , ϕ t -1 ) , where z i ∼ ν and m i ∼ q u,t -1 , i ∈ { 1 , . . . , B } , such that E z ∼ ν [ ˜ f ( z , m, ϕ )] = f ( m,ϕ ) . If we ensure that q u,t approaches q ∗ f,ϕ t over time, optimization steps based on q u,t will eventually follow q ∗ f,ϕ t and allow for the optimum ϕ ∗ to be reached.

## C.1 Regularity assumptions

We make the following assumptions about the function f and the observation noise.

Assumption C.1. The objective f is a sample from a zero-mean Gaussian process prior with a bounded, positive-semidefinite covariance function κ : ( M × Φ) 2 → R , which is continuous over Φ .

The GP assumption allows us to derive closed-form expressions for predictions over f and their associated uncertainty. The continuity assumption on κ is easily satisfied by most practical covariance functions and ensures that, if ϕ t converges to some ϕ ∗ , GP-based estimates f ( m,ϕ ∗ ) will also converge for every m ∈ M . To model predictions over f with closed-form GP updates, we also need Gaussian assumptions about the observation noise, which is given by:

<!-- formula-not-decoded -->

However, as we will show in our analysis, sub-Gaussian tails are enough for GP modeling, which we formalize next.

Assumption C.2. The observation noise is σ 2 ϵ -sub-Gaussian, i.e., given any m ∈ M and ϕ ∈ Φ , we have:

<!-- formula-not-decoded -->

This mild assumption is satisfied, for example, when ν is a zero-mean Gaussian distribution and ˜ f is Lipschitz continuous on its first argument, in which case σ ϵ only depends on ˜ f through its Lipschitz constant [4, 40].

## C.2 Gaussian process model

Under the GP assumption f ∼ GP (0 , κ ) , the posterior over f is again a Gaussian process. Suppose at each iteration t ≥ 1 of stochastic gradient descent we sample a mini-batch { m t,i } B i =1 from a variational posterior approximating q ∗ f,ϕ at ϕ = ϕ t -1 . Given a batch of observations B t := { ϕ t -1 , m t,i , y t,i } B i =1 , the GP posterior f | B 1 ,...,t ∼ GP ( µ t , κ t ) has its mean and covariance described by the following recursive equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where κ t -1 ( m,ϕ ) := [ κ t -1 ( m,ϕ,m t,i , ϕ t -1 )] B i =1 ∈ R B , K t -1 := [ κ t -1 ( m t,i , ϕ t -1 , m t,j , ϕ t -1 )] B i,j =1 ∈ R B × B , and µ t -1 := [ µ t -1 ( m t,i , ϕ t -1 )] B t =1 ∈ R B , with µ 0 = 0 and κ 0 = κ . Any pointwise prediction is then modeled as f ( m,ϕ ) | B 1 ,...,t ∼ N ( µ t ( m,ϕ ) , σ 2 t ( m,ϕ )) , where σ 2 t ( m,ϕ ) := κ t ( m,ϕ,m,ϕ ) , for ( m,ϕ ) ∈ M × Φ .

## Algorithm 1 Stochastic optimization with UCB sampling

<!-- formula-not-decoded -->

## C.3 Upper confidence bound (UCB) algorithm

Given the GP posterior, we formulate an upper confidence bound algorithm [49] with:

<!-- formula-not-decoded -->

where β t &gt; 0 is a parameter controlling the size of the confidence bound, which we will discuss in our analysis. We then derive a sampling distribution based on using the UCB as a surrogate for f as:

<!-- formula-not-decoded -->

The solution to this optimization is available in closed form as the UCB softmax:

<!-- formula-not-decoded -->

Equipped with this UCB-based sampling distribution, we follow the generic procedure outlined in Algorithm 1. The algorithm starts by sampling from the current UCB distribution. A sample-based estimate of the optimization objective E m ∼ ϕ t [ f ( m,ϕ t -1 )] ≈ 1 B ∑ B i =1 ˜ f ( z t,i , m t,i , ϕ t -1 ) is then passed to the algorithm responsible for updating the parameters ϕ t , e.g., a stochastic gradient descent update. Once the parameters are updated, we reevaluate the objective and update our GP. The procedure then repeats up to a given number of iterations N ∈ N .

## C.4 Approximation errors under sub-Gaussian noise

In the following, we derive generic concentration bounds for GP predictions under sub-Gaussian observation noise. We start by showing that the approximation error between the GP mean and the true function is subGaussian.

Lemma C.1. Let f ∼ GP (0 , κ ) be a zero-mean Gaussian process with a given positive-definite covariance function κ : S × S → R . Assume we are given a sequence of observations y n = f ( x n ) + ϵ n , where x n ∈ S and ϵ n is σ 2 ϵ -sub-Gaussian noise, for all n ∈ N . Let µ n and σ 2 n denote the predictive mean and variance, respectively, of the GP posterior under the assumption that the noise is zero-mean Gaussian with variance given by σ 2 ϵ . Then, for all n ≥ 0 and all x ∈ S , we have that f ( x ) -µ n ( x ) is σ 2 n ( x ) -sub-Gaussian.

Proof. For n = 0 , the proof is trivial as, without observations, we only have the prior with µ 0 ( x ) = 0 and σ 2 0 ( x ) = κ ( x, x ) . Now let X n := { x i } n i =1 ⊂ S denote a set of n ≥ 1 observed locations. For any given x ∈ S , expanding the GP posterior mean from its definition, the approximation error can be decomposed as:

<!-- formula-not-decoded -->

where κ ( x, X n ) := [ κ ( x, x 1 ) , . . . , κ ( x, x n )] , K n := [ κ ( x i , x j )] n i,j =1 , f n := [ f ( x i )] n i =1 , and ϵ n := [ ϵ i ] n i =1 . The last term on the right-hand side above is sub-Gaussian, since E [ ϵ n ] = 0 and, letting α n := ( K n + σ 2 ϵ I ) -1 κ ( X n , x ) , we have a sum of independent sub-Gaussian random variables, see e.g. [40], Lemma 1.1:

<!-- formula-not-decoded -->

which follows from the definition of sub-Gaussian noise (cf. Assumption C.2). The remaining term on the right-hand side of equation 26 is a zero-mean Gaussian random variable with variance given by:

<!-- formula-not-decoded -->

As equation 26 describes the sum of two independent sub-Gaussian random variables, we can follow similar reasoning to the one applied in equation 27 to show that ∆ n ( x ) is s 2 n ( x ) -sub-Gaussian for some s 2 n ( x ) &gt; 0 . The resulting sub-Gaussian parameter s 2 n ( x ) is then bounded by the sum of the individual sub-Gaussian parameters in equations 27 and 28 as:

<!-- formula-not-decoded -->

which concludes the proof.

## C.5 Convergence guarantees

Now we apply the error bounds above to the general optimization problem in equation 17.

Assumption C.3. The sequence of parameters { ϕ t } ∞ t =1 is a Cauchy sequence, i.e.:

<!-- formula-not-decoded -->

The assumption above can be guaranteed by, e.g., diminishing step sizes during (stochastic) gradient descent. It essentially means that ϕ t will converge to some ˆ ϕ ∈ Φ ⊆ R n ϕ , though not requiring it to be the optimum.

Assumption C.4. The prior p ( m ) has full support over M .

Such assumption ensures that the prior would not wrongly assign zero probability to plausible models.

Lemma C.2. Let assumptions C.1 to C.4 hold, and set β t = β &gt; 0 , for all t ∈ { 0 , 1 , 2 , . . . } . Then the following almost surely holds:

<!-- formula-not-decoded -->

Proof. Consider the following upper bound on the predictive variance of a GP model [51, Lem. D.3]:

<!-- formula-not-decoded -->

where N t ( m,ϕ ) denotes the number of observations collected at ( m,ϕ ) ∈ M × Φ up to time t ≥ 1 . In addition, letting H t denote the σ -algebra generated by the history of all random variables measurable at time t , and setting ˆ ϕ := lim t →∞ ϕ t , the second Borel-Cantelli lemma [12] tells us that: 4

<!-- formula-not-decoded -->

Therefore, for σ 2 t ( m, ˆ ϕ ) → 0 , we need the series above to diverge. To ensure the latter, we can show that the conditional probabilities in Equation (35) have a nonzero lower bound or, if they converge to zero, that they do so slowly enough.

We now derive a lower bound on the sampling probabilities. First, observe that:

<!-- formula-not-decoded -->

4 More precisely, the second Borell-Cantelli lemma shows that the two sides of Equation 33 are proportional to each other, while equality holds if the right-hand side diverges.

where ∥ f ( · , ϕ ) ∥ ∞ = sup m ∈ M | f ( m,ϕ ) | denotes the supremum norm of f ( · , ϕ ) , and we applied Jensen's inequality in the last step. Since the kernel κ is continuous and bounded, the sub-Gaussian parameter σ 2 t ( · , ϕ t ) has a maximum in M , which is finite. As the expected value of the maximum of a finite collection of subGaussian random variables is bounded [see, e.g., 4, Thr. 2.5], it follows that the GP mean µ t is almost surely bounded at all times (by, e.g., Markov's inequality). Considering the model sampling probabilities and that p min := min m ∈ M p ( m ) &gt; 0 by Assumption C.4, we then have that the following almost surely holds:

As, for every m ∈ M , the sequence { σ 2 t ( m,ϕ t ) } ∞ t =0 is non-negative and non-increasing, it has a limit by the monotone convergence theorem. Let σ ∗ := lim t →∞ max m ∈ M σ t ( m,ϕ t ) , and let m ∗ ∈ M be one of the maximizers of lim t →∞ σ t ( · , ϕ t ) . If σ ∗ &gt; 0 , by Equation 35, we have for m ∗ that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies N t ( m ∗ , ˆ ϕ ) → ∞ by Equation 33. However, in that case, we must have σ 2 ∗ = lim t →∞ σ 2 t ( m ∗ , ϕ t ) = 0 by Equation 32, which is a contradiction. Therefore, σ ∗ = 0 , and consequently lim t →∞ σ t ( m,ϕ t ) ≤ σ ∗ = 0 , for all m ∈ M .

Finally, we show that σ 2 t ( · , ϕ t ) ∈ O ( t -1 ) . As we have seen that lim t →∞ σ t ( · , ϕ t ) = 0 above, applying the limit to equation 35, we see that P [ m t = m | H t -1 ] → b m &gt; 0 , for each m ∈ M . Hence, N t ( m,ϕ t ) -1 ∈ O ( t -1 ) , implying that σ 2 t ( · , ϕ t ) is O ( t -1 ) asymptotically by Equation 33, which concludes the proof.

<!-- formula-not-decoded -->

Definition C.1. Let { ξ t } t ∈ N be a real-valued stochastic process. We say that ξ t ∈ O P ( g ( t )) , for a positive function g : N → (0 , ∞ ) , if:

or equivalently that:

Theorem C.3. Under the assumptions in Theorem C.2, we have that the following holds in probability:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Expanding from the definition of the KL divergence and the variational distributions, we have that:

<!-- formula-not-decoded -->

Under assumptions C.1 and C.2, given any β &gt; 0 , applying standard sub-Gaussian concentration results [4] and a union bound, we have that, for all t ≥ 0 :

<!-- formula-not-decoded -->

With probability at least 1 -δ β , it then follows that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all m ∈ M . Hence, with the same probability, it holds that:

<!-- formula-not-decoded -->

By Theorem C.2, we know that σ t ( m,ϕ t ) ∈ O ( t -1 / 2 ) , so that there exists C &gt; 0 such that σ t ( m,ϕ t ) ≤ Ct -1 / 2 , for all m ∈ M . We then have that:

<!-- formula-not-decoded -->

which concludes the proof.

Remark C.4 . The result in Theorem C.3 is similar to Corollary 1 in Oliveira et al. [36], which also derives the concentration bound for the KL divergence between a surrogate-based approximation of a posterior and the true posterior. However, Oliveira et al.'s result only provides an asymptotic convergence rate requires an upper bound on the information gain of the surrogate model of order o ( √ t ) and an appropriately scaled UCB parameter β t , whereas our result shows that we do not need either of these assumptions whenever a sampling lower bound can be guaranteed, i.e., inf t ∈ N ,m ∈ M P [ m t +1 = m | H t ] &gt; 0 . In addition, Oliveira et al. [36] only deals with the static setting where the target posterior does not change over time, while in our case we have a changing ϕ t that leads to different targets per optimization step. This non-stationarity requires additional care with the convergence analysis.

## D Bijective equivalence between discrete distributions

Proposition D.1. Every finite discrete distribution over a finite support M = { m 1 , m 2 , . . . , m k } has a unique representation as a categorical distribution. Specifically, there exists a bijective mapping between the set of all finite discrete distributions on M and the set of categorical distributions parameterized by probability vectors ψ ζ over M .

Proof. Let P denote the set of all finite discrete distributions over M , and let C denote the set of categorical distributions parameterized by θ .

̸

Injectivity: Suppose θ and ϕ are two distinct probability vectors in C . Then, there exists at least one index i such that θ i = ϕ i . Consequently, the corresponding distributions assign different probabilities to m i , implying θ = ϕ .

̸

Surjectivity: For any finite discrete distribution p ∈ P , define θ = p . Since p satisfies θ i ≥ 0 and ∑ k i =1 θ i = 1 , θ is a valid parameterization in C . Thus, every p corresponds to some θ .

Since the mapping is both injective and surjective, it is bijective. Therefore, every finite discrete distribution has a unique categorical distribution representation.

## E Monte Carlo gradients via score function estimation

An alternative to the reparameterization trick is the score function estimator (SFE), which circumvents the issue of non-differentiable samples from discrete distributions by using the log trick to compute the gradients of a function with respect to variational parameters. In the case of the distribution of models, we have the identity

<!-- formula-not-decoded -->

By the Leibniz integral rule, the gradient of the expectation in equation 9 with respect to the parameters of the discrete distribution is

<!-- formula-not-decoded -->

In practice, the variance of this estimator can be very high when the batch size is not large. However, there are techniques to reduce this variance for general applications. The simplest of which is to use a control variate ς in the form

<!-- formula-not-decoded -->

By simply choosing ς = E t ∈{ 1 ,...,T } [ L ( ϕ, ψ )] , where the expectation is estimated online over the iterations of the optimizer, we can reduce variance of ∇ ψ L ( ϕ, ψ ) . See Appendix E.1 for implementation details.

## E.1 Control variate for score function estimator

We adopt the approach used in Kingma and Ba [26] for obtaining an unbiased running first moment of the loss function. At iteration t we draw B samples { m t,n } B n =1 and compute

<!-- formula-not-decoded -->

With fixed decay β ∈ (0 , 1) , update the (biased) first moment exactly as in the approach of the Adam optimizer [26]:

<!-- formula-not-decoded -->

To remove the initialization bias,

<!-- formula-not-decoded -->

Using ς t := µ t as a baseline, the Monte Carlo gradient estimator becomes

<!-- formula-not-decoded -->

Because ς t is independent of each m t,n , the estimator remains unbiased while the baseline substantially reduces its variance.

## E.2 Controlling learning rate via the information gain

When using stochastic gradient descent for optimization over parameters of both q ψ and q ϕ , it is necessary to use careful scaling of the estimated gradients to ensure the optimizer does not 'drop off a cliff' into a local minimum. Such phenomena has been observed in related fields such as proximal policy gradients [46] where the authors demonstrate empirically such a necessity in reinforcement learning problems. In essence, we want to control the learning rate of ψ with respect to the convergence of ϕ → ϕ ∗ . We show empirical results for controlling this rate and leave any mathematical properties for the optimal scaling to future research.

One approach is to control the rate of information gain (IG) of q ψ during the simultaneous optimization over both ψ and ϕ . By assuming a bounded rate of information gain for q ϕ (achieved via gradient clipping) we only

need to consider computing the IG over successive q ( t ) ψ for steps t = 1 , . . . , T . Defining the IG in terms of entropy, we have

<!-- formula-not-decoded -->

When q ψ is a categorical distribution, this quantity is available analytically. However, in general this is not available, but it can be estimated via Monte Carlo integration and importance sampling using available quantities (see Appendix E.3). We choose to set a threshold for the IG between steps, denoted β IG( ψ ) , and then at each successive step t we scale ∇ ψ using an iterative method such as bisection 5 .

## E.3 Monte Carlo estimation of information

The below procedure assumes q ψ represents a distribution over strings of Bernoulli variables. Let ψ ∈ R n ψ parameterize a masked autoencoder that determines logits for a product Bernoulli distribution

<!-- formula-not-decoded -->

with MADE logits NN ( i ) ψ ( · ) and σ (NN ψ ) = (1 + e -NN ψ ) -1 . After an SGD proposal ψ ′ = ψ -α ∇ ψ , we estimate the entropy reduction

<!-- formula-not-decoded -->

To reduce computation at the expense of introducing some estimation bias, we employ importance weights to re-use the current sample of model indicators in an iterative search to scale the gradient step. Draw a mini-batch { m ( n ) } N n =1 i.i.d. ∼ q ψ once ; no re-sampling is needed afterwards. Because the expectation in equation 49 switches from q ψ to q ψ ′ , rewrite

<!-- formula-not-decoded -->

For a Bernoulli product the weight factorizes:

<!-- formula-not-decoded -->

implemented stably via σ (NN ( i ) ψ ′ ) σ (NN ( i ) ψ ) = exp [ log(1 + e -NN ( i ) ψ ) -log(1 + e -NN ( i ) ψ ′ ) ] .

The mini-batch estimator is therefore

<!-- formula-not-decoded -->

Given a tolerance ε &gt; 0 , reduce α ← 0 . 5 α until

<!-- formula-not-decoded -->

If no α &gt; 10 -20 satisfies equation 53, discard the update by setting the gradient to 0 . Otherwise, apply the accepted scaled gradient.

5 In preliminary investigations, other approaches for implementation of this threshold such as constrained optimization and computation of Lagrange multipliers were trialed without success, possibly due to the geometry of the optimization landscape.

## F Robust variable selection example details and additional results

The likelihood is

<!-- formula-not-decoded -->

with priors p ( γ ) = 2 -p and p ( β ) = N (0 , σ 2 β I ) . Each of the parameters in the likelihood are described in Table 2 under the Misspecification:None column. The data generating setup in Table 2 describes three levels of misspecification to induce poor identifiability and thus a posterior that is challenging to fit using simple variational density families, such as mean field inference. This exemplifies the use of normalizing flows for this experiment. While many parameters are shared, some differ strongly between the likelihood and DGP. In particular, notice the difference in σ 1 , σ 2 . Also, for the highly misspecified DGP, correlation between included covariates i and excluded covariates j is induced by a factor of ρ i,j = 0 . 1 for a proportion of j , making the recovery of the DGP using any inference method a challenging and improbable task. For every data set, β will be either β 1 or β 2 with probability 0 . 5 .

Table 2: Data generating setup

<!-- image -->

| Parameter                                                         | Misspecification to likelihood   | Misspecification to likelihood   | Misspecification to likelihood   |
|-------------------------------------------------------------------|----------------------------------|----------------------------------|----------------------------------|
|                                                                   | None                             | Mid                              | High                             |
| Number of data points &#124; x &#124;                             | 50                               | 50                               | 50                               |
| Dimension of β                                                    | 8                                | 8                                | 8                                |
| Dimension of γ                                                    | 7                                | 7                                | 7                                |
| &#124;M&#124;                                                     | 2 7 = 128                        | 2 7 = 128                        | 2 7 = 128                        |
| Probability of inclusion P ( γ i = 1)                             | 0 . 4                            | 0 . 4                            | 0 . 4                            |
| Non-outlier σ 1                                                   | 1                                | 2                                | 4                                |
| Outlier σ 2                                                       | 10                               | 5                                | 4                                |
| Probability of correlation P ( ρ i,j > 0 &#124; γ i = 1 ,γ j = 0) | 0                                | 0                                | 0.4                              |
| Total correlation factor ∑ j ρ i,j                                | 0                                | 0                                | 0.1                              |
| β 1                                                               | 0.5                              | 0.5                              | 0.5                              |
| β 2                                                               | 0.5                              | 1.5                              | 1.5                              |
| Outlier probability α                                             | 0.1                              | 0.1                              | 0.1                              |

Lastly, during the inference process, we consider two separate experiments for each DGP: a 'focused-prior' experiment where σ β = 1 . 5 , and a 'wide-prior' experiment where σ β = 10 . These two scenarios cause a significant difference between the inferred reversible jump MCMC model probabilities and the inferred VTI model probabilities, as can be seen in the subsequent figures.

VTI inference was conducted on a cluster of GPU nodes with mixed Nvidia RTX3090 and H100 cards. On the former we used float32 precision for MLP architectures, the latter used float64.

## F.1 Focused versus wide priors

Each of Figures 5-10 is a replicate of Figure 2 in the main text, showing a sweep of 10 randomly generated data sets (indicated by different colours) according to the corresponding setup in Table 2 using three different variational families: diagonal Gaussian MLP (a CoSMIC mean-field variational family), a composition of 5 affine masked autoregressive flows each with 5 hidden blocks, and a composition of 4 rational quadratic spline masked autoregressive flows each with 6 hidden blocks. The expressiveness of each variational family increases from left to right in each figure.

In the σ β = 1 . 5 focused prior setting (Figures 5, 7, 9) performance is generally good, as per Figure 2 in the main text: (i) the model probability estimates (top row) tend to move closer to the y = x line as the expressiveness of the variational family increases (left to right plots); (ii) the slight S-shape of the model probability estimates around the y = x line is easily interpretable as the the variational objective L ( ψ, ϕ ) (equation 9) will naturally favour models with higher posterior model probability over those with lower probabilities; (iii) the true data generating process models (triangles) are generally given high posterior model probabilities; and (iv) individual model posteriors are better estimated for higher probability models (negative slope on the bottom rows).

For the σ β = 10 wide prior setting (Figures 6, 8, 10) performance at first glance appears much worse, particularly in terms of estimating model probabilities. However, on closer inspection this is not the case. It is well known

(e.g. [18]) that the marginal likelihood (a.k.a. model evidence; a component of the posterior model probability) can be highly sensitive to diffuse priors. In such cases (as with σ β = 10 ) the posterior will tend to unreasonably favour those models with fewer parameters, and particularly (in the case of regression models) the null model with no predictors, even in the presence of a very clear relationship between predictors and response. This effect can be clearly seen in Figures 6, 8, 10 (top row), where the null model (indicated by a circle) is given far higher posterior model probability on the π ( m ) axis than the actual data generating process (triangles). In contrast, the true data generating process (triangles) is generally given a high posterior model probability (comparable with the focused prior setting in Figures 5, 7, 9) under the VTI approximation. From these results we conclude that: (i) the posterior model probabilities that depend on the marginal likelihood (i.e., the estimates of π ( m ) on the x -axis) have been affected by the wide prior to unreasonably favour models with less parameters; (ii) the VTI-based posterior model probability estimates suggest that they are less sensitive to the undesirable effects of this prior; and (iii) in combination the resulting plots in Figures 6, 8, 10 (top row) only appear to indicate worse performance of VTI compared to the gold standard than is actually the case.

## F.2 Within model comparison

Figure 11 illustrates a typical comparison between the reversible jump MCMC estimated posterior distribution and the VTI approximation. The figure shows the posterior of the data generating process model from the first high misspecification dataset in Figure 2 (main text). While there are some small differences, the main features of the posterior appear to be well captured.

0

Figure 5: As Figure 2 (main text), but under: no misspecification ( σ 1 = 1 , σ 2 = 10 ), focused prior ( σ β = 1 . 5 ) . Circles indicate the null model (constant only, no predictors); triangles indicate the data generating process.

<!-- image -->

Figure 6: As Figure 2 (main text), but under: no misspecification ( σ 1 = 1 , σ 2 = 10 ), wide prior ( σ β = 10 ) . Circles indicate the null model (constant only, no predictors); triangles indicate the data generating process.

<!-- image -->

Figure 7: As Figure 2 (main text), but under: mid misspecification ( σ 1 = 2 , σ 2 = 5 ), focused prior ( σ β = 1 . 5 ) . Circles indicate the null model (constant only, no predictors); triangles indicate the data generating process.

<!-- image -->

Figure 8: As Figure 2 (main text), but under: mid misspecification ( σ 1 = 2 , σ 2 = 5 ), wide prior ( σ β = 10 ) . Circles indicate the null model (constant only, no predictors); triangles indicate the data generating process.

<!-- image -->

Figure 9: As Figure 2 (main text), but under: high misspecification ( σ 1 = 4 , σ 2 = 4 ), focused prior ( σ β = 1 . 5 ) . Circles indicate the null model (constant only, no predictors); triangles indicate the data generating process.

<!-- image -->

Figure 10: As Figure 2 (main text), but under: high misspecification ( σ 1 = 4 , σ 2 = 4 ), wide prior ( σ β = 10 ) . Circles indicate the null model (constant only, no predictors); triangles indicate the data generating process.

<!-- image -->

RJMCMC posterior misspecified robust variable selection

Variational posterior misspecified robust variable selection

Figure 11: Multivariate plot comparison between reversible jump MCMC (left) and VTI (right) using spline flow composition of four layers and six blocks on the first synthetic narrow-prior highmisspecification data set from the Figure 2 (main text) example.

<!-- image -->

## F.3 Baseline reversible jump MCMC for robust variable selection

Consider the linear model y = Xβ + ε with ε ∼ N (0 , σ 2 I ) . We introduce a binary mask m ∈ { 0 , 1 } p to indicate active coefficients in β ∈ R p . The reversible jump MCMC algorithm explores the model space by proposing bit-flips in m , corresponding to adding (birth) or removing (death) predictors.

Jacobian Determinant : For bit-flipping moves in a saturated space where the dimensionality remains constant ( dim( m ′ ) = dim( m ) ), the transformation is bijective with a Jacobian determinant of 1:

<!-- formula-not-decoded -->

Thus, the Jacobian does not affect the acceptance probability.

Birth Move ( m,β ) → ( m ′ , β ) : A birth move flips a bit in m → m ′ from 0 to 1. Given the birth/death move ratio the acceptance probability is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Death Move ( m,β ) → ( m ′ , β ) : A death move flips a bit in m → m ′ from 1 to 0. Using the same birth/death move ratio, the acceptance probability is:

<!-- formula-not-decoded -->

Within-Model Gaussian Proposal β → β ′ : Within a fixed model m , propose a new β ′ using a symmetric random-walk:

Since the proposal is symmetric, the proposal densities cancel out in the acceptance probability.

<!-- formula-not-decoded -->

## G Example description: Bayesian inference of multi-layer-perceptron directed acyclic graph discovery

## Notation:

$$N d number of nodes in graph n number of data samples X ∈ R n × N d (rows are i.i.d. samples) P ∈ P N d permutation matrix (node order) U ∈ { 0 , 1 } N d × N d strictly upper-triangular edge mask A = P ⊤ UP adjacency in canonical order (code default) pa A ( j ) = { i < j : U ij = 1 } parents of node j in the sorted order.$$

Node-wise conditional mean: Fix hidden width H and a model indicator m = ( P , U ) . For each non-root node j = 2 , . . . , N d define parameters

<!-- formula-not-decoded -->

with W (1) j ∈ R H × ( j -1) , b (1) j ∈ R H , W (2) j ∈ R 1 × H , b (2) j ∈ R . Let u j := U 1:( j -1) , j be the ( j -1) -vector of active parents. Writing X 1: j -1 to denote the 1 , . . . , j -1 columns of X ,

<!-- formula-not-decoded -->

Gaussian likelihood: Let ϖ be the permutation associated with P (so X ϖ ( j ) is column j after sorting). With homoscedastic noise σ 2 ,

<!-- formula-not-decoded -->

Parameter prior (masked i.i.d. Gaussian): Let C ( m ) ⊆ { 1 , . . . , dim θ } be the index set that survives the mask. Then

<!-- formula-not-decoded -->

(parameters outside C ( m ) are handled by a reference density).

## Structural prior:

<!-- formula-not-decoded -->

with P a permutation matrix and U strictly upper triangular.

The un-normalised log-posterior is the sum of the three boxed terms above.

## G.1 Data generating process

The data generating procedure generally follows the simulation design in Thompson et al. [52].

## Global hyper-parameters:

N d : number of nodes , H : hidden width , σ 2 : noise variance , ρ Edge ∈ (0 , 1) : edge probability , σ 0 &gt; 0 : parameter prior scale .

Sample graph structure:

<!-- formula-not-decoded -->

Sample node parameters: Let the bias flag β ∈ { 0 , 1 } ( β = 1 keeps both bias vectors, β = 0 sets them to 0 ). For each non-root node j = 2 , . . . , N d draw independently

<!-- formula-not-decoded -->

while the root node has θ (1) = ∅ . Note the active parameters are drawn uniformly from a non-zero range rather than from the prior.

Context-to-mask map: For m = ( P , U ) , C ( m ) = C ( U ) ⊆ { 1 , . . . , dim θ } keeps exactly the coordinates satisfying the conditions:

1. Column i of W (1) j is active iff U ij = 1 ;
2. If ∑ i&lt;j U ij = 0 then all parameters in θ ( j ) are masked.

(The permutation P has no effect on the mask.)

Data generation (topological order): Let ϖ be the permutation induced by P . For each sample s = 1 , . . . , n generate sequentially

<!-- formula-not-decoded -->

where ε js iid ∼ N (0 , 1) and

<!-- formula-not-decoded -->

Collecting the n draws gives

<!-- formula-not-decoded -->

## G.2 Comparison metrics

Given knowledge of a 'true' adjacency matrix A , each experiment uses four scores for comparison with the estimated posterior: F1, structured Hamming distance (SHD), Brier score, and area under the receiver operating characteristic curve (AUROC). This follows the experiment setup in Thompson et al. [52].

## G.3 Common inference setup

For each data set in both the simulation study and real data example, VTI is run a total of 10 replicates using different random seeds, and the posterior is selected where the terminal loss is minimized. For DAGMA, the sparsity hyperparameter is swept from λ min = 10 -3 to λ max = 1 over 10 logarithmically spaced values. For the autoregressive flow, we use Affine(5,5) (see Appendix A.2) with a context encoder designed as follows:

<!-- formula-not-decoded -->

where σ ⌈× 2 ⌉ ( x ) := Wx + b broadcasts from | x | to the first power of 2 greater than or equal to 2 | x | . The final dimension of δ ( P , U ) is 4096.

<!-- formula-not-decoded -->

## G.4 Simulation design

In the simulation study, the configuration of the MLP is as follows. We set the hidden layer width to H = 10 . We set the number of nodes to N d = 10 . We omit the bias parameters b (1) j , b (2) j for all edges, i.e. set β = 0 . The edge inclusion probability is set to ρ Edge = 0 . 5 . For VTI, the model prior p ( m ) is uniform (i.e. the sparsity parameter is set to λ = 0 ).

We generate 10 i.i.d. complete data sets of length n max = 2 10 from the above process. The experiment compares data size against the metrics from Appendix G.2. The range of data sizes are n = 16 , 32 , 4 , 128 , 256 , 512 , 1024 , where n &lt; n max simply takes the first n samples.

VTI inference was conducted on a cluster of GPU nodes with mixed Nvidia RTX3090 and H100 cards. On the former we used float32 precision for MLP architectures, the latter used float64.

In the DAGMA setup, a sweep of the regularization tuning parameter λ was conducted for each dataset. The resulting adjacency matrix with the closest number of active edges to the data-generating graph was selected. This resulted in a higher-than-usual score for DAGMA results in the simulation study when compared to other methods. For DiBS/DiBS+, the inference ran for 5 , 000 steps over 10 'particles' (each an individual Stein variational gradient descent optimization, see [32]). JSP-GFN was configured to use a batch size of 1024 over 50 , 000 iterations.

## G.5 Real data example

For VTI, we chose to use a penalized structural model prior p ( m ) that induces 'extra' sparsity via further down-weighting the probability of graphs with more edges in order to reach an acceptable level of closeness to the 'consensus' graph in Sachs et al. [44]. It should be noted that in no other experiment do we use sparsity-inducing priors. We set λ = 200 and set the number of hidden nodes per edge to H = 5 and include the bias terms, i.e. β = 1 .

For DAGMA non-linear, DiBS/DiBS+, and JSP-GFN, we use 10 hidden nodes per edge and no bias term.

## G.6 DAG Model indicator construction: Lehmer Code Decoding

A permutation of the ordered set { 1 , 2 , . . . , N d } is represented by a Lehmer code c = ( c 1 , c 2 , . . . , c N d ) , where c i ∈ { 0 , 1 , . . . , N d -i } . At step i (1 ≤ i ≤ N d ) we choose the ( c i +1) -th unused index in the remaining ascending list.

Example. For N d = 5 and c = (2 , 1 , 0 , 0 , 0)

```
c 1 = 2 : { 1 , 2 , 3 , 4 , 5 }→ 3 , c 2 = 1 : { 1 , 2 , 4 , 5 }→ 2 , c 3 = 0 : { 1 , 4 , 5 }→ 1 , c 4 = 0 : { 4 , 5 }→ 4 , c 5 = 0 : { 5 }→ 5 .
```

Permutation-matrix representation. The permutation ϖ is stored as a one-hot P ∈ { 0 , 1 } N d × N d with P r,i = 1 iff row r is chosen at column i .

Algorithm 2 decodes each column in parallel. For column i the code k ∈ [0 , N d -i ] specifies 'pick the ( k +1) -th leftover row.' The Boolean mask marks currently unused rows; broadcasting the flattened one-hot vector onto the corresponding ( batch , row ) pairs writes the unit entries. Column N d is filled by the single row that remains unassigned. This implementation gives a compact ( B,N d ) tensor, expanded by the decoder to ( B,N d , N d ) for efficient batched linear algebra in our DAG-inference pipeline.

## Algorithm 2 Vectorized Lehmer decode via leftover mask

Require: P code ∈ N B × N d {batch of Lehmer codes} Ensure: P ∈ { 0 , 1 } B × N d × N d

- 2: P ← 0 bs × N d × N d
- 1: bs ← B

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 4: k ← P code [: , i ]

<!-- formula-not-decoded -->

- 8: Idx ← nonzero( Mask )
- 7: Mask ← ( Used = 0)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 10: end for
- 12: Last ← nonzero( Used = 0)
- 14: return P

<!-- formula-not-decoded -->

## G.7 Model identifier for directed acyclic graphs

We encode a permutation matrix P ∈ { 0 , 1 } N d × N d using a compressed Lehmer code consisting of N d -1 categorical variables { ρ cat 1 , . . . , ρ cat N d -1 } . Here ρ cat i has N d -i +1 outcomes.

Concretely, ρ cat 1 ∈ { 0 , 1 , . . . , N d -1 } , ρ cat 2 ∈ { 0 , 1 , . . . , N d -2 } , . . . , ρ cat N d -1 ∈ { 0 , 1 } . Once the first N d -1 columns are fixed, the last column is forced.

Each ρ cat i = k is mapped to a one-hot vector of length N d . The value k selects the ( k +1) -st available row for the i -th column; previously taken rows remain zero, preserving the permutation property.

Given P we form an upper-triangular mask U ∈ { 0 , 1 } N d × N d with zero diagonal. Each entry above the diagonal ( i &lt; j ) is a Bernoulli variable, so U flattens to N d ( N d -1) 2 bits. The adjacency matrix is A = P ⊤ UP , giving a DAG.

We concatenate the N d -1 categorical codes with the N d ( N d -1) 2 Bernoulli bits, yielding a vector z of length ( N d -1) + N d ( N d -1) 2 . MADE + consumes z together with a multiplier \_ fn specifying the parameter count for each entry.

Let z j denote the j -th component of z :

<!-- formula-not-decoded -->

The architecture yields the autoregressive factorization

<!-- formula-not-decoded -->

The identifier { ρ cat 1 , . . . , ρ cat N d -1 , U binary } is modelled autoregressively by a single MADE + network, yielding A = P ⊤ UP upon sampling.

We employ a structural prior over the space of models with the edge-penalty term γ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that when γ = 0 , the prior is uniform.

## G.8 Neural probability mass function for model indicators over large spaces: MADE +

To represent a distribution over binary strings, we use the Masked Autoencoder for Density Estimation (MADE) [19] implementation found in the Durkan et al. [14] repository. To represent a more complex discrete distribution such as that required by the P , U representation of a directed acyclic graph, we apply a simple extension to this architecture to allow us to vary the output dimension multiplier. For presentational clarity we call this extension MADE + . The key change in MADE + is the introduction of a per-dimension output multiplier function r ( i ) that determines how many parameters are emitted for the i -th input dimension in the autoregressive factorization.

In the original MADE, all features share a common multiplier k , yielding an output dimensionality of k × d when there are d input features. Mathematically, if x ∈ R d , the network outputs ( h 1 , h 2 , . . . , h kd ) ∈ R kd .

In MADE + , a function r : { 0 , 1 , . . . , d -1 } → N is provided, and the final output dimension is ∑ d -1 i =0 r ( i ) . For each input dimension x i , the network outputs r ( i ) parameters. Concretely, where d is the number of input features, the final output dimension becomes total \_ out \_ features = ∑ d -1 i =0 r ( i ) . In other words, each input x i can be associated with a custom number of distributional parameters (e.g., to handle discrete variables of different cardinalities). The masking logic is preserved by replicating each degree, deg( x i ) , exactly r ( i ) times in the final layer.

Below is a simplified, side-by-side pseudocode comparing MADE (left) and MADE + (right). Changes in MADE + are highlighted in green.

| Algorithm 3 Original MADE (Final Layer Construction)                                                                                                                                       | Algorithm 4 MADE + (Final Layer Construction)                                                                                                                                                                                         |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| out_features = features * output_multiplier final_layer = MaskedLinear( in_degrees = prev_out_degrees, out_features = out_features, autoregressive_features = features, is_output = True ) | total_out_features = ∑ features - 1 i =0 r ( i ) final_layer = MaskedLinear( in_degrees = prev_out_degrees, out_features = total_out_features, autoregressive_features = features, is_output = True, output_multiplier_fn = r ( i ) ) |

By allowing each input dimension X i to have its own output multiplier r ( i ) , the MADE + architecture provides a more flexible autoregressive decomposition:

<!-- formula-not-decoded -->

where now the conditional distribution for x i can be parameterized by r ( i ) parameters (e.g., logits for a categorical variable of size r ( i ) , or a mean/variance pair, etc.).

Hence, one can naturally combine discrete variables of varying dimensions such as Bernoulli and categorical variables. For example, if x 1 is categorical with 10 categories and x 2 is a Bernoulli variable, one can specify r (0) = 10 and r (1) = 1 , so that the overall conditional densities (or probability mass functions) multiply to form a richer joint model adapting precisely to each variable's nature.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Claims made in the abstract and introduction/contributions are met in Sections 2, 3, and demonstrated in numerical experiments in Section 5.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: While there is no explicitly-titled Limitations section, (i) the effects of using normalising flows with poorer/stronger degrees of expressivity on the quality of the VTI approximation are explicitly discussed in Section 5 (Experiments) and Appendix F (Additional Results); the nature of the VTI approximation over transdimensional model space tending to focus more on high posterior model probability models is discussed in Section 6 (Discussion); and the limitations of particular methods of approximating q ψ ( m ) , depending on the cardinality of the model space |M| are discussed in Sections 3.2, 3.3 and 6 (Discussion).

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The methods presented in Section 2 and Section 3 are formally analyzed in Section A and Section C respectively. The authors are unaware of any unaddressed assumptions.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Every effort was made to disclose all of the information needed to reproduce the experimental results. Section F and Section G contain details of the setup of experiments that are not included in Section 5.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Code will be provided in the supplementary materials.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: Almost all details are provided in the main paper or supplementary material appendices. Any smaller details not provided in these sources can be determined from the supplementary code itself. These include the hyperparameters of the optimizer, learning rate schedulers, and so on.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The experiment design runs multiple data sets and optimizer seeds. Figures 4 and 3 show error bars. No statistical tests are performed.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Compute time and resources were not considered to be the main focus of the paper's contribution (which is the development of the first approach to extend variational inference to the transdimensional setting), however Section F and Section G.7 mention the GPU architectures used for each experiment.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: To the best knowledge of the authors, the paper conforms, in every respect, with the NeurIPS Code of Ethics.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: To the best knowledge of the authors, the paper describes a novel methodology that has no direct societal impact.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The experiments use only freely-available data sources.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All data sources are referenced and the licence and terms are properly respected. Only open source libraries were used in the code.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: No new assets are released apart from the eventual release of the code used for experiments with the appropriate CC license.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No crowdsourcing experiments were conducted.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No crowdsourcing experiments were conducted.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: LLM technology was not used for any purpose other than grammar and minor textual edits.