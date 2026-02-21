## Multilevel neural simulation-based inference

## Yuga Hikida

Aalto University yuga.hikida@aalto.fi

Niall Jeffrey University College London n.jeffrey@ucl.ac.uk

## Ayush Bharti

Aalto University ayush.bharti@aalto.fi

François-Xavier Briol University College London f.briol@ucl.ac.uk

## Abstract

Neural simulation-based inference (SBI) is a popular set of methods for Bayesian inference when models are only available in the form of a simulator. These methods are widely used in the sciences and engineering, where writing down a likelihood can be significantly more challenging than constructing a simulator. However, the performance of neural SBI can suffer when simulators are computationally expensive, thereby limiting the number of simulations that can be performed. In this paper, we propose a novel approach to neural SBI which leverages multilevel Monte Carlo techniques for settings where several simulators of varying cost and fidelity are available. We demonstrate through both theoretical analysis and extensive experiments that our method can significantly enhance the accuracy of SBI methods given a fixed computational budget.

## 1 Introduction

Simulation-based inference (SBI) [Cranmer et al., 2020] is a set of methods used to estimate parameters of complex models for which the likelihood is intractable but simulating data is possible. It is particularly useful in fields such as cosmology [Jeffrey et al., 2021], epidemiology [Kypraios et al., 2017], ecology [Beaumont, 2010], synthetic biology [Lintusaari et al., 2017], and telecommunications engineering [Bharti et al., 2022a], where models describe intricate physical or biological processes such as galaxy formation, spread of diseases, the interaction of cells, or propagation of radio signals.

For a long time, SBI was dominated by methods such as approximate Bayesian computation (ABC) [Beaumont et al., 2002, Beaumont, 2019], which compared summary statistics of simulations and of the observed data. However, SBI methods using neural networks to approximate likelihoods [Papamakarios et al., 2019, Lueckmann et al., 2019, Boelts et al., 2022], likelihood ratios [Thomas et al., 2022, Durkan et al., 2020, Hermans et al., 2020], or posterior distributions [Papamakarios and Murray, 2016, Lueckmann et al., 2017, Greenberg et al., 2019, Radev et al., 2022] are now quickly becoming the preferred approach. These neural SBI methods are often favoured because they allow for amortisation [Zammit-Mangion et al., 2024], meaning that they require a large offline cost to train the neural network, but once the network is trained, the method can rapidly infer parameters for new observations or different priors without requiring additional costly simulations. This is particularly useful when the simulator is computationally expensive, as it reduces the need to repeatedly run simulations for each new inference task, making the overall process significantly less costly.

Nevertheless, the initial training phase of neural SBI methods typically still requires a large number of simulations, preventing their application on computationally expensive (and often more realistic) models which can take hours of compute time for simulating a single datapoint. Examples include most tsunami [Behrens and Dias, 2015], wind farm [Kirby et al., 2023], nuclear fusion [Hoppe et al., 2021] and cosmology [Jeffrey et al., 2025] simulators.

One avenue to mitigate this issue is multi-fidelity methods [Peherstorfer et al., 2018]: we often have access to a sequence of simulators with increasing computational cost and accuracy which we may be able to use to refine existing methods based on a single simulator.

Figure 1: Low- and high-fidelity cosmological simulations from the CAMELS data [VillaescusaNavarro et al., 2023] studied in Section 5.4.

<!-- image -->

This setting is quite common. One example is simulators requiring the numerical solution of ordinary, partial, or stochastic differential equations, where the choice of mesh size or stepsize affects both the accuracy of the solution and the computational cost. Another example arises when modelling complex physical, chemical, or biological processes, where low-fidelity simulators can be obtained by neglecting certain aspects of the system. This is exemplified in Figure 1, where a high-fidelity cosmological simulation includes baryonic astrophysics and thus appears smoother than the low-fidelity simulation.

The key idea behind multi-fidelity methods is to leverage cheaper, less accurate simulations to supplement the more expensive, high-fidelity simulations, ultimately improving efficiency without sacrificing accuracy. This has been popular in the emulation literature since the seminal work of Kennedy and O'Hagan [2000], but applications to SBI are much more recent and include Jasra et al. [2019], Warne et al. [2018], Prescott and Baker [2020, 2021], Warne et al. [2022], Prescott et al. [2024], who proposed multi-fidelity versions of ABC. More recently, [Krouglova et al., 2025] also proposed a multi-fidelity method to enhance neural SBI approximations of the posterior based on transfer learning. However, their method is not supported by theoretical guarantees.

In this paper, we propose a novel multi-fidelity method which is broadly applicable to neural SBI methods. Taking neural likelihood and neural posterior estimation as the main case studies, we show that our approach is able to significantly reduce the computational cost of the initial training through the use of multilevel Monte Carlo [Giles, 2015, Jasra et al., 2020] estimates of the training objective. The approach has strong theoretical guarantees; our main result (Theorem 1) directly links the reduction in computational cost to the accuracy of the low-fidelity simulators, and we demonstrate (in Theorem 2) how to best balance the number of simulations at each fidelity level in the process. Our extensive experiments on models from finance, synthetic biology, and cosmology also demonstrate the significant computational advantages provided by our method.

## 2 Background

We first recall the basics of SBI methods and the related works on reducing computational cost in Section 2.1, then provide a brief introduction to multilevel Monte Carlo in Section 2.2.

## 2.1 Simulation-based inference

Let { P θ } θ ∈ Θ be a parametric family of distributions on some space X ⊆ R d X parameterised by θ ∈ Θ ⊆ R d Θ . We assume that this model is available in the form of a computer code, i.e. as a simulator , where simulating from P θ is straightforward, but the likelihood function p ( · | θ ) associated with P θ is intractable. Simulators can be characterised by a pair ( U , G θ ) , where U is a distribution (typically simple, such as a uniform or a Gaussian) on a space U ⊆ R d U which captures all of the randomness, and G θ : U ↦→ X is a (deterministic) parametric map called the generator . Simulating x ∼ P θ can be achieved by first simulating u ∼ U , and then applying the generator x = G θ ( u ) . In this paper, we consider Bayesian inference for the parameters θ of this simulator-based model given independent and identically distributed (iid) data x o 1: m = { x o j } j m =1 ∈ X m collected from some data-generating process. Specifically, we are interested in approximating the posterior with density π ( θ | x o 1: m ) ∝ ∏ m j =1 p ( x o j | θ ) π ( θ ) , where π ( θ ) is the prior density. As introduced below, this can be achieved via a neural SBI method which approximates the likelihood or posterior.

Neural likelihood estimation (NLE). NLEs [Papamakarios et al., 2019, Lueckmann et al., 2019, Boelts et al., 2022, Radev et al., 2023a] are extensions of the synthetic likelihood approach [Wood, 2010, Price et al., 2018] that use flexible conditional density estimators, typically normalising flows

[Rezende and Mohamed, 2015, Papamakarios et al., 2021], as surrogate models for the likelihood function associated with P θ . The surrogate conditional density q NLE ϕ : X × Θ → [0 , ∞ ) where q NLE ϕ ( · | θ ) is a density function for each θ ∈ Θ and ϕ ∈ Φ ⊆ R d Φ denotes its learnable parameters, is trained by minimising the negative log-likelihood with respect to ϕ on simulated samples. More precisely, let { ( θ i , x i ) } n i =1 be the training data such that θ i ∼ π are realisations from the prior and x i ∼ P θ i are realisations from the simulator. Then ˆ ϕ MC := arg min ϕ ∈ Φ ℓ NLE MC ( ϕ ) , where ℓ NLE MC is an empirical (Monte Carlo) estimate of negative expected log-density:

<!-- formula-not-decoded -->

Once the surrogate likelihood is trained, Markov chain Monte Carlo (MCMC) or variational inference methods are used to sample from the (approximate) posterior distribution π NLE ( θ | x o 1: m ) ∝ ∏ m j =1 q NLE ˆ ϕ MC ( x o j | θ ) π ( θ ) . NLEs can therefore be regarded as being partially amortised-the surrogate likelihood need not be trained for a new observed dataset, however, MCMC needs to be carried out again to obtain the new posterior.

For a computationally costly simulator, we note that obtaining training samples can become a bottleneck, which affects the accuracy of estimating the expected loss. Thus, estimating the loss accurately with fewer samples is key to handling costly simulators.

Neural posterior estimation (NPE). Instead of learning a surrogate likelihood, NPEs learn a mapping x ↦→ p ( θ | x ) from the data to the posterior using conditional density estimators. These are often based on mixture density networks [Bishop, 1994, Papamakarios and Murray, 2016] or normalising flows [Dinh et al., 2014, Papamakarios et al., 2017, Radev et al., 2022]. Similar to NLE, the conditional density q NPE ϕ : Θ ×X m → [0 , ∞ ) is trained by minimising the negative log likelihood with respect to ϕ using data { ( θ i , x 1: m,i ) } n i =1 generated by first sampling from the prior θ i ∼ π and then the simulator x 1: m,i = ( x 1 ,i , . . . , x m,i ) ∼ P θ i :

<!-- formula-not-decoded -->

Once ϕ is estimated, the NPE posterior is obtained as π NPE ( θ | x o 1: m ) = q NPE ˆ ϕ MC ( θ | x o 1: m ) . Although training q NPE ϕ incurs an upfront cost, this is a one-time cost as approximate posteriors for new observed datasets are obtained by a simple forward pass of x o 1: m through the trained networks, making NPEs fully amortised (in contrast with the partial amortisation of NLE). Similarly to NLE, the computationally costly step in NPE is the generation of training samples from running expensive simulators. Note that both q NLE ϕ and q NPE ϕ usually include a summary function (often architecturally implicit in NPE). This is helpful when X is high-dimensional or the number of observations m is large [Alsing et al., 2018, Radev et al., 2022], and for NPE it allows conditioning on datasets of different sizes. Recently, alternative training objectives for NPE based on flow matching [Wildberger et al., 2023], diffusion [Geffner et al., 2023, Sharrock et al., 2024, Gloeckler et al., 2024], and consistency models [Schmitt et al., 2024b] have been proposed.

Related work. We briefly note that beyond the aforementioned multi-fidelity methods, other works also aim to reduce the computational cost of SBI. In the context of ABC, adaptive sampling of the posterior using either sequential Monte Carlo techniques [Sisson et al., 2007, Beaumont et al., 2009, D. Moral et al., 2011] or Gaussian process surrogates [Gutmann and Corander, 2016, Meeds and Welling, 2014] has been a popular approach. A similar approach has been applied to neural SBI [Papamakarios and Murray, 2016, Lueckmann et al., 2017, Greenberg et al., 2019, Papamakarios et al., 2019, Hermans et al., 2020, Durkan et al., 2020], where sequential training schemes are employed to reduce the number of calls to the simulator. Other works have tackled this problem using cost-aware sampling [Bharti et al., 2025], side-stepping high-dimensional estimation [Jeffrey and Wandelt, 2020], early stopping of simulations [Prangle, 2016], dependent simulations [Niu et al., 2023, Bharti et al., 2023], expert-in-the-loop methods [Bharti et al., 2022b], self-consistency properties [Schmitt et al., 2024a], parallelisation of computations [Kulkarni and Moritz, 2023], and the Markovian structure of certain simulators [Gloeckler et al., 2025]. Our proposed method, introduced in Section 3, can be combined with all of these compute-efficient SBI methods and is hence complementary to them.

## 2.2 Multilevel Monte Carlo method

Consider some square-integrable function f : Z → R and distribution µ on a domain Z ⊆ R d Z . We consider the task of estimating E z ∼ µ [ f ( z )] . A first approach is standard Monte Carlo (MC) [Robert and Casella, 2000, Owen, 2013], which yields the following estimator: 1 / n ∑ n i =1 f ( z i ) , where z 1 , . . . , z n ∼ µ . The root-mean-squared error (RMSE) of this estimator converges at a rate O ( n -1 / 2 ) , where the rate constant is controlled by Var [ f ( z )] [Owen, 2013, Ch. 2]. In cases where f is expensive to evaluate or µ is expensive to sample from, the RMSE can therefore be relatively large.

This issue can be mitigated by multilevel Monte Carlo (MLMC) , which was first proposed by Heinrich [2001], Giles [2008], and more recently reviewed in Giles [2015], Jasra et al. [2020]. Suppose we have a sequence of square-integrable functions f l : Z → R for l ∈ { 0 , 1 , . . . , L } which are approximations of f and which are ordered such that f L = f , and both the cost of evaluation C l and the accuracy (or fidelity ) of f l increase with l . In that case, MLMC consists of expressing E z ∼ µ [ f ( z )] through a telescoping sum and approximating each term through MC based on samples z l i ∼ µ for i , . . . , n and l , , . . . , L :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that this can also be thought of as using the low-fidelity functions as approximate control variates. By carefully balancing the number of samples n 0 , . . . , n L according to the costs C 0 , . . . , C L and variances Var [ f 0 ( z )] , Var [ f 1 ( z ) -f 0 ( z )] , . . . , Var [ f L ( z ) -f L -1 ( z )] at each level, one can show that MLMC can significantly improve on the accuracy of MC given a fixed computational budget. For this reason, MLMC has found numerous applications in statistics and machine learning, including for optimisation [Asi et al., 2021, Hu et al., 2023, Yang et al., 2024], sampling [Dodwell et al., 2019, Jasra et al., 2020], variational inference [Fujisawa and Sato, 2021, Shi and Cornish, 2021], probabilistic numerics [Li et al., 2023, Chen et al., 2025] and the design of experiments [Goda et al., 2020, 2022].

## 3 Methodology

We now present NLE and NPE versions of our approach, termed multilevel-NLE and multilevel-NPE .

MLMCfor NLE and NPE. Recall that we are performing inference for a simulator ( G θ , U ) with a prior π on the parameter θ ∈ Θ . We reparameterise the NLE and the NPE objective in terms of u instead of x (akin to the reparametrisation trick in variational inference), and express them more broadly using an arbitrary loss ℓ : Φ → R (to represent either ℓ NLE or ℓ NPE ) and an arbitrary function f ϕ : U m × Θ → R (to represent either f NLE ϕ or f NPE ϕ ) as:

<!-- formula-not-decoded -->

where for NLE we have f NLE ϕ ( u, θ ) := -log q NLE ϕ ( G θ ( u ) | θ ) = -log q NLE ϕ ( x | θ ) with m = 1 , and for NPE we have f NPE ϕ ( u 1: m , θ ) := -log q NPE ϕ ( θ | G θ ( u 1 ) , . . . , G θ ( u m )) = -log q NPE ϕ ( θ | x 1: m ) . Hereafter, we present our methods using f ϕ and ℓ in order to avoid duplication.

The MC estimator of the loss ℓ ( ϕ ) is given by ℓ MC ( ϕ ) := 1 n ∑ n i =1 f ϕ ( u 1: m,i , θ i ) . Note that the variance of this MC estimator depends on the number of iid samples n . Hence, a small n owing to a computationally expensive simulator will lead to a poor estimator for the loss. Now suppose that we have access to a sequence of generators G 0 θ ( u ) , . . . , G L -1 θ ( u ) with varying fidelity levels. Then, for l = 0 , . . . , L -1 , we can define a corresponding sequence of functions f l ϕ : U m × Θ → R :

such that G L θ ( u ) := G θ ( u ) and f L ϕ ( u 1: m , θ ) := f ϕ ( u 1: m , θ ) . This sequence of functions gives evaluations of the log conditional density at evaluations of the (approximate) simulator. Recall that the larger the value of l , the more accurate (and computationally expensive) such simulations will tend to be. At this point, we can re-express the objective using a telescoping sum as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (1) follows by adding then subtracting some terms, and using linearity of expectations. Now suppose that we can simulate from each of these approximate simulators to obtain:

<!-- formula-not-decoded -->

for i = 1 , . . . , n l and l = 0 , . . . , L . We can then use these simulations to approximate each term in the telescoping sum through a Monte Carlo estimator as follows:

<!-- formula-not-decoded -->

This corresponds to an (unbiased) MLMC estimator of our original objective in (1). The first term, h 0 ( ϕ ) , approximates ℓ ( ϕ ) , but is biased since it uses f 0 ϕ (i.e. the lowest fidelity simulator) rather than f L ϕ (the highest fidelity simulator). The additional terms, h 1 ( ϕ ) , . . . , h L ( ϕ ) , correct this bias by estimating the expected difference between the objectives at consecutive fidelity levels.

Within each term h l ( ϕ ) , the functions f l ϕ and f l -1 ϕ are evaluated on the same samples u l 1: m and θ l , i.e., they are seed-matched . This ensures that f l ϕ ( u l 1: m,i , θ l i ) and f l -1 ϕ ( u l 1: m,i , θ l i ) are coupled and highly correlated, which leads to a reduction in variance [Owen, 2013, Ch. 8] since

<!-- formula-not-decoded -->

and the covariance will be large. This covariance will be particularly large the more similar the two functions f l ϕ and f l -1 ϕ are, and we therefore expect Var [ h l ( ϕ )] to be smallest in those settings. We can also immediately see that without seed-matching, the covariance will be small and the variance will be large, highlighting why seed-matching is essential for MLMC.

Computational cost. Let C l be the computational cost of sampling one x from the l th level generator G l θ and evaluating f l ϕ , and recall that C 0 &lt; C 1 &lt; . . . &lt; C L . Then, the cost of MLMC is

<!-- formula-not-decoded -->

̸

while that of the MC estimator is Cost ( ℓ MC ( ϕ ); n ) = O ( nC L ) . Using solely the high-fidelity generator (as is customary in SBI) would require a large n in order to reasonably estimate θ , thus increasing the total cost. However, with multiple lower-fidelity generators available, we can have a different number of simulated samples per level (i.e. we can take n 0 = n 1 = . . . = n L ), and can select n 0 , . . . , n L such that n l &lt; n l -1 . This allows us to take a much larger number of samples from the cheaper (or low C l ) approximations of the simulator, and a much smaller number of samples from the expensive (or high C l ) approximations of the simulator, making MLMC particularly attractive for reducing the total computational cost of simulation in neural SBI.

̸

̸

Extensions. There are several straightforward extensions of our approach which are not covered above so as to not overload notation. Firstly, each simulator could have its own base measure U l , which could be defined on spaces {U l } L l =1 of different dimensions. This is not a problem since we could simply consider U to be the tensor product measure and U to be the corresponding tensor product space, in which case all equations above remain valid. Similarly, the parameter space may differ across simulators. However, to ensure the best possible performance, it will still be essential to seed-match random numbers where there is overlap; see Owen [2013, Ch. 8] for more details on the use of common random numbers, and Section 5 for a study of this issue for multilevel neural SBI.

Secondly, although our discussion has pertained to multilevel-NLE and multilevel-NPE so far, it is straightforward to extend the MLMC approach to other neural SBI methods such as neural ratio estimation [Hermans et al., 2020, Durkan et al., 2020, Miller et al., 2022], score-based NPE [Geffner et al., 2023], flow-matching NPE [Wildberger et al., 2023], and GAN-based NPE [Ramesh et al., 2022] since these are all based on objectives which can be expressed as MC estimators.

Optimisation. Gradient-based optimisation of the MLMC objective ℓ MLMC ( ϕ ) can be challenging due to the 'conflicting' gradients which appear in consecutive terms ∇ ϕ h l ( ϕ ) and ∇ ϕ h l +1 ( ϕ ) . More precisely, the term ∇ ϕ h l ( ϕ ) always contains

<!-- formula-not-decoded -->

which approximates ∇ ϕ E [ f l ϕ ] , whilst the term ∇ ϕ h l +1 ( ϕ ) always contains

<!-- formula-not-decoded -->

which approximates -∇ ϕ E [ f l ϕ ] . In the infinite-sample limit, ∇ ϕ E [ f l ϕ ] and -∇ ϕ E [ f l ϕ ] cancel out, but this is not the case for ζ l, + ϕ and ζ l, -ϕ since we are typically working with only a small number of expensive

## Algorithm 1 MLMCgradient adjustment

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

simulations. When naively applying standard gradient-based optimisation methods on this loss, we observe that the training dynamics is typically dominated by only one of these two quantities until approaching stationarity, at which point the conflicting gradients lead to unstable updates and, ultimately, often cause divergence.

To mitigate this issue, we use a combination of gradient adjustments summarised in Algorithm 1. Firstly, we rescale ζ l, + ϕ and ζ l, -ϕ to ensure that they have comparable norms and that their difference remains small and stable. Secondly, we apply the gradient projection technique of Liu et al. [2020], projecting the gradients of h 0 ( ϕ ) and h c ( ϕ ) := ∑ L l =1 h l ( ϕ ) onto each other's normal planes to reduce the impact of conflicting gradients. We observe empirically that combining these two techniques significantly improves the stability of the optimisation throughout the training and leads to better performance; see Appendix B.6 for a detailed comparison.

## 4 Theory

We now present our main theoretical results. Theorem 1 expresses the variance of each of the terms in the telescoping sum approximation (see (2)) as a function of the number of simulations per level, and the magnitude of the difference in generators between consecutive levels.

We say that µ is log-concave if it has a density of the form exp( -ψ ( z )) for some convex function ψ : Z → R . We recall that for r ∈ [1 , ∞ ) , d, d ′ ∈ N and a non-empty, open, connected set Z ⊆ R d , the space of vector-valued r -integrable functions with respect to a probability distribution µ is given by L r ( µ ) := { g : Z → R d ′ : ∥ g ∥ L r ( µ ) := ( ∫ Z ∥ g ( x ) ∥ r 2 µ ( dx )) 1 / r &lt; ∞} . For τ ∈ N , the corresponding Sobolev space of vector-valued functions of smoothness τ is given by W τ,r ( µ ) := { g : Z → R d ′ : ∥ g ∥ W τ,r ( µ ) = ( ∑ | α |≤ τ ∥ D α g ∥ r L r ( µ ) ) 1 / r &lt; ∞} , where for a multi-index α ∈ N d , D α is the weak derivative operator corresponding to α . Finally, we recall that a function g is locally K Lip -smooth if its gradient is locally Lipschitz continuous with Lipschitz constant K Lip &gt; 0 ; i.e. for all z, z ′ in some open set of Z , we have that ∥∇ g ( z ) -∇ g ( z ′ ) ∥ 2 ≤ K Lip ∥ z -z ′ ∥ 2 . For simplicity, we will write ˜ q ϕ ( x 1: m , θ ) for the conditional density model used for either NLE (in which case ˜ q ϕ ( x 1: m , θ ) = q NLE ϕ ( x 1 | θ ) ) and NPE (in which case ˜ q ϕ ( x 1: m , θ ) = q NPE ϕ ( θ | x 1: m ) ).

Theorem 1. Let ϕ ∈ Φ and suppose the following assumptions hold:

- (A1) The prior π and the base measure U are log-concave distributions.
- (A2) The generators satisfy ∥ ∥ G l ∥ ∥ W 1 , 4 ( π × U ) ≤ S l for l ∈ { 0 , 1 , . . . , L } .
- (A3) log ˜ q ϕ is continuously differentiable, locally K Lip ( ϕ ) -smooth and satisfies the growth condition ∥∇ log ˜ q ϕ ( x 1: m , θ ) ∥ 2 ≤ K grow ( ϕ )( ∑ m i =1 ∥ x i ∥ 2 + ∥ θ ∥ 2 + 1) for some K Lip ( ϕ ) , K grow ( ϕ ) &gt; 0 .

Then, for l ∈ { 1 , . . . , L } and K 0 ( ϕ ) , . . . , K L ( ϕ ) &gt; 0 independent of n 0 , . . . , n L , we have that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

See Appendix A.1 for the proof, and we emphasise that the variance is over both parameters θ and noise, but conditional on ϕ . (A1) requires log-concavity, which is a strong condition, but this is only required of the prior and the base measure. However, in SBI these tend to be simple distributions such as Gaussians or uniforms, which satisfy this assumption; see Saumard and Wellner [2014] for how to verify log-concavity in practice. (A2) is relatively mild: it asks that G 0 , . . . , G L have at least one derivative in θ and u , and for these generators and their derivatives to have a fourth moment. This will hold when the simulators are used to define sufficiently well-behaved data-generating processes, but will be violated for sufficiently heavy-tailed distributions (e.g. the Cauchy). Finally, (A3) is mild; it holds when the gradient of the log conditional density estimator is Lipschitz continuous, which is for example the case when the conditional density is twice continuously differentiable and the Hessian is bounded (see e.g. Lemma 2.3 of Wright and Recht [2022]). It also holds for models such as mixtures of Gaussians, which are used in mixture density networks and for normalising flows with sufficiently regular transformations; see Table 1 in Liang et al. [2022] for some known Lipschitz continuous transformations. There are two key implications of Theorem 1. The first is a bound on the variance of the MC objective:

<!-- formula-not-decoded -->

which is obtained by noticing that the bound on Var [ h 0 ( ϕ )] is simply a bound on a Monte Carlo objective. From this, we immediately notice that the bound will be large whenever the high-fidelity simulator is expensive and n is small, or whenever the simulator is complex as measured in this Sobolev norm. The second implication is a bound on the variance of the MLMC objective:

<!-- formula-not-decoded -->

In order to make this bound small, we need to make each term small. Typically, ∥ G 0 ∥ W 1 , 4 ( π × U ) will be large, but this will be counterbalanced by taking n 0 to be large. For the higher-fidelity levels, the number of samples n l will typically be smaller, but this will be counter-balanced by the fact that ∥ G l -G l -1 ∥ W 1 , 4 ( π × U ) is small whenever G l -1 is a good approximation of G l . As we now show, we can directly use (5) to get an indication of how to select n 0 , . . . , n L .

Theorem 2. Suppose the assumptions of Theorem 1 hold. Then, the values of n 0 , . . . , n L which minimise the upper bound on Var [ ℓ MLMC ( ϕ )] in (5) given a fixed computational budget; i.e. for Cost ( ℓ MLMC ( ϕ ); n 0 , . . . , n L ) ≤ C budget for some C budget &gt; 0 , are given by

<!-- formula-not-decoded -->

See Appendix A.2 for the proof. This result provides useful intuition on how to select the number of samples per level. For instance, consider the case L = 1 , i.e., two levels. If G 0 is known to be a good approximation of G 1 , it makes sense to allocate a large budget to generating low-fidelity simulations while only a small number of high-fidelity simulations would be sufficient. On the other hand, if G 0 and G 1 differ substantially, allocating a larger budget to high-fidelity simulations makes sense, despite the higher cost, as it can help accurately capture this difference. Although Theorem 2 provides intuition, it may be hard to obtain the optimal n 0 , . . . , n L exactly since it requires computing quantities which are often unknown. For example, computing Sobolev norms can be challenging, and the results implicitly depend on ϕ through the constants K 0 ( ϕ ) , . . . , K L ( ϕ ) of Theorem 1 (which are unlikely to be tight). This is a common limitation of MLMC theory; see [Giles, 2015, Sec. 2-3].

Before concluding this section, we note that our theory focussed on the variance of ℓ MLMC ( ϕ ) , but another important quantity for gradient-based optimisation will be the variance of the gradient ∇ ϕ ℓ MLMC ( ϕ ) of this objective. It turns out that similar results are straightforward to prove for this quantity under very minor modifications of the assumptions, see Appendix A.3.

Figure 2: Performance of our ML-NLE and ML-NPE method on the g-and-k example. (a) KLdivergence ( ↓ ) between the estimated and the (almost) exact density for ML-NLE under different high-fidelity samples n 1 . We compare it with NLE (low) trained on only low-fidelity data ( n = 10 4 ) and NLE (high), trained on only high-fidelity data ( n = 300 ). (b) Negative log-posterior density (NLPD ↓ ) for ML-NPE, NPE (low) with n = 10 3 , and NPE (high) with n = 100 . (c) One instance of learned densities using NLE. (d) Empirical coverage plot for ML-NPE, NPE (low), and NPE (high).

<!-- image -->

## 5 Numerical Experiments

We compare the performance of our multilevel version of NLE and NPE, termed ML-NLE and ML-NPE, respectively, against their standard counterpart with the MC loss. We use the sbi library [Tejero-Cantero et al., 2020] implementation for NLE and NPE , see Appendix B for the details. The code to reproduce our experiments is available at https://github.com/yugahikida/multilevel-sbi.

## 5.1 The g-and-k distribution: an illustrative example

We first consider the g-and-k distribution [Prangle, 2020] as an illustrative example. This is a very flexible univariate distribution that is defined via its quantile function and has four parameters, controlling the mean, variance, skewness, and kurtosis respectively, making it challenging for SBI methods. It does not typically have a low-fidelity simulator, so we construct one through a Taylor approximation of the quantile function, see Appendix B.1 for details. This makes it a slightly contrived example, but the fact that the g-and-k allows for an efficient approximation of the likelihood will make it particularly convenient to study the performance of NLE-based methods.

We fix the number of low-fidelity samples ( n 0 = 10 4 for ML-NLE and n 0 = 10 3 for ML-NPE) and vary the number of high-fidelity samples n 1 to asses the improvement in performance of our methods as n 1 increases. For ML-NLE, we compute the Kullback-Leibler divergence (KLD) between the estimated conditional density and a numerical approximation of the likelihood. For ML-NPE, we use the negative log-posterior density (NLPD) of the true θ under the estimated posterior density as the metric. The results in Figure 2a and 2b show that the multilevel versions of NLE and NPE perform better than their standard counterparts (with MC loss) using just a fraction of the high-fidelity samples. Unsurprisingly, the performance of MLE-NLE and ML-NPE improves as n 1 increases.

In Figure 2c, we show an example of the NLE densities learned, with additional examples in Appendix B.1. Our ML-NLE method is able to approximate the almost exact g-and-k density the best. The coverage plot in Figure 2d shows that ML-NPE yields slightly conservative posteriors as opposed to the overconfident posteriors obtained from NPE trained on either all low- or all high-fidelity data.

## 5.2 Ornstein-Uhlenbeck process: a popular financial model

Our next experiment involves the Ornstein-Uhlenbeck (OU) process-a stochastic differential equation model commonly used in financial analysis [Minenna, 2003], which in our case outputs a 100 -dimensional Markovian time-series and has three parameters. The process is known to converge to a stationary Gaussian distribution, which we use as the low-fidelity simulator, see Appendix B.3. The example is reproduced from Krouglova et al. [2025], who used it for benchmarking their transfer learning approach to NPE (TL-NPE). TL-NPE first trains an NPE network on low-fidelity simulations, and then uses a small set of high-fidelity data to refine the network parameters until a stopping criterion is met. We implement TL-NPE with n 0 = 1100 and n 1 = { 10 , 100 } , and compare against ML-NPE with n 0 = 1000 and n 1 = { 10 , 100 } . These values are picked to keep the simulation

Figure 3: Performance of ML-NPE and TL-NPE measured by NLPD ( ↓ ) and KL divergence ( ↓ ) with different number of high-fidelity samples n 1 on the OU process. (a) When n 1 = 10 , ML-NPE outperforms on both metric. (b) When n 1 = 100 , TL-NPE outperforms on both metric. Note that for visualisation purpose, we removed outliers, see Appendix B.3 for the result with all the data.

<!-- image -->

budget the same for both methods. TL-NPE uses the default early stopping criterion from the sbi library [Tejero-Cantero et al., 2020], which terminates training if the validation loss increases for 20 epochs. Once the criterion is met, the network parameters achieving the lowest validation loss are selected. This criterion is used for both the low- and the high-fidelity training stages. We use 20% of the data as validation set for early stopping.

We run both methods 20 times and compute the NLPD of the ground-truth parameter θ under the estimated posterior across 500 test points. Additionally, we report the KLD between the posterior obtained using TL-NPE or ML-NPE and a reference NPE posterior trained with n = 10 , 000 highfidelity simulations. Results are aggregated across the 20 runs and all test points, and reported in Figure 8. We observe that both the methods achieve comparable performance: for n 1 = 10 , ML-NPE performs slightly better on both metrics, whereas for n 1 = 100 , TL-NPE slightly outperforms ML-NPE on both.

We additionally investigate a challenging scenario where the high- and low-fidelity simulators have differing dimensionalities of θ , in which our method exhibits some limitations, possibly due to instability during optimisation; see Appendix B.3.1 for details.

## 5.3 Toggle-switch model: a Systems Biology example

We now consider the toggle-switch model [Bonassi et al., 2011, Bonassi and West, 2015]. This model describes the interaction between two genes over time, and has seven parameters and a scalar observation at the end of a time interval. Simulators typically use time-discretisation, and the total number of time-steps T acts as a fidelity parameter: running the model with large T incurs larger computational cost but leads to accurate simulations, while smaller T leads to cheap but inaccurate samples. Thus, this model illustrates a setting with more than two fidelity levels (by taking T to be more than two values). We take T 0 = 50 , T 1 = 80 , and T 2 = 300 to be the number of steps for the three fidelity levels with n 0 = 10 4 , n 1 = 500 , and n 2 = 100 . Guided by the intuition from Theorem 2, we allocate a large budget to estimating the difference between the low- and the medium-fidelity simulators, leveraging prior knowledge that this difference is substantial. See Appendix B.4.1 for results under alternative budget allocations. Note that in this case each fidelity level has a different base measure of dimension 2 T +1 ; however,

Figure 4: MMD ( ↓ ) across 5000 parameter values for NLE and ML-NLE.

<!-- image -->

there is still sufficient seed-matching (see Appendix B.4 for details), and therefore variance reduction, thanks to MLMC. We compare our ML-NLE method with the standard NLE trained on samples from either the low- ( n = 12 , 060 ), the medium- ( n = 7537 ), or the high-fidelity simulator ( n = 2010 ). The number of training data n in each case is selected so as to match the total computational cost of simulation between ML-NLE and NLE, see Appendix B.4. Figure 4 reports the maximum mean discrepancy (MMD) [Gretton et al., 2012] between 500 samples from the learned conditional densities and 500 samples from the high-fidelity simulator across for 5000 different parameter values. We observe that ML-NLE performs better than all the NLE baselines for the same computational cost.

## 5.4 Cosmological Simulations

We now consider a cosmological simulator using the CAMELS suite [Villaescusa-Navarro et al., 2021, 2023]-one of the most computationally intensive cosmological simulations to date-which comprises both low- and high-fidelity data. These are state-of-the-art simulations being used with realworld data. Developing surrogate, multilevel techniques for cosmology has become a key research focus [Chartier et al., 2021, Chartier and Wandelt, 2022]. Our task is to infer a standard cosmological target parameter using a 39-dimensional power spectra of cosmology data, see Appendix B.5 for an example. The low-fidelity simulations are gravity-only N-body simulations, whose physical behaviour is controlled only by the parameter and some Gaussian fluctuations of the initial conditions. The high-fidelity hydrodynamic simulations have additional physics, controlled by an additional five parameters. We include these additional parameters as part of the U space, constituting a case of partial common random numbers between the low- and high-fidelity simulators, similar to Section 5.3.

Here, the high-fidelity simulations can be orders of magnitude ( &gt; × 100 ) slower to generate than the low-fidelity ones, making this a representative problem for our method. Assuming we only have access to n = n 1 = 20 high-fidelity simulations, we wish to ascertain the improvement in inference accuracy by including 1000 lowfidelity simulations (i.e. n 0 = 980 ) using our ML-NPE method. To that end, we measure the NLPD and empirical coverage of the estimated posteriors for 980 test data. The result in Figure 5 shows that ML-NPE performs better than

Figure 5: NLPD ↓ and empirical coverage of NPE and ML-NPE for the cosmological inference task.

<!-- image -->

standard NPE for both the metrics. Standard NPE tends to produce overconfident posteriors, while ML-NPE yields calibrated or underconfident posteriors for most confidence levels. Thus, including low-fidelity samples using our method leads to better inference outcomes. Before concluding, we note that some recent papers demonstrating the potential of multi-fidelity SBI methods in cosmology appeared around the same time as our paper [Saoulis et al., 2025, Thiele et al., 2025]. These more in-depth studies clearly highlight the potential for impact of advanced multi-fidelity SBI methods.

## 6 Conclusion

This paper demonstrated how to reduce the cost of SBI using MLMC, but could more broadly be seen as a way to perform multilevel training of conditional density estimators in scenarios where data from different sources with different accuracy levels needs to be combined. Our method can be readily applied to scenarios with more than two fidelity levels. It is also particularly appealing since it is complementary to other compute-efficient SBI methods. For example, Tatsuoka et al. [2025] recently proposed to train an NPE network on low-fidelity data, and to then use the resulting posterior approximation to guide sampling from the high-fidelity simulator. This approach could easily be combined with our method.

In terms of limitations, our method involves gradient adjustments during optimisation, and we did observe a minor increase of roughly 15%-20% in training time compared to that of standard SBI methods (see Appendix B.2). However, this is not a significant issue as training time is usually negligible compared to costly high-fidelity simulations. Another limitation of our approach is that it is not applicable in cases where seed-matching of the low- and high-fidelity simulators is not possible. This was not an issue in any of the examples we encountered, but limit its applicability in some cases.

## Acknowledgments

The authors are grateful to Sam Power, Tim Sullivan and David Warne for helpful discussions, and to the authors of Krouglova et al. [2025] for sharing their code and identifying a bug in our implementation of their method in a preprint version of this paper. YH and AB were supported by the Research Council of Finland grant no. 362534. FXB was supported by the EPSRC grant [EP/Y022300/1]. NJ was supported by the ERC-selected UKRI Frontier Research Grant EP/Y03015X/1 and by the Simons Collaboration on Learning the Universe.

## References

- J. Alsing, B. Wandelt, and S. Feeney. Massive optimal data compression and density estimation for scalable, likelihood-free inference in cosmology. Monthly Notices of the Royal Astronomical Society , 477(3):2874-2885, 2018. 3
- H. Asi, Y. Carmon, A. Jambulapati, Y. Jin, and A. Sidford. Stochastic bias-reduced gradient methods. In Advances in Neural Information Processing Systems , volume 34, pages 10810-10822, 2021. 4
- M. A. Beaumont. Approximate Bayesian computation in evolution and ecology. Annual Review of Ecology, Evolution, and Systematics , 41(1):379-406, 2010. 1
- M. A. Beaumont. Approximate Bayesian computation. Annual Review of Statistics and Its Application , 6(1):379-403, 2019. 1
- M. A. Beaumont, W. Zhang, and D. J. Balding. Approximate Bayesian computation in population genetics. Genetics , 162(4):2025-2035, 2002. 1
- M. A. Beaumont, J-M. Cornuet, J-M. Marin, and C. P. Robert. Adaptive approximate Bayesian computation. Biometrika , 96(4):983-990, 2009. 3
- J. Behrens and F. Dias. New computational methods in tsunami science. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences , 373(2053):20140382, 2015. 1
- A. Bharti, F-X. Briol, and T. Pedersen. A general method for calibrating stochastic radio channel models with kernels. IEEE Transactions on Antennas and Propagation , 70(6):3986-4001, 2022a. 1
- A. Bharti, L. Filstroff, and S. Kaski. Approximate Bayesian computation with domain expert in the loop. In Proceedings of the 39th International Conference on Machine Learning , volume 162, pages 1893-1905, 2022b. 3
- A. Bharti, M. Naslidnyk, O. Key, S. Kaski, and F-X. Briol. Optimally-weighted estimators of the maximum mean discrepancy for likelihood-free inference. In Proceedings of the 40th International Conference on Machine Learning , volume 202, pages 2289-2312, 2023. 3
- A. Bharti, D. Huang, S. Kaski, and F-X. Briol. Cost-aware simulation-based inference. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics , volume 258, pages 28-36, 2025. 3
- C. M. Bishop. Mixture density networks . Aston University, 1994. 3, 29
- S. G. Bobkov. Isoperimetric and analytic inequalities for log-concave probability measures. The Annals of Probability , 27(4):1903-1921, 1999. 17
- J. Boelts, J-M. Lueckmann, R. Gao, and J. H. Macke. Flexible and efficient simulation-based inference for models of decision-making. Elife , 11:e77220, 2022. 1, 2
- F. V. Bonassi and M. West. Sequential Monte Carlo with adaptive weights for approximate Bayesian computation. Bayesian Analysis , 10(1), 2015. 9
- F. V. Bonassi, L. You, and M. West. Bayesian learning from marginal data in bionetwork models. Statistical applications in genetics and molecular biology , 10(1), 2011. 9
- N. Chartier, B. Wandelt, Y. Akrami, and F. Villaescusa-Navarro. CARPool: Fast, accurate computation of large-scale structure statistics by pairing costly and cheap cosmological simulations. Monthly Notices of the Royal Astronomical Society , 503(2):1897-1914, 2021. ISSN 13652966. 10, 30
18. Nicolas Chartier and Benjamin D. Wandelt. CARPool covariance: Fast, unbiased covariance estimation for large-scale structure observables. Monthly Notices of the Royal Astronomical Society , 509(2):2220-2233, 2022. ISSN 13652966. doi: 10.1093/mnras/stab3097. 10, 30
- Z. Chen, M. Naslidnyk, and F-X. Briol. Nested expectations with kernel quadrature. arXiv:2502.18284 , 2025. 4

- K. Cranmer, J. Brehmer, and G. Louppe. The frontier of simulation-based inference. Proceedings of the National Academy of Sciences , 117(48):30055-30062, 2020. 1
- P. D. Moral, A. Doucet, and A. Jasra. An adaptive sequential Monte Carlo method for approximate Bayesian computation. Statistics and Computing , 22(5):1009-1020, 2011. 3
3. Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516 , 2014. 3
- T. J. Dodwell, C. Ketelsen, R. Scheichl, and A. L. Teckentrup. Multilevel Markov chain Monte Carlo. SIAM Review , 61(3):509-545, 2019. 4
- C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios. Neural spline flows. In Advances in Neural Information Processing Systems , volume 32, pages 7511-7522, 2019. 26
- C. Durkan, I. Murray, and G. Papamakarios. On contrastive learning for likelihood-free inference. In International Conference on Machine Learning , volume 119, pages 2771-2781, 2020. 1, 3, 5
- M. Fujisawa and I. Sato. Multilevel Monte Carlo variational inference. Journal of Machine Learning Research , 22(278):1-44, 2021. 4
- M. Gatti et al. Dark Energy Survey Year 3 results: Simulation-based cosmological inference with wavelet harmonics, scattering transforms, and moments of weak lensing mass maps. II. cosmological results. Physical Review D , 111(6):063504, 2025. 30
- T. Geffner, G. Papamakarios, and A. Mnih. Compositional score modeling for simulation-based inference. In Proceedings of the 40th International Conference on Machine Learning , volume 202, pages 11098-11116, 2023. 3, 5
- M. B. Giles. Multilevel Monte Carlo path simulation. Operations Research , 56(3):607-617, 2008. 4
- M. B. Giles. Multilevel Monte Carlo methods. Acta Numerica , 24:259-328, 2015. 2, 4, 7
- M. Gloeckler, M. Deistler, C. D. Weilbach, F. Wood, and J. H. Macke. All-in-one simulation-based inference. In Proceedings of the 41st International Conference on Machine Learning , volume 235, pages 15735-15766, 2024. 3
- M. Gloeckler, S. Toyota, K. Fukumizu, and J. H. Macke. Compositional simulation-based inference for time series. In The Thirteenth International Conference on Learning Representations , 2025. 3
- T. Goda, T. Hironaka, and T. Iwamoto. Multilevel Monte Carlo estimation of expected information gains. Stochastic Analysis and Applications , 38(4):581-600, 2020. 4
- T. Goda, T. Hironaka, W. Kitade, and A. Foster. Unbiased MLMC stochastic gradient-based optimization of Bayesian experimental designs. SIAM Journal on Scientific Computing , 44(1): A286-A311, 2022. 4
- D. Greenberg, M. Nonnenmacher, and J. H. Macke. Automatic posterior transformation for likelihoodfree inference. In Proceedings of the 36th International Conference on Machine Learning , pages 2404-2414, 2019. 1, 3
17. A Gretton, K Borgwardt, M J Rasch, and B Scholkopf. A kernel two-sample test. Journal of Machine Learning Research , 13:723-773, 2012. 9
- M. U. Gutmann and J. Corander. Bayesian optimization for likelihood-free inference of simulatorbased statistical models. Journal of Machine Learning Research , 17(125):1-47, 2016. 3
- S. Heinrich. Multilevel Monte Carlo methods. In Large-Scale Scientific Computing , volume 24, pages 58-67. Springer, 2001. 4
- J. Hermans, V. Begy, and G. Louppe. Likelihood-free MCMC with amortized approximate ratio estimators. In Proceedings of the 37th International Conference on Machine Learning , pages 4239-4248, 2020. 1, 3, 5

- M. Hoppe, O. Embreus, and T. Fülöp. Dream: A fluid-kinetic framework for tokamak disruption runaway electron simulations. Computer Physics Communications , 268:108098, 2021. 1
- Y. Hu, J. Wang, Y. Xie, A. Krause, and D. Kuhn. Contextual stochastic bilevel optimization. In Thirty-seventh Conference on Neural Information Processing Systems , 2023. 4
- A. Jasra, S. Jo, D. Nott, C. Shoemaker, and R. Tempone. Multilevel Monte Carlo in approximate Bayesian computation. Stochastic Analysis and Applications , 37(3):346-360, 2019. 2
- A. Jasra, K. Law, and C. Suciu. Advanced Multilevel Monte Carlo Methods. International Statistical Review , 88(3):548-579, 2020. 2, 4
5. N Jeffrey and B. D. Wandelt. Solving high-dimensional parameter inference: marginal posterior densities &amp; Moment Networks. Third Workshop on Machine Learning and the Physical Sciences, NeurIPS 2020 , art. arXiv:2011.05991, 2020. 3
- N. Jeffrey, J. Alsing, and F. Lanusse. Likelihood-free inference with neural compression of DES SV weak lensing map statistics. Monthly Notices of the Royal Astronomical Society , 501(1):954-969, 2021. 1
- N. Jeffrey et al. Dark energy survey year 3 results: likelihood-free, simulation-based wCDM inference with neural compression of weak-lensing map statistics. Monthly Notices of the Royal Astronomical Society , 536(2):1303-1322, 2025. 1, 30
- M. C. Kennedy and A. O'Hagan. Predicting the output from a complex computer code when fast approximations are available. Biometrika , 87:1-13, 2000. 2
- O. Key, A. Gretton, F.-X. Briol, and T. Fernandez. Composite goodness-of-fit tests with kernels. Journal of Machine Learning Research , 26(51):1-60, 2025. 28
- D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In International Conference for Learning Representations , 2015. 25
- A. Kirby, F-X. Briol, T. D. Dunstan, and T. Nishino. Data-driven modelling of turbine wake interactions and flow resistance in large wind farms. Wind Energy , 26(9):875-1011, 2023. 1
- A. N. Krouglova, H. R. Johnson, B. Confavreux, M. Deistler, and P. J. Gonçalves. Multifidelity simulation-based inference for computationally expensive simulators. arXiv:2502.08416 , 2025. 2, 8, 10
- S. Kulkarni and C. A. Moritz. Improving effectiveness of simulation-based inference in the massively parallel regime. IEEE Transactions on Parallel and Distributed Systems , 34(4):1100-1114, 2023. 3
- T. Kypraios, P. Neal, and D. Prangle. A tutorial introduction to Bayesian inference for stochastic epidemic models using approximate Bayesian computation. Mathematical Biosciences , 287:42-53, 2017. 1
- K. Li, D. Giles, T. Karvonen, S. Guillas, and F-X. Briol. Multilevel Bayesian quadrature. In International Conference on Artificial Intelligence and Statistics , pages 1845-1868, 2023. 4
- F. Liang, L. Hodgkinson, and M. W. Mahoney. Fat-tailed variational inference with anisotropic tail adaptive flows. In Proceedings of the 39th International Conference on Machine Learning , volume 162, pages 13257-13270, 2022. 7
- J. Lintusaari, M. U. Gutmann, R. Dutta, S. Kaski, and J. Corander. Fundamentals and recent developments in approximate Bayesian computation. Systematic Biology , 66:66-82, 2017. 1
- A. Liu, J. Z. Liu, J-S. Denain, K. Gimpel, S. Sidor, S. Levine, and P. Abbeel. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems , volume 33, pages 5824-5836, 2020. 6
19. J-M. Lueckmann, P. J. Gonçalves, G. Bassetto, K. Öcal, M. Nonnenmacher, and J. H. Macke. Flexible statistical inference for mechanistic models of neural dynamics. In Advances in Neural Information Processing Systems , page 1289-1299, 2017. 1, 3

- J-M. Lueckmann, G. Bassetto, T. Karaletsos, and J. H. Macke. Likelihood-free inference with emulator networks. In Symposium on Advances in Approximate Bayesian Inference , pages 32-53, 2019. 1, 2
- E. Meeds and M. Welling. GPS-ABC: Gaussian process surrogate approximate Bayesian computation. In Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence , page 593-602, 2014. 3
- B. K. Miller, C. Weniger, and P. Forré. Contrastive neural ratio estimation. In Advances in Neural Information Processing Systems , 2022. 5
- M. Minenna. The detection of market abuse on financial markets: A quantitative approach. Quaderni di finanza , 54, 2003. 8
- Z. Niu, J. Meier, and F-X Briol. Discrepancy-based inference for intractable generative models using quasi-Monte Carlo. Electronic Journal of Statistics , 17(1):1411-1456, 2023. 3
- A. B. Owen. Monte Carlo theory, methods and examples . https://artowen.su.domains/mc/ , 2013. 4, 5
- G. Papamakarios and I. Murray. Fast ϵ -free inference of simulation models with Bayesian conditional density estimation. In Advances in Neural Information Processing Systems , pages 1036-1044, 2016. 1, 3
- G. Papamakarios, D. Sterratt, and I. Murray. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics , pages 837-848, 2019. 1, 2, 3
- G. Papamakarios, E. Nalisnick, S. Rezende, D. J.and Mohamed, and B. Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research , 22 (57):1-64, 2021. 3
- George Papamakarios, Theo Pavlakou, and Iain Murray. Masked autoregressive flow for density estimation. In Advances in neural information processing systems , volume 30, 2017. 3
- S. Paszke, A.and Gross, S. Chintala, E. Chanan, G.and Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in pytorch. In NIPS-W , 2017. 25
- B. Peherstorfer, K. Willcox, and M. Gunzburger. Survey of multifidelity methods in uncertainty propagation, inference, and optimization. SIAM Review , 60(3):550-591, 2018. 2
- D. Prangle. Lazy ABC. Statistics and Computing , 26:171-185, 2016. 3, 26
- D. Prangle. gk: An R Package for the g-and-k and Generalised g-and-h Distributions. The R Journal , 12(1):7, 2020. 8
- T. P. Prescott and R. E. Baker. Multifidelity approximate Bayesian computation. SIAM-ASA Journal on Uncertainty Quantification , 8(1):114-138, 2020. 2
- T. P. Prescott and R. E. Baker. Multifidelity approximate Bayesian computation with sequential Monte Carlo parameter sampling. SIAM-ASA Journal on Uncertainty Quantification , 9(2):788-817, 2021. 2
- T. P. Prescott, D. J. Warne, and R. E. Baker. Efficient multifidelity likelihood-free Bayesian inference with adaptive computational resource allocation. Journal of Computational Physics , 496:112577, 2024. 2
- W. H. Press. Numerical recipes 3rd edition: The art of scientific computing . Cambridge university press, 2007. 26
- L. F. Price, C. C. Drovandi, A. Lee, and D. J. Nott. Bayesian synthetic likelihood. Journal of Computational and Graphical Statistics , 27(1):1-11, 2018. 2

- S. T. Radev, U. K. Mertens, A. Voss, L. Ardizzone, and U. Köthe. Bayesflow: Learning complex stochastic models with invertible neural networks. IEEE Transactions on Neural Networks and Learning Systems , 33(4):1452-1466, 2022. 1, 3
- S. T. Radev, M. Schmitt, V. Pratz, U. Picchini, U. Köthe, and P-C. Bürkner. Jana: Jointly amortized neural approximation of complex bayesian models. In Uncertainty in Artificial Intelligence , pages 1695-1706, 2023a. 2
- S. T. Radev, M. Schmitt, L. Schumacher, L. Elsemüller, V. Pratz, Y. Schälte, U. Köthe, and P-C. Bürkner. Bayesflow: Amortized bayesian workflows with neural networks. Journal of Open Source Software , 8(89):5702, 2023b. 25
- P. Ramesh, J-M. Lueckmann, J. Boelts, Á. Tejero-Cantero, D. S. Greenberg, P. J. Goncalves, and J. H. Macke. GATSBI: Generative adversarial training for simulation-based inference. In International Conference on Learning Representations , 2022. 5
- G. D. Rayner and H. L. MacGillivray. Numerical maximum likelihood estimation for the g-and-k and generalized g-and-h distributions. Statistics and Computing , 12(1):57-75, 2002. 25
- D. Rezende and S. Mohamed. Variational inference with normalizing flows. In International conference on machine learning , pages 1530-1538, 2015. 3
- C. P. Robert and G. Casella. Monte Carlo Statistical Methods . Springer, 2000. 4
- A. A. Saoulis, D. Piras, N. Jeffrey, A. Spurio-Mancini, A. M. G. Ferreira, and B. Joachimi. Transfer learning for multifidelity simulation-based inference in cosmology. arXiv:2505.21215 , 2025. 10
- A. Saumard and J. A. Wellner. Log-concavity and strong log-concavity: A review. Statistics Surveys , 8:45-114, 2014. 7, 17, 19
- M. Schmitt, D. R. Ivanova, D. Habermann, U. Köthe, P-C. Bürkner, and S. T. Radev. Leveraging self-consistency for data-efficient amortized Bayesian inference. In Proceedings of the 41st International Conference on Machine Learning , 2024a. 3
- M. Schmitt, V. Pratz, U. Köthe, P-C. Bürkner, and S. Radev. Consistency models for scalable and fast simulation-based inference. Advances in Neural Information Processing Systems , 37: 126908-126945, 2024b. 3
- L. Sharrock, J. Simons, S. Liu, and M. Beaumont. Sequential neural score estimation: Likelihood-free inference with conditional score based diffusion models. In Proceedings of the 41st International Conference on Machine Learning , volume 235, pages 44565-44602, 2024. 3
- Y. Shi and R. Cornish. On multilevel monte carlo unbiased gradient estimation for deep latent variable models. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics , volume 130, pages 3925-3933, 2021. 4
- S. A. Sisson, Y. Fan, and Mark M. Tanaka. Sequential Monte Carlo without likelihoods. Proceedings of the National Academy of Sciences , 104(6):1760-1765, 2007. 3
- C. Tatsuoka, M. Yang, D. Xiu, and G. Zhang. Multi-fidelity parameter estimation using conditional diffusion models. arXiv:2504.01894 , 2025. 10
- A. Tejero-Cantero, J. Boelts, M. Deistler, J-M. Lueckmann, C. Durkan, P. J. Gonçalves, D. S. Greenberg, and J. H. Macke. sbi: A toolkit for simulation-based inference. Journal of Open Source Software , 5(52):2505, 2020. 8, 9, 25
17. Leander Thiele, Adrian E. Bayer, and Naoya Takeishi. Simulation-efficient cosmological inference with multi-fidelity SBI. 2025. 10
- O. Thomas, R. Dutta, J. Corander, S. Kaski, and M. U. Gutmann. Likelihood-free inference by ratio estimation. Bayesian Analysis , 17(1):1-31, 2022. 1
- F. Villaescusa-Navarro et al. The CAMELS Project: Cosmology and Astrophysics with Machinelearning Simulations. The Astrophysical Journal , 915(1):71, 2021. 10, 30

- F. Villaescusa-Navarro et al. The CAMELS Project: Public Data Release. The Astrophysical Journal Supplement Series , 265(2):54, 2023. 2, 10, 30
- P. Virtanen et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods , 17:261-272, 2020. 25
- D. J. Warne, R. E. Baker, and M. J. Simpson. Multilevel rejection sampling for approximate bayesian computation. Computational Statistics &amp; Data Analysis , 124:71-86, 2018. 2
- D. J. Warne, T. P. Prescott, R. E. Baker, and M. J. Simpson. Multifidelity multilevel Monte Carlo to accelerate approximate Bayesian parameter inference for partially observed stochastic processes. Journal of Computational Physics , 469:111543, 2022. 2
- J. B. Wildberger, M. Dax, S. Buchholz, S. R. Green, J. H. Macke, and B. Schölkopf. Flow matching for scalable simulation-based inference. In Thirty-seventh Conference on Neural Information Processing Systems , 2023. 3, 5
- S. N. Wood. Statistical inference for noisy nonlinear ecological dynamic systems. Nature , 466(7310): 1102-1104, 2010. 2
- S. J. Wright and B. Recht. Optimization for Data Analysis . Cambridge University Press, 2022. 7
- S. Yang, V. Zankin, M. Balandat, S. Scherer, K. Carlberg, N. Walton, and K. J. H. Law. Accelerating look-ahead in Bayesian optimization: Multilevel Monte Carlo is all you need. In International Conference on Machine Learning , pages 56722-56748, 2024. 4
- A. Zammit-Mangion, M. Sainsbury-Dale, and R. Huser. Neural methods for amortized inference. Annual Review of Statistics and Its Application , 2024. 1

## Supplemental Material

The appendix is arranged as follows: Appendix A contains the proofs of the theoretical results presented in Section 4. Appendix B consists of the experimental details and additional results.

## A Proof of theoretical results

## A.1 Proof of Theorem 1

Proof. Note that here, and throughout the rest of this proof, we will simplify the notation. We will drop the subscript for variances and expectations, and these should all be understood as being u i ∼ U for i ∈ { 1 , . . . , m } and θ ∼ π . We will also use the variable z to denote a vector containing u 1: m and θ , so that z ∈ R md U + d Θ . Finally, we will write L 4 and W 1 , 4 to denote the spaces L 4 ( π × U ) and W 1 , 4 ( π × U ) .

The proof will be structured as follows. We will express the variance of each term using the variance of individual samples. We will then use a Poincaré-type inequality to bound the variance of the objective in terms of the expected squared norm of its gradient, then we will upper bound the norm using only terms which depend on constants and terms expressing how well G l -1 approximates G l .

Since each term in the MLMC expansion is an MC estimator (and therefore based on independent samples), we can express the variances of each term as follows:

<!-- formula-not-decoded -->

where we use the fact that the variance of a sum of independent random variables is the sum of variances. Similarly for the other levels,

<!-- formula-not-decoded -->

For the first step, we use a version of a Poincaré-type inequality for log-concave measures due to [Bobkov, 1999] (note that a simpler statement and additional discussion is provided in Proposition 10.1 (b) of Saumard and Wellner [2014] for the case of strongly log-concave measures). This result shows that for any log-concave measure µ , there exists K Poin &gt; 0 such that for any sufficiently regular integrand f : Z → R where Z ⊆ R d Z , Var [ f ( z )] ≤ K Poin E z ∼ µ [ ∥∇ f ( z ) ∥ 2 2 ] . Applying this Poincaré inequality to each term of the learning objective (i.e. to the terms in (6) and (7)), we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for l ∈ { 1 , . . . , L } . Here, we emphasise again that the expectation is over z , which encompasses both θ and u 1: m , and the vector ∇ z f l ϕ ( z ) ∈ R d U m + d Θ for any l ∈ { 1 , . . . , L } . This result requires the joint distribution of the prior π and m times the base measure U to be log-concave, which holds since the product of log-concave densities is also log-concave (since the sum of convex functions is convex) and we have assumed that the prior and base measures are independent and separately log-concave through Assumption (A1).

To simplify notation, we now introduce the vector-valued function g l ( z ) = g l ( θ, u 1: m ) = ( G l θ ( u 1 ) ⊤ , . . . , G l θ ( u m ) ⊤ , θ ⊤ ) ⊤ so that ˜ q ϕ ( x 1: m , θ ) = ˜ q ϕ ( g l ( θ, u )) = ˜ q ϕ ( g l ( z )) . We will now derive our first bound, which looks at the first term in the MLMC expansion. To do so, we simplify

(8) as follows

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, we have that (10) follows due to the chain rule, (11) follows due to the definition of the matrix 2-norm, (12) follows due to the Cauchy-Schwarz inequality of expectations.

We now bound each term in (12) separately. For the first term, we get

<!-- formula-not-decoded -->

̸

where (13) and (14) follow from the fact that the two norm squared of a matrix is less than the sum of the two norm squared of sub-matrices constructed through rows and columns, (15) follows by noticing that ∥∇ θ θ ∥ 2 2 = d Θ and ∥∇ u i G 0 θ ( u j ) ∥ 2 2 = 0 whenever i = j , and (16) follows similarly to (13) and (14). Taking squares and an expectation, we get:

<!-- formula-not-decoded -->

where (18) follows from ( ∑ n i =1 a i ) 2 ≤ n ∑ n i =1 a 2 i , (19) follows from the definition of the Sobolev norm and the fact that u 1 , . . . , u m have the same distribution and hence the same expectation, and (20) follows by grouping constants together.

We now move on to bounding the second term in (12). Since we assumed in Assumption (A3) that ∇ log ˜ q ϕ satisfies a linear growth condition, we must have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, (21) uses the growth condition, (22) holds by applying ( ∑ n i =1 a i ) 4 ≤ n 3 ∑ n i =1 a 4 i , (23) follows from the definition of Sobolev norm, and (24) uses the fact that E [ ∥ θ ∥ 4 2 ] is upper bounded by a constant since the expectation is against π , which a log-concave distribution and hence all its moments are finite (see Section 5.1. of Saumard and Wellner [2014]).

Combining (6), (8), (12), (20), and (24) therefore gives:

<!-- formula-not-decoded -->

where K 0 ( ϕ ) is used to combine all of the constants. This now concludes the first part of our results, which bounds the variance of the first term in the MLMC telescoping sum.

We can now derive a similar bound for the other terms (i.e. to simplify (9)). We do this by first only considering the norm inside of the expectation:

<!-- formula-not-decoded -->

Here, (26) follows from the chain rule, (27) follows by adding and subtracting the term ∇ z g l ( z ) ∇ log ˜ q ϕ ( g l -1 ( z )) and using the triangle inequality, and (28) follows from the CauchySchwarz inequality of the 2-norm. Squaring both sides of this inequality and taking expectations, we

then obtain:

<!-- formula-not-decoded -->

where (29) follows from that fact that for a, b ∈ R , we have ( a + b ) 2 ≤ 2 a 2 +2 b 2 , and (30) follows from the Cauchy-Schwarz inequality for expectations. To conclude this proof, we notice that the derivation from (13) to (16) can be modified by replacing G 0 by G l gives:

<!-- formula-not-decoded -->

where the last inequality holds thanks to Assumption (A2). Similarly, replacing G 0 by G l -G l -1 and following the derivations from (13) to (20) gives

<!-- formula-not-decoded -->

We notice that we lose the additive term since the last columns of the matrix ∇ z g l ( z ) -∇ z g l -1 ( z ) form a zero matrix. This is because

<!-- formula-not-decoded -->

The non-zero lower-right block, i.e., ∇ θ θ in (14), was the cause of the additive term in (20) and the final result (the upper-right block is always zero since ∇ u θ = 0 ), which is now cancelled out. Additionally, we could replace G 0 by G l -1 in the the bound from (22) to (24) in order to get:

<!-- formula-not-decoded -->

where once again we used Assumption (A2).

Finally, we split the following expression to make use of our local Lipschitz property:

<!-- formula-not-decoded -->

Here, (37) follows due to the law of total expectation. For (38), the bound on the first term follows due to ∥ a -b ∥ 4 2 ≤ 8( ∥ a ∥ 4 2 + ∥ b ∥ 4 2 ) and Markov's inequality, and the bound on the second term follows due to the local-Lipschitz condition (i.e. Assumption (A3)) and the fact that a probability is always upper bounded by 1 . Then, (39) follows using (35). Finally, (40) follows by grouping all constants together and noting that

<!-- formula-not-decoded -->

Combining all of the above (i.e. (7), (9), (28), (31), (32), (35), and (39)), we end up with

<!-- formula-not-decoded -->

where K l ( ϕ ) combines all constants. This proves our second result and therefore concludes our proof.

## A.2 Proof of Theorem 2

Proof. We first recall both the cost and the variance of our estimator, modifying our notation slightly to emphasise the number of samples n 0 , . . . , n L . The total cost of this method is given by

<!-- formula-not-decoded -->

where C l is the cost of evaluating f l at level l . In addition, we also have the following upper bound on the variance using Theorem 1:

<!-- formula-not-decoded -->

where once again we write ∥ · ∥ W 1 , 4 to denote the norm ∥ · ∥ W 1 , 4 ( π × U ) . Overall, we would like to solve the following optimisation problem:

<!-- formula-not-decoded -->

where C budget is the computational budget. We relax the problem slightly by minimising the upper bound on the variance instead, thus the problem can be expressed in the following Lagrangian form:

<!-- formula-not-decoded -->

This can be solved by setting all partial derivatives with respect to ( n 0 , . . . , n L ) and ν equal to zero and solving the associated system of equations. Firstly, we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and for l ∈ { 1 , . . . , L } , we have:

<!-- formula-not-decoded -->

Finally, taking the partial derivative with respect to ν confirms that our constraint is active (i.e. we are on the boundary of the feasible region):

<!-- formula-not-decoded -->

Plugging the results of (44) and (45) into (46), we get:

<!-- formula-not-decoded -->

which gives

<!-- formula-not-decoded -->

We can now use the expression for ν that we obtained in (47) to obtain a simplified expression for n 0 , . . . , n L (using (44) and (45)):

<!-- formula-not-decoded -->

This completes our proof.

## A.3 Extension of Theorem 1 to the gradient

We provide an upper bound on the variance for each element of the gradient ∇ ϕ ℓ MLMC ( ϕ ) , that is, on each partial derivative ∇ ϕ j ℓ MLMC ( ϕ ) for j ∈ { 1 , . . . , d ϕ } . The partial derivatives are given by

<!-- formula-not-decoded -->

Theorem 3. Let ϕ ∈ Φ ⊆ R d Φ and suppose the following assumption hold in addition to A1-A2 in theorem 1:

- (A3') ∇ ϕ j log ˜ q ϕ is continuously differentiable, locally K j Lip ( ϕ ) -smooth and satisfies the growth condition ∥∇ ϕ j ∇ log ˜ q ϕ ( x 1: m , θ ) ∥ 2 ≤ K j grow ( ϕ )( ∑ m i =1 ∥ x i ∥ 2 + ∥ θ ∥ 2 + 1) for some K j Lip ( ϕ ) , K j grow ( ϕ ) &gt; 0 for all j = 1 , . . . , d Φ .

Then, for l ∈ { 1 , . . . , L } , K j 0 ( ϕ ) , . . . , K j L ( ϕ ) &gt; 0 , j ∈ { 1 , . . . , d ϕ } independent of n 0 , . . . , n L , we have that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The variance of each term can be expressed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assume that ∇ ϕ j f ϕ is sufficiently regular, applying Poincaré inequality to (48) and (49) (as in (8) and (9)) gives:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the expectation is over z . We can simplify (50) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here (53) is due to the chain rule and (54) follows (11) - (12). The bound for the first expectation is given by (20). For the second expectation, we have:

<!-- formula-not-decoded -->

by Assumption (A3') and following (21) - (24). Combining (48), (50), (54), (20), and (55) gives:

<!-- formula-not-decoded -->

where K j 0 ( ϕ ) is a constant depending on ϕ and j , independent of n 0 . Now, we derive an upper bound on Var [ ∇ ϕ j h l ( ϕ ) ] . Following (26) - (30), we have:

<!-- formula-not-decoded -->

Bound for the first expectation is given by (31) and the third expectation is given by (32). For the forth expectation, from (55) (replacing 0 with l ) and Assumption (A2), we have:

<!-- formula-not-decoded -->

For the second expectation, following (36)-(40), and using our local Lipschitz property, we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where K j score-diff ( ϕ ) combines all the constant, independent of n l and the difference in the simulators. Finally, applying the bounds for the each expectation; (31), (61), (32), and (59), to (58), and combining it with (49) gives:

<!-- formula-not-decoded -->

where K l combines all the constant. Combining (56) and (63) gives a bound on the variance of the partial derivative:

<!-- formula-not-decoded -->

## B Experimental details &amp; additional results

For all the experiments, we use a Mac M4 CPU with 16 GB memory for the training of neural networks. Running all the experiments roughly takes half a day. For optimisation, we use batch gradient descent with the Adam optimiser [Kingma and Ba, 2015] using Pytorch [Paszke et al., 2017]. For the construction of the conditional density estimator, we use the SBI package [Tejero-Cantero et al., 2020]. We also use the BayesFlow package [Radev et al., 2023b] for some of the figures. For each experiment, we fix the number of epochs unless stated otherwise. We use the same setting and stopping criterion for the NPE and the NLE baseline as in the default implementation of the SBI package. We set the learning rate to 10 -4 for all the experiments.

## B.1 The g-and-k distribution

Simulator setup. The g-and-k distribution can be written in simulator form as follows:

<!-- formula-not-decoded -->

where Φ -1 is the quantile function of the standard normal distribution and erf -1 is the inverse of error function. We set the prior distribution to be a tensor product of marginal distributions on each parameter: θ 1 , θ 2 , θ 3 ∼ Unif ([0 , 3] 3 ) and θ 4 ∼ Unif ([0 , exp(0 . 5)]) . Note that we always resort to an approximation method for the evaluation of erf -1 ( · ) . For the high-fidelity simulator, we use an accurate approximation implemented in Scipy [Virtanen et al., 2020]. For the low-fidelity simulator, we use a Taylor expansion of erf -1 up to the third order:

<!-- formula-not-decoded -->

An advantage of the g-and-k distribution is that its density can be approximated numerically almost exactly. Following Rayner and MacGillivray [2002], we first numerically obtain F θ ( x ) = G -1 θ ( x )

by solving numerically for x i -G θ ( u i ) = 0 using a root-solving algorithm [Press, 2007] 1 , then we obtain the density function by taking ˆ p ( x | θ ) := F ′ θ ( x ) = ∂F θ ( x ) /∂x . For this second step, we typically use a finite difference approximation. Seed-matching is simply done by using same the u and θ .

## Neural network details

NLE: We use the neural spline flow (NSF) [Durkan et al., 2019]. We pick 10 bins, span of [ -7 , 7] and 1 coupling layer since we only have one dimensional input. The conditioner for the NSF is a multilayer perceptron neural network (MLP) with 3 hidden layers of 50 units, and 10% dropout, trained for 10 , 000 epochs.

NPE: We use NSF with 3 bins, span of [ -3 , 3] and 3 coupling layers. The conditioner for the NSF is a MLP with 2 hidden layers of 50 units, and 10% dropout, trained for 800 epochs. For each true parameter values, we produce m = 1000 iid samples and obtain quantile based 4 -dimensional summary statistics following Prangle [2016].

## Evaluations details

NLE: We calculate the forward KL-divergence between the almost exact density and the approximated density over 2000 equidistant points in [ -30 , 30] .

NPE: We calculate NLPD over 500 simulations. Empirical coverage is estimated using average value of 500 simulated datasets, where we draw 2000 posterior samples from each simulations. We then calculate 1 -β -highest posterior density credible interval, where β is 101 equidistant points between 0 and 1 .

## B.2 Training time comparison

We report training time per epoch for our multilevel versions of NLE and NPE and their standard counterparts for the g-and-k experiment; see Appendix B.2. For both methods, we picked n = 2000 for the standard NLE/NPE with MC loss and n 0 = 1000 , n 1 = 500 for our ML-NLE/ML-NPE with MLMC loss such that the total number of samples are the same. Each network is trained independently 10 times to assess uncertainty, with a training budget of 500 epochs for NLE and 100 epochs for NPE.

Table 1: Average training time per epoch (standard deviation in gray).

| Method   | Training Time per Epoch (s × 10 - 3 )   |
|----------|-----------------------------------------|
| ML-NLE   | 1.82 (±0.05)                            |
| NLE      | 1.58 (±0.06)                            |
| ML-NPE   | 8.04 (±0.17)                            |
| NPE      | 6.56 (±0.14)                            |

## B.3 Ornstein-Uhlenbeck process

Simulator setup. Given T = 10 (total time), ∆ t = 0 . 1 (time step), x 0 = 2 . 0 (initial value), θ = [ γ, µ, σ ] ⊤ , N = ⌊ T/ ∆ t ⌉ = 100 (number of steps), the high- and the low-fidelity OUP simulators are defined as follows:

## High-fidelity:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

1 Note that since G θ ( u ) is a quantile function, F θ ( x ) := G -1 θ ( x ) is a cumulative distribution function.

Figure 6: Additional results for the g-and-k experiment. (a) Histogram of 1000 seed-matched samples from low- and high-fidelity simulator. (b) ISE (integrated squared error) for NLE ( ↓ ): sum of the squared distance between (almost) exact density and approximated density over 2000 points in the interval [ -30 , 30] . The performance of ML-NLE improves as n 1 increases. (c) Approximated density and (almost) exact density for NLE across four simulations. The first row of (c) shows that we get better approximation of the density when combining low and fidelity samples than using them separately. The second row of (c) shows the improvement of the performance of our ML-NLE as n 1 increases. These results demonstrate the effectiveness of our method. (d) Recovery plot for NPE: It measures how well the ground truth value is captured by the median of approximated posterior. The x-axis shows the ground truth parameter value and the y-axis shows the median of the posterior distribution with the median absolute deviation, i.e. ± median ( | θ -median ( θ ) | ) , shown around the points. Here r and R 2 denotes the correlation coefficient ( ↑ ) and the coefficient of determination ( ↑ ) between the ground truth values and the estimated median, respectively. The dashed diagonal line indicates perfect recovery of the true parameter. Using MLMC leads to significant improvements in the parameter recovery, which become more pronounced as n 1 increases.

<!-- image -->

## Low-fidelity:

<!-- formula-not-decoded -->

We set the prior distribution θ ∼ Unif ([0 . 1 , 1 . 0] × [0 . 1 , 3 . 0] × [0 . 1 , 0 . 6]) . The resulting x is a 100 -dimensional time series. Seed-matching is simply done by using the same u and θ .

Neural network details. We use NSF with 2 bins, span of [ -2 , 2] and 2 coupling layers. The conditioner for the NSF is a MLP with 1 hidden layer of 32 units, and 10% dropout. For ML-NPE, we trained the network for 500 epochs. For TL-NPE, we set a high maximum number of epochs and used on early stopping. The stopping criterion is based on the validation loss computed on 20% of the data, with a patience parameter controlling how many epochs to wait before stopping. For each true parameter values, we produce m = 1 sample and use 5 representative data points as a summary statistics. The subsamples are taken at equal interval in log space such that we take 0 , 3 , 10 , 31 , 99 th data points.

Evaluations details. We train each ML-NPE and TL-NPE 20 times. For each trained network, we compute the NLPD and KL divergence against reference posterior across 500 simulated datasets.

Figure 7: Additional results for the OUP experiment. (a) Example of one seed-matched sample from high and low-fidelity simulators. (b) Recovery plot for the reference NPE posterior with n = 10 4 . (c)-(f) Recovery plots for ML-NPE and TL-NPE for n 1 = 10 and n 1 = 100 .

<!-- image -->

Figure 8: Figure 8 with all the data, without removing outliers. Outliers are identified using IQR method with factor 4 , which removes only extreme outliers.

<!-- image -->

## B.3.1 Experiment with different dimension of θ

We now conduct an experiment to explore a failure mode of our method. We include an additional parameter in the high-fidelity simulator, termed 'initial value' θ 4 = x 0 ∼ uniform ([0 , 4]) , instead of fixing it at x 0 = 2 . 0 as we did previously. For this experiment, we picked n 0 = 1 , 000 and n 1 = 100 . We additionally trained the conditional density estimator using only n = 100 data from the high-fidelity simulator. All the other settings remain the same.

Table 2: Mean and standard deviation of NLPD and KLD.

|        | ML-NPE   | NPE (high only)   |
|--------|----------|-------------------|
| NLPD ↓ | -0.18    | -1.21             |
|        | (0.04)   | (0.53)            |

We observe that it notably underperforms in NLPD compared to using only high-fidelity data. This may be due to our training objective being more unstable and therefore more sensitive to large differences in simulator outputs introduced by the additional parameter x 0 .

## B.4 Toggle switch model

Simulator setup. We follow the implementation by Key et al. [2025]. Given parameters θ = [ α 1 , α 2 , β 1 , β 2 , µ, σ, γ ] ⊤ , we can sample from the simulator by x ∼ ˜ N ( µ + u T , µσ/u γ T ) , where ˜ N denotes the truncated normal distribution on R + . Here, u T is given by the discretised-time equation

Figure 9: Additional results for the Toggle-switch experiment. (a) Histograms of 1000 seed-matched samples from the high-, the medium-, and the low-fidelity simulators. Note that the difference between the low- and the medium-fidelity simulator is larger than the difference between the mediumand the high-fidelity simulator. This suggests that we should take n 1 &gt; n 2 . (b) Approximated density and samples from high-fidelity simulator across four simulations. ML-NLE demonstrates superior performance compared to the NLE baselines trained exclusively on high-, medium-, and low-fidelity data with the same total simulation budget. In particular, the density estimated by ML-NLE aligns more closely with high-fidelity simulations, highlighting the improved accuracy of our method.

<!-- image -->

as follows:

<!-- formula-not-decoded -->

We set the initial state u 0 = u 1 = 10 and prior distribution θ ∼ Unif ([0 . 01 , 50] × [0 . 01 , 50] × [0 . 01 , 5] × [0 . 01 , 5] × [250 , 450] × [0 . 01 , 0 . 5] × [0 . 01 , 0 . 4]) . The cost of the simulator increases linearly with T , and T →∞ leads to an exact simulation. Different choice of T leads to the simulator with different fidelity levels, having different cost and precision. The expression for the cost of simulating data for the MLMC and MC is given by

<!-- formula-not-decoded -->

This yields a total data generation cost for ML-NLE with ( n 0 , n 1 , n 2 ) = (10 4 , 500 , 100) , and ( T 0 , T 1 , T 2 ) = (50 , 80 , 300) as C MLMC = 603000 , assuming a unit cost c = 1 . We then allocate the same computational budget to NLE at different fidelities. This results in the following sample sizes: n 2 = C ML-NLE /T 2 = 2010 , n 1 = C ML-NLE /T 1 = 7537 , and n 0 = C ML-NLE /T 0 = 12060 .

Note that U varies with the fidelity level of the simulator, as it is given by R 2 T +1 . This discrepancy can complicate seed-matching across fidelities. To address this, we define a unified domain for the noise U = ⋃ L l =0 U l shared across all fidelity levels, and assume that for lower fidelities (smaller l ), certain components of the input are simply disregarded. For seed-matching, we share θ and u l ∩ u l -1 = u l -1 where u l ∈ U l ⊆ U , ∀ l .

Neural network details. We use Gaussian mixture density network [Bishop, 1994] with two mixture components. We use MLP with 2 hidden layers and 20 units to estimate the parameters: means, variances, and mixture weights, trained for 10 , 000 epochs.

Evaluations details. We sample m = 500 from both high-fidelity simulator and each approximated densities, and calculate MMD over 5000 simulations. The length scale is estimated by the median heuristic.

## B.4.1 Experiment with different allocation of samples n l

We now repeat the same experiment, but with different allocations of n 0 , n 1 , n 2 , keeping the total computational cost roughly the same. Apart from the choice of n 0 = 10 , 000 , n 1 = 500 , n 2 = 100 that we use in Section 5.3 (option A) which is guided by our theory, we include two other alternatives: (B) n 0 = 9260 , n 1 = 200 , n 2 = 300 and (C) n 0 , n 1 , n 2 = 1077 . Option B places more budget on learning the difference between the medium- and the high-fidelity simulator (opposite of option A), and option C allocates equal number of samples for all the terms. The results in Table 3 indicate that option A is the best performing, indicating performance can be improved by careful assignment of the computational budget led by insights from our theory.

Table 3: Mean and standard deviation of MMD. The results for option A, only high, medium, and low remains the same as in Section 5.3.

|          | Option A   | Option B   | Option C   | only high   | only medium   | only low   |
|----------|------------|------------|------------|-------------|---------------|------------|
| MMD( ↓ ) | 0.16       | 0.33       | 0.29       | 0.43        | 0.37          | 0.59       |
| MMD( ↓ ) | (0.23)     | (0.26)     | (0.28)     | (0.29)      | (0.45)        | (0.65)     |

## B.5 Cosmological simulations

Simulator setup. We use the CAMELS simulation suite [Villaescusa-Navarro et al., 2021, 2023], a benchmark dataset for machine learning in astrophysics, to study multi-fidelity simulation-based inference in cosmology. The simulation is one of the most expensive cosmological suites ever run; a small fraction of the next generation are being run on the UK DIRAC HPC Facility with 15M CPUh.

The dataset includes both low-fidelity (gravity-only N-body) and high-fidelity (hydrodynamic) simulations of 25 Mpc /h cosmological volumes. The original inference task involves two cosmological parameters, θ = (Ω m , σ 8 ) , where Ω m is the matter density and σ 8 the amplitude of fluctuations, with mock power spectrum measurements P ( k ) used to infer the parameters. However, due to the limited availability of both high- and low-fidelity simulations, we focus on inferring only σ 8 and treat Ω m as part of the nuisance parameter. We found that attempting to jointly infer both parameters led to poor performance (expected due to physical σ 8 -Ω m degeneracy) even when using 90% of the high-fidelity data.

In this setup, low-fidelity simulations are governed solely by θ and the initial condition seed u , while high-fidelity simulations additionally incorporate complex astrophysical processes (e.g., feedback), modelled via four extra parameters. These high-fidelity simulations are significantly more computationally expensive-often more than 100× slower to generate than low-fidelity ones. Given these limitations, cosmological analyses often rely on conservative data cuts to exclude small-scale modes (e.g., k ≳ 0 . 1 , h/ Mpc), where simulation inaccuracies are most pronounced [e.g. Jeffrey et al., 2025, Gatti et al., 2025]. Multi-fidelity approaches aim to mitigate such constraints by leveraging both inexpensive and high-fidelity simulations to improve the accuracy of cosmological inference.

Note that the idea of combining low- and high- fidelity simulations has previously been proposed in cosmology; see for example the work of Chartier et al. [2021], Chartier and Wandelt [2022] who use low-fidelity simulations to construct approximations of quantities of interest which can be used as control variates. This work differs from our proposed approach in that it targets quantities such as means and covariances, rather than f the training objective of neural SBI.

Neural network details. We use NSF with 3 bins, span of [ -3 , 3] and 3 coupling layers. The conditioner for the NSF is an MLP with 2 hidden layer of 30 units, and 10% dropout, trained for 400 epochs.

Evaluations details. We use the same setting as the g-and-k experiment for NPE except that we use 980 test simulations for evaluation.

## B.6 Ablation study of the gradient adjustment technique

We conduct experiments to evaluate the effect of the different gradient adjustment techniques we employ on the performance of our method. We train both ML-NPE and ML-NLE using (i) our gradient adjustment approach which involves both rescaling and projection, (ii) only rescaling, (iii)

Figure 10: Additional results for the cosmological simulator experiment. (a) 1 seed-matched sample from high and low-fidelity simulators. The one with noise is used to train the neural networks. (b)-(c) Recovery plot of ML-NPE and NPE. Adding low-fidelity samples lead better recovery of the parameter.

<!-- image -->

only projection, and (iv) no gradient adjustment (standard training). We then compare the performance of NPE and NLE on the g-and-k and NLE on the toggle switch experiment. Other than the choice of the gradient adjustment, the training method, hyperparameters, and evaluation methods remain the same. The results are shown in Table 4.

Table 4: Mean and standard deviation of the metrics. For g-and-k NPE, n 0 = 1000 , n 1 = 100 , m = 1000 and for g-and-k NLE, n 0 = 10 4 , n 1 = 300 , m = 1 as in Section 5.1. For toggle switch NLE, n 0 = 10000 , n 1 = 500 , n 2 = 100 as in Section 5.3.

|                            | (i) both     | (ii) only rescaling   | (iii) only projection   | (iv) standard   |
|----------------------------|--------------|-----------------------|-------------------------|-----------------|
| g-and-k NPE (NLPD ↓ )      | -0.30 (0.31) | -0.21 (0.25)          | -0.11 (0.11)            | -0.13 (0.27)    |
| g-and-k NLE (KLD ↓ )       | 0.22 (0.47)  | 0.21 (0.45)           | 0.24 (0.46)             | 0.24 (0.28)     |
| Toggle-switch NLE (MMD ↓ ) | 0.26 (0.25)  | 0.50 (0.37)           | 0.35 (0.27)             | 0.59 (0.31)     |

We observe that when the MLMC loss diverges under standard training, as is the case for the toggle switch experiment and g-and-k with NPE, our gradient adjustment approach that combines both gradient rescaling and projection yields the best results. In the case of NLE on the g-and-k simulator, the loss does not diverge and standard training is sufficient. In such cases, the gradient adjustment yields similar results as standard training, albeit with an increase in the variance of the metric. Therefore, we suggest optimising using our gradient adjustment approach when using ML-NLE or ML-NPE as it avoids the need to first detect whether the loss is diverging or not.

We further compare the training losses with and without gradient adjustment. With the gradient adjustment, the loss diverges, whereas with it, the loss curve exhibits convergence; see Figure 11. To clarify the reason for this behaviour, we also provide an illustration of our gradient scaling procedure in Figure 12.

We note that the optimisation requires some form of regularisation, and the one we used is one possible choice. In our experiments, the gradient adjustment approach performed the best among the options we tried (e.g., regularising the contribution of the difference term when it dominates the first term, or penalising high variance in the difference term).

Figure 11: Comparison of training losses with and without gradient adjustment. (a) Without gradient adjustment, the contribution of ζ 0 , -ϕ begins to dominate the loss around epoch 1000. The resulting conflict between gradient components leads to unstable optimisation, as evidenced by strong fluctuations in the loss and eventual divergence. (b) With gradient adjustment, all components contribute more stably, and the overall loss decreases steadily eventually, indicating convergence. Loss functions are shown for the g-and-k experiment with NLE.

<!-- image -->

Figure 12: Illustration of the gradient scaling with two levels. The coloured arrows indicate gradient of the correction term, total gradient, and scaled gradient respectively. (a) Ideal case: ||∇ ϕ h c ( ϕ ) || remains small and works as a correction to ∇ ϕ h 0 ( ϕ ) . (b) In the later stage of training, ∇ ϕ h 0 ( ϕ ) and ζ 1 , + ϕ diminishes and ζ 0 , -ϕ starts to dominate the optimisation. (c) Gradient scale adjustment: We scale ζ 0 , -ϕ such that || ζ 1 , + ϕ || ≈ || ζ 0 , -ϕ || and ||∇ ϕ h c ( ϕ ) || remains small throughout the training as intended.

<!-- image -->

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: In Section 4, we provide theoretical analysis of our method, and in Section 5, we empirically show sample efficiency of our method compared to standard NPE/NLE and the related work.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations of our method are discussed in Section 6.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Assumptions and theoretical results are provided in Section 4 and complete proof is provided in Appendix A.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The loss function is provided in (2), training method is provided in Section 3. All the experimental details are provided in Appendix B.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Detailed experimental settings, including all hyperparameters, are provided in Appendix B. Reproducible code, along with the simulator and evaluation data, is included in the submission.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: All the hyperparameters for the training and the evaluation methods are provided in Appendix B.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report performance as box and violin plots over multiple runs and/or multiple independently trained networks in Section 5. All the details are in Appendix B.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: It is provided in the first paragraph of Appendix B.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: All data used in this paper is synthetically generated through simulation, posing no ethical or privacy concerns.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the potential positive societal impacts of our method in Section 1, and we are not aware of any significant negative impacts.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: We do not use data or models with high risk for misuse.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All software used in this work is open-source and has been properly cited.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: Reproducible code is provided, along with instructions to replicate all experiments presented in the paper.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve any such experiments.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve any such experiments.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard component.