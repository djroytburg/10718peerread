## Flow based approach for Dynamic Temporal Causal models with non-Gaussian or Heteroscedastic Noises

## Abdellah Rahmani

LTS4, EPFL Lausanne, Switzerland abdellah.rahmani@epfl.ch

## Pascal Frossard

LTS4, EPFL

Lausanne, Switzerland pascal.frossard@epfl.ch

## Abstract

Understanding causal relationships in multivariate time series is crucial in many scenarios, such as those dealing with financial or neurological data. Many such time series exhibit multiple regimes, i.e., consecutive temporal segments with a priori unknown boundaries, with each regime having its own causal structure. Inferring causal dependencies and regime shifts is critical for analyzing the underlying processes. However, causal structure learning in this setting is challenging due to (1) non-stationarity, i.e., each regime can have its own causal graph and mixing function, and (2) complex noise distributions, which may be non-Gaussian or heteroscedastic. Existing causal discovery approaches cannot address these challenges, since generally assume stationarity or Gaussian noise with constant variance. Hence, we introduce FANTOM, a unified framework for causal discovery that handles non-stationary processes along with non-Gaussian and heteroscedastic noises. FANTOM simultaneously infers the number of regimes and their corresponding indices and learns each regime's Directed Acyclic Graph. It uses a Bayesian Expectation Maximization algorithm that maximizes the evidence lower bound of the data log-likelihood. On the theoretical side, we prove, under mild assumptions, that temporal heteroscedastic causal models, introduced in FANTOM's formulation, are identifiable in both stationary and non-stationary settings. In addition, extensive experiments on synthetic and real data show that FANTOM outperforms existing methods.

## 1 Introduction

Causal structure learning from multivariate time series (MTS) is a fundamental problem with diverse applications in traffic modeling [9], biology [44], climate science [43], or healthcare [47]. However, identifying causal relationships in MTS poses several challenges. First, real-world time series are often non-stationary, exhibiting multiple unknown regimes, each potentially governed by different causal relationships. Examples include changing dependencies across climate conditions [27], financial markets [21], and epileptic seizure stages [56]. Second, many MTS display complex noises, e.g., non-Gaussian or even heteroscedastic noise, whose variance depends on both instantaneous and lagged causes. This occurs in fMRI data [46], EEG measurements [22], or financial data [18].

Recent causal discovery methods capture linear and nonlinear interactions with instantaneous and lagged effects [41, 35, 51]. More recently, Gong et al. [15] and Wang et al. [55] explored structural equation models for a single stationary regime governed by a one causal graph with historically dependent noise, where noise variance depends solely on time-lagged variables, neglecting heteroscedasticity. Existing multi-regime methods include RPCMCI [45], which identifies only linear, time-lagged interactions and requires prior knowledge of regime numbers and transitions; and CD-NOD [21], which handles causal discovery from non-stationary MTS, but is limited to homoscedastic noise, cannot infer individual causal graph for each regime, and is incapable of

t

0

1

2

3

4

5

6

1

-

1

t

x

-

<!-- image -->

0

1

2

3

0 1 2 3 4 5 6 1 1 Figure 1: Illustration of FANTOM processing a MTS with two ground truth regimes ( K = 2 ). The algorithm recovers the regime indices I ∗ 1 and I ∗ 2 and learns a temporal causal graph for each regime (dashed edges represent time-lagged links; solid arrows indicate instantaneous links). In the E-step, posterior probabilities p ( z t = r | x t , x &lt;t ) are estimated, where z t = r means x t belongs to regime r . The M-step then infers causal graphs within each regime. Here, N w is the number of regimes that converges to K = 2 .

-

1 identifying recurring regimes. CASTOR [39], infers both regime indices and separate causal graphs per regime, accommodating instantaneous and lagged causal relationships, yet it is still restricted to normal noise. Consequently, the previous methods cannot jointly infer the number of regimes, their indices, their causal graphs, nor effectively manage either non-Gaussian or heteroscedastic noises.

1 To address these limitations, we propose FANTOM, a new framework for Structural Equation Models (SEMs) in multi-regime MTS under either non-Gaussian or heteroscedastic noises. FANTOM is, to the best of our knowledge, the first method to handle heteroscedasticity in both stationary and non-stationary MTS, as well as non-Gaussianity in non-stationary settings. Given a MTS with multiple regimes, FANTOM simultaneously learns each regime's causal graph and mixing function, determines the number of regimes, and infers their indices (Figure 1). It uses a Bayesian Expectation Maximization (BEM) [12] procedure to optimize the evidence lower bound (ELBO), alternatively assigning regime indices (Expectation step) and inferring causal relationships in each regime (Maximization step). Unlike Gaussian-based approaches, FANTOM employs conditional normalizing flows [12] to handle complex distributions and compute regime membership probabilities. It uses Bayesian structure learning that averages across all plausible graphs and naturally filters out spurious edges. Under mild assumptions, we prove that temporal heteroscedastic causal models are identifiable for both stationary and multi-regime MTS. Across extensive comparisons with existing multi-regime causal discovery methods, we show that FANTOM consistently achieves superior performance in structure learning and regime detection. Moreover, it outperforms stationary models, even when they are provided with ground-truth regime partitions, on synthetic and two real-world datasets. The main contributions of this paper can be summarized as follows:

- We introduce FANTOM, a unified framework for causal discovery in multi-regime MTS that handles both homoscedastic non-Gaussian and heteroscedastic noises while simultaneously discovering the number of regimes, their indices, and their corresponding causal graphs.
- Under mild assumptions in causal discovery, we prove identifiability of the temporal heteroscedastic causal models in the stationary case and show that the number of regimes, their indices, and their graphs are identifiable (up to permutation) in the non-stationary setting.
- We demonstrate, via extensive experiments, that FANTOM outperforms state-of-the-art methods on both synthetic and real-world datasets.

Related work. Many works tackle causal structure learning from stationary MTS , Granger causality is the primary approach used for this purpose [32, 8]. However, it is unable to accommodate

4

5

I

N

w

6

1

instantaneous effects. DYNOTEARS [35], learns instantaneous and time lagged structures and leverages the acyclicity constraint, introduced by Zheng et al. in [60], to turn the DAG learning problem to a purely continuous optimization problem. However, DYNOTEARS is limited to linear SEMs. Runge et al. [42] proposed a two-stage algorithm PCMCI+ that can scale to large time series, PCMCI+ is able also to handle non linear relationships. Neverthless, DYNOTEARS and PCMCI+ are restricted to homoscedastic noises where variance is a constant over time. For this reason, Rhino [15] and SCOTCH [55] introduced models that tackle stationary MTS with historical noise, where the noise variance is a function of solely time lagged parents. However Rhino, DYNOTEARS and PCMCI+, cannot handle heteroscedastic noise and are limited to stationary MTS.

Several studies have sought to tackle the challenge of causal discovery in non-stationary MTS [21, 17, 45, 33]. Remarkably, Huang et al. [21] address the setting of time series composed of different regimes by modulating causal relationships through a regime index. CD-NOD detects change points and outputs a single summary causal graph, but it overlooks recurring regimes and provides neither regime-specific graphs nor their count. RPCMCI [45] provides a graph for each regime, yet it assumes prior knowledge of the number of regimes, restricts edges to time-lagged links, and offers no identifiability guarantees. Balsells-Rodas et al. [5] establish identifiability for first-order, regime-dependent causal discovery in multi-regime MTS with Gaussian noise and offer a practical algorithm, but their framework allows only a single time-lagged edge and excludes instantaneous links. Finally, CASTOR [39] jointly infers regime labels and their causal graphs, capturing both instantaneous and lagged links, under an equivariant Gaussian noise assumption. However, none of these models can simultaneously learn the number of regimes, their indices, and their structures under non-Gaussian noise, nor do they offer identifiability guarantees. FANTOM fills this gap and even generalises to richer heteroscedastic noise settings while providing identifiability results. More detailed related work is provided in Appendix A).

## 2 Problem formulation

In this Section, we introduce our notation, define a temporal causal graph, and then we present a new Structural Equation Models (SEMs) for multi-regime MTS with non-Gaussian/Heteroscedastic noise.

Notation. Matrices, vectors, and scalars are denoted by uppercase bold G τ , lowercase bold x t and lowercase letters x i t -τ , respectively. Ground-truth variables are indicated with an asterisk, such as G ∗ . We assume that all distributions have densities p ( x t ) w.r.t. the Lebesgue measure. The notation [ | 0 : L | ] represents the set of integers { 0 , ..., L } and | · | denotes set cardinality. The notation ( x t ) t ∈T = ( x i t ) i ∈ V ,t ∈T represents a MTS of | V | = d components and length |T | , T is the time index set and x &lt;t refers to { x t -L , ..., x t -1 } .

̸

Definition 2.1 (Temporal Causal Graph [39]) . The temporal causal graph, associated with the MTS ( x t ) t ∈T , is defined by a Directed Acyclic Graph (DAG) G = ( V , E ) , represented by a collection of adjacency matrices G τ ∈ [ | 0: L | ] = { G 0 , . . . , G L } , and a fixed maximum lag L . Its vertices V consist of the set of components x 1 t ′ , . . . , x d t ′ for each t ′ ∈ [ | t -L : t | ] . The edges E of the graph are defined as follows: ∀ τ ∈ [ | 1 : L | ] the variables x i t -τ and x j t are connected by a lag-specific directed link x i t -τ → x j t in G pointing forward in time if and only if x i at time t -τ causes x j at time t . Then the coefficient [ G τ ] ij associated with the adjacency matrix G τ ∈ M d ( R ) will be non-zero and x i ∈ Pa j G ( &lt; t ) ; the lagged parents of node i in G . For instantaneous links ( τ = 0 ), we cannot have self loops i.e. i = j . If τ = 0 , we have an edge x i t → x j t and x i ∈ Pa j G ( t ) ; the instantaneous parents of j at the current time t , if and only if x i at time t causes x j at time t .

In many real world scenarios, a non-stationary MTS ( x t ) t ∈T may exhibit K distinct, non-overlapping regimes, where non-stationarity is not modeled by continuous changes but rather by sequences of piecewise-constant regimes, as in climate science [27], finance [18], or epileptic recordings [56]. Every regime r is a stationary MTS block, has its own temporal causal graph G r (Definition 2.1). At each time t = 1 , 2 , . . . , |T | there is a discrete latent state z t ∈ { 1 , 2 , . . . , K } that models the regime partition, i.e., z t = r means that x t belongs to regime r and we denote I r = { t | z t = r, ∀ t ∈ T } the set of all time indices at which regime r appears. We gather these sets into I = ( I r ) r ∈ [ | 1: K | ] , yielding a unique time partition of the MTS ( x t ) t ∈T composed of K regimes. In addition, the observation x t follows, a novel and general SEM that takes into account non stationarity, and handles both

non-Gaussian case and heteroscedastic setting, that we introduce as follows, ∀ r ∈ [ | 1 : K | ] , ∀ t ∈ I r :

where f i,r and g i,r are general differentiable functions, with g i,r strictly positive, and ϵ i,r t following an arbitrary probability density. We assume E ( ϵ i,r t ) = 0 and E (( ϵ i,r t ) 2 ) = 1 without loss of generality. We denote the set of these temporal causal graphs as G = ( G r ) r ∈ [ | 1: K | ] . In the case of non-stationary MTS, regimes appear sequentially with at least a minimum duration ζ , and a subsequent regime v (where v = r +1 ) begins only after at least ζ time units from the start of regime r . Additionally, if regime r reoccurs, its duration in the second appearance is also no less than ζ samples. We refer to the phenomenon, where each regime persists for at least ζ consecutive time steps, as the regime persistent dynamics assumption and we define it as follows:

<!-- formula-not-decoded -->

Assumption 2.2. We consider a MTS with K multiple regimes ( x t ) t ∈T where the SEM is defined in Eq.(1). Given variable i , we assume that a regime r ∈ [ | 1 : K | ] is ζ -persistent if the parents ( Pa i G r ( &lt; t ) , Pa i G r ( t )) and functional dependencies ( f i,r , g i,r , ϵ i,r t ) are stationary for ζ consecutive time steps t .

The persistence assumption permits to capture different regime dynamics, whether arising from changes in the causal graph across regimes, commonly observed in climate science [27] and epileptic recordings [56], or from shifts in functional dependencies, which correspond to soft interventions in causal discovery. Our newly introduced SEM in Eq.(1) generalizes several existing approaches in three novel aspects: (1) when K = 1 , if g i, 1 ( Pa i G 1 ( &lt; t ) , Pa i G 1 ( t )) = 1 for all i ∈ [ | 1 : d | ] , we recover the classical additive noise models in causal discovery. Thus, allowing g i, 1 to be a strictly positive and differentiable function, not only extends Rhino's SEM [15] but also yields a new, general SEM for stationary multivariate time series with heteroscedastic noise . (2) When K &gt; 1 , if g i, 1 ( Pa i G r ( &lt; t ) , Pa i G r ( t )) = 1 , and ϵ i,r t ∼ N (0 , σ r ) for all i ∈ [ | 1 : d | ] , we recover the setting introduced in [39, 4]. Then, allowing ϵ i,r t to follow an arbitrary probability density then yields the first SEM for non-stationary MTS composed of multiple regimes with non-Gaussian noise . Finally, (3) permitting g i,r to be a strictly positive, differentiable function leads, to the best of our knowledge, to the first general SEM for non-stationary MTS with heteroscedastic noise .

## 3 Flow based approach for Dynamic Temporal Causal models

## 3.1 ELBO formulation

Many real-world time series e.g., EEG data [22, 40], climate data [27, 16], and financial data [18] are non-stationary and exhibit complex noise distributions. Existing causal discovery methods cannot jointly recover the number of regimes, their indices, and their structures under either non-Gaussian or heteroscedastic noises. With FANTOM, our objective is to close this gap, simultaneously learning the number of regimes K , their indices I = ( I r ) r ∈ [ | 1: K | ] , and DAGs G = ( G r ) r ∈ [ | 1: K | ] in both homoscedastic non-Gaussian and general heteroscedastic settings. Because integrating over latent regimes makes the data log-likelihood intractable, we instead maximise its Evidence Lower BOund (ELBO). Proposition 3.1, proved in Appendix G, formalises this ELBO for N w &gt; K provisional regimes. Section 3.2 outlines the initialization trick that instantiates these N w regimes, and the E-step that progressively merges them until N w settles at K .

Proposition 3.1. Let ( x t ) t ∈T be a MTS composed of multiple regimes and following the SEM described in Eq.(1). The data likelihood admits the following evidence lower bound (ELBO):

<!-- formula-not-decoded -->

where ∀ t ∈ T : z t ∈ [ | 1 : N w | ] are the discrete latent variables and N w is the number of regimes.

Here, Θ are all the learnable parameters of our model (detailed in Section 3.2), log p θ z t ( x t | x &lt;t , G z t ) represents the observational log-likelihood of x t belonging to regime z t ∈ [ | 1 , N w | ] , q ϕ r ( G r ) represents the variational distribution that approximates the intractable posterior p θ r ( G r | x t ∈I r ) , and p ( z t | x t , x &lt;t ) represents the posterior distribution of the latent variables z t . The distribution p ( G r ) is the graph prior and p ( z t ) represents our prior belief about the membership of samples to the causal models; typically we model it as a time varying function.

## 3.2 Model parametrization

In this Section, we will define and motivate FANTOM design choices. FANTOM maximises the ELBO (Eq.(2)) with a Bayesian Expectation Maximization (BEM) scheme. Because BEM normally needs the number of regimes a priori, we first describe an initialisation trick that removes this requirement. Next, we motivate our prior over the latent regime indicator z t and show how a Temporal Graph Neural Network, combined with CNFs, models the regime-specific likelihood p θ z t ( x t | x &lt;t , G z t ) .

Initialization trick. FANTOM initially divides the MTS into N w &gt; K equal time windows in the initialisation step (the length of the initialized windows is greater than ζ minimum regime duration), where each window represents one initial regime estimate. Our initialization scheme builds some initial pure regimes (regimes composed of samples from the same ground truth regime) and other impure ones (regimes composed of samples from two neighboring ground truth regimes). After initialization, FANTOM alternates between two phases. The E-step (subsection 3.4) updates the regime indices I = ( I r ) r ∈ [ | 1 : N w | ] and removes regimes with too few samples, updating N w . The M-step (subsection 3.3.1) learns the graphs ( G r ) r ∈ [ | 1 : N w | ] and models heteroscedastic noise with a Bayesian structure-learning method that uses conditional normalizing flows (CNFs). These phases repeat until the algorithm reaches the maximum number of iterations.

BEM choice motivation. We argue that inferring regimes and learning their associated DAGs are interdependent tasks, making the BEM algorithm particularly well-suited for this problem. A two-step alternative, that detects change points with KCP [1], then runs causal discovery on each segment breaks down: First, change point detection methods like KCP [1] fail to detect regime shifts driven by changes in causal mechanisms, because those involve shifts in conditional distributions (See Appendix F.3.3). It also treats recurring regimes as distinct, forcing redundant model fits and raising computation costs. Second, heteroscedastic noise further degrades existing causal methods (Table 2). FANTOM addresses all three issues by coupling CNFs, which capture heteroscedasticity, with Bayesian structure learning that prunes spurious edges.

Time varying weight modeling. We use time-varying weights, initially proposed for financial data modeling [59, 57], as priors for the discrete latent variables z t . To support smooth regime transitions consistent with our persistence assumption (Assumption 2.2), we adopt a flexible functional form based on the softmax transformation of learnable parameter ω r ∈ R 2 and time index t : p ( z t = r ) = π t ( ω r ) = exp( ω r 1 · t + ω r 0 ) ∑ Nw j =1 exp ( ω j 1 · t + ω j 0 ) . This formulation encourages that, if x t belongs to regime r in the current iteration, it is only allowed to remain in the same regime r or smoothly transition to neighboring regimes ( r -1 , r +1 ) in the next iteration. See Section 3.4 and Appendix B for details.

Bayesian structure learning. Following [14, 15, 55], FANTOM employs Bayesian structure learning. We approximate the intractable posterior p θ ( G | x t ∈T ) using the variational distribution q ϕ ( G ) = ∏ N w r =1 q ϕ r ( G r ) δ ( θ r ) , where δ denotes the Dirac delta function. Following [14, 15, 55], we model q ϕ r ( G r ) as a product of independent Bernoulli variables and compute its expectation with a single Monte Carlo sample using the Gumbel-Softmax trick [26]. Additional details are in Appendix C.

Likelihood of SEM. Using the functional form Eq.(1), we have then we can write the observational likelihood:

<!-- formula-not-decoded -->

where p hetero refers to the density function of the heteroscedastic conditions. To estimate f i,r , we build upon the model formulation of [14], which uses neural networks to describe the functional

<!-- formula-not-decoded -->

relationship f i,r θ r : R d → R . Specifically, we propose flexible functional designs for f i,r , which must respect the relations encapsulated in G r . Namely, if x j t -τ / ∈ Pa i G r ( &lt; t ) ∪ Pa i G r ( t ) , then ∂f i,r /∂x j t -τ = 0 . We design where ψ r and ϑ r are neural network blocks illustrated in Figure 2 with all the other colored blocks. Instead of using a neural network block per node, we adopt a weight-sharing mechanism by using a trainable embeddings e τ,i for τ ∈ [ | 0 : L | ] and i ∈ { 1 , · · · , d } . For the heteroscedastic term, we introduce an invertible mapping ℓ i,r θ r : R → R such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where n i,r t ∼ N (0 , 1) . The design of ℓ i,r θ r needs to properly balance the flexibility and tractability of the transformed noise density. We choose a conditional normalizing flows, for heteroscedastic noise, called conditional spline flow [10], that transforms our heteroscedastic noise distribution to a fixed normal noise n i,r t for regime r . The spline parameters are predicted using a hyper-network with a similar form as Eq.(4) to incorporate heteroscedasticity. Due to the invertibility of ℓ i,r θ r , the noise likelihood conditioned on all parents is:

Figure 2: Temporal graph neural network (TGNN) used by FANTOM.

<!-- image -->

<!-- formula-not-decoded -->

∣ ∣ where p n i,r ( · ) is the standard normal density. In the non-Gaussian, non-heteroscedastic case, FANTOM learns a base distribution based on a composite affine-spline transformation of a standard Gaussian. Finally, the system parameters comprise the learnable parameters of the time varying weights ω r , of the variational inference ϕ r and of the neural networks θ r . We use Θ , introduced in Eq.(2), to note all the learnable parameters of FANTOM, and we have the set of parameters is: Θ = { ( ω r , ϕ r , θ r ) } N w r =1 .

## 3.3 BEMalgorithm

## 3.3.1 Mstep: graph learning

FANTOMapplies BEM to maximize the ELBO described in Proposition 2. It begins by initializing the regime partitions β r t = p ( z t = r | x t , x &lt;t ) using equally sized windows, selected via a hyperparameter. Note that these binary regime indices β r t are updated during the E-step. Then, in the M-step, FANTOM incorporates β r t in the ELBO Eq.(2) to estimate the DAGs for each regime and learn the parameters ω r that align π t ( ω r ) with β r t . We have:

<!-- formula-not-decoded -->

(6)

The maximization of the above equation can be decomposed into two distinct and separate maximization problems. The first problem, regime alignment, focuses on aligning π t ( ω r ) with β r t . While the

second one, graph learning, involves estimating DAGs for every regime.

log p θ r ( x t | x &lt;t , G r ) = ∑ d i =1 log p hetero ( y i,r t | Pa i G r ( &lt; t ) , Pa i G r ( t ) ) , where p hetero and y i,r t are defined in Eq.(5). The parameters θ r , ϕ r are learned by maximizing the graph learning maximization problem Eq.(6), where the Gumbel-softmax gradient estimator is used [26]. We also leverage augmented Lagrangian training similar to [35, 39, 14], to anneal α, ρ .

For the graph prior p ( G r ) for all r ∈ [ | 1 : N w | ] have to combine two components: DAG constraint and graph sparseness prior. Inspired by [60, 15, 14], we propose the following unnormalised prior p ( G r ) ∝ exp ( -λ s ∥ G r 0: K ∥ 2 F -ρh 2 ( G r 0 ) -αh ( G r 0 ) ) . Using Eq.(3), we have

## 3.4 E step: Regime learning

In the E-step, FANTOM updates the posterior probability β r t = p ( z t = r | x t , x &lt;t ) (see Eq.(7), with derivation provided in Appendix B):

<!-- formula-not-decoded -->

where p ( x t | x &lt;t , z t = r, G r ) denotes the observational likelihood of x t being generated by the SEM from Eq (1) for regime r . This probability is computed using the CNFs trained during the M-step, following the same reasoning as in Eq.(3).

The probability of x t belonging to regime r is influenced by two main factors: the observation's position within its current regime and whether that regime is designated as pure or impure. For example, if x t is in a pure regime r but is near the boundary in the current iteration, π t ( ω r ) and π t ( ω r +1 ) are nearly equal (e.g., π t ∈ [1100 , 1500] ( ω 1 ) vs. π t ∈ [1100 , 1500] ( ω 2 ) in Figure 3). Nonetheless, since regime r was learned from pure data, p ( x t | x &lt;t , z t = r, G r ) stays high, keeping β r t at its maximum value and maintaining x t in regime r for the next iteration. In the other hand, if x t is in an impure regime r +1 near the boundary during the current iteration, π t ( ω r ) and π t ( ω r +1 ) are also close in value (e.g., π t ∈ [1501 , 1800] ( ω 1 ) vs. π t ∈ [1501 , 1800] ( ω 2 ) in Figure 3). However, because the causal graph for regime r is more reliable (having been derived from pure data), p ( x t | x &lt;t , z t = r, G r ) &gt; p ( x t | x &lt;t , z t = r +1 , G r +1 ) . As a result, x t moves from regime r +1 to r in the next iteration.

Figure 3: Illustration of π t ( ω r ) after Fantom's first iteration with equal windows of 1500 samples for an MTS of 4500 samples with two ground-truth regimes: I ∗ 1 = [ | 0 : 1999 | ] and I ∗ 2 = [ | 2000 : 4500 | ] .

<!-- image -->

For simplicity reasons, we explicit these cases from one border but the same thing happens in the other border which accelerates convergence. More details about other cases and Figures that illustrate the idea could be found in Appendix B.

After updating β r t , for each sample x t , FANTOM assigns a value of 1 to the most probable regime r (with the highest β r t ), and 0 to others. Additionally, FANTOM filters out regimes with insufficient samples (fewer than ζ , the minimum regime duration, defined as a hyper-parameter). Discarded regime samples are then reassigned to the nearest regime in terms of probability β r t in the subsequent iteration which is in general a neighboring regime ensured by the way we set up the probability β r t ∝ π t ( ω r ) p ( x t | x &lt;t , z t = r, G r ) .

## 4 Identifiability results

Identifiability is an important statistical property to ensure that the causal discovery problem is meaningful. In causal analysis, the whole point is to find out which variable causes others; if the model is not identifiable, the analysis is not possible at all. This section proves that causal discovery from multi-regime MTS is identifiable in the FANTOM framework namely, when (i) the noise is

non-Gaussian or heteroscedastic and (ii) the latent variable z t has a time-varying-parent prior. We first formalize identifiability for this setting, then state three theorems covering both stationary and multi-regime MTS under the two noise assumptions.

Definition 4.1. The conditional distribution of multi-regime MTS with a time varying prior is: p ( x t | x &lt;t ) = ∑ K r =1 π t ( ω r ) p θ r ( x t | x &lt;t , G r ) . We say this model is identifiable up to permutation and translation, if:

- For any two models with parameters ( ω r , θ r , G r ) K r =1 and (˜ ω r , ˜ θ r , ˜ G r ) ˜ K r =1 , such that for any t ∈ T : p ( x t | x &lt;t ) = ˜ p ( x t | x &lt;t ) , we have K = ˜ K and it exists a permutation σ and translation function ϱ : R 2 → R 2 such that θ r = ˜ θ σ ( r ) and ω r = ϱ (˜ ω σ ( r ) )
- -∀ r ∈ [ | 1 : K | ] the causal model ( θ r , G r ) is identifiable.

Following [15, 39, 7], we use the common assumptions of causal discovery settings (Causal Markov property H.2, stationarity H.2, minimality H.2, sufficiency H.2), see Appendix H.1 for precise statements. We present our first theoretical results, Theorem 4.2, states that for any stationary MTS, composed of K = 1 regime and following Eq.(1) with ϵ i t ∼ N (0 , 1) the ground truth solution G ∗ is uniquely identifiable, the detailed proof can be found in Appendix H.

Theorem 4.2 (Identifiability of Temporal Heteroscedastic Gaussian noise model (THGNM)) . Assume Causal Markov property, stationarity, minimality, sufficiency and let ( x t ) t ∈T be a MTS following a THGNM, ∀ t ∈ T :

where f i and g i are differentiable functions, with g i strictly positive and ϵ i t ∼ N (0 , 1) are mutually independent normal noises. The THGNM is identifiable if 1 g i is not a polynomial of degree two.

<!-- formula-not-decoded -->

Identifiability of Temporal Restricted Heteroscedastic noise model (TRHNM). We present and prove in Appendix H.3 our second identifiability results of a TRHNM (Theorem H.10), where ϵ i t can follow any arbitrary density distribution. We states the results for bivariate time series, in which we show that, if a backward model exists, a differential equation will always hold. Then inspired from Peters et al. [38], Immer et al. [25], and Strobl et al. [50], we define TRHNM and show its identifiability.

Our last theoretical result states that the mixture of identifiable temporal causal models with either non-Gaussian or heteroscedastic noises is also identifiable as defined in definition 4.1.

Theorem 4.3 (Identifiability of the mixture of identifiable temporal causal models) . Let F be a family of K identifiable temporal causal models, F = ( p θ r ( ·|· , G r )) r ∈ N ∗ that are linearly independent and let M K be the family of all K -finite mixtures of elements from F , i.e.,

Then the family M K is identifiable as defined in definition 4.1.

<!-- formula-not-decoded -->

Identifiability is important in causal discovery as shown in several papers [7, 15, 35, 5]. We extend these guarantees to FANTOM's settings by proving identifiability for stationary temporal models with heteroscedastic noise, showing that weaker assumptions suffice when the noise ϵ i t is simplified, recovering identifiability under explicit restrictions for arbitrary noise, and covering multi-regime MTS scenarios in both cases. Although convergence rates or finite-sample bounds remain elusive because the BEM objective is non-convex, our experiments demonstrate that FANTOM still converges in non-Gaussian and heteroscedastic settings. Further convergence rates or finite-data bounds are however extremely challenging due to the non-convexity of the acyclicity constraint in a BEM procedure. Yet, we empirically demonstrate, in the experiments section, that FANTOM converges in both non-Gaussian and heteroscedastic cases.

Table 1: Average SHD, F1 scores, NHD and Ratio for different models with d = 10 nodes and K = 3 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ), and Type classifies the graph as window (W) or summary (S). Inst. refers to instantaneous links, and Lag to time-lagged edges.

|           |       |      | Homoscedastic non-Gaussian noise   | Homoscedastic non-Gaussian noise   | Homoscedastic non-Gaussian noise   | Homoscedastic non-Gaussian noise   | Homoscedastic non-Gaussian noise   | Homoscedastic non-Gaussian noise   | Homoscedastic non-Gaussian noise   | Homoscedastic non-Gaussian noise   | Homoscedastic non-Gaussian noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   |
|-----------|-------|------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| Model     | Split | Type | Inst.                              | Inst.                              | Inst.                              | Lag                                | Lag                                | Lag                                |                                    |                                    | Regime                             | Inst.                   | Inst.                   | Inst.                   | Lag                     | Lag                     | Lag                     |                         |                         | Regime                  |
|           |       |      | SHD ↓                              | F1 ↑                               | NHD ↓                              | Ratio ↓                            | SHD ↓                              | F1 ↑                               | NHD ↓                              | Ratio ↓                            | Acc.                               | SHD ↓                   | F1 ↑                    | NHD ↓                   | Ratio ↓                 | SHD ↓                   | F1 NHD                  | ↓ Ratio                 | ↓                       | Acc.                    |
| PCMCI+    | ×     | W    | 17.5                               | 74.9                               | 0.02                               | 0.24                               | 14.5                               | 88.3                               | 0.01                               | 0.11                               | ×                                  | 46.1                    | 11.1                    | 0.05                    | 0.88                    | 46.0                    | ↑ 19.0                  | 0.05                    | 0.80                    | ×                       |
| Rhino     | ×     | W    | 2.50                               | 96.8                               | 0.002                              | 0.03                               | 6.00                               | 95.2                               | 0.006                              | 0.04                               | ×                                  | 44.5                    | 5.11                    | 0.06                    | 0.94                    | 53.5                    | 64.7                    | 0.07                    | 0.35                    | ×                       |
| DYNOTEARS | ×     | W    | 42.0                               | 54.4                               | 0.06                               | 0.45                               | 21.5                               | 82.1                               | 0.02                               | 0.17                               | ×                                  | 89.5                    | 31.5                    | 0.14                    | 0.68                    | 118.0                   | 37.5                    | 0.17                    | 0.61                    | ×                       |
| CASTOR    | ×     | W    | 22.0                               | 66.2                               | 0.03                               | 0.33                               | 17.0                               | 84.4                               | 0.01                               | 0.15                               | ×                                  | 104.0                   | 23.4                    | 0.19                    | 0.76                    | 133.5                   | 34.8 0.24               |                         | 0.64                    | ×                       |
| RPCMCI    | ✓     | W    | -                                  | -                                  | -                                  | -                                  | -                                  | -                                  | -                                  | -                                  | -                                  | -                       | -                       | -                       | -                       | -                       | -                       | -                       | -                       | -                       |
| CASTOR    | ✓     | W    | 47.0                               | 34.8                               | 0.05                               | 0.65                               | 59.5                               | 39.4                               | 0.07                               | 0.60                               | 51.6                               | -                       | -                       | -                       | -                       | -                       |                         | -                       | -                       | -                       |
| FANTOM    | ✓     | W    | 0.33                               | 99.5                               | 0.00                               | 0.00                               | 12.5                               | 89.1                               | 0.01                               | 0.10                               | 99.4                               | 5.67                    | 93.3                    | 0.006                   | 0.06                    | 12.3                    | - 90.9                  | 0.012                   | 0.08                    | 97.1                    |
|           |       |      | SHD ↓                              | SHD ↓                              | F1 ↑                               | F1 ↑                               | NHD ↓                              | NHD ↓                              | Ratio ↓                            | Ratio ↓                            | Acc.                               | SHD ↓                   | SHD ↓                   | F1 ↑                    | F1 ↑                    | NHD ↓                   | NHD ↓                   | Ratio ↓                 | Ratio ↓                 | Acc.                    |
| CD-NOD    | ×     | S    | 42.5                               | 42.5                               | 31.8                               | 31.8                               | 0.42                               | 0.42                               | 0.67                               | 0.67                               | ×                                  | 42                      | 42                      | 6.15                    | 6.15                    | 0.61                    | 0.61                    | 0.93                    | 0.93                    | ×                       |
| FANTOM    | ✓     | S    | 4.5                                | 4.5                                | 95.6                               | 95.6                               | 0.04                               | 0.04                               | 0.04                               | 0.04                               | 96.6                               | 4.5                     | 4.5                     | 96.5                    | 96.5                    | 0.04                    | 0.04                    | 0.03                    | 0.03                    | 96.8                    |

## 5 Experiments

## 5.1 Synthetic data

Data generation. We conduct extensive experiments to evaluate FANTOM's performance on synthetic datasets. For ground truth graph generation, we use the Barabási-Albert model (degree 4) for instantaneous links and the Erdos-Rényi model (degree 1-2) for time-lagged relationships. For data generation process, f r i , g r i are chosen to be randomly initialized MLPs with one hidden layer and activation functions randomly chosen from { Tanh, Exp } . We evaluate the different models on multiple complex noise distributions; non-stationary MTS with either (1) heteroscedastic noise or (2) non-Gaussian homoscedastic noise, details in Appendix E.1. We consider L = 1 , while additional experiments with multiple lags are provided in the Appendix F.1. Regime durations are randomly selected from { 1000 , 1500 , 2000 , 2500 } . We test different numbers of nodes { 5 , 10 , 20 , 40 } and varying regime counts ( K ∈ { 2 , 3 } ). Each combination of K and d nodes is repeated three times, resulting in over 24 distinct datasets 1 .

Benchmarks. We benchmark our model against several baselines, including causal discovery methods for MTS with multiple regimes, such as CASTOR [39], CD-NOD [21] and RPCMCI [45]. Since CD-NOD returns a summary graph (see Appendix F.1), we compute a comparable summary graph from FANTOM's output for fair evaluation. FANTOM is also compared with models for single-regime MTS, including Rhino [15], PCMCI+ [42], DYNOTEARS [35]. Given that these models cannot deal with multiple regimes, we put them in a more favorable position than ours and provide these models with the true regime partition information. This is done by training the aforementioned models on each pure regime separately (regime governed by the same causal model) .

Evaluation Metrics. We assess the performance of our proposed method for learning the DAGs using four key metrics: 1) F1 score, representing the harmonic mean of precision and recall; 2) Structural Hamming Distance (SHD), which counts discrepancies (e.g., reversed, missing, or redundant edges) between two DAGs; 3) Normalized Hamming Distance (NHD) measures how many edges differ normalized by the total number of possible edges; 4) Ratio NHD computes the ratio between the NHDand the baseline NHD of an output with the same number of edges but with all of them incorrect. For the regime detection task, we use Accuracy (Reg Acc) metric.

Results and discussion. Table 1 shows results on MTS with multiple regimes under heteroscedastic noise (right part of the table) and homoscedastic non-Gaussian noise (left part of the table). In the homoscedastic, non-Gaussian scenario , baselines generally perform better in the graph learning task, yet FANTOM still surpasses them on regime detection, with 99.4% accuracy, and instantaneous links, 99.5% F1. For the regime detection task, CASTOR succeeds to detect the regimes but with low accuracy (51.6%) compared to FANTOM (99.4%) and this due to the fact that CASTOR assumes equivariance normal noise with, however RPCMCI does not converge in this case too. For timelagged connections, Rhino (95.2% F1), that has access to the ground-truth regime labels by training it on pure regime separately, slightly outperforms FANTOM (89.1% F1) that learns simultaneously the number of regimes, their indices and structures. In the heteroscedastic setting , FANTOM consistently outperforms both multi-regime baselines (CASTOR, RPCMCI, CD-NOD) and stationary approaches. It achieves the top scores on all metrics: for instantaneous links, an F1 of 93.3%, a 60%

1 https://github.com/arahmani1/fantom.git

improvement over the second-best, and a ratio of 0.06, 0.64 lower than the next-best DYNOTEARS. For time-lagged links, an F1 of 90.9%, 25% higher than Rhino, and a ratio of 0.08. FANTOM also detects the correct number of regimes and their indices with over 96% accuracy. By contrast, RPCMCI struggles to converge due to its homoscedastic assumption and time-lag-only dependencies, CASTOR relies on Gaussian noise and cannot detect regime labels in the absence of ground truth, DYNOTEARS, PCMCI+, CD-NOD, and Rhino likewise fail in this heteroscedastic scenario, even when regime labels are given a priori. Although Rhino models history-dependent noise, it does not handle general heteroscedasticity. Overall, the table shows that FANTOM is the only method that remains robust across both noise scenarios, matching or surpassing specialised baselines while simultaneously discovering regimes and graphs.

Appendices F.2.1 and F.2.2 report additional results with standard deviations, confirming that FANTOM sustains its performance when scaled to graphs of 20 and 40 nodes. Ablation study in the Appendix F.1 further shows FANTOM's robustness towards the choice of initialized window and ζ .

## 5.2 Real world data

Wind Tunnel. We use the wind tunnel datasets from Gamella et al. [13], featuring two controllable fans pushing air through a chamber, barometers measuring air pressure at various locations, and a hatch controlling an external opening. The dataset comprises 16 variables across two regimes of 10,000 samples each: the first is observational, while the second involves soft interventions on five variables (see Appendix E.4.1). We compare FANTOM to the aforementioned baselines, with results in Table 2. FANTOM is the only model that detects

Table 2: Performance on Wind Tunnel data evaluated on summary causal graph.

|           | Split   |   SHD ↓ |   F1 ↑ | Ratio ↓   | Reg Acc.   |
|-----------|---------|---------|--------|-----------|------------|
| PCMCI+    | ×       |      37 |   22.9 | 0.77      | ×          |
| DYNOTEARS | ×       |      34 |    0   | 1         | ×          |
| CASTOR    | ×       |     104 |   17.2 | 0.82      | ×          |
| CD-NOD    | ×       |      40 |   20   | 0.80      | ×          |
| Rhino     | ×       |      47 |   32   | 0.68      | ×          |
| CASTOR    | ✓       |     120 |   19.5 | 0.80      | 49.9       |
| FANTOM    | ✓       |      29 |   38.5 | 0,61      | 99.9       |

the regime with 99.9% accuracy and outperforms all baselines on the causal graph learning task, achieving 38.5% on F1 score. Notably, FANTOM surpasses all baselines even when they are given the ground-truth regime partitions.

Epilepsy detection. Huizenga et al. [22] show that scalp potential fields are contaminated by heteroscedastic noise in EEG measurements. We evaluate FANTOM's performance in detecting epileptic regimes using EEG signals from 10 different patients in the Temple University Hospital EEG Seizure Corpus (TUSZ) dataset [40, 52]. We treat this as an unsupervised regime detection problem, analyzing roughly 100 seconds of recordings at a 250 Hz sampling rate for each patient, capturing both normal and seizure states (see Appendix E.4.2). The recordings consist of 19 electrodes, each considered a causal variable. FANTOM detects the correct regime partitions with an average 82.7% accuracy across all patients. The seizure regime's learned graph is denser and more connected than that of the normal state, which aligns with the generalized seizures affecting multiple brain regions. Full details and illustrations are provided in Appendix E.4.2.

## 6 Conclusion

We introduced FANTOM, a unified framework for multi-regime MTS that jointly infers (i) the number of regimes, (ii) their boundaries, and (iii) their corresponding causal DAG, under either non-Gaussian or heteroscedastic noises. Under mild assumptions in causal discovery, we prove identifiability of the temporal heteroscedastic causal models in the stationary case and show that the number of regimes, their indices, and their graphs are identifiable (up to permutation) in the non-stationary setting. Extensive experiments on synthetic and real-world data show consistent gains over strong baselines. FANTOM offers a principled means to uncover regime-specific causal dynamics, enhancing regime detection, and causal discovery with potential applications in various domains such as finance, climate science, and neuroscience.

## Acknowledgment

We thank Alessandro Favero, Nikolaos Dimitriadis, and Ortal Senouf for helpful feedbacks and comments. This work was supported by the SNSF Sinergia project 'PEDESITE: Personalized Detection of Epileptic Seizure in the Internet of Things (IoT) Era'

## References

- [1] Sylvain Arlot, Alain Celisse, and Zaid Harchaoui. A kernel multiple change-point algorithm via model selection. Journal of machine learning research , 2019.
- [2] Charles K Assaad, Emilie Devijver, and Eric Gaussier. Survey and evaluation of causal discovery methods for time series. Journal of Artificial Intelligence Research , 2022.
- [3] Karim Assaad, Emilie Devijver, Eric Gaussier, and Ali Ait-Bachir. A mixed noise and constraintbased approach to causal inference in time series. In ECML PKDD , 2021.
- [4] Carles Balsells-Rodas, Yixin Wang, and Yingzhen Li. On the identifiability of switching dynamical systems. arXiv preprint arXiv:2305.15925 , 2023.
- [5] Carles Balsells-Rodas, Yixin Wang, and Yingzhen Li. On the identifiability of switching dynamical systems. In Forty-first International Conference on Machine Learning , 2024.
- [6] Albert-László Barabási and Réka Albert. Emergence of scaling in random networks. Science , 1999.
- [7] Philippe Brouillard, Sébastien Lachapelle, Alexandre Lacoste, Simon Lacoste-Julien, and Alexandre Drouin. Differentiable causal discovery from interventional data. In Advances in Neural Information Processing Systems , 2020.
- [8] Bart Bussmann, Jannes Nys, and Steven Latré. Neural additive vector autoregression models for causal discovery in time series. In Discovery Science , 2021.
- [9] Yuxiao Cheng, Ziqian Wang, Tingxiong Xiao, Qin Zhong, Jinli Suo, and Kunlun He. Causaltime: Realistically generated time-series for benchmarking of causal discovery. arXiv preprint arXiv:2310.01753 , 2023.
- [10] Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows. Advances in neural information processing systems , 2019.
- [11] Doris Entner and Patrik O Hoyer. On causal discovery from time series data using fci. Probabilistic graphical models , 2010.
- [12] Nir Friedman. The bayesian structural em algorithm. arXiv preprint arXiv:1301.7373 , 2013.
- [13] Juan L Gamella, Jonas Peters, and Peter Bühlmann. Causal chambers as a real-world physical testbed for ai methodology. Nature Machine Intelligence , 7(1):107-118, 2025.
- [14] Tomas Geffner, Javier Antoran, Adam Foster, Wenbo Gong, Chao Ma, Emre Kiciman, Amit Sharma, Angus Lamb, Martin Kukla, Nick Pawlowski, et al. Deep end-to-end causal inference. arXiv preprint arXiv:2202.02195 , 2022.
- [15] Wenbo Gong, Joel Jennings, Cheng Zhang, and Nick Pawlowski. Rhino: Deep causal temporal relationship learning with history-dependent noise. Preprint arXiv:2210.14706 , 2022.
- [16] Wiebke Günther, Urmi Ninad, Jonas Wahl, and Jakob Runge. Conditional independence testing with heteroskedastic data and applications to causal discovery. Advances in Neural Information Processing Systems , 2022.
- [17] Wiebke Günther, Oana-Iuliana Popescu, Martin Rabel, Urmi Ninad, Andreas Gerhardus, and Jakob Runge. Causal discovery with endogenous context variables. Advances in Neural Information Processing Systems , pages 36243-36284, 2024.

- [18] James D Hamilton and Raul Susmel. Autoregressive conditional heteroskedasticity and changes in regime. Journal of econometrics , 1994.
- [19] Uzma Hasan, Emam Hossain, and Md Osman Gani. A survey on causal discovery methods for iid and time series data. arXiv preprint arXiv:2303.15027 , 2023.
- [20] Stefan Haufe, Klaus-Robert Müller, Guido Nolte, and Nicole Krämer. Sparse causal discovery in multivariate time series. In causality: objectives and assessment . PMLR, 2010.
- [21] Biwei Huang, Kun Zhang, Jiji Zhang, Joseph Ramsey, Ruben Sanchez-Romero, Clark Glymour, and Bernhard Schölkopf. Causal discovery from heterogeneous/nonstationary data. In Journal of Machine Learning Research , 2020.
- [22] Hilde M Huizenga and Peter CM Molenaar. Equivalent source estimation of scalp potential fields contaminated by heteroscedastic and correlated noise. Brain topography , 1995.
- [23] Antti Hyttinen, Frederick Eberhardt, and Matti Järvisalo. Constraint-based causal discovery: Conflict resolution with answer set programming. In UAI , 2014.
- [24] Aapo Hyvärinen, Kun Zhang, Shohei Shimizu, and Patrik O Hoyer. Estimation of a structural vector autoregression model using non-gaussianity. Journal of Machine Learning Research , 2010.
- [25] Alexander Immer, Christoph Schultheiss, Julia E Vogt, Bernhard Schölkopf, Peter Bühlmann, and Alexander Marx. On the identifiability and estimation of causal location-scale noise models. In International Conference on Machine Learning . PMLR, 2023.
- [26] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144 , 2016.
- [27] Soufiane Karmouche, Evgenia Galytska, Jakob Runge, Gerald A Meehl, Adam S Phillips, Katja Weigel, and Veronika Eyring. Regime-oriented causal model evaluation of atlantic-pacific teleconnections in cmip6. Earth System Dynamics , 2023.
- [28] Nan Rosemary Ke, Olexa Bilaniuk, Anirudh Goyal, Stefan Bauer, Hugo Larochelle, Bernhard Schölkopf, Michael C Mozer, Chris Pal, and Yoshua Bengio. Learning neural causal models from unknown interventions. Preprint arXiv:1910.01075 , 2019.
- [29] Ilyes Khemakhem, Ricardo Monti, Robert Leech, and Aapo Hyvarinen. Causal autoregressive flows. In International conference on artificial intelligence and statistics . PMLR, 2021.
- [30] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014.
- [31] Lars Lorch, Jonas Rothfuss, Bernhard Schölkopf, and Andreas Krause. Dibs: Differentiable bayesian structure learning. In Advances in Neural Information Processing Systems , 2021.
- [32] Sindy Löwe, David Madras, Richard Zemel, and Max Welling. Amortized causal discovery: Learning to infer causal graphs from time-series data. In Conference on Causal Learning and Reasoning , 2022.
- [33] Sarah Mameche, Lénaïg Cornanguer, Urmi Ninad, and Jilles Vreeken. Spacetime: Causal discovery from non-stationary time series. arXiv preprint arXiv:2501.10235 , 2025.
- [34] Mark Newman. Networks . Oxford university press, 2018.
- [35] Roxana Pamfil, Nisara Sriwattanaworachai, Shaan Desai, Philip Pilgerstorfer, Konstantinos Georgatzis, Paul Beaumont, and Bryon Aragam. Dynotears: Structure learning from time-series data. In International Conference on Artificial Intelligence and Statistics , 2020.
- [36] Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. Causal inference on time series using restricted structural equation models. Advances in neural information processing systems , 2013.
- [37] Jonas Peters, Dominik Janzing, and Bernhard Schölkopf. Elements of causal inference: foundations and learning algorithms . The MIT Press, 2017.

- [38] Jonas Peters, Joris M Mooij, Dominik Janzing, and Bernhard Schölkopf. Causal discovery with continuous additive noise models. The Journal of Machine Learning Research , 2014.
- [39] Abdellah Rahmani and Pascal Frossard. Causal temporal regime structure learning. In The 28th International Conference on Artificial Intelligence and Statistics , 2025.
- [40] Abdellah Rahmani, Arun Venkitaraman, and Pascal Frossard. A meta-gnn approach to personalized seizure detection and classification. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , 2023.
- [41] Jakob Runge. Causal network reconstruction from time series: From theoretical assumptions to practical estimation. Chaos: An Interdisciplinary Journal of Nonlinear Science , 2018.
- [42] Jakob Runge. Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. In Conference on Uncertainty in Artificial Intelligence . PMLR, 2020.
- [43] Jakob Runge, Peer Nowack, Marlene Kretschmer, Seth Flaxman, and Dino Sejdinovic. Detecting and quantifying causal associations in large nonlinear time series datasets. Science advances , 2019.
- [44] Karen Sachs, Omar Perez, Dana Pe'er, Douglas A Lauffenburger, and Garry P Nolan. Causal protein-signaling networks derived from multiparameter single-cell data. Science , 2005.
- [45] Elena Saggioro, Jana de Wiljes, Marlene Kretschmer, and Jakob Runge. Reconstructing regimedependent causal relationships from observational time series. Chaos: An Interdisciplinary Journal of Nonlinear Science , 2020.
- [46] Christof Seiler and Susan Holmes. Multivariate heteroscedasticity models for functional brain connectivity. Frontiers in neuroscience , 2017.
- [47] Xinpeng Shen, Sisi Ma, Prashanthi Vemuri, and Gyorgy Simon. Challenges and opportunities with causal discovery algorithms: application to alzheimer's pathophysiology. Scientific reports , 2020.
- [48] Le Song, Mladen Kolar, and Eric Xing. Time-varying dynamic bayesian networks. In Advances in Neural Information Processing Systems , 2009.
- [49] Peter Spirtes, Clark N Glymour, Richard Scheines, and David Heckerman. Causation, prediction, and search . MIT press, 2000.
- [50] Eric V Strobl and Thomas A Lasko. Identifying patient-specific root causes with the heteroscedastic noise model. Journal of Computational Science , 2023.
- [51] Xiangyu Sun, Oliver Schulte, Guiliang Liu, and Pascal Poupart. Nts-notears: Learning nonparametric dbns with prior knowledge. arXiv preprint arXiv:2109.04286 , 2021.
- [52] Siyi Tang, Jared A Dunnmon, Khaled Saab, Xuan Zhang, Qianying Huang, Florian Dubost, Daniel L Rubin, and Christopher Lee-Messer. Self-supervised graph neural networks for improved electroencephalographic seizure analysis. arXiv preprint arXiv:2104.08336 , 2021.
- [53] Sofia Triantafillou and Ioannis Tsamardinos. Constraint-based causal discovery from multiple interventions over overlapping variable sets. The Journal of Machine Learning Research , 2015.
- [54] Sumanth Varambally, Yi-An Ma, and Rose Yu. Discovering mixtures of structural causal models from time series data. arXiv preprint arXiv:2310.06312 , 2023.
- [55] Benjie Wang, Joel Jennings, and Wenbo Gong. Neural structure learning with stochastic differential equations. arXiv preprint arXiv:2311.03309 , 2023.
- [56] Xiaojia Wang, Yanchao Liu, and Chunfeng Yang. Ictal-onset localization through effective connectivity analysis based on rnn-gc with intracranial eeg signals in patients with epilepsy. Brain Informatics , 2024.
- [57] Chun S Wong and Wai K Li. On a logistic mixture autoregressive model. Biometrika , 2001.

- [58] Sidney J Yakowitz and John D Spragins. On the identifiability of finite mixtures. The Annals of Mathematical Statistics , 1968.
- [59] Shuguang Zhang, Minjing Tao, Xu-Feng Niu, and Fred Huffer. Time-varying gaussian-cauchy mixture models for financial risk management. arXiv preprint arXiv:2002.06102 , 2020.
- [60] Xun Zheng, Bryon Aragam, Pradeep K Ravikumar, and Eric P Xing. Dags with no tears: Continuous optimization for structure learning. Advances in neural information processing systems , 31, 2018.
- [61] Shengyu Zhu, Ignavier Ng, and Zhitang Chen. Causal discovery with reinforcement learning. Preprint arXiv:1906.04477 , 2019.

## Table of content

| A   | Detailed related works                                   | Detailed related works                                                                                                             | 16   |
|-----|----------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|------|
| B   | Expectation step: Derivation, intuition and illustration | Expectation step: Derivation, intuition and illustration                                                                           | 17   |
| C   | Maximization step:                                       | Maximization step:                                                                                                                 | 18   |
|     | C.1                                                      | Mathematical Derivation: Equation (2) to (6) . . . . . . . . . . . . . .                                                           | 18   |
|     | C.2                                                      | Variational Inference Details . . . . . . . . . . . . . . . . . . . . . . .                                                        | 19   |
| D   | Limitations &Risk of spurious causality                  | Limitations &Risk of spurious causality                                                                                            | 19   |
| E   | Data generation and Baselines                            | Data generation and Baselines                                                                                                      | 20   |
|     | E.1                                                      | Synthetic data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                       | 20   |
|     | E.2                                                      | Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                        | 20   |
|     | E.3                                                      | Optimization parameters . . . . . . . . . . . . . . . . . . . . . . . . .                                                          | 21   |
|     | E.4                                                      | Real world data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                        | 21   |
|     |                                                          | E.4.1 Causal Chambers data . . . . . . . . . . . . . . . . . . . . . .                                                             | 21   |
|     | E.4.2                                                    | Epilepsy data . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                | 22   |
| F   | Additional Experiments                                   | Additional Experiments                                                                                                             | 24   |
|     | F.1                                                      | Ablation studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                       | 24   |
|     | F.1.2                                                    | Robustness to the pruning step, minimum regime duration, and tialization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |      |
|     |                                                          |                                                                                                                                    | 25   |
|     | F.1.3                                                    | Robustness to data standardization . . . . . . . . . . . . . . . .                                                                 | 26   |
|     | F.2                                                      | Additional results on synthetic data . . . . . . . . . . . . . . . . . . . .                                                       | 27   |
|     | F.2.1                                                    | Heteroscedastic noise with different number of nodes and regimes                                                                   | 27   |
|     | F.2.2                                                    | Non-Gaussian noise with different number of nodes and regimes                                                                      | 29   |
|     | F.3                                                      | Additional experiments for L=2 . . . . . . . . . . . . . . . . . . . . .                                                           | 31   |
|     | F.3.1 F.3.2                                              | Illustrations of learned graphs . . . . . . . . . . . . . . . . . .                                                                | 32   |
|     |                                                          | Time complexity analysis . . . . . . . . . . . . . . . . . . . .                                                                   | 35   |
| G   | Proof of proposition                                     | Proof of proposition                                                                                                               | 37   |
|     | Proofs of our theoretical contributions                  | Proofs of our theoretical contributions                                                                                            |      |
| H   |                                                          |                                                                                                                                    | 38   |
|     | H.1 Assumptions . . .                                    | . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                          | 38   |
|     | H.2 Proof of theorem 4.2 H.3 Identifiability results in  | H.2 Proof of theorem 4.2 H.3 Identifiability results in                                                                            |      |
|     |                                                          | the case of Temporal General Heteroscedastic Noise                                                                                 | 40   |

Figure 4: Graphical model of FANTOM. Observed variables ( x t ) are in gray, while latent variables ( z t ) and parameters ( Θ ) are in white. Blue edges represent parameter-variable interactions.

<!-- image -->

## A Detailed related works

Causal structure learning from IID data. Causal structure learning has become an active area of research. hasan et al. [19] recently presented a comprehensive review of causal discovery methods for IID data and time series. For IID data, several approaches rely on conditional independence to infer causal relationships from observational data, such as the classical PC algorithm [49]. Additionally, some methods extend beyond observational data, incorporating interventional data to enhance causal inference, including COmbINE [53] and HEJ [23]. These approaches utilize data collected from controlled interventions to uncover causal relationships. A novel research direction introduced in [60] addresses the combinatorial challenges of structure learning by formulating it as a continuous constrained optimization problem, thus avoiding computationally costly combinatorial searches. Similarly, Zhu et al. [61] utilize the acyclicity constraint but employ reinforcement learning techniques for estimating directed acyclic graphs (DAGs). In contrast, Ke et al. [28] propose an approach that learns DAGs from interventional data by optimizing an unconstrained objective function. [7] provide a comprehensive analysis of continuous-constrained methods, offering a generalized framework applicable to interventional data scenarios. Another significant method, DiBS [31], estimates the full posterior distribution over Bayesian networks from limited observations, enabling quantification of uncertainty and assessment of confidence in causal discovery.

Causal structure learning from stationary MTS. The previously mentioned state-of-the-art methods primarily target independent observations rather than temporal dependencies. Assaad et al. [2] provide a comprehensive review of approaches specifically designed for causal discovery from MTS. To model causal relationships involving time dependencies, researchers frequently employ Dynamic Bayesian Networks (DBNs), which effectively capture discrete-time temporal dynamics within directed graphical frameworks. Some methods neglect contemporaneous (instantaneous) dependencies and focus exclusively on recovering time-lagged causal links [20, 48], and tsFCI [11], the latter adapting the Fast Causal Inference algorithm [49] for time series data. Runge et al. [43] introduced PCMCI, a scalable two-stage algorithm for time series, initially focusing only on time-lagged relationships. They subsequently extended it to PCMCI+ [42], enabling the identification of contemporaneous causal connections. Additionally, models addressing non-Gaussian instantaneous effects have been developed, such as VARLINGAM [24], which integrates nonGaussian instantaneous models with autoregressive components. Another significant method is Timeseries Models with Independent Noise (TiMINo) [36], which studies nonlinear and instantaneous effects using constrained SEMs. Pamfil et al. [35] recently proposed DYNOTEARS, leveraging an algebraic characterization of graph acyclicity from [60] to estimate both instantaneous and timelagged relationships from time series data. DYNOTEARS utilizes a score-based DBN learning approach optimized via an augmented Lagrangian framework, enabling causal graph inference without assumptions on the underlying topology. In contrast, methods like NBCB [3], a noisebased/constraint-based approach, aim to learn a summary causal graph directly from observational time series data, going beyond Markov equivalence constraints even in the presence of instantaneous relationships. Rhino [15] introduced the first model that tackle stationary MTS with historical noise,where they assume that the noise variance changes over time as a function of solely time lagged

Figure 5: Illustration of π t ( ω r ) after Fantom's first iteration with equal windows of 1500 samples for an MTS of 4500 samples with two ground-truth regimes: I ∗ 1 = [ | 0 : 1999 | ] and I ∗ 2 = [ | 2000 : 4500 | ] .

<!-- image -->

I 1 I 2 I 3 Figure 6: Example of initialization with N w = 3 windows, I 1 and I 3 are pure regimes while I 2 is impure (composed of samples from two ground-truth regimes I ∗ 1 and I ∗ 2 , K = 2 ).

parents. All the aforementioned methods does not handle heteroscedastic setting and also fail short in the case of MTS with multiple regimes.

1 Causal structure learning from MTS with multiple regimes. Some research have aimed to address this challenge by developing methods for causal discovery in heterogeneous data. An example of such a method is CD-NOD [21], tackles time series with various regimes. By using the time stamp IDs as a surrogate variable, CD-NOD output one summary causal graph where the parents of each variable are identified as the union of all its parents in graphs from different regimes. Then it detects the change points by using a non-stationary driving force that estimates the variability of the conditional distribution p ( x i | union parents of x i ) over the time index surrogate. While CD-NOD provides a summary graph capturing behavioral changes across regimes, it falls short in inferring individual causal graphs, also CD-NOD cannot handle either non-Gaussian or heteroscedastic noise. The overall summary graph does not effectively highlight changes between regimes. Additionally, CD-NOD detects the change points but fails to determine the regime indices, rendering it incapable of inferring the precise number of regimes. In scenarios involving recurring regimes, CD-NOD is unable to detect this crucial information. Another relevant work dealing with MTS composed of multiple regimes is RPCMCI [45]. In this paper, the model [45] learns a temporal graph for each regime. However, they focus initially on inferring only time-lagged relationships and require prior knowledge of the number of regimes and transitions between them. [5] addresses first-order regime-dependent causal discovery from MTS with multiple regimes. They proved that first-order Markov switching models with non-linear Gaussian transitions are identifiable up to permutations. Their work offers also a practical algorithms for regime-dependent causal discovery in time series data. However, its primary limitation is the assumption of solely time-lagged relationships, with the theory being restricted to a single time lag. CASTOR [39] learns regime indices and their corresponding causal graphs, including instantaneous and time-lagged relationships, under the assumption of normally distributed noise with equivariance. However, like other causal discovery methods for non-stationary MTS, it does not address non-Gaussian or heteroscedastic noise. [54] tackle a different setting in which they aim to discover a mixtures of Structural Causal Models from a datasets of MTS. They assume that they have different stationary MTS in the same dataset, one regime per MTS, and each one could be explained by one causal model in the mixture. In their case, they assume that every MTS in the dataset is stationary but the whole dataset is a mixture. In our case, we assume that we have only one non-stationary MTS and it is composed of different regimes where we do not know when the regime starts and ends, and our goal is to identify the regimes and the corresponding causal graphs.

## B Expectation step: Derivation, intuition and illustration

<!-- formula-not-decoded -->

<!-- image -->

where p ( x t | x &lt;t , z t = r, G r ) denotes the likelihood of x t being generated by the SEM from Equation (1) for regime r . This probability is computed using the normalizing flows trained during the M-step, following the same reasoning as in Eq(3).

The probability of x t belonging to regime r is influenced by two main factors: the observation's position within its current regime and whether that regime is designated as pure or impure. In order to clarify the intuition behind pure and impure regimes, Figure 6 shows an example of such case. It presents an initialization of three equal windows while the MTS is composed of two ground truth regimes presented by green color I ∗ 1 and red one I ∗ 2 . In such case, the regimes I 1 and I 3 are pure, because they are composed of samples coming from the same ground truth regime ( I ∗ 1 for the regime I 1 and I ∗ 2 for the regime I 3 ), while I 2 is an impure regime and has samples from the two ground truth regimes.

We highlight all the different cases for a sample x t either near to the border or not of regime r and also either the causal graph learned for r is on pure or impure data:

Case 1: If x t is in a pure regime r and is far from the boundary in the current iteration, π t ( ω r ) takes a high value (for example, π t ∈ [0 , 1000] ( ω 1 ) in Figure 3). Because regime r was trained on pure data, its causal graph is more accurate, leading to a high likelihood p ( x t | x &lt;t , z t = r, G r ) . Consequently, β r t ∝ π t ( ω r ) p ( x t | x &lt;t , z t = r, G r ) remains dominant, causing x t to stay in regime r at the next iteration.

Case 2: If x t is in a pure regime r but is near the boundary in the current iteration, π t ( ω r ) and π t ( α u +1 ) are nearly equal (e.g., π t ∈ [1100 , 1500] ( ω 1 ) vs. π t ∈ [1100 , 1500] ( ω 2 ) in Figure 3)). Nonetheless, since regime r was learned from pure data, p ( x t | x &lt;t , z t = r, G r ) stays high, keeping β r t at its maximum value and maintaining x t in regime r for the next iteration.

Case 3: If x t is in an impure regime r +1 near the boundary during the current iteration, π t ( ω r ) and π t ( ω r +1 ) are also close in value (e.g., π t ∈ [1501 , 1800] ( ω 1 ) vs. π t ∈ [1501 , 1800] ( ω 2 ) in Figure 3). However, because the causal graph for regime r is more reliable (having been derived from pure data), p ( x t | x &lt;t , z t = r, G r ) &gt; p ( x t | x &lt;t , z t = r +1 , G r +1 ) . As a result, x t moves from regime r +1 to r in the next iteration.

Case 4: x t belongs to impure regime r + 1 and is far from the border (e.g., t ∈ [1801 , 2500] in Figure 3). In this case, it's uncertain whether x t will switch regimes in the next iteration. However, as the pure regime r expands with each iteration, x t will eventually be near the border of regime r +1 , bringing us back to Case 3.

## C Maximization step:

## C.1 Mathematical Derivation: Equation (2) to (6)

We perform the maximization of ELBO presented in proposition 3.1, using a BEM procedure, where we alternate between E-step (updating posterior probabilities while fixing all the parameters Θ = { θ r , ϕ r , ω r } N w r =1 ) and M step (Updating the parameters while using the posteriors learned in the E-step), we can summarize the process as follows:

- In the Estep: we learn β r t = p ( z t | x t , x &lt;t , Θ old ) , where Θ old is Θ of the previous iteration.
- In the M-step: we fix the learned posterior probabilities to the values β r t and we update Θ .
- By fixing the posterior probabilities in the M-step, the entropy of these probabilities H ( p ( z t | x t , x &lt;t , Θ old )) is a constant, which allow us to discard it.

The detail derivation from Eq.(2) to Eq.(6) is the following:

<!-- formula-not-decoded -->

We replace p ( z t | x t , x &lt;t , Θ old ) by β r t , p ( z t ) by our prior choice π t ( ω r ) , and we discard the entropy of the posterior because it is a constant in these steps. Hence, we got the Eq.(6):

<!-- formula-not-decoded -->

After the maximization of Eq.(6), FANTOM update the posteriors p ( z t | x t , x &lt;t , θ old ) following Eq.(7).

## C.2 Variational Inference Details

We provide the detailed formulations for q ϕ r ( G r ) . In order to model the temporal adjacency matrices G r τ where τ ∈ [ | 1 : K | ] , we use two learnable matrices U τ , Q τ ∈ R d × d such that:

<!-- formula-not-decoded -->

For instantaneous graphs G r 0 , we used the same trick as in [15, 54], in which we employ three lower triangular learnable matrices U 0 , Q 0 , E 0 ∈ R d × d to characterise three scenarios: (1) i → j ; (2) j → i ; (3) no edge between them. For node i &gt; j :

<!-- formula-not-decoded -->

With this formulation, the instantaneous adjacency matrix is free of self-loops, eliminating any length-1 cycles.

## D Limitations &amp; Risk of spurious causality

FANTOM's performance deteriorates when a regime contains only a handful of samples or is recorded at an extremely low sampling rate. This shortcoming is not surprising, estimating a separate causal graph for each regime is intrinsically difficult in the presence of multiple regimes and heteroscedastic noise. Yet this ability to pinpoint which edges vanish or emerge from one regime to the next is what makes FANTOM valuable in domains such as healthcare and climate science, where regime shifts

carry substantive meaning. Importantly, in realistic settings where each regime offers sufficient data, for example, epileptic seizures that last several minutes at 250 Hz, FANTOM delivers strong results and provides insights unattainable with existing methods.

FANTOM requires a suitable initial segmentation to effectively learn the regime indices. Our ablation studies demonstrate that selecting a reasonable initial window size establishes a basis for accurate regime detection, which is crucial for achieving high performance. However, we highlight a critical limitation regarding the subsequent pruning step: if the initial window is too high (over-pruning), it can drastically reduce the number of initial regimes below the ground truth. This initialization leads to a loss of essential information and a significant deterioration of FANTOM's final performance.

Spurious causality is a practical risk, so we advocate using the learned graphs as decision support rather than as ground truth. Before acting on them, especially in medical or financial settings, practitioners should adopt safeguards: (i) expert vetting of proposed edges and mechanisms; (ii) robustness checks (e.g., stability under resampling or perturbations, alternative specifications); and (iii) interventional validation when feasible.

## E Data generation and Baselines

## E.1 Synthetic data

We employ the Erdos-Rényi (ER) [34] model with mean degrees of 1 or 2 to generate lagged graphs, and the Barabasi-Albert (BA) [6] model with mean degrees 4 for instantaneous graphs. The maximum number of lags, L , is set at 1. We experiment with varying numbers of nodes { 10 , 20 , 40 } and different numbers of regimes { 2 , 3 } , each representing diverse causal graphs or mixing functions. The length of each regime is randomly sampled from the set { 1000 , 1500 , 2000 , 2500 , 3000 } .

- Heteroscedastic case. In heteroscedastic settings, noise variance shifts across both variables and observations, making the underlying DAG much harder to recover from data. Given a random set if directed acyclic graphs G = ( G r ) r ∈ [ | 1: K | ] , we generate observations from the SEMs in Eq 1 as follows:

where f i,r , g i,r are chosen to be randomly initialized MLPs with one hidden layer of size number of nodes and tanh activation functions. ϵ i,r t follows either a normal distribution N (0 , 1) or a more complex one obtained by transforming samples from a standard Gaussian with an MLP with random weights and sin activation function.

<!-- formula-not-decoded -->

- Homoscedastic non-Gaussian case. The formulation used to generated the data is:

where f i,r is a general differentiable non-linear function. The function f i,r is a random combination between a linear transformation and a randomly chosen function from the set: { Tanh, Exp } . ϵ i,r t follows either a Triangular distribution or a more complex one obtained by transforming samples from a standard Gaussian with an MLP with random weights and sin activation function.

<!-- formula-not-decoded -->

## E.2 Baselines

DYNOTEARS [35]. DYNOTEARS formulates causal discovery for multivariate time series through a linear vector autoregressive (V AR) model that simultaneously captures lagged and instantaneous causal effects. Its key innovation is the DAGness penalty a smooth, continuously differentiable relaxation of the acyclicity constraint optimized via an augmented Lagrangian scheme alongside a mean squared error loss. DYNOTEARS emerges as the special case of FANTOM obtained by setting K = 1 , using linear component functions f i, 1 , fixing the noise scaling to g i, 1 = 1 and ϵ i, 1 t ∼ N (0 , 1) in Eq(1). For comparing with this model, we use publicly available package causalnex 2 .

PCMCI+ [42]. a scalable two-stage algorithm for time series, enabling the identification of contemporaneous causal connection. As DYNOTEARS, PCMCI+ is a special case of FANTOM, obtained

2 https://causalnex.readthedocs.io/en/latest/

by setting K = 1 , using linear or non linear component functions f i, 1 , fixing the noise scaling to g i, 1 = 1 and allowing ϵ i, 1 t to follow any distribution. For the comparison, we use publicly available package Tigramite 3 .

Rhino [15]. Gong et al. propose the first structural equation models with historically dependent noise, where noise variance depends solely on time-lagged variables, the Rhino's SEM is as follow:

where ϵ i t ∼ N (0 , 1) . Rhino neglects heteroscedasticity, and assumes a single stationary regime governed by a one causal graph. By our SEM proposed in Eq(1), we can recover the Rhino's SEM by setting K = 1 and making g i, 1 a function of only time lagged parents. In Rhino, they took a non linear transformation of normal noise which is equivalent in our case to allowing ϵ i, 1 t to follow any distribution. To compare with Rhino, we used the open package causica 4 .

<!-- formula-not-decoded -->

RPCMCI [45]. RPCMCI learns regime indices and time lagged causal relationships from multiregime MTS. FANTOM's SEM in Eq(1) generalize it. We can recover RPCMCI settings by making f i,r depends only on time lagged relations and g i,r = 1 . For the comparison, we use publicly available package Tigramite 5 .

CASTOR [39]. CASTOR learns number of regimes their indices and also their corresponding DAGs including instantaneous and time lagged causal relationships from multi-regime MTS. But they assume that they only have gaussian noise with equivariance. FANTOM's SEM in Eq(1) generalize it. We can recover CASTOR settings by making g i,r = 1 and ϵ i,r t ∼ N (0 , 1) . For the comparison, we use publicly available code CASTOR 6 .

## E.3 Optimization parameters

Heteroscedastic settings. Unless noted (i.e., in the synthetic-data study), we set the model lag to the true value of 1 and allow FANTOM to capture instantaneous effects. The variational posterior q ϕ r ( G r ) is initialized to prefer sparse graphs (edge probability &lt; 0 . 5 ). Heteroscedastic noise is modeled with conditional normalizing flows (CNFs). Every neural block is a two-layer MLP with 32 hidden units, residual connections, and layer normalization.

Gradients for discrete edges are estimated via the Gumbel-Softmax trick, using a hard forward pass and a soft backward pass with temperature 0.25. All spline flows employ 128 bins, and each transformation uses an embedding dimension equal to the number of nodes.

The sparsity penalty is fixed at λ s = 50 . For graphs with 10 or 20 nodes we use ρ = 1 and α = 0 , whereas for 40 nodes we set ρ = 0 . 001 . Models are optimized with Adam [30] at a learning rate of 0.005. We establish ζ = 900 as the minimum regime duration, and we use 1000 as initial window size.

Homoscedastic non-Gaussian settings. Unless noted (i.e., in the synthetic-data study), we set the model lag to the true value of 1 and allow FANTOM to capture instantaneous effects. We use the same parameters as the heteroscedastic settings. The main difference is the use of a composite of affine-spline transformation. All spline flows employ 16 bins, and each transformation uses an embedding dimension equal to the number of nodes.

The sparsity penalty is fixed at λ s = 5 . For graphs with 10 or 20 nodes we use ρ = 1 and α = 0 , whereas for 40 nodes we set ρ = 0 . 001 . Models are optimized with Adam [30] at a learning rate of 0.005.

## E.4 Real world data

## E.4.1 Causal Chambers data

We use the wind tunnel datasets from Gamella et al. [13], featuring two controllable fans pushing air through a chamber, barometers measuring air pressure at various locations, and a hatch controlling an

3 https://jakobrunge.github.io/tigramite/

4 https://github.com/microsoft/causica/blob/main/README.md

5 https://jakobrunge.github.io/tigramite/

6 https://github.com/arahmani1/CASTOR

Figure 7: Figure taken from Gamella et al. [13]. Diagrams of the wind tunel causal chamber and its main components, including the amplification circuit that drives the speaker of the wind tunnel. The variables measured by the chamber are displayed in black math print. Sensor measurements are denoted by a tilde. Manipulable variables, that is, actuators and sensor parameters, are shown in bold symbols.

<!-- image -->

external opening. The tunnel is a chamber with two controllable fans that push air through it and barometers that measure air pressure at different locations. A hatch precisely controls the area of an additional opening to the outside (see Figure 7). The dataset comprises 16 variables: controllable load of the two fans L in , L out , their measurable speed (˜ ω in , ¯ ω out ) , the current draw by the fans ( ˜ C in , ˜ C out ) ( ˜ C in , ˜ C out ), the resulting air pressure inside the chamber ( ˜ P dw , ˜ P up ) or at its intake ( ˜ P int ), and the hatch H . In the circuit that drives the speaker, we can manipulate the potentiometers ( A 1 , A 2 ) that control the amplification, monitoring the resulting signal at different points of the circuit ( ˜ S 1 , ˜ S 2 ) and through the microphone output ( ˜ M ) . ˜ ˜ P amb is the ambient pressure measure by the outer barometer.

We evaluate all the models on two regimes of 10,000 samples each: the first is observational, while the second involves soft interventions on five variables:

- T out , the resolution of the tachometer timer that measures the elapsed time between successive revolutions of the fan. Choosing microseconds yields a higher resolution in the fan-speed measurement. Hence intervention on T out yields to a change on ¯ ω out .
- O up the oversampling rates when taking measurements of the current ( ˜ C in , ˜ C out ), amplifier ( ˜ S 1 , ˜ S 2 ) and microphone signals ( M ) , and of air pressure at the different barometers ( ˜ P up , ˜ P dw , ˜ P amb , ˜ P int ) .
- R in , R out , R 2 the reference voltages, in volts, of the sensors used to measure the current ( ˜ C in , ˜ C out ), and amplifier ( ˜ S 2 ), respectively.

For the training procedure, we start by a window of 6000 samples, which gives us three different initial regimes then FANTOM converges smoothly to the exact number of regimes K = 2 . We set our lag to 8 time lagged and we allow the presence of instantaneous parents. Regarding the parameters, we use a sparsity coefficient equal to 50, spline of 8 bins and MLPs of size 32.

We compare FANTOM to the baselines mentioned in the main text, with results in Table 2. FANTOM is the only model that detects the regime with 99.9% accuracy and outperforms all baselines on the graph learning task, achieving 38.5% on F1 score. Notably, FANTOM surpasses all the models tailored for stationary MTS, even when they are given the ground-truth regime partitions.

## E.4.2 Epilepsy data

Huizenga et al. [22] show that scalp potential fields are contaminated by heteroscedastic noise in EEG measurements. We evaluate FANTOM's performance in detecting epileptic regimes using EEG

<!-- image -->

Time (s)

Figure 8: EEG signals for patient of id 7170, session 1 in the TUSZ data.

signals from 10 different patients in the Temple University Hospital EEG Seizure Corpus (TUSZ) dataset [52]. The dataset encompasses multiple seizure types; in this study, we focus on generalized seizures, which engage the entire brain. Each patient's record contains scalp EEG signals from 19 channels (Figure 8), each considered a causal variable.

State-of-the-art methods [52] use graph neural networks (GNNs) for seizure detection, typically building a fixed distance graph and feeding Fast Fourier Transform (FFT) coefficients from 10-12 seconds EEG windows as node features. These approaches (i) train a single model for all patients, offering no personalization; (ii) cannot operate in zero-shot or unsupervised settings; and (iii) reuse an identical graph for both seizure and normal periods. FANTOM addresses these limitations: it detects seizures in a personalized manner without any training data and learns distinct temporal causal graphs for normal and seizure states.

Preprocessing. We apply FANTOM to 10 different patients from TUSZ dataset, we treat this as an unsupervised regime detection problem, analyzing roughly 100 seconds of recordings at a 250 Hz sampling rate for each patient, capturing both normal and seizure states. Before running FANTOM, we filter out the EEG signals by a band pass filter of order 6, the lower frequency is 0.5Hz while the highest frequency is 50Hz.

Table 3: Seizure detection accuracy using FANTOM for 10 different patients

| Patient id   | Regime Acc     |
|--------------|----------------|
| 0002         | 84.8           |
| 0021         | 80.8           |
| 0302         | 76.6           |
| 0492         | 86.6           |
| 6440         | 86.1           |
| 6520         | 80.2           |
| 7128         | 94.1           |
| 7170         | 81.1           |
| 7936         | 82.0           |
| 8303         | 75.1           |
| Avg          | 82 . 7 ± 5 . 2 |

Figure 9: Figure illustrates the summary causal graph per regime learned by FANTOM for different patients

<!-- image -->

We segment each EEG recording into fixed 12 s windows (3000 samples). FANTOM is initialized with eight initial regimes and reliably converges to the two ground-truth states; quantitative results appear in Table 3. We employ a temporal lag of eight samples and allow instantaneous parental links. Averaged over all patients, FANTOM achieves 82.7% regime-assignment accuracy. The graph learned for the seizure state is substantially denser and more interconnected than that of the normal state (Figure 9), consistent with the widespread neural involvement of generalized seizures.

## F Additional Experiments

## F.1 Ablation studies

## F.1.1 Impact of Heteroscedastic Modeling and Flow Architecture

To assess the contribution of heteroscedastic modelling and the flow architecture, we ran an ablation on MTS comparing three FANTOM variants: (i) Gaussian output (no flow, FANTOM-Gauss), (ii) a simple coupling-spline flow (FANTOM-Spline), and (iii) the full model with a conditional normalizing flow (CNF, FANTOM). In the single-regime setting, removing or simplifying the CNF consistently degrades performance. In the multi-regime setting, only the CNF variant remains stable and converges to the two ground truth regimes; the alternatives collapse to one regime. Results are reported in the table below:

Table 4: Importance of heteroscedastic modelling and flow architecture for 10 nodes. - means the method collapses to 1 regime.

| Heteroscedastic noise   | Heteroscedastic noise             | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   |
|-------------------------|-----------------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| K                       | model                             | Inst                    | Inst                    | Lag                     | Lag                     | Regime                  |
|                         |                                   | SHD                     | F1                      | SHD                     | F1                      | Accuracy                |
| 1                       | FANTOM-Gauss FANTOM-Spline FANTOM | 39 17 0                 | 0.0 0.0 100             | 44 18 0                 | 42.5 0 100              | - - -                   |
| 2                       | FANTOM-Gauss FANTOM-Spline FANTOM | - - 1                   | - - 97.8                | - - 1                   | - - 98.9                | - - 98.6                |

## F.1.2 Robustness to the pruning step, minimum regime duration, and regime initialization

We evaluated robustness to the pruning step, minimum regime duration ( ζ ), and regime initialization. Across these variations, FANTOM's predictive performance remains stable. We find that changing (window size, ζ ) primarily affects runtime, especially at long horizons, without materially impacting graph and regime learning performances. To show efficiency under long-horizon conditions, we used regimes with lengths [2000, 3000, 2000, 2500].

Table 5: Performance of FANTOM with varying hyperparameters (window size and ζ ) on 10 node graphs with 2, 3, and 4 regimes.

| Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   |
|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| (window, ζ )            | K                       | Numb                    | Running                 | Inst.                   | Inst.                   | Lag                     | Lag                     | Regime                  |
| (window, ζ )            | K                       | Numb                    | Running                 | SHD                     | F1                      | SHD                     | F1                      | Acc.                    |
| (1000,900)              | 2                       | 4                       | 17'8s                   | 5                       | 91.2                    | 13                      | 85.7                    | 96.5                    |
| (1500,1200)             | 2                       | 4                       | 10'12s                  | 1                       | 97.9                    | 9                       | 89.8                    | 95.1                    |
| (1000,900)              | 3                       | 4                       | 30'33s                  | 5                       | 94.2                    | 8                       | 94.4                    | 96.4                    |
| (1500,1200)             | 3                       | 4                       | 16'2s                   | 3                       | 96.7                    | 10                      | 91.9                    | 91.8                    |
| (1000,900)              | 4                       | 4                       | 40'2s                   | 5                       | 95.8                    | 12                      | 92.7                    | 92.9                    |
| (1500,1200)             | 4                       | 4                       | 25'38s                  | 3                       | 97.5                    | 16                      | 90.5                    | 91.5                    |

Table 6: Performance of FANTOM with varying hyperparameters (window size and ζ ) on 40 node graphs with 2, 3, and 4 regimes.

| Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   |
|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| (window, ζ )            | K                       | Numb                    | Running                 | Inst.                   | Inst.                   | Lag                     | Lag                     | Regime                  |
| (window, ζ )            | K                       | Numb                    | Running                 | SHD                     | F1                      | SHD                     | F1                      | Acc.                    |
| (1000,900)              | 2                       | 4                       | 22'15s                  | 33                      | 81.9                    | 27                      | 91.9                    | 100                     |
| (1500,1200)             | 2                       | 4                       | 10'20s                  | 35                      | 83.8                    | 25                      | 91.7                    | 100                     |
| (1000,900)              | 3                       | 4                       | 34'25s                  | 49                      | 85.6                    | 58                      | 88.6                    | 99.8                    |
| (1500,1200)             | 3                       | 4                       | 15'52s                  | 43                      | 87.3                    | 72                      | 87.3                    | 99.9                    |
| (1000,900)              | 4                       | 5                       | 66'30s                  | 34                      | 86.5                    | 53                      | 91.9                    | 99.7                    |
| (1500,1200)             | 4                       | 4                       | 21'38s                  | 32                      | 87.1                    | 54                      | 91.8                    | 99.6                    |

From the table, FANTOM scales to 40 nodes and converges to four regimes under different initializations. With a window initialization of 1500 and ζ = 1200 , it starts with six initial regimes and converges smoothly to four with 99.6% accuracy in 21min38s. With a window initialization of

1000 and ζ = 900 , it starts with nine initial regimes and converges to four with 99.7% accuracy in 66min30s. All rebuttal experiments were run on a Tesla V100-SXM2 (26GB). These results indicate that FANTOM scales without requiring large GPU memory.

To test scalability, we ran FANTOM on a 40 node dataset with four regimes and long horizons (lengths 2000, 3000, 2000, 2500). As expected, additional regimes and longer horizons increase runtime. With 40 nodes and two regimes, training finishes in 10 min 20 s, whereas the four regime setting requires 21 min 38 s.

We showed in the table above (Table 5 and 6) that FANTOM is robust to the initialization parameter (window size that impacts directly the initial regime count and the pruning threshold), e.g. in the case of 40 nodes with 4 regimes, with a window initialization of 1500 and ζ = 1200 , it starts with six initial regimes and converges smoothly to four in 21min38s. With a window initialization of 1000 and ζ = 900 , it starts with nine initial regimes and converges to four with 99.6% accuracy in 66 min 30s.

## F.1.3 Robustness to data standardization

We investigated FANTOM's robustness to data standardization, a common challenge for most optimization-based causal discovery models. The results in Table 7 clearly demonstrate FANTOM's resilience. This robustness stems from its use of Bayesian structure learning to estimate a distribution over plausible graphs and conditional normalizing flows to effectively model complex data distributions. In contrast, models such as CASTOR are sensitive to standardization and even fail when provided with ground-truth regime labels.

Table 7: Robustness of FANTOM to data standardization (10 nodes).

| Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   |
|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| model                   | K                       | data                    | Inst.                   | Inst.                   | Lag                     | Lag                     | Regime                  |
|                         |                         |                         | SHD                     | F1                      | SHD                     | F1                      | Acc.                    |
| CASTOR                  | 2                       | Raw standardized        | 90 28                   | 26.9 0.0                | 28 33                   | 29.7                    | x x                     |
| FANTOM                  | 2                       | Raw standardized        | 4 5                     | 93.3 91.5               | 7 7                     | 0.0 90.9 90.9           | 97.1 91.1               |
| CASTOR                  | 3                       | Raw standardized        | 135 37                  | 24.1 0.0                | 135 48                  | 29.6 0.0                | x x                     |
| FANTOM                  | 3                       | Raw standardized        | 3 2                     | 96.1 97.3               | 10 9                    | 91.5 92.3               | 92.2 92.2               |

From the Table 7, FANTOM achieves the same results on raw and standardized data while CASTOR fails in both cases even when we give it access to the ground truth regime labels. In the case of standardized data, CASTOR predicts adjacency matrices full of zeros due to its incapability of handling such scenarios.

## F.2 Additional results on synthetic data

## F.2.1 Heteroscedastic noise with different number of nodes and regimes

Table 8: Average SHD, F1 scores, NHD and Ratio for different models with d = 10 nodes and K = 2 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

| Heteroscedastic noise, K = 2 and d = 10   | Heteroscedastic noise, K = 2 and d = 10   | Heteroscedastic noise, K = 2 and d = 10   | Heteroscedastic noise, K = 2 and d = 10   | Heteroscedastic noise, K = 2 and d = 10   | Heteroscedastic noise, K = 2 and d = 10   | Heteroscedastic noise, K = 2 and d = 10   | Heteroscedastic noise, K = 2 and d = 10   | Heteroscedastic noise, K = 2 and d = 10   | Heteroscedastic noise, K = 2 and d = 10   | Heteroscedastic noise, K = 2 and d = 10   |
|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| Model                                     | Split                                     | Inst.                                     | Inst.                                     | Inst.                                     | Inst.                                     | Lag                                       | Lag                                       | Lag                                       | Lag                                       | Regime                                    |
|                                           |                                           | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | Acc.                                      |
| PCMCI+                                    | ×                                         | 34.3 ± 3 . 7                              | 14.2 ± 2 . 5                              | 0.09 ± 0 . 0                              | 0.85 ± 0 . 1                              | 26.0 ± 4 . 3                              | 25.6 ± 4 . 8                              | 0.07 ± 0 . 0                              | 0.74 ± 0 . 0                              | ×                                         |
| Rhino                                     | ×                                         | 28.0 ± 5 . 6                              | 8.56 ± 4 . 9                              | 0.09 ± 0 . 0                              | 0.92 ± 0 . 0                              | 38.5 ± 6 . 3                              | 58.4 ± 11 . 7                             | 0.13 ± 0 . 0                              | 0.43 ± 0 . 0                              | ×                                         |
| DYNOTEARS                                 | ×                                         | 58.5 ± 4 . 9                              | 31.2 ± 2 . 8                              | 0.22 ± 0 . 0                              | 0.68 ± 0 . 0                              | 79.5 ± 3 . 5                              | 33.9 ± 4 . 4                              | 0.27 ± 0 . 0                              | 0.66 ± 0 . 0                              | ×                                         |
| CASTOR                                    | ×                                         | 90.0 ± 0 . 0                              | 28.1 ± 2 . 9                              | 0.38 ± 0 . 0                              | 0.74 ± 0 . 0                              | 90.5 ± 3 . 2                              | 35.6 ± 3 . 4                              | 0.42 ± 0 . 0                              | 0.64 ± 0 . 0                              | ×                                         |
| RPCMCI                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| CASTOR                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| FANTOM                                    | ✓                                         | 4.67 1 . 5                                | 91.7 3 . 2                                | 0.005 0 . 0                               | 0.04 0 . 0                                | 11.6 1 . 1                                | 88.2 2 . 1                                | 0.02 0 . 0                                | 0.08 0 . 0                                | 96.6 1 . 1                                |

±

±

±

±

±

±

±

±

±

Table 9: Average SHD, F1 scores, NHD and Ratio for different models with d = 10 nodes and K = 3 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

| Heteroscedastic noise, K = 3 and d = 10   | Heteroscedastic noise, K = 3 and d = 10   | Heteroscedastic noise, K = 3 and d = 10   | Heteroscedastic noise, K = 3 and d = 10   | Heteroscedastic noise, K = 3 and d = 10   | Heteroscedastic noise, K = 3 and d = 10   | Heteroscedastic noise, K = 3 and d = 10   | Heteroscedastic noise, K = 3 and d = 10   | Heteroscedastic noise, K = 3 and d = 10   | Heteroscedastic noise, K = 3 and d = 10   | Heteroscedastic noise, K = 3 and d = 10   |
|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| Model                                     | Split                                     | Inst.                                     | Inst.                                     | Inst.                                     | Inst.                                     | Lag                                       | Lag                                       | Lag                                       | Lag                                       | Regime                                    |
|                                           |                                           | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | Acc.                                      |
| PCMCI+                                    | ×                                         | 46.1 ± 4 . 7                              | 11.1 ± 2 . 1                              | 0.05 ± 0 . 0                              | 0.88 ± 0 . 0                              | 46.0 ± 0 . 8                              | 19.0 ± 3 . 3                              | 0.05 ± 0 . 0                              | 0.80 ± 0 . 0                              | ×                                         |
| Rhino                                     | ×                                         | 44.5 ± 6 . 5                              | 5.11 ± 1 . 9                              | 0.06 ± 0 .                                | 0.94 ± 0 . 0                              | 53.5 ± 1 . 5                              | 64.7 ± 4 . 6                              | 0.07 ± 0 . 0                              | 0.35 ± 0 . 0                              | ×                                         |
| DYNOTEARS                                 | ×                                         | 89.5 ± 3 . 5                              | 31.5 ± 0 . 4                              | 0 0.14 ± 0 . 0                            | 0.68 ± 0 . 0                              | 118.0 ± 4 . 0                             | 37.5 ± 1 . 2                              | 0.17 ± 0 . 0                              | 0.61 ± 0 . 0                              | ×                                         |
| CASTOR                                    | ×                                         | 104. ± 3 . 7                              | 23.4 ± 0 . 9                              | 0.19 ± 0 . 0                              | 0.76 ± 0 . 0                              | 133.5 ± 2 . 8                             | 34.8 ± 1 . 2                              | 0.24 ± 0 . 0                              | 0.64 ± 0 . 0                              | ×                                         |
| RPCMCI                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| CASTOR                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| FANTOM                                    | ✓                                         | 5.67 3 . 3                                | 93.3 4 . 2                                | 0.006 0 . 0                               | 0.06 0 . 0                                | 12.3 6 . 3                                | 90.9 1 . 0                                | 0.012 0 . 0                               | 0.08 0 . 0                                | 97.1 0 . 0                                |

±

±

±

±

±

±

±

±

±

Under heteroscedastic conditions, d = 10 node graphs with either two or three regimes, FANTOM outperforms every baseline. Among methods explicitly designed for multi-regime MTS, it is the only one that converges to the true regime partitions, attaining 97.1% (Table 9) in regime detection accuracy. CASTOR and RPCMCI break down in heteroscedastic settings.

To give stationary MTS baselines the best possible chance, we supply them with the ground truth regime partitions. This is done by training the aforementioned models on each pure regime separately (regime governed by the same causal model). Yet, FANTOM still dominates the structure learning task, while learning the number of regime and their indices as well, achieving an F1 of 93.3 % and an NHD of 0.006 (Table 9). Because NHD penalizes every missing, extra, or mis-oriented edge, a value of 0.006 implies that FANTOM not only recovers the graph skeleton but orients edges with high precision.

Table 10: Average SHD, F1 scores, NHD and Ratio for different models with d = 20 nodes and K = 2 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

| Heteroscedastic noise, K = 2 and d = 20   | Heteroscedastic noise, K = 2 and d = 20   | Heteroscedastic noise, K = 2 and d = 20   | Heteroscedastic noise, K = 2 and d = 20   | Heteroscedastic noise, K = 2 and d = 20   | Heteroscedastic noise, K = 2 and d = 20   | Heteroscedastic noise, K = 2 and d = 20   | Heteroscedastic noise, K = 2 and d = 20   | Heteroscedastic noise, K = 2 and d = 20   | Heteroscedastic noise, K = 2 and d = 20   | Heteroscedastic noise, K = 2 and d = 20   |
|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| Model                                     | Split                                     | Inst.                                     | Inst.                                     | Inst.                                     | Inst.                                     | Lag                                       | Lag                                       | Lag                                       | Lag                                       | Regime                                    |
|                                           |                                           | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | Acc.                                      |
| PCMCI+                                    | ×                                         | 84.0 ± 17 .                               | 39.7 ± 2 . 6                              | 0.03 ± 0 . 0                              | 0.6 ± 0 . 0                               | 42.5 ± 12 .                               | 76.9 ± 1 . 0                              | 0.14 ± 0 . 0                              | 0.22 ± 0 . 0                              | ×                                         |
| Rhino                                     | ×                                         | 70.0 ± 6 . 0                              | 1.26 ± 1 . 2                              | 0.04 ± 0 . 0                              | 0.98 ± 0 . 0                              | 188.0 ± 30 .                              | 26.3 ± 9 . 8                              | 0.14 ± 0 . 0                              | 0.73 ± 0 . 0                              | ×                                         |
| DYNOTEARS                                 | ×                                         | 221.5 ± 9 . 5                             | 26.9 ± 1 . 2                              | 0.22 ± 0 . 0                              | 0.72 ± 0 . 0                              | 45.0 ± 7 . 0                              | 61.5 ± 3 . 9                              | 0.02 ± 0 . 0                              | 0.38 ± 0 . 0                              | ×                                         |
| CASTOR                                    | ×                                         | 377.5 ± 1 . 5                             | 14.9 ± 0 . 1                              | 0.41 ± 0 . 0                              | 0.84 ± 0 . 0                              | 379.5 ± 1 . 5                             | 19.1 ± 1 . 0                              | 0.41 ± 0 . 0                              | 0.80 ± 0 . 0                              | ×                                         |
| RPCMCI                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| CASTOR                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| FANTOM                                    | ✓                                         | 9.00 3 . 0                                | 89.1 5 . 7                                | 0.006 0 . 0                               | 0.10 0 . 0                                | 26.0 5 . 0                                | 85.4 1 . 3                                | 0.01 0 . 0                                | 0.14 0 . 0                                | 97.8 0 . 2                                |

±

±

±

±

±

±

±

±

±

Table 11: Average SHD, F1 scores, NHD and Ratio for different models with d = 20 nodes and K = 3 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

| Heteroscedastic noise, K = 3 and d = 20   | Heteroscedastic noise, K = 3 and d = 20   | Heteroscedastic noise, K = 3 and d = 20   | Heteroscedastic noise, K = 3 and d = 20   | Heteroscedastic noise, K = 3 and d = 20   | Heteroscedastic noise, K = 3 and d = 20   | Heteroscedastic noise, K = 3 and d = 20   | Heteroscedastic noise, K = 3 and d = 20   | Heteroscedastic noise, K = 3 and d = 20   | Heteroscedastic noise, K = 3 and d = 20   | Heteroscedastic noise, K = 3 and d = 20   |
|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| Model                                     | Split                                     | Inst.                                     | Inst.                                     | Inst.                                     | Inst.                                     | Lag                                       | Lag                                       | Lag                                       | Lag                                       | Regime                                    |
|                                           |                                           | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | Acc.                                      |
| PCMCI+                                    | ×                                         | 83.5 ± 15 .                               | 39.9 ± 1 . 6                              | 0.03 ± 0 . 0                              | 0.59 ± 0 . 0 0.98 ± 0 . 0                 | 38.0 ± 2 . 0 292.0 ±                      | 77.8 ± 0 . 9 25.8 8 . 9                   | 0.01 ± 0 . 0                              | 0.22 ± 0 . 0                              | × ×                                       |
| Rhino                                     | ×                                         | 108.5 ± 11 .                              | 1.6 ± 1 . 2                               | 0.02 ± 0 . 0                              |                                           | 62 .                                      | ±                                         | 0.09 ± 0 . 0                              | 0.74 ± 0 . 0                              |                                           |
| DYNOTEARS                                 | ×                                         | 325. ± 14 .                               | 27.6 ± 1 . 0                              | 0.14 ± 0 . 0                              | 0.72 ± 0 . 0                              | 55.5 ± 0 . 5                              | 69.8 ± 7 . 6                              | 0.01 ± 0 . 0                              | 0.29 ± 0 . 0                              | ×                                         |
| CASTOR                                    | ×                                         | 412.3 ± 2 . 5                             | 13.8 ± 0 . 0                              | 0.20 ± 0 . 0                              | 0.86 ± 0 . 0                              | 570. ± 2 . 0                              | 17.5 ± 0 . 4                              | 0.28 ± 0 . 0                              | 0.81 ± 0 . 0                              | ×                                         |
| RPCMCI                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| CASTOR                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| FANTOM                                    | ✓                                         | 13.5 6 . 5                                | 88.2 6 . 2                                | 0.05 0 . 0                                | 0.11 0 . 0                                | 46.5 9 . 5                                | 84.6 2 . 5                                | 0.01 0 . 0                                | 0.14 0 . 0                                | 97.8 1 . 4                                |

±

±

±

±

±

±

±

±

±

Under heteroscedastic conditions, d = 20 node graphs with either two or three regimes, FANTOM outperforms all the baselines. Among methods explicitly designed for multi-regime MTS, it is the only one that converges to the true regime partitions, attaining 97.8% in regime detection accuracy (Table 10). CASTOR and RPCMCI break down in heteroscedastic settings, also in the case of 20 nodes.

To give stationary MTS baselines the best possible chance, we supply them with the ground truth regime partitions. This is done by training the aforementioned models on each pure regime separately (regime governed by the same causal model). Yet, FANTOM still dominates the structure learning task, while learning the number of regime and their indices as well, achieving an F1 of 89.1 % and an NHD of 0.006 for instantaneous links and an F1 of 85.4% and and an NHD 0.01% for time lagged (Table 10).

Table 12: Average SHD, F1 scores, NHD and Ratio for different models with d = 40 nodes and K = 2 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

| Heteroscedastic noise, K = 2 and d = 40   | Heteroscedastic noise, K = 2 and d = 40   | Heteroscedastic noise, K = 2 and d = 40   | Heteroscedastic noise, K = 2 and d = 40   | Heteroscedastic noise, K = 2 and d = 40   | Heteroscedastic noise, K = 2 and d = 40   | Heteroscedastic noise, K = 2 and d = 40   | Heteroscedastic noise, K = 2 and d = 40   | Heteroscedastic noise, K = 2 and d = 40   | Heteroscedastic noise, K = 2 and d = 40   | Heteroscedastic noise, K = 2 and d = 40   |
|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| Model                                     | Split                                     | Inst.                                     | Inst.                                     | Inst.                                     | Inst.                                     | Lag                                       | Lag                                       | Lag                                       | Lag                                       | Regime                                    |
|                                           |                                           | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | Acc.                                      |
| PCMCI+                                    | ×                                         | 146.5 ± 5 . 0                             | 25.8 ± 0 . 1                              | 0.02 ± 0 . 0                              | 0.74 ± 0 . 0                              | 109.0 ± 15 .                              | 58.0 ± 4 . 5                              | 0.01 ± 0 . 0                              | 0.42 ± 0 . 0                              | ×                                         |
| Rhino                                     | ×                                         | 137.0 ± 7 . 1                             | 0.0 ± 0 . 0                               | 0.02 ± 0 . 0                              | 1.00 ± 0 . 0                              | 700.0 ± 87 .                              | 28.2 ± 5 . 7                              | 0.12 ± 0 . 0                              | 0.71 ± 0 . 0                              | ×                                         |
| DYNOTEARS                                 | ×                                         | 137.5 ± 9 . 2                             | 17.1 ± 1 . 4                              | 0.020 ± 0 . 0                             | 0.82 ± 0 . 0                              | 129.0 ± 15                                | 36.6 ± 4 . 9                              | 0.01 ± 0 . 0                              | 0.63 ± 0 . 0                              | ×                                         |
| CASTOR                                    | ×                                         | 151.0 ± 11.                               | 0.0 ± 0.0                                 | 0.03 ± 0 . 0                              | 1.00 ± 0 . 0                              | . 333.0 ± 9 . 3                           | 19.6 ± 5 . 6                              | 0.050 ± 0 . 0                             | 0.80 ± 0 . 0                              | ×                                         |
| RPCMCI                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| CASTOR                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| FANTOM                                    | ✓                                         | 27.00 6 . 0                               | 85.6 3 . 7                                | 0.005 0 . 0                               | 0.14 0 . 0                                | 26.5 0 . 5                                | 91.9 1 . 0                                | 0.004 0 . 0                               | 0.08 0 . 0                                | 99.9 0 . 1                                |

±

±

±

±

±

±

±

±

±

Table 13: Average SHD, F1 scores, NHD and Ratio for different models with d = 40 nodes and K = 3 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

| Heteroscedastic noise, K = 3 and d = 40   | Heteroscedastic noise, K = 3 and d = 40   | Heteroscedastic noise, K = 3 and d = 40   | Heteroscedastic noise, K = 3 and d = 40   | Heteroscedastic noise, K = 3 and d = 40   | Heteroscedastic noise, K = 3 and d = 40   | Heteroscedastic noise, K = 3 and d = 40   | Heteroscedastic noise, K = 3 and d = 40   | Heteroscedastic noise, K = 3 and d = 40   | Heteroscedastic noise, K = 3 and d = 40   | Heteroscedastic noise, K = 3 and d = 40   |
|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|
| Model                                     | Split                                     | Inst.                                     | Inst.                                     | Inst.                                     | Inst.                                     | Lag                                       | Lag                                       | Lag                                       | Lag                                       | Regime                                    |
|                                           |                                           | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | SHD ↓                                     | F1 ↑                                      | NHD ↓                                     | Ratio ↓                                   | Acc.                                      |
| PCMCI+                                    | ×                                         | 222.0 ± 1 . 4                             | 25.2 ± 0 . 4                              | 0.01 ± 0 . 0                              | 0.75 ± 0 . 0                              | 142.0 ± 17 .                              | 62.4 ± 1 . 0                              | 0.01 ± 0 . 0                              | 0.37 ± 0 . 0                              | ×                                         |
| Rhino                                     | ×                                         | 210.5 ± 9 . 2                             | 0.0 ± 0 . 0                               | 0.01 ± 0 . 0                              | 1.00 ± 0 . 0                              | 1005. ± 146 .                             | 29.3 ± 5 . 4                              | 0.08 ± 0 . 0                              | 0.7 ± 0 . 0                               | ×                                         |
| DYNOTEARS                                 | ×                                         | 203.0 ± 14 .                              | 15.2 ± 2 . 2                              | 0.01 ± 0 . 0                              | 0.85 ± 0 . 0                              | 161.0 ± 1 . 4                             | 51.3 ± 15 .                               | 0.01 ± 0 . 0                              | 0.49 ± 0 . 1                              | ×                                         |
| CASTOR                                    | ×                                         | 224.0 ± 10 . 2                            | 0.00 ± 0 . 0                              | 0.01 ± 0 . 0                              | 1.00 ± 0 . 0                              | 501.0 ± 21 .                              | 23.1 ± 1 . 2                              | 0.03 ± 0 . 0                              | 0.76 ± 0 . 0                              | ×                                         |
| RPCMCI                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| CASTOR                                    | ✓                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         | -                                         |
| FANTOM                                    | ✓                                         | 52.0 9 . 0                                | 82.2 5 . 1                                | 0.004 0 . 0                               | 0.19 0 . 0                                | 65.0 7 . 0                                | 87.9 0 . 6                                | 0.004 0 . 0                               | 0.11 0 . 0                                | 99.8 0 . 0                                |

±

±

±

±

±

±

±

±

±

Under heteroscedastic conditions, d = 40 node graphs with either two or three regimes, FANTOM outperforms every baselineand shows that it can scale to large graphs even in this complex setting. Among methods explicitly designed for multi-regime MTS, it is the only one that converges to the true regime partitions, attaining 99.9% in regime detection accuracy. CASTOR and RPCMCI break

down in heteroscedastic settings (Table 12). Large scale graphs helps FANTOM to differentiate between the different regimes and increases the regime detection accuracy by 2% compared to 20 node graphs settings.

To give stationary MTS baselines the best possible chance, we supply them with the ground truth regime partitions. This is done by training the aforementioned models on each pure regime separately (regime governed by the same causal model). Yet, FANTOM still dominates the structure learning task, while learning the number of regime and their indices as well, achieving an F1 of 85.6 % and an NHD of 0.14 (Table 12).

## F.2.2 Non-Gaussian noise with different number of nodes and regimes

Table 14: Average SHD, F1 scores, NHD and Ratio for different models with d = 10 nodes and K = 2 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

| Homoscedastic non-Gaussian noise, K = 2 and d = 10   | Homoscedastic non-Gaussian noise, K = 2 and d = 10   | Homoscedastic non-Gaussian noise, K = 2 and d = 10   | Homoscedastic non-Gaussian noise, K = 2 and d = 10   | Homoscedastic non-Gaussian noise, K = 2 and d = 10   | Homoscedastic non-Gaussian noise, K = 2 and d = 10   | Homoscedastic non-Gaussian noise, K = 2 and d = 10   | Homoscedastic non-Gaussian noise, K = 2 and d = 10   | Homoscedastic non-Gaussian noise, K = 2 and d = 10   | Homoscedastic non-Gaussian noise, K = 2 and d = 10   | Homoscedastic non-Gaussian noise, K = 2 and d = 10   | Homoscedastic non-Gaussian noise, K = 2 and d = 10   |
|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| Model                                                | Split                                                | Type                                                 | Inst.                                                | Inst.                                                | Inst.                                                | Inst.                                                | Lag                                                  | Lag                                                  | Lag                                                  | Lag                                                  | Regime                                               |
|                                                      |                                                      |                                                      | SHD ↓                                                | F1 ↑                                                 | NHD                                                  | Ratio ↓                                              | SHD ↓                                                | F1 ↑                                                 | NHD ↓                                                | Ratio ↓                                              | Acc.                                                 |
| PCMCI+                                               | ×                                                    | W                                                    | 6.00 ± 0 . 0                                         | 88.1 ± 0 . 5                                         | ↓ 0.01 ± 0 . 0                                       | 0.12 ± 0 . 0                                         | 8.00 ± 5 . 0                                         | 90.7 ± 6 . 1                                         | 0.01 ± 0 . 0                                         | 0.09 ± 0 . 0                                         | ×                                                    |
| Rhino                                                | ×                                                    | W                                                    | 2.50 ± 0 . 5                                         | 96.0 ± 0 . 1                                         | 0.004 ± 0 . 0                                        | 0.04 ± 0 . 0                                         | 4.00 ± 1 . 0                                         | 96.0 ± 0 . 4                                         | 0.006 ± 0 . 0                                        | 0.04 ± 0 . 0                                         | ×                                                    |
| DYNOTEARS                                            | ×                                                    | W                                                    | 42.0 ± 11 .                                          | 51.8 ± 7 . 9                                         | 0.12 ± 0 . 0                                         | 0.48 ± 0 . 0                                         | 8.00 ± 1 . 0                                         | 86.8 ± 0 . 5                                         | 0.02 ± 0 . 0                                         | 0.12 ± 0 . 0                                         | ×                                                    |
| CASTOR                                               | ×                                                    | W                                                    | 17.0 ± 3 . 0                                         | 62.6 ± 8 . 5                                         | 0.055 ± 0 . 0                                        | 0.37 ± 0 . 0                                         | 11.0 ± 1 . 0                                         | 85.2 ± 0 . 7                                         | 0.02 ± 0 . 0                                         | 0.15 ± 0 . 0                                         | ×                                                    |
| RPCMCI                                               | ✓                                                    | W                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    |
| CASTOR                                               | ✓                                                    | W                                                    | 34.0 ± 20 .                                          | 42.5 ± 27 .                                          | 0.09 ± 0 . 0                                         | 0.57 ± 0 . 2                                         | 45.0 ± 31 .                                          | 47.0 ± 25 .                                          | 0.13 ± 0 . 1                                         | 0.52 ± 0 . 2                                         | 77.0 ± 13 .                                          |
| FANTOM                                               | ✓                                                    | W                                                    | 1.00 ± 0 . 0                                         | 98.2 ± 0 . 1                                         | 0.002 ± 0 . 0                                        | 0.01 ± 0 . 0                                         | 8.00 ± 4 . 0                                         | 91.0 ± 4 . 9                                         | 0.02 ± 0 . 0                                         | 0.08 ± 0 . 05                                        | 98.6 ± 0 . 2                                         |
|                                                      |                                                      |                                                      | SHD ↓                                                | SHD ↓                                                | F1 ↑                                                 | F1 ↑                                                 | NHD ↓                                                | NHD ↓                                                | Ratio ↓                                              | Ratio ↓                                              | Acc.                                                 |
| CD-NOD                                               | ×                                                    | S                                                    | 33.0 ± 3 . 0                                         | 33.0 ± 3 . 0                                         | 47.0 ± 5 . 9                                         | 47.0 ± 5 . 9                                         | 0.41 ± 0 . 0                                         | 0.41 ± 0 . 0                                         | 0.52 ± 0 . 0                                         | 0.52 ± 0 . 0                                         | ×                                                    |
| FANTOM                                               | ✓                                                    | S                                                    | 7.5 2 . 5                                            | 7.5 2 . 5                                            | 93.1 2 . 2                                           | 93.1 2 . 2                                           | 0.07 0 . 0                                           | 0.07 0 . 0                                           | 0.06 0 . 0                                           | 0.06 0 . 0                                           | 99.6 0 . 0                                           |

±

±

±

±

±

Table 15: Average SHD, F1 scores, NHD and Ratio for different models with d = 10 nodes and K = 3 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

| Homoscedastic non-Gaussian noise, K = 3 and d = 10   | Homoscedastic non-Gaussian noise, K = 3 and d = 10   | Homoscedastic non-Gaussian noise, K = 3 and d = 10   | Homoscedastic non-Gaussian noise, K = 3 and d = 10   | Homoscedastic non-Gaussian noise, K = 3 and d = 10   | Homoscedastic non-Gaussian noise, K = 3 and d = 10   | Homoscedastic non-Gaussian noise, K = 3 and d = 10   | Homoscedastic non-Gaussian noise, K = 3 and d = 10   | Homoscedastic non-Gaussian noise, K = 3 and d = 10   | Homoscedastic non-Gaussian noise, K = 3 and d = 10   | Homoscedastic non-Gaussian noise, K = 3 and d = 10   | Homoscedastic non-Gaussian noise, K = 3 and d = 10   |
|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| Model                                                | Split                                                | Type                                                 | Inst.                                                | Inst.                                                | Inst.                                                | Inst.                                                | Lag                                                  | Lag                                                  | Lag                                                  | Lag                                                  | Regime                                               |
|                                                      |                                                      |                                                      | SHD ↓                                                | F1 ↑                                                 | NHD ↓                                                | Ratio ↓                                              | SHD ↓                                                | F1 ↑                                                 | NHD ↓                                                | Ratio ↓                                              | Acc.                                                 |
| PCMCI+                                               | ×                                                    | W                                                    | 17.5 ± 2 . 1                                         | 74.9 ± 5 . 0                                         | 0.02 ± 0 . 0                                         | 0.24 ± 0 . 0                                         | 14.5 ± 2 . 1                                         | 88.3 ± 1 . 6                                         | 0.01 ± 0 . 0                                         | 0.11 ± 0 . 0                                         | ×                                                    |
| Rhino                                                | ×                                                    | W                                                    | 2.50 ± 0 . 7                                         | 96.8 ± 1 . 3                                         | 0.002 ± 0 . 0                                        | 0.03 ± 0 . 0                                         | 6.00 ± 1 . 4                                         | 95.2 ± 1 . 6                                         | 0.006 ± 0 . 0                                        | 0.04 ± 0 . 0                                         | ×                                                    |
| DYNOTEARS                                            | ×                                                    | W                                                    | 42.0 ± 33 .                                          | 54.4 ± 11 . 2                                        | 0.06 ± 0 . 0                                         | 0.45 ± 0 . 1                                         | 21.0 ± 1 . 4                                         | 82.1 ± 1 . 6                                         | 0.02 ± 0 . 0                                         | 0.17 ± 0 . 0                                         | ×                                                    |
| CASTOR                                               | ×                                                    | W                                                    | 22.0 ± 2 . 8                                         | 66.2 ± 2 . 5                                         | 0.030 ± 0 . 0                                        | 0.33 ± 0 . 0                                         | 17.0 ± 4 . 2                                         | 84.4 ± 1 . 8                                         | 0.01 ± 0 . 0                                         | 0.15 ± 0 . 0                                         | ×                                                    |
| RPCMCI                                               | ✓                                                    | W                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    |
| CASTOR                                               | ✓                                                    | W                                                    | 47.0 ± 17 .                                          | 34.8 ± 30 .                                          | 0.05 ± 0 . 0                                         | 0.65 ± 0 . 2                                         | 59.5 ± 31                                            | 39.4 ± 19 .                                          | 0.07 ± 0 . 0                                         | 0.60 ± 0 . 2                                         | 51.6 ± 8 . 9                                         |
| FANTOM                                               | ✓                                                    | W                                                    | 0.33 ± 0 . 4                                         | 99.5 ± 0 . 6                                         | 0.00 ± 0 . 0                                         | 0.00 ± 0 . 0                                         | . 12.5 ± 6 . 8                                       | 89.1 ± 3 . 7                                         | 0.01 ± 0 . 0                                         | 0.10 ± 0 . 0                                         | 99.4 ± 0 . 1                                         |
|                                                      |                                                      |                                                      | SHD ↓                                                | SHD ↓                                                | F1 ↑                                                 | F1 ↑                                                 | NHD ↓                                                | NHD ↓                                                | Ratio ↓                                              | Ratio ↓                                              | Acc.                                                 |
| CD-NOD                                               | ×                                                    | S                                                    | 42.5 ± 2 . 5                                         | 42.5 ± 2 . 5                                         | 31.8 ± 1 . 1                                         | 31.8 ± 1 . 1                                         | 0.42 ± 0 . 0                                         | 0.42 ± 0 . 0                                         | 0.67 ± 0 . 0                                         | 0.67 ± 0 . 0                                         | ×                                                    |
| FANTOM                                               | ✓                                                    | S                                                    | 4.5 2 . 0                                            | 4.5 2 . 0                                            | 95.6 1 . 5                                           | 95.6 1 . 5                                           | 0.04 0 . 0                                           | 0.04 0 . 0                                           | 0.04 0 . 0                                           | 0.04 0 . 0                                           | 96.6 0 . 2                                           |

±

±

±

±

±

Under homoscedastic, non-Gaussian noise with d = 10 node graphs and either two or three regimes, FANTOM outperforms every baseline on instantaneous-link inference, reaching an F1 of 98.2 % and an NHD of 0.002. Among methods explicitly designed for multi-regime MTS, both FANTOM and CASTOR recover the exact number of regimes, whereas RPCMCI fails to converge to the true partitions. FANTOM further surpasses CASTOR in regime detection (98.6 % vs. 77.0 %) and in DAG learning (F1 = 98.2 % vs. 42.5 %).

To give the stationary-MTS baselines their best chance, we supply them with the ground-truth regime labels and train each model on the corresponding pure regime. Even in this favorable setting, only Rhino exceeds FANTOM on time-lagged links, achieving an F1 of 96.0 % compared with FANTOM's 91.0 %.

Table 16: Average SHD, F1 scores, NHD and Ratio for different models with d = 20 nodes and K = 2 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

| Homoscedastic non-Gaussian noise, K = 2 and d = 20   | Homoscedastic non-Gaussian noise, K = 2 and d = 20   | Homoscedastic non-Gaussian noise, K = 2 and d = 20   | Homoscedastic non-Gaussian noise, K = 2 and d = 20   | Homoscedastic non-Gaussian noise, K = 2 and d = 20   | Homoscedastic non-Gaussian noise, K = 2 and d = 20   | Homoscedastic non-Gaussian noise, K = 2 and d = 20   | Homoscedastic non-Gaussian noise, K = 2 and d = 20   | Homoscedastic non-Gaussian noise, K = 2 and d = 20   | Homoscedastic non-Gaussian noise, K = 2 and d = 20   | Homoscedastic non-Gaussian noise, K = 2 and d = 20   | Homoscedastic non-Gaussian noise, K = 2 and d = 20   |
|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| Model                                                | Split                                                | Type                                                 | Inst.                                                | Inst.                                                | Inst.                                                | Inst.                                                | Lag                                                  | Lag                                                  | Lag                                                  | Lag                                                  | Regime                                               |
|                                                      |                                                      |                                                      | SHD ↓                                                | F1 ↑                                                 | NHD ↓                                                | Ratio ↓                                              | SHD ↓                                                | F1 ↑                                                 | NHD ↓                                                | Ratio ↓                                              | Acc.                                                 |
| PCMCI+                                               | ×                                                    | W                                                    | 46.0 ± 2 . 8                                         | 54.9 ± 0 . 6                                         | 0.03 ± 0 . 0                                         | 0.45 ± 0 . 0                                         | 17.0 ± 4 . 2                                         | 88.7 ± 0 . 5                                         | 0.009 ± 0 . 0                                        | 0.11 ± 0 . 0                                         | ×                                                    |
| Rhino                                                | ×                                                    | W                                                    | 17.5 ± 14 .                                          | 82.5 ± 16 .                                          | 0.007 ± 0 . 0                                        | 0.17 ± 0 . 1                                         | 29.5 ± 3 . 5                                         | 83.5 ± 2 . 1                                         | 0.01 ± 0 . 0                                         | 0.16 ± 0 . 0                                         | ×                                                    |
| DYNOTEARS                                            | ×                                                    | W                                                    | 44.5 ± 3 . 5                                         | 44.9 ± 3 . 1                                         | 0.03 ± 0 . 0                                         | 0.55 ± 0 . 0                                         | 46.5 ± 12 .                                          | 55.8 ± 5 . 2                                         | 0.025 ± 0 . 0                                        | 0.44 ± 0 . 0                                         | ×                                                    |
| CASTOR                                               | ×                                                    | W                                                    | 139.5 ± 13 .                                         | 41.8 ± 5 . 3                                         | 0.10 ± 0 . 0                                         | 0.58 ± 0 . 0                                         | 186.0 ± 28                                           | 40.1 ± 2 . 3                                         | 0.13 ± 0 . 0                                         | 0.60 ± 0 . 0                                         | ×                                                    |
| RPCMCI                                               | ✓                                                    | W                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    |
| CASTOR                                               | ✓                                                    | W                                                    | 68.0 ± 8 . 5                                         | 37.2 ± 12 .                                          | 0.04 ± 0 . 0                                         | 0.62 ± 0 . 1                                         | 86.0 ± 9 . 9                                         | 37.4 ± 7 . 8                                         | 0.06 ± 0 . 0                                         | 0.64 ± 0 . 0                                         | 46.6 ± 19 .                                          |
| FANTOM                                               | ✓                                                    | W                                                    | 3.50 ± 1 . 5                                         | 97.2 ± 1 . 2                                         | 0.002 ± 0 . 0                                        | 0.02 ± 0 . 0                                         | 11.5 ± 2 . 5                                         | 93.1 ± 1 . 6                                         | 0.006 ± 0 . 0                                        | 0.06 ± 0 . 0                                         | 100. ± 0 . 0                                         |
|                                                      |                                                      |                                                      | SHD ↓                                                | SHD ↓                                                | F1 ↑                                                 | F1 ↑                                                 | NHD ↓                                                | NHD ↓                                                | Ratio ↓                                              | Ratio ↓                                              | Acc.                                                 |
| CD-NOD                                               | ×                                                    | S                                                    | 106. ± 4 . 0                                         | 106. ± 4 . 0                                         | 26.0 ± 4 . 1                                         | 26.0 ± 4 . 1                                         | 0.31 ± 0 . 0                                         | 0.31 ± 0 . 0                                         | 0.73 ± 0 . 0                                         | 0.73 ± 0 . 0                                         | ×                                                    |
| FANTOM                                               | ✓                                                    | S                                                    | 4.0 0 . 0                                            | 4.0 0 . 0                                            | 98.3 0 . 0                                           | 98.3 0 . 0                                           | 0.01 0 . 0                                           | 0.01 0 . 0                                           | 0.01 0 . 0                                           | 0.01 0 . 0                                           | 100. 0 . 0                                           |

±

±

±

±

±

Under homoscedastic, non-Gaussian noise with d = 20 node graphs and two regimes, FANTOM outperforms every baseline on instantaneous-link and time lagged link inference, reaching an F1 of 97.2 % and an NHD of 0.002 for instantaneous links and an F1 of 93.1% on time lagged relationships. Among methods explicitly designed for multi-regime MTS, both FANTOM and CASTOR recover the exact number of regimes, whereas RPCMCI fails to converge to the true partitions. FANTOM further surpasses CASTOR in regime detection (100. % vs. 46.6 %) and in DAG learning (F1 = 97.2 %vs. 37.2 %).

To give the stationary-MTS baselines their best chance, we supply them with the ground-truth regime labels and train each model on the corresponding pure regime. Even in this favorable setting, FANTOM out performs all the baselines, achieving an F1 of 97.2 %.

Table 17: Average SHD, F1 scores, NHD and Ratio for different models with d = 20 nodes and K = 3 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

|           | Homoscedastic non-Gaussian noise, K = 3 and d = 20   | Homoscedastic non-Gaussian noise, K = 3 and d = 20   | Homoscedastic non-Gaussian noise, K = 3 and d = 20   | Homoscedastic non-Gaussian noise, K = 3 and d = 20   | Homoscedastic non-Gaussian noise, K = 3 and d = 20   | Homoscedastic non-Gaussian noise, K = 3 and d = 20   | Homoscedastic non-Gaussian noise, K = 3 and d = 20   | Homoscedastic non-Gaussian noise, K = 3 and d = 20   | Homoscedastic non-Gaussian noise, K = 3 and d = 20   | Homoscedastic non-Gaussian noise, K = 3 and d = 20   | Homoscedastic non-Gaussian noise, K = 3 and d = 20   |
|-----------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| Model     | Split                                                | Type                                                 | Inst.                                                | Inst.                                                | Inst.                                                | Inst.                                                | Lag                                                  | Lag                                                  | Lag                                                  | Lag                                                  | Regime                                               |
|           |                                                      |                                                      | SHD ↓                                                | F1 ↑                                                 | NHD ↓                                                | Ratio ↓                                              | SHD ↓                                                | F1 ↑                                                 | NHD ↓                                                | Ratio ↓                                              | Acc.                                                 |
| PCMCI+    | ×                                                    | W                                                    | 66.5 ± 2 . 1                                         | 55.1 ± 2 . 5                                         | 0.02 ± 0 . 0                                         | 0.45 ± 0 . 0                                         | 24.5 ± 12                                            | 88.9 ± 3 . 0                                         | 0.007 ± 0 . 0                                        | 0.11 ± 0 . 0                                         | ×                                                    |
| Rhino     | ×                                                    | W                                                    | 27.5 ± 14 .                                          | 82.1 ± 11 .                                          | 0.007 ± 0 . 0                                        | 0.17 ± 0 . 1                                         | . 50.5 ± 3 . 5                                       | 81.8 ± 1 . 4                                         | 0.01 ± 0 . 0                                         | 0.18 ± 0 . 0                                         | ×                                                    |
| DYNOTEARS | ×                                                    | W                                                    | 69.0 ± 4 . 2                                         | 43.6 ± 3 . 7                                         | 0.02 ± 0 . 0                                         | 0.56 ± 0 . 0                                         | 54.5 ± 10 . 6                                        | 66.3 ± 2 . 4                                         | 0.01 ± 0 . 0                                         | 0.34 ± 0 . 0                                         | ×                                                    |
| CASTOR    | ×                                                    | W                                                    | 167.0 ± 9 . 9                                        | 38.3 ± 4 . 6                                         | 0.05 ± 0 . 0                                         | 0.61 ± 0 . 0                                         | 264.5 ± 34 .                                         | 41.2 ± 1 . 5                                         | 0.08 ± 0 . 0                                         | 0.58 ± 0 . 0                                         | ×                                                    |
| RPCMCI    | ✓                                                    | W                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    |
| CASTOR    | ✓                                                    | W                                                    | 121.5 ± 27 .                                         | 34.3 ± 9 . 6                                         | 0.03 ± 0 . 0                                         | 0.65 ± 0 . 1                                         | 181.5 ± 17 .                                         | 33.7 ± 7 . 8                                         | 0.05 ± 0 . 0                                         | 0.66 ± 0 . 0                                         | 79.8 ± 4 . 3                                         |
| FANTOM    | ✓                                                    | W                                                    | 7.00 ± 0 . 0                                         | 96.1 ± 0 . 0                                         | 0.001 ± 0 . 0                                        | 0.03 ± 0 . 0                                         | 45.5 ± 23 . 0                                        | 85.2 ± 6 . 1                                         | 0.01 ± 0 . 0                                         | 0.14 ± 0 . 0                                         | 100. ± 0 . 0                                         |
|           |                                                      |                                                      | SHD ↓                                                | SHD ↓                                                | F1 ↑                                                 | F1 ↑                                                 | NHD ↓                                                | NHD ↓                                                | Ratio ↓                                              | Ratio ↓                                              | Acc.                                                 |
| CD-NOD    | ×                                                    | S                                                    | 133. ± 1 . 5                                         | 133. ± 1 . 5                                         | 31.5 ± 1 . 8                                         | 31.5 ± 1 . 8                                         | 0.43 ± 0 . 0                                         | 0.43 ± 0 . 0                                         | 0.61 ± 0 . 0                                         | 0.61 ± 0 . 0                                         | ×                                                    |
| FANTOM    | ✓                                                    | S                                                    | 6.0 1 . 0                                            | 6.0 1 . 0                                            | 98.2 0 . 2                                           | 98.2 0 . 2                                           | 0.01 0 . 0                                           | 0.01 0 . 0                                           | 0.01 0 . 0                                           | 0.01 0 . 0                                           | 100. 0 . 0                                           |

±

±

±

±

±

Under homoscedastic, non-Gaussian noise with d = 20 node graphs and three regimes, FANTOM outperforms every baseline on instantaneous-link inference, reaching an F1 of 96.1 % and an NHD of 0.001. Among methods explicitly designed for multi-regime MTS, both FANTOM and CASTOR recover the exact number of regimes, whereas RPCMCI fails to converge to the true partitions. FANTOM further surpasses CASTOR in regime detection (100. % vs. 79.8 %) and in DAG learning (F1 = 96.1 % vs. 34.3 %).

To give the stationary-MTS baselines their best chance, we supply them with the ground-truth regime labels and train each model on the corresponding pure regime. For this scenario of 20 nodes and 3 regimes and benefiting from regime labels, PCMCI+ exceeds FANTOM and Rhino on time-lagged links, achieving an F1 of 88.9 % compared with FANTOM's 85.2%.

Table 18: Average SHD, F1 scores, NHD and Ratio for different models with d = 40 nodes and K = 2 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

| Homoscedastic non-Gaussian noise, K = 2 and d = 40   | Homoscedastic non-Gaussian noise, K = 2 and d = 40   | Homoscedastic non-Gaussian noise, K = 2 and d = 40   | Homoscedastic non-Gaussian noise, K = 2 and d = 40   | Homoscedastic non-Gaussian noise, K = 2 and d = 40   | Homoscedastic non-Gaussian noise, K = 2 and d = 40   | Homoscedastic non-Gaussian noise, K = 2 and d = 40   | Homoscedastic non-Gaussian noise, K = 2 and d = 40   | Homoscedastic non-Gaussian noise, K = 2 and d = 40   | Homoscedastic non-Gaussian noise, K = 2 and d = 40   | Homoscedastic non-Gaussian noise, K = 2 and d = 40   | Homoscedastic non-Gaussian noise, K = 2 and d = 40   |
|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| Model                                                | Split                                                | Type                                                 | Inst.                                                | Inst.                                                | Inst.                                                | Inst.                                                | Lag                                                  | Lag                                                  | Lag                                                  | Lag                                                  | Regime                                               |
|                                                      |                                                      |                                                      | SHD ↓                                                | F1 ↑                                                 | NHD                                                  | Ratio ↓                                              | SHD ↓                                                | F1 ↑                                                 | NHD ↓                                                | Ratio ↓                                              | Acc.                                                 |
| PCMCI+                                               | ×                                                    | W                                                    | 91.0 ± 2 . 8                                         | 56.7 ± 0 . 1                                         | ↓ 0.01 ± 0 . 0                                       | 0.43 ± 0 . 0                                         | 25.0 ± 1 . 4                                         | 91.9 ± 0 . 8                                         | 0.003 ± 0 . 0                                        | 0.08 ± 0 . 0                                         | ×                                                    |
| Rhino                                                | ×                                                    | W                                                    | 24.5 ± 2 . 1                                         | 90.7 ± 0 . 9                                         | 0.004 ± 0 . 0                                        | 0.09 ± 0 .                                           | 34.5 ± 5 . 0                                         | 89.8 ± 0 . 9                                         | 0.005 ± 0 . 0                                        | 0.10 ± 0 . 0                                         | ×                                                    |
| DYNOTEARS                                            | ×                                                    | W                                                    | 100.0 ± 5 . 7                                        | 40.8 ± 3 . 7                                         | 0.015 ± 0 . 0                                        | 0 0.59 ± 0 . 0                                       | 58.5 ± 17 .                                          | 76.4 ± 5 . 4                                         | 0.009 ± 0 . 0                                        | 0.21 ± 0 . 0                                         | ×                                                    |
| CASTOR                                               | ×                                                    | W                                                    | 94.0 ± 46 .                                          | 67.3 ± 5 . 0                                         | 0.010 ± 0 . 0                                        | 0.33 ± 0 . 0                                         | 100.0 ± 49                                           | 78.6 ± 3 . 4                                         | 0.01 ± 0 . 0                                         | 0.21 ± 0 . 0                                         | ×                                                    |
| RPCMCI                                               | ✓                                                    | W                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    |
| CASTOR                                               | ✓                                                    | W                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    |
| FANTOM                                               | ✓                                                    | W                                                    | 29.5 ± 1 . 5                                         | 88.1 ± 0 . 2                                         | 0.004 ± 0 . 0                                        | 0.11 ± 0 . 0                                         | 32.3 ± 0 . 5                                         | 90.8 ± 0 . 2                                         | 0.005 ± 0 . 0                                        | 0.08 ± 0 . 0                                         | 100. ± 0 . 0                                         |
|                                                      |                                                      |                                                      | SHD ↓                                                | SHD ↓                                                | F1 ↑                                                 | F1 ↑                                                 | NHD ↓                                                | NHD ↓                                                | Ratio ↓                                              | Ratio ↓                                              | Acc.                                                 |
| CD-NOD                                               | ×                                                    | S                                                    | 260. ± 5 . 0                                         | 260. ± 5 . 0                                         | 17.8 ± 3 . 6                                         | 17.8 ± 3 . 6                                         | 0.18 ± 0 . 0                                         | 0.18 ± 0 . 0                                         | 0.82 0 . 0                                           | 0.82 0 . 0                                           | ×                                                    |
| FANTOM                                               | ✓                                                    | S                                                    | 39.5 4 . 5                                           | 39.5 4 . 5                                           | 92.4 0 . 6                                           | 92.4 0 . 6                                           | 0.02 0 . 0                                           | 0.02 0 . 0                                           | ± 0.07 0 . 0                                         | ± 0.07 0 . 0                                         | 100. 0 . 0                                           |

±

±

±

±

±

Under homoscedastic, non-Gaussian noise with d = 40 -node graphs and two regimes, FANTOM is the only method that recovers the correct number of regimes, whereas CASTOR and RPCMCI fail to converge. FANTOM achieves 100% regime-detection accuracy and an 88 . 1% F1 score in DAG recovery.

For a fair comparison with stationary-MTS baselines, we provide these models with the ground-truth regime labels and train them separately on each pure regime. In this advantaged setting, Rhino surpasses FANTOM on instantaneous links ( F1 = 90 . 7% vs. 88 . 1% ), while PCMCI+ leads on time-lagged links ( F1 = 91 . 9% vs. 90 . 8% ).

Table 19: Average SHD, F1 scores, NHD and Ratio for different models with d = 40 nodes and K = 3 regimes. Split denotes whether regime separation is automatic ( ✓ ) or manual ( × ). Inst. refers to instantaneous links, and Lag to time-lagged edges.

| Homoscedastic non-Gaussian noise, K = 3 and d = 40   | Homoscedastic non-Gaussian noise, K = 3 and d = 40   | Homoscedastic non-Gaussian noise, K = 3 and d = 40   | Homoscedastic non-Gaussian noise, K = 3 and d = 40   | Homoscedastic non-Gaussian noise, K = 3 and d = 40   | Homoscedastic non-Gaussian noise, K = 3 and d = 40   | Homoscedastic non-Gaussian noise, K = 3 and d = 40   | Homoscedastic non-Gaussian noise, K = 3 and d = 40   | Homoscedastic non-Gaussian noise, K = 3 and d = 40   | Homoscedastic non-Gaussian noise, K = 3 and d = 40   | Homoscedastic non-Gaussian noise, K = 3 and d = 40   | Homoscedastic non-Gaussian noise, K = 3 and d = 40   |
|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| Model                                                | Split                                                | Type                                                 | Inst.                                                | Inst.                                                | Inst.                                                | Inst.                                                | Lag                                                  | Lag                                                  | Lag                                                  | Lag                                                  | Regime                                               |
|                                                      |                                                      |                                                      | SHD ↓                                                | F1 ↑                                                 | NHD ↓                                                | Ratio ↓                                              | SHD ↓                                                | F1 ↑                                                 | NHD ↓                                                | Ratio ↓                                              | Acc.                                                 |
| PCMCI+                                               | ×                                                    | W                                                    | 141.5 ± 12 .                                         | 56.5 ± 4 . 5                                         | 0.008 ± 0 . 0                                        | 0.43 ± 0 . 0                                         | 37.5 ± 9 . 2                                         | 91.6 ± 2 . 8                                         | 0.003 ± 0 . 0                                        | 0.08 ± 0 . 0                                         | ×                                                    |
| Rhino                                                | ×                                                    | W                                                    | 53.5 ± 26 .                                          | 85.5 ± 7 . 4                                         | 0.004 ± 0 . 0                                        | 0.14 ± 0 . 0                                         | 52.0 ± 5 . 7                                         | 89.2 ± 2 . 2                                         | 0.004 ± 0 . 0                                        | 0.11 ± 0 . 0                                         | ×                                                    |
| DYNOTEARS                                            | ×                                                    | W                                                    | 152.0 ± 4 . 2                                        | 40.8 ± 3 . 7                                         | 0.01 ± 0 . 0                                         | 0.59 ± 0 . 0                                         | 91.0 ± 17 .                                          | 75.8 ± 3 . 4                                         | 0.006 ± 0 . 0                                        | 0.24 ± 0 . 0                                         | ×                                                    |
| CASTOR                                               | ×                                                    | W                                                    | 93.5 ± 2 . 1                                         | 65.9 ± 7 . 6                                         | 0.009 ± 0 . 0                                        | 0.34 ± 0 . 0                                         | 94.0 ± 8 . 5                                         | 78.6 ± 4 . 5                                         | 0.009 ± 0 . 0                                        | 0.21 ± 0 . 0                                         | ×                                                    |
| RPCMCI                                               | ✓                                                    | W                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    |
| CASTOR                                               | ✓                                                    | W                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    | -                                                    |
| FANTOM                                               | ✓                                                    | W                                                    | 35.3 ± 0 . 7                                         | 90.9 ± 0 . 4                                         | 0.002 ± 0 . 0                                        | 0.09 ± 0 . 0                                         | 46.3 ± 0 . 3                                         | 90.7 ± 1 . 3                                         | 0.003 ± 0 . 0                                        | 0.09 ± 0 . 0                                         | 100. ± 0 . 0                                         |
|                                                      |                                                      |                                                      | SHD ↓                                                | SHD ↓                                                | F1 ↑                                                 | F1 ↑                                                 | NHD ↓                                                | NHD ↓                                                | Ratio ↓                                              | Ratio ↓                                              | Acc.                                                 |
| CD-NOD                                               | ×                                                    | S                                                    | 342. ± 4 . 3                                         | 342. ± 4 . 3                                         | 17.1 ± 3 . 2                                         | 17.1 ± 3 . 2                                         | 0.24 ± 0 . 0                                         | 0.24 ± 0 . 0                                         | 0.82 ± 0 . 0                                         | 0.82 ± 0 . 0                                         | ×                                                    |
| FANTOM                                               | ✓                                                    | S                                                    | 57.0 2 . 5                                           | 57.0 2 . 5                                           | 92.8 0 . 7                                           | 92.8 0 . 7                                           | 0.03 0 . 0                                           | 0.03 0 . 0                                           | 0.07 0 . 0                                           | 0.07 0 . 0                                           | 100. 0 . 0                                           |

±

## F.3 Additional experiments for L=2

Table 20: Performance of FANTOM on L = 2 time series under varying node counts d ∈ { 10 , 20 , 40 } and regime settings K = 2 and 3 .

| Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   | Heteroscedastic noise   |
|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| k                       | d                       |                         | Inst.                   | Inst.                   | Lag                     | Regime                  |
|                         |                         | SHD                     | F1                      | SHD                     | F1                      | Acc.                    |
|                         | 10                      | 1                       | 97.7                    | 0                       | 100.0                   | 94.3                    |
| 2                       | 20                      | 14                      | 87.0                    | 4                       | 98.7                    | 99.0                    |
| 40                      |                         | 15                      | 94.5                    | 3                       | 99.5                    | 98.9                    |
| 3                       | 10                      | 1                       | 98.5                    | 1                       | 99.5                    | 91.5                    |
|                         | 20                      | 5                       | 97.5                    | 11                      | 97.4                    | 97.9                    |
| 40                      |                         | 43                      | 88.6                    | 3                       | 99.0                    | 99.0                    |

±

±

±

±

FANTOM maintains similar performance with a larger lag ( L = 2 ). It achieves an F1 score above 87% on instantaneous links for 10, 20, and 40 nodes in MTS with 2 or 3 regimes. For time-lagged links, we evaluate using the summary graph of lagged relations; the table shows that FANTOM detects them effectively.

## F.3.1 Illustrations of learned graphs

Figure 10: The estimated temporal causal graphs for two regimes, in Heteroscedastic case, consist of one matrix of 10 rows and 10 columns representing instantaneous links and another of 10 rows and 10 columns delineating time-lagged relations (with a maximum lag L = 1 in this case). Dark red indicates a value of one (presence of an edge), while gray symbolizes a value of 0 (absence of an edge). The second column displays the ground-truth causal graphs, and the final column highlights the difference between the estimated and true graphs.

<!-- image -->

Figure 11: The estimated temporal causal graphs for three regimes, in Heteroscedastic case, consist of one matrix of 10 rows and 10 columns representing instantaneous links and another of 10 rows and 10 columns delineating time-lagged relations (with a maximum lag L = 1 in this case). Dark red indicates a value of one (presence of an edge), while gray symbolizes a value of 0 (absence of an edge). The second column displays the ground-truth causal graphs, and the final column highlights the difference between the estimated and true graphs.

<!-- image -->

Figure 12: The estimated temporal causal graphs for three regimes, in Heteroscedastic case, consist of one matrix of 40 rows and 40 columns representing instantaneous links and another of 40 rows and 40 columns delineating time-lagged relations (with a maximum lag L = 1 in this case). Dark red indicates a value of one (presence of an edge), while gray symbolizes a value of 0 (absence of an edge). The second column displays the ground-truth causal graphs, and the final column highlights the difference between the estimated and true graphs.

<!-- image -->

Figure 13: The estimated temporal causal graphs for three regimes, in Non-Gaussian case, consist of one matrix of 40 rows and 40 columns representing instantaneous links and another of 40 rows and 40 columns delineating time-lagged relations (with a maximum lag L = 1 in this case). Dark red indicates a value of one (presence of an edge), while gray symbolizes a value of 0 (absence of an edge). The second column displays the ground-truth causal graphs, and the final column highlights the difference between the estimated and true graphs.

<!-- image -->

## F.3.2 Time complexity analysis

We start first by computing the time complexity of our Temporal Graph Neural network illustrated in Figure 2. We note d the input size, e the embedding size used for e ij and h hidden layer size. After the first NN block,

<!-- formula-not-decoded -->

then after the matrix multiplication block and the second NN block, we have :

where L is the maximum lag.

<!-- formula-not-decoded -->

Using the same architecture for a Conditional normalizing flow has a time complexity of :

The complexity of FANTOM per iteration is O ( N w |T | (2 Ld 2 + h ( e + dK ) ) , where K is the number of bins, N w is the number of regimes, and |T | is the number of samples.

<!-- formula-not-decoded -->

## F.3.3 Regime detection experiments

We compare FANTOM to CASTOR [39] and KCP [1] in the task of regime detection. KCP is a multiple change-point detection method designed to handle univariate, multivariate, or complex data. Being non-parametric, KCP does not necessitate knowing the true number of change points in advance. It detects abrupt changes in the complete distribution of the data by employing a characteristic kernel.

CASTOR is a causal discovery model specifically designed for multi-regime MTS. CASTOR is learns number of regime and their indices and their corresponding causal graphs without any prior knowledge. But it is limited to normal noise with equivariance. FANTOM learns the regime indices, while handling heteroscedastic noises.

Weopted to perform regime detection with 10 nodes and three different regimes. For a fair comparison, we chose three regimes without re-occurrence, as KCP only detect change points and cannot identify the re-occurrence of a specific regime.

Regarding the models employed, we use the open-source code of CASTOR implemented in Python by the authors 7 . For KCP, we employ the Rupture package 8 .

Figure 14: Comparison between FANTOM, CASTOR and KCP on regime detection for a MTS with heteroscedastic noise and composed of 3 regimes using accuracy metric. The number of nodes is d = 10 .

<!-- image -->

From Figure 14, it is evident that FANTOM outperform CASTOR and the change-point detection method KCP. This outcome can be attributed to the limitation of KCP in detecting changing points within causal mechanisms that are represented by conditional distributions. FANTOM outperforms

7 https://github.com/arahmani1/CASTOR

8 https://centre-borelli.github.io/ruptures-docs/

CASTOR in detecting regime indices. This result can be explained by the fact that CASTOR fails in handling heteroscedastic noises and fails to learn meaningful graphs which also lead to poor performance in regime detection.

From this analysis and the other experiments shown in the different tables, we can conclude that in scenarios involving MTS with multiple regimes with non-Gaussian or Heteroscedastic noises, FANTOM offers a robust solution. Additionally, employing other methods to split the regimes and learn the causal graph through traditional causal discovery methods may not be an optimal solution:

- We demonstrate that regime indices are not well recoverable by other state-of-the-art change point detection method KCP. Therefore, employing KCP to learn the regimes and subsequently using methods like DYNOTEARS, PCMCI+, or Rhino to learn the graph may not constitute an optimal solution.
- In cases of regime recurrence, the aforementioned methods are unable to accurately detect the exact number of regimes. Therefore, if a user employs KCP and subsequently uses the regime partitions revealed by KCP as an input to a causal discovery method (such as PCMCI+, DYNOTEARS, Rhino, etc.), the running time will be significantly high.
- We show throughout all our experiments, that FANTOM outperforms all causal discovery method in DAG learning task, even when these models are in more favorable scenarios, by having access to the regime labels beforehand.

## G Proof of proposition

In this section, we are going to provide the entire proof of our proposition 3.1. As we state before, we note G = ( G r ) r ∈ [ | 1: N w | ] .

Proof. we have:

<!-- formula-not-decoded -->

Let's focus on the first term of our last inequality log p Θ ( x t | x &lt;t , G ) , we have:

<!-- formula-not-decoded -->

Including this result in the previous equation gives us the following:

<!-- formula-not-decoded -->

we note that the priors p ( G r ) and the variational estimations q ϕ r ( G r ) are independents.

## H Proofs of our theoretical contributions

In this section, we concentrate on establishing the identifiability of regimes and causal graphs within the FANTOM framework. Before diving into the details, let us set and clarify the required assumptions.

## H.1 Assumptions

Definition H.1 (Causal Stationarity, [41]) . A stationary time series process ( x t ) t ∈T with graph G is called causally stationary over a time index set T if and only if for all links x i t -τ → x j t in the graph

<!-- formula-not-decoded -->

This elucidates the inherent characteristics of the time-series data generation mechanism, thereby validating the choice of the auto-regressive model.

Assumption H.2 (Causal Stationarity for MTS with multiple regime) . A MTS ( x t ) t ∈T with K regimes, graph set ( G u ) u ∈ [ | 1: K | ] , and regime partition E = ( E u ) u ∈ [ | 1: K | ] is causally stationary over the time index set T if, for each regime u ∈ [ | 1 : K | ] , the sub-series ( x t ) t ∈E u is causally stationary with graph G u as defined in definition H.1.

Definition H.3. (Causal Markov Property, [37]). Given a DAG G and a joint distribution p , this distribution is said to satisfy causal Markov property w.r.t. the DAG G if each variable is independent of its non-descendants given its parents.

This is a common assumptions for the distribution induced by an SEM. With this assumption, one can deduce conditional independence between variables from the graph.

Assumption H.4 (Causal Markov Property (CMP)) . A set of joint distributions ( p ( ·|G r )) r ∈ [ | 1: K | ] satisfies the CMP with respect to the DAGs ( G r ) r ∈ [ | 1: K | ] if, for each r ∈ [ | 1 : K | ] , the distribution p ( ·|G r ) satisfies the CMP relative to the DAG G r . Specifically, in every regime r , each variable is independent of its non-descendants given its parents.

Assumption H.5 (Causal Minimality) . Given a set of DAGs ( G r ) r ∈ [ | 1: K | ] and a set of joint distribution ( p ( ·|G r )) r ∈ [ | 1: K | ] , we say that this set of distributions satisfies causal minimality w.r.t. the set of DAGs ( G r ) r ∈ [ | 1: K | ] if for every r : p ( ·|G r ) is Markovian w.r.t the DAG G r but not to any proper subgraph of G r .

Assumption H.6 (Causal Sufficiency) . A set of observed variables V is causally sufficient for a process x t if and only if in the process every common cause of any two or more variables in V is in V or has the same value for all units in the population.

This assumption implies there are no latent confounders present in the time-series data.

Table 21: Summary of the main assumptions of algorithms considered in the paper. For causal graphs, S means that the algorithm provides a summary causal graph and W means that the algorithm provides a window causal graph; F corresponds to faithfulness and M to minimality. An empty cell mean that the information given in the corresponding column was not discussed by the authors of the corresponding algorithm.

|           | Causal graph   | Causal Markov   | Causal sufficiency   | Faithfulness / Minimality   | Heteroscedastic noise   | Stationarity per regime   |
|-----------|----------------|-----------------|----------------------|-----------------------------|-------------------------|---------------------------|
| DYNOTEARS | W              | ✓               | ✓                    |                             | ×                       | ×                         |
| PCMCI+    | W              | ✓               | ✓                    | F                           | ×                       | ×                         |
| RPCMCI    | W              | ✓               | ✓                    | F                           | ×                       | ✓                         |
| Rhino     | W              | ✓               | ✓                    | M                           | ×                       | ×                         |
| CD-NOD    | S              | ✓               | ✓                    | F                           | ×                       | ✓                         |
| CASTOR    | W              | ✓               | ✓                    | M                           | ×                       | ✓                         |
| FANTOM    | W              | ✓               | ✓                    | M                           | ✓                       | ✓                         |

The table 21 illustrates that most assumptions (causal sufficiency, causal Markov, faithfulness/minimality) are commonly shared among various state-of-the-art models in causal discovery.

However, FANTOM, CASTOR, RPCMCI, and CD-NOD relax the assumption of stationarity and instead assume that the MTS (Multivariate Time Series) are composed of different regimes. While CD-NOD predicts only a summary causal graph, FANTOM, CASTOR and RPCMCI predict a window causal graph, which can subsequently be used to reconstruct a summary graph. FANTOM is the only model that can handle heteroscedastic noise.

## H.2 Proof of theorem 4.2

We start first by proving theorem 4.2. To do so, we will prove identifiability in the case of bivariate time series, Lemma H.7. Then we will prove identifiability in the case of MTS.

LemmaH.7. Assume Causal Markov property, minimality, stationarity, sufficiency and ( x 1 t , x 2 t ) t ∈T be a bivariate time series such that ( x 1 t , x 2 t ) t ∈T following Eq(1) where K = 1 and ϵ i t ∼ N (0 , 1) . We have x 1 t , x 2 t follow Gaussian distribution, if f 1 , f 2 are non linear and 1 g 1 , 1 g 2 are not a polynomial of degree two then the bivariate Temporal heteroscedastic Gaussian noise (THGNM) model is identifiable.

Proof of the Lemma. Let's assume we have two temporal causal graph G and G ′ for the bivariate TGHNM.

Disagreement in time lagged relationships. Assume that G and G ′ do not differ in the instantaneous effects. ∀ i ∈ { 1 , 2 } : Pa i G ( t ) = Pa i G ′ ( t ) . Hence and Wlog, there is some k &gt; 0 and an edge x 1 t -k → x 2 t in G but not in G ′ . From G ′ and the Causal Markov property, we have that x 1 t -k ⊥ ⊥ x 2 t | S , where S = ( { x i t -l , 1 ≤ l ≤ L, i ∈ { 1 , 2 }} ∪ ND t ) \ { x 1 t -k , x 2 t } , and ND t are all X i t that are nondescendants (wrt instantaneous effects) of x 2 t . Applied to G , causal minimality leads to a contradiction because x 1 t -k ̸ ⊥ ⊥ x 2 t | S in G , and the above reasoning shows that it exists a subgraph G ′ of G that is Markovian to the joint distribution of the data.

Disagreement on instantaneous parents. Now, let's assume we have a forward model, ∀ t ∈ T :

We will prove by contradiction that a backward model can not exists.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We note h t = Pa 1 G ( &lt; t ) ∪ Pa 2 G ( &lt; t ) . We know that ϵ 2 t ⊥ ⊥ ( x 1 t , Pa 1 G ( &lt; t ) , Pa 2 G ( &lt; t )) and ϵ 1 t ⊥ ⊥ ( x 2 t , Pa 1 G ( &lt; t ) , Pa 2 G ( &lt; t )) , using Lemma 36 in Peters et al. [38], we have:

where x i t | h t is x i t conditioned on h t . This last result contradicts the theorem states by Khemakhem et al. [29]. Hence, our bivariate TGHNM is identifiable.

Let's prove this results in the case of MTS (Theorem 4.2). In the case of Disagreement in time lagged relationships, we can use the same proof for the bivariate case.

Disagreement on instantaneous parents. Let's assume we have two temporal causal graph G and G ′ such that G ̸ = G ′ . According to the Propostion 28 in Peters et al [38], for G and G ′ be two different DAGs over a set of variables V , such that x t is generated by our HNM and satisfies the Markov condition and causal minimality with respect to G and G ′ . Then there are variables x 1 t , x 2 t ∈ V such that for the set Q := Pa 1 G ( t ) \{ x 2 t } , Y := Pa 2 G ′ ( t ) \{ x 1 t } and S := Q ∪ Y , we have: 1) x 2 t → x 1 t in G and x 1 t → x 2 t in G ′ . 2) S ⊆ ND G x 1 t \{ x 2 t } and S ⊆ ND G ′ x 2 t \{ x 1 t } . Pa 1 G ( t ) is the set of parent variables of x 1 t in graph G . ND G x 1 t is the set of non-descendant (wrt instantaneous effects) of x 1 t in graph G .

We consider S = s with p ( s ) &gt; 0 . Denote x 1 , ∗ t := x 1 t | S = s and x 2 , ∗ t := x 2 t | S = s . Lemma 37 in Peters et al. [38] states that if p ( x t ) is generated according to the SEM models as follows:

<!-- formula-not-decoded -->

with corresponding DAG G , then for a variable x i t ∈ V , if S ⊆ ND G x i t then ϵ i t ⊥ ⊥ S . Our TGHNM can be viewed one specific class of the SEM in the aforementioned equation. Hence, Lemma 37 holds under our TGHNM and renders ϵ 1 t ⊥ ⊥ ( x 2 t , S ) and ϵ 2 t ⊥ ⊥ ( x 1 t , S ) , using Lemma 36 in Peters et al. [38], we have:

<!-- formula-not-decoded -->

where x i t | h t is x i t conditioned on h t . This results contradict our previous proved Lemma, then THGNM is identifiable model under the conditions stated in the theorem.

Theorem H.8 (Identifiability of Temporal Non Gaussian noise model (TNGNM)) . Assume Causal Markov property, stationarity, minimality, sufficiency and let ( x t ) t ∈T be a MTS following a TNGNM, ∀ t ∈ T :

where f i is a differentiable function, and ϵ i t are mutually independent noises and follow a non Gaussian distribution. The TNGNM is identifiable.

<!-- formula-not-decoded -->

Proof. The proof of this theorem could be concluded from theorem 1 in Rhino [15]. Eq(14) is a special case of Rhino SEMs.

## H.3 Identifiability results in the case of Temporal General Heteroscedastic Noise Models

In this section, we will present our identifiability results for the case of Temporal General Heteroscedastic Noise, where a MTS has the following SEM : ∀ t ∈ T :

where f i and g i are differentiable functions, with g i strictly positive and ϵ i t are mutually independent normal noises and can have any arbitrary density distribution. We assume E ( ϵ i t ) = 0 and E (( ϵ i t ) 2 ) = 1 without loss of generality.

<!-- formula-not-decoded -->

We will start first by showing that if backward model, respects to instantaneous links, exists in the bivariate case then, the data generating mechanism must fulfill the a Partial Differential Equation (PDE). Then, following Peters et al. [38] and Strobl et al. [50] for defining Restricted SEM on iid, we will define a Temporal Restricted Heteroscedastic Noise model and show its identifiability.

LemmaH.9. Assume Causal Markov property, minimality, stationarity, sufficiency and ( x 1 t , x 2 t ) t ∈T be a bivariate time series. Then we have time lagged parents are identifiable, and a backward model with respect to instantaneous links can be fit i.e. ∀ t ∈ T :

We note h t = Pa 1 ( &lt; t ) ∪ Pa 2 ( &lt; t ) , and let ν 1 ( · ) and ν 2 ( · ) be the twice differentiable log densities of ˜ X 1 t and ϵ 2 t respectively. For compact notation, define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assume that f 1 , g 1 , f 2 , and g 2 are twice differentiable. Then, the data generating mechanism must fulfill the following PDE for all (˜ x 2 t , ˜ x 1 t ) with G (˜ x 2 t , ˜ x 1 t ) = 0 .

<!-- formula-not-decoded -->

̸

We drop the time-lagged parent in Eq (16) to simplify the notation, since conditioning on the history makes it redundant.

Proof of the Lemma. Let's assume we have two temporal causal graph G and G ′ for the bivariate temporal heteroscedastic causal models where the noise distribution could follow any arbitrary distribution.

Disagreement in time lagged relationships. Assume that G and G ′ do not differ in the instantaneous effects. ∀ i ∈ { 1 , 2 } : Pa i G ( t ) = Pa i G ′ ( t ) . Hence and Wlog, there is some k &gt; 0 and an edge x 1 t -k → x 2 t in G but not in G ′ . From G ′ and the Causal Markov property, we have that x 1 t -k ⊥ ⊥ x 2 t | S , where S = ( { x i t -l , 1 ≤ l ≤ L, i ∈ { 1 , 2 }} ∪ ND t ) \ { x 1 t -k , x 2 t } , and ND t are all X i t that are nondescendants (wrt instantaneous effects) of x 2 t . Applied to G , causal minimality leads to a contradiction because x 1 t -k ̸ ⊥ ⊥ x 2 t | S in G , and the above reasoning shows that it exists a subgraph G ′ of G that is Markovian to the joint distribution of the data.

Disagreement in instantaneous parents. Now, let's assume we have a forward model, after conditioning on h t = Pa 1 G ( &lt; t ) ∪ Pa 2 G ( &lt; t ) , ∀ t ∈ T :

We want to prove that if a backward model

<!-- formula-not-decoded -->

exists then the PDE in Eq(17) is fulfilled.

Our conditioning trick on time lagged parents makes the use of Immer et al. [25] theorem 1 feasible in our case. We employ the change of variables from { ˜ x 2 t , ϵ 1 t } to { ˜ x 1 t , ϵ 2 t } and the proof will be the same as Immer et al.. Hence, we leverage Theorem 1 of Immer el al. and we conclude that if a backward model exists the PDE Eq(17) is verified.

<!-- formula-not-decoded -->

Theorem H.10 (Identifiability of Temporal Restricted Heteroscedastic noise model (TRHNM)) . Assume Causal Markov property, minimality, sufficiency and let ( x t ) t ∈T be a MTS following Eq(1) where K = 1 . The graph G is uniquely identified if ∀ i ∈ [ | 1 : d | ] , ∀ j : x j t ∈ Pa i G ( t ) and S such that ( Pa i G ( t ) \ x j t ) ⊆ S ⊆ ( Nd ( x i t ) \ x j t ) , there exists S = s where p ( s ) &gt; 0 h t = Pa i G ( &lt; t ) ∪ Pa j G ( &lt; t ) and p ( x i t , x j t | s, h t ) do not satisfy PDE of Equation 17, and we call the model that verify this condition, the Temporal Restricted Heteroscedastic noise model.

Proof of the theorem. We will follow the same steps in the proof of theorem 4.2. Let's assume we have two temporal causal graph G and G ′ for the multivariate TRHNM. We assume also that ∀ i ∈ [ | 1 : d | ] , ∀ j : x j t ∈ Pa i G ( t ) and S such that ( Pa i G ( t ) \ x j t ) ⊆ S ⊆ ( Nd ( x i t ) \ x j t ) , there exists S = s where p ( s ) &gt; 0 h t = Pa i G ( &lt; t ) ∪ Pa j G ( &lt; t ) and p ( x i t , x j t | s, h t ) do not satisfy PDE of Equation 17.

We will start by showing that time lagged parents are identifiable. Same reasoning in the bivariate case.

Disagreement in time lagged relationships. Assume that G and G ′ do not differ in the instantaneous effects. ∀ i ∈ { 1 , 2 } : Pa i G ( t ) = Pa i G ′ ( t ) . Hence and Wlog, there is some k &gt; 0 and an edge x 1 t -k → x 2 t in G but not in G ′ . From G ′ and the Causal Markov property, we have that x 1 t -k ⊥ ⊥ x 2 t | S , where S = ( { x i t -l , 1 ≤ l ≤ L, i ∈ { 1 , 2 }} ∪ ND t ) \ { x 1 t -k , x 2 t } , and ND t are all X i t that are nondescendants (wrt instantaneous effects) of x 2 t . Applied to G , causal minimality leads to a contradiction

because x 1 t -k ̸ ⊥ ⊥ x 2 t | S in G , and the above reasoning shows that it exists a subgraph G ′ of G that is Markovian to the joint distribution of the data.

Disagreement on instantaneous parents. Let's now assume we have two temporal causal graph G and G ′ such that G ̸ = G ′ . According to the Propostion 29 in Peters et al [38], for G and G ′ be two different DAGs over a set of variables V , such that x t is generated by our TRHNM and satisfies the Markov condition and causal minimality with respect to G and G ′ . Then there are variables x 1 t , x 2 t ∈ V such that for the set Q := Pa 1 G ( t ) \{ x 2 t } , Y := Pa 2 G ′ ( t ) \{ x 1 t } and S := Q ∪ Y , we have: 1) x 2 t → x 1 t in G and x 1 t → x 2 t in G ′ . 2) S ⊆ ND G x 1 t \{ x 2 t } and S ⊆ ND G ′ x 2 t \{ x 1 t } . Pa 1 G ( t ) is the set of parent variables of x 1 t in graph G . ND G x 1 t is the set of non-descendant (wrt instantaneous effects) of x 1 t in graph G .

We consider S = s with p ( s ) &gt; 0 . Lemma 37 in Peters et al. [38] states that if p ( x t ) is generated according to the SEM models as follows:

<!-- formula-not-decoded -->

Using now Lemma 36 in Peters et al. [38], and we denote x 1 , ∗ t := x 1 t | S = s and x 2 , ∗ t := x 2 t | S = s . We have:

with corresponding DAG G , then for a variable x i t ∈ V , if S ⊆ ND G x i t then ϵ i t ⊥ ⊥ S . Our TRHNM can be viewed one specific class of the SEM in the aforementioned equation. Hence, Lemma 37 holds under our TRHNM and applying it to x 1 t renders ϵ 1 t ⊥ ⊥ ( x 2 t , S ) and x 2 t ϵ 2 t ⊥ ⊥ ( x 1 t , S ) .

<!-- formula-not-decoded -->

where x i, ∗ t | h t is x i t conditioned on h t , S and h t = Pa 1 G ( &lt; t ) ∪ Pa 2 G ( &lt; t ) in this case, which is also equal to h t = Pa 1 G ′ ( &lt; t ) ∪ Pa 2 G ′ ( &lt; t ) , because we proved identifiability of time lagged parents. Eq(18) raise a contradiction because, having these forward and backward models imply the verification of PDE 17. But we chose s such that this PDE is not verified hence contradiction. Then, the identifiability of our Temporal Restricted Heteroscedastic noise.

## H.4 Proof of theorem 4.3

In this section, we want to prove the identifiability of mixture of Temporal causal models either in the case of Temporal Heteroscedastic Gaussian noise, Temporal Restricted Heteroscedastic noise and Homoscedastic NonGaussian noise.

Proof. Let F be a family of identifiable temporal causal models either from TGHNM or TNGNM, F = ( p θ r ( ·|· , G r )) r ∈ N ∗ that are linearly independent and let M K be the family of all K -finite mixtures of elements from F , i.e.

First, we introduce a result from Yakowitz &amp; Spragins [58] that established a necessary and sufficient condition for the identifiability of finite mixtures of multivariate distributions.

<!-- formula-not-decoded -->

Theorem H.11 (Identifiability of finite mixtures of distributions, Yakowitz &amp; Spragins [58]) . . Let F = { F ( x ; α ) , α ∈ R m , x ∈ R n } be a finite mixture of distributions. Then F is identifiable if and only if F is a linearly independent set over the field of real numbers.

We will further assume that it exists two distribution such that ∀ ( x t , x &lt;t ) covering the space value of random variables ( X t , X &lt;t ) :

<!-- formula-not-decoded -->

Our objective is to show first that K = ˜ K , it exists a permutation σ and a translation function ϱ : R 2 → R 2 : ( θ r , G r ) = ( ˜ θ σ ( r ) , ˜ G σ ( r ) ) and ω r = ϱ (˜ ω σ ( r ) ) .

Using that F =( p θ r ( ·|· , G r )) K r =1 are linearly independent and fixing t :

<!-- formula-not-decoded -->

and this true ∀ ( x t , x &lt;t ) covering the space value of the random variable Y = ( X t , X &lt;t ) . By using theorem H.11, we can conclude that: K = ˜ K and it exists a permutation σ such that: ( θ r , G r ) = ( ˜ θ σ ( r ) , ˜ G σ ( r ) ) and ∀ t ∈ T : π t ( ω r ) = π t ( ˜ ω r ) .

<!-- formula-not-decoded -->

To proof our identifiability as defined in definition 4.1, we still need to prove that ω r = ϱ (˜ ω σ ( r ) ) . We have ∀ t ∈ T : π t ( ω r ) = π t ( ˜ ω r ) , we take two indices r, s ∈ [ | 1 : K | ] :

To handle time varying weights identifiability, we will consider ratios of mixture weights:

By Equation 20:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As a consequence of the last equation, we have for all the indices:

<!-- formula-not-decoded -->

Hence it exists a translation function ϱ : R 2 → R 2 , such that ∀ r ∈ [ | 1 : K | ] :

<!-- formula-not-decoded -->

Hence our mixture of temporal causal models is identifiable as defined in definition 4.1.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: Yes the main claims are clear in the abstract, introduction and in the whole paper

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Yes, we have a section in appendix D in which we talk about the limitation of our work.

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

Justification: Our work offers three principal theoretical contributions. All assumptions and complete proofs appear in Appendix H. Due to space constraint, we state two theorems in the main text and the third in the appendix, with each theorem explicitly citing its underlying assumptions.

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

Justification: Yes, we provide in the main text figures that illustrate the exact architecture used for our model. In the appendices, we provide the exact hyper-parameters needed to reproduce our results in all the presented experiments. Our code is also provided in the supplementary material.

Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.

In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Our code, with all the instruction to reproduce our results, is provided in the supplementary materials

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

Justification: All the detailed needed to run our code and to reproduce our results are presented in Appendix E.3.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: In the main text, we presented only average scores for the different metric. But, in the appendix F we provide erro bars for our model and all the other baselines.

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

Justification: We provide time complexity and running time in appendices F.3.2 and F.1 Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: -

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: In our conclusion, we talk about paper's broader impact.

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

Justification:

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: -

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

Justification: We presented all the details about our model: parameters, architecture, data. Also the detailed proofs are provided

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: -

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: -

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: -

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.