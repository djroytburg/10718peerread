## Dynamical Decoupling of Generalization and Overfitting in Large Two-Layer Networks

## Andrea Montanari

Department of Statistics and Department of Mathematics, Stanford University

## Pierfrancesco Urbani

Université Paris-Saclay, CNRS, CEA, Institut de Physique Théorique, 91191, Gif-Sur-Yvette, France

## Abstract

Understanding the inductive bias and generalization properties of large overparametrized machine learning models requires to characterize the dynamics of the training algorithm. We study the learning dynamics of large two-layer neural networks via dynamical mean field theory, a well established technique of nonequilibrium statistical physics. We show that, for large network width m , and large number of samples per input dimension n/d , the training dynamics exhibits a separation of timescales which implies: ( i ) The emergence of a slow time scale associated with the growth in Gaussian/Rademacher complexity of the network; ( ii ) Inductive bias towards small complexity if the initialization has small enough complexity; ( iii ) A dynamical decoupling between feature learning and overfitting regimes; ( iv ) A non-monotone behavior of the test error, associated 'feature unlearning' regime at large times.

## 1 Introduction

Machine learning (ML) models are trained using stochastic gradient descent (SGD), or one of its variants to minimize the error on training data (empirical risk function). Classically, their good behavior on unseen test data is explained by the fact that model complexity is kept small by regularization techniques: these models do not 'overfit.' Traditional ML theory decouples the analysis of the model from the optimization algorithm, which is assumed to converge to an approximate global minimizer [47].

In contrast, in modern ML, the empirical risk is highly non-convex, the number of parameters is comparable with the number of training samples, and the model complexity is only weakly controlled. As a consequence, there can be many assignments of the model parameters (many global empirical risk minimizers) that perfectly interpolate the data -even when these are noisy. While all of these interpolators are indistinguishable on the training data, they behave very differently (and some of them very poorly) on test data. It has been hypothesized that models trained by SGD generalize well to test data because the algorithm selects a near global minimizer with low complexity, although a mechanistic understanding of this process is lacking. For this reason, the generalization properties cannot be decoupled from the training dynamics.

Several striking consequences of this lack of decoupling are documented in the literature (and have long been familiar to practitioners): ( i ) Test error after training is observed to depend strongly on the initial weights distribution [28]; ( ii ) Test error depends strongly on the optimization algorithm (SGD, RMSProp, ADAM, to name a few), even when these algorithms achieve the same train error [55]; ( iii ) Careful choice of the hyperparameters in the optimization algorithm is crucial [34, 59], and the optimal choice is often different from the one that minimizes train error; ( iv ) Models learned by training for a shorter time have smaller complexity and can generalize better [44, 11].

Figure 1: Three dynamical regimes of learning in a two-layer neural networks, with m hidden neurons. Training data comprises n points in d dimensions distributed according to a single index model. We assume n, m, d all large with n/md = α (here α = 0 . 3 ). Blue: test error. Purple: train error. Red: ℓ 1 norm of second-layer weights (a proxy for model complexity).

<!-- image -->

These observations have motivated a broad effort to encapsulate the effect of the dynamics as 'implicit regularization' [48, 3, 15, 56]: the algorithm selects an empirical risk minimizer that also minimizes a specific notion of model complexity. While this implicit regularization hypothesis has been fruitful, it can only be validated if we can precisely understand the training dynamics.

In this work we leverage tools from theoretical physics to directly analyze the training dynamics and derive quantitative predictions on the implicit bias of neural network training, in a simple setting. This allows us to capture feature learning and lazy/overfitting regimes within the same unified picture. We discover a time-scale separation in the training dynamics, between an early stage in which the model learns the relevant features representation of the data, and a late stage of training that is characterized by overfitting, feature 'unlearning,' and hence test error that increases with training. While the regularizing effect of early stopping has been an important object of study (for simpler models) in the past [44, 11, 61, 57], our work is the first to point out a time-scale separation between feature learning (on a faster timescale) and overfitting (on a slower time scale), thus reconciling the feature learning and neural tangent theories of learning.

We study two-layer fully connected neural networks f ( · ; θ ) : R d → R , i.e.

where θ = ( a , W ) , where W = ( w 1 , . . . , w m ) ∈ R d × m and a = ( a 1 , . . . , a m ) ∈ R m are, respectively, first- and second-layer weights. For convenience, we fix the normalization ∥ w i ∥ = 1 , and assume that σ does not depend on m . We apply model (1.1) to a supervised learning task. We are given i.i.d. data ( y i , x i ) , i ≤ n , with y i ∈ R a response variable and x i ∈ R d a feature vector, and try to learn a model f ( · ; θ ) to predict the response y new corresponding to a new input x new. We use gradient flow (GF) to minimize the empirical risk under square loss, namely

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here P θ is a projection matrix that guarantees that w i ( t ) ∈ S d -1 at all times. The factor n/d is introduced for convenience and simply amounts to a rescaling of time. We will typically initialize the training by setting ( w i ) i ≤ m ∼ iid Unif ( S d -1 ) , and a i = a 0 for all i ≤ m , and study the dependence of the training dynamics on three key parameters:

Network width: m, Overparametrization ratio: α := n md , Initialization scale: a 0 .

Alongside the train error, we will be interested in the test error at time t , i.e. R ( θ ( t )) := E { ( y new -f ( x new ; θ ( t ))) 2 } / 2 , and the generalization error R ( θ ( t )) -R n ( θ ( t )) .

̂ Model (1.1) is much simpler than state-of-the-art architectures [52], but is rich enough to investigate several general questions, which we summarize below:

When the network is sufficiently overparametrized ( α small) and a 0 is large, neural tangent kernel (NTK) theory predicts that GF converges to an interpolator [30, 22, 16] .

- Q1. For which region of α, a 0 does convergence take place, beyond NTK theory?
- Q2. Does the selected model provide good generalization or not [27, 37]?

In contrast, when a 0 is small, gradient-based algorithms can learn non-linear low-dimensional representation of the data [5, 21, 1, 6]. In these results, the difference between train and test error (generalization error) is negligible: the model does not overfit.

- Q3. Can we reconcile this feature-learning/no-overfitting behavior with the lazytraining/overfitting regime described previously?

In the early phase of training, the generalization error vanishes. However, training longer times can be beneficial, despite leading to overfitting.

Q4. When does the test error start increasing with training time? When should we stop training?

Finally, scaling with the network size is crucial:

- Q5. How does the generalization error depend on network size and number of iterations?
- Q6. Does overfitting start earlier for larger networks or later?

In Section 2, we will present our analysis using theoretical physics techniques. Section 3 presents rigorous results confirming the picture emerging from this analysis. Finally, in Section 4 we discuss how our results address the above questions.

## 2 Main results: Dynamical mean field theory

Westudy the dynamics of model (1.1) under the simplest data distribution in which genuine non-linear learning is required to efficiently learn a good prediction rule, the so called k -index model . Namely, we assume x i ∼ N (0 , I d ) and y i that depends on a low-dimensional projection U T x i :

<!-- formula-not-decoded -->

where the noise ε i is independent of x i , U ∈ R d × k is an orthogonal matrix ( U T U = I k ) and φ : R k → R is a nonlinear function, E { φ ( g ) 2 } &lt; ∞ for g standard Gaussian.

An important aspect of this data distribution is that (for large d ) it presents the largest possible gap between linear/kernel learning, which requires sample size to be superpolynomial in d [27, 58], and nonlinear/neural network learning which only requires n = O ( d ) (generically, for constant k ). When the dimension d becomes large, discovering the latent features U T x is crucial for learning and requires nonlinear processing of the labels y i [5, 21, 1, 6].

Our main focus will be on the simplest case, namely k = 1 , with φ a generic function (in particular E { φ ( G ) G } ̸ = 0 for G ∼ N (0 , 1) , which corresponds information exponent equal to one according to the classification of [4].). Some of our results apply to k -index models for general fixed k (in particular, the rigorous results of Section 3). We defer to future work a more complete analysis of the DMFT for k ≥ 2 .

We discover a separation of time scales at large m (or large n/d ), for sufficiently small initialization a 0 : feature learning takes place on a fast time scale, followed by overfitting/reversal to kernel learning. This scenario is summarized in Figure 1, which plots numerical evaluations of our theoretical predictions at k = 1 , τ &gt; 0 data distribution, in the limit n, d, m →∞ at overparametrization ratio α = 0 . 3 .

More precisely, we observe three regimes (below W 2nd := a /m is the vector of second-layer weights in model (1.1)):

( i ) Mean field feature learning. t = O (1) . The network learns the low-dimensional features U T x ; the train error and test error decrease while their difference (generalization error) is negligible; the second layer weights remain small ∥ W 2nd ∥ 1 = O (1) .

( iii ) Overfitting and feature unlearning. t ≳ m . Train error and test error diverge significantly, i.e. R ( θ ( t )) -̂ R n ( θ ( t )) becomes of order one. At the end of this regime, the train error converges to 0 , i.e. the neural network interpolates the noisy data. The test error instead grows, and its limit value is the one of a (data independent) kernel method: in other words, the model unlearns the low-dimensional structure. Finally, the second weights grow to ∥ W 2nd ∥ 1 ≍ √ m , which indeed is the scale required for interpolation.

( ii ) Extended feature learning. 1 ≪ t ≪ m . The train error decreases slowly; the generalization error increases is small, i.e. R ( θ ( t )) -̂ R n ( θ ( t )) = o (1) ; the test error can evolve non-monotonically, but remains approximately constant. Second-layer weights become large 1 ≪∥ W 2nd ∥ 1 ≪ √ m .

In this section we outline our results based on 'dynamical mean field theory' (DMFT). The next section will present rigorous results that are proven independently.

## 2.1 Technique

Our DMFT analysis is based on the following two steps:

Step 1: We leverage techniques from theoretical physics to derive an approximate asymptotic characterization of the gradient flow dynamics (1.2) in the limit n, d → ∞ , with n/d → α . This characterization consists of a set of integral-differential equations for the following asymptotic quantities (here p-lim denotes limit in probability, and we use the superscripts n to emphasize the dependence of the right-hand side on n, d )

<!-- formula-not-decoded -->

A rigorous derivation of the DMFT in a setting that includes two-layer networks is given in [13].

However, the asymptotically exact DMFT characterization of [13] is rather complex to integrate numerically or to study analytically. In order to circumvent this problem, we use a DMFT that is is asymptotically exact for a well-defined Gaussian version of the original model. Namely, we observe that the empirical risk of Eq. (1.2) takes the form

<!-- formula-not-decoded -->

̂ The Gaussian approximation comes with an error which we show analytically is vanishing on time scales of order one ( indeed on these time scales we correctly recover the mean field theory of [38, 14]) and we demonstrate empirically to be small on larger time scales ( see for instance example Fig. 4.) The curves in Fig. 1 were obtained by solving numerically the DMFT equations, see Appendix C for details.

where F : ( S d -1 ) m × R m → R n is s stochastic process with i.i.d. components F i ( θ ) = y i -f ( x i ; θ ) . We replace these by Gaussian processes with matching mean and covariance, and study the DMFT for gradient flow with respect to the associated risk R g n ( θ ) .

Step 2: We study this DMFT, with special attention to the large network limit m →∞ , and large sample size α →∞ , with α = α/m fixed, for a generic single index model ( k = 1 ). We obtain a separation of time scales in the dynamics, corresponding to distinct learning regimes.

a

6

5

4

3

2

1

0

10

-

1

m

m

m

m

m

m

= 2

= 2

= 2

= 2

= 2

= 2

4

5

6

7

8

9

10

0

10

t

Figure 2: Evolution of second-layer weights (left) and train error (right) when fitting pure noise data . Here we use mean field initialization, h ( z ) = (9 / 10) z + (1 / 6) z 3 , α = 0 . 4 and τ = 0 . 6 . Symbols: SGD results on actual 2-layer networks with d = 200 , n = αmd (averaged over 10 simulations). Continuous viridis lines: Numerical solution of the DMFT equations. Note that the second layer weights are given in terms of a scalar quantity as the result of the statistically symmetric initialization.

<!-- image -->

The analysis of the DMFT equations in the double limit m,t → ∞ is an example of singular perturbation theory [9, 29]. Making this type of analysis rigorous is notoriously challenging and we proceed by a combination of numerical solutions and analytical derivations.

In the following, we will first consider the simplest possible setting, pure noise data, and subsequently consider the single-index model. The structure of the activation function and target nonlinearity will be encoded in the functions h ( q ) := E { σ ( G 1 ) σ ( G q ) } , ̂ φ ( q ) := E { φ ( G 1 ) σ ( G q ) } , where G 1 , G q are standard jointly Gaussian with E { G 1 G q } = q . The relation between σ, φ and h, ̂ φ is conveniently expressed in terms of the expansions in Hermite polynomials σ ( x ) = ∑ k ≥ 0 s k He k ( x ) , φ ( x ) = ∑ k ≥ 0 f k He k ( x ) , which corresponds to the analytic expansion h ( q ) = ∑ k ≥ 0 s 2 k q k , φ ( q ) = ∑ k ≥ 0 s k f k q k .

In Section 3 we present rigorous results that do not require either of these simplifying assumptions.

̂ As mentioned above, we assume throughout n, d → ∞ , with n/d → α ∈ (0 , ∞ ) , with the limit m,α →∞ taken afterwards. To further simplify our analysis, we assume a symmetric initialization whereby a i (0) = a 0 is independent of i ≤ m and ( w i (0) : i ≤ m ) ∼ iid Unif ( S d -1 ) . Throughout, we use 'with high probability' for 'with probability converging to one as n, d →∞ .'

## 2.2 Training on pure noise

We begin by the case in which the data is pure noise: y i = ε i ∼ N (0 , τ 2 ) . A by-now-classic experiment [60] showed that deep learning models have sufficient capacity to achieve vanishing training error even when actual labels are replaced by random ones: they 'interpolate pure noise.'

The ability of a model F Θ = ( f ( · ; θ ) : θ ∈ Θ) to interpolate pure noise is intimately connected to its Gaussian complexity G ( F Θ ; n ) := E sup θ ∈ Θ ⟨ g , f ( X ; θ ) ⟩ /n [53] (where g ∼ N ( 0 , I n ) is independent of f ( X , ; θ ) = ( f ( x i ; θ ) : i ≤ n ) . Indeed, interpolation is impossible unless G ( F Θ ; n ) ≥ τ . Viceversa, G ( F Θ ; n ) ≪ τ ensures good generalization.

By a theorem of [7] for the network (1.1), G ( F Θ ; n ) ≤ L σ ∥ a /m ∥ 1 √ d/n (with L σ depending uniquely on σ ). This means that, in order to interpolate noise, the average magnitude of second layer weights must be ∥ a /m ∥ 1 ≥ L -1 σ τ √ n/d = ( L -1 σ α 1 / 2 ) τ √ m . However, complexity bounds do not have implications on the convergence of GF to an interpolator.

Figure 2 compares the DMFT predictions to simulations using SGD to train an actual two layer networks. In this figure we initialize a (0) = 1 , and let a ( t ) evolve with GF alongside the first layer weigths. We observe that the theory describes well the empirical results, despite the Gaussian

1

10

2

10

3

<!-- image -->

tm

t

Figure 3: Train/test error (right) when fitting data from a single index model . We set h ( z ) = ̂ φ ( z ) = (9 / 10) z + z 2 / 2 , τ = 0 . 3 and α = 0 . 3 . Lines correspond to predictions from the DMFT (continuous: train error; dashed: test error). Black continuous line is the m → ∞ value. Right: Same data plotted versus t .

approximation in our DMFT and the difference between SGD and GF. We also observe that secondlayer weights remain roughly constant until a large time t # ( m ) , which appears to increase with m . Roughly at the same time, train error starts to decrease and converges to zero.

In Section G.1 of the appendix, we will make precise the above picture of the evolution of a ( t ) . Here, we consider a simplified setting in which a ( t ) = γ √ m with γ independent of m , not evolving with training. Note that G ( F Θ ; n ) ≍ γ/ √ α and hence such a network can interpolate pure noise if γ is larger than threshold depending on α . Our DMFT predicts a sharp phase transition. For α ∈ (0 , 1) , GF converges to vanishing train error with high probability if γ &gt; γ GF ( α, m ) τ , and converges to a strictly positive training error if γ &lt; γ GF ( α, m ) τ . The threshold γ GF ( α, m ) converges to a limit γ ∗ GF ( α ) ∈ (0 , 1) as m →∞ .

A rephrasing of the same phenomenon states that lim n,d →∞ R g n ( θ ( t )) = e tr ( t ; m,γ ) , and

<!-- formula-not-decoded -->

Informally γ ∗ GF ( α ) is the minimum complexity γ for a very large network to interpolate noise via gradient flow. The functions γ ∗ GF ( α ) , e ∗ ( γ ) will play an important role below.

We will next consider training on data from a single-index model. The initial scale of secondlayer weights ∥ a (0) /m ∥ 1 plays a crucial role and we will separately analyze lazy and mean field initializations.

## 2.3 Training on data with latent structure: lazy initialization

We initialize a (0) = γ 0 √ m , and let a ( t ) evolve according to GF alongside first-layer weights. DMFT predicts the emergence of three dynamical regimes for large m and large α (with n/d → α ). For an illustration, we refer to Fig. 3.

First dynamical regime: t = O (1 /m ) . Second layer weights do not change significantly γ ( t ) = γ 0 + o m (1) , while first layer-weights move by ∥ w i ( t ) -w i (0) ∥ = Θ(1 / √ m ) . Because the weights a i ( t ) are of order √ m , even an O (1 / √ m ) change in the w i leads to a significant decrease in test error and train error.

Train and test error are close to each other. Namely, the following limits are well defined

<!-- formula-not-decoded -->

For large scaled time ˆ t , the error e lz1 ( ˆ t ; φ, γ 0 , α ) converges to the error of the best linear approximation to f ∗ . This dynamical regime follows the qualitative predictions of NTK theory, and is essentially linear in the weights w i , but the time is too short for the model to overfit the data.

Second dynamical regime: t = Θ(1) . Second layer weights do not change significantly: γ ( t ) = γ 0 + o m (1) , while first layer weights change significantly ∥ w i ( t ) -w i (0) ∥ = Θ(1) . However they change orthogonally to the latent subspace U and hence the test error does not change: no actual learning takes place in this regime, but the model starts to overfit the data.

More formally, train and test error have well defined limits as the network width diverges:

<!-- formula-not-decoded -->

However, the scaling function e lz2 ts ( t ; φ, γ 0 , α ) for the test error is constant in time and equal to the value achieved at the end of the first dynamical regime. Namely

<!-- formula-not-decoded -->

Since the w i 's move orthogonally to the latent space, their dynamics is equivalent (for large m ) to the one in the pure noise setting, modulo a redefinition of h . The right plot in Fig. 3 illustrates this.

Third dynamical regime: t = Θ( m ) . The qualitative properties of this regime depend whether or not γ 0 is larger than an interpolation threshold γ ∗ GF ( α, φ, τ ) , which generalizes the threshold γ ∗ GF ( α ) = γ ∗ GF ( α, 0 , 1) introduced in the pure noise case. Because the dynamics of weights w i in the subspace orthogonal to U is equivalent to dynamics in pure noise, we expect the interpolation threshold γ ∗ GF ( α, φ, τ ) to be given in terms of pure noise threshold γ ∗ GF ( α ) as follows:

<!-- formula-not-decoded -->

For γ 0 &gt; γ ∗ GF ( α, φ, τ ) , interpolation is achieved during the second dynamical regime, no further evolution takes place.

For γ 0 &lt; γ ∗ GF ( α, φ, τ ) , a non-trivial evolution takes place for t = Θ( m ) . Introducing the rescaled time z ∈ (0 , ∞ ) , we obtain, as m →∞ ,

<!-- formula-not-decoded -->

Further, for large values of the rescaled time z →∞ , γ lz3 ( z ) grows to γ ∗ GF ( α, φ, τ ) ≈ γ ∗ GF ( α, φ, τ ) , while e lz3 tr ( z ) decreases to 0 . In other words, interpolation is achieved on this third regime.

Further the test error e lz3 ts ( z ) increases from e lz2 ts ( t ; φ, γ 0 , α ) to e lz2 ts ( t ; φ, γ ∗ GF , α ) , with γ ∗ GF = γ ∗ GF ( α, φ, τ ) whereby e lz2 ts ( · · · ) is given by Eq. (2.7).

## 2.4 Training on data with latent structure: mean field initialization

We initialize a (0) = a 0 , independent of m and let second layer weights evolve. Note that at initialization the network's Rademacher complexity is small, namely of order a 0 √ d/n = a 0 / √ αm . Our DMFT analyisis predicts two dynamical regimes for large m . We will refer to them as 'first' and 'third regime' for consistency with other settings ( see Sec.G.2 of the appendix). For an illustration, we refer to Figs. 4 and 5.

First dynamical regime: t = O (1) . Both first and second layer weights change by order one: a ( t ) = a 0 +Θ(1) and ∥ w i ( t ) -w i (0) ∥ = Θ(1) . and as a consequence test and train error decrease significantly. In this regime, the two errors remain close to each other and their evolution is well captured by the mean field theory of [38, 14], as specialized to the case of spherically invariant distributions [10, 2].

Namely, lim m →∞ a ( t ) = a mf1 ( t ) , lim m →∞ v ( t ) = v mf1 ( t ) , and DMFT reduces to a system of k +1 ordinary differential equations for the k +1 scalar variables ( a mf1 ( t ) , v mf1 ( t ))

<!-- formula-not-decoded -->

where Q v := I k -vv T . As mentioned above, train and test error coincide in the large width limit

<!-- formula-not-decoded -->

<!-- image -->

t

t

Figure 4: Training dynamics under a single-index model. We set h ( q ) = ̂ φ ( q ) = (9 / 10) q + q 3 / 6 , τ = 0 . 3 and α = 0 . 3 , under mean field initialization. Left: second-layer weights. Right: train and test error. Symbols are empirical results for SGD with actual two-layer neural networks with d = 200 , n = αmd (averaged over 10 simulations). Lines correspond to predictions from the DMFT (on the right, continuous: train error; dashed: test error).

An explicit formula for e mf1 ( t ) is given in Appendix G.2.1. In the case k = 1 and ̂ φ ( z ) = h ( z ) , we have that a mf1 = 1 , v mf1 = 1 is a fixed point of Eq. (2.10), and indeed the only fixed point with v mf1 &gt; 0 . If h ′ (0) &gt; 0 , then, we have ( a mf1 ( t ) , v mf1 ( t )) → (1 , 1) as t →∞ , and therefore test and train error converge to the Bayes error e mf1 ( t ) → τ 2 / 2 . This is significantly smaller than the test error achieved with lazy initialization. The separation between lazy and mean-field initialization is expected because feature learning takes place in the mean field regime.

Third dynamical regime: t = Ω( m ) . Computing the local stability of DMFT solutions around the mean field asymptotics (see Appendix G.2.2) suggests that the latter breaks down for t = Θ( m ) . For t ≳ m , we observe that the second layer weights grow to achieve a ( t ) ≍ √ m , the projection onto the latent space decreases to v ( t ) ≍ 1 / √ m , and train and test error diverge, eventually achieving e tr ( t ) ≈ 0 and test error significantly larger than the Bayes error achieved earlier. We refer to this phenomenon as 'feature unlearning.'

<!-- formula-not-decoded -->

Denoting by t 0 ( m ; c ) the time at which a ( t ) = c √ m (for c a small constant), we expect the existence of a window size w ( m ) such that where γ mf3 ( z ) , e mf3 tr ( z ) , e mf3 ts ( z ) are scaling functions describing the dynamics on this timescale. We expect t 0 ( m ; c ) = t ∗ ( c ) m + o ( m ) , and w ( m ) ≲ t 0 ( m ; c ) , but our numerical solutions are not sufficient to determine the precise scaling. On the other hand, it appears that at large times, the complexity converges close the interpolation threshold:

<!-- formula-not-decoded -->

Finally, the evolution of train and test error for a ( t ) ≍ √ m appears to match the behavior at fixed second-layer weights. Namely, we define two functions

<!-- formula-not-decoded -->

We observe that the limit curves ( γ, ε mf tr ( γ )) , ( γ, ε mf ts ( γ )) , match closely asymptotic train and test error obtained by fixing a ( t ) = γ √ m , and not letting second-layer weight evolve. This confirms the hypothesis that γ ( t ) is a slow variable, while others converge as if γ was fixed.

## 3 Lower bounding the overfitting timescale

In this section we rigorously establish two results that confirm elements of the scenario outlined in the previous sections. We emphasize that the result presented here are non-asymptotic, i.e. hold at finite

Figure 5: Left: second layer weights on the scale √ m as a function of t/m . Curves appear to collapse on a master curve. The red arrow denotes γ ∗ GF and the curves appear to converge to that limit. Center: the projection of the first layer weights on the latent space in the single index model as a function of time on timescales of order m . Right: difference between test and train error as a function of the second layer weights on the scale √ m . The finite m curve are approaching a scaling curve which coincides with the one obtained by evaluating the same quantity but with a lazy initialization and fixed second layer weights.

<!-- image -->

n, m, d modulo unspecified absolute constants. Further, we do not assume a symmetric initialization of the weights. Throughout this section setting, it is more convenient to rescale time defining ˆ t = tα . Hence. instead of the flow (1.2), we study

For α = Θ(1) the parametrizations t and ˆ t are equivalent.

<!-- formula-not-decoded -->

The first result of this section implies that (under mean field initialization) overfitting cannot take place on times of order one.

Theorem 1. Under the GF dynamics (1.2) , and the data distribution in the introduction (with k arbitrary), further assume ∥ σ ∥ Lip , ∥ σ ∥ ∞ ≤ L , | φ (0) | , ∥ φ ∥ Lip ≤ L , ∥ a (0) ∥ ∞ ≤ a 0 , for some a 0 ≥ 1 and that the w i (0) , i ≤ m are independent of the data { ( y i , x i ) : i ≤ n } . Finally assume n ≥ d ∨ m . Then, there exist universal constants C 0 , C 1 , and the following holds for all ˆ t ≥ 0 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under mean field initialization, a 0 is a fixed constant and hence a 1 is also bounded, whence the generalization error in Eq. (3.3) is small as long as ˆ t = o (( n/d ) 1 / 4 ) (equivalently, for α fixed, ˆ t = o ( m 1 / 4 ) ).

By itself, this result implies a separation of timescales between learning and overfitting, thus confirming the picture developed within DMFT, but falls short of characterizing the overfitting timescale.

The second result implies that, up to time-scale of order one, the dynamics is closely tracked by the mean field equations (2.10). Since the a i (0) at initialization are not necessarily all equal, these are generalized as

<!-- formula-not-decoded -->

The mean field prediction for test error is the same as for training error and given by

<!-- formula-not-decoded -->

Theorem 2. Under the the GF dynamics (1.2) , and the data distribution in the introduction (with k arbitrary), further assume that ∥ φ ∥ ∞ , ∥ φ ′ ∥ ∞ , ∥ φ ′ ∥ Lip ≤ L , ∥ σ ∥ ∞ , ∥ σ ′ ∥ ∞ , ∥ σ ′ ∥ Lip ≤ L . Further

assume | a i (0) | ≤ L for all i ≤ m , ( w i (0)) i ≤ m ∼ iid Unif ( S d -1 ) . Then for any δ &gt; 0 there exist constants c 0 c 1 , C depending on L, τ, δ, k such that, letting T lb = c 0 (log m ) 1 / 3 ∧ (log n/d ) 1 / 3 , the following happens with probability at least 1 -2 exp( -c 1 d ) ,

<!-- formula-not-decoded -->

Remark 3.1. While the analysis in the previous section requires m → ∞ after n, d → ∞ , neither Theorem 3.1 nor Theorem 3.2 make the assumption. In particular, Eq. (3.3) implies that the generalization error is small for ˆ t = o (( n/d ) 1 / 4 ) irrespective of m .

<!-- formula-not-decoded -->

Similarly, Eqs. (3.5), (3.6) imply that the mean field theory of [38, 14, 45] captures well the evolution of the system for times t = o ((log m ) 1 / 3 ∧ (log n/d ) 1 / 3 ) .

## 4 Discussion

We conclude by highlighting a few qualitative conclusions of our work, and how they address questions raised in Section 1. In the following remarks, we consider α = n/md as constant.

Interpolation mechanism. In the current setting, the neural model complexity is proportional to ∥ a ( t ) ∥ 1 / √ m = γ ( t )+ o n (1) . Weobserve two alternative scenarios. If the complexity at initialization is large enough γ 0 &gt; γ ∗ GF ( α ) τ , then the gradient flow rapidly converges to a near interpolator without significant change in γ ( t ) . If instead, γ 0 &lt; γ ∗ GF ( α ) τ , then γ ( t ) grows to reach the interpolation threshold at which point the training error converges to 0 .

Adiabatic evolution of model complexity. In the latter case, the complexity γ ( t ) evolves on a slower time scale than other degrees of freedom. The dynamics on shorter timescales is well approximated by the one at fixed γ (given by the current value γ ( t ) ). The generalization error becomes of order one only when γ ( t ) is of order one.

Decoupling of learning and overfitting. When γ 0 = o m (1) , the fact that γ ( t ) acts as a slow variable implies a largem decoupling between learning (which takes place on faster timescales, as long as γ ( t ) = o m (1) ), and overfitting (which takes place on slower timescales, when γ ( t ) = Ω m (1) ). This has several implications for the questions outlined in the introduction.

Q3 : Lazy initialization a (0) ≍ √ m leads to poor generalization because the feature-learning phase is skipped either partially or altogether.

- Q2 : Training until interpolation is generally suboptimal.
- Q4 : The optimal tradeoff is obtained at the end of the first phase.
- Q5, Q6 : Further, at fixed overerparametrization n/md = α , overfitting starts later for larger models.

Overfitting and feature unlearning. The above description points at a non-monotonicity of the model quality, which improves on short time scales, and deteriorates at larger time scales. Reciprocally, early stopping acts as a regularization. While this phenomenon is well understood for linear models [24, 57], our analysis provides an analogous (quantitative) scenario for training neural network models. In particular, it clarifies the underlying mechanism: in the same dynamical regime in which network complexity grows ( γ ( t ) becomes of order one), and training error becomes negligible, the low-dimensional latent features are 'unlearned' ( v ( t ) becomes of order 1 / √ m ). We expect that these findings also allow to understand the beneficial effect of regularization on the second layer.

## Acknowledgments

This work was supported by the NSF through award DMS-2031883, the Simons Foundation through Award 814639 for the Collaboration on the Theoretical Foundations of Deep Learning, and the ONR grant N00014-18-1-2729. This work was supported by the French government under the France 2030 program (PhOM - Graduate School of Physics) with reference ANR-11-IDEX-0003.

## References

- [1] Emmanuel Abbe, Enric Boix Adsera, and Theodor Misiakiewicz. The merged-staircase property: a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer neural networks. In Conference on Learning Theory , pages 4782-4887. PMLR, 2022.
- [2] Luca Arnaboldi, Ludovic Stephan, Florent Krzakala, and Bruno Loureiro. From highdimensional and mean-field dynamics to dimensionless odes: A unifying approach to sgd in two-layers networks. In The Thirty Sixth Annual Conference on Learning Theory , pages 1199-1227. PMLR, 2023.
- [3] Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. Implicit regularization in deep matrix factorization. Advances in Neural Information Processing Systems , 32, 2019.
- [4] Gerard Ben Arous, Reza Gheissari, and Aukosh Jagannath. Online stochastic gradient descent on non-convex losses from high-dimensional inference. Journal of Machine Learning Research , 22(106):1-51, 2021.
- [5] Jimmy Ba, Murat A Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu, and Greg Yang. Highdimensional asymptotics of feature learning: How one gradient step improves the representation. Advances in Neural Information Processing Systems , 35:37932-37946, 2022.
- [6] Boaz Barak, Benjamin Edelman, Surbhi Goel, Sham Kakade, Eran Malach, and Cyril Zhang. Hidden progress in deep learning: SGD learns parities near the computational limit. Advances in Neural Information Processing Systems , 35:21750-21764, 2022.
- [7] Peter Bartlett. For valid generalization the size of the weights is more important than the size of the network. Advances in neural information processing systems , 9, 1996.
- [8] Gérard Ben Arous, Amir Dembo, and Alice Guionnet. Cugliandolo-kurchan equations for dynamics of spin-glasses. Probability theory and related fields , 136(4):619-660, 2006.
- [9] Nils Berglund. Perturbation theory of dynamical systems. arXiv preprint math/0111178 , 2001.
- [10] Raphaël Berthier, Andrea Montanari, and Kangjie Zhou. Learning time-scales in two-layers neural networks. Foundations of Computational Mathematics , pages 1-84, 2024.
- [11] Christopher M Bishop. Regularization and complexity control in feed-forward networks. In Proceedings International Conference on Artificial Neural Networks ICANN'95 , pages 141-148, 1995.
- [12] Blake Bordelon and Cengiz Pehlevan. Self-consistent dynamical field theory of kernel evolution in wide neural networks. Advances in Neural Information Processing Systems , 35:32240-32256, 2022.
- [13] Michael Celentano, Chen Cheng, and Andrea Montanari. The high-dimensional asymptotics of first order methods with random data. arXiv:2112.07572 , 2021.
- [14] Lenaic Chizat and Francis Bach. On the global convergence of gradient descent for overparameterized models using optimal transport. Advances in neural information processing systems , 31, 2018.
- [15] Lenaic Chizat and Francis Bach. Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss. In Conference on learning theory , pages 1305-1338. PMLR, 2020.
- [16] Lenaic Chizat, Edouard Oyallon, and Francis Bach. On lazy training in differentiable programming. Advances in neural information processing systems , 32, 2019.
- [17] Andrea Crisanti, Heinz Horner, and H J Sommers. The spherical p-spin interaction spin-glass model: the dynamics. Zeitschrift für Physik B Condensed Matter , 92:257-271, 1993.
- [18] Leticia F Cugliandolo. Recent applications of dynamical mean-field methods. Annual Review of Condensed Matter Physics , 15, 2023.

- [19] Leticia F Cugliandolo and David S Dean. Full dynamical solution for a spherical spin-glass model. Journal of Physics A: Mathematical and General , 28(15):4213, 1995.
- [20] Leticia F Cugliandolo and Jorge Kurchan. Analytical solution of the off-equilibrium dynamics of a long-range spin-glass model. Physical Review Letters , 71(1):173, 1993.
- [21] Alexandru Damian, Jason Lee, and Mahdi Soltanolkotabi. Neural networks can learn representations with gradient descent. In Conference on Learning Theory , pages 5413-5452. PMLR, 2022.
- [22] Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global minima of deep neural networks. In International conference on machine learning , pages 1675-1685. PMLR, 2019.
- [23] Giampaolo Folena, Silvio Franz, and Federico Ricci-Tersenghi. Rethinking mean-field glassy dynamics and its relation with the energy landscape: The surprising case of the spherical mixed p-spin model. Physical Review X , 10(3):031045, 2020.
- [24] Jerome Friedman, Trevor Hastie, and Robert Tibshirani. Additive logistic regression: a statistical view of boosting (with discussion and a rejoinder by the authors). The annals of statistics , 28(2):337-407, 2000.
- [25] Yan V Fyodorov. A spin glass model for reconstructing nonlinearly encrypted signals corrupted by noise. Journal of Statistical Physics , 175:789-818, 2019.
- [26] Yan V Fyodorov and Rashel Tublin. Optimization landscape in the simplest constrained random least-square problem. Journal of Physics A: Mathematical and Theoretical , 55(24):244008, 2022.
- [27] Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Linearized two-layers neural networks in high dimension. The Annals of Statistics , 49(2), 2021.
- [28] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics , pages 249-256. JMLR Workshop and Conference Proceedings, 2010.
- [29] Mark Holmes. Introduction to Perturbation Methods . Springer, 2013.
- [30] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems , 31, 2018.
- [31] Persia Jana Kamali and Pierfrancesco Urbani. Dynamical mean field theory for models of confluent tissues and beyond. SciPost Physics , 15(5):219, 2023.
- [32] Persia Jana Kamali and Pierfrancesco Urbani. Stochastic gradient descent outperforms gradient descent in recovering a high-dimensional signal in a glassy energy landscape. arXiv preprint arXiv:2309.04788 , 2023.
- [33] Jaron Kent-Dobias. On the topology of solutions to random continuous constraint satisfaction problems. arXiv preprint arXiv:2409.12781 , 2024.
- [34] Yuanzhi Li, Colin Wei, and Tengyu Ma. Towards explaining the regularization effect of initial large learning rate in training neural networks. Advances in neural information processing systems , 32, 2019.
- [35] Stefano Sarao Mannelli, Florent Krzakala, Pierfrancesco Urbani, and Lenka Zdeborova. Passed &amp;spurious: Descent algorithms and local minima in spiked matrix-tensor models. In international conference on machine learning , pages 4333-4342. PMLR, 2019.
- [36] Andreas Maurer. A vector-contraction inequality for Rademacher complexities. In Algorithmic Learning Theory: 27th International Conference , pages 3-17. Springer, 2016.

- [37] Song Mei, Theodor Misiakiewicz, and Andrea Montanari. Generalization error of random feature and kernel methods: hypercontractivity and kernel matrix concentration. Applied and Computational Harmonic Analysis , 59:3-84, 2022.
- [38] Song Mei, Andrea Montanari, and Phan-Minh Nguyen. A mean field view of the landscape of two-layer neural networks. Proceedings of the National Academy of Sciences , 115(33):E7665E7671, 2018.
- [39] Marc Mézard, Giorgio Parisi, and Miguel Angel Virasoro. Spin glass theory and beyond , volume 9. World Scientific, 1987.
- [40] Francesca Mignacco, Florent Krzakala, Pierfrancesco Urbani, and Lenka Zdeborová. Dynamical mean-field theory for stochastic gradient descent in gaussian mixture classification. Advances in Neural Information Processing Systems , 33:9540-9550, 2020.
- [41] Francesca Mignacco and Pierfrancesco Urbani. The effective noise of stochastic gradient descent. Journal of Statistical Mechanics: Theory and Experiment , 2022(8):083405, 2022.
- [42] Andrea Montanari and Eliran Subag. Solving overparametrized systems of random equations: I. model and algorithms for approximate solutions. arXiv:2306.13326 , 2023.
- [43] Andrea Montanari and Eliran Subag. On Smale's 17th problem over the reals. arXiv:2405.01735 , 2024.
- [44] Nelson Morgan and Hervé Bourlard. Generalization and parameter estimation in feedforward nets: Some experiments. Advances in neural information processing systems , 2, 1989.
- [45] Grant Rotskoff and Eric Vanden-Eijnden. Trainability and accuracy of artificial neural networks: An interacting particle system approach. Communications on Pure and Applied Mathematics , 75(9):1889-1935, 2022.
- [46] Mark Sellke. The threshold energy of low temperature Langevin dynamics for pure spherical spin glasses. Communications on Pure and Applied Mathematics , 77(11):4065-4099, 2024.
- [47] Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to algorithms . Cambridge University Press, 2014.
- [48] Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. The Journal of Machine Learning Research , 19(1):2822-2878, 2018.
- [49] Eliran Subag. Concentration for the zero set of random polynomial systems. arXiv preprint arXiv:2303.11924 , 2023.
- [50] Michel Talagrand. Mean field models for spin glasses: Volume I: Basic examples , volume 54. Springer Science &amp; Business Media, 2010.
- [51] Pierfrancesco Urbani. A continuous constraint satisfaction problem for the rigidity transition in confluent tissues. Journal of Physics A: Mathematical and Theoretical , 56(11):115003, 2023.
- [52] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS) , volume 30, pages 5998-6008. Curran Associates, Inc., 2017.
- [53] Roman Vershynin. High-dimensional probability: An introduction with applications in data science , volume 47. Cambridge university press, 2018.
- [54] Nikhil Vyas, Yamini Bansal, and Preetum Nakkiran. Limitations of the ntk for understanding generalization in deep learning. arXiv preprint arXiv:2206.10012 , 2022.
- [55] Ashia C Wilson, Rebecca Roelofs, Mitchell Stern, Nati Srebro, and Benjamin Recht. The marginal value of adaptive gradient methods in machine learning. Advances in neural information processing systems , 30, 2017.

- [56] Blake Woodworth, Suriya Gunasekar, Jason D Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models. In Conference on Learning Theory , pages 3635-3673. PMLR, 2020.
- [57] Yuan Yao, Lorenzo Rosasco, and Andrea Caponnetto. On early stopping in gradient descent learning. Constructive Approximation , 26(2):289-315, 2007.
- [58] Gilad Yehudai and Ohad Shamir. On the power and limitations of random features for understanding neural networks. Advances in neural information processing systems , 32, 2019.
- [59] Kaichao You, Mingsheng Long, Jianmin Wang, and Michael I Jordan. How does learning rate decay help modern neural networks? arXiv preprint arXiv:1908.01878 , 2019.
- [60] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM , 64(3):107115, 2021.
- [61] Tong Zhang and Bin Yu. Boosting with early stopping: Convergence and consistency. Annals of Statistics , pages 1538-1579, 2005.
- [62] Jean Zinn-Justin. Quantum field theory and critical phenomena . Oxford University Press, 2021.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes] ,

Justification: We conduct a theoretical analysis that is described by the abstract and that answers the questions detailed in the introduction.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: In the introduction section we discuss how the contribution compares to previous literature and limitations related to the use of non-rigorous mathematical techniques. Guidelines:

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

Justification: We conduct a theoretical analysis of training dynamics. The method that we use is non-rigorous but well established in theoretical physics. We show that the method correctly reproduces observations and it is checked against simulations. We prove two theorems that support our analysis.

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

Justification: We detail the numerical simulations in the appendix

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

Justification: Our paper is theoretical in nature and simulations are fairly standard and only play a support role.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- While we encourage the release of code and data, we understand that this might not be possible, so 'No' is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The theoretical results and figures are detailed with the corresponding settings that we used to produce them.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes] .

Justification: The paper contains all the details about the numerical simulations we used.

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

Answer: [NA] .

Justification: There are no extensive or complex experiments we have performed. The paper is theoretical in nature and aims at understanding simple yet paradigmatic models.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Answer: [Yes]

Justification: We conform with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA] .

Justification: Our work is theoretical in nature and aims at understanding neural network models rather to extend their use in technological applications.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA] .

Justification: Our work is theoretical in nature and aims at understanding neural network models rather to extend their use in technological applications.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA] .

Justification: We do not use existing datasets or codes.

Guidelines:

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA] .

Justification: We do not produce any new asset. Our study is purely theoretical in nature.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA] .

Justification: Our work is theoretical in nature and aims at understanding neural network models rather to extend their use in technological applications. We do not perform experiments with humans.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA] .

Justification: We do not conduct experiments with humans.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

## Answer: [NA] .

Justification: We do not use LLM.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.

## A Setting

We recall for reference some basic definitions and notations. We consider the 2-layer network defined by

<!-- formula-not-decoded -->

Throughout, we assume an offset to be subtracted so that E σ ( G ) = 0 , for G ∼ N (0 , 1) . The network input x is a d -dimensional real vector and the output is a scalar variable. The parameters of the network are the weights of the first layer collected in the matrix W defined as

<!-- formula-not-decoded -->

We will assume that ∥ w i ∥ 2 = 1 . The weights of the second layer are instead ( a 1 , . . . , a m ) and are real, possibly unbounded, variables.

We consider a dataset of n points independent and identically distributed ( y i , x i ) i ≤ n where x i ∼ N (0 , I d ) , and the labels y i are generated according to the following k -index models:

<!-- formula-not-decoded -->

Therefore, labels depend on the projection of the covariates on a fixed subspace U ∈ R d × k , with U T U = I k (there is no loss of generality in assuming U orthogonal). Efficient learning requires to estimate this subspace. Since we consider learning with square loss, we assume

<!-- formula-not-decoded -->

We now discuss the covariance structure of the network given by Eq. (A.1). For two sets of weights ( a 1 , W 1 ) and ( a 2 , W 2 ) we have where g ∼ N (0 , I k ) . We refer to the case φ = 0 as the 'pure noise case' or 'pure noise data'.

<!-- formula-not-decoded -->

The average in the rhs of Eq. (A.4) is over the data distribution while the function h ( q ) is defined as

<!-- formula-not-decoded -->

for ( G 1 , G 2 ) centered jointly Gaussian with E { G 2 i } = 1 , E { G 1 G 2 } = q .

Furthermore we have that:

where ̂ φ is given by

<!-- formula-not-decoded -->

for G ∼ N (0 , I k ) independent of G 0 ∼ N (0 , 1) .

<!-- formula-not-decoded -->

We consider Gaussian process f g ( a , W ) , φ g with the same covariance function defined above and define the empirical risk under Gaussian approximation as

<!-- formula-not-decoded -->

where f g ( · · · ) = ( f g i ( · · · ) : i ≤ n ) , φ g = ( φ g i : i ≤ n ) , ε = ( ε i : i ≤ n ) are vectors containing n i.i.d. copies of the above processes. We will also write y g = φ g + ε .

Given a model with estimated parameters ˆ a , W , the test error is given by where the expectation in the first line is over a triple ( f g , φ g , ε ) independent of the data, and in the second line with respect to x . The two expectations coincide because they depend uniquely on the second moments of these processes.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The Lagrange multipliers ν i are added to enforce the spherical constraint ∥ w i ( t ) ∥ 2 = 1 . While we consider the case of normalized first-layer weights, our approach can be generalized to unconstrained weights or to include weight decay (ridge regularization). As explained in the main text, we will replace this by gradient flow in the Gaussian model ̂ R g n ( a , W ) . Werefer to Section K for a discussion of DMFT in the original non-Gaussian model. In our analysis we will always consider the proportional asymptotics

<!-- formula-not-decoded -->

We typically index sequences and limits by n , but it is understood that d = d ( n ) → ∞ as well. After n, d → ∞ proportionally, we will consider the large network asymptotics m → ∞ at fixed α = α/m .

In the following we will drop the superscript g and write, for instance ̂ R n ( a , W ) instead of ̂ R g n ( a , W ) whenever clear from the context. All of our analytical predictions (except for Section 3) are obtained within the Gaussian model.

## B Technique

Notice that each fitting error F i ( θ ) = y i -f ( x i ; θ ) , i ∈ { 1 , . . . , n } is a random function of the model parameters θ . The randomness is due to the randomness in x i and in the noise ε i . The empirical risk in Eq. (1.2) can be rewritten as

<!-- formula-not-decoded -->

Our key approximation consists in replacing the i.i.d. random functions ( F i ) i ≤ n by i.i.d. Gaussian processes ( F g i ) i ≤ n with matching mean and covariance. While DMFT equations have been recently proven without recurring to this approximation (see [13] and appendices), their structure is simpler in the Gaussian case, which allows us to carry out the largem analysis.

Computing the covariance of F ( · ) is a straightforward exercise. We assume for simplicity that an intercept is subtracted so that E [ σ ( G )] = 0 , E [ φ ( G )] = 0 and otherwise these functions are generic ( G , G 1 , G and so on will denote standard Gaussian vectors). We then have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ Recall that θ = ( a , W ) where a ∈ R m , W = ( w 1 , . . . , w m ) ∈ R d × m are the first layer weights Finally, h : R → R , ̂ φ : R k → R encode the activations σ and the target function φ , with h applied entrywise to the matrix W T 1 W 2 .

The covariance of F i ( θ ) = y i -f i ( x ; θ ) is easily computed from the above, and this defines completely the corresponding Gaussian process ( F g i ) i ≤ n . We denote the associated risk function R g n ( θ ) := ∥ F g ( θ ) ∥ 2 / 2 n .

̂ Let us emphasize that the cost function ̂ R g n ( θ ) remains highly non-trivial despite the fact that the functions F i are replaced by Gaussian processes. Near-minima of high-dimensional Gaussian processes have a very rich structure, which is a central theme in spin glass theory [39, 50]. Additional layers of complexity arise here for two reasons. First, ̂ R g n ( θ ) is a sum of squares of Gaussians and, second, the underlying Gaussian process has a significantly more intricate covariance than in standard spin glasses (where typically depends only on the inner product ⟨ θ 1 , θ 2 ⟩ ). Recent work explored the simpler case in which F g i ( · ) is a Gaussian process with covariance E { F g i ( θ 1 ) F g i ( θ 2 ) } = ξ ( ⟨ θ 1 , θ 2 ⟩ ) depending uniquely on the inner product [25, 26, 51, 49, 42, 43, 33]. Gradient descent dynamics on these models has been recently studied via DMFT in [31, 32]: our work builds on these advances. DMFTwas leveraged before to address other questions in high-dimensional statistics and ML [35, 12]. We refer to [8, 13] for mathematical results on the DMFT approach.

While ̂ R g n ( θ ) has a non-trivial structure, methods from statistical physics can be brought to bear to derive an asymptotic characterization. Namely, define the functions

These functions are random (because of the random initialization and the randomness in F g ) and depend on n, d . However, as n, d → ∞ with n/d → α , they converge to non-random limits ( C ij ( t 1 , t 2 )) i&lt;j ≤ m , ( v i ( t )) i ≤ m , ( a i ( t )) i&lt;j ≤ m that are the unique solution of a set of coupled integrodifferential equations, see the appendices. We refer to these as to the DMFT equations.

<!-- formula-not-decoded -->

̸

Our main focus is on the behavior of the solutions of these equations for large m and, at first sight, the complexity of the DMFT increases with m . An important simplification arises when choosing a symmetric initial condition a i (0) = a 0 for all i ≤ m , and ( w i (0)) i ≤ m ∼ iid Unif ( S d -1 ) . Namely, the solution of the DMFT equations is symmetric under permutations of the neurons: C ii ( t 1 , t 2 ) = C d ( t 1 , t 2 ) for i ≤ m and C ij ( t 1 , t 2 ) = C o ( t 1 , t 2 ) for i = j ≤ m , while v i ( t ) = v ( t ) , a i ( t ) = a ( t ) for i ≤ m . We then have a reduction to a set of integro-differential equations on k +3 functions, that depend parametrically on m .

We use two approaches to study these equations (see appendix):

- ( a ) Numerical integration for increasing values of m under different initial conditions.
- ( b ) Asymptotics as m →∞ (at fixed α = α/m ) via singular perturbation theory [9, 29].

For ( b ) , a specific dynamical regime is identified by a scaling of the time variable, which in our case will take the form t = t # ( m ) · ˆ t for a certain fixed function t # ( m ) and ˆ t = O (1) a scaled time. The asymptotics of DMFT quantities in that regime takes the form

<!-- formula-not-decoded -->

## C Dynamical Mean Field Theory (DMFT)

In this section we state the results of Dynamical Mean Field Theory (DMFT). We will outline a heuristic derivation in Section L. We first introduce the general DMFT equations in Section C.1 and the corresponding predictions for certain observable of interest in Section C.2. These are a set of Θ( m 2 ) integro-differential equations in as many unknown functions.

We then specialize these equations to the case of a symmetric initialization, in which w i (0) ∼ Unif ( S d -1 ) and a i (0) = a 0 for all i ≤ m , see Section C.3 In this case, the dynamics is characterized by a set of k +3 equations which are stated in Sections C.4 and C.5.

## C.1 General DMFT equations

Let a n i ( t ) , w n i ( t ) , ν n i ( t ) the the solution of Eq (A.10) when the dynamics is initialized at non-random a n i (0) = a 0 ,i , i ≤ n and possibly random, w n i (0) such that ⟨ w n i (0) , w n j (0) ⟩ → C 0 ij for i, j ≤ n , U T w n i (0) → v 0 i for i ≤ n . While random, the w n i (0) are assumed here to be independent of the random processes f g , φ g , ε .

For t, s ≥ 0 consider the quantities

<!-- formula-not-decoded -->

Then DMFT predicts that these quantities have a well defined non-random limit as n, d →∞ ,

<!-- formula-not-decoded -->

where the limits are understood to hold in almost sure sense. These limits are the unique solution of a set of integro-differential equations in the unknowns { C ij ( t, s ) , R ij ( t, s ) , v i ( t ) , a i ( t ) : i, j ≤ m } , which we next state as three sets: (1) Dynamical equations; (2) Equations for auxiliary functions; (3) Boundary conditions. Before that, we mention some constraints that need to be satisfied by the solution of these equations.

- (0) Constraints. The functions C ij ( t, s ) , R ij ( t, s ) satisfy:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first condition in particular implies the following useful relation:

<!-- formula-not-decoded -->

We refer to the property (C.5) (and similar ones for R functions appearing below) as 'causality constraint.'

(1) Dynamical equations. These equations determine the dynamics of { C ij ( t, s ) , R ij ( t, s ) , v i ( t ) , a i ( t ) : i, j ≤ m } , and involve the auxiliary functions (memory kernels) M C ij ( t, s ) , M R ij ( t, s ) and (Lagrange multipliers) ν i ( t ) (the last equations assume implicitly t a &gt; t b ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We point out that the δ ( t a -t b ) in the last equation (together with Eq. (C.5)) has to be interpreted as follows: R ij ( t, t ′ ) = 0 for t &lt; t ′ while, for ε &gt; 0 , R ij ( t + ε, t ) = δ ij + o ε (1) .

Equations (C.9) and (C.10) can also be written in terms of an effective stochastic process in R m : w e ( t ) = ( w e i ( t ) : i ≤ m ) . This is defined as the solution of the following set of ODEs (for

i ∈ { 1 , . . . , m } ):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( η i ( t ) : i ≤ m ) is a centered Gaussian process with covariance

<!-- formula-not-decoded -->

Define b ( t ) = ( b i ( t ) : i ≤ m ) . The solution of Eqs. (C.9) and (C.10) can be written as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In fact the stochastic process of Eq. (C.11) is expected to describe the limit distribution of the secondlayer weights W ( t ) . Namely, for i ≤ d , define ˜ w i ( t ) = W ( t ) e i ∈ R m be a vector containing the i -th coordinate of each neuron. Then, for any fixed i and any T ,

<!-- formula-not-decoded -->

Here d ⇒ denotes convergence in distribution as n, d →∞ , in C ([0 , T ] , R m ) .

(2) Equations for auxiliary functions. The memory kernels M R and M C are defined by

<!-- formula-not-decoded -->

where the functions R A and C A satisfy the symmetry properties C A ( t, s ) = C A ( s, t ) and R A ( t, s ) = 0 for t &lt; s , and are the unique solution where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The Lagrange multipliers ν i ( t ) have to be fixed to enforce the constraint C ii ( t, t ) = 1 which follows from w α ∈ S d -1 . The corresponding equations are

- (3) Boundary conditions. The dynamical equations (C.7) to (C.10) can be integrated from a set of initial conditions that reflect initial conditions of the GF dynamics:

<!-- formula-not-decoded -->

## C.2 Expressions for train and test error

The asymptotics of many quantities of interest can be expressed in terms of the solutions of the DMFT equations stated in the last section. In particular, the train error ̂ R n ( W ( t ) , a ( t )) and test error R ( W ( t ) , a ( t )) at time t have well defined limits under the proportional asymptotics:

The functions e tr ( t ) e ts ( t ) are given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

More generally, C A ( t, s ) gives the asymptotics of the correlation of residuals:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we recall that y g = φ g + ε .

## C.3 Symmetric initialization and solutions

As anticipated, we consider the uninformative initialization w n i (0) ∼ Unif ( S d -1 ) and a n i (0) = a 0 for all i ≤ m . This results in the following initialization for the DMFT equations of

̸

<!-- formula-not-decoded -->

̸

̸

This initialization is invariant under permutations of the m neurons. Since the DMFT equations of Section C.1 are equivariant under such permutations, their solution is also invariant under permutations. This means that it takes the form:

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

As a consequence, the memory kernels in Eq. (C.18) take the form

<!-- formula-not-decoded -->

̸

̸

We will refer to the reduced DMFT under symmetry as to the SymmDMFT .

## C.4 DMFT equations for symmetric initialization ( SymmDMFT )

(1) Dynamical equations. Substituting the ansats of the previous section in the equations of Section C.1, we obtain the following equations for the functions a ( t ) , v ( t ) , C d ( t, s ) , C o ( t, s ) , R d ( t, s ) , R o ( t, s ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(2) Equations for auxiliary functions. The memory kernels M ( s ) R ( t, s ) , M ( o ) R ( t, s ) and M ( s ) C ( t, s ) , M ( o ) C ( t, s ) are given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Further, C A ( t, s ) , R A ( t, s ) are given by the same equations (C.19), where Σ C , Σ R are simplified as follows:

<!-- formula-not-decoded -->

Finally, the Lagrange multipliers are determined by

<!-- formula-not-decoded -->

- (3) Boundary conditions. As anticipated the SymmDMFT is initialized as

<!-- formula-not-decoded -->

## C.5 Expressions for train and test error under symmetric initialization

The general expression for train and test error given in Section C.2 specialize to:

<!-- formula-not-decoded -->

## D Numerical integration of the DMFT equations

## D.1 Integration technique

We integrate the SymmDMFT equations (C.32) to (C.37) using a standard Euler discretization. Namely, we discretize time on an equi-spaced grid t ∈ T := { 0 , η, 2 η, . . . } and approximate derivatives by differences and integrals by sums on this grid. As an example, Eq. (C.32) is replaced by

<!-- formula-not-decoded -->

The discretization of Eq. (C.10) deserves an additional clarification because of the delta-function. For t a ≥ t b , t a , t b ∈ N η , we compute

<!-- formula-not-decoded -->

with boundary condition

<!-- formula-not-decoded -->

Of course, the solution of this system of difference equation does not coincide with the solution of the original equations (C.32) to (C.37), and in this section we will write a ( t ; η ) , C o ( t, s ; η ) and so on to emphasize the distinction.

Equations (C.42) can be directly interpreted as determining Σ C ( t, s ) and Σ R ( t, s ) on the grid t, s ∈ T . Finally, we discretize Eq. (C.19) as

<!-- formula-not-decoded -->

Note that we dropped the integration limits here, since they are enforced by the causality constraints implying Σ R ( t, s ) = 0 , R A ( t, s ) = 0 for t &lt; s . Defining the matrices Σ R = (Σ R ( t, s ) : t, s ∈ T ) , and similarly for Σ C , C A , R A , we can rewrite (D.2) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We truncate these matrices (which are infinite) to a maximum time T (e.g., redefine Σ R = (Σ R ( t, s ) : t, s ∈ T , s, t ≤ T ) ) and solve these equations by matrix inversion:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We denote by a ( t ; η ) , v ( t ; η ) , C o ( t, s ; η ) , C d ( t, s ; η ) , R o ( t, s ; η ) , R d ( t, s ; η ) , the functions obtained via the Euler integration scheme. We will assume that this solution is interpolated continuously for t, s ̸∈ T . For instance, for i, j ∈ N a, b ∈ [0 , 1) , we let

<!-- formula-not-decoded -->

Finally, while we described the discretization procedure for the SymmDMFT , the discussion above applies verbatimly for the full DMFT of Section C.1.

The DMFT equations and their symmetric specialization have a causal structure which means that they can be integrated by progressively by increasing T . Furthermore there is no self-consistency condition in the integration scheme at variance with the non-Gaussian settings, see for example [40]. This simplification allows to investigate the long time behavior of the dynamics in a numerical, rather efficient, way.

## D.2 Accuracy of the numerical integration scheme

The discretization of DMFT is expected to converge to the actual solution with errors of order η . Namely, we expect

<!-- formula-not-decoded -->

and similarly for the other functions. We refer to [13] for related examples in which the convergence was proved rigorously, and to [31] for an empirical study in a closely related model.

In order to test the accuracy of our approach, and the correctness of the DMFT equations, we simulated the gradient descent (GD) dynamics for the Gaussian model. Namely, we generate realizations of the process f g ( a , W ) = ( f g i ( a , W ) : i ≤ n ) with the prescribed covariance (A.4), and the vector φ g = ( φ g i : i ≤ n ) with same covariance as in Eq. (A.6) (see Section D.4.) We define ̂ R n ( a , W ) via Eq. (A.8) and implement the following GD iteration

<!-- formula-not-decoded -->

̸

where P S d -1 is the projector to the unit sphere, i.e. P S d -1 ( x ) = x / ∥ x ∥ if x = 0 and P S d -1 ( 0 ) = 0 . Note that the trajectories of Eq. (D.9) depend on the sample size n (and hence the dimension d = d n ) and the stepsize η GD. To emphasize this dependence, we also use the notation a n ( t ; η GD ) W n ( t ; η GD ) .

We expect the GD trajectories defined by Eq. (D.9) approach the GF trajectories defined by Eq. (A.10) as η GD → 0 uniformly in n, d . Namely,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the limits are understood to hold in probability for any fixed t . Informally, for fixed small η GD, GD dynamics is a good approximation to GF dynamics, irrespective of the dimension.

We generate several realizations of the processes f g , φ g , and of the gradient descent trajectories (D.9). We average observables of interest over these realizations and compare these with the Euler discretization of the DMFT equations. For instance, consider the correlation functions C ij ( t, s ) . Then we can compare:

- C n ij ( t, s ; η GD ) = E ⟨ w n i ( t ; η GD ) , w n j ( s ; η GD ) ⟩ where the expectation is taken with respect to the GD process (D.9).
- C ij ( t, s ; η ) , the solution of the Euler discretization of the DMFT, described in the previous section.

Some results of this comparison are presented in the next subsection. This comparison allows us to gauge two types of systematic effects:

1. The effect of finite n, d . Indeed, the DMFT equations characterize the n, d →∞ limit of the GD dynamics (D.9).
2. The non-zero stepsize η . Note that the effect of discretization introduced in the DMFT equations are different from the ones in the gradient descent (D.9). Therefore the disagreement between the two is a measure of the nonzeroη effects.

To clarify further the last point, we emphasize that, despite the notation, C ij ( t, s ; η ) is not the n, d →∞ limit of C n ij ( t, s ; η ) .

We note in passing that it is possible to derive DMFT equations for GD, hence characterizing lim n →∞ C n ij ( t, s ; η GD ) . Similar characterizations were obtained for related (simpler) models in [40, 13, 41, 31, 32]. We defer the analysis of GD with large stepsizes to future work.

## D.3 Testing the numerical accuracy

Figures 6 and 7 we present examples of the numerical comparison described in the previous section, under two different settings, as described below.

Setting 1. We assume pure noise data with τ = 1 and train a network with m = 4 neurons and covariance structure given by h ( z ) = z/ 10 + z 2 / 2 . We simulate GD trajectories, according to Eq. (D.9) with d = 100 , n = 150 , and correspondingly evaluate the Euler discretization of DMFT, cf. Section D.1 for α = n/d = 1 . 5 .

We choose an initialization that is not symmetric and therefore we have to use the full DMFT equations of Section C.1. More precisely, we initialize second layer weights as follows:

<!-- formula-not-decoded -->

The weights of the first layer are instead initialized by generating two random vectors y 1 , y 2 ∼ Unif ( S d -1 ) , and setting

<!-- formula-not-decoded -->

This initialization results in initializing the DMFT equations with

<!-- formula-not-decoded -->

Figure 6: Comparison between discretized DMFT and GD dynamics for the Gaussian model (labeled as 'Simulations'). GD results are averaged over N = 10 4 realizations of the Gaussian process, under Setting 1 described in the main text. Left frame: Second layer for DMFT and GD with η GD = η = 0 . 1 . Right frame: Train error and correlation function for DMFT with a few values of η , and for GD.

<!-- image -->

Figure 7: Comparison between discretized DMFT and GD dynamics (labeled as 'Simulations'). GDresults are averaged over N = 10 4 realizations of the Gaussian process, under Setting 2 described in the main text. Results for GD are averaged over N = 10 4 samples.

<!-- image -->

Both for the discretized DMFT and for GD for several values of the stepsize. The results of this analysis are plotted in Fig. 6.

Setting 2. We consider again pure noise with τ = 1 , a network with m = 4 , input dimension d = 100 and sample size n = 150 . We use hidden neurons with the same covariance structure as in the Setting 1.

However, we change the initialization with respect to Setting 1. First layer are initialized independently and uniformly at random. It follows that

<!-- formula-not-decoded -->

Second layer weights are initialized according to

<!-- formula-not-decoded -->

We use stepsize η = 0 . 1 .

## D.4 Construction of the Gaussian process f g ( · )

The Gaussian process f g can be constructed as follows. Define a sequence of independent Gaussian tensors J ( k ) ∈ ( R d ) ⊗ k , k ≥ 1 , with entries ( J ( k ) i 1 ,...,i k : i j ≤ d ) ∼ iid N (0 , 1) . We then let

<!-- formula-not-decoded -->

It is easy to check that this stochastic process has the prescribed covariance, with

<!-- formula-not-decoded -->

has long as the series above has radius of convergence larger than 1 . An analogous construction holds for φ g .

## E Dynamical regimes: General preliminaries

In the next two sections, we will study the SymmDMFT equations of Section C.4 and characterize different dynamical regimes in the large network limit. From a technical viewpoint, we develop a singular perturbation theory of the DMFT equations as m →∞ for fixed overparametrization ratio α = α/m .

While singular perturbation theory is a classical domain of mathematics [9, 29], making this type of analysis rigorous is notoriously challenging. We will proceed heuristically as follows: ( i ) Hypothesize a certain asymptotic behavior of the DMFT solution in a specific time-scale; ( ii ) Check consistency with the DMFT equations; ( iii ) Check that this behavior is observed in the numerical solution of the DMFT equations.

More precisely, a specific dynamical regime is identified by a scaling of the time variable, which in our case will take the form t = t # ( m ) · ˆ t for a certain fixed function t # ( m ) and ˆ t a scaled time of order one. The asymptotics of DMFT quantities in that regime takes the form (for instance)

<!-- formula-not-decoded -->

where c # ( m ) , c o ( ˆ t, ˆ s ; α ) are two fixed functions, the limit is understood to hold at fixed ˆ t, ˆ s, α ∈ (0 , ∞ ) , and we made explicit the dependence of C o on m , α . More concisely, we will often write the above formula as

<!-- formula-not-decoded -->

and we will typically use t , s instead of ˆ t , ˆ s for the dummy variables.

The behavior of the DMFT equations depends in a crucial way in the initialization of the second layer weights:

- In Section F, we will consider the case of a 'lazy initialization,' i.e. we will assume a (0) = γ 0 √ m for some constant γ 0 ∈ (0 , ∞ ) independent of m .
- In Section G, we will consider the 'mean field initialization' i.e. assume a (0) = a 0 to be constant and independent of m .

## F Dynamical regimes: Lazy initialization

As anticipated, in this section we study dynamical regimes under lazy initialization. In subsection F.1, we will consider the case of pure noise data and in subsection F.2 the k -index model.

Throughout this section, we let γ ( t ) = a ( t ) / √ m (in particular, γ (0) = γ 0 ).

Figure 8: Training of pure noise data: first dynamical regime. Rescaled correlation function m ( C d ( t, s ) -1) in the first dynamical regime as a function of the scaled time tm for a model initialized with a lazy scaling and fixed second layer weights. Different curves correspond to the numerical integration of the SymmDMFT equations at various values of m . They appear to converge to the scaling solution in the large m limit described by Eqs. (F.6). Here α = 0 . 5 , ˜ h ( z ) = (3 / 10) z + z 2 / 2 and τ = 1 .

<!-- image -->

## F.1 Pure noise model

Under the pure noise model, we have φ = ˆ φ = 0 . Further, the variable v ( t ) is not defined and can be dropped (equivalently, we can set v ( t ) = 0 ).

We identify three dynamical regimes:

1. t = O (1 /m ) : γ ( t ) = γ 0 + o m (1) , train error decreases, and the network approximates the null function (Section F.1.1).
2. t = Θ(1) : γ ( t ) = γ 0 + o m (1) , first-layer weights move significantly and train error converges to a limit e ∗ ( γ 0 ) (Section F.1.2). If γ 0 is larger than the interpolation threshold, then train error vanishes in this regime.
3. t = Θ( m ) : This regime emerges only if γ 0 is smaller than the interpolation threshold. (We discuss the identification of the interpolation transition of gradient flow in Section F.1.3.) If this is the case, γ ( t ) grows on the time scale t = Θ( m ) until it crosses the interpolation threshold. At that point the train error vanishes (Section F.1.4).

Since in the first two regimes γ ( t ) does not change appreciably, the dynamics in these time scales is essentially equivalent to the one of a network in which second-layer weights are fixed and do not evolve by GF. In Section F.1.1 and F.1.2 we first consider this case.

We note that the pure noise model is unchanged if we rescale τ → cτ , γ 0 → cγ 0 . More precisely, this results in a rescaling of the risk by c 2 and hence of time by the same factor. As a consequence quantities of interest often depend on γ, τ uniquely through their ratio γ/τ .

## F.1.1 First dynamical regime: t = O (1 /m )

We first consider the case in which the (scaled) second layer weights are not updated and fixed to their initialization, i.e. γ ( t ) = γ 0 .

It is possible to check that, up to higher-order terms, the SymmDMFT equations are solved by functions of the form (the first equation holds in weak sense, i.e. after integrating against a test function)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where C lz1 A , C lz1 d , C lz1 o , ν lz1 and R lz1 o are suitable functions independent of m . Here and below, we use the notation ϑ ( t ) = 1 ( t &gt; 0) .

Note that Eq. (F.3) implies that on this dynamical regime the weights of the first layer change by order 1 /m .

Plugging the asymptotic form in Eqs. (F.1) to (F.4) into the SymmDMFT equations and matching the leading orders for large m , we obtain that the functions C lz1 A , C lz1 d , C lz1 o , ν lz1 and R lz1 o must satisfy

These are a set of ordinary differential equations that can be solved explicitly. We get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, Eqs. (F.6) imply

<!-- formula-not-decoded -->

Recalling Eq. (F.2) we conclude that

<!-- formula-not-decoded -->

or, using the interpretation of C o ,

<!-- formula-not-decoded -->

̸

In other words, at the end of this dynamical regime, the first-layer weights form a regular simplex, with center w ( t/m ) := m -1 ∑ m i =1 w i ( t/m ) satisfying ∥ w ( t/m ) ∥ 2 = o m (1) .

Hence, at the end of the first dynamical regime, the first-layer weights are such that the linear component of the activation function σ is removed. In other words, for t a large constant, we have

<!-- formula-not-decoded -->

where σ nl G ( w ) is a Gaussian process with covariance structure given by h ( z ) -zh ′ (0) , and err is small in mean square.

Notice also that this is achieved by a O (1 / √ m ) change in each of the first layer weights. Indeed, by Eq. (F.3), we have

<!-- formula-not-decoded -->

Figure 9: Training with pure noise data under lazy initialization: second dynamical regime t = Θ(1) . Left panel: First-layer weights correlation function C d ( t, 0) measuring the inner product between neurons at time 0 and time t , plotted versus t for several values of m , and compared with the large m -asymptotics C lz2 d . Right panel: training error a e tr ( t, γ 0 , m ) plotted versus t for several values of m , and compared with the largem asymptotics in this regime e lz2 tr ( t, 1) . Notice the two-steps decrease of the training error, corresponding to the two regimes t = O (1 /m ) and t = Θ(1) . Inset: Same curves plotted versus tm , and compared with the asymptotic prediction e lz1 tr ( · , 1) in the first dynamical regime. For both panels we use α = 0 . 5 , ˜ h ( z ) = (3 / 10) z + z 2 / 2 , γ 0 = 1 and τ = 1 .

<!-- image -->

Equations (F.1) to (F.4) can be used to compute the behavior of the train error in this dynamical regime:

<!-- formula-not-decoded -->

Using Eqs. (F.6), we get the expression:

<!-- formula-not-decoded -->

In particular, the train error at the end of this dynamical regime is

<!-- formula-not-decoded -->

This is in agreement with (F.10). Indeed, note that

<!-- formula-not-decoded -->

This picture is confirmed by the fact that Eqs. (F.6) depend on h only through h ′ (0) . This means that the dynamics on timescales of order 1 /m is controlled by the linear part of the covariance structure of the hidden layer.

Training in this timescale attempts to minimize ∥ f g ( a , W ) ∥ 2 without fitting the noise.

In Fig.8 we test the correctness of the asymtotic ansatz of Eqs. (F.1) to (F.4). Namely, we compare the results of numerical integration of the SymmDMFT equations for various values of m , with the prediction of Eqs. (F.6). The match is excellent.

So far we assumed that second-layer weights are not optimized and γ ( t ) = γ 0 . What happens if drop this constraint? It can be checked that the form given in Eqs. (F.1)-(F.4) still solves the SymmDMFT equations when a ( t ) is allowed to evolve, and γ ( t/m ) = γ 0 + o m (1) for all fixed t ∈ (0 , ∞ ) . In other words, second layer weights do not change significantly during this dynamical regime.

## F.1.2 Second dynamical regime: t = Θ(1)

The second dynamical regime arises when t = Θ(1) . Recall from the previous subsection that, for t = o m (1) , the train error remains close (for large m ) to the plateau characterized at the end of the first dynamical regime, see Eq. (F.14). When t is of order one, the first layer weights start changing by an amount of order one as well, and the model starts to fit the noise.

As before, we begin by considering the simplified setting in which γ ( t ) = γ 0 is fixed and not optimized by GF.

We claim that the SymmDMFT equations are solved by the following ansatz, up to lower order terms as m →∞ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here C lz2 d , R lz2 d , C lz2 o , R lz2 o and ν lz2 are certain functions independent of m . Equations (F.19), (F.20) state in particular that C lz2 o ( t, t ′ ) = -C lz2 d ( t, t ′ ) and R lz2 o ( t, t ′ ) = -R lz2 d ( t, t ′ ) , and the therefore we are left with the task of determining C lz2 d ( t, t ′ ) , R lz2 d ( t, t ′ ) . By substituting Eqs. (F.16) to (F.20) into the SymmDMFT equations and matching leading order terms, we get a set of two integral-differential equations for C lz2 d ( t, t ′ ) , R lz2 d ( t, t ′ ) , which we next state.

We first define

<!-- formula-not-decoded -->

then we define R lz2 A and C lz2 A as the solution of

<!-- formula-not-decoded -->

We next define the asymptotic form for the memory kernels

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and we have defined

The equations for ν lz2 , C lz2 d and R lz2 d are then given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(As before, in the second and last equation, it is understood that t ≥ t ′ , and the last equation is understood to hold in weak sense.)

Given the constraints on C d , R d , we have the following constraints on C lz2 d , R lz2 d ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, the last condition, together with Eq. (F.26) implies R lz2 d ( t + , t ) = 1 .

<!-- image -->

GF

-

Figure 10: Training with pure noise data under lazy initialization: algorithmic interpolation threshold. Left Panel. We plot the train error as a function of time for different values of α at γ 0 = 10 / 7 . The train error has an exponential decay to zero for α below the interpolation threshold. Right Panel. We plot the time for GF to converge to near-zero training error as a function of α , for various values of γ 0 , as computed using the theory of Section F.1.2. The divergence of t rel signals the phase transition for GF interpolation α ∗ GF . Inset: τ rel versus α in linear scale. Main panel: t rel versus α ∗ GF -α (with the fitted value of α ∗ GF ). Here we use h ( z ) = (3 / 10) z + z 2 / 2 .

The evolution of the the train error in this second dynamical regime is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have made explicit the dependence on the initialization of second-layer weights γ 0 .

Note that Eq. (F.22) implies C lz2 A (0 , 0) = -Σ lz2 C (0 , 0) , and Eq. (F.21) yields Σ lz2 C (0 , 0) = τ 2 + γ 2 0 h (1) -γ 2 0 h ′ (0) . Therefore

In other words, this second dynamical regime captures the decrease of the training error which starts at the plateau reached in the first regime, cf. Eq. (F.14). which coincides with the long time extrapolation of the first dynamical regime.

<!-- formula-not-decoded -->

This second dynamical regime is fully non-linear and depends on the entire covariance function ˜ h . Further, the first order weights move by an amount ∥ w i ( t ) -w i (0) ∥ = Θ(1) , as follows from the fact that C lz2 d ( t, 0) &lt; 1 strictly.

In order to confirm the ansatz (F.16) to (F.20), we compared the solution of the full SymmDMFT equations, with the solution of the asymptotic equations (F.25), (F.27). An example of such a comparison is presented in Fig. 9: the agreement is excellent.

The treatment above assumed the constraint γ ( t ) = γ 0 . However, as in the first dynamical regime, if we let second layer weights evolve, they do not change appreciably. Namely, the asymptotic form given in Eqs. (F.16) to (F.20) still solves the SymmDMFT equations when a ( t ) is allowed to evolve. We have γ ( t ) = γ 0 + o m (1) on this timescale.

## F.1.3 The algorithmic interpolation transition

For the discussion in this section, we denote by e tr ( t, γ 0 , m, α ) the train error as a function of t , where we emphasized the dependence on the initial condition γ 0 , on the number of neurons m , and on the overparametrization ratio α . We further assume that second layer weigths are not evolved and therefore γ ( t ) = γ 0 for all t . We define the asymptotic train error achieved by GF as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Again, in this definition a i = γ 0 √ m is kept fixed and does not evolve with time.

Notice that it is in principle possible that lim n →∞ ̂ R n ( a , W ( t n )) is strictly smaller than e tr , ∞ ( γ 0 , m, α ) if we let t n diverge with n at sufficiently fast rate. However, based on results on related models in spin-glass theory we expect this not to be the case as long as t n is polynomial in n . Explicitly, we expect that, for any sequence t n →∞

<!-- formula-not-decoded -->

Using the reduced equations for t = Θ(1) timescale, i.e. Eqs. (F.25) to (F.27), we can also define

<!-- formula-not-decoded -->

A natural question is whether the large m limit of e tr , ∞ ( γ 0 , m, α ) coincides with e lz2 tr , ∞ ( γ 0 , α ) . This amounts to asking whether there exists dynamical regime with timescale t ( m ) diverging with m at which e tr ( t ( m ) , γ 0 , m, α ) starts diverging significantly from the value at the end of the second dynamical regime namely e lz2 tr , ∞ ( γ 0 , α ) . If e lz2 tr , ∞ ( γ 0 , α ) = 0 then of course lim m →∞ e tr ( t ( m ) , γ 0 , m, α ) = 0 as well.

If however e lz2 tr , ∞ ( γ 0 , α ) &gt; 0 , then the answer depends upon whether the second layer weights are evolved with GF:

- In the constrained setting in which second-layer weights do not evolve, we observe (from numerical solutions of SymmDMFT ) that

<!-- formula-not-decoded -->

- In the next section we will see that if γ ( t ) evolves with GF then the train error achieved on a diverging timescale t = Θ( m ) is strictly smaller than e lz2 tr , ∞ ( γ 0 , α ) and vanishes for large enough t .

Note that e tr , ∞ ( γ 0 , m, α ) and e lz2 tr , ∞ ( γ 0 , α ) also depend on the noise variance τ 2 . However, because of the invariance under rescaling discussed at the beginning of this section (adding τ as an argument):

<!-- formula-not-decoded -->

and similarly for e tr , ∞ ( γ 0 , m, α ) . Because of this relation, we can think that τ 2 is fixed throughout, e.g. τ 2 = 1 .

We expect e tr , ∞ ( γ 0 , m, α ) , e lz2 tr , ∞ ( γ 0 , α ) to be non-increasing in γ 0 , and define the thresholds

<!-- formula-not-decoded -->

(These definitions need to be modified if γ 0 ↦→ e tr , ∞ ( γ 0 , m, α ) is non-monotone.)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The numerical solution of the SymmDMFT equations imply that the curve γ ∗ GF ( α ) is monotone increasing with α , as also suggested by the Gaussian complexity bound (see Section 2.2 in the main text). Hence we can invert it to get a threshold α ∗ GF ( γ 0 ) : the two descriptions are equivalent.

In order to determine α ∗ GF ( γ 0 ) , we adopt a procedure already implemented in [31] for a simpler model. The procedure is based on the observation (from numerical solutions) that when e lz2 tr , ∞ ( γ 0 , α ) = 0 , e lz2 tr ( t, γ 0 , α ) = exp( -t/t rel ( α ; ε ) + o ( t )) for some t rel ( α ) &gt; 0 which diverges as α ↑ α GF .

1. Define a grid of values of α , A 0 = { α 1 , α 2 , . . . , α K } , which are expected to be smaller than α ∗ GF ( γ 0 ) .

Of course, Eq. (F.38) implies

Figure 11: Training with pure noise data under lazy initialization: second layer weights in the third dynamical regime. Evolution of the (rescaled) weights of the second layer as a function of t/m . Here τ = 2 . 5 and γ 0 = 1 , α = 0 . 3 , and covariance structure for the neurons given by h ( z ) = (9 / 10) z + z 2 / 2 .

<!-- image -->

2. For each value α ∈ A 0 , integrate numerically the reduced equations (F.25) to (F.27). Verify that e lz2 tr ( t, γ 0 , α ) appear to converge to 0 with t →∞ . Let A ⊆ A 0 be the subset of values for which this happens.
3. For each α ∈ A , define t rel ( α i ; ε ) := inf { t : e lz2 tr ( t, γ 0 , α i ) &lt; ε · τ 2 } where ε is a small threshold value (we use ε = 10 -7 ).
4. Estimate parameters α ∗ GF ( γ 0 ) , c, ν by fitting the relation t rel ( α i ; ε ) ∼ c ( α ∗ GF -α i ) -ν .

Figure 10 illustrates the calculation of α ∗ GF ( γ 0 ) for three values of γ 0 . In the inset we plot t rel for three values of γ 0 as a function of α . In the main panel, we demonstrate the divergence of t rel when ( α ∗ GF -α ) vanishes. In practice, we observe ν = 2 fit well the data across a variety of settings, suggesting this is the universal exponent for the divergence of t rel .

## F.1.4 Third dynamical regime: t = Θ( m )

In the first two dynamical regimes, the largem behavior did not depend on whether we would let second layer evolve with GF or we kept them fixed, i.e. γ ( t ) = γ 0 .

In contrast, the behavior on timescales diverging with m depends significantly on the dynamics of second-layer weights.

- If second layer weights are fixed, no significant further evolution takes place. In particular, the training error does not decrease significantly below the value reached at the end of the second dynamical regime, i.e. e ℓ tr , ∞ ( γ 0 , α ) . This is stated formally in Eq. (F.38).
- If second layer weights evolve according to GF, then the dynamics on time-scales diverging with m can be non-trivial and depends on the second-layer weights initialization γ 0 . If γ 0 &gt; γ ∗ GF ( α ) , then GR reaches vanishing training error during the second dynamical regime, and no further evolution takes place.

However, if γ 0 &lt; γ ∗ GF ( α ) , second layer weights start evolving when t = Θ( m ) , thus giving rise to a third dynamical regime. This is the object of the present subsection.

In Fig. 11, left frame, we plot the rescaled second layer weights γ ( t ) (as predicted by numerical integration of the SymmDMFT equation) as a function of time for several values of m . Here, obviously, we do not constrain γ ( t ) = γ (0) .

We observe that γ ( t ) changes only when t = Θ( m ) . Indeed, when plotted against t/m , curves obtained for different values of m collapse onto each other. This suggests that, for t = o ( m ) γ ( t ) = γ (0) + o m (1) (recall that γ (0) = γ 0 by definition). Further, the curve collapse suggests that,

for any fixed ˆ t ∈ (0 , ∞ ) :

<!-- formula-not-decoded -->

where we have made explicit the dependence on γ 0 . Of course, the case γ 0 &gt; γ ∗ GF ( α ) fits in this framework with γ lz3 ( z, γ 0 ) = γ 0 identically.

We next consider the evolution of the train error. In Fig. 12, left frame, we plot the train error (again, as predicted by numerical integration of the SymmDMFT equation) as a function of time for several values of m .

Again, when plotted as a function of t/m , curves for different values of m reach a plateau, and collapse below the plateau. This suggests the following limit behavior, which is consistent with Eq. (F.43)

<!-- formula-not-decoded -->

(Here we use ˜ e tr ( ˆ t m, γ 0 ) to denote the train error when second-layer weights evolve, in contrast with e tr ( ˆ t m, γ 0 ) which we used for the setting in which second-layer weights are constrained.)

Matching the present dynamical regime ( t = Θ( m ) ) with previous one ( t = Θ(1) , cf. Section F.1.2), implies that

<!-- formula-not-decoded -->

In other words, the function e lz3 tr describes the decrease of the train error below the level e lz2 tr , ∞ ( γ 0 ) achieved during the second dynamical regime.

In order to characterize the scaling function e lz3 tr , in Fig. 12, right frame, we plot parametrically the the train error for different values of m as a function of the second layer weights γ ( t ) . We also plot the curve ( γ, e lz2 tr , ∞ ( γ )) . This plot is consistent with the following behavior as m → ∞ . In a first regimes (corresponding to t = o ( m ) ) the train error has a drop that becomes vertical in the m →∞ limit, implying that γ ( t ) does not evolve while the train error decreases until it reaches e ℓ tr , ∞ ( γ 0 ) . In the last regime (corresponding to t = Θ( m ) ), γ ( t ) increases together with the decrease of the train error e ℓ tr ( t, γ 0 ) . Remarkably, they follow the curve ( γ, e lz2 tr , ∞ ( γ )) .

In order to describe the last regime, we point out that t ↦→ γ lz3 ( t ) is monotone increasing. Therefore we can re-parametrize time by the value of the second layer weights. Namely, define ˜ γ -1 the inverse function, so that

<!-- formula-not-decoded -->

Using this reparametrization of time, the behavior in Fig. 12 can be formalized as

<!-- formula-not-decoded -->

The collapse on finite m curves in Fig. 12, right frame, onto the curve ( γ, e lz2 tr , ∞ ( γ )) suggests that

<!-- formula-not-decoded -->

In other words, the dynamics on timescales of order m is adiabatic : at each increase of γ ( t ) on timescales of order m , the train error relaxes to the the value it would have had if the second layer weights would have been fixed in time at the corresponding value of γ .

A remarkable consequence of Eq. (F.48) is that that

<!-- formula-not-decoded -->

In words, in the large network limit, the norm of second-layer weights at the end of training is asymptotically the minimum norm that allows for interpolation.

## F.2 Multi-index model

In this section we generalize the computations of Section F.1 to the case in which the dataset has a structure produced via a k -index model. The weights of the second layer are set to a ( t ) = γ ( t ) √ m and evolve with GF. The initialization scale γ (0) = γ 0 is fixed and independent of m .

As in the pure noise case, we identify three dynamical regimes:

Figure 12: Training with pure noise data under lazy initialization: third dynamical regime. Left frame: Train error on timescales of order m . Right frame: GF trajectories in the plane γ (second layer weights) e tr (train error). Black dots represent pairs ( γ, e lz2 tr , ∞ ( γ )) , where e lz2 tr , ∞ ( γ ) is the train error achieved at the end of the first dynamical regime, cf. Section F.1.3. The data has been produced from the same model as in Fig. 11.

<!-- image -->

√

Figure 13: SymmDMFT predictions and large network scaling for lazy training in a single index model. Left: Projection v ( t ) of the first layer weights onto the latent direction on timescales of the order 1 /m . The result for m → ∞ , v lz1 , has been obtained by integrating analytically Eq. (F.55). Right: The behavior of C d ( t, 0) on timescales t = Θ(1) , compared with the scaling theory for m →∞ , namely C lz2 d . In both cases with h ( z ) = ˆ φ ( z ) = (9 / 10) z + z 2 / 2 , τ = 0 . 3 and α = 0 . 3 , γ 0 = 1 .

<!-- image -->

1. t = O (1 /m ) : γ ( t ) = γ 0 + o m (1) , ∥ w i ( t ) -w i (0) ∥ = Θ(1 / √ m ) . On this scale the network only learns a linear approximation of the target. Test and train error remain close to each other (Section F.2.1).
2. t = Θ(1) : γ ( t ) = γ 0 + o m (1) , ∥ w i ( t ) -w i (0) ∥ = Θ(1) . Test error does not change but train error decreases significantly (Section F.2.2).
3. t = Θ( m ) : This regime only emerges if γ 0 is below a certain interpolation threshold, i.e. γ 0 &lt; γ ∗ GF ( α, φ, τ ) . In this regime γ ( t ) grows until the threshold, and train error decreases to 0 while test error decreases to 0 (Section F.2.5).

## F.2.1 First dynamical regime: t = O (1 /m )

On this timescale, the SymmDMFT equations are solved, up to higher order terms, by the following ansatz:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with

In particular, Eq. (F.50) implies ∥ w i (0) -w i ( t/m ) ∥ = o m (1) : weights of the first layer change by small amount.

<!-- formula-not-decoded -->

The scaling functions defined in Eqs. (F.50)-(F.53) satisfy a set of equations that can be derived directly from the SymmDMFT equations:

Note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The solution of Eqs. (F.55) implies that

<!-- formula-not-decoded -->

Furthermore, on this timescale, the train and test error coincide and are given by

<!-- formula-not-decoded -->

The corresponding asymptotic value is given by where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The interpretation of this dynamical regime is analogous to the one of the same regime in the purenoise setting, as confirmed by Eq. (F.59) : the network learns the linear component of the data distribution.

In the left panel of Fig. 13 we test the scaling theory in this dynamical regime, as given by Eqs. (F.50) to (F.53). We plot the solution of the SymmDMFT equations, versus tm , for increasing values of m : the curve collapse well on their conjectured m →∞ limit.

## F.2.2 Second dynamical regime: t = Θ(1)

We next consider t = Θ(1) . One can show that the SymmDMFT equations are solved, up to higher order terms as m →∞ , by the following ansatz

<!-- formula-not-decoded -->

with γ ( t ) = γ + o m (1) and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In other words, on this time scale first layer weights move by order one ∥ w i ( t ) -w i (0) ∥ = Θ(1) , but in a linear subspace that is orthogonal to the latent space. Second layer weights do not move appreciably. As a consequence, no additional learning takes place in this regime, but the model begins to overfit the data.

Note that the above scaling form is compatible with the long time limit of the previous dynamical regime.

In order to define the equations for the functions on the right-hand side of Eq. (F.61) we define R lz2 A and C lz2 A to be the solution of where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting the ansatz (F.61) into the SymmDMFT equations, and using Eqs. (F.62), we obtain the following equations for C lz2 d ( t, t ′ ) , R lz2 d ( t, t ′ ) , ν lz2 ( t )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, the train and test errors converge to well defined limits for t fixed and m →∞ :

where

<!-- formula-not-decoded -->

Note that, using Eqs. (F.62), (F.64), and the fact that C lz2 d ( t, t ) = 1 (because of the unit norm constraint on the first layer weights), we get

<!-- formula-not-decoded -->

<!-- image -->

tm

t

Figure 14: SymmDMFT predictions and large network scaling for lazy training in a single index model: train and test error. Left frame: train and test error on the time scale t = Θ(1 /m ) for several values of m , together with the asymptotic prediction as m → ∞ on this time scale e lz1 tr ( ˆ t, γ 0 ) = e lz1 ts ( ˆ t, γ 0 ) . Right: train and test error on the time scale t = Θ(1) for several values of m , together with the asymptotic prediction as m →∞ on this time scale e lz2 tr ( t, γ 0 ) . Here γ 0 = 1 , h ( z ) = ˆ φ ( z ) = (9 / 10) z + z 2 / 2 , τ = 0 . 3 , α = 0 . 3 .

Using Eq. (F.57), we obtain that the asymptotic test error in this dynamical regime is constant and equal to the test error achieved at the end of the previous regime, namely e lz2 ts ( t, γ 0 ) = e lz1 ts , ∞ , cf. Eq. (F.59). As anticipated, no learning takes place on this timescale.

The predictions of Eqs. (F.61) are tested in the right panel of Fig. 13. We plot the correlation function C d ( t, 0) for several values of m , as obtained by solving the SymmDMFT equations. We compare these results with the m → ∞ prediction C lz2 d ( t, 0) obtained by solving Eqs. (F.66) to (F.68). We observe collapse of finite m curves on the large m asymptotics supporting our conclusions.

In Fig. 14 we plot the behavior of the train and test error both on timescales t = Θ(1 /m ) (left frame, plotting e tr ( t, γ 0 ) , e ts ( t, γ 0 ) versus tm ) and t = Θ(1) (right frame, plotting e tr ( t, γ 0 ) , e ts ( t, γ 0 ) versus tm ). We the solutions of SymmDMFT equations at increasing values of m with the theory scaling theory presented in the previous section (for t = Θ(1 /m ) , left frame) and in this section (for t = Θ(1) , right frame). As anticipated, we observe the following:

- On the time scale t = Θ(1 /m ) (left panel), test and train error collapse (as m →∞ ) on a common limiting curve e lz1 tr ( ˆ t, γ 0 ) = e lz1 ts ( ˆ t, γ 0 ) which converges, for large ˆ t , to the positive limiting value e lz1 ts , ∞ characterized in the previous section.
- On the time scale t = Θ(1) (right panel), test and train error collapse (as m → ∞ ) on two distinct limiting curves. The first one is constant and equal to e lz1 ts , ∞ . The second one decreases from e lz1 ts , ∞ to 0 and is predicted by the asymptotic theory in this section, cf. Eq. (F.70).

Note that, in the example of Fig. 14, the initialization γ 0 is sufficiently large that the train error decreases to zero on the time scale Θ(1) , namely γ 0 &gt; γ ∗ GF ( α, φ, τ ) , for a suitable threshold γ ∗ GF ( α, φ, τ ) . As we will see in the next section, a third dynamical regime emerges when γ 0 &lt; γ ∗ GF ( α, φ, τ ) .

## F.2.3 The algorithmic interpolation threshold

The asymptotic theory within the second dynamical regime, described in Section F.2.2, turns out to be equivalent to the one in the pure-noise model, Section F.1.2, up to a change of variables. Namely, defining

<!-- formula-not-decoded -->

with initial condition ˜ C o (0 , 0) = -1 , reduce the equations of Section F.2.2 to the ones of Section F.1.2 with noise level τ replaced by

<!-- formula-not-decoded -->

Figure 15: The asymptotic behavior of the test error as a function of m for different h ( z ) = ˆ φ ( z ) . We observe that soon as h ( z ) contains a z 2 term, the NTK limit for m → ∞ is approached from below (left panel). Furthermore the speed of the convergence to the limiting value depends crucially on whether a z 2 monomial is present in the Taylor expansion of h ( z ) (right panel). The data has been produced with α = 0 . 3 and τ = 0 . 6 .

<!-- image -->

The interpretation of this reduction is simple. On the time scale t = Θ(1) , the first layer weights move orthogonally to the latent subspace spanned by U . Hence, the dynamics on this timescale is not affected by the signal and only attempts to fit the labels noise. The noise is inflated as per Eq. (F.73), because the network is not able to fit beyond the linear part of the target distribution.

As a corollary of the above equivalence, the interpolation threshold of the k -index model coincides with with the interpolation threshold on pure noise data with noise level given by Eq. (F.73). Using the extended notation γ ∗ GF ( α, φ, τ ) to indicate the dependence on the underlying data distribution (which is parametrized by φ, τ ), we can write the stated relation as

<!-- formula-not-decoded -->

(Here we used the invariance under rescaling in the pure noise model, which implies γ ∗ GF ( α, 0 , τ 2 ) = τγ ∗ GF ( α, 0 , 1) .)

## F.2.4 Dependence on m

Within NTK theory, it is normally assumed that optimal models are achieved at very large network sizes m → ∞ . Empirical results contradicting this expectation have been put forward in [54], but no theoretical analysis was provided either in [54] or in subsequent work. We can use the SymmDMFT theory to fill this gap and study the dependence of test error on the number of neurons m under lazy initialization. We choose γ 0 &gt; γ ∗ GF ( α, φ, τ ) , and therefore vanishing training error is reached during the second dynamical regime, i.e. for t = Θ(1) : this is therefore the last dynamical regime. Throughout this regime, we have γ ( t ) = γ 0 + o m (1) .

Recalling that e ts ( t, γ 0 , m, α ) is the test error at time t in this setting, as predicted by SymmDMFT we consider the limit

<!-- formula-not-decoded -->

We note that, for γ 0 &gt; γ ∗ GF ( α, φ, τ ) , we expect

<!-- formula-not-decoded -->

to be given by Eq. (F.59).

In Fig. 15 we plot the SymmDMFT prediction for e ℓ ∞ ( γ 0 , m, α ) as a function of m for several choices of h (we use h = ˆ φ here). The limit m → ∞ of these curves matches e lz1 ts , ∞ as expected. However we empirically observe that e ℓ ∞ ( γ 0 , m, α ) approaches e lz1 ts , ∞ in two qualitatively different ways:

Figure 16: Train and test error on different timescales when training on single index data and lazy initialization. Train error (solid curves) and test error (dashed curves) for a model trained on a single index data with h ( z ) = (9 / 10) z + z 2 / 2 = ˆ φ ( z ) . The noise level is τ = 2 . 5 and initialization a (0) = γ 0 √ m , γ 0 &lt; γ ∗ GF ( α, φ, τ ) . Left panel: timescales of order one. The grey dashed line corresponds to the scaling solution for m →∞ when the second layer does not evolve with GF. Right panel: same data plotted versus t/m , to explore timescales of order m . The arrows show scaling appearing and curves collapsing on a master curve.

<!-- image -->

̸

- In the cases we consider that have h ′′ (0) = 0 , e lz1 ts , ∞ is approached from below as m →∞ , and e ℓ ∞ ( γ 0 , m, α ) is non-monotone. We also observe that, for the values of m we consider, the approach to the asymptotic value is compatible with a rate m -1 / 2 : e ℓ ∞ ( γ 0 , m, α ) = e lz1 ts , ∞ -Θ( m -1 / 2 ) .

̸

- In the cases we consider that have h ′′ (0) = 0 , then e lz1 ts , ∞ is approached from above as m →∞ , and e ℓ ∞ ( γ 0 , m, α ) is typically monotone. In this case the approach to the limiting behavior is compatible with a rate m -1 : e ℓ ∞ ( γ 0 , m, α ) = e lz1 ts , ∞ +Θ( m -1 ) .

The first scenario is the generic one, and similar to what is observed in [54] for actual neural networks. An intuitive explanation is that -at finite m - the projection of neurons onto the latent space ∥ v lz1 ∞ ∥ = Θ(1 / √ m ) is sufficient for the network to partially learn the quadratic component of the target function. In order to establish on more solid grounds these empirical observations one should study the 1 /m corrections to the scaling theory developed here. This is left for future work.

## F.2.5 Third dynamical regime: t = Θ( m )

As for the pure noise case, beyond the time scale t = Θ(1) , we distinguish two situations. If γ 0 &gt; γ ∗ GF ( α, φ, τ ) , then vanishing training error is reached within the second dynamical regime t = Θ(1) . If γ 0 &lt; γ ∗ GF ( α, φ, τ ) , GF dynamics develops an additional regime for t = Θ( m ) . In this section, we study this third regime.

In Figure 16, we plot the SymmDMFT predictions for train and test errors as a function of time for several values of m , for a setting with γ 0 &lt; γ ∗ GF ( α, φ, τ ) . In particular, in Fig.16-left we plot train and test error as a function of t . The curves for the train error for increasing value of m collapse on limit curve given by e lz2 tr ( t, γ 0 ) characterized in Section F.2.2. In other words, the dynamics on this timescales follows the scaling theory of Section F.2.2. However in this case γ 0 &lt; γ ∗ GF ( α, φ, τ ) , whence by definition e lz2 tr , ∞ &gt; 0 . This correspond to the limit curve in Fig. 16-left having a strictly positive asymptote.

Figure 16-right shows train and test error plotted against t/m . We observe that curves training error curves collapse on a common limit, that decreases from e lz2 tr , ∞ to 0 , while test error curves increase above the plateau e lz1 ts , ∞ . This suggests the following limit behavior

<!-- formula-not-decoded -->

Figure 17: Training a two layer network in the same setting of Figure 16. Left panel: second layer weights on the timescale of order m . The black arrow corresponds to the interpolation threshold for a model, γ ∗ GF ( α, τ ) obtained by fitting the relaxation time as a function of the weights of an lazy initialized model for γ 0 &gt; γ ∗ GF ( α, τ ) . The second layer weights, at finite m develop a plateau at long time. In the inset we show the approach of this plateaus to the limiting value given by γ ∗ GF ( α, φ, τ ) . Right panel: parametric plot of the train error as a function of the scaled weights of the second layer. The dashed gray dashed line corresponds to the extrapolated train error for an network with second layer weights fixed to the corresponding value in the m → ∞ (as extracted from the numerical integration of the scaling theory).

<!-- image -->

In order to further explore the GF dynamics in this regime, in Fig. 17-left we plot the evolution of the second layer rescaled weights against t/m . The curves for increasing values of m collapse on a master curve, suggesting the existence of a limit

<!-- formula-not-decoded -->

The limit curve γ lz3 ( ˆ t, γ 0 ) increases from γ 0 to a limit value:

<!-- formula-not-decoded -->

As in Section F.2.5 we consider the inverse function of t ↦→ γ lz3 ( t, γ 0 ) , denoted by γ ↦→ ˜ γ -1 ( γ, γ 0 ) . In Fig. 17-right we plot the train error as a function of the second layer weights γ ( t ) . Again, for increasing values of m the curves collapse on a master curve which is given by

<!-- formula-not-decoded -->

We then also plot in Fig.17-right the asymptotic value of the train error for a network initialized with second layer weights blocked at an initialization scale γ , call it e lz2 tr , ∞ ( γ ) .

The curves ε ( γ, γ 0 ) appear to have a vertical segment (corresponding to t = o ( m ) ) in which the training error decreases, while γ ( t ) = γ 0 + o m (1) is nearly unchanged, and a continuously decreasing segment in which γ ( t ) increases while e tr ( t, γ 0 ) decreases to 0 (corresponding to t = Θ( m ) ). In the second phase, the curves appear to converge to e lz2 tr , ∞ ( γ ) as m →∞ . This suggests

<!-- formula-not-decoded -->

In other words the dynamics on timescales of order m is adiabatic also in the multi index case. For a small change of the second layer weights on a scale of order √ m , the train error relaxes to its asymptotic value on timescales of order one. This graph suggests that the limit value of γ ( t ) coincides with the critical value for interpolation. Namely recalling the definition (F.79) for the asymptotic value of γ ( t ) , we have

<!-- formula-not-decoded -->

where the interpolation threshold in the multi-index model γ ∗ GF ( α, φ, τ ) is related to the interpolation threshold in the pure noise model via Eq. (F.74).

## G Dynamical regimes: Mean field initialization

In this section we assume the initialization of the weights of the second layer is kept of order one. To be definite, we set a (0) = a 0 , independent of m . This corresponds to the mean field initialization studied in [38, 14, 45].

Specializing to the data distribution considered here, earlier work characterized the dynamics up to time T , under a few settings (which prove equivalent in this regime):

- One-pass SGD, with stepsize ε ≪ 1 /d and therefore time horizons such that T ≪ d/n (the latter inequality follows from T ≤ nε for one-pass SGD). In this case, the dynamics is characterized by a set of ODEs for for the projections of the weights on the latent space and inner products between weights.
- Gradient flow in the population risk, which admits the same characterization and corresponds to the limit n →∞ of the above.
- The limit of the above regimes for large width m →∞ . This is characterized by a partial differential equation for the distribution of projections of first layer weights onto the latent space, provided T ≤ c 0 log m , for c 0 a sufficiently small constant.

We refer to [5, 21, 1, 6, 2, 10] for a few pointers to this literature. In all of these settings, the train error remains close to the test error. In contrast, the analysis presented here allow us to explore the overfitting regime.

Section G.1, we will focus on a pure noise data distribution, while Section G.2, considers a multiindex model. As in the case of lazy initializations, we consider first the limit n, d →∞ at n/md = α and m fixed (hence characterized by SymmDMFT ) and subsequently study dynamical regimes emerging as m →∞ at n/md = α fixed.

## G.1 Pure noise model

Under the pure noise model, we have φ = ˆ φ = 0 . We identify three distinct dynamical regimes:

- t = O (1) : a ( t ) = a 0 + o m (1) , e tr ( t ) = τ 2 / 2 + o m (1) , and ∥ w i ( t ) -w i (0) ∥ = o m (1) . In words, the weights change minimally and the train error remains close to the one of the null network f ( x ; θ ) ≈ 0 (Section G.1.1).
- t = Θ( m ) . In this regime a ( t ) = √ mγ ( t/m ) + o m (1) , and therefore the network complexity becomes large enough for it to fit the noise. The dynamics on this timescale is closely related to the one under lazy initialization, studied in Section F.1.4. In particular, γ ( ˆ t ) converge to the interpolation threshold γ ∗ GF ( α, τ ) if ˆ t →∞ (after m →∞ ). (Section G.1.3).
- t = Θ( √ m ) : a ( t ) = Θ(1) , e tr ( t ) = τ 2 / 2+ o m (1) , and ∥ w i ( t ) -w i (0) ∥ = Θ(1) . Namely, weights change but the train error does not change significantly. (Section G.1.2).

## G.1.1 First dynamical regime: t = O (1)

In this dynamical regime, the SymmDMFT equations are solved by the following scaling ansatz

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Furthermore we have

<!-- formula-not-decoded -->

Plugging the scaling ansatz in the SymmDMFT , we obtain equations determining the scaling functions C mf1 o , R mf1 o . Defining

<!-- formula-not-decoded -->

<!-- image -->

t

t

Figure 18: Training on pure noise data under mean-field initialization: t = Θ(1) regime. We plot C o ( t, 0) and C o ( t, t ) as given by solving the SymmDMFT equations for different values of m and compare them with the asymptotic solution of Section G.1.1. Here we use τ = 0 . 6 , α = 0 . 3 and h ( z ) = (9 / 10) z + z 3 / 6 . Note that the vertical axis is multiplied by a factor m , in agreement with the prediction of Eq. (G.2).

we have

<!-- formula-not-decoded -->

In particular

<!-- formula-not-decoded -->

The equations (G.4) imply that the train error is constant in this regime and equal to

<!-- formula-not-decoded -->

̸

The above predictions are tested in Fig. 18 where we plot C o ( t, t ) and C o ( t, 0) for different values of m and check their approach to the scaling functions C mf1 o ( t, 0) and C mf1 o ( t, t ) .

In other words, in this regime both first and second layer weights change minimally and the resulting error remains close to the one to the null function f ( x ; θ ) ≈ 0 . We will see that this regime is significantly more interesting for the case of data with a signal, see Section G.2. We note in passing that the limit value ⟨ w j , w j ⟩ ≈ τ 2 -ρ 0 mρ 0 for i = j corresponds to minimizing the empirical risk under the linear approximation in which σ ( z ) is replaced by √ h ′ (0) z .

## G.1.2 Second dynamical regime: t = Θ( √ m )

We now consider the case in which time scales as √ m . The following asymptotic forms can be checked to solve the SymmDMFT equations, up to higher order terms, for suitable choices of the scaling functions on the right-hand side:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 19: Training on pure noise data under mean-field initialization: t = Θ( √ m ) regime , under the same setting as in Fig. 18. We plot the solutions of the SymmDMFT equations for several values of m as a function of t/ √ m . We compare these to the m → ∞ scaling theory of Section G.1.2, i.e. to numerical solutions of Eqs. (G.14) to (G.17).

<!-- image -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Plugging this scaling ansatz into the SymmDMFT equations we get the constraints

<!-- formula-not-decoded -->

Wealso obtain that the following equations must be satisfied by C mf2 d ( t, t ′ ) , R mf2 d ( t, t ′ ) , a mf2 ( t ) , ν mf2 ( t ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with initial conditions given by

<!-- formula-not-decoded -->

We test these predictions in Fig. 19. We plot several quantities in the solution of the SymmDMFT equations for increasing values of m and compare them with the solution of the asymptotic equations (G.14) to (G.17). We observe convergence to the predicted asymptotic behavior.

Equations (G.14) to (G.17) can be further simplified. The right-hand side of Eq. (G.17) is a positive. Therefore a mf2 ( t ) is a monotone increasing function. Define the time change

<!-- formula-not-decoded -->

and the corresponding time-changed scaling functions

<!-- formula-not-decoded -->

Equations (G.14) to (G.17) imply that these time-changed function functions satisfy

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where again ˜ h ( z ) = h ( z ) -h ′ (0) z .

Equations (G.21), (G.23) are independent of the dynamics of the second layer weights. These equations are nothing but the DMFT equations describing gradient descent dynamics of the celebrated spherical mixed p -spin glass model [17, 20, 8, 23], whose definition we recall next. Consider a random cost function H ( x ) indexed x ∈ S d -1 , which is a centered Gaussian process with covariance structure given by

<!-- formula-not-decoded -->

Define the gradient flow dynamics

<!-- formula-not-decoded -->

where P ⊥ x ( t ) is the projector orthogonal to x ( t ) . Then the high-dimensional asymptotics of this dynamics is characterized by Eqs. (G.21), (G.23). In particular lim d →∞ ⟨ x ( t ) , x ( s ) ⟩ = ˜ C mf d ( t, t ′ ) almost surely.

A particularly interesting quantity is the asymptotic energy value in the mixed p -spin model:

<!-- formula-not-decoded -->

The DMFT analysis for this problem implies that

<!-- formula-not-decoded -->

Figure 20: Evolution of second layer weights, as predicted by the numerical solution of Eq. (G.17) . Here we use h ( z ) = (9 / 10) z + z 3 / 6 , α = 0 . 3 and τ = 0 . 6 . The straight line is just a guide to the eyes to test the prediction of Eq. (G.29).

<!-- image -->

For ˜ h ( z ) = c 2 k z k , k ≥ 2 , we have the explicit expression [20, 19, 46]

<!-- formula-not-decoded -->

An explicit expression for E for general covariance structure is an unknown [23].

The asymptotic energy E has an interesting interpretation for the dynamics of two-layer networks -within the SymmDMFT theory. Eq. (G.28) implies that

<!-- formula-not-decoded -->

In Fig. 20 we test the prediction of Eq. (G.29) by integrating numerically Eqs. (G.14) to (G.17) and plotting the prediction for the second-layer weigths a mf ( t ) . We observe that at large t , a mf2 ( t ) ≈ A ∞ t , with A ∞ given by Eq. (G.29)as predicted.

We also note that C A ( t, t ) = -τ 2 also in this timescale, and hence the train error does not change significantly. Namely , for any constant t , we have

<!-- formula-not-decoded -->

If we use heuristically Eq. (G.28) and Eq. (G.12) beyond the √ t time scale, we obtain

<!-- formula-not-decoded -->

This suggests that a ( t ) becomes of order √ m on timescale of order m . When this happens, the network complexity is large enough to allow for interpolation, and hence we expect the dynamics to change. Indeed a new dynamical regime emerges for t = Θ( m ) , as we will study next.

## G.1.3 Third dynamical regime: t = Θ( m )

As anticipated, an additional regime arises on timescales of order m . In Figure 21 we plot the evolution of the weights of the second layer as a function of t/m for increasing values of the width m . The different curves collapse suggesting the following limit to exist

<!-- formula-not-decoded -->

The limit curve appears to grows linearly at small t , γ mf3 ( t ) = A ∞ t + o ( t ) , where A ∞ is the coefficient computed in the previous section, cf. Eq. (G.29). Hence, this third dynamical regime matches directly with the previous one. As can be seen from the right plot, there appear to be a finite limit lim t →∞ γ mf3 ( t ) &lt; ∞ .

√

Figure 21: Evolution of the second layer weigths when training on pure noise data under mean field initialization for t = Θ( m ) . Rescaled second layer weights a ( t ) / √ m as a function of t/m . We plot solutions of the SymmDMFT equations for the setting of Fig. 18.

<!-- image -->

Figure 22: Train error and Lagrange multiplier ν ( t ) on timescales of order m under mean field initialization for pure noise data. Solutions of the SymmDMFT equations for the setting of Fig. 18. Finite m curves accumulate on master curves suggesting the existence of scaling functions.

<!-- image -->

We now turn to the analysis of the train error. Recall that on the previous timescales, the train error stays approximately constant, and equal to the train error of the null network, namely e tr ( t √ m ) = τ 2 / 2 + o m (1) for any fixed t . In Fig. 22 we plot both the train error and the Lagrange multiplier ν as a function of t/m . Again, as m grows, these curve converge to limit curves. This suggests the existence of the following limits

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that in this case, differently from the lazy initialization setting, the corresponding scaling function do not depend on the initialization parameter a 0 .

In order to characterize the limits in Eqs. (G.33)-(G.34), we proceed as in Sec. F.1.4. Namely, in Fig. 23 we plot the train error and the Lagrange multiplier ν as a function of the rescaled second layer weights γ = a ( t ) / √ m . We also plot the asymptotic value of train error and Lagrange multiplier under the constrained GF dynamics in which second layer weigths are fixed to a ( t ) = γ √ m and do not evolve with time: e lz2 tr , ∞ ( γ ) := lim t →∞ e lz2 tr ( t, γ ) and ν lz2 ∞ ( γ ) := lim t →∞ ν lz2 ( t, γ ) . These are computed by integration of the scaling theory in Section F.1.2.

The good collapse on these curves suggests to consider the the following construction, analogous to Sec. F.1.4. Define the inverse function of t ↦→ γ mf3 ( t ) , denoted by ( γ mf3 ) -1 . Then, define

<!-- formula-not-decoded -->

Figure 23: Train error, rescaled second-layer weights and the Lagrange multiplier ν ( t ) on timescales of order m under mean field initialization. Left Panel: parametric plot of the train error as a function of the weights of the second layer on the scale √ m , namely γ = a ( t ) / √ m . The inset shows the same data on a logarithmic scale. Right Panel: same plot for the Lagrange multiplier ν . Data is the same as in Fig. 18.

<!-- image -->

Figure 23 suggests that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equations (G.36), (G.36) imply that on timescales of order m the dynamics is adiabatic. For each incremental change of a on a the scale √ m , all one-time quantities relax to the asymptotic value which turns out to be the same as a constrained model with a ( t ) / √ m = γ fixed.

The consequence of Eqs. (G.36)-(G.36) is that

<!-- formula-not-decoded -->

where γ ∗ GF ( α, τ ) corresponds to the interpolation value of the initialization scale of a lazy model.

## G.2 Multi-index model

In this section we consider the case in which the dataset is distributed according to a multi-index model. For time scales beyond t = O (1) , we will assume that h ( z ) = ˆ φ ( z ) . This simplifies the asymptotics for t large but of order one.

We identify two dynamical regimes emerging as m →∞ :

- t = O (1) : a ( t ) = O (1) but is not constant. Also, the projection v ( t ) of first layer weights onto the latent space evolve as well as do train and test error. We further have e tr ( t ) = e ts ( t ) + o m (1) : there is no overfitting. This evolution is captured the mean field theory of [38, 14] which we recover as m →∞ limit of SymmDMFT .
- t = Θ( m ) : a ( t ) = Θ( √ m ) , v ( t ) decreases towards 0 and train and test error diverge. In this dynamical regime the network unlearns to a large extent the latent structure of the data and overfit it.

## G.2.1 First dynamical regime: t = Θ(1)

For timescales of order one, the SymmDMFT equations are solved, up to subleading terms as m →∞ , by the following ansatz

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 24: Gradient flow dynamics under mean field initialization in the first dynamical regime t = O (1) . for data distributed according to a single index model. Curves are numerical solutions of the SymmDMFT equations: we plot v ( t ) and a ( t ) for different values of m and compare them to the mean field predictions. Data is distributed according to a single index model with h t ( z ) = ˆ φ ( z ) = h ( z ) = (9 / 10) z + z 3 / 6 with τ = 0 . 6 and α = 0 . 3 .

<!-- image -->

Figure 25: Evolution of the train and test error on different timescales under mean field initialization a (0) = 1 . The train (solid curves) and test (dashed curves) errors as a function of time t (left panel) and scaled time t/m (right panel). Curves are numerical solutions of the SymmDMFT equations for h ( z ) = (9 / 10) z + z 3 / 6 , ˆ φ ( z ) = h ( z ) , τ = 0 . 6 and α = 0 . 3 . The arrow on the right panel corresponds to the asymptotic test error for a model with second layer weights fixed to the corresponding interpolation threshold.

<!-- image -->

The corresponding scaling equations are then given by

<!-- formula-not-decoded -->

These equations are solved by setting:

<!-- formula-not-decoded -->

<!-- image -->

t

t

Figure 26: Finite width corrections. The 1 /m corrections to the second layer weights and the projection on the latent space of the single index model on timescales of order 1 . Dashed lines are obtained by integrating numerically Eqs. (G.56) to (G.59) determining the limits m → ∞ . Here, ˆ φ ( z ) = h ( z ) = (9 / 10) z + z 3 / 6 with τ = 0 . 6 and α = 0 . 3 .

with v mf1 ( t ) , a mf1 ( t ) the solution of

<!-- formula-not-decoded -->

with initial conditions given by v mf1 (0) = 0 and a mf1 (0) = a 0 .

Equations (G.44) coincide with the mean field theory of [38, 14, 45], when the latter are specialized to the multi-index model studied here, under symmetric initializations [10]. (See also [2].) Using the ansatz of Eqs. (G.39) to (G.41) in the formulas for training and test error (C.45), (C.46), we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with

Aparticularly simple case is the one in which k = 1 (single index model) and φ = σ (whence ˆ φ = h ). For a class of such activations with h ′ (0) &gt; 0 , we have a mf1 ( t ) , v mf1 ( t ) → 1 as t →∞ and therefore

In other words, neurons align perfectly with latent direction, the generalization error vanishes, and and train and test error converge for large constant t to the Bayes error τ 2 / 2 .

In Fig. 24 we compare the solution of Eqs. (G.44) with the numerical integrations of the SymmDMFT equations for a range of values of m . As m increases, the SymmDMFT solutions converge to the asymptotic predictions v mf1 ( t ) , a mf1 ( t ) , confirming the above ansatz.

Similarly, in Fig. 25-left panel we compute the train and test error by solving the SymmDMFT equations and compare the results to the asymptotic prediction provided by Eq. (G.46). We observe that -as predicted- train and test error match on an increasingly long time interval. At a certain point, they diverge: we will next characterize the timescale on which this happens.

## G.2.2 Escape from the mean field dynamical regime

In order to understand on which time scale the dynamics diverges from mean field theory described above, we will study small deviations from this theory. We expect that these deviations will diverge with time. Characterizing this divergence will allow to determine time scale on which we exit the mean field regime.

We focus on the case of a single index model k = 1 , with ˆ φ = h , and set a (0) = 1 . We believe that the qualitative conclusions obtained in this case apply more generally. We also assume h to be such

that the long time asymptotics of mean field dynamical solutions is

<!-- formula-not-decoded -->

As mentioned in the previous section, this holds for a broad class of activations. In other words, for time t large and yet of order one, the neurons are very well aligned.

We next study the corrections to the mean field solution. We claim that such corrections are of order 1 /m and define the functions ˜ a ( t ) , ˜ v ( t ) , dots , ˜ R o ( t, t ′ ) , via

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting the above form into the SymmDMFT equations and matching the next-to-leading order in m we can obtain the equations for the 1 /m corrections. It turns out that equations for ˜ a, ˜ v, ˜ C o and ˜ ν decouple from the equations for ˜ C d , ˜ R d and ˜ R o . Given that we are interested in the former quantities we only report the corresponding equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, we used the following auxiliary functions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that Eqs. (G.56) to (G.59) are a set of four integral-differential equations for the four functions ˜ a ( t ) , ˜ v ( t ) , ˜ ν ( t ) , ˜ C o ( t, t ′ ) . The original SymmDMFT equations involve three other functions: ˜ C d ( t, t ′ ) , ˜ R d ( t, t ′ ) , ˜ R o ( t, t ′ ) ? We also remark that: ( i ) These equations are linear in the unknowns ˜ a ( t ) , ˜ v ( t ) , ˜ ν ( t ) , ˜ C o ( t, t ′ ) ; ( ii ) They can be integrated numerically with the same strategy used to integrate the SymmDMFT equations.

In Fig. 26 we plot the deviations from the mean field limit m ( a ( t ) -a mf1 ( t )) and m ( v ( t ) -v mf1 ( t )) as a function of time t , as obtained by solving the SymmDMFT equations 1 , for several values of m . We also plot the predicted limits ˜ a ( t ) , ˜ v ( t ) , which are obtained by integrating Eqs. (G.56) to (G.59) As m gets large, the finitem curves appear to converge to the predictions ˜ a ( t ) , ˜ v ( t ) .

In Figure 27 we plot the result of integrating Eqs. (G.56) to (G.59) over a wider time window. We observe that ˜ v , ˜ a , ˜ ν and ˜ C o ( t, t ) diverge linearly with t .

This suggests the following asymptotics for these corrections

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The values of the constant a ∗ , v ∗ , ν ∗ and c ∗ can be obtained analytically by using the above ansatz in Eqs. (G.56) to (G.59). We obtain that they solve the following linear equations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The asymptotic linear behavior predicted by Eqs. (G.69), (G.70), with the coefficients determined by Eqs. (G.71)-(G.74) is plotted in Fig. 27. We observe good agreement with the numerical integration of Eqs. (G.56) to (G.59).

where

Figure 27: Finite width corrections to dynamical observables under mean field initialization. The 1 /m corrections to v ( t ) , a ( t ) , C o ( t, t ) and ˜ ν ( t ) as a function of time as extracted from the numerical integration of the corresponding equations. The dashed lines are the asymptotic predictions for t →∞ which show that the divergence of all quantities is linear with time. Here, ˆ φ ( z ) = h ( z ) = (9 / 10) z + z 3 / 6 with τ = 0 . 6 and α = 0 . 3 .

<!-- image -->

√

Figure 28: econd layer weights and projection on of the first layer weigths onto the latent structure of the data for gradient flow under mean field initialization on timescales of order m . Left: rescaled second layer weights a ( t ) /m as a function of the rescaled time t/m . The arrow on the right points at the threshold γ ∗ GF ( α, φ, τ ) for interpolation under gradient flow, see Section F.2.3. Right: projection of the first layer weights on the latent space in the single index model as a function of rescaled time t/m . Here, ˆ φ ( z ) = h ( z ) = (9 / 10) z + z 3 / 6 with τ = 0 . 6 and α = 0 . 3 . v = 1 /γ in (F.57).

<!-- image -->

√

<!-- image -->

Figure 29: Parametric plot of the rescaled projection onto the latent direction v √ m against rescaled second layer weights γ = a/ √ m . Same data as in Fig. 28. Dashed line is v √ m = 1 /γ .

<!-- image -->

tr

Figure 30: Train and test error of gradient flow under mean field initialization, for increasing values of m . Left: train error as a function of rescaled weights a ( t ) / √ m . Dashed line is the Bayes error τ 2 / 2 . Curves are traversed in time from top to bottom. Right: test error versus train error. Curves are traversed in time from right to left. Here ˆ φ ( z ) = h ( z ) = (9 / 10) z + z 3 / 6 with τ = 0 . 6 and α = 0 . 3 .

<!-- image -->

Figure 31: The difference between test and train error for the single index data. Left panel: the difference between test and train error plotted as a function of a/ √ m and compared to what is obtained from a model with fixed second layer weights initialized with Lazy scaling. Right panel: the difference between test and train error on timescales of order m . Here, ˆ φ ( z ) = h ( z ) = (9 / 10) z + z 3 / 6 with τ = 0 . 6 and α = 0 . 3 .

The above analysis implies that (considering to be definite second layer weights, and projection of first layer weigths onto the latent direction), for m ≫ 1 , t ≫ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where lim m →∞ ∆ a/v ( t, m ) = 0 . If we neglect the error terms, and assume that this expression holds for t larger than O (1) in m , then it indicates that a ( t ) , v ( t ) differ significantly from the mean field prediction when t/m becomes of order one. We expect therefore a third dynamical regime for t = Θ( m ) , which will be the object of the next section.

## G.2.3 Second dynamical regime: t = Θ( m ) and beyond

As pointed out at the end of the previous section, we expect a third dynamical regime when t = Θ( m ) . By this time, the stability calculation in the previous section indicates that second layer weights become of order √ m . Figure 28 confirms this, and shows that, in the same regime v ( t ) becomes small. In fact, numerical solution of the SymmDMFT equations are consistent with a ( t ) = Θ( √ m ) , v ( t ) = Θ(1 / √ m ) , and a ( t ) v ( t ) ≈ 1 for t = Θ( m ) .

For a small constant c denote by t 0 ( m ; c ) the time at which a ( t 0 ( m ; c )) = c √ m . We then expect that the following exists

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

provided w ( m ) is a suitable function (with w ( m ) = O ( t 0 ( m ; c )) ). The stability analysis in the previous section suggests that t 0 ( m ; c ) ≤ t ∗ ( c ) m + o ( m ) . Our numerical solutions do not cover a large enough range of values of m to verify this ansatz, and determine the scaling of w ( m ) with m . On the other hand, they indicate that indeed t 0 ( m ; c ) = Θ( m ) .

Since the second layer weights become of order √ m in this dynamical regime, train and test error start to differ significantly. We expect

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This picture is confirmed by Fig. 30, which reports train and test error as predicted by numerical solutions of the SymmDMFT equations for increasing values of m . On the left, we plot the train error as a function of the rescaled second layer weights γ = a/ √ m . We observe that curves for different values of m decrease until they reach the Bayes error τ 2 . On this phase however different curves do not collapse corresponding to the fact that γ vanishes. In the second phase, γ grows to be of order one and correspondingly the train error decreases below the Bayes error: this is the third dynamical regime. Overfitting takes place at this point.

In the right frame of Fig. 30, we plot test error versus train error. We observe, again, the two phases emerging for large m . In the first phase train error and test error are closely matched. In the second phase, train error decreases and test error correspondingly increases. Again, this takes place when t = Θ( m ) .

Finally, in Fig. 31, we repeat similar plots for the generalization error (difference between test and train error).

When t/m is large, the train error vanishes. We observe from Figure 28, left frame that, as t → ∞ , rescaled second layer weights reach a finite limit that is close to the interpolation threshold characterized in Section F.2.3. Namely

<!-- formula-not-decoded -->

1 We note that solving the SymmDMFT equations accurately enough to capture these corrections requires either to use very fine discretization, or a higher-order integration method.

<!-- image -->

m

α

GF

-

α

Figure 32: The interpolation transition for pure noise data and a network with second layer weights that do not evolve with time, fixed at a = 1 , see Section H. The noise level is fixed to τ = 1 and we considered h ( z ) = (3 / 10) z + z 2 / 2 . Top left panel: relaxation time ((rate for convergence to vanishing error) for different values of m . Top right panel: logarithmic plot of the relaxation time. The value of the algorithmic threshold for different values of m is a fitting parameter. Bottom left panel: values of the algorithmic thresholds as a function of m . Bottom right panel: the relaxation time as extracted from the scaling limit of the SymmDMFT equations in the m → ∞ limit. The algorithmic threshold is in this case α GF ( ∞ ) ≈ 1 . 18 which fits well the behavior plotted in the left bottom plot.

## H Dynamics under mean field initialization for n/d = α fixed

## H.1 Interpolation threshold at fixed a ( t ) = a 0

In this section, we consider an alternative scaling in the large width limit. As before, we use the SymmDMFT equations, and therefore study the limit n, d → ∞ with n/d → α . In the previous sections we studied the large width limit m →∞ with α = α/m fixed. In that setting interpolation is only possible when the network complexity scales, i.e. second-layer weights are a = Θ( √ m )

Here instead we keep a ( t ) = 1 and do not let evolve second-layer weights with GF. We consider pure noise data, and show that interpolation takes place if α &lt; α GF ( m ) , while the train error remains bounded away from zero for α &gt; α GF ( m ) . As expected from Gaussian complexity considerations, the threshold α GF ( m ) has a finite limit as m →∞ . In particular, for any α &gt; 0 , a network with a bounded cannot interpolate pure noise data.

As thorough in Sec.F we fix α and integrate numerically the SymmDMFT equations for finite but increasing values of m . We fix the initialization scale a 0 and the noise level τ and change only α .

We observe that for α small enough the train error decreases exponentially fast to zero. Namely, recalling that e tr ( t ; α ) := lim n,d →∞ ̂ R n ( a ( t ) , W ( t )) , we have that α &lt; α GF ( m ) ⇒ e tr ( t ; α ) = exp {-t/t ∗ rel ( α, m ) + o ( t ) } . (H.1)

However, the relaxation time time t ∗ rel ( α, m ) increases as α ↑ α GF ( m ) . Concretely, we define t rel ( α, m, c ) as the infimum time such that e tr ( t ; α ) ≤ c , where c is some small constant. In practice, we set c = 10 -7 . The results are plotted as a function of α for several values of m in Fig. 32, top left plot.

Figure 33: heck of the convergence of the numerical solution of the SymmDMFT for α fixed to the scaling solution for m →∞ . The left panel shows the behavior of the train error while the right panel shows the behavior of the correlation C d ( t, 0) . Both panels refer to a model where the teacher is pure noise with τ = 1 and the student is made of of neurons whose covariance structure is given by h ( z ) = (3 / 10) z + z 2 / 2 .

<!-- image -->

For each value of m the relaxation time appears to diverge at the critical point α GF ( m ) as an inverse power of α GF ( m ) -α , namely:

<!-- formula-not-decoded -->

The estimated interpolation thresholds α GF ( m ) are plotted as a function of m in the bottom left of Fig. 32. These data are consistent with the existence of a finite limit

The exponent ν appears to be independent of m . We fit this form to our data and extract the interpolation thresholds α rel ( m ) . In Fig. 32, top right, we plot t rel ( α, m, c ) /m as a function of the gap to this threshold. This plot confirms the form (H.2), with exponent ν ≈ 2 . Also, the fact that different curves superimpose indicate that L ( m,c ) ≈ L ∗ ( c ) m .

<!-- formula-not-decoded -->

and numerically α GF ( ∞ ) ≈ 1 . 18 .

In the next subsection, we derive equations describing the m →∞ limit for α = O (1) , a = O (1) fixed. Studying these equations yields further support to Eq. (H.3).

## H.2 Infinite width limit at fixed α

In order to study the limit m →∞ at fixed α , we discuss the limit of the SymmDMFT equations when m →∞ . As we have seen previously, the relaxation time of the train error is proportional to m . This is clearly visible in Fig. 32-top/left. This suggests that for m →∞ , dynamics takes place on timescales of order m . Therefore we propose the following asymptotic ansatz

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which defines a set of functions, ˜ R α d , ˜ C α d , ˜ R α o , ˜ C α o and ˜ ν α . We now describe the equations that these scaling functions satisfy satisfy. First we define ˜ C α A and ˜ R α A as the solution of

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Then we define the limit memory kernels:

<!-- formula-not-decoded -->

Substituting the above ansatz in the SymmDMFT equations and matching the leading order terms, we get the following equations that determine ˜ R α d , ˜ C α d , ˜ R α o , ˜ C α o and ˜ ν α :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

These are to be solved with boundary condition

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The scaling behavior of the train error is then given by

<!-- formula-not-decoded -->

In order to test the accuracy of the asymptotic analysis developed in this sections, we solved numerically the SymmDMFT equations for increasing values of m and compare the results to the numerical integration of Eqs. (H.10), (H.14) presented in this section. Some results of this comparison are presented in Fig. 33, which shows good agreement between finitem curves and m →∞ limit.

We conclude by emphasizing that, throughout this section α ( t ) = 1 and τ = 1 were fixed. If we generalize to arbitrary α ( t ) = a 0 and arbitrary τ &gt; 0 , the threshold α GF ( m ) will of course on these quantities through the ratio a 0 /τ .

The solution of Eqs. (H.10), (H.14) provides another route to estimate the largem interpolation threshold α GF ( ∞ ) at fixed a ( t ) = 1 . Namely, we solve the equations numerically and extract the t rel ( α, ∞ , c ) , which is defined analogously to above. We then fit the divergence of t rel ( α, ∞ , c ) at α GF ( ∞ ) according to Eq. (H.2). We obtain α GF ( ∞ ) ≈ 1 . 18 , in agreement with the threshold obtained by extrapolating the finitem thresholds α GF ( m ) . In the bottom right plot of Fig. 32 we plot t rel ( α, ∞ , c ) as function of α GF ( ∞ ) -α . This confirms the behavior of Eq. (H.2) with ν ≈ 2 .

## I Details about SGD simulations

In this appendix we provide some details about the numerical simulations with stochastic gradient descent (SGD) presented in Figures 2, 4.

We generate data according to the pure noise model y i = ε i (Fig. 2), y i = φ ( w T ∗ x i ) + ε i (Fig. 4), i ≤ n . We learn the two-layer network of Eq. (1.1), see below for the class definition.

```
class Net(nn.Module): def __init__(self, a, m, d): super().__init__() self.m = m self.lin1 = nn.Linear(d,m,bias=False) self.lin1.weight.data = (1/np.sqrt(d))*torch.randn((m,d)) self.lin2 = nn.Linear(m,1,bias=False) self.lin2.weight.data[0,:] = a self.act = Myact() self.project() def forward(self, x ): x1 = self.act(self.lin1(x)) return self.lin2(x1)/self.m def project(self, epsilon): row_norms = torch.norm(self.lin1.weight.data, dim=1, keepdim= True) row_norms = torch.clamp(row_norms , min=epsilon) self.lin1.weight.data = self.lin1.weight.data/row_norms
```

As shown in this code, we use the initialization where P B projects first layer weights to the unit ball:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We use the standard SGD iteration without weight decay and constant stepsize η , and batch size b :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The optimizer is defined in the code below

```
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0., weight_decay=0.) lambda_step = lambda epoch: 1 scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer , lr_lambda= lambda_step)
```

In the simulations of Figures 2, and 4 we use batch size b = 100 and step size η = 0 . 1 . Each symbol reports the average of N sim = 10 simulations.

## J Lower bounding the overfitting timescale

Throughout this appendix we use t to denote the rescaled time ˆ t introduced in Section 3.

## J.1 Proof of Theorem 3.1

By computing the derivative ∂ a i R n ( a ( t ) , W ( t )) , we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where, for n ≥ d , the last inequality holds with probability at least 1 -2 exp( -cn ) (for some universal c &gt; 0 ) by standard upper bounds on the norm of random matrices [53]. Further

<!-- formula-not-decoded -->

where in ( a ) it is understood that σ is applied entrywise to Xw i ∈ R n and in ( b ) we have g ∼ N ( 0 , I n ) , and φ is applied row-wise to XU ∈ R n × k . By using standard concentration on the norm of random matrices, also with probability 1 -exp( -cn ) , we have (for m ≤ n )

Summarizing the above bounds, we have

<!-- formula-not-decoded -->

which implies the first claim by integration.

<!-- formula-not-decoded -->

To prove the second claim, we consider the following sets of parameters W ∞ m,d ( a ) ⊆ W m,d ( a ) (which will also prove useful in the next section)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The second claim follows in turn if we prove that there exists a universal constant C such that

<!-- formula-not-decoded -->

This is a standard estimate, that we reproduce for the readers' convenience.

We begin by bounding the expectation of the supremum by symmetrization and contraction inequalities. Letting ( ξ i ) i ≤ n ∼ iid Unif ( { +1 , -1 } ) , we have

<!-- formula-not-decoded -->

We begin by bounding E 1 :

<!-- formula-not-decoded -->

where in ( a ) we applied the contraction inequality of [36] to the function ψ i ( t ) = y i σ ( t/ ( | y i | +1)) . We next bound term E 2 :

<!-- formula-not-decoded -->

where inequality ( b ) follows by applying the contraction inequality of [36] to ψ ( t 1 , t 2 ) = σ ( t 1 ) σ ( t 2 ) which is CL 2 -Lipschitz because ∥ σ ∥ Lip , ∥ σ ∥ ∞ ≤ L .

Summarizing, we proved that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In order to complete the proof of Eq. (J.3), we will show that the supremum concentrates around its expectation. For fixed ( a , W ) ∈ W m,d ( a ) , we have

∣ ∥ We write ̂ R n ( X ; a , W ) to emphasize the dependence of the risk on X Letting r ( x i ; a , W ) = φ ( U T x ) -f ( x ; a , W ) , we have

Hence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality holds on an event that has probability at least 1 -e -n . Defining Z n,d,m ( a ) := sup ( a , W ) ∈W m,d ( a ) ∣ ∣ ̂ R n ( a , W ) -R ( a , W ) ∣ , Borell inequality yields

<!-- formula-not-decoded -->

Together with Eq. (J.4), we thus obtain that the following holds with probability 1 -2 e -t -e -n

This yields the desired claim.

<!-- formula-not-decoded -->

## J.2 Proof of Theorem 3.2

We introduce the notations:

<!-- formula-not-decoded -->

We begin by establishing a uniform convergence lemma.

<!-- formula-not-decoded -->

Lemma J.1. Under the data distribution of Section A, assume ∥ φ ∥ ∞ ≤ L and the activation function to be bounded differentiable with Lipschitz continuous first derivative ∥ σ ∥ ∞ , ∥ σ ′ ∥ ∞ , ∥ σ ′ ∥ Lip ≤ L . Then there exists a universal constant C 1 , and a constant c 0 &gt; 0 dependent on L, τ, α such that, with probability at least 1 -2 exp( -nc 0 ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Gradient with respect to w ℓ . By a concentration argument, it is sufficient to consider the expected supremum. Writing the formula for ∇ w ℓ ̂ R n and using a standard symmetrization argument, we get

<!-- formula-not-decoded -->

where the ξ i are i.i.d. Radamacher random variables and in the last two lines it is understood that the supremum is over ∥ w ∥ , ∥ w ∥ , ∥ u ∥ ≤ 1 . Consider the second term in the last expression. Defining η ( x ) = x 1 | x |≤ M + M ( 1 x&gt;M -1 x&lt; -M ) , and η ( x ) = x -η ( x ) , we have

<!-- formula-not-decoded -->

Further defining ϕ ( t 1 , t 2 , t 3 ) := σ ( t 1 ) σ ′ ( t 2 ) η ( t 3 ) (which is CL 2 M -Lipschitz for M ≥ 1 ), we have

<!-- formula-not-decoded -->

Using the contraction inequality of [36], we get

<!-- formula-not-decoded -->

Next consider B 2 , 2 :

<!-- formula-not-decoded -->

where the last inequality holds because u T x i is Gaussian with variance ∥ u ∥ 2 , and using again the contraction inequality. Collecting various terms and optimizing over M ≥ 1 , we obtain

<!-- formula-not-decoded -->

The proof of Eq. (J.7) is completed by bounding B 1 along the same lines.

<!-- formula-not-decoded -->

Consider term D 2 , and define the L 2 -Lipschitz function ψ ( t 1 , t 2 ) := σ ( t 1 ) σ ( t 2 ) ,

<!-- formula-not-decoded -->

Term D 1 is controlled analogously, yielding the proof of Eq. (J.8).

We next prove some continuity properties of the population risk R . It is useful to recall the form:

<!-- formula-not-decoded -->

Lemma J.2. Under the data distribution of Section A, assume ∥ φ ∥ ∞ ≤ L that φ and σ are bounded differentiable with Lipschitz continuous first derivative, ∥ σ ∥ ∞ , ∥ σ ′ ∥ ∞ , ∥ σ ′ ∥ ∞ ≤ L , ∥ φ ∥ ∞ , ∥∇ φ ∥ ∞ , ∥∇ φ ∥ Lip ≤ L , L ≥ 1 . Then, there exists an absolute constant C such that for any ( a , W ) , ( a , ˜ W ) ∈ W m,d ( a ) :

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. As a preliminary remark, the assumptions on φ , σ imply similar smoothness properties of ̂ φ , h . In particular, recall that h ( q ) = E [ σ ( G 1 ) σ ( G q )] for ( G 1 , G q ) jointly Gaussian, centered with unit variance and covariance E [ G 1 , G q ] = 1 , whence its k -th derivative is h ( k ) ( q ) = E [ σ ( k ) ( G 1 ) σ ( k ) ( G q )] (whenever σ ∈ C ( k ) ( R ) ). Therefore, the assumptions on σ imply that ∥ h ′ ∥ ∞ , ∥ h ′ ∥ Lip ≤ L 2 . Similarly, ∥∇ ̂ φ ∥ ∞ , ∥∇ ̂ φ ∥ Lip ≤ CL 2 . Proof of Eq. (J.11) . Differentiating Eq. (J.10)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore

<!-- formula-not-decoded -->

m | a ℓ | ∥ ∥ ∇ w ℓ R ( a , ˜ W ) -∇ w ℓ R ( a , W ) ∥ ∥ ≤ ∥ ∥ ∇ ̂ φ ( U T ˜ w ℓ ) -∇ ̂ φ ( U T w ℓ ) ∥ + m ∑ j =1 | a j | m ∥ ∥ h ′ ( ˜ w T ℓ ˜ w j ) ˜ w j -h ′ ( w T ℓ w j ) w j ∥ ∥ ≤ CL 2 ∥ ˜ w ℓ -w ℓ ∥ + a max j ≤ m ∥ ∥ h ′ ( ˜ w T ℓ ˜ w j ) ˜ w j -h ′ ( w T ℓ w j ) w j ∥ ∥ . Further, by the above smoothness properties of h ,

Substituting above, this yields the claim (J.11).

Proof of Eq. (J.12) . We proceed analogously to the previous point. Namely

<!-- formula-not-decoded -->

whence

<!-- formula-not-decoded -->

m ∣ ∣ ∂ a ℓ R ( a , ˜ W ) -∂ a ℓ R ( a , W ) ∣ ∣ ≤ ∣ ∣ ̂ φ ( U T ˜ w ℓ ) -̂ φ ( U T w ℓ ) ∣ ∣ + m ∑ j =1 | a j | m ∣ ∣ h ( ˜ w T ℓ ˜ w j ) -h ( w T ℓ w j ) ∣ ∣ ≤ CL 2 ∥ ˜ w ℓ -w ℓ ∥ + CaL 2 ( ∥ ˜ w ℓ -w ℓ ∥ + ∥ ˜ w j -w j ∥ ) , which implies immediately Eq. (J.12)

Proof of Eq. (J.13) . Recalling Eq. (J.15), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which proves the desired claim.

Proof of Eq. (J.13) . Recalling Eq. (J.16), we have

<!-- formula-not-decoded -->

Using the last lemma and triangle inequality we get the following.

Corollary J.3. Under the assumptions of Lemma J.2, there exists an absolute constant C such that, for all ( a , W ) , ( a , ˜ W ) ∈ W ∞ m,d ( a ) :

<!-- formula-not-decoded -->

We next consider a ( t ) , W ( t ) that follows GF with respect to the empirical risk, as per Eq. (A.10), which we rewrite as and denote by a 0 ( t ) , W 0 ( t ) the GF with respect to population risk:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma J.4. Under the data distribution of Section A, there exists constant c ∗ = c ∗ ( δ ) , c 0 = c 0 ( δ ) depending uniquely on δ &gt; 0 , and an absolute constant C such that the following holds. Assume φ, σ to be bounded, differentiable with Lipschitz continuous first derivative ∥ φ ∥ ∞ , ∥ φ ′ ∥ ∞ , ∥ φ ′ ∥ Lip ≤ L . ∥ σ ∥ ∞ , ∥ σ ′ ∥ ∞ , ∥ σ ′ ∥ Lip ≤ L , Further assume n/d ≥ exp( c 0 L 2 ) , L ≥ 1 . Let ( a ( t ) , W ( t )) , ( a 0 ( t ) , W 0 ( t )) , be defined as above, with W (0) = W 0 (0) and a (0) = a 0 (0) such that ∥ a (0) ∥ ∞ = ∥ a 0 (0) ∥ ∞ ≤ a 0 . Define

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

Proof. We will prove that the desired bound holds on the high-probability event of Lemma J.1, where by we set a = ( c 1 L -2 log ne/d ) 1 / 3 . Throughout the proof, we use c 0 , c 1 , C to denote constants that might change from line to line, with dependence on the parameters of the problem as per the statement of the lemma.

We start by noting that, letting v i = -m ∇ w i ̂ R n ( a , W ) and v 0 ,i = -( n/d ) ∇ w 0 ,i R ( a 0 , W 0 , and P ⊥ w := I -ww T the projector orthogonal to w .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, comparing the evolution of w i ( t ) and w 0 ,i ( t ) , we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since we are working on the event of Lemma J.1, and using Corollary J.3, we get, for t ≤ T ∗ ( m ; c ) .

Further

<!-- formula-not-decoded -->

Collecting all the terms, and using a ≥ 1 , we get

<!-- formula-not-decoded -->

We next consider the evolution of second-layer weights:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using the last bound together with Eq. (J.21), we get whence the claim follows by Gromwall inequality for sufficiently small c 1 .

We finally need a lemma from [10] approximating GF in the population risk by the mean field dynamics.

Lemma J.5 (Corollary 1 and Proposition 3 [10]) . Let a 0 ( t ) , W 0 ( t ) be GF with respect to the population risk (J.18) with initialization | a 0 ,i (0) | ≤ a 0 and ( w 0 ,i (0)) i ≤ m ∼ Unif ( S d -1 ) . Recall that a mf1 i ( t ) , v mf1 ( t ) is the solution of the ODEs (3.4) with initialization a mf1 i (0) = a 0 ,i (0) , v mf1 i ( t ) = 0 . Under the assumptions of Theorem 3.2, for any ε &gt; 0 there exists constants c 0 , c 1 depending uniquely on L , and an absolute constant C such that letting T lb ( m ) = (( c 0 /ε ) log m ) 1 / 3 , the following happens with probability at least 1 -2 exp( -c 1 d ) ,

<!-- formula-not-decoded -->

Proof of Theorem 3.2. Throughout the proof L, τ, α are assumed to be fixed, and constants C, c 0 , . . . depend on them and can change from line to line. We will further work on the high probability events of Theorem 3.1, Lemma J.4, and Lemma J.5. By Theorem 3.1, for all t ≤ T lb ( m ) we have ∥ a ( t ) ∥ ∞ ≤ c 2 (log 2 m ) 1 / 3 (where the constant c 2 can be made sufficiently small, by eventually reducing c 1 ). An analogous of of Theorem 3.1 for the population risk implies ∥ a 0 ( t ) ∥ ∞ ≤ c 2 (log 2 m ) 1 / 3 as well for all t ≤ T lb ( m ) . Hence we can apply Lemma J.4 and Lemma J.5, which yields the claim.

## K Dynamical mean field theory for non-Gaussian model

The DMFT equations for GF in the original non-Gaussian model can be derived from the general theory of [13].

Given a (positive semi-definite) kernel Q : R ≥ 0 × R ≥ 0 → R m × m , ( t, z ) ↦→ Q ( t, s ) , we write z ∼ GP (0 , Q ) if z is a centered Gaussian process with values in R m and covariance E [ z ( t ) z ( s ) T ] = Q ( t, s ) .

The DMFT equations can be interpreted as a set of fixed point equations for the functions C ij , R ij , a i . We define the deterministic processes a ( t ) , ν i ( t ) and stochastic processes w e ( t ) = ( w e i ( t ) : i ≤ m ) , r ( t ) = ( r i ( t ) : i ≤ m ) , as the solution of

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, in the first equation, ( w e (0) , u ) ∼ N (0 , I m ) ⊗ N (0 , I k ) are independent of η . In the second equation, y = φ ( r 0 ) + ε with ( r 0 , ε ) ∼ N (0 , I k ) ⊗ N (0 , τ 2 ) independent of ξ .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In solving the above, the random functions ∂w e i ( t ) ∂η j ( s ) and ∂r i ( t ) ∂ξ j ( s ) (for t &gt; s ) are defined to be solutions of the following linear ODEs:

<!-- formula-not-decoded -->

with boundary condition ∂w e i ( t ) ∂η j ( t ) = δ ij for the first equation.

## L Derivation of the dynamical mean field theory equations

The study of the dynamics in such high-dimensional limit can be done via dynamical mean field theory (DMFT) [18]. The theoretical technology that we will employ is an evolution of the one first derived in [31, 32] to study gradient flow and stochastic gradient descent on models that are very much related to the Gaussian process we are discussing here [51, 42, 33]. We remark that the formalism considered here can be used to study both the single index model and the pure noise case. To obtain the pure noise model, one can set h t = ˆ φ = 0 . Furthermore, the extension to multi-index models can be also done easily on the same lines.

The analysis of Eqs. (A.10) can be done by recasting them into a path integral representation. We follow the same procedure presented in [31]. Eqs.(A.10) can be packed into a dynamical partition function where the path measure D a ( t ) D ˜ a ( t ) D W D ˆ W is implicitly defined. The action A reads

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Eq. (L.2) can be rewritten by introducing Grassmann variables [62]. Call ˆ a = ( t a , θ a ) a supertime coordinate, with θ a a Grassmann variable. Define, with a slight abuse of notation

<!-- formula-not-decoded -->

Eq. (L.2) can be written as

<!-- formula-not-decoded -->

The first two terms of the sum describe the kinetic terms of the dynamical equations of motion. The last term instead contains the interaction between the weights of the network. The empirical risk ̂ R n depends on the training dataset. We are interested in understanding the behavior of the dynamics of gradient flow when we average over its realizations. Since the dynamical partition function is identically one we can average it directly over the dataset 2 . In this way we have

<!-- formula-not-decoded -->

2 We emphasize anyway that the average over the dataset is not mandatory: the resulting DMFT equations are self-averaging.

Performing standard manipulation, see [31], the dynamical partition function, for d →∞ , can be written as

The dynamical action S dyn is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where α = n/d and

<!-- formula-not-decoded -->

The kinetic kernels K and ˜ K are implicitly defined in such a way that they reproduce the time derivative part of the dynamical equations (A.10).

In the large d limit, fixing m and α , the path integral in Eq. (L.5) concentrates on its saddle point. The corresponding equations are

<!-- formula-not-decoded -->

and where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If Lagrange multipliers are added to constrain the norm of the the weights of the first layer, one should provide additional equations for them. Finally the equations for the dynamics of the second layer weights are given by

Eqs. (L.9)-(L.12) contain all the information about the dynamics. In order to fully specify the behavior of physical quantities such has the train and test error, it is useful to unfold the Grassmann structure of Eqs. (L.9)-(L.12).

<!-- formula-not-decoded -->

## L.1 Unfolding the Grassmann structure

Causality of the dynamics implies that the following parametrization is the most general solution of the saddle point equations

<!-- formula-not-decoded -->

Plugging this parametrization into the saddle point equations we get that the correlators in Eqs. (L.13) satisfy the following DMFT equations

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that we used the notation according to which the prime sign denotes the derivatives of the functions with respect to their argument. The memory kernels M R and M C are defined by

<!-- formula-not-decoded -->

The kernels in Eq. (L.18) depend on R A and C A that are defined in Eqs. (L.13). The corresponding equations are

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

The Lagrange multipliers ν α ( t ) have to be fixed self-consistently to enforce that C α,α ( t, t ) = 1 given that w α ∈ S d -1 . The corresponding equations are

<!-- formula-not-decoded -->

Finally we need to add a set of equation to propagate the diagonal elements of the correlation matrix:

<!-- formula-not-decoded -->

These dynamical equations can be integrated from a set of initial conditions that fully specify the initial status of the neurons. We will consider a random initial condition for the weights of the first layer so that

̸

<!-- formula-not-decoded -->

̸

Finally, the initial conditions for the weights of the last layer a α (0) are completely arbitrary. The solution of the DMFT equations gives access to the dynamics of the train and test error. The train error as a function of time is defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A simple way to derive the expression of e tr as a function of the solution of the DMFT equations in the d →∞ limit is to consider a deformation of Eq. (L.5) which consists in replacing

For P (ˆ a ) = 1 we get back the original expression. The main idea of the derivation is to use P (ˆ a ) as a source field. In particular we have that

∣ Note that the deformed dynamical partition function Z dyn [ P ] does not equal 1 for generic P so that the formula above makes perfectly sense. The deformation of the partition function produces a deformation of S dyn in Eq. (L.7) which consist in replacing

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Performing explicitly the derivatives with respect to P one gets

<!-- formula-not-decoded -->

The computation of the test error can be done in analogous way

<!-- formula-not-decoded -->

The average in Eq. (L.29) is performed over the training set and an additional datapoint, not presented in the training set and having the same statistical structure.

In summary, the solution of the DMFT equations gives access to the train and test error dynamics in the large dimensional limit. These equations can be integrated numerically very efficiently. Our goal is to understand their behavior for infinite number of neurons, m →∞ at fixed sample complexity α . We will be mostly interested in two types of questions: first, given a dataset that is pure noise, what are the sample complexities at which the network is able to interpolate the dataset. Second: given a dataset built out of a single index process what is the dynamics of the test and train error.