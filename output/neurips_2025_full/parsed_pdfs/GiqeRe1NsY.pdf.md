## Understanding Softmax Attention Layers: Exact Mean-Field Analysis on a Toy Problem

Elvis Dohmatob 1 , 2 , 3

1 Concordia University 2 Mila-Quebec AI Institute 3 Meta elvis.dohmatob@concordia.ca

## Abstract

Self-attention has emerged as a fundamental component driving the success of modern transformer architectures which power large language models (ChatGPT, Llama, etc.) and various other types of systems. However, a theoretical understanding of how such models actually work is still under active development. The recent work of (Marion et al., 2025) introduced the so-called "single-location regression" problem, which can provably be solved by a simplified self-attention layer but not by linear models, thereby demonstrating a striking functional separation. A rigorous analysis of self-attention with softmax for this problem is challenging due to the coupled nature of the model. In the present work, we use ideas from the classical random energy model in statistical physics to analyze softmax self-attention on the single-location problem. Our analysis yields exact analytic expressions for the population risk in terms of the overlaps between the learned model parameters and those of an oracle. Moreover, we derive a detailed description of the gradient descent dynamics for these overlaps and prove that, under broad conditions, the dynamics converge to the unique oracle attractor. Our work not only advances the understanding of self-attention but also provides key theoretical ideas that are likely to find use in further analyses of even more complex transformer architectures.

## 1 Introduction

Understanding the theoretical foundations of transformer layers [Bahdanau et al., 2015] (also see [Schmidhuber, 1992]), particularly self-attention (SA) [Vaswani et al., 2017], remains a critical and largely unresolved challenge in machine learning. SA stands as a cornerstone of modern large language models (ChatGPT, Llama, Gemini, Mistral, Deepseek, Cluade, etc.), driving their unprecedented success across diverse tasks. Unlike classical layers such as feedforward, convolutional, or recurrent architectures, SA enables capabilities that these traditional mechanisms cannot replicate, including efficient capture of long-range dependencies and context-aware representations in a single forward pass. While classical layers benefit from decades of rigorous analysis and well-established theory, the inner workings of SA remain poorly understood, with only a sparse body of research attempting to unravel its operational principles. Moreover, the training dynamics of SA, or how it evolves during optimization, present additional complexities that are crucial for developing more robust, efficient, and interpretable models. A deeper theoretical understanding of SA and its learning behavior is therefore essential not only to explain the empirical successes of transformers but also to unlock their full potential and inform future advancements in neural network design.

Recently, Marion et al. [2025] have considered a simplified version of SA and showed that unlike linear models, it can solve a so-called single-location regression problem: the teacher model f ∗ sees an incoming input X = ( x 1 , . . . , x L ) made of the embedding vectors x ℓ in R d for a sequence of L tokens (e.g words), and must correctly locate a secret block x ℓ ∗ at a random secret index ℓ ∗ ∈ [ L ] . Refer to Figure 1. This token index is special in that except for additive Gaussian noise,

Figure 1: The Single-Location Regression Problem. Each x ℓ ∈ R d corresponds a token embedding. The embedding for the secret token at index ℓ ∗ contains signal aligned with a hidden vector u ∗ . The embeddings for tokens at all other indices are pure Gaussian noise. The label y is computed using only this secret token, and all other tokens are ignored. Optionally, we also introduce a link function σ to capture non-linear problems ( σ was taken to be the identity function in [Marion et al., 2025]).

<!-- image -->

̸

it is aligned with an unknown unit-vector u ∗ which can be thought of as encoding the position of ℓ ∗ ; all the other blocks x ℓ = ℓ ∗ are Gaussian noise. Once this block is identified, the model must then approximate the output given by y = f ∗ ( x ) := x ⊤ ℓ ∗ v ∗ , where v ∗ is another unit-vector perpendicular to u ∗ . Thus, presumably, the model must somehow figure out the directions u ∗ and v ∗ in order to solve this problem. This problem captures some aspects of the sparse parity problem, except it is considerably simpler; for example, it does suffer from the the well-known exponential query complexity lower-bound which characterizes the latter problem. Marion et al. [2025] considered a simplified transformer model

<!-- formula-not-decoded -->

for some point-wise activation function θ and inverse temperature parameter λ . The parameters of this student model are a pair of unit-vectors ( u, v ) .

Going beyond [Marion et al., 2025] which considered pointwise/separable SA (1), we consider softmax SA which better reflects what is actually used in transformers. Our student model is then

<!-- formula-not-decoded -->

In our extension, we also include a possibly nonlinear link function σ (known to the learner) used to compute the labels using the the embedding of of a token at a secret index σ ( x ⊤ ℓ ∗ v ∗ ) (instead of x ⊤ ℓ ∗ v ∗ ) for the true labels, and σ ( x ⊤ ℓ v ) (instead of x ⊤ ℓ v as in (1)) for the values in our attention model (2). This link function should not be confused with the softmax layer which is always present in the setting we consider in our work. These are two separate extensions of (1).

Importantly, our theoretical analysis is valid for all L up to a limit which is super-polynomial in the dimension d , i.e., log L = O ( d ) . In contrast, the analysis of Marion et al. [2025] is only limited to sub-linear number of blocks L = o ( d ) .

Main Contributions. Our contributions can be summarized as follows:

- Exact Analytic Formulae for the Risk. In an appropriate asymptotic scaling regime for d and L (refer to (9)), we obtain precise analytic expressions for the population risk of our softmax self-attention model (2) (Proposition 1 and Proposition 2). Our approach uses ideas from the classical analysis of the Random Energy Model (REM) [Derrida, 1981, Lucibello and Mézard, 2024] to handle the softmax, which maps to the Gibbs distribution induced by the disorder in corresponding REM. In order to incorporate the nonlinearity σ , we extend a recent result of [Zavatone-Veth and Pehlevan, 2025]. See Proposition 12, Proposition 13, and their corollaries (Appendix C).
- Optimization Dynamics. We study the optimization landscape of projected gradient-descent on the population risk relative a manifold corresponding to spherical constraints on the model parameters. We classify the stationary points and show that for a large variety of link functions, the induced dynamics always has the optimal model parameters as an attractor (Propositions 8, 10, and 9).
- No Need for Special Initialization. In Proposition 4 we focus on the linear link function σ ( t ) ≡ t and remove a critical initialization assumption which was made in [Marion et al., 2025]. Indeed some

of the main results about optimization in the aforementioned paper assumed that initialization be selected from a peculiar manifold, which effectively assumes some knowledge of the teacher / oracle parameters ( u ∗ , v ∗ ) , which is not feasible a feasible requirement in practice.

## 2 Related Work

The self-attention mechanism, introduced by Vaswani et al. [2017] drives much of modern deep learning, notably in natural language processing and computer vision. By adeptly capturing longrange dependencies, it eclipsed recurrent neural networks in many tasks. Empirical breakthroughs like BERT [Devlin et al., 2019] and Vision Transformers [Dosovitskiy et al., 2021] expanded its reach, sparking theoretical exploration of its mechanics.

Asparse literature of studies have unpacked aspects self-attention's expressive power, complexity, and optimization. Schlag et al. [2021] crafted a minimal attention model without sans positional encoding or normalization, to study its core behavior, while Cui et al. [2024] explored a solvable dot-product attention model, identifying a phase transition between positional and semantic learning driven by positional encoding. Another common simplification of attention, is the so-called linear attention mechanism, where the softmax layer is removed altogether. Such models have been intensively studied in the setting of in-context learning to derive theoretical insights on the internal workings of transformers [Ahn et al., 2023, Von Oswald et al., 2023, Zhang et al., 2024, Lu et al., 2024].

The work which is most related to ours is [Marion et al., 2025] which considers a simplified attention model (1) with the softmax layer replaced by a pointwise function, and study the generalization profile and the optimization landscape induced by such models. In contrast, we consider the more difficult (and practically relevant) case of softmax attention and provide a complete theoretical picture.

Let us note that Dong et al. [2021] showed that except if MLP layers and skip-connections are also used, single-layer SA transformers have a strong inductive bias to converge to rank-1 matrices, a simplicity bias which would limit the applicability to complex problems. Fortunately, the singlelocation problem studied here and in the reference work [Marion et al., 2025] is just simple enough to be captured by a single-layer SA transformer without need of MLP layers or skip-connections.

## 3 Preliminaries

## 3.1 Problem Setup

Data Distribution. Let d and L be positive integers, ϵ &gt; 0 and γ ∈ (0 , 1) be real numbers, and consider the following data distribution: P on [ L ] × R L × d × R given by ( ℓ ∗ , X, y ) ∼ P iff

<!-- formula-not-decoded -->

(Secret Token Features)

x

ℓ

∗

∼ N

̸

(

c

du

∗

, γ

I

d

)

,

with

c

:=

1

-

γ

,

(4)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We will denote the marginal distribution of the features X by P X . Here, u ∗ , v ∗ ∈ S d -1 , where S d -1 is the ( d -1) -dimensional unit-sphere in R d , while σ : R → R is a link function. The unit-vectors u ∗ and v ∗ are fixed but unknown to the learner. The constant γ ∈ (0 , 1) controls the signal-to-noise ratio (SNR) of the problem. As argued in [Marion et al., 2025] in the limit γ → 0 + the dummy features vanish, and the problem reduces to the usual Gaussian linear signal + noise model, which is solvable via simple linear regression with a large enough sample from P . The situation is graphically illustrated in Figure 1.

The Single-Locator Regression Problem and Softmax Self-Attention. As already mentioned in the introduction, the task is to approximate the true label function X ↦→ σ ( x ⊤ ℓ ∗ v ∗ ) . Of course, neither the sample-dependent index ℓ ∗ , nor the unit-vectors u ∗ and v ∗ are known to the learner. For a pair

2

√

2

√

of unit-vectors ( u, v ) ∈ S 2 d -1 , consider the following simplified softmax self-attention (SA) model f introduced in (2). The inverse-temperature λ &gt; 0 controls the sharpness of f ( X ; u, v ) . We will impose the following inverse-temperature scaling λ = β √ d, with fixed β &gt; 0 . Within this class of models, the one with parameters ( u ∗ , v ∗ ) will be referred to as the teacher / oracle model and denoted f ∗ . We shall see in Proposition 2 that this oracle model can indeed approximate the true label function if the feature noise parameter γ is not too large. On the other hand, if the learnable parameter vector u is close to the oracle version u ∗ , then the softmax will concentrate its mass around the right index ℓ ∗ , allowing the model to select the value σ ( x ⊤ ℓ ∗ v ) from all the other values σ ( x ⊤ ℓ v ) . Then, if v is itself close to the oracle version v ∗ , the output of the model f will approximate the true labels σ ( x ⊤ ℓ ∗ v ∗ ) . Thus, goal is to learn the oracle parameter ( u ∗ , v ∗ ) .

We work in the following asymptotic regime where d and L are large but L is exponential in d , i.e

<!-- formula-not-decoded -->

In our theory, taking the limit α → 0 + will correspond to the extreme case where L is at most sub-exponential in d (e.g., polynomial, or even constant as in [Marion et al., 2025]).

Risk / Test-Error. We will be interested in the average L 2 -squared error of the parametrized model f defined in (2), relative to the data distribution P , i.e

<!-- formula-not-decoded -->

which measure how well the model solves the single-location task. The offset ϵ 2 corresponds to the irreducible error of the Bayes model f Bayes : X ↦→ E [ y | X ] , due to the label noise.

Figure 2: Illustrating the optimization dynamics for various for different choices of link function σ . For this experiment, we use input-dimension d = 100 , L = 20 blocks, (normalized) inversetemperature β = 1 , γ = 1 / √ 2 , and label-noise level ϵ = 0 . 1 . The Riemannian gradient-descent scheme is used (29) with step-size s = 0 . 01 . The population risk R is replaced by an empirical version ˆ R = n -1 ∑ n i =1 ( f ( X i ; u, v ) -y i ) 2 , where ( X 1 , y 1 ) , . . . , ( X n , y n ) is an iid sample of size n = 1000 from the data distribution P . The final risk R ( u k , v k ) shown is evaluated on an independent test sample of size 10000 . Broken lines correspond to our theoretical predictions (Proposition 1). Notice the perfect agreement between experiment and our theory. The oscillations in the curves for the ReLU (3rd sub-plot) are reminiscent of the non-smoothness this link function.

<!-- image -->

## 4 A Mean-Field Approximation

## 4.1 Main Idea: Equivalence to the Random Energy Model

Fix the parameters ( u, v ) ∈ S 2 d -1 of a softmax self-attention model f as defined in (2). For a random data point ( ℓ ∗ , X, y ) ∼ P , one can express the output of f as a convex combination like so

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

We have isolated the contribution of the secret token index ℓ ∗ ∈ [ L ] . The other terms (captured in f 2 ) is linked to a d -dimensional random energy model (REM) [Derrida, 1981] with L -1 = e αd configurations with random energy levels E ℓ = √ dx ⊤ ℓ u drawn iid from N (0 , d ) . It turns out that the mean-field description for such a system is completely captured as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Condensation. The value T crit = 1 /β crit ∈ (0 , ∞ ) is the so-called condensation temperature . For temperatures T = 1 /β less than this value (i.e for β &gt; β crit ), the system freezes; only a handful of equilibrium configurations carry the maximum energy, which is of order E := max ℓ = ℓ ∗ E ℓ ≃ ψ √ d . Finally, in the limit α → 0 + in which L is now at most sub-exponentially large in d , the parameters β crit and r vanish and the the system becomes permanently frozen / condensed, for all values of β . Our main results for optimization (Section 5) will focus on this regime.

Remark 1. The above is mean-field description in the sense that it ignores corrections of order 1 /d which could cause statistical fluctuations. Such corrections would require more advanced treatment of the REM, as done in [Bovier and Kurkova, 2004], for example.

## 4.2 Mean-Field Representation of Models and Their Risk

For theoretical analysis, the softmax in the parametrized model class (2) is troublesome because all the blocks are interacting (via the partition function Z ). This is precisely the reason Marion et al. [2025] decided to forgo it pointwise / de-correlated self-attention (1) instead.

Overlaps. It turns out that in the high-dimensional limit (9), (2) admits a simple description in terms of the "thermodynamic" quantities r , ϕ , ψ introduced earlier in (17) and (14), and the overlaps µ, ν, ζ, η, ρ ∈ [ -1 , 1] defined by

<!-- formula-not-decoded -->

In particular, µ and ν capture the alignment between self-attention transformer f given in (2) with parameters ( u, v ) and the oracle parameters ( u ∗ , v ∗ ) .

Now, the µ overlap parameter will enter the picture via the following function p : [ -1 , 1] → [0 , 1] which will play a crucial role in our analysis

<!-- formula-not-decoded -->

This is effectively the high-dimensional limit of the probability that f correctly locates the secret index ℓ ∗ ∈ L is a random data point X = ( x 1 , . . . , x L ) ∼ P X .

Remark 2. Notice a sharp phase-transition: for µ &lt; ψ/c , the probability p ( µ ) is exponentially close to zero (because the input dimension d is large), while for µ &gt; ψ/c , it is exponentially close to 1 .

Auxiliary Functions. Given any ρ, ζ ∈ [ -1 , 1] , define an auxiliary function σ γ,ζ,ρ : R → R by

<!-- formula-not-decoded -->

In particular, we write σ γ ( t ) := σ γ, 0 , 0 ( t ) = σ ( γt ) . The so-called "dual kernel" associated with σ γ , denoted ¯ σ γ : [ -1 , 1] → R , will play a crucial role in our results, and is defined by

<!-- formula-not-decoded -->

where ( G 1 , G 2 ) ∼ N ν , a bi-variate Gaussian with unit variance and correlation coefficient ν . For example, for the linear link function σ ( t ) := t , we get σ γ,ζ,ρ ( t ) ≡ γt +( cζ -ρr ) √ d and ¯ σ γ ( ν ) ≡ γ 2 ν .

We shall also need the auxiliary functions H 1 , H 2 : [ -1 , 1] 3 → R defined by

<!-- formula-not-decoded -->

for ( G 1 , G 2 ) ∼ N ν . Note the implicit dependence of the H k functions on feature noise level γ and the thermodynamic parameter r introduced in (17).

Also define simplified versions h 1 , h 2 : [ -1 , 1] 2 → R of the H k 's corresponding to setting ζ = 0 , i.e

<!-- formula-not-decoded -->

Remark 3. Note that H 2 ( ν, 0 , ρ ) ≡ H 1 ( ν, 0 , ρ ) and H k ( ν, 0) ≡ ¯ σ γ ( ν ) for k = 1 , 2 . Also note that if r = 0 , then H k ( ν, ζ, ρ ) ≡ h k ( ν, ζ ) , and the H k ( ν, ζ, ρ ) doesn't vary with ρ .

Our mean-field analysis will need the following technical condition on the link function σ .

Condition 1. (A) The link function σ is square integrable w.r.t N (0 , 1) , and (B) σ is positivehomogeneous, meaning that there exists m&gt; 0 such that σ ( ut ) = u m σ ( t ) for all u &gt; 0 , t ∈ R .

Examples include: linear link function σ ( t ) := t ; sign link function σ ( t ) := sign( t ) ; ReLU σ ( t ) := max( t, 0) ; quadratic link function σ ( t ) := t 2 ; power link function σ ( t ) := t m (with m&gt; 0) ; etc.

The following is one of our main results.

Proposition 1. Suppose Condition 1 prevails. Then, for any model parameters ( u, v ) ∈ S 2 d -1 and for a random data point ( ℓ ∗ , X ) ∼ P , it holds in the limit (9) that

<!-- formula-not-decoded -->

where p = p ( µ ) is as defined in (20) , and µ , ν , η , ζ , and ρ are as defined in (19) .

Furthermore, the population risk of the model is given by R ( u, v ) ≃ ¯ R ( µ, ν, ζ, ρ ) , where

<!-- formula-not-decoded -->

Proof Sketch. The backbone of the proof (provided in the Corollary 3 of the Appendix) uses ideas from the classical analysis of the REM [Derrida, 1981, Lucibello and Mézard, 2024] to establish (25). In particular, we extend a recent result of Zavatone-Veth and Pehlevan [2025] for our purposes.

Once formula (25) is established, the formula for the risk is a matter of Gaussian integration.

In order to apply Proposition 1, one needs to compute the auxiliary functions H 1 and H 2 defined in (23). This is done in Proposition 11 of Appendix B. The said formulae can then be readily exploited to get explicit expressions for the surrogate risk ¯ R appearing in Proposition 1.

Linear Link Function. Consider the special case of the identity link function σ ( t ) := t . In this case, thanks again to Proposition 11, we know that H k functions appearing in Proposition 1 are:

<!-- formula-not-decoded -->

with a := cζ √ d and b := ρr √ d . We obtain the following corollary.

Corollary 1. Under the conditions of Proposition 1, and in the limit (9) , it holds that f ( X ; u, v ) ≃ px ⊤ ℓ ∗ v + (1 -p ) ρr √ d . Moreover, the population risk is given by R ( u, v ) ≃ ¯ R ( µ, ν, ζ, ρ ) , where ¯ R ( µ, ν, ζ, ρ ) = ( pcζ +(1 -p ) ρr ) 2 d + γ 2 ( p 2 -2 pν +1) .

A Geometric Insight from Corollary 1. For any fixed u ∈ S d -1 , the restriction of the population risk v ↦→ R ( u, v ) on the set { v ∈ S d -1 | u ⊤ v = 0 } behaves like a quadratic well R ≃ ( pv ⊤ u ∗ ) 2 d + γ 2 ∥ pv -v ∗ ∥ 2 , with deepest point v ( u ) given by

<!-- formula-not-decoded -->

In the above calculation, we have used the Sherman-Morrison formula and the fact that u ⊤ ∗ v ∗ = 0 . Further, if cu ⊤ u ∗ ≥ (1 + Ω(1)) ψ (i.e u is within a spherical cap around u ∗ ), then

<!-- formula-not-decoded -->

and so v ( u ) = v ∗ /p ≃ v ∗ . It is then easy to see that any such ( u, v ( u )) minimizes the population risk R . Thus, we only need to get an Ω(1) alignment of the u parameter with the oracle counterpart u ∗ .

## 4.3 Bayes-optimality of the Oracle Model: A Sharp Phase-Transition

To be sure we are actually in business, we must ensure that the best possible choice of the parameters ( u, v ) ∈ S d -1 for the parametrized family (2), namely the oracle parameters ( u ∗ , v ∗ ) , does indeed achieve zero risk R = 0 . As the next result shows, it turns out that Condition 2 is a necessary and sufficient condition for this purpose.

Proposition 2. The oracle parameter ( u ∗ , v ∗ ) are indeed optimal for the risk functional R restricted to the parametrized family (2) , i.e R ( u ∗ , v ∗ ) = inf ( u,v ) ∈ S 2 d -1 R ( u, v ) .

Moreover, we have the following sharp phase-transition (recall that c := √ 1 -γ 2 ).

- (A) If c ≥ (1 + δ ) ψ for some δ ∈ (0 , 1) , then in the limit (9) , it holds that R ( u ∗ , v ∗ ) → 0 . That is, the oracle model with parameters ( u ∗ , v ∗ ) is Bayes-optimal.
- (B) If c ≤ (1 -δ ) ψ , then in the limit (9) , it holds that R ( u ∗ , v ∗ ) = Ω(1) , more precisely, R ( u ∗ , v ∗ ) → ¯ σ γ (1) &gt; 0 . That is, learning is not possible!

Proof. Indeed, by definition ( u ∗ , v ∗ ) is optimal for the population risk functional R . Moreover, thanks to Proposition 1, we have R ( u ∗ , v ∗ ) ≃ ¯ R (1 , 1 , 0 , 0) , with

<!-- formula-not-decoded -->

where p = p (1) := 1 / (1 + e -( c -ψ ) βd ) . Finally, since ¯ σ γ (1) &gt; 0 , observe that RHS in the above display vanishes for large d iff c &gt; ψ , and the result follows.

Motivated by the above result, we shall need the following condition in the sequel.

Condition 2 (Realizability) . c ≥ (1 + Ω(1)) ψ , i.e c ≥ ψ (1 + δ ) for some δ &gt; 0 , where ψ is as defined in (14) and we recall that c := √ 1 -γ 2 . For most of our analysis, we can work under the weaker condition c &gt; ψ .

For example, the condition is always satisfied for α → 0 + corresponding to sequence length L which is sub-exponential (e.g., polynomial) in the dimension d , because we have ψ → 0 + in this case and so ψ ≤ (1 -δ ) c trivially, for any δ &gt; 0 and sufficiently large d .

## 5 Learning and Optimization

From Proposition 2, we know that under Condition 2, the oracle model parameters ( u ∗ , v ∗ ) solve the single-locator problem. But, can numerical optimization actually find it? As mentioned in the introduction, unlike the case of pointwise attention considered in [Marion et al., 2025], the analysis is complicated by the softmax (2) which characterizes genuine attention layers [Vaswani et al., 2017] used in practice.

## 5.1 Preliminaries

Projected Gradient-Descent/Gradient-Flow. For simplicity of analysis, we shall consider a learner who has infinite samples, and therefore can directly optimize the population risk R over the parametrized family (2). We shall study the dynamics of the following projected gradient descent (PGD) scheme

<!-- formula-not-decoded -->

where s &gt; 0 is the step-size and P ⊥ u := I d -uu ⊤ is the orthogonal projector onto the tangent space to unit-sphere S d -1 at the point u . The projection step on the second line ensures that the iterates ( u k +1 , v k +1 ) remain on the the manifold S 2 d -1 . For sufficiently small step size s , the dynamics (29) are captured by projected gradient-flow (PGF) on the manifold S 2 d -1 given by

<!-- formula-not-decoded -->

The dynamics (29) and (30) induce a corresponding evolution equation for the order parameters ( µ, ν, η, ζ, ρ ) which will be our main object of study.

Some of our results will concern the following submanifold M⊆ S 2 d -1

<!-- formula-not-decoded -->

introduced by [Marion et al., 2025]. It is clear that M contains the oracle parameters ( u ∗ , v ∗ ) . On this submanifold the dynamics reduce to just µ and ν , a two-dimensional system.

Condition 3 (Sub-exponential block length) . In this section, we shall shall work in the frozen regime r = 0 , corresponding to the case where β crit → 0 + , that is the number of blocks L is sub-exponential in the input-dimension d , i.e log L = o ( d ) . This condition is very mild, and subsumes the regime L = o ( d ) considered in [Marion et al., 2025] as a special case.

Since ρ := u ⊤ v only enters the picture via the product ρr , in this regime the effect ρ is permanently lost; this variable disappears from the picture under the above condition.

## 5.2 Analysis of Optimization Dynamics: Arbitrary Initialization

Proposition 3. For any u, v ∈ S d -1 , define µ := u ⊤ u ∗ , ν := v ⊤ v ∗ , η := u ⊤ v ∗ , ζ := v ⊤ u ∗ , ρ := u ⊤ v . If ρ = 0 , then in the limit (9) , we have the following:

<!-- formula-not-decoded -->

uniformly on w ∈ S d -1 , where the functions T 1 , T 2 , T 2 : [ -1 , 1] 3 → R are as given in Appendix E.1.

Thus, (asymptotically) the gradients of the risk R are trapped in the 2-dimensional subspace of R d spanned by the oracle parameters ( u ∗ , v ∗ ) .

Corollary 2. For sufficiently small step-size in the iteration scheme (29) , the equations of motion for overlaps ( µ, ν, η, ζ ) are given by the following 4-dimensional gradient-flow:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Stronger Results in the Case of Linear Link Function. Even in the greatly simplified case of point-wise / non-softmax attention (1) considered in the reference work [Marion et al., 2025], their analysis was further restricted to the setting where the optimization scheme (29) is initialized on the sub-manifold M . This is problematic because by definition, choosing a point on M presupposes knowledge of the oracle parameter ( u ∗ , v ∗ ) . Our next result closes this gap and shows that this sub-manifold indeed contains all the stationary points, and is therefore eventually attained by the dynamics (29), irrespective of the initialization.

Proposition 4. For the linear link function σ ( t ) := t , the sub-manifold M contains all the stationary points of the 4-dimensional dynamics (30) . In fact, the only stationary points are ( ± 1 , ± 1 , 0 , 0) , of which (1 , 1 , 0 , 0) (corresponding to the oracle model parameters ( u ∗ , v ∗ ) ) is the only stable one (more precisely, it is an attractor/sink); the others are saddles and sources.

## 5.3 Dynamics on the Submanifold M

We now consider the dynamics (29) in the regime where it has entered the submanifold M defined in (31) (e.g., via initialization), and show a drastic simplification of the picture.

Proposition 5. The 4-dimensional dynamics (29) fixes the submanifold M . That is, once the dynamics (29) enters M , it remains there.

Proposition 6. In the limit of vanishing step-size, the equations of motion of the overlap variables µ and ν , induced (30) are given by the following 2-dimensional gradient-flow:

<!-- formula-not-decoded -->

where the scalar fields F 0 1 , F 0 2 : [ -1 , 1] 2 → R are as defined in (71) and (72) respectively.

Stationary Points. Let E = { ( µ, ν ) ⊆ [ -1 , 1] 2 | F 0 1 ( µ, ν ) = F 0 2 ( µ, ν ) = 0 } be the stationary points of the 2-dimensional dynamics dynamics (35). Consider the following subsets of [ -1 , 1] 2

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition 7. E = E 1 ∪ E 2 ∪ E 3 ∪ E 4 is a partitioning of the set of stationary points of the 2-dimensional dynamics (35) .

Proof. The E of stationary points correspond to pairs ( µ, ν ) for which F 0 1 ( µ, ν ) = F 0 2 ( µ, ν ) = 0 . By definition this translates to

<!-- formula-not-decoded -->

But A ( µ ) := p ( µ )(1 -p ( µ )) cβd and p ( µ ) are always positive (since p ( µ ) ∈ (0 , 1) ) and so they can be canceled out in the above equations to give

<!-- formula-not-decoded -->

By manipulating the equations, it is easy to see that the all the solutions can be organized into the sets E 1 , E 2 , E 3 , and E 4 respectively. Each E k vanishes exactly one of the two factors in either of the two equations above (four possibilities in total).

A complete classification of the stationary points is provided in Appendix A.

## 5.4 Convergence to A Stationary Point

Our analysis would be incomplete without showing that the PGD dynamics (29) actually converges to a stationary point. The following result is proved in Appendix G.

Proposition 8. Under some smoothness conditions on the link function σ (made explicit in the appendix), the following holds: If ( u 0 , v 0 ) ∈ M , then the PGD dynamics (29) converges to a stationary point of the population risk functional R .

Taken together with with Proposition 4, Proposition 5, Proposition 7, alongside the results in Appendix A on the classification of stationary points, we infer the following:

- In the case of the identity link functions σ ( t ) ≡ t , (projected) gradient descent on the population risk function R converges to the group truth model parameters ( u ∗ , v ∗ ) , irrespective of the initialization ( u 0 , v 0 ) . This result is much stronger that [Marion et al., 2025] which required ( u 0 , v 0 ) ∈ M , even though the latter considered the much simpler case of pointwise self-attention (1).
- In the general case, we have the same convergence as above, provided the initialization ( u 0 , v 0 ) is on the manifold M .

## 6 Concluding Remarks

We present an end-to-end theories of softmax self-attention and an interesting task, the single-locator regression problem proposed in [Marion et al., 2025]. Building on the pointwise/non-softmax analysis of Marion et al. [2025], we give closed-form formulas for the population risk and gradient flow, pinpointing why attention solves tasks that defeat linear models. Our proof unites tools from statistical physics, mean-field theory, and Riemannian optimization, while eliminating the delicate initialization assumptions of previous work. Thus a ubiquitous yet opaque layer becomes one whose behavior can now be predicted, analyzed, and engineered.

Limitations and Future Directions. A nature next step would be to extend our proposed theory to (1) empirical risk minimization (i.e finite-sample analysis) and (2) multi-locator regression problem, where instead of a single secret token index [Marion et al., 2025], several secret indices must be recovered. The latter would encompass tasks such as the well-known sparse parity problem, opening the door to richer combinatorial analyses of attention. With some extra effort, the core ideas technical ideas developed in our work would directly extend to the this setting.

## References

- Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings , 2015.
- Anton Bovier and Irina Kurkova. Derrida's generalised random energy models 1 : models with finitely many hierarchies. Annales de l'I.H.P. Probabilités et statistiques , 40(4), 2004.
- Hugo Cui, Jaron Kent-Dobias, Florent Krzakala, and Lenka Zdeborová. A phase transition between positional and semantic learning in a solvable model of dot-product attention. Physical Review Letters , 132(13), 2024.
- Bernard Derrida. Random-energy model: An exactly solvable model of disordered systems. Phys. Rev. B , 24, 1981.
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Association for Computational Linguistics . Association for Computational Linguistics, 2019.
- Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: pure attention loses rank doubly exponentially with depth. In Proceedings of the 38th International Conference on Machine Learning , volume 139 of Proceedings of Machine Learning Research , pages 2793-2803. PMLR, 2021.
- Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations , 2021.
- Yue M. Lu, Mary I. Letey, Jacob A. Zavatone-Veth, Anindita Maiti, and Cengiz Pehlevan. Asymptotic theory of in-context learning by linear attention. arXiv e-prints , 2024.
- Carlo Lucibello and Marc Mézard. Exponential capacity of dense associative memories. Phys. Rev. Lett. , 132, Feb 2024.
- Pierre Marion, Raphaël Berthier, Gérard Biau, and Claire Boyer. Attention layers provably solve single-location regression. In The Thirteenth International Conference on Learning Representations , 2025.
- Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning , volume 139 of Proceedings of Machine Learning Research . PMLR, 2021.
- Jürgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation , 4(1), 1992.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 30. Curran Associates, Inc., 2017.
- Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, Joao Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In Proceedings of the 40th International Conference on Machine Learning , volume 202 of Proceedings of Machine Learning Research . PMLR, 2023.
- Jacob A Zavatone-Veth and Cengiz Pehlevan. Nadaraya-watson kernel smoothing as a random energy model. Journal of Statistical Mechanics: Theory and Experiment , 2025, jan 2025.
- Ruiqi Zhang, Spencer Frei, and Peter L. Bartlett. Trained transformers learn linear models in-context. Journal of Machine Learning Research , 25(49), 2024.

## Contents

| A   | Classification of Stationary Points: Sinks, Sources, Saddles    | Classification of Stationary Points: Sinks, Sources, Saddles                                                                                                                |   11 |
|-----|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| B   | Misc: Additional Theoretical Results                            | Misc: Additional Theoretical Results                                                                                                                                        |   12 |
| C   | Relevant Statistical Physics for Machine Learning               | Relevant Statistical Physics for Machine Learning                                                                                                                           |   12 |
|     | C.1                                                             | High-dimensional Analysis of Abstract Nadaraya-Watson Estimator (Local Learning)                                                                                            |   12 |
|     | C.2                                                             | A Novel Extension of the Nadaraya-Watson Estimator . . . . . . . . . . . . . . . .                                                                                          |   14 |
| D   | High-Dimensional Representation of Estimator and Its Risk       | High-Dimensional Representation of Estimator and Its Risk                                                                                                                   |   16 |
|     | D.1                                                             | Case of Linear Link Function: Proof of Corollary 1 . . . . . . . . . . . . . . . . .                                                                                        |   16 |
|     | D.2                                                             | Proof of Proposition 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                    |   16 |
| E   | Equations of Motion                                             | Equations of Motion                                                                                                                                                         |   16 |
|     | E.1                                                             | Auxiliary Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                     |   16 |
|     | E.2                                                             | Proof of Proposition 3 (High-Dimensional Representation of Gradient of Population Risk) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   17 |
|     | E.3                                                             | Proof of Corollary 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                    |   19 |
| F   | Proofs of Classification Theorems                               | Proofs of Classification Theorems                                                                                                                                           |   19 |
|     | F.1                                                             | Jacobian Matrices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                     |   19 |
|     | F.2                                                             | Proof of Proposition 9 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                    |   20 |
|     | F.3                                                             | Proof of Proposition 10 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                     |   20 |
|     | F.4                                                             | Proof of Proposition 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                    |   22 |
| G   | Proof of Proposition 8 (Convergence of PGD to Stationary Point) | Proof of Proposition 8 (Convergence of PGD to Stationary Point)                                                                                                             |   23 |
|     | G.1                                                             | Step 1: A Descent Lemma . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .                                                                                         |   23 |
|     | G.2                                                             | Step 2: Convergence to Stationary Point . . . . . . . . . . . . . . . . . . . . . . .                                                                                       |   25 |

## A Classification of Stationary Points: Sinks, Sources, Saddles

We start with the following general result which shows that oracle model with parameters ( u ∗ , v ∗ ) corresponding to µ = ν = 1 is a stable stable stationary point of the dynamics (35), even in the case of nonlinear link function σ .

Proposition 9. The point (1 , 1) (corresponding to the oracle model parameters ( u ∗ , v ∗ ) ) is always an attractor/sink of the 2-dimensional dynamics (35) .

As with all the other results in this section, the proof of the above result is provided in Section F.

## Appendix

We now give a complete classification of the stationary points E = ∪ k E k for various choices of the link function σ .

Proposition 10. We have the following classification of the stationary points of the 2-dimensional dynamics (35) induced by different choice of the link function σ .

(A) Linear link function. For σ ( t ) := t , the stationary points are: ( ± 1 , ± 1) , of which (1 , 1) is a sink (stable), (1 , -1) is a source (unstable), and ( -1 , ± 1) are saddles (unstable).

(B) Quadratic link function. For σ ( t ) := t 2 , the stationary points are: ( ± 1 , ± 1) , ( ± 1 , 0) , and ( ψ, 0) , where ψ is the thermodynamic parameter introduced in (14) . Moreover, (1 , ± 1) are sinks (stable); ( -1 , ± 1) , ( -1 , 0) , and ( ψ, 0) are saddles (unstable); (1 , 0) is a source (unstable).

Note that because of the evenness of this link function, the stable stationary points (1 , ± 1) both correspond to the oracle parameters ( u ∗ , v ∗ ) .

(C) ReLU link function. Consider the link function σ ( t ) := max( t, 0) , and suppose γ ∈ [1 / 2 , 1) . Then, the dynamics has 4 stationary points: ( ± 1 , ± 1) , of which (1 , 1) is a sink (stable), (1 , -1) is a saddle (unstable), and ( ± 1 , -1) are degenerate.

Thus, for all these link functions, the iteration scheme (29) with sufficiently small step-size is guaranteed to converge to the oracle parameters ( u ∗ , v ∗ ) .

## B Misc: Additional Theoretical Results

Proposition 11. For any ν, ζ, ρ ∈ [ -1 , 1] , set a := cζ √ d , h = a/γ , b := ρr √ d , b 0 := max( b, 0) , and q := a Φ( h ) + γφ ( h ) . For different choices of the link function σ , the function dual function ¯ σ γ , and the H k 's in Proposition 1 have the following closed form expressions for any ν, ζ, ρ ∈ [ -1 , 1] .

- (A) Linear link function. If σ ( t ) := t , then ¯ σ γ ( ν ) = γ 2 ν , and

<!-- formula-not-decoded -->

(B) Quadratic link function. If σ ( t ) := t 2 , then ¯ σ γ ( ν ) = γ 4 (1 + 2 ν 2 ) , and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(C) ReLU link function. If σ ( t ) := max( t, 0) , then ¯ σ γ ( ν ) = ( √ 1 -ν 2 + ν arccos( -ν )) γ 2 / (2 π ) , the well-known "arc-cosine" kernel, and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where φ is the standard Gaussian pdf, Φ is the corresponding cdf, and Φ 2 ( · , · ; ν ) is the cdf of the standard bi-variate Gaussian with unit variance and correlation coefficient ν .

The proof for the case linear and quadratic link functions reveals that the results can be readily generalized to general powers by making use of hyper-geometric functions of type 2 F 1 .

## C Relevant Statistical Physics for Machine Learning

## C.1 High-dimensional Analysis of Abstract Nadaraya-Watson Estimator (Local Learning)

Fix unit-vectors u, v ∈ R N . Consider a sizen random energy model (REM) [Derrida, 1981] in N dimensions, with energy levels E 1 , . . . , E n , where E i := √ Nx ⊤ i u , i.e independent energy levels from N (0 , N ) , and consider the sum

<!-- formula-not-decoded -->

for some link function σ ∈ L 2 ( N (0 , 1)) , which doesn't depend on N . We seek a deterministic equivalent in the limit

<!-- formula-not-decoded -->

The following result is an adaptation of the main result of Zavatone-Veth and Pehlevan [2025] to the case of Gaussian covariates.

Proposition 12. In the limit (44) , it holds a.s that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We use Laplace method of integration coupled with the recipe of Lucibello and Mézard [2024], Zavatone-Veth and Pehlevan [2025]. Thus, we write

<!-- formula-not-decoded -->

where we have introduced overlaps

<!-- formula-not-decoded -->

The potential ϕ has an energetic part βt and an entropic part α -s ( t, q ) . In order to determine the function s , we use large-deviation methods. Consider the random bi-variate Gaussian random vector z = ( x ⊤ i u, x ⊤ i v ) , with covariance matrix N Σ , where

<!-- formula-not-decoded -->

The (normalized) log-MGF of w is given by

<!-- formula-not-decoded -->

We take s to be the Legendre transform of ζ , i.e

<!-- formula-not-decoded -->

for any c = ( t, q ) . The condensation region the corresponds to s ( c ) ≥ α , i.e c ⊤ Σ c/ 2 ≥ α , which is an ellipsoid in c -space. Consider the re-parametrization in polar coordinates

<!-- formula-not-decoded -->

That is, q = r sin θ and t = r √ 1 -ρ 2 cos θ + ρr sin θ . This gives s ( r, θ ) = r 2 / 2 , and our potential takes the form

<!-- formula-not-decoded -->

and we must now maximize w.r.t ( r, θ ) outside the condensation region. Maximizing Φ( r, θ ) w.r.t θ gives

<!-- formula-not-decoded -->

This in turn gives cos θ = √ 1 -ρ 2 and sin θ = ρ . Plugging into ϕ gives

<!-- formula-not-decoded -->

Moreover, the condensation region is now given by α ≤ s ( r, θ ) = r 2 / 2 , i.e r ≥ √ 2 α . Outside of this region (i.e for r &lt; √ 2 α ), maximizing Φ w.r.t r gives r = β .

Putting things together, we get the following final form for our potential

<!-- formula-not-decoded -->

It is a standard result for the REM that ( βN ) -1 log Z → ψ = ϕ/β . Finally, plugging everything into the RHS of (48) then gives g ≃ ( e ϕN /Z ) σ ( ρr ) ≃ ( e ϕN /e ϕN ) σ ( ρr ) = σ ( ρr ) , as claimed.

For our purposes, we would like to compute sums of the form g = ∑ i p i σ ( x ⊤ i v ) and not ∑ i p i σ ( x ⊤ i v/ √ N ) as in Proposition 12. To address this, we can't simply consider a new function h N ( t ) := σ ( t √ N ) to write f = ∑ i h N ( x ⊤ i v/ √ N ) and then apply Proposition 12. The issues is that h N is not fixed in L 2 ( N (0 , 1)) but itself varies with the dimension N which is tending to infinity. In the special case when σ is positive-homogeneous, we can effectively factor out this dimension-dependence and correctly apply Proposition 12. Viz,

Corollary 3. Let ( p i ) i be the Gibbs distribution from Proposition 12. If σ ∈ L 2 ( N (0 , 1)) is positively-homogeneous, then in the limit (44) , it holds that

<!-- formula-not-decoded -->

Proof. Indeed, positive-homogeneity means that there exists m &gt; 0 such that σ ( ut ) = u m σ ( t ) for all u &gt; 0 and t ∈ R . In particular, we have σ ( x ⊤ i v/ √ N ) = σ ( x ⊤ i v ) /N m/ 2 , i.e. σ ( x ⊤ i v ) = N m/ 2 σ ( x ⊤ i v/ √ N ) . The result then follows from Proposition 12:

<!-- formula-not-decoded -->

## C.2 A Novel Extension of the Nadaraya-Watson Estimator

We now extend Proposition to the case of multivariate link functions. A slight modification of our arguments from Gaussian to spherical data immediately gives a non-trivial extension of the main result in [Zavatone-Veth and Pehlevan, 2025].

Let u, v 1 , . . . , v k ∈ R N be unit-vectors, and let ρ j := u ⊤ v j be the cosine of the angle that u makes with each v j . Let x 1 , . . . , x n be iid from N (0 , I N ) , and consider the following generalized NW estimator

<!-- formula-not-decoded -->

where the F is an L 2 ( N (0 , I k )) function and k ≥ 1 is a fixed integer. We seek a deterministic equivalent for g in the limit (44).

Proposition 13. In the limit (44) , it holds that g → F ( ρr 1 , . . . , ρr k ) a.s.

Proof. Once again, we will follow the line of thought of Lucibello and Mézard [2024]. The Laplace method let's us write:

<!-- formula-not-decoded -->

with the definition of overlaps

<!-- formula-not-decoded -->

Now, we need to compute the log-MGF of the Gaussian random vector z := ( x ⊤ i u, x ⊤ i v 1 , . . . , x ⊤ i v k ) . Its covariance matrix is N Σ , where Σ ∈ R ( k +1) × ( k +1) is the covariance matrix of z/ √ N , given by

<!-- formula-not-decoded -->

We thus compute the normalized log-MGF of z as

<!-- formula-not-decoded -->

The Legendre transform of ζ is of course given by

<!-- formula-not-decoded -->

Recall that the condensation region in c -space is then the exterior of the set by s ( c ) ≤ α , i.e the ellipsoid E ⊆ R k +1 given by

<!-- formula-not-decoded -->

Now, for any fixed value r 2 / 2 of s ( c ) , the sought-for potential ϕ has an energetic part βt and an entropic part α -s ( c ) = α -r 2 / 2 , i.e has the form

<!-- formula-not-decoded -->

We now maximize ϕ subject to the constraint s ( c ) = r 2 / 2 , i.e c ⊤ Σ -1 c = r 2 . This is a linear maximization problem (since t = c ⊤ e 1 ) with quadratic constraint. The method of Lagrange multipliers gives c ∝ Σ e 1 , i.e c is proportional to the first row of Σ which is the vector (1 , ρ 1 , . . . , ρ k ) . Plugging the constraint c ⊤ Σ -1 c = r 2 gives c = r Σ e 1 = ( r, ρr 1 , . . . , ρr k ) . The potential then takes the form

<!-- formula-not-decoded -->

The condensation region E ′ is then r ≥ √ 2 α =: β crit . In this region, maximizing ϕ w.r.t r gives r = β . Thus, we must have r = min( β, β crit ) . Combining with (55), we deduce that

<!-- formula-not-decoded -->

as claimed.

We have the following generalization of Corollary 4.

Corollary 4. Let F be as in Proposition 13. If in addition F is positive-homogeneous w.r.t to each input (the orders of the homogeneity are allowed to be different), then in the limit (44) , it holds a.s that

<!-- formula-not-decoded -->

Recall that F being positive-homogeneity of order ( m 1 , . . . , m k ) ∈ R k + means that

<!-- formula-not-decoded -->

for any t 1 , . . . , t k ∈ R and u 1 , . . . , u k &gt; 0 .

Proof of Corollary 4. Indeed, taking u j ≡ 1 / √ N and t j ≡ x ⊤ i v j gives

<!-- formula-not-decoded -->

Thanks to Proposition 13, we know that RHS ≃ F ( rρ 1 , . . . , rρ k ) . We deduce that

<!-- formula-not-decoded -->

as claimed.

## D High-Dimensional Representation of Estimator and Its Risk

## D.1 Case of Linear Link Function: Proof of Corollary 1

̸

We start with an instructive self-contained proof of Corollary 1. Following the decomposition (11), we only need to estimate f 2 = ∑ ℓ = ℓ ∗ q ℓ x ⊤ ℓ v , where q ℓ := e βE ℓ /Z -ℓ ∗ , E ℓ := √ dx ⊤ ℓ u , Z := ∑ ℓ = ℓ ∗ e βE ℓ . We can decompose x ⊤ ℓ v = ρx ⊤ ℓ u + √ 1 -ρ 2 z ℓ , where the z ℓ 's are N (0 , 1) , and independent of Xu . This gives

̸

<!-- formula-not-decoded -->

̸

Conditioned on Xu , the second sum is a centered Gaussian distribution with variance equal to (1 -ρ 2 ) ∑ ℓ = ℓ ∗ q 2 ℓ ≤ 1 -ρ 2 ≤ 1 . For the first sum, observe that

̸

̸

<!-- formula-not-decoded -->

̸

and the result follows. The step " → " is a classical result in the analysis of the REM, while the last step ' ∂ β ϕ = r follows from the definition of ϕ and r in (14) and (17). We deduce that f 2 ≃ ρr √ d , and so f ( X ; u, v ) ≃ px ⊤ ℓ ∗ v + (1 -p ) ρr √ d as claimed. The claimed formula for the population risk R ( u, v ) is then obtained by plugging the previous formula for f into definition (10), and then computing basic Gaussian integrals.

## D.2 Proof of Proposition 1

̸

Indeed, following the decomposition (11), we know that f ≃ pσ ( x ⊤ ℓ ∗ v )+(1 -p ) f 2 , and we only need to estimate f 2 = ∑ ℓ = ℓ ∗ q ℓ σ ( x ⊤ ℓ v ) , where q ℓ := e βE ℓ /Z -ℓ ∗ , E ℓ := √ dx ⊤ ℓ u , Z := ∑ ℓ = ℓ ∗ e βE ℓ , and p = p ( u ⊤ u ∗ ) is as defined in formula (20). Corollary 3 with N = d and n = L -1 gives

We deduce that f ≃ pσ ( x ⊤ ℓ ∗ v ) + (1 -p ) σ ( ρr √ d ) as claimed.

<!-- formula-not-decoded -->

The claimed formula for the risk R ( u, v ) is then a matter of direct Gaussian integration.

## E Equations of Motion

## E.1 Auxiliary Functions

Define auxiliary functions A : [ -1 , 1] → R , and B,T 1 , T 2 , T 3 , F 1 , F 2 , F 4 : [ -1 , 1] 3 → R , and F 3 : [ -1 , 1] 4 → R by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where h k ( ν, ζ ) := H k ( ν, ζ, 0) , and the H k are as defined in (23). In particular, when η = ζ = 0 (as on the sub-manifold M ), the above formulae drastically reduce to

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

̸

## E.2 Proof of Proposition 3 (High-Dimensional Representation of Gradient of Population Risk)

Recall R ( u, v ) = E [ δ ( X ; u, v ) 2 ] , where δ ( X ; u, v ) := f ( X ; u, v ) -σ ( x ⊤ ℓ ∗ v ∗ ) . Differentiating w.r.t u and v gives

<!-- formula-not-decoded -->

We already know that

<!-- formula-not-decoded -->

We now need to estimate ∇ u f ( X ; u, v ) and ∇ v f ( X ; u, v ) . Setting λ := β √ d , one computes

̸

<!-- formula-not-decoded -->

̸

where p = p ( µ ) := 1 / (1 + e -( cµ -ψ ) βd ) and µ := u ⊤ u ∗ as usual.

Now, for any w ∈ S d -1 , Corollary 3 and Corollary 4 (with N = d and n = L -1 ) give

<!-- formula-not-decoded -->

δ

≃

p

(

σ

(

x

⊤

ℓ

∗

v

)

-

σ

(

ρr

d

))

-

(

σ

(

x

⊤

ℓ

∗

v

∗

)

-

σ

(

ρr

d

))

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

From the above display, we deduce that

<!-- formula-not-decoded -->

̸

Likewise, we have

<!-- formula-not-decoded -->

Putting things together, we have shown that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Gradient w.r.t u Parameter. Proceeding from (81) and (82), we compute

<!-- formula-not-decoded -->

Since r = 0 by assumption, the Remark 3 tells us that all the H k functiosn no longer depend on ρ , and we get

<!-- formula-not-decoded -->

where B ( µ, ν, ζ ) := 2( p ( µ ) h 1 (1 , ζ ) -h 2 ( ν, ζ )) as usual. Recalling the definition

<!-- formula-not-decoded -->

we can write the above as

<!-- formula-not-decoded -->

This proves that ∇ u R ( u, v ) ≃ T 1 ( µ, ν, ζ ) u ∗ as claimed.

Gradient w.r.t v Parameter. Using (80) and (82), one computes

<!-- formula-not-decoded -->

Now, using the well know formulae g ∇ g = (1 / 2) ∇ g 2 and g ∇ h = ∇ gh -h ∇ g , we have

<!-- formula-not-decoded -->

Observe that ∇ v σ γ, 0 ,ρ ( z ⊤ v ∗ ) = -∇ v σ ( ρr √ d ) , and so

<!-- formula-not-decoded -->

Putting everything (once again under the assumption that r = 0 ), we get

<!-- formula-not-decoded -->

We conclude that ∇ v R ( u, v ) ≃ T 3 ( µ, ν, ζ ) u ∗ + T 2 ( µ, ν, ζ ) v ∗ as claimed.

## E.3 Proof of Corollary 2

From (30), the dynamics for ( µ, ν, η, ζ is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

On the other hand, one computes

<!-- formula-not-decoded -->

## F Proofs of Classification Theorems

For ease of notation, we shall use the following shorthand

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F.1 Jacobian Matrices

Classical theory of dynamical systems tells us that the classification of the different stationary points ( µ, ν ) ∈ E can be done by studying the signs of the real parts of the eigenvalues of the Jacobian matrices

<!-- formula-not-decoded -->

One can use these Jacobian matrices to classify the stationary points as sources ( J ( µ, ν ) only has positive eigenvalues), sinks ( J ( µ, ν ) only has negative eigenvalues), and saddles ( J ( µ, ν ) has a positive and a negative eigenvalue).

Let us now compute the entries of each J ( µ, ν ) . From the definition of T 1 and T 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, for any stationary point ( µ, ν ) ∈ E , we have

<!-- formula-not-decoded -->

We deduce the following Lemma which will be crucial in the proofs.

Lemma 1. Recall the definition of A ( µ ) and B ( µ, ν ) from (86) . For any stationary point ( µ, ν ) ∈ E = E 1 ∪ E 2 ∪ E 3 ∪ E 4 of the dynamics (35) , the Jacobian matrix (88) has the following form.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## F.2 Proof of Proposition 9

Thanks to Lemma 1, the Jcobian matrix is given by

<!-- formula-not-decoded -->

Note that by definition the A and B functions in from (86) that

<!-- formula-not-decoded -->

because p (1) ∈ (0 , 1) and ¯ σ γ (1) &gt; 0 . On the other hand, -4 p (1)¯ σ ′ γ (1) is negative since ¯ σ ′ γ (1) = ∑ n ≥ 1 nc 2 n &gt; 0 . We deduce that J (1 , 1) has only negative eigenvalues, and the result follows.

## F.3 Proof of Proposition 10

Wenow consider a few important choices for the link function σ , and provide a complete classification of all the stationary points of the induced ( µ, ν ) -dynamics. The picture will of course depend on the underlying link function σ .

Linear Link Function. Consider the case σ ( t ) ≡ t . It is clear that

<!-- formula-not-decoded -->

It is then easy to see that E 2 = E 3 = E 4 = ∅ , and so the set of stationary points is

<!-- formula-not-decoded -->

Observe that for any stationary point ( µ, ν ) ∈ E , one has

<!-- formula-not-decoded -->

Now, T 1 ( ± 1 , ν ) = ( p ( ± 1) -ν ) A , T 2 ( ± 1 , ν ) = -2 γp ( ± 1) , with A := p (1)(1 -p (1)) cβd &gt; 0 . The Jacobian matrices at each stationary point is then given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(98)

Since A &gt; 0 , p ( µ ) + 1 &gt; 0 and p ( µ ) -1 &lt; 0 for all µ , we deduce that

- J ( -1 , ± 1) each have one negative and positive eigenvalue: these are saddles.
- J (1 , -1) has both eigenvalues positive: this is a source.
- J (1 , 1) has both eigenvalues negative: this is a sink.

This proves Proposition 10(A).

Quadratic Link Function. Here σ ( t ) ≡ t 2 and so thanks to Proposition 11, ¯ σ γ ( ν ) = γ 4 (1 + 2 ν 2 ) . Consequently, we have

<!-- formula-not-decoded -->

It is then clear that E 2 := { ( ± 1 , ν ) | ν ∈ ( -1 , 1) , ¯ σ ′ ( ν ) = 0 } = { ( ± 1 , 0) } .

We now consider E 3 := { ( µ, ± 1) | p ( µ )(¯ σ ′ γ ( ν ) -¯ σ γ ( ± 1) = 0 } . Now, 0 = p ( µ )¯ σ γ (1) -¯ σ γ ( ± 1) iff

<!-- formula-not-decoded -->

which is impossible. Thus, E 3 = ∅ .

Finally, we compute E 4 . By definition,

<!-- formula-not-decoded -->

Now, 0 = p ( µ )¯ σ γ (1) -¯ σ γ (0) iff which is feasible iff ψ &lt; 1 .

Therefore, the stationary points are E = E 1 ∪ E 2 ∪ E 4 , with E 1 = { ( ± 1 , ± 1) } , E 2 = { ( ± 1 , 0) } , and E 4 = { ( ψ, 0) } . Recall A ( µ ) and B ( µ, ν ) from (86). We have the following classification

- ( ± 1 , ± 1) at each such stationary point, the Jacobian is given by

<!-- formula-not-decoded -->

Note, that for any µ ∈ [ -1 , 1] , it holds that

<!-- formula-not-decoded -->

since ¯ σ γ ( -1) = ¯ σ γ (1) &gt; 0 . Thus, sign( A ( µ ) B ( µ, ν ) µ ) = -sign( µ ) for all ( µ, ν ) ∈ { ( ± 1 , ± 1) } . Thus, J ( -1 , ± 1) each have one negative and positive eigenvalues and J (1 , ± 1) each have only positive eigenvalues. We conclude that ( -1 , ± 1) are saddles while, (1 , ± 1) are sinks (stable stationary points). It should comfort the reader to know that (1 , ± 1) are equivalent representations of the oracle (i.e. Bayes-optimal) parameters ( u ∗ , v ∗ ) because due to the evenness of the quadratic link function, replacing v ∗ by -v ∗ doesn't change the oracle model.

<!-- formula-not-decoded -->

- ( ± 1 , 0) : Here, J ( µ, ν ) = [ 2 A ( µ ) B ( µ, 0) µ 0 0 2 p ( µ )¯ σ ′′ γ (0) ] . In particular, we see that J ( -1 , 0) has one negative and one positive eigenvalue, while (1 , 0) has only positive eigenvalues. We conclude that ( -1 , 0) is a saddle, while (1 , 0) is a source.
- ( ψ, 0) : This is a stationary point only if 0 ≤ ψ &lt; 1 . In that case, we have

<!-- formula-not-decoded -->

Thus, J ( ψ, 0) has one negative and positive eigenvalue. We conclude that ( ψ, 0) is a saddle.

This proves Proposition 10(B).

ReLU Link Function. Consider the case where σ ( t ) ≡ ( t ) + , thanks to Proposition 11, we have

<!-- formula-not-decoded -->

One then readily computes

<!-- formula-not-decoded -->

and so

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, the stationary points are E = E 1 = { ( ± 1 , ± 1) } . Now, for any ( µ, ν ) ∈ E 1 , the Jacobian of the dynamics is

<!-- formula-not-decoded -->

More explicitly, we have

<!-- formula-not-decoded -->

Since B ( µ, 1) := 2( p ( µ )¯ σ γ (1) -¯ σ γ (1)) = -2(1 -p ( µ ))¯ σ γ (1) &lt; 0 (because ¯ σ γ (1) &gt; 0 under the ongoing constraints), we deduce that each of J ( ± 1 , -1) has one negative and one zero eigenvalue; while J ( -1 , 1) has one positive and one negative eigenvalue; J (1 , 1) has only negative eigenvalues. We conclude (1 , 1) is a sink (stable stationary point), ( -1 , 1) is a saddle.

This completes the proof of Proposition 10C.

## F.4 Proof of Proposition 4

For linear link function σ ( t ) ≡ t , we have for any µ, ν, ζ ∈ [ -1 , 1] ,

h

1

(

ν, ζ

) =

γ

ν

+

a

,

h

2

(

ν, ζ

) =

γ

ν, with

a

:=

cζ

d,

2

2

2

√

<!-- formula-not-decoded -->

We can thus simplify the auxiliary functions F 1 , F 2 , F 3 , F 4 appearing in Section E.1 like so:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, the stationary points ( µ, ν, η, ζ ) of the equations of motion given in Proposition 2 are defined by the equations F 1 = F 2 = F 3 = F 4 = 0 . We will prove that any such point must verify η = ζ = 0 .

̸

We prove that ζ = 0 . Suppose on the contrary that ζ = 0 . Then, the equations F 2 = F 4 = 0 give ν = pγ 2 /C and ζ 2 = 1 -( pγ 2 ) 2 /C 2 , where C := c √ d ≫ pγ 2 . This implies

<!-- formula-not-decoded -->

̸

Thus, we must have B = 0 , and so the equation F 1 = 0 gives u ⊤ u ∗ = µ = ± 1 . Now, µ = u ⊤ u ∗ = ± 1 implies u = ± u ∗ , and so ζ := v ⊤ u ∗ = ± v ⊤ u = ± ρ = 0 . By reductio ad absurdum , we must conclude ζ = 0 .

Note that with η = 0 , the equation F 2 = 0 now gives ν = ± 1 . This means v = ± v ∗ and so η := u ⊤ v ∗ = ± u ⊤ v = ± ρ = 0 . We conclude that every stationary has η = ζ = 0 .

We now show that the only stationary points are ( ± 1 , ± 1) . Now, plugging η = ζ = 0 , the stationary point must satisfy

<!-- formula-not-decoded -->

̸

Because p ( µ ) γ 2 &gt; 0 , second equation tells us that we must have ν = ± 1 . Plugging this into the first equation and dividing through by the factor 2 A ( µ ) &gt; 0 gives (1 -µ 2 )( p ( µ ) ∓ 1) = 0 . But p ( µ ) ∓ 1 = 0 because p ( µ ) ∈ (0 , 1) for all µ , and we conclude that µ = ± 1 . This shows that the stationary points are ( ± 1 , ± 1 , 0 , 0) as claimed.

The classification of the stationary points then follows from Proposition 10(A).

## G Proof of Proposition 8 (Convergence of PGD to Stationary Point)

## G.1 Step 1: A Descent Lemma

We shall need the following regularity assumption for the link function σ .

Assumption 1. The link function σ is C 2 on R with ∥ σ ′ ∥ ∞ , ∥ σ ′′ ∥ ∞ &lt; ∞ . The case of ReLU activation needs special treatment (not provided here).

One can show that on S 2 d -1 , the R functional

<!-- formula-not-decoded -->

is L -smooth on S 2 d -1 for some finite L &gt; 0 . Now, consider the following canonical the extension r of R from S 2 d -1 to all of R 2 d

<!-- formula-not-decoded -->

Note that ∇ u r ( u, v ) = P ⊥ u ∇ u R ( u, v ) for all ( u, v ) ∈ S 2 d -1 ⊆ S 2 d -1 ( δ ) ⊆ R 2 d . Then, one can show that for any δ ∈ (0 , 1] , the functional r is L δ -smooth with L δ := 4 L/ (1 -δ ) 2 , on the tube

<!-- formula-not-decoded -->

This means that

<!-- formula-not-decoded -->

for any u, v, u ′ , v ′ ∈ S 2 d -1 ( δ ) , with ∆ = ( u ′ -u, v ′ -v ) ∈ R 2 d . Furthermore, because r is radially symmetric, we know that ∇ u r ( u, v ) ⊥ u and ∇ v r ( u, v ) ⊥ v for all non-zero u, v ∈ R d .

Now, define g k = ( a k , b k ) , where a k , b k ∈ R d are defined by

<!-- formula-not-decoded -->

In (109) above, taking u = u k , v = v k , u ′ = ˜ u k +1 := u k -sa k (as in (28)), and v ′ := ˜ v k +1 = u k -sg k , we get

<!-- formula-not-decoded -->

We shall now control the deviation of ( u k +1 , v k +1 ) from (˜ u k +1 , ˜ v k +1 ) . By definition, we have

<!-- formula-not-decoded -->

Now, because ˜ u k +1 = u k -sg u k with ∥ u k ∥ = 1 and u k ⊥ g u k , we have

<!-- formula-not-decoded -->

where the last step uses the elementary inequality 1 + a ≤ 1 + a/ 2 for all a ≥ 0 . Similarly, for v k +1 we have

<!-- formula-not-decoded -->

On the other hand, (28) and (29) give

<!-- formula-not-decoded -->

We deduce that

<!-- formula-not-decoded -->

Using this in (109) above gives with u = ˜ u k +1 , v = ˜ v k +1 , u ′ := u k +1 = ˜ u k +1 , v ′ := u k +1 , so that ∆ = ζ k := ( u k +1 -˜ u k +1 , v k +1 -˜ v k +1 ) gives

<!-- formula-not-decoded -->

where we have used the fact that ∇ r (˜ u k +1 , ˜ v k +1 ) ⊥ ζ k . Combining with (110) above gives

<!-- formula-not-decoded -->

Now, one can show that ∥∇ r ( u, v ) ∥ ∞ ≤ M δ &lt; ∞ uniformly on S d -1 ( δ ) . Thus, if 0 &lt; s &lt; 1 /M δ , then s 4 ∥ g k ∥ 4 = ( s 2 ∥ g k ∥ 2 ) s 2 ∥ g k ∥ 2 ≤ s 2 ∥ g k ∥ 2 , and we get

<!-- formula-not-decoded -->

provided the stepsize s is sufficiently small in the sense that

<!-- formula-not-decoded -->

Noting that r = R on S 2 d -1 ⊆ S 2 d -1 ( δ ) , we get the following descent lemma.

Lemma 2. If the step size s is sufficiently small in the sense that 0 &lt; s &lt; min(1 / (2 L δ ) , M δ ) , then

<!-- formula-not-decoded -->

where we recall that g k = ( a k , b k ) , a k = P ⊥ u k ∇ u R ( u k , v k ) , b k = P ⊥ v k ∇ v R ( u k , v k ) .

## G.2 Step 2: Convergence to Stationary Point

The above inequality can be rewritten as sL δ ∥ g k ∥ 2 / 4 ≤ R k -R k +1 , and summing both sides gives

<!-- formula-not-decoded -->

where R min := min u,v ∈ S d -1 R ( u, v ) = 0 . We deduce that ∥ g k ∥ → 0 , and so g k → 0 in the limit k → 0 . Now, one computes

<!-- formula-not-decoded -->

Analogously, we get ∥ v k +1 -v k ∥ ≤ (3 / 2) s ∥ b k ∥ . Combining gives

<!-- formula-not-decoded -->

We deduce that the PGD iterates ( u k , v k ) given in (29) form a Cauchy sequence in S 2 d -1 . Due to completeness of S 2 d -1 , this sequence has a limit ( u ∞ , v ∞ ) ∈ S 2 d -1 . We now show that ( u ∞ , v ∞ ) is a stationary point of the risk functional R .

For simplicity of presentation, we focus on the case of linear link function σ ( t ) ≡ t .

Thanks to Proposition 3, if ( u 0 , v 0 ) ∈ M , then ( u k , v k ) ∈ M for all k and one computes,

<!-- formula-not-decoded -->

where T j = T j ( µ k , ν k , 0) for j = 1 , 2 , 3 , as defined in Appendix E.1, with µ k := u ⊤ k u ∗ and ν k := v ⊤ k v ∗ . Note that we have used the fact that T 3 ( µ k , ν k , 0) = 0 thanks to the equation (107). We deduce that

<!-- formula-not-decoded -->

Thus, the limit point ( u ∞ , v ∞ ) is such that

<!-- formula-not-decoded -->

This is precisely the characterization of stationary points risk functional R established in Proposition 7. This concludes the proof of Proposition 8.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes] .

Justification: The main contributions outlined in the introduction are clearly linked to the appropriate sections and adequately support each claim.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes] .

Justification: Yes we discuss limitations in the last section of the main paper.

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

Justification: We provide the exact assumptions in the main body for each theorem/proposition and corollary and the complete proof is provided in the appendix.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## [Yes]

Justification: We provide all the experimental details in dedicated sections in the appendix (supplemental).

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

Answer: [NA] .

Justification: This is purely a theoretical work. Question doesn't apply.

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

Justification: We provide all the experimental details in dedicated sections in the appendix (supplemental).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No] .

Justification: This is a pure theoretical work.

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

Answer: [Yes] .

Justification: This is a pure theoretical work. Experiments were run with a single CPU on a laptop, and took less than 30 minutes in total. We provide all the experimental details in dedicated sections in the appendix (supplemental).

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer:[Yes] .

Justification: We have read the code of ethics carefully and we conform with it.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer:[NA] .

Justification: Our work proposes a theoretical analysis of self-attention, the main main building block of transformer-based models which power LLMs like ChatGPT, Llama, DeepSeek, etc. Building a rigorous understanding transformers has the potential to help understand how they fail (e.g hallucinate), or drastically perform well on tasks previously though to be unsolvable with ML/AI.

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

Justification: This is a purely theoretical work. No new gadgets (models, architecture, datasets, etc.) are being proposed here.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The original creators of all models, datasets and algorithms used in this work are properly credited, with citations in the manuscript.

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

Answer: [NA] .

Justification: We do not introduce new assets.

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

Answer: [NA] .

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [No]

Justification: LLMs are not part of the core, method development of this research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.