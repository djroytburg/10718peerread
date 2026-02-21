## Smoothed Agnostic Learning of Halfspaces over the Hypercube

## Yiwen Kou

Department of Computer Science, UCLA Los Angeles, CA, US evankou@cs.ucla.edu

## Raghu Meka

Department of Computer Science, UCLA Los Angeles, CA, US raghum@cs.ucla.edu

## Abstract

Agnostic learning of Boolean halfspaces is a fundamental problem in computational learning theory, but it is known to be computationally hard even for weak learning. Recent work (Chandrasekaran et al., 2024) proposed smoothed analysis as a way to bypass such hardness, but existing frameworks rely on additive Gaussian perturbations, making them unsuitable for discrete domains. We introduce a new smoothed agnostic learning framework for Boolean inputs, where perturbations are modeled via random bit flips. This defines a natural discrete analogue of smoothed optimality generalizing the Gaussian case. Under strictly subexponential assumptions on the input distribution, we give an efficient algorithm for learning halfspaces in this model, with runtime and sample complexity ˜ O ( n poly( 1 σϵ ) ) . Previously, such algorithms were known only with strong structural assumptions for the discrete hypercube-for example, independent coordinates or symmetric distributions. Our result provides the first computationally efficient guarantee for smoothed agnostic learning of halfspaces over the Boolean hypercube, bridging the gap between worst-case intractability and practical learnability in discrete settings.

## 1 Introduction

Halfspaces, or linear threshold functions (LTFs), are one of the most fundamental concept classes in machine learning. In the realizable setting (Valiant, 1984), they are efficiently learnable by classical algorithms such as the Perceptron (Rosenblatt, 1958; Novikoff, 1963), Winnow (Littlestone, 1987), large-margin methods like Support Vector Machines (Cortes and Vapnik, 1995), or by linear programming. These methods exploit linear separability and can perform well even in the presence of irrelevant features.

In contrast, the agnostic learning framework (Haussler, 1992; Kearns et al., 1992), which allows for arbitrary label noise, poses significant algorithmic challenges. In this setting, the goal is to find a hypothesis that competes with the best in a concept class, without assuming that the data is linearly separable. However, agnostic learning of halfspaces is computationally hard in the worst case: even weak learning-achieving error marginally better than random guessing-is NP-hard under standard complexity assumptions, both in continuous domains (Feldman et al., 2009) and on the Boolean hypercube (Guruswami and Raghavendra, 2009).

To overcome these barriers, several restricted models have been studied. For example, under random classification noise (RCN), halfspaces remain learnable using modified Perceptron algorithms or linear programming (Blum et al., 1996; Cohen, 1997). Under Massart noise, where adversarial flips are bounded in probability, recent work has led to efficient learning algorithms (Awasthi et al., 2015; Diakonikolas et al., 2019, 2020, 2021). Other lines of work exploit structure in the input distribution: Kalai et al. (2008) gave improper agnostic learning algorithms under uniform, spherical,

or log-concave distributions by approximating halfspaces with low-degree polynomials, extending earlier Fourier-based methods (Linial et al., 1993).

A more recent and promising direction is based on smoothed analysis, which was introduced to explain the practical performance of algorithms that are worst-case hard (Spielman and Teng, 2001). In learning theory, Chandrasekaran et al. (2024) proposed a smoothed agnostic framework in which the learner competes with the best classifier under slight random perturbations of the inputs. This relaxation enables efficient algorithms for learning low-dimensional concepts, even when worst-case learning is intractable. However, their approach is tailored to continuous domains and relies on additive Gaussian noise, and hence is not suitable for discrete domains such as the Boolean hypercube.

Our contribution. We develop a discrete analogue of smoothed agnostic learning for Boolean concept classes over {± 1 } n , where additive Gaussian noise is ill-defined. Instead of perturbing examples in Euclidean space, we introduce bit-flipping noise: each input coordinate is independently flipped with probability σ . This gives rise to a new benchmark for learning that captures robustness to small discrete perturbations, interpolating between classical agnostic learning ( σ = 0 ) and random guessing ( σ = 1 / 2 ).

To formalize this, we begin by recalling the standard agnostic learning objective.

Definition 1.1 (Agnostic Optimality) Let X be a domain and F be a class of functions f : X → {± 1 } . Let D be a distribution over labeled examples ( x , y ) ∈ X × {± 1 } . The agnostic error of F under D is defined as

̸

<!-- formula-not-decoded -->

We now define a smoothed variant of this benchmark, in which each input is perturbed before evaluation. This definition is general and does not assume any particular structure of the domain or distribution.

Definition 1.2 (Smoothed Optimality) Let X be a domain and F be a class of functions f : X → {± 1 } . Let D be a distribution over labeled examples ( x , y ) ∈ X ×{± 1 } and P σ ( x ) be a perturbation distribution of x over X . Define the smoothed agnostic error as:

̸

<!-- formula-not-decoded -->

In standard agnostic learning, the goal is to compete with opt under D . Following Chandrasekaran et al. (2024), we instead compete with opt σ in Definition 1.2.

Definition 1.3 (Smoothed Agnostic Learning) Fix ϵ, σ &gt; 0 and δ ∈ (0 , 1) . An algorithm A learns the class F in the σ -smoothed agnostic setting if, given i.i.d. samples from D , it outputs a hypothesis h : X → {± 1 } such that with probability at least 1 -δ :

̸

<!-- formula-not-decoded -->

As an example, Chandrasekaran et al. (2024) study the case where X = R n and the perturbation distribution is additive Gaussian noise: P σ ( x ) = N ( x , σ 2 I n ) . This formulation relies on the Euclidean structure of R n and does not extend to discrete domains. In our setting, we consider X = {± 1 } n and define perturbations via random bit flips: P σ ( x ) = x ⊙ z where z ∼ N σ is a product distribution with z i = -1 with probability σ and 1 otherwise. This defines a natural smoothed learning model for the Boolean hypercube that avoids embedding the domain into R n .

Under this framework, we show that halfspaces over the Boolean cube are efficiently learnable in the smoothed agnostic model under mild distributional assumptions. Our approach extends the classical L 1 -polynomial regression framework to this smoothed setting. The key idea is that every Boolean halfspace, when composed with small random bit-flip perturbations, admits a low-degree polynomial approximation under the input distribution. To establish this, we analyze a smoothed version of the halfspace defined via a noise operator (Definition 3.2), and construct approximators using Berry-Esseen-type arguments combined with critical index analysis to handle irregular weight vectors. We obtain the following result:

Theorem 1.4 (Subgaussian-Informal, see also Theorem 4.8) Let D be a distribution on {± 1 } n × {± 1 } with sub-gaussian x -marginal of variance proxy σ 2 0 . There exists an algorithm that learns

the class of linear threshold functions in the σ -smoothed setting with N = n poly( σ 0 /σϵ ) log(1 /δ ) samples and poly( n, N ) runtime.

This is the first result that establishes efficient smoothed agnostic learning of halfspaces over the Boolean hypercube. While our algorithm is improper, it achieves strong generalization guarantees under natural distributions. Previously, such results for the hypercube were only known under very restricted distributions as discussed below.

## 2 Related Work

Distributional Assumptions in Halfspace Learning: It is well-understood that agnostically learning halfspaces is intractable in the worst case (Feldman et al., 2009; Guruswami and Raghavendra, 2009), even under relatively benign noise models (Diakonikolas et al., 2022). This has motivated a long line of distribution-specific algorithms that guarantee learnability by leveraging assumptions on the data distribution. Early work focused on uniform or product distributions, where powerful Fourier-analytic techniques yield low-degree approximations (Linial et al., 1993; Klivans et al., 2004a; Blais et al., 2010). Under the uniform hypercube distribution, halfspace concepts exhibit strong Fourier concentration and low noise sensitivity (O'Donnell, 2021), enabling efficient learning via low-degree polynomial approximation (Klivans et al., 2004a). This was extended to symmetric distributions in Wimmer (2010) and to arbitrary product distributions in Blais et al. (2010). However, beyond these there are very few general classes of distributions over they hypercube where halfspaces are agnostically learnable.

For continuous distributions, halfspaces were shown to be agnostically learnable under log-concave distributions by Kalai et al. (2008) and this was later extended to intersections and other functions of halfspaces in Kane et al. (2013). Much like the discrete setting, until the recent work of Chandrasekaran et al. (2024), most positive results required strong structural assumptions on the marginal distribution of the examples. This work introduced a new smoothed agnostic model which led to several new results for learning halfspaces and functions of halfspaces for a much broader class of distributions (e.g., sub-gaussian or sub-exponential densities). Our work continues this progression to very general distributions, but focuses on the Boolean domain and shows that only mild tail bounds (strictly sub-exponential) suffice for efficient learning in the smoothed setting.

Noise Models and Smoothed Analysis: In parallel to distributional assumptions on X , a complementary line of work has tackled label noise models and smoothed analysis. The classical noise models include random classification noise (RCN), where each label is independently flipped with some probability. Blum et al. (1996) gave the first polynomial-time algorithm for learning a halfspace under random classification noise, exploiting the fact that a halfspace's margin makes it relatively robust to independent label flips. A stronger noise model is the Massart noise model, which bounds the adversary by a flipping probability η &lt; 1 / 2 on each example. Diakonikolas et al. (2020) gave an efficient algorithm for learning halfspaces with Massart noise over log-concave distributions. On the other hand, with adversarial (malicious) noise, learning halfspaces requires additional assumptions. Klivans et al. (2009) designed efficient algorithms for origin-centered halfspaces under malicious noise by assuming isotropic log-concave distribution and small noise rate. In smoothed analysis of learning, one assumes that either the data (Blum and Dunagan, 2002; Kane et al., 2013) or the target concept (Chandrasekaran et al., 2024) is randomly perturbed, so that pathological arrangements are avoided. Chandrasekaran et al. (2024) introduced a smoothed agnostic PAC model in R d where the learner competes against the best classifier that is robust to slight Gaussian perturbations of examples. Our work can be seen as a Boolean analogue of this idea: rather than perturbing continuous inputs, we require the optimal halfspace to be stable under small random label flips.

## 3 Preliminaries

We review relevant definitions from Boolean function analysis that will allow us to define a discrete smoothing operator and justify using it in place of the original linear threshold function. We use definitions from the analysis of Boolean functions over product spaces, following the framework of Mossel et al. (2005). Let (Ω 1 , µ 1 ) , . . . , (Ω n , µ n ) be finite probability spaces and let (Ω , µ ) denote their product. In our setting, we take Ω i = {± 1 } and define µ to be the product distribution N σ , where each coordinate is 1 with probability 1 -σ and -1 with probability σ , independently.

Definition 3.1 ( ρ -noisy copy) Given x ∈ Ω and ρ ∈ [0 , 1] , a ρ -noisy copy of x is a random vector y ∼ N ρ ( x ) , where each coordinate y i is independently set to x i with probability ρ and to an independent draw from µ i with probability 1 -ρ .

Definition 3.2 (Noise operator T ρ ) For any function f : Ω → R and ρ ∈ [0 , 1] , the noise operator T ρ is defined as

<!-- formula-not-decoded -->

This definition generalizes the Bonami-Beckner operator (Kahn et al., 1988) when µ is the uniform distribution on the hypercube. Intuitively, T ρ f is a smoothed version of f , computed by averaging f over a neighborhood of x with geometric decay controlled by ρ . In particular, T 1 f = f , and as ρ decreases from 1, T ρ f suppresses high-frequency components of f . This operator will be used as our main tool for constructing smoothed approximations to Boolean threshold functions.

Definition 3.3 (Noise stability and noise sensitivity) For any f : Ω → R , the noise stability at parameter ρ is defined as

<!-- formula-not-decoded -->

If f : Ω →{± 1 } , the noise sensitivity at parameter δ ∈ [0 , 1] is given by

̸

<!-- formula-not-decoded -->

̸

Equivalently, NS δ ( f ) = Pr x,y [ f ( x ) = f ( y )] where x and y have Hamming correlation 1 -2 δ . This quantity captures the robustness of f to small input perturbations.

It is well-known that natural Boolean functions with low total influence or low-degree Fourier concentration exhibit low noise sensitivity. In particular, linear threshold functions are noise-stable under both uniform (Peres, 2004) and general product distributions (Blais et al., 2010). The following lemma bounds the noise sensitivity of halfspaces over arbitrary product spaces.

Lemma 3.4 (Theorem 3.2 in Blais et al. (2010)) Let f : Ω →{± 1 } is a linear threshold function, where the domain Ω = Ω 1 × · · · × Ω n has the product distribution µ = µ 1 × · · · × µ n . Then NS δ ( f ) ≤ 5 4 √ δ .

This bound implies that for ρ = 1 -δ close to 1 , the smoothed function T ρ f closely approximates the original threshold function f . This justifies our strategy of working with T ρ f instead of f in the smoothed learning setting: any learner that performs well on T ρ f will, up to a small error, also succeed on f .

Notation. We use small boldface characters for vectors and capital bold characters for matrices. We use [ d ] to denote the set { 1 , 2 , · · · , d } . For a vector x ∈ R d and i ∈ [ d ] , x i denotes the i -th coordinate of x , and ∥ x ∥ 2 := √ ∑ d i =1 x 2 i the ℓ 2 norm of x . For x , y ∈ R d , we use ⟨ x , y ⟩ = ∑ d i =1 x i y i as the inner product between them and x ⊙ y = ( x 1 y 1 , · · · , x d y d ) as the Hadamard product between them. We use 1 {E} to be the indicator function of some event E . For ( x , y ) distributed according to D , we denote D x to be the marginal distribution of x .

## 4 Technical Overview

In this section, we outline the main steps of our analysis. Our approach follows a reduction-based strategy: we reduce smoothed agnostic learning of Boolean halfspaces to the problem of approximating a smoothed halfspace by a low-degree polynomial, which can then be learned via L 1 regression (Section 4.1). We begin by replacing the original target f x ( z ) = f ( x ⊙ z ) with a smoothed surrogate T 1 -ρ f x ( z ) (Definition 3.2), facilitating approximation by low-degree polynomials (Section 4.2).

To handle the biased distribution arising from noise perturbation, we introduce a rerandomization and conditioning trick that rewrites each bit as a mixture involving uniform random variables. This allows us to express the smoothed function as a conditional expectation over uniformly random inputs, making it amenable to quantitative central-limit theorems (Berry-Esseen estimates; Section 4.3). We then use a case analysis facilitated by a decomposition of the weight vector (Section 4.4):

1. If a small number of large coordinates (the 'head') dominate, the halfspace's output is primarily determined by those coordinates, and we can approximate the function directly.
2. Otherwise, the remaining 'tail' is regular , and we apply the Berry-Esseen theorem to approximate the Boolean sum by a Gaussian. This reduces the problem to the continuous setting, where we leverage Gaussian-based techniques (the density ratio method from Chandrasekaran et al. (2024)) to construct low-degree polynomial approximations.

Together, these ingredients yield an efficient smoothed learner for Boolean halfspaces under strictly sub-exponential input distributions.

## 4.1 High-Level Approach via L 1 egression

Our starting point is the L 1 -polynomial regression method for agnostic learning. In particular, Kalai et al. (2008) established a powerful reduction from agnostic learnability to low-degree polynomial approximation.

Algorithm 1 L 1 Polynomial Regression Algorithm

Input: Sample S = { ( x 1 , y 1 ) , . . . , ( x N , y N ) } , degree bound d

1: Find polynomial p of degree ≤ d to minimize

<!-- formula-not-decoded -->

(This can be done by expanding examples to include all monomials of degree ≤ d and then performing L 1 linear regression.)

- 2: Output hypothesis h ( x ) = sign( p ( x ) -t ) , where t ∈ [ -1 , 1] is chosen to minimize the classification error on S .

̸

Theorem 4.1 (Theorem 5 in Kalai et al. (2008)) Suppose min deg( p ) ≤ d E D x [ | p ( x ) -c ( x ) | ] ≤ ϵ for some degree d and any c in the concept class C . Then, for h output by the degreed L 1 polynomial regression algorithm with N = poly( n d /ϵ ) examples, E S ∼D N [ P ( x ,y ) ∼D [ h ( x ) = y ]] ≤ opt + ϵ , where opt = min f ∈C P ( x ,y ) ∼D [ f ( x ) = y ] . If we repeat the algorithm r = O (log(1 /δ ) /ϵ ) times with fresh examples each, and let h be the hypothesis with lowest error on an independent test set of size O (log(1 /δ ) /ϵ 2 ) , then with probability at least 1 -δ , P ( x ,y ) ∼D [ h ( x ) = y ] ≤ opt + ϵ .

̸

Theorem 4.1 says that if the target function f can be approximated in L 1 by a low-degree polynomial p with error at most ϵ , then one can efficiently learn f to misclassification error opt + ϵ , where opt is the Bayes-optimal error rate under distribution D . Once such a polynomial is shown to exist, Theorem 4.1 implies a computationally efficient learning algorithm with sample complexity N = poly( n d /ϵ ) log(1 /δ ) .

## 4.2 Smoothed Learning as Non-Worst-Case Approximation

The challenge is that an arbitrary halfspace f ( x ) = sign( ⟨ w , x ⟩ -θ ) might not be well-approximated by any low-degree polynomial over worst-case input distributions. Following Chandrasekaran et al. (2024), we view smoothed learning as a form of non-worst-case approximation. In this smoothed agnostic setting, the learner's 'effective' target concept is the mapping ( x , z ) ↦→ f ( x ⊙ z ) , where z ∈ {± 1 } n is a random noise vector independent of x with σ close to 0 meaning only a tiny fraction of bits are flipped on average. We extend the L 1 -regression reduction to handle this scenario. In particular, we prove an analogue of Kalai et al. (2008)'s result tailored to the smoothed model:

̸

Theorem 4.2 Suppose min deg( p z ) ≤ d E z ∼D σ , x ∼D x [ | p z ( x ) -f ( x ⊙ z ) | ] ≤ ϵ for some degree d and any halfspace f , where D x is any distribution on {± 1 } n . Then, for h output by the degreed L 1 polynomial regression algorithm with N = poly( n d /ϵ ) examples, E S ∼D N [ P ( x ,y ) ∼D [ h ( x ) = y ]] ≤ opt σ + ϵ . If we repeat the algorithm r = O (log(1 /δ ) /ϵ ) times with fresh examples each, and let h be the hypothesis with lowest error on an independent test set of size O (log(1 /δ ) /ϵ 2 ) , then with probability at least 1 -δ , P ( x ,y ) ∼D [ h ( x ) = y ] ≤ opt σ + ϵ .

̸

̸

After this reduction, our task reduces to a purely approximation-theoretic problem: we need to construct, for each noise vector z , a polynomial p z ( x ) in the variable x such that the expected L 1 error over the smoothing process remains small:

<!-- formula-not-decoded -->

To achieve this, we treat the smoothing noise z and consider x as a fixed parameter. This reduces the problem to approximating the function f x ( z ) = f ( x ⊙ z ) . We replace f x ( z ) with its smooth approximation by applying the generalized Bonami-Beckner operator (Definition 3.2) on z :

<!-- formula-not-decoded -->

Applying Lemma 3.4 with ρ = O ( ϵ 2 ) , we obtain:

<!-- formula-not-decoded -->

Therefore, if we can find a low-degree polynomial that approximates T 1 -ρ f x ( z ) well in L 1 , that polynomial will also succeed in approximating f x ( z ) . The remainder of our technical approach will be devoted to constructing such a polynomial approximator for the smoothed halfspace T 1 -ρ f x ( z ) .

## 4.3 From Biased to Uniform Distribution on the Hypercube

To construct low-degree polynomial approximations, we analyze the noise-smoothed function T 1 -ρ f x . Recall that for a fixed input x , we define f x ( z ) = f ( x ⊙ z ) , and suppose f ( · ) = sign( ⟨ w , ·⟩ -θ ) . Then we have:

<!-- formula-not-decoded -->

Here, z ∼ N σ denotes a product distribution over {± 1 } n where each bit z i is 1 with probability 1 -σ and -1 with probability σ . The vector y is a (1 -ρ ) -noisy copy of z (Definition 3.1) with probability 1 -ρ , y i = z i ; otherwise, y i is redrawn independently from N σ with probability ρ . Therefore, y ∼ N σ , correlated with z , follows a biased distribution on the hypercube. To facilitate polynomial approximation, we aim to reduce this to a form where the randomness comes from a uniform distribution. To achieve this, we introduce a rerandomization trick that rewrites each coordinate y i as:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

with ϵ i being a Radmacher random variable (uniform over {± 1 } ) .

This decomposition captures the full noise process: l i is an indicator that determines whether the coordinate is kept as z i (with probability 1 -ρ ) or resampled as τ i ∼ ( N σ ) i (with probability ρ ). The variable m i is then used to rerandomize τ i , since τ i can be viewed as taking the value 1 with probability 1 -2 σ (when m i = 0) or a uniform random bit ϵ i with probability 2 σ (when m i = 1 ).

Akey benefit is that, conditional on l and m , the random component ϵ follows the uniform distribution on {± 1 } n . We now condition on ( l , m ) and express the smoothed function as:

<!-- formula-not-decoded -->

where b is a deterministic shift depending on the coordinates fixed by l , m .

Given that ϵ is uniform distribution on hypercube, the inner sum behaves like a sum of independent {± 1 } random variables. Under mild regularity condition (Definition 4.3) on the weight vector u , we can apply the Berry-Esseen Theorem to approximate this inner distribution by a Gaussian. Specifically, we approximate:

<!-- formula-not-decoded -->

Substituting into the earlier expression yields the Gaussian-smoothed approximation:

<!-- formula-not-decoded -->

This reduces our setting to the Gaussian noise model analyzed in Chandrasekaran et al. (2024) for which efficient low-degree polynomial approximations are known. In particular, the density ratio method developed in that work can be applied to approximate ˜ T 1 -ρ f x ( z ) with a small L 1 error.

<!-- formula-not-decoded -->

## 4.4 Handling Irregularity via Critical Index Analysis

Recall that the approximation in (4.1) relies on the Berry-Esseen Theorem, which introduces a uniform approximation error of O ( ( ∥ u ⊙ l ⊙ m ∥ 3 ∥ u ⊙ l ⊙ m ∥ 2 ) 3 ) for the cumulative density function. This can be further bounded by O ( ∥ u ⊙ l ⊙ m ∥ ∞ ∥ u ⊙ l ⊙ m ∥ 2 ) . Note that each coordinate l i m i is equal to 1 with probability 2 ρσ and 0 otherwise. By concentration, we have ∥ u ⊙ l ⊙ m ∥ 2 ≈ (2 ρσ ) 1 / 2 ∥ u ∥ 2 , so the approximation error becomes O ( ( ρσ ) -1 / 2 ∥ u ∥ ∞ ∥ u ∥ 2 ) . This motivates the following regularity condition:

Definition 4.3 (regularity) For vector w ∈ R n , w is α -regular if ∥ w ∥ ∞ ≤ α · ∥ w ∥ 2 .

Given this definition, we see that if u is α -regular, then the approximation in (4.1) holds with L ∞ error O (( ρσ ) -1 / 2 α ) . Since u = w ⊙ x and x ∈ {± 1 } n , the regularity of u is equivalent to that of w . For such 'good' (i.e., α -regular with small α ) weight vectors w , we can construct low-degree polynomial approximators by reducing to the Gaussian setting analyzed in Section 4.3 and Chandrasekaran et al. (2024).

However, we must also handle the 'bad' or irregular cases, where ⟨ u , l ⊙ m ⊙ ϵ ⟩ deviates significantly from Gaussian behavior. To deal with such irregular w , we employ critical index analysis, a standard tool in the analysis of Boolean halfspaces (Servedio, 2006; Matulef et al., 2010; Diakonikolas et al., 2010; Meka and Zuckerman, 2010; O'Donnell and Servedio, 2011; Diakonikolas and Servedio, 2013).

Definition 4.4 ( α -critical index) For u ∈ R n , assume that | u 1 | ≥ · · · ≥ | u n | . We define the α -critical index ℓ ( α ) of a halfspace h ( x ) = sign( ⟨ u , x ⟩ -θ ) as the smallest index i ∈ [ n ] for which | u i | ≤ α · σ i , where σ i := √ ∑ n j = i u 2 j .

Intuitively, the α -critical index is the first index i such that the tail weight vector ( u i , · · · , u n ) is α -regular. Our earlier argument covers the case i = 1 , where the entire vector is regular. Using this framework, we obtain the following structural result:

Lemma 4.5 (Critical Index Decomposition) Without loss of generality, let u = w ⊙ x with entries sorted in non-increasing magnitude, i.e., | u 1 | ≥ · · · ≥ | u n | . Suppose x follows a ( α, λ ) -strictly sub-exponential distribution on {± 1 } n . For any fixed z , there exists a threshold K = K ( α, ϵ ) = O ( log(1 + λ ) /α 2 +log(1 /ϵ ) log(1 /α ) /ρσα 2 ) such that one of the following two conditions holds:

1. For some H &lt; K , the tail vector u T = ( u H +1 , · · · , u n ) is α -regular, where α is to be choosen later.

̸

<!-- formula-not-decoded -->

This lemma is proved by analyzing two cases, depending on whether the critical index ℓ ( α ) satisfies 1 &lt; ℓ ( α ) &lt; K , or ℓ ( α ) ≥ K . In the former case, we set H = ℓ ( α ) -1 , and u T is α -regular. In the latter case, the head vector forms a sufficiently long geometrically decaying sequence | u 1 | ≥ · · · ≥ | u H | to ensure that the influence of the remaining tail vector u T on the halfspace output is negligible. That is, with high probability, sign( ⟨ u , y ⟩ -θ ) ≈ sign( ⟨ u H , y H ⟩ -θ ) .

We now show how to construct low-degree polynomial approximators in both cases.

Case 1: When u T is α -regular, we condition on y H . For each fixed y H , the function becomes a regular halfspace in y T :

<!-- formula-not-decoded -->

We apply the techniques of Chandrasekaran et al. (2024) to approximate this with a low-degree polynomial. One subtlety is that directly applying their construction leads to a degree polynomial in | ˜ θ | / ∥ u T ∥ 2 . To address this, we use an indicator trick to define:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˜ p y H ( x ) can be constructed using the idea from Chandrasekaran et al. (2024) since | ˜ θ | / ∥ u T ∥ 2 is controlled. The indicator functions are low-degree polynomials of degree at most H , since they only depend on H variables and any function f : {± 1 } k → R can be represented by a degree at most k multilinear polynomial.

Case 2: If the second condition of the lemma holds, we approximate T 1 -ρ f x ( z ) directly using sign( ⟨ u H , y H ⟩ -θ ) . Since this depends only on the first H coordinates, it can be exactly represented as a polynomial of degree at most H .

In either case, we obtain a low-degree polynomial approximator for the smoothed function T 1 -ρ f x ( z ) .

## 4.5 Results

Using this framework, we establish the following approximation bound:

Definition 4.6 (Strictly Sub-exponential Distributions) A distribution D on R d is ( α, λ ) -strictly sub-exponential if for all ∥ v ∥ 2 = 1 , P x ∼D [ |⟨ x , v ⟩| &gt; t ] ≤ 2 · e -( t/λ ) 1+ α .

Lemma 4.7 Fix ϵ &gt; 0 and a sufficiently large universal constant C &gt; 0 . Let D be a ( α, λ ) -strictly sub-exponential distribution on {± 1 } n . Let f : {± 1 } n → {± 1 } be a linear threshold function. There exists a family of polynomials p z parameterized by z of degree at most O ( ( Cσ -1 2 λ log(1 /ϵ ) /ϵ ) 6(1+ 1 α ) 3 ) such that E z ∼N σ E x ∼ D [ | p z ( x ) -f x ( z ) | ] is at most ϵ .

Given the polynomial approximation and the degree upper bound, one can directly run L 1 polynomial regression (Algorithm 1) as stated in Theorem 4.2. We now can get our main theorem for strictly sub-exponential distributions.

̸

Theorem 4.8 Let D be a distribution on {± 1 } n ×{± 1 } such that the marginal distribution is ( α, λ ) -strictly sub-exponential. There exists an algorithm that draws N = n poly(( λ/σϵ ) (1+1 /α ) 3 ) log(1 /δ ) samples, runs in time poly( n, N ) , and computes a hypothesis h ( x ) such that, with probability at least 1 -δ , it holds that P ( x ,y ) ∼D [ y = h ( x )] ≤ opt σ + ϵ .

Our main theorem shows that any Boolean halfspace on {± 1 } n can be learned agnostically in the smoothed model under strictly sub-exponential input distributions. This result holds in a general and challenging setting where prior techniques fail, and it achieves efficient runtime and sample complexity. Table 1 compares our guarantees with the most relevant prior works. Conceptually, our contributions extend the scope of agnostic halfspace learning in two fundamental directions:

Relaxing distributional assumptions via smoothed optimality: A key technical contribution is our use of a smoothed benchmark opt σ (Definition 1.3) instead of the worst-case error opt (Definition 1.1), enabling learning under substantially weaker distributional assumptions. In particular, we show that halfspaces remain efficiently learnable under general strictly sub-exponential marginals, which is a significant relaxation compared to the strong structural assumptions required in earlier work. For example, the Fourier-based techniques of Klivans et al. (2004b); Kalai et al. (2008); Blais et al. (2010) exploit spectral concentration under uniform or product distributions to obtain low-degree polynomial approximations. To go beyond the product setting, Wimmer (2010) generalized previous techniques to symmetric group to handle permutation-invariant distributions. However, these methods break down when the input has more dependencies or heavier tails. In contrast, our approach succeeds under strictly sub-exponential marginals by combining bit-flip smoothing with a critical index decomposition and Berry-Esseen approximation, enabling polynomial approximation without requiring coordinate independence or permutation invariant structure.

Extending the smoothed learning framework to the Boolean hypercube: A second core contribution is our extension of smoothed agnostic learning to the Boolean domain {± 1 } n , where additive Gaussian perturbations used in prior smoothed models are not well-defined. In continuous domains, several tools, including Gaussian surface area bounds (Klivans et al., 2008), log-concave concentration inequalities (Kane et al., 2013), and Gaussian smoothing combined with density ratio techniques (Chandrasekaran et al., 2024), enable efficient agnostic learning of halfspaces. However, none of these directly apply to the hypercube. Our analysis circumvents this barrier by performing a case analysis based on the critical index of the weight vector: either a small number of head coordinates

Table 1: Comparison of agnostic learning of halfspaces. We ignore the polynomial logarithmic factors in 1 /δ .

| Work                         | Domain   | Distribution     | Bench.   | Smooth        | Complexity                           |
|------------------------------|----------|------------------|----------|---------------|--------------------------------------|
| Kalai et al. (2008)          | {± 1 } n | Uniform          | opt      | None          | n O ( 1 ϵ 4 ) n O ( 1 ϵ 4 )          |
| Kalai et al. (2008)          | S n - 1  | Uniform          | opt      | None          |                                      |
| Kalai et al. (2008)          | R n      | Log-concave      | opt      | None          | n O ( d ( ϵ ))                       |
| Klivans et al. (2008)        | R n      | Gaussian         | opt      | None          | n O ( 1 ϵ 4 )                        |
| Wimmer (2010)                | [ B ] n  | Perm-Inv         | opt      | None          | n O ( 1 ϵ 4 ) (log log( 1 )) ˜ O (1) |
| Kane et al. (2013)           | R n      | Sub-exp          | opt σ    | Input noise   | n exp( σϵ σ 4 ϵ 4                    |
| Chandrasekaran et al. (2024) | R n      | Strictly Sub-exp | opt σ    | Concept noise | n poly(( λ σϵ ) (1+ 1 α ) 3 )        |
| Ours                         | {± 1 } n | Strictly Sub-exp | opt σ    | Concept noise | n poly(( λ σϵ ) (1+ 1 α ) 3 )        |

dominate and effectively determine the output, or the remaining tail is regular, allowing us to invoke the Berry-Esseen theorem to approximate the Boolean tail sum by a Gaussian, thereby enabling the use of continuous tools developed in prior work (Chandrasekaran et al., 2024).

## 5 Conclusion and Open Problems

In this work, we extended the smoothed agnostic learning framework to the Boolean hypercube, and demonstrated that halfspaces are efficiently learnable with respect to a broad class of input distributions (strictly sub-exponential marginals). Our approach combines tools from smoothing analysis, conditional polynomial approximation, and critical index decomposition to construct lowdegree polynomial approximators in a discrete setting where standard analytic techniques are not applicable. By competing with a smoothed benchmark opt σ , our guarantee circumvents known hardness results for agnostic learning over the hypercube, while matching the sample and runtime complexity of prior work in continuous domains.

Our current techniques apply only to single halfspaces, and the polynomial degree and runtime degrade as the smoothing parameter σ becomes small. In addition, our analysis requires strictly subexponential tail assumptions, and it remains unclear whether comparable guarantees are achievable under weaker conditions. Our results also suggest a potential link to agnostic learning under smoothed input distributions, analogous to the Gaussian framework in continuous domains (Kalai and Teng, 2008; Kalai et al., 2009; Kane et al., 2013; Chandrasekaran et al., 2024). Formalizing this connection in the Boolean setting appears subtle, due to the lack of Euclidean geometry and the discrete nature of bit-flip noise, and we leave it as an intriguing direction for future work.

An important open question is whether these techniques can be extended to intersections of multiple halfspaces. While our framework theoretically supports such generalizations under smoothed optimality, a major technical challenge arises in adapting critical index analysis to this setting. For a single halfspace, sorting the coordinates of the weight vector by magnitude plays a central role in identifying regular and irregular components. However, in the case of multiple halfspaces, each weight vector may induce a different ordering over coordinates, making it difficult to define a unified notion of 'head' and 'tail' variables. As a result, applying a shared conditioning or decomposition strategy becomes nontrivial. Developing new structural insights or approximation techniques that can handle this multi-directional irregularity remains an open problem.

More broadly, this raises the question of how far the smoothed learning framework can be pushed. Can it yield efficient algorithms for learning other complex Boolean concept classes (e.g., DNF formulas, decision lists, or polynomial threshold functions of higher degree) under heavy-tailed distributions? Can it be made adaptive to unknown noise levels or to distributions that do not satisfy strict tail bounds? We leave these questions for future work.

## Acknowledgments and Disclosure of Funding

We sincerely thank the anonymous reviewers for their helpful comments. The authors acknowledge support in part from the National Science Foundation under Award CCF-2217033 (EnCORE: Institute for Emerging CORE Methods in Data Science).

## References

- AWASTHI, P., BALCAN, M.-F., HAGHTALAB, N. and URNER, R. (2015). Efficient learning of linear separators under bounded noise. In Conference on Learning Theory . PMLR.
- BLAIS, E., O'DONNELL, R. and WIMMER, K. (2010). Polynomial regression under arbitrary product distributions. Machine learning 80 273-294.
- BLUM, A. and DUNAGAN, J. (2002). Smoothed analysis of the perceptron algorithm for linear programming. In Proceedings of the Thirteenth Annual ACM-SIAM Symposium on Discrete Algorithms . SODA '02, Society for Industrial and Applied Mathematics, USA.
- BLUM, A., FRIEZE, A. M., KANNAN, R. and VEMPALA, S. S. (1996). A polynomial-time algorithm for learning noisy linear threshold functions. Algorithmica 22 35-52.
- CHANDRASEKARAN, G., KLIVANS, A., KONTONIS, V., MEKA, R. and STAVROPOULOS, K. (2024). Smoothed analysis for learning concepts with low intrinsic dimension. In The Thirty Seventh Annual Conference on Learning Theory . PMLR.
- COHEN, E. (1997). Learning noisy perceptrons by a perceptron in polynomial time. In Proceedings 38th Annual Symposium on Foundations of Computer Science .
- CORTES, C. and VAPNIK, V. (1995). Support-vector networks. Machine learning 20 273-297.
- DIAKONIKOLAS, I., GOPALAN, P., JAISWAL, R., SERVEDIO, R. A. and VIOLA, E. (2010). Bounded independence fools halfspaces. SIAM Journal on Computing 39 3441-3462.
- DIAKONIKOLAS, I., GOULEAKIS, T. and TZAMOS, C. (2019). Distribution-independent pac learning of halfspaces with massart noise. Advances in Neural Information Processing Systems 32 .
- DIAKONIKOLAS, I., IMPAGLIAZZO, R., KANE, D. M., LEI, R., SORRELL, J. and TZAMOS, C. (2021). Boosting in the presence of massart noise. In Conference on Learning Theory . PMLR.
- DIAKONIKOLAS, I., KANE, D., MANURANGSI, P. and REN, L. (2022). Cryptographic hardness of learning halfspaces with massart noise. Advances in Neural Information Processing Systems 35 3624-3636.
- DIAKONIKOLAS, I., KONTONIS, V., TZAMOS, C. and ZARIFIS, N. (2020). Learning halfspaces with massart noise under structured distributions. In Conference on Learning Theory . PMLR.
- DIAKONIKOLAS, I. and SERVEDIO, R. A. (2013). Improved approximation of linear threshold functions. computational complexity 22 623-677.
- FELDMAN, V., GOPALAN, P., KHOT, S. and PONNUSWAMI, A. K. (2009). On agnostic learning of parities, monomials, and halfspaces. SIAM Journal on Computing 39 606-645.
- GURUSWAMI, V. and RAGHAVENDRA, P. (2009). Hardness of learning halfspaces with noise. SIAM Journal on Computing 39 742-765.
- HAUSSLER, D. (1992). Decision theoretic generalizations of the pac model for neural net and other learning applications. Information and Computation 100 78-150.
- KAHN, J., KALAI, G. and LINIAL, N. (1988). The influence of variables on boolean functions. In [Proceedings 1988] 29th Annual Symposium on Foundations of Computer Science .
- KALAI, A. T., KLIVANS, A. R., MANSOUR, Y. and SERVEDIO, R. A. (2008). Agnostically learning halfspaces. SIAM Journal on Computing 37 1777-1805.

- KALAI, A. T., SAMORODNITSKY, A. and TENG, S.-H. (2009). Learning and smoothed analysis. In Proceedings of the 2009 50th Annual IEEE Symposium on Foundations of Computer Science . FOCS '09, IEEE Computer Society, USA.
- KALAI, A. T. and TENG, S.-H. (2008). Decision trees are pac-learnable from most product distributions: a smoothed analysis. arXiv preprint arXiv:0812.0933 .
- KANE, D., KLIVANS, A. and MEKA, R. (2013). Learning halfspaces under log-concave densities: Polynomial approximations and moment matching. In Conference on Learning Theory . PMLR.
- KEARNS, M. J., SCHAPIRE, R. E. and SELLIE, L. M. (1992). Toward efficient agnostic learning. In Proceedings of the Fifth Annual Workshop on Computational Learning Theory . COLT '92, Association for Computing Machinery, New York, NY, USA.
- KLIVANS, A. R., LONG, P. M. and SERVEDIO, R. A. (2009). Learning halfspaces with malicious noise. Journal of Machine Learning Research 10 2715-2740.
- KLIVANS, A. R., O'DONNELL, R. and SERVEDIO, R. A. (2004a). Learning intersections and thresholds of halfspaces. Journal of Computer and System Sciences 68 808-840.
- KLIVANS, A. R., O'DONNELL, R. and SERVEDIO, R. A. (2004b). Learning intersections and thresholds of halfspaces. Journal of Computer and System Sciences 68 808-840. Special Issue on FOCS 2002.
- KLIVANS, A. R., O'DONNELL, R. and SERVEDIO, R. A. (2008). Learning geometric concepts via gaussian surface area. 2008 49th Annual IEEE Symposium on Foundations of Computer Science 541-550.
- LINIAL, N., MANSOUR, Y. and NISAN, N. (1993). Constant depth circuits, fourier transform, and learnability. Journal of the ACM (JACM) 40 607-620.
- LITTLESTONE, N. (1987). Learning quickly when irrelevant attributes abound: A new linearthreshold algorithm. In 28th Annual Symposium on Foundations of Computer Science (sfcs 1987) .
- MATULEF, K., O'DONNELL, R., RUBINFELD, R. and SERVEDIO, R. A. (2010). Testing halfspaces. SIAM Journal on Computing 39 2004-2047.
- MEKA, R. and ZUCKERMAN, D. (2010). Pseudorandom generators for polynomial threshold functions. In Proceedings of the Forty-second ACM Symposium on Theory of Computing .
- MOSSEL, E., O'DONNELL, R. and OLESZKIEWICZ, K. (2005). Noise stability of functions with low influences: invariance and optimality. In 46th Annual IEEE Symposium on Foundations of Computer Science (FOCS'05) . IEEE.
- NOVIKOFF, A. B. (1963). On convergence proofs on perceptrons. In Proceedings of the Symposium on the Mathematical Theory of Automata . Polytechnic Institute of Brooklyn.
- O'DONNELL, R. (2021). Analysis of boolean functions. arXiv preprint arXiv:2105.10386 .
- O'DONNELL, R. and SERVEDIO, R. A. (2011). The chow parameters problem. SIAM Journal on Computing 40 165-199.
- PERES, Y. (2004). Noise stability of weighted majority. arXiv preprint math/0412377 .
- ROSENBLATT, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain. Psychological review 65 386.
- SERVEDIO, R. (2006). Every linear threshold function has a low-weight approximator. In 21st Annual IEEE Conference on Computational Complexity (CCC'06) .
- SPIELMAN, D. and TENG, S.-H. (2001). Smoothed analysis of algorithms: Why the simplex algorithm usually takes polynomial time. In Proceedings of the thirty-third annual ACM symposium on Theory of computing .
- VALIANT, L. G. (1984). A theory of the learnable. Commun. ACM 27 1134-1142.
- WIMMER, K. (2010). Agnostically learning under permutation invariant distributions. In 2010 IEEE 51st Annual Symposium on Foundations of Computer Science .

## A Bonami-Beckner Operator Approximation

We show that for any linear threshold function f : {± 1 } n →{± 1 } the approximation error L 1 of the operator T 1 -ρ f to f can be upper bounded by O ( √ ρ ) .

Lemma A.1 For any linear threshold function f : {± 1 } n →{± 1 } and σ, ρ ∈ [0 , 1] , it holds that

<!-- formula-not-decoded -->

Proof By triangle inequality and special case of Lemma 3.4 when Ω = {± 1 } n and µ = N σ , we have

̸

<!-- formula-not-decoded -->

Therefore, choosing ρ = O ( ϵ 2 ) makes this error at most ϵ/ 2 .

## B Polynomial Approximation for T 1 -ρ f x ( z )

We now approximate T 1 -ρ f x ( z ) using a polynomial for the more general class of strictly subexponential distributions.

Definition B.1 (Strictly Sub-exponential Distributions) A distribution D on R d is ( α, λ ) -strictly sub-exponential if for all ∥ v ∥ 2 = 1 , P x ∼D [ |⟨ x , v ⟩| &gt; t ] ≤ 2 · e -( t/λ ) 1+ α .

Our main goal in this section is to prove the following polynomial approximation result in Lemma 4.7. Suppose f ( x ) = sign( ⟨ w , x ⟩ -θ ) . Denote w ⊙ x as u . Without loss of generality, suppose that | u 1 | ≥ | u 2 | ≥ · · · ≥ | u n | . Then, we have

<!-- formula-not-decoded -->

To obtain a polynomial approximation of T ρ f x ( z ) , we first prove Lemma 4.5.

## Proof Let

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If α -critical index ℓ ( α ) &lt; L ( α, ϵ ) , then the first condition holds by taking H = ℓ ( α ) -1 . If the α -critical index ℓ ( ϵ ) ≥ L ( α, ϵ ) , we will show that the second condition holds.

̸

By Lemma 5.5 in Diakonikolas et al. (2010), there exist a set of nicely separated coordinates G = { i 1 , i 2 , · · · , i t } ⊆ H where i 1 &lt; i 2 &lt; · · · &lt; i t and i k +1 -i k = ⌈ 4 log(1 /α ) /α 2 ⌉ such that | u i k +1 | ≤ | u i k | / 3 for any k ∈ [ t -1] . Then by Claim 5.7 in Diakonikolas et al. (2010), for any two points x 1 = x 2 ∈ {± 1 } t , we have |⟨ u G , x 1 ⟩ - ⟨ u G , x 2 ⟩| ≥ | u i t | . Take t = ⌈ log(1 /ϵ ) /ρσ ⌉ . For any

fixed assignment to the variables in H \ G , we have

<!-- formula-not-decoded -->

where the second inequality is because there's at most one point in an interval of length | u i t | given that ⟨ u G , x 1 ⟩ are well-separated. The third inequality is because

<!-- formula-not-decoded -->

By our choice of L ( α, ϵ ) , t, i t , we have

<!-- formula-not-decoded -->

By applying Lemma 5.5 in Diakonikolas et al. (2010), we have

<!-- formula-not-decoded -->

Therefore, by Lemma C.1, for at least 1 -ϵ fraction of x , it holds with probability at least 1 -ϵ of y that

<!-- formula-not-decoded -->

Then, it follows that with probability at least 1 -ϵ of y we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now we are ready to construct our polynomial. We consider the two cases in Lemma 4.5 separately.

Case 1: If the weight vector w of the LTF falls into the second case of Lemma 4.5, notice that sign( ⟨ u H , y H ⟩ -θ ) = sign( ⟨ w H ⊙ x H , y H ⟩ -θ ) can be represented as a polynomial p y ( x ) of degree at most H = K since only H coordinates of x are relevant. In this case, we take our final polynomial as

<!-- formula-not-decoded -->

Let ∆( x ) be defined as the error term E z ∼N σ [ | p z ( x ) -T 1 -ρ f x ( z ) | ] . We have that for at least 1 -ϵ fraction of x it holds that

̸

<!-- formula-not-decoded -->

It follows that the final L 1 approximation error

<!-- formula-not-decoded -->

Here 'good' x refers to the at least 1 -ϵ fraction of x such that the approximation in (4.2) holds.

Case 2: If the weight vector w of the LTF falls into the first case of Lemma 4.5, we consider the following approximation

<!-- formula-not-decoded -->

where C = (1 -2 ρσ ) · λ log 1 1+ α (2 /ϵ ) + √ 2 log(2 /ϵ ) and ˜ p y H ( x ) will be choosen later as (B.5). In this case, we take our final polynomial as

<!-- formula-not-decoded -->

Let ∆( x ) be defined as the L 1 error term E z ∼N σ [ | p z ( x ) -T 1 -ρ f x ( z ) | ] . For notation simplicity, we denote { y H : |⟨ u H , y H ⟩ -θ | &gt; C · ∥ u T ∥ 2 } as event E . Then, we have

<!-- formula-not-decoded -->

Notice that by Lemma C.1, for at least 1 -ϵ fraction of x , |⟨ u T , y T ⟩| ≤ C · ∥ u T ∥ 2 holds for at least 1 -ϵ fraction of y . For such x and y , under event E , we have

<!-- formula-not-decoded -->

then it follows that

<!-- formula-not-decoded -->

Then, for at least 1 -ϵ fraction of x we have

<!-- formula-not-decoded -->

Here 'good' y refers to the at least 1 -ϵ fraction of y such that |⟨ u T , y T ⟩| ≤ C · ∥ u T ∥ 2 holds. Therefore, we have

<!-- formula-not-decoded -->

Here 'good' x refers to the at least 1 -ϵ fraction of x such that P y [ |⟨ u T , y T ⟩| ≤ C · ∥ u T ∥ 2 ] ≥ 1 -ϵ holds.

Next we consider bounding ∆ 2 ( x ) by constructing proper low-degree polynomial ˜ p y ( x ) as follows. Recall that y ∼ N 1 -ρ ( z ) where y i = z i with probability 1 -ρ and y i randomly drawn from N σ with

probability ρ , we use the following rerandomization trick for random vector y T : for each coordinate of y T , let

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and ϵ i is a Radmacher random variable. Let random variable A = ⟨ u , y ⟩ -θ . Then by (B.1) we have

<!-- formula-not-decoded -->

where v T := ( 1 T -l T ) ⊙ z T + l T ⊙ ( 1 T -m T ) . Then we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

By Theorem C.3, we have

<!-- formula-not-decoded -->

where C ′ is a constant. Therefore, we have

<!-- formula-not-decoded -->

Notice that in this case u T is α -regular and l i m i takes 1 with probability 2 ρσ and 0 with probability 1 -2 ρσ , then by Lemma C.2, for at least 1 -ϵ fraction of ( l T , m T ) it holds that

<!-- formula-not-decoded -->

as long as the condition α ≤ ρσ/ √ log(1 /ϵ ) / 2 holds.

Notice that

<!-- formula-not-decoded -->

let

<!-- formula-not-decoded -->

where E 1 is the event regarding the randomness of ( l T , m T ) such that (B.2) holds, then

<!-- formula-not-decoded -->

as long as ρ = O ( ϵ 2 log(1 /ϵ ) /σ ) . Then, we have

<!-- formula-not-decoded -->

Now we only need to consider polynomial approximation for T ρ f x ( z ) . We can recenter the expectation around zero as follows:

<!-- formula-not-decoded -->

where Q ( s ) = e -| s | / 2 .

For the simplicity of the following analysis, we define random variable

<!-- formula-not-decoded -->

Since x is ( α, λ ) -strictly sub-exponential, then x satisfies the following concentration inequality:

<!-- formula-not-decoded -->

Then, we can rewrite:

<!-- formula-not-decoded -->

For simplicity, denote

<!-- formula-not-decoded -->

Notice that under condition ( l T , m T ) ∈ E 1 and condition y H ∈ E c , we have

<!-- formula-not-decoded -->

where C = (1 -2 ρσ ) · λ log 1 1+ α (4 /ϵ ) + √ 2 log(2 /ϵ ) . We also have

<!-- formula-not-decoded -->

and hence

<!-- formula-not-decoded -->

Then, we have

<!-- formula-not-decoded -->

We now define a polynomial ˜ p z ( x ) approximating T ρ f x ( z ) . To do this, we approximate e -1 2 a 2 x 2 and e a ( s -c ) x using polynomials in x . First, we use a polynomial p 1 ( x ) to approximate e -1 2 a 2 x 2 . This polynomial is given by the following lemma. We choose the parameters later.

Lemma B.2 Let t ∈ Z + . Let x be a random variable satisfying the ( α, λ ) -strictly sub-exponential tail bound. Then there exists a polynomial q of degree

<!-- formula-not-decoded -->

where C is a sufficiently large constant such that the approximation error E x [( q ( x ) -e -1 2 a 2 x 2 ) b ] is upper bounded by 2 ϵ .

Second, to approximate e a ( s -c ) x , we use the function p 2 ( x, s ) = p k ( a ( s -c ) x ) 1 [ | s | ≤ T ] where p k ( x ) = 1 + ∑ k -1 i =1 x i i ! is the degree k -1 Taylor approximation of e x . We choose degree k and threshold T later. Thus our final approximation of T ρ f x ( z ) is

<!-- formula-not-decoded -->

We now want to bound the L 1 error term E x ∼ D x E z ∼N σ [ | ˜ p z ( x ) -T ρ f x ( z ) | ] . To help us analyse the error, we define the 'hybrid' function ¯ p z ( x ) such that

<!-- formula-not-decoded -->

We have that

<!-- formula-not-decoded -->

We now bound ∆ 3 ( x ) and ∆ 4 ( x ) separately. We have that

<!-- formula-not-decoded -->

Observe that ∆ 3 ( x ) can be bounded as the expected sum of the following two terms:

<!-- formula-not-decoded -->

where the first term's bound comes from the fact that | p k ( x ) -e x | ≤ e | x | k ! · | x | k .

We first bound ∆ 31 . We have that

<!-- formula-not-decoded -->

where the second inequality is by e | a ( s -c ) x | ≤ e a ( s -c ) x + e -a ( s -c ) x . Then, we have

<!-- formula-not-decoded -->

Under condition ( l T , m T ) ∈ E 1 and condition y H ∈ E c , we have (B.3) and (B.4) and hence

<!-- formula-not-decoded -->

when

<!-- formula-not-decoded -->

and C ′ is a large enough constant. The second inequality used Lemma C.5. We now bound ∆ 32 . We have that

<!-- formula-not-decoded -->

The third inequality is based on the following claim.

Lemma B.3 Define the distribution Q on R with density function Q ( s ) = e -| s | / 2 . Then there exist a universal constant C such that for every b ∈ R , it holds that

<!-- formula-not-decoded -->

Thus, we have that

<!-- formula-not-decoded -->

when

<!-- formula-not-decoded -->

By (B.3) and (B.4), we can take

<!-- formula-not-decoded -->

The third last inequality is by Lemma C.6. Plugging this into the bound for k in (B.6), we can take

<!-- formula-not-decoded -->

for constant C ′ , C ′′ . By (B.3) and (B.4), we know that

<!-- formula-not-decoded -->

where C ′′′ is a large constant.

We now bound ∆ 4 ( x ) . We have that

<!-- formula-not-decoded -->

Notice that

<!-- formula-not-decoded -->

where the last inequality is by Lemma B.3. Thus, we have

<!-- formula-not-decoded -->

Denote

<!-- formula-not-decoded -->

We have

<!-- formula-not-decoded -->

when δ is chosen accordingly. The third inequality is by Lemma C.5.

By Lemma B.2 and taking the exponent b as 2, the degree of p 1 ( x ) required to get the error is

<!-- formula-not-decoded -->

By using (B.3) and (B.4) and plugging in the order of k, T in (B.7) and (B.8), we know the degree of p 1 ( x ) is

<!-- formula-not-decoded -->

Putting everything together, we get that the degree of p z ( x ) is at most 2 H + deg( p 1 ) + deg( p 2 ) which is

<!-- formula-not-decoded -->

Plugging in the order of ρ = Ω( ϵ 2 ) and α = Ω( ρσ/ √ log(1 /ϵ )) , the degree can be bounded by

<!-- formula-not-decoded -->

## C Proofs of Auxiliary Lemmas

## C.1 Proof of Theorem 4.2

Proof Let f ∗ be the optimal halfspace that achieves opt σ . Let p z be the polynomial of degree at most d such that

<!-- formula-not-decoded -->

Consider the sample dataset S = { ( x i , y i ) } N i =1 in a single run of Algorithm 1. Let p S be the polynomial chosen by the algorithm and let h S be the corresponding hypothesis that the algorithm outputs. By the proof of Theorem 4.1 in Kalai et al. (2008), we have that

̸

<!-- formula-not-decoded -->

Notice that p S is the minimizer of the error, and thus beats any polynomial p z we choose, we have

<!-- formula-not-decoded -->

By (C.1), we can bound ∆ 1 ( S ) as follows:

<!-- formula-not-decoded -->

By the optimality of f ∗ , we have

<!-- formula-not-decoded -->

Thus, we obtain

<!-- formula-not-decoded -->

̸

Since our hypothesis h S is a polynomial threshold function of degree d on n variables, VC theory tells us that for N = poly( n d /ϵ ) , we have that

̸

<!-- formula-not-decoded -->

By Markov's inequality, on any single repetition of the algorithm, we have that

̸

<!-- formula-not-decoded -->

̸

Hence, after r = O (log(1 /δ ) /ϵ ) repetitions of the algorithm, with probability at least 1 -δ/ 2 , one of them will have P ( x ,y ) ∼D [ h S ( x ) = y ] ≤ opt σ + 7 8 ϵ . In this case, using an independent set of size O (log(1 /δ ) /ϵ 2 ) , we probability at most δ/ 2 , we will choose one with error &gt; opt σ + ϵ .

̸

## C.2 Proof of Lemma B.3

Proof The proof below is straightforward calculation by completing the squares.

<!-- formula-not-decoded -->

## C.3 Proof of Lemma B.2

Proof Let p ( x ) = ∑ deg( p ) i =0 c i x i be the polynomial obtained from Lemma C.4 with error ϵ/ 2 and T = ω (log(1 /ϵ )) to be choosen later. Our final polynomial is q ( x ) = p ( 1 2 a 2 x 2 ) . Clearly, deg( q ) = 2 · deg( p ) = O ( √ T log(1 /ϵ )) . We now bound the error.

<!-- formula-not-decoded -->

where the last inequality is by the tail bound of ( α, λ ) -strictly sub-exponential random variable.

We now bound E x [ ( q ( x ) -e -1 2 a 2 x 2 ) 2 b ] . We have that

<!-- formula-not-decoded -->

where the third last inequality is by Lemma C.5. Here C, C ′ , C ′′ are large enough constant. Putting it all together, we get that

<!-- formula-not-decoded -->

Choosing

<!-- formula-not-decoded -->

where C ′′′ is a large constant makes the total error less than 2 ϵ . Since T is ω (log(1 /ϵ )) , the degree of the final polynomial is O ( √ T log(1 /ϵ )) which is

<!-- formula-not-decoded -->

## C.4 Other Auxiliary Lemmas

Lemma C.1 Suppose x is a distribution on {± 1 } n that is ( α, λ ) -strictly subexponential and y distributed as N 1 -ρ ( z ) . For any fixed T ⊆ [ n ] and fixed z , with probability at least 1 -ϵ of x , it holds that

<!-- formula-not-decoded -->

where u = w ⊙ x and C = (1 -2 ρσ ) · λ log 1 1+ α (4 /ϵ ) + √ 2 log(2 /ϵ ) .

Proof By Hoeffding's inequality, we have

<!-- formula-not-decoded -->

Therefore, with probability at least 1 -ϵ it holds that

<!-- formula-not-decoded -->

Notice that

<!-- formula-not-decoded -->

since x is ( α, λ ) -strictly subexponential, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Thus, with probability at least 1 -ϵ of x it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that x is distributed on {± 1 } n and hence ∥ u T ∥ 2 = ∥ w T ⊙ x T ∥ 2 = ∥ w T ∥ 2 , then with probability at least 1 -ϵ of y it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

Lemma C.2 Suppose that the vector w ∈ R n is α -regular, i.e., ∥ w ∥ ∞ ≤ α · ∥ w ∥ 2 . Suppose u is a n -dimensional random vector where each coordinate is 1 with probability ρ and 0 with probability 1 -ρ independently. If α ≤ ρ/ √ log(1 /δ ) / 2 , then with probability at least 1 -δ over the randomness of u it holds that

<!-- formula-not-decoded -->

where c = ( ρ -√ log(1 /δ ) / 2 · α ) -1 2 . If condition α ≤ ρ/ √ 2 log(1 /δ ) holds, then with probability at least 1 -δ over the randomness of u it holds that

<!-- formula-not-decoded -->

Proof By Hoeffding's inequality and notice that E [( w i u i ) 2 ] = w 2 i · E [ u i ] = ρw 2 i and 0 ≤ ( w i u i ) 2 ≤ w 2 i , we obtain

<!-- formula-not-decoded -->

where the second inequality is by the definition of the infinity norm, and the third inequality is by the α -regularity condition. Therefore, with probability at least 1 -δ , we can get

<!-- formula-not-decoded -->

Then, with probability at least 1 -δ , it holds that

<!-- formula-not-decoded -->

Theorem C.3 (Berry-Esseen CLT) Let X 1 , X 2 , ..., be independent random variables with E [ X i ] = 0 , E [ X 2 i ] = σ 2 i &gt; 0 , and E [ | X i | 3 ] = ρ i &lt; ∞ . Also, let

<!-- formula-not-decoded -->

be the normalized n -th partial sum. Denote F n the cdf of S n , and Φ the cdf of the standard normal distribution. Then for all n there exists an absolute constant C such that

<!-- formula-not-decoded -->

Lemma C.4 (Lemma D.11 in Chandrasekaran et al. (2024)) For T &gt; 0 and error ϵ &gt; 0 , there exists a polynomial p such that

1. sup x ∈ [0 ,T ] | p ( x ) -e -x | ≤ ϵ .
2. deg( p ) ≤ O ( √ T log(1 /ϵ )) , if T = ω (log(1 /ϵ )) .
3. p ( x ) = ∑ deg( p ) i =1 c i x i where | c i | ≤ e C √ T log(1 /ϵ ) for all i ≤ deg( p ) . Here C is a large enough constant.

Lemma C.5 If x is ( α, λ ) -strictly sub-exponential random variable satisfying P x [ | x | &gt; t ] ≤ 2 · e -( t/λ ) 1+ α , then the k -th moment is upper bounded by:

<!-- formula-not-decoded -->

where C is a universal constant.

Proof By the layer cake representation and the tail bound, we have

<!-- formula-not-decoded -->

Making the substitution s = ( t/λ ) 1+ α , i.e. t = λs 1 1+ α and dt = λ 1+ α s -α 1+ α ds , we get

<!-- formula-not-decoded -->

Notice that Γ( x ) ≤ Cx x -1 / 2 e -x e 1 / (12 x ) for a positive constant C , then we have

<!-- formula-not-decoded -->

Lemma C.6 If x is ( α, λ ) -strictly sub-exponential random variable satisfying P x [ | x | &gt; t ] ≤ 2 · e -( t/λ ) 1+ α , then for any a &gt; 0

<!-- formula-not-decoded -->

Proof We split the integral into two parts at some threshold T , which we choose later:

<!-- formula-not-decoded -->

By the layer cake representation and the tail bound, we have

<!-- formula-not-decoded -->

Choose T to be (2 a ) 1 /α λ 1+1 /α . Then, we have ( t/λ ) 1+ α ≥ 2 at for t ≥ T and hence

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus,

## NeurIPS Paper Checklist

The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

- You should answer [Yes] , [No] , or [NA] .
- [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
- Please provide a short (1-2 sentence) justification right after your answer (even for NA).

The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

## IMPORTANT, please:

- Delete this instruction block, but keep the section heading 'NeurIPS Paper Checklist" ,
- Keep the checklist subsection headings, questions/answers and guidelines below.
- Do not modify the questions and only use the provided macros for your answers .

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state the paper's main contribution: a discrete analogue of smoothed agnostic learning for Boolean halfspaces under strictly sub-exponential distributions, along with a computationally efficient learning algorithm. These claims are supported by the theoretical results in the main body (Section 4.5).

## Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

## Answer: [Yes]

Justification: The limitations of our work are discussed in the conclusion section of this paper (Section 5).

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

Justification: All theoretical results are clearly stated with assumptions, and full proofs are provided in the main paper or the supplementary material. Lemmas and tools from prior work are cited and contextualized.

Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: The paper is entirely theoretical and does not include empirical experiments. Guidelines:

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

Justification: No datasets or code are used or required, as the paper does not include experiments.

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

Answer: [NA]

Justification: No experimental setting is involved; the paper is theoretical.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include empirical results or plots requiring error bars or statistical tests.

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

Answer: [NA]

Justification: No experiments were conducted; no compute resources were used.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: The research is theoretical, complies with NeurIPS ethical guidelines, and does not raise ethical concerns regarding data use, privacy, or fairness.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper is purely theoretical.

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

Justification: The paper does not release data, models, or tools with potential misuse risk.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: No external datasets, models, or software assets are used.

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

Justification: No new datasets or software assets are introduced.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: No human subjects or crowdsourcing are involved.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: No IRB approval is necessary, as no human subjects are involved.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: No LLMs were used in developing or supporting the core scientific contributions of this research.

Guidelines:

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy ( https://neurips.cc/Conferences/2025/LLM ) for what should or should not be described.